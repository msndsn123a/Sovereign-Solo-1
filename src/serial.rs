//! Direct Hardware Serial Telemetry (16550 UART driver for COM1).
//!
//! Provides raw, zero-dependency serial communication over I/O port 0x3F8,
//! allowing telemetry even after UEFI boot services are terminated.

use core::fmt;

/// Standard COM1 base I/O port.
pub const COM1_BASE: u16 = 0x3F8;
pub const COM2_BASE: u16 = 0x2F8;
pub const INPUT_PREAMBLE: [u8; 2] = *b"SO";
pub const OUTPUT_PREAMBLE: [u8; 2] = *b"SR";
pub const UPDATE_PREAMBLE: [u8; 2] = *b"NU";
pub const INPUT_FRAME_SIZE: usize = 2 + 64;
pub const OUTPUT_FRAME_SIZE: usize = 2 + core::mem::size_of::<i32>();

/// Line Status Register bit 5: Transmitter Holding Register Empty (THRE).
pub const LSR_THRE: u8 = 0x20;

/// Inline assembly helper to write a byte to an x86 I/O port.
#[inline(always)]
pub unsafe fn outb(port: u16, val: u8) {
    core::arch::asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack, preserves_flags)
    );
}

/// Inline assembly helper to read a byte from an x86 I/O port.
#[inline(always)]
pub unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    core::arch::asm!(
        "in al, dx",
        in("dx") port,
        out("al") val,
        options(nomem, nostack, preserves_flags)
    );
    val
}

/// Inline assembly helper to write a 16-bit word to an x86 I/O port.
#[inline(always)]
pub unsafe fn outw(port: u16, val: u16) {
    core::arch::asm!(
        "out dx, ax",
        in("dx") port,
        in("ax") val,
        options(nomem, nostack, preserves_flags)
    );
}

/// Initializes standard COM1 UART (0x3F8) for 115200 baud, 8 data bits, no parity, 1 stop bit (8-N-1).
/// Interrupts are disabled and the 14-byte FIFO buffer is enabled.
pub unsafe fn init() {
    init_port(COM1_BASE, false);
}

/// Initializes a second UART as the binary streaming port.
pub unsafe fn init_wire_port() {
    init_port(COM2_BASE, false);
}

/// Initializes a 16550-compatible UART at the supplied base I/O port.
pub unsafe fn init_port(base: u16, enable_rx_interrupt: bool) {
    // 1. Disable all UART interrupts (IER = 0x00)
    outb(base + 1, 0x00);

    // 2. Enable Divisor Latch Access Bit (DLAB = 1) in Line Control Register
    outb(base + 3, 0x80);

    // 3. Set baud rate divisor to 1 (115200 baud = 115200 / 1)
    outb(base, 0x01); // Divisor latch low byte
    outb(base + 1, 0x00); // Divisor latch high byte

    // 4. Configure 8 bits, no parity, 1 stop bit (8-N-1) and clear DLAB (0x03)
    outb(base + 3, 0x03);

    // 5. Enable FIFO, clear TX/RX FIFO queues, set 14-byte threshold (0xC7)
    outb(base + 2, 0xC7);

    // 6. Set RTS and DTR, enable Auxiliary Output 2 (0x0B)
    outb(base + 4, 0x0B);

    if enable_rx_interrupt {
        outb(base + 1, 0x01);
    }
}

/// Enables Receiver Data Available in IER. Polling remains the consumer; callers
/// should keep CPU interrupts disabled unless an IRQ handler has been installed.
pub unsafe fn enable_rx_interrupt(base: u16) {
    outb(base + 1, 0x01);
}

/// Returns a COM1 byte if the UART reports Data Ready.
#[inline(always)]
pub fn poll_byte() -> Option<u8> {
    poll_byte_from(COM1_BASE)
}

/// Returns one byte without blocking when the selected UART reports Data Ready.
#[inline(always)]
pub fn poll_byte_from(base: u16) -> Option<u8> {
    unsafe {
        if inb(base + 5) & 0x01 != 0 {
            Some(inb(base))
        } else {
            None
        }
    }
}

/// Fixed-length frame reader for `SO` followed by 64 raw signed-byte values.
pub struct FrameReader {
    state: u8,
    index: usize,
    payload: [i8; 64],
    preamble_first: u8,
    preamble_tsc: u64,
    update_lba_bytes: [u8; 8],
}

pub struct ReceivedFrame {
    pub payload: [i8; 64],
    pub preamble_tsc: u64,
    pub ingress_tsc: u64,
}

pub enum FrameEvent {
    Input(ReceivedFrame),
    Update {
        lba: u64,
    },
}

impl FrameReader {
    pub const fn new() -> Self {
        Self {
            state: 0,
            index: 0,
            payload: [0; 64],
            preamble_first: 0,
            preamble_tsc: 0,
            update_lba_bytes: [0; 8],
        }
    }

    /// Consumes available UART bytes and returns a complete frame when found.
    pub fn poll_frame(&mut self, base: u16) -> Option<FrameEvent> {
        loop {
            let byte = match poll_byte_from(base) {
                Some(byte) => byte,
                None => return None,
            };
            if let Some(event) = self.consume_byte(byte) {
                return Some(event);
            }
        }
    }

    fn consume_byte(&mut self, byte: u8) -> Option<FrameEvent> {
        match self.state {
            0 => {
                if byte == INPUT_PREAMBLE[0] || byte == UPDATE_PREAMBLE[0] {
                    self.state = 1;
                    self.preamble_first = byte;
                    self.preamble_tsc = unsafe { crate::timer::read_tsc() };
                }
            }
            1 => {
                if self.preamble_first == INPUT_PREAMBLE[0] && byte == INPUT_PREAMBLE[1] {
                    self.state = 2;
                    self.index = 0;
                } else if self.preamble_first == UPDATE_PREAMBLE[0]
                    && byte == UPDATE_PREAMBLE[1]
                {
                    self.state = 3;
                    self.index = 0;
                } else if byte == INPUT_PREAMBLE[0] || byte == UPDATE_PREAMBLE[0] {
                    self.preamble_first = byte;
                    self.preamble_tsc = unsafe { crate::timer::read_tsc() };
                } else {
                    self.state = 0;
                }
            }
            3 => {
                self.update_lba_bytes[self.index] = byte;
                self.index += 1;
                if self.index == self.update_lba_bytes.len() {
                    self.state = 0;
                    self.index = 0;
                    return Some(FrameEvent::Update {
                        lba: u64::from_le_bytes(self.update_lba_bytes),
                    });
                }
            }
            2 => {
                self.payload[self.index] = byte as i8;
                self.index += 1;
                if self.index == self.payload.len() {
                    self.state = 0;
                    self.index = 0;
                    return Some(FrameEvent::Input(ReceivedFrame {
                        payload: self.payload,
                        preamble_tsc: self.preamble_tsc,
                        ingress_tsc: unsafe { crate::timer::read_tsc_with_aux() }.0,
                    }));
                }
            }
            _ => self.state = 0,
        }
        None
    }
}

/// Encodes the fixed six-byte `SR` response frame.
pub const fn encode_response_frame(value: i32) -> [u8; OUTPUT_FRAME_SIZE] {
    let scalar = value.to_le_bytes();
    [OUTPUT_PREAMBLE[0], OUTPUT_PREAMBLE[1], scalar[0], scalar[1], scalar[2], scalar[3]]
}

/// Writes a single byte to COM1, translating `\n` to `\r\n` for standard serial terminals.
/// Spins until the Transmitter Holding Register Empty (THRE) bit is set.
pub unsafe fn write_byte(b: u8) {
    if b == b'\n' {
        write_byte_raw_from(COM1_BASE, b'\r');
    }
    write_byte_raw_from(COM1_BASE, b);
}

/// Writes a raw byte to a UART without newline translation.
#[inline(always)]
pub unsafe fn write_byte_raw_from(base: u16, b: u8) {
    // Wait until LSR bit 5 (Transmitter Holding Register Empty) is 1
    while (inb(base + 5) & LSR_THRE) == 0 {
        core::hint::spin_loop();
    }
    outb(base, b);
}

/// Writes binary bytes without text newline translation.
pub unsafe fn write_raw_bytes_from(base: u16, bytes: &[u8]) {
    for &byte in bytes {
        write_byte_raw_from(base, byte);
    }
}

/// Writes a string slice to COM1.
pub unsafe fn write_str(s: &str) {
    for &b in s.as_bytes() {
        write_byte(b);
    }
}

/// Zero-sized type implementing `core::fmt::Write` for formatted serial printing.
pub struct SerialPort;

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        unsafe {
            crate::serial::write_str(s);
        }
        Ok(())
    }
}

/// Formatted print macro for COM1 serial telemetry.
#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut serial = $crate::serial::SerialPort;
            let _ = write!(serial, $($arg)*);
        }
    };
}

/// Formatted println macro for COM1 serial telemetry with newline.
#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($($arg:tt)*) => {
        $crate::serial_print!("{}\n", format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn so_frame_ingests_exactly_64_signed_bytes() {
        let payload = core::array::from_fn(|index| (index as u8).wrapping_mul(17) as i8);
        let mut reader = FrameReader::new();
        assert!(reader.consume_byte(b'S').is_none());
        assert!(reader.consume_byte(b'O').is_none());
        let mut event = None;
        for byte in payload.map(|value| value as u8) {
            event = reader.consume_byte(byte);
        }
        match event {
            Some(FrameEvent::Input(frame)) => assert_eq!(frame.payload, payload),
            _ => panic!("SO plus 64 payload bytes must emit exactly one input frame"),
        }
        assert_eq!(INPUT_FRAME_SIZE, 66);
    }

    #[test]
    fn sr_response_is_six_bytes_with_little_endian_scalar() {
        let value = -0x0102_0304i32;
        let frame = encode_response_frame(value);
        assert_eq!(&frame[..2], b"SR");
        assert_eq!(&frame[2..], &value.to_le_bytes());
        assert_eq!(frame.len(), 6);
    }

    #[test]
    fn nu_control_parses_little_endian_lba() {
        let expected_lba = 264_192u64;
        let mut reader = FrameReader::new();
        assert!(reader.consume_byte(b'N').is_none());
        assert!(reader.consume_byte(b'U').is_none());
        let mut event = None;
        for byte in expected_lba.to_le_bytes() {
            event = reader.consume_byte(byte);
        }
        assert!(matches!(event, Some(FrameEvent::Update { lba }) if lba == expected_lba));
    }
}
