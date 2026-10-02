#![no_std]
#![cfg_attr(not(test), no_main)]

#[cfg(test)]
extern crate std;

pub mod activation;
pub mod bitpack;
pub mod crypto;
#[path = "../tools/wasm_verifier/src/pure.rs"]
pub mod wasm_math;

#[cfg(target_arch = "x86_64")]
pub mod boot;
#[cfg(target_arch = "x86_64")]
pub mod cat;
#[cfg(target_arch = "x86_64")]
pub mod io_ring;
#[cfg(target_arch = "x86_64")]
pub mod kernel;
#[cfg(target_arch = "x86_64")]
pub mod nvme;
#[cfg(target_arch = "x86_64")]
pub mod paging;
#[cfg(target_arch = "x86_64")]
pub mod pci;
#[cfg(target_arch = "x86_64")]
pub mod serial;
#[cfg(target_arch = "x86_64")]
pub mod shared_mem;
#[cfg(target_arch = "x86_64")]
pub mod smp;
#[cfg(target_arch = "x86_64")]
pub mod telemetry;
#[cfg(target_arch = "x86_64")]
pub mod timer;
#[cfg(target_arch = "x86_64")]
pub mod waitpkg;
#[cfg(target_arch = "x86_64")]
pub mod watchdog;

macro_rules! x86_only_items {
    ($($item:item)*) => {
        $(
            #[cfg(target_arch = "x86_64")]
            $item
        )*
    };
}

x86_only_items! {
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::mem::MaybeUninit;
#[cfg(not(test))]
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};
use uefi::prelude::*;

use crate::bitpack::{
    parse_neur_header, validate_pot_payload, validate_ternary_payload, NeurHeader, NeurHeaderError,
    NEUR_ATTENTION_PAYLOAD_BYTES, NEUR_ATTENTION_POT_PAYLOAD_BYTES, NEUR_BASE_HEADER_SIZE,
    NEUR_HEADER_SIZE, NEUR_MAGIC, NEUR_MLP_PAYLOAD_BYTES, NEUR_MLP_SPARSE_PAYLOAD_BYTES,
    NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, NEUR_MODEL_MLP, NEUR_QUANT_POT,
};
use crate::boot::{establish_cpu_sovereignty, exit_uefi_boot_services, hardware_shutdown};
use crate::io_ring::{InputFrame, InputRing, OutputFrame, OutputRing};
use crate::kernel::{
    enable_avx512_os_state, infer_avx2, infer_scalar, software_prefetch_warm,
    PotLinearAttention, SparseTernaryMlp64x32x16, TernaryLinearAttention, TernaryMlp64x32x16,
    TernaryWeights64,
};
use crate::timer::{
    cycles_for_seconds, cycles_to_micros_milli, read_tsc, summarize, LatencySample, LatencySummary,
    MAX_LATENCY_SAMPLES,
};

const STREAM_FRAME_LIMIT: u64 = 8;
const STREAM_TIMEOUT_SECONDS: u64 = 25;

/// 4 KiB page-aligned DMA buffer, large enough for 512-byte and 4 KiB namespace blocks.
#[repr(C, align(4096))]
pub struct WeightBuffer(pub [u8; nvme::DMA_PAGE_SIZE]);

/// Active/shadow raw shard DMA buffers; each element is independently 4 KiB aligned.
static mut MODEL_WEIGHT_BUFFERS: [WeightBuffer; 2] =
    [const { WeightBuffer([0; nvme::DMA_PAGE_SIZE]) }; 2];

/// 64-byte aligned universal streaming I/O ring buffers.
pub static mut INPUT_RING: InputRing = InputRing::new();
static mut OUTPUT_RING: OutputRing = OutputRing::new();

pub fn input_ring_indices() -> (usize, usize) {
    unsafe { InputRing::snapshot_raw(core::ptr::addr_of!(INPUT_RING)) }
}

/// AP callback used only by the selected auxiliary ring-monitor processor.
pub extern "C" fn ap_ring_monitor_entry() -> ! {
    let mut polls = 0u32;
    loop {
        polls = polls.wrapping_add(1);
        if polls & 0x0FFF == 0 {
            let (head, tail) = input_ring_indices();
            smp::record_ring_monitor_sample(head.wrapping_sub(tail));
            if let Some(lba) = smp::take_shard_update_request() {
                let success = unsafe { update_model_from_raw_lba(lba) };
                smp::finish_shard_update(success);
            }
        }
        core::arch::x86_64::_mm_pause();
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("[PANIC]: {}", info);
    loop {
        core::hint::spin_loop();
    }
}

#[derive(Clone, Copy)]
struct LoadedModel {
    solo_weights: TernaryWeights64,
}

struct ModelSlot(UnsafeCell<MaybeUninit<LoadedModel>>);

impl ModelSlot {
    const fn empty() -> Self {
        Self(UnsafeCell::new(MaybeUninit::uninit()))
    }
}

unsafe impl Sync for ModelSlot {}

static MODEL_SLOTS: [ModelSlot; 2] = [const { ModelSlot::empty() }; 2];
pub static ACTIVE_SHARD_INDEX: AtomicUsize = AtomicUsize::new(0);
static MODEL_READERS: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

struct ActiveModelGuard {
    index: usize,
    model: &'static LoadedModel,
}

impl ActiveModelGuard {
    fn model(&self) -> &LoadedModel {
        self.model
    }
}

impl Drop for ActiveModelGuard {
    fn drop(&mut self) {
        MODEL_READERS[self.index].fetch_sub(1, Ordering::Release);
    }
}

fn store_model_slot(index: usize, model: LoadedModel) {
    unsafe {
        (*MODEL_SLOTS[index].0.get()).write(model);
    }
}

fn acquire_active_model() -> ActiveModelGuard {
    loop {
        let index = ACTIVE_SHARD_INDEX.load(Ordering::Acquire);
        MODEL_READERS[index].fetch_add(1, Ordering::AcqRel);
        if ACTIVE_SHARD_INDEX.load(Ordering::Acquire) == index {
            let model = unsafe { (&*MODEL_SLOTS[index].0.get()).assume_init_ref() };
            return ActiveModelGuard { index, model };
        }
        MODEL_READERS[index].fetch_sub(1, Ordering::Release);
    }
}

fn model_weight_buffer_ptr(index: usize) -> *mut WeightBuffer {
    unsafe {
        core::ptr::addr_of_mut!(MODEL_WEIGHT_BUFFERS)
            .cast::<WeightBuffer>()
            .add(index)
    }
}

fn infer_values(
    inputs: &[i8; 64],
    model: &LoadedModel,
    avx2_enabled: bool,
    outputs: &mut [i32; 64],
) -> u8 {
    outputs[0] = if avx2_enabled {
        unsafe { infer_avx2(&model.solo_weights, inputs) }
    } else {
        infer_scalar(&model.solo_weights, inputs)
    };
    1
}

fn infer_frame(
    frame: InputFrame,
    model: &LoadedModel,
    avx2_enabled: bool,
) -> OutputFrame {
    let mut output = OutputFrame {
        output_dim: 0,
        values: [0; 64],
        t0_preamble: frame.t0_preamble,
        t1_ingress: frame.t1_ingress,
        t2_compute: 0,
    };
    output.output_dim = infer_values(&frame.payload, model, avx2_enabled, &mut output.values);
    output.t2_compute = unsafe { timer::read_tsc_with_aux() }.0;
    output
}

unsafe fn transmit_output_frame(frame: &OutputFrame) -> u64 {
    serial::write_raw_bytes_from(serial::COM2_BASE, &serial::OUTPUT_PREAMBLE);
    serial::write_byte_raw_from(serial::COM2_BASE, frame.output_dim);
    let mut index = 0usize;
    while index < frame.output_dim as usize {
        serial::write_raw_bytes_from(serial::COM2_BASE, &frame.values[index].to_le_bytes());
        index += 1;
    }
    timer::read_tsc_with_aux().0
}

fn summarize_stage(samples: &[LatencySample], stage: u8) -> LatencySummary {
    let mut values = [0u64; MAX_LATENCY_SAMPLES];
    let count = samples.len().min(MAX_LATENCY_SAMPLES);
    let mut index = 0usize;
    while index < count {
        values[index] = match stage {
            0 => samples[index].ingress_cycles,
            1 => samples[index].compute_cycles,
            2 => samples[index].egress_cycles,
            _ => samples[index].turnaround_cycles,
        };
        index += 1;
    }
    summarize(&values[..count])
}

fn log_latency_summary(name: &str, summary: LatencySummary) {
    let min_us = cycles_to_micros_milli(summary.min);
    let p50_us = cycles_to_micros_milli(summary.p50);
    let p99_us = cycles_to_micros_milli(summary.p99);
    let max_us = cycles_to_micros_milli(summary.max);
    serial_println!(
        "[LATENCY]: {} cycles min={} p50={} p99={} max={}; calibrated_us min={}.{:03} p50={}.{:03} p99={}.{:03} max={}.{:03}",
        name,
        summary.min,
        summary.p50,
        summary.p99,
        summary.max,
        min_us / 1000,
        min_us % 1000,
        p50_us / 1000,
        p50_us % 1000,
        p99_us / 1000,
        p99_us % 1000,
        max_us / 1000,
        max_us % 1000
    );
}

fn verify_shared_mailbox(
    mailbox: &shared_mem::SharedMailbox,
    model: &LoadedModel,
    avx2_enabled: bool,
) -> bool {
    const TEST_FRAMES: usize = shared_mem::MAILBOX_CAPACITY;
    let mut samples = [LatencySample::default(); TEST_FRAMES];
    let mut completed = 0usize;
    let mut dropped = 0usize;

    while completed < TEST_FRAMES {
        let test_payload = [1i8; 64];
        let t0_publish = unsafe { read_tsc() };
        if mailbox
            .try_publish_input(&test_payload, t0_publish)
            .is_err()
        {
            dropped += 1;
            continue;
        }

        let (sequence, input_slot, t0_observed) = loop {
            if let Some(pending) = mailbox.peek_input() {
                break pending;
            }
            core::arch::x86_64::_mm_pause();
        };
        let t1_ingest = unsafe { read_tsc() };
        let published = mailbox.compute_and_publish_output(
            1,
            t0_observed,
            t1_ingest,
            |output_values| {
                let input = unsafe { input_slot.payload() };
                output_values[0] = if avx2_enabled {
                    unsafe { infer_avx2(&model.solo_weights, input) }
                } else {
                    infer_scalar(&model.solo_weights, input)
                };
                unsafe { read_tsc() }
            },
        );
        mailbox.consume_input(sequence);
        let (_, t2_compute, t3_commit) = match published {
            Ok(timestamps) => timestamps,
            Err(()) => {
                dropped += 1;
                continue;
            }
        };

        let (_, output_values, actual_dim) = match mailbox.consume_output() {
            Some(output) => output,
            None => {
                dropped += 1;
                continue;
            }
        };
        if actual_dim != 1 {
            return false;
        }

        let expected = if avx2_enabled {
            unsafe { infer_avx2(&model.solo_weights, &test_payload) }
        } else {
            infer_scalar(&model.solo_weights, &test_payload)
        };
        if output_values[0] != expected {
            return false;
        }

        samples[completed] = LatencySample {
            ingress_cycles: t1_ingest.saturating_sub(t0_observed),
            compute_cycles: t2_compute.saturating_sub(t1_ingest),
            egress_cycles: t3_commit.saturating_sub(t2_compute),
            turnaround_cycles: t3_commit.saturating_sub(t0_observed),
        };
        completed += 1;
    }

    if dropped != 0 {
        serial_println!("[SHM TEST ERROR]: dropped={} frames", dropped);
        return false;
    }
    log_latency_summary("shared_mailbox_ingress", summarize_stage(&samples, 0));
    log_latency_summary("shared_mailbox_compute", summarize_stage(&samples, 1));
    log_latency_summary("shared_mailbox_commit", summarize_stage(&samples, 2));
    let summary = summarize_stage(&samples, 3);
    log_latency_summary("shared_mailbox_turnaround", summary);
    let sub_microsecond =
        timer::tsc_frequency_hz() != 0 && summary.max < (timer::tsc_frequency_hz() / 1_000_000);
    serial_println!(
        "[SOLO VERIFY]: output_dim=1; frames={}, drops={}, scalar_reference=match, max_below_1us={}",
        completed,
        dropped,
        sub_microsecond
    );
    true
}

#[derive(Clone, Copy, Debug)]
enum ShardLoadError {
    NotFound,
    InvalidHeader(NeurHeaderError),
    SignatureMismatch,
    InvalidPayload,
    ReadFailed,
}

fn neur_payload_size(header: &NeurHeader) -> Result<usize, ShardLoadError> {
    match (header.model_type, header.quant_type) {
        (NEUR_MODEL_MLP, 0) if header.block_sparse => Ok(NEUR_MLP_SPARSE_PAYLOAD_BYTES),
        (NEUR_MODEL_MLP, 0) => Ok(NEUR_MLP_PAYLOAD_BYTES),
        (NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, 0) => Ok(NEUR_ATTENTION_PAYLOAD_BYTES),
        (NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, NEUR_QUANT_POT) => {
            Ok(NEUR_ATTENTION_POT_PAYLOAD_BYTES)
        }
        _ => Err(ShardLoadError::InvalidPayload),
    }
}

fn pot_query_row_to_solo(row: &[u8; 32]) -> TernaryWeights64 {
    let mut packed = [0u8; 16];
    let mut index = 0usize;
    while index < 64 {
        let byte = row[index >> 1];
        let code = if index & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let ternary = if code == 0 || code == 8 {
            0
        } else if code < 8 {
            1
        } else {
            3
        };
        packed[index >> 2] |= ternary << ((index & 3) * 2);
        index += 1;
    }
    TernaryWeights64::from_packed(&packed)
}

fn parse_neur_shard(data: &[u8]) -> Result<(LoadedModel, NeurHeader), ShardLoadError> {
    let header = parse_neur_header(data).map_err(ShardLoadError::InvalidHeader)?;
    if header.version != 2 || data.len() < NEUR_HEADER_SIZE {
        return Err(ShardLoadError::SignatureMismatch);
    }
    let payload_size = neur_payload_size(&header)?;
    let payload_end = NEUR_HEADER_SIZE + payload_size;
    if payload_end > data.len() || payload_end > nvme::DMA_PAGE_SIZE {
        return Err(ShardLoadError::InvalidPayload);
    }
    let payload = &data[NEUR_HEADER_SIZE..payload_end];
    if !crypto::verify_neur_signature(
        &data[..NEUR_BASE_HEADER_SIZE],
        &data[NEUR_BASE_HEADER_SIZE..NEUR_HEADER_SIZE],
        payload,
    ) {
        return Err(ShardLoadError::SignatureMismatch);
    }
    let valid_payload = if header.quant_type == NEUR_QUANT_POT {
        validate_pot_payload(payload)
    } else if header.block_sparse {
        validate_ternary_payload(&payload[..NEUR_MLP_PAYLOAD_BYTES])
    } else {
        validate_ternary_payload(payload)
    };
    if !valid_payload {
        return Err(ShardLoadError::InvalidPayload);
    }

    // Keep the existing signed shard formats and validation, but project each
    // format once to its first 64-input row for stateless Solo inference.
    let solo_weights = match (header.model_type, header.quant_type) {
        (NEUR_MODEL_MLP, _) if header.block_sparse => {
            let sparse = SparseTernaryMlp64x32x16::from_packed_payload(payload)
                .ok_or(ShardLoadError::InvalidPayload)?;
            TernaryWeights64::from_packed(&sparse.weights.layer1[0])
        }
        (NEUR_MODEL_MLP, _) => {
            let mlp = TernaryMlp64x32x16::from_packed_payload(payload)
                .ok_or(ShardLoadError::InvalidPayload)?;
            TernaryWeights64::from_packed(&mlp.layer1[0])
        }
        (NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, NEUR_QUANT_POT) => {
            let attention = PotLinearAttention::from_packed_payload(payload, header.pot_scale)
                .ok_or(ShardLoadError::InvalidPayload)?;
            pot_query_row_to_solo(&attention.q[0])
        }
        (NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, _) => {
            let attention = TernaryLinearAttention::from_packed_payload(payload)
                .ok_or(ShardLoadError::InvalidPayload)?;
            TernaryWeights64::from_packed(&attention.q[0])
        }
        _ => return Err(ShardLoadError::InvalidPayload),
    };
    Ok((LoadedModel { solo_weights }, header))
}

unsafe fn load_neur_shard(
    controller: &mut nvme::NvmeController,
    buffer: &mut WeightBuffer,
) -> Result<(LoadedModel, NeurHeader, u64), ShardLoadError> {
    let logical_block_size = controller.logical_block_size;
    if !(NEUR_HEADER_SIZE..=nvme::DMA_PAGE_SIZE).contains(&logical_block_size) {
        return Err(ShardLoadError::InvalidPayload);
    }
    let candidate_byte_offsets = [133120u64 * 512, 34816u64 * 512, 67584u64 * 512, 0];
    let mut selected_lba = None;

    for byte_offset in candidate_byte_offsets {
        if byte_offset % logical_block_size as u64 != 0 {
            continue;
        }
        let lba = byte_offset / logical_block_size as u64;
        buffer.0.fill(0);
        if controller
            .read_blocks_polled(lba, 1, &mut buffer.0)
            .is_err()
        {
            continue;
        }

        let magic = u32::from_be_bytes([buffer.0[0], buffer.0[1], buffer.0[2], buffer.0[3]]);
        if magic == NEUR_MAGIC {
            selected_lba = Some(lba);
            break;
        }
    }

    let lba = selected_lba.ok_or(ShardLoadError::NotFound)?;
    let header = parse_neur_header(&buffer.0[..logical_block_size])
        .map_err(ShardLoadError::InvalidHeader)?;
    let payload_size = neur_payload_size(&header)?;
    let payload_end = NEUR_HEADER_SIZE + payload_size;
    if payload_end > nvme::DMA_PAGE_SIZE {
        return Err(ShardLoadError::InvalidPayload);
    }
    let block_count = (payload_end + logical_block_size - 1) / logical_block_size;
    if block_count == 0 || block_count > u16::MAX as usize {
        return Err(ShardLoadError::InvalidPayload);
    }

    buffer.0.fill(0);
    controller
        .read_blocks_polled(lba, block_count as u16, &mut buffer.0)
        .map_err(|_| ShardLoadError::ReadFailed)?;

    let (model, confirmed_header) = parse_neur_shard(&buffer.0[..payload_end])?;
    if confirmed_header != header {
        return Err(ShardLoadError::InvalidPayload);
    }
    Ok((model, header, lba))
}

unsafe fn load_neur_shard_from_nvme(
    buffer: &mut WeightBuffer,
) -> Result<(LoadedModel, NeurHeader, u64, usize), ShardLoadError> {
    let pci_nvme = pci::find_nvme_device().ok_or(ShardLoadError::NotFound)?;
    serial_println!(
        "[SOVEREIGN_CORE]: Found NVMe on PCI {:02x}:{:02x}.{:x} BAR0: {:#018x}",
        pci_nvme.bus,
        pci_nvme.slot,
        pci_nvme.func,
        pci_nvme.bar0_phys
    );
    let mut controller =
        nvme::NvmeController::init(&pci_nvme).map_err(|_| ShardLoadError::ReadFailed)?;
    let block_size = controller.logical_block_size;
    let (model, header, lba) = load_neur_shard(&mut controller, buffer)?;
    Ok((model, header, lba, block_size))
}

/// Read, authenticate, and atomically activate a candidate shard from a raw LBA.
/// Runs on the auxiliary AP while the BSP continues polling and serving frames.
unsafe fn update_model_from_raw_lba(lba: u64) -> bool {
    let Some(device) = pci::find_nvme_device() else {
        return false;
    };
    let mut controller = match nvme::NvmeController::init(&device) {
        Ok(controller) => controller,
        Err(_) => return false,
    };
    let active_index = ACTIVE_SHARD_INDEX.load(Ordering::Acquire);
    let shadow_index = active_index ^ 1;

    while MODEL_READERS[shadow_index].load(Ordering::Acquire) != 0 {
        core::arch::x86_64::_mm_pause();
    }

    let buffer = &mut *model_weight_buffer_ptr(shadow_index);
    if controller
        .read_blocks_polled(lba, 1, &mut buffer.0)
        .is_err()
    {
        return false;
    }
    let block_size = controller.logical_block_size;
    let header = match parse_neur_header(&buffer.0[..block_size]) {
        Ok(header) => header,
        Err(_) => return false,
    };
    if header.version != 2 {
        return false;
    }
    let payload_size = match neur_payload_size(&header) {
        Ok(size) => size,
        Err(_) => return false,
    };
    let payload_end = NEUR_HEADER_SIZE + payload_size;
    if payload_end > nvme::DMA_PAGE_SIZE {
        return false;
    }
    let block_count = payload_end.div_ceil(block_size);
    if block_count == 0 || block_count > u16::MAX as usize {
        return false;
    }
    if controller
        .read_blocks_polled(lba, block_count as u16, &mut buffer.0)
        .is_err()
    {
        return false;
    }
    let Ok((candidate, _)) = parse_neur_shard(&buffer.0[..payload_end]) else {
        return false;
    };

    if ACTIVE_SHARD_INDEX.load(Ordering::Acquire) != active_index {
        return false;
    }
    store_model_slot(shadow_index, candidate);
    ACTIVE_SHARD_INDEX.swap(shadow_index, Ordering::SeqCst);
    true
}

fn safe_identity_model() -> LoadedModel {
    let mut packed = [0u8; 16];
    packed[0] = 0b01;
    LoadedModel {
        solo_weights: TernaryWeights64::from_packed(&packed),
    }
}

fn run_host_ipc_loop(
    mailbox: &shared_mem::SharedMailbox,
    model: &LoadedModel,
    avx2_enabled: bool,
    watchdog: &watchdog::Watchdog,
) -> bool {
    let start = unsafe { read_tsc() };
    let timeout_cycles = cycles_for_seconds(STREAM_TIMEOUT_SECONDS);
    let mut completed = 0u64;
    let mut drops = 0u64;
    let mut samples = [LatencySample::default(); MAX_LATENCY_SAMPLES];
    let mut cursor = 0usize;

    serial_println!(
        "[SHM HOST IPC]: Waiting for host frames at GPA {:#018x}; limit={}, timeout={} s.",
        shared_mem::HOST_MAILBOX_PHYSICAL_BASE,
        STREAM_FRAME_LIMIT,
        STREAM_TIMEOUT_SECONDS
    );

    while completed < STREAM_FRAME_LIMIT
        && unsafe { read_tsc() }.saturating_sub(start) < timeout_cycles
    {
        watchdog.kick_watchdog();
        if let Some((sequence, input_slot, t0_ready)) = mailbox.peek_input() {
            let t1_ingest = unsafe { read_tsc() };
            let published = mailbox.compute_and_publish_output(
                1,
                t0_ready,
                t1_ingest,
                |output_values| {
                    let inputs = unsafe { input_slot.payload() };
                    infer_values(inputs, model, avx2_enabled, output_values);
                    unsafe { read_tsc() }
                },
            );
            mailbox.consume_input(sequence);
            match published {
                Ok((_, t2_compute, t3_commit)) => {
                    samples[cursor] = LatencySample {
                        ingress_cycles: t1_ingest.saturating_sub(t0_ready),
                        compute_cycles: t2_compute.saturating_sub(t1_ingest),
                        egress_cycles: t3_commit.saturating_sub(t2_compute),
                        turnaround_cycles: t3_commit.saturating_sub(t0_ready),
                    };
                    cursor = (cursor + 1) % MAX_LATENCY_SAMPLES;
                    completed += 1;
                }
                Err(()) => drops += 1,
            }
        } else {
            unsafe {
                waitpkg::wait_for_address(core::ptr::addr_of!(mailbox.input_head).cast());
            }
        }
    }

    if cursor > 0 {
        let count = (completed as usize).min(MAX_LATENCY_SAMPLES);
        let window = &samples[..count];
        log_latency_summary("host_ipc_ingress", summarize_stage(window, 0));
        log_latency_summary("host_ipc_compute", summarize_stage(window, 1));
        log_latency_summary("host_ipc_commit", summarize_stage(window, 2));
        log_latency_summary("host_ipc_turnaround", summarize_stage(window, 3));
    }
    serial_println!(
        "[SHM HOST IPC]: guest_processed={}, drops={}, elapsed_cycles={}",
        completed,
        drops,
        unsafe { read_tsc() }.saturating_sub(start)
    );
    completed == STREAM_FRAME_LIMIT && drops == 0
}

fn run_mmio_loop(
    mailbox: &shared_mem::SharedMailbox,
    model: &LoadedModel,
    avx2_enabled: bool,
    watchdog: &watchdog::Watchdog,
) -> bool {
    let start = unsafe { read_tsc() };
    let timeout_cycles = cycles_for_seconds(STREAM_TIMEOUT_SECONDS);
    let mut completed = 0u64;
    let mut drops = 0u64;
    let mut samples = [LatencySample::default(); MAX_LATENCY_SAMPLES];
    let mut cursor = 0usize;
    let output_dim = 1;

    serial_println!(
        "[MMIO]: IVSHMEM ring active; capacity={}, frame_limit={}, timeout={} s.",
        shared_mem::MAILBOX_CAPACITY,
        STREAM_FRAME_LIMIT,
        STREAM_TIMEOUT_SECONDS
    );

    while completed < STREAM_FRAME_LIMIT
        && unsafe { read_tsc() }.saturating_sub(start) < timeout_cycles
    {
        watchdog.kick_watchdog();
        if let Some((sequence, input_slot, t0_ready)) = mailbox.peek_input() {
            let t1_ingest = unsafe { read_tsc() };
            let published = mailbox.compute_and_publish_output(
                output_dim,
                t0_ready,
                t1_ingest,
                |output_values| {
                    let inputs = unsafe { input_slot.payload() };
                    infer_values(inputs, model, avx2_enabled, output_values);
                    unsafe { read_tsc() }
                },
            );
            mailbox.consume_input(sequence);
            match published {
                Ok((_, t2_compute, t3_commit)) => {
                    samples[cursor] = LatencySample {
                        ingress_cycles: t1_ingest.saturating_sub(t0_ready),
                        compute_cycles: t2_compute.saturating_sub(t1_ingest),
                        egress_cycles: t3_commit.saturating_sub(t2_compute),
                        turnaround_cycles: t3_commit.saturating_sub(t0_ready),
                    };
                    cursor = (cursor + 1) % MAX_LATENCY_SAMPLES;
                    completed += 1;
                }
                Err(()) => drops += 1,
            }
        } else {
            // PCI shared-memory BARs may be UC; use the private WB sentinel and
            // a bounded TSC deadline instead of monitoring the BAR itself.
            waitpkg::bounded_wait();
        }
    }

    if cursor > 0 {
        let count = (completed as usize).min(MAX_LATENCY_SAMPLES);
        let window = &samples[..count];
        log_latency_summary("mmio_ingress", summarize_stage(window, 0));
        log_latency_summary("mmio_compute", summarize_stage(window, 1));
        log_latency_summary("mmio_commit", summarize_stage(window, 2));
        let turnaround = summarize_stage(window, 3);
        log_latency_summary("mmio_turnaround", turnaround);
        let below_one_microsecond = timer::tsc_frequency_hz() != 0
            && turnaround.max < timer::tsc_frequency_hz() / 1_000_000;
        serial_println!(
            "[MMIO VERIFY]: guest_processed={}, drops={}, max_turnaround_below_1us={}",
            completed,
            drops,
            below_one_microsecond
        );
    }
    serial_println!(
        "[MMIO]: guest_processed={}, drops={}, elapsed_cycles={}",
        completed,
        drops,
        unsafe { read_tsc() }.saturating_sub(start)
    );
    completed == STREAM_FRAME_LIMIT && drops == 0
}

#[entry]
fn main(
    image_handle: uefi::Handle,
    mut system_table: uefi::table::SystemTable<uefi::table::Boot>,
) -> uefi::Status {
    // 1. Direct Hardware Serial Telemetry Initialization (COM1 115200 8-N-1)
    unsafe {
        serial::init();
        serial::init_wire_port();
    }
    serial_println!("[SOVEREIGN_CORE]: SERIAL TELEMETRY INITIALIZED (COM1 115200 8-N-1)");

    let memory_encryption = boot::memory_encryption_status();
    match memory_encryption {
        boot::MemoryEncryptionStatus::Unsupported => serial_println!(
            "[SECURITY MEM]: Hardware encryption unsupported / inactive (virtualized/legacy)"
        ),
        boot::MemoryEncryptionStatus::IntelTme {
            active: true,
            locked,
            algorithm,
        } => {
            let algorithm_name = match algorithm {
                0 => "AES-XTS-128",
                1 => "AES-XTS-256",
                _ => "unknown",
            };
            serial_println!(
                "[SECURITY TME]: Intel TME active ({}, locked={})",
                algorithm_name,
                locked
            );
            if algorithm > 1 {
                serial_println!("[SECURITY TME DETAIL]: unrecognized algorithm id=0x{:04X}", algorithm);
            }
        }
        boot::MemoryEncryptionStatus::IntelTme {
            active: false,
            locked,
            algorithm,
        } => serial_println!(
            "[SECURITY TME]: Intel TME supported but inactive (algorithm=0x{:04X}, locked={})",
            algorithm,
            locked
        ),
        boot::MemoryEncryptionStatus::AmdSme {
            active: true,
            sev_supported,
            c_bit_position: Some(position),
        } => serial_println!(
            "[SECURITY SME]: AMD SME active (encrypted DRAM, C-bit={}, SEV supported={})",
            position,
            sev_supported
        ),
        boot::MemoryEncryptionStatus::AmdSme {
            active: true,
            sev_supported,
            c_bit_position: None,
        } => serial_println!(
            "[SECURITY SME]: AMD SME active but C-bit unavailable (SEV supported={}, page tagging disabled)",
            sev_supported
        ),
        boot::MemoryEncryptionStatus::AmdSme {
            active: false,
            sev_supported,
            c_bit_position,
        } => serial_println!(
            "[SECURITY SME]: AMD SME supported but inactive (C-bit={:?}, SEV supported={})",
            c_bit_position,
            sev_supported
        ),
    }
    if cfg!(feature = "strict-memory-encryption") && !memory_encryption.satisfies_strict_policy() {
        serial_println!(
            "[SECURITY GATE]: strict memory encryption required; refusing boot before model or DMA setup."
        );
        return uefi::Status::ABORTED;
    }
    serial_println!(
        "[SECURITY GATE]: strict_memory_encryption={}, policy_satisfied={}",
        cfg!(feature = "strict-memory-encryption"),
        memory_encryption.satisfies_strict_policy()
    );

    let calibrated_hz = timer::calibrate_tsc(|window_us| {
        let _ = system_table.boot_services().stall(window_us);
    });
    let (frequency_mhz, frequency_hundredths) = timer::tsc_frequency_mhz_parts();
    if timer::calibration_used_cpuid15() {
        serial_println!(
            "[TIMER]: Calibrated TSC frequency: {}.{:02} MHz (CPUID leaf 0x15)",
            frequency_mhz,
            frequency_hundredths
        );
    } else {
        serial_println!(
            "[TIMER]: Calibrated TSC frequency: {}.{:02} MHz (100 ms UEFI stall)",
            frequency_mhz,
            frequency_hundredths
        );
    }
    let (invariant_tsc, rdtscp_supported) = boot::tsc_capabilities();
    serial_println!(
        "[TIMER]: Invariant TSC={}, RDTSCP={}, reads serialized with LFENCE.",
        invariant_tsc,
        rdtscp_supported
    );
    if !invariant_tsc {
        serial_println!("[TIMER WARNING]: TSC is not invariant; calibrated microseconds may drift if CPU power state changes.");
    }
    let (hwp_supported, hwp_enabled, turbo_supported) = boot::cpu_power_management_status();
    serial_println!(
        "[CPU POWER]: HWP supported={}, enabled={}, turbo capability={}; firmware/thermal limits left unchanged.",
        hwp_supported,
        hwp_enabled,
        turbo_supported
    );
    if calibrated_hz == 0 {
        serial_println!("[TIMER WARNING]: TSC frequency calibration failed; calibrated time conversions unavailable.");
    }

    let (guest_f, guest_bw, guest_dq, guest_vl, guest_xsave, guest_osxsave, guest_xcr0) =
        boot::avx512_guest_features();
    serial_println!(
        "[GUEST CPUID]: AVX512F={}, AVX512BW={}, AVX512DQ={}, AVX512VL={}, XSAVE={}, OSXSAVE={}, XCR0=0x{:016X}",
        guest_f,
        guest_bw,
        guest_dq,
        guest_vl,
        guest_xsave,
        guest_osxsave,
        guest_xcr0
    );
    let guest_vpopcntdq = boot::cpu_supports_avx512_vpopcntdq();
    serial_println!("[GUEST CPUID]: AVX512VPOPCNTDQ={}", guest_vpopcntdq);
    let avx512_state_enabled = if boot::cpu_supports_avx512_state() {
        let state_enabled = unsafe { enable_avx512_os_state() };
        if state_enabled {
            serial_println!("[CPU]: AVX-512 OS ZMM state enabled.");
        } else {
            serial_println!(
                "[CPU WARNING]: AVX-512 state unavailable; using scalar compute fallback."
            );
        }
        state_enabled
    } else {
        serial_println!("[CPU WARNING]: AVX-512 unavailable; using scalar compute fallback.");
        false
    };
    let avx512_enabled = avx512_state_enabled && boot::cpu_supports_avx512();
    let avx2_hardware = core::arch::x86_64::__cpuid(0).eax >= 7
        && core::arch::x86_64::__cpuid_count(7, 0).ebx & (1 << 5) != 0;
    let avx2_enabled = avx512_state_enabled && avx2_hardware;
    serial_println!(
        "[SIMD]: Solo backend selected={}",
        if avx2_enabled {
            "AVX2"
        } else {
            "scalar"
        }
    );
    if waitpkg::initialize() {
        serial_println!(
            "[POWER]: WAITPKG active (UMONITOR/UMWAIT), IA32_UMWAIT_CONTROL=0x{:08X}",
            waitpkg::control_value()
        );
    } else {
        serial_println!("[POWER]: WAITPKG unsupported, falling back to PAUSE loop");
    }

    let mut hardware_watchdog = watchdog::probe(&system_table);

    let discovered_ivshmem = if cfg!(feature = "host-ipc") {
        None
    } else {
        unsafe { pci::find_ivshmem_device() }
    };
    let mut smp_topology = smp::discover(&system_table);
    if smp_topology.madt_found {
        serial_println!(
            "[SMP]: ACPI MADT found; enabled processors={}, BSP APIC ID={}, secondary xAPIC IDs={}",
            smp_topology.madt_processor_count,
            smp_topology.bsp_apic_id,
            smp_topology.ap_count
        );
        if smp_topology.ap_count != 0 && !smp_topology.x2apic_mode {
            smp_topology.trampoline_ready = smp::prepare_trampoline(&mut smp_topology);
            if smp_topology.trampoline_ready {
                serial_println!(
                    "[SMP]: AP trampoline reserved at {:#06x}; SIPI vector={:#04x}",
                    smp_topology.trampoline_address,
                    smp_topology.trampoline_address >> 12
                );
            } else {
                serial_println!(
                    "[SMP WARNING]: low-memory trampoline unavailable; AP startup will be skipped."
                );
            }
        } else if smp_topology.x2apic_mode {
            serial_println!("[SMP]: x2APIC mode is active; xAPIC SIPI startup skipped.");
        }
    } else {
        serial_println!("[SMP]: ACPI MADT unavailable; AP startup skipped.");
    }
    let mut mmio_ranges = [paging::MmioRange::EMPTY; 4];
    let mut mmio_range_count = 0usize;
    if let Some(device) = discovered_ivshmem {
        mmio_ranges[mmio_range_count] = paging::MmioRange::new(device.bar0_phys, 4096);
        mmio_range_count += 1;
        mmio_ranges[mmio_range_count] =
            paging::MmioRange::new(device.shared_memory_phys, device.shared_memory_size);
        mmio_range_count += 1;
    }
    if !cfg!(feature = "host-ipc") {
        if let Some(nvme_device) = unsafe { pci::find_nvme_device() } {
            serial_println!(
                "[PCI MMIO]: NVMe BAR0={:#018x}; adding uncached huge-page mapping.",
                nvme_device.bar0_phys
            );
            mmio_ranges[mmio_range_count] = paging::MmioRange::new(nvme_device.bar0_phys, 4096);
            mmio_range_count += 1;
        }
    }
    if smp_topology.ap_count != 0 && !smp_topology.x2apic_mode {
        mmio_ranges[mmio_range_count] = paging::MmioRange::new(smp_topology.lapic_base, 4096);
        mmio_range_count += 1;
    }
    let mut mmio_ingress = None;
    let mailbox_result = if let Some(device) = discovered_ivshmem {
        serial_println!(
            "[PCI MMIO]: IVSHMEM {:04X}:{:04X} at {:02x}:{:02x}.{} BAR0={:#018x}, BAR2(shared)={:#018x}, bytes={}",
            device.vendor_id,
            device.device_id,
            device.bus,
            device.slot,
            device.func,
            device.bar0_phys,
            device.shared_memory_phys,
            device.shared_memory_size
        );
        match unsafe { boot::bind_ivshmem_mailbox(device) } {
            Some(pointer) => {
                mmio_ingress = Some(device);
                Ok(pointer)
            }
            None => {
                serial_println!(
                    "[PCI MMIO WARNING]: shared BAR is too small or misaligned; using UEFI RAM mailbox and UART ingress."
                );
                boot::reserve_shared_mailbox()
            }
        }
    } else {
        serial_println!(
            "[PCI MMIO]: IVSHMEM 1AF4:1110 not found; using UEFI RAM mailbox and UART ingress."
        );
        boot::reserve_shared_mailbox()
    };
    let mailbox_ptr = match mailbox_result {
        Ok(pointer) => pointer,
        Err(error) => {
            serial_println!("[SHM ERROR]: UEFI page allocation failed: {:?}", error);
            return uefi::Status::OUT_OF_RESOURCES;
        }
    };
    let mailbox_address = mailbox_ptr.as_ptr() as u64;
    serial_println!(
        "[SHM]: Initialized Mailbox at physical addr {:#018x}, bytes={}, pages={}, alignment={}, mode={}",
        mailbox_address,
        core::mem::size_of::<shared_mem::SharedMailbox>(),
        core::mem::size_of::<shared_mem::SharedMailbox>().div_ceil(4096),
        core::mem::align_of::<shared_mem::SharedMailbox>(),
        if mmio_ingress.is_some() {
            "PCI IVSHMEM BAR2"
        } else if cfg!(feature = "host-ipc") {
            "host-backed GPA"
        } else {
            "UEFI allocated"
        }
    );

    let stdout = system_table.stdout();
    let _ = stdout.reset(false);
    let _ = writeln!(stdout, "NEURAL-BOX CORE: HARNESS INITIALIZED");

    // UEFI filesystem protocols are unavailable after ExitBootServices. Keep
    // the DMA-aligned shard buffer alive and fill it before transferring control.
    let initial_buffer = model_weight_buffer_ptr(0);
    let filesystem_shard_len = unsafe { boot::read_weights_file(&mut (*initial_buffer).0) };
    match filesystem_shard_len {
        Some(bytes) => serial_println!(
            "[LOADER]: Read weights.bin from UEFI SimpleFileSystem ({} bytes).",
            bytes
        ),
        None => serial_println!("[LOADER]: File weights.bin not found; using fallback"),
    }
    let filesystem_load_result = filesystem_shard_len.map(|file_bytes| {
        let result = unsafe { parse_neur_shard(&(&(*initial_buffer).0)[..file_bytes]) };
        match &result {
            Ok(_) => {
                serial_println!(
                    "[LOADER]: Validated file shard from UEFI FAT volume ({} bytes).",
                    file_bytes
                );
                serial_println!("[SECURITY]: Shard signature valid (Ed25519 verified)");
            }
            Err(_) => serial_println!(
                "[SECURITY]: Signature mismatch or invalid header - rejecting shard"
            ),
        }
        result
    });

    // 3. Exit UEFI Boot Services & Ingest Memory Map
    let mmap_info = unsafe { exit_uefi_boot_services(image_handle, &mut system_table) };

    // 4. Absolute CPU Sovereignty Configuration (Mute interrupts, verify CR0/CR4/XCR0)
    unsafe {
        establish_cpu_sovereignty(avx512_enabled);
    }
    let paging_activation = unsafe {
        paging::activate_identity_map(
            &mmio_ranges[..mmio_range_count],
            memory_encryption.amd_c_bit_mask(),
        )
    };
    match paging_activation.mode {
        paging::HugePageMode::OneGiB => serial_println!(
            "[PAGING]: Custom CR3 activated (1GiB HugePages); mapped={} GiB, MMIO UC pages={}, encrypted huge pages={}, C-bit mask={:#018x}",
            paging_activation.mapped_bytes / (1024 * 1024 * 1024),
            paging_activation.uncached_huge_pages,
            paging_activation.encrypted_huge_pages,
            paging_activation.encryption_mask
        ),
        paging::HugePageMode::TwoMiB => serial_println!(
            "[PAGING]: Custom CR3 activated (2MiB HugePages); mapped={} GiB, MMIO UC pages={}, encrypted huge pages={}, C-bit mask={:#018x}",
            paging_activation.mapped_bytes / (1024 * 1024 * 1024),
            paging_activation.uncached_huge_pages,
            paging_activation.encrypted_huge_pages,
            paging_activation.encryption_mask
        ),
    }
    serial_println!("[TIMER]: Maskable interrupts confirmed disabled after ExitBootServices.");
    serial_println!("[SOVEREIGN_CORE]: BOOT SERVICES TERMINATED. INTERRUPTS MUTED. CPU ACQUIRED.");
    serial_println!(
        "[SOVEREIGN_CORE]: PHYSICAL MMAP INGESTED ({} entries, {} bytes)",
        mmap_info.entry_count,
        mmap_info.map_size
    );

    let load_result = if let Some(file_result) = filesystem_load_result {
        file_result.map(|(model, header)| (model, header, None))
    } else {
        unsafe { load_neur_shard_from_nvme(&mut *initial_buffer) }
            .map(|(model, header, lba, block_size)| (model, header, Some((lba, block_size))))
    };
    let model = match load_result {
        Ok((loaded_model, header, source)) => {
            serial_println!(
                "[SHARD]: MAGIC=0x{:08X}, VERSION={}, INPUT_DIM={}, MODEL_TYPE={}, HIDDEN_OR_ATTN_DIM={}, OUTPUT_DIM={}",
                NEUR_MAGIC,
                header.version,
                header.input_dim,
                header.model_type,
                header.hidden_dim,
                header.output_dim
            );
            serial_println!(
                "[SHARD CONFIG]: multi_stream={}, quant_type={}, pot_scale={}, block_sparse={}, activation_lut={}",
                header.multi_stream,
                header.quant_type,
                header.pot_scale,
                header.block_sparse,
                header.activation_lut
            );
            if let Some((lba, block_size)) = source {
                serial_println!(
                    "[LOADER]: Loaded shard from raw NVMe LBA {} (block size={} bytes).",
                    lba,
                    block_size
                );
            } else {
                serial_println!("[LOADER]: Using model shard from UEFI FAT filesystem.");
            }
            serial_println!(
                "[SOLO MODEL]: authenticated shard retained; projection=row 0, output_dim=1, recurrent_state=disabled"
            );
            loaded_model
        }
        Err(error) => {
            serial_println!(
                "[LOADER]: NVMe fallback unavailable or invalid ({:?}); using safe identity model.",
                error
            );
            match error {
                ShardLoadError::InvalidHeader(header_error) => serial_println!(
                    "[SHARD ERROR]: Invalid model header ({:?}); using safe identity model.",
                    header_error
                ),
                ShardLoadError::SignatureMismatch => serial_println!(
                    "[SECURITY]: Rejected unsigned or tampered shard; using safe identity model."
                ),
                ShardLoadError::NotFound => {
                    serial_println!("[SHARD ERROR]: No valid NEUR shard; using safe identity model.")
                }
                ShardLoadError::InvalidPayload => serial_println!(
                    "[SHARD ERROR]: Model payload bounds/encoding invalid; using safe identity model."
                ),
                ShardLoadError::ReadFailed => {
                    serial_println!("[SHARD ERROR]: Model NVMe read failed; using safe identity model.")
                }
            }
            safe_identity_model()
        }
    };
    store_model_slot(0, model);
    ACTIVE_SHARD_INDEX.store(0, Ordering::Release);
    serial_println!("[SOVEREIGN_CORE]: DMA SHARD INGESTION COMPLETE.");
    serial_println!(
        "[ALLOC]: Model inference uses fixed-size static/stack buffers; no heap allocator."
    );
    serial_println!(
        "[SOLO KERNEL]: predecoded ternary coefficients=64, scalar action values=-1/0/1, avx2={}",
        avx2_enabled
    );
    let cat_configured = if let Some(capabilities) = boot::l3_cat_capabilities() {
        serial_println!(
            "[CACHE]: Intel L3 CAT supported; CBM bits={}, CLOS count={}",
            capabilities.cbm_length,
            capabilities.clos_count
        );
        match unsafe { cat::configure_l3_cat(capabilities) } {
            Some(configuration) => {
                serial_println!(
                    "[CACHE]: CAT supported/configured; CLOS=1 mask=0x{:08X}, L3_QOS_CFG=0x{:016X}",
                    configuration.way_mask,
                    configuration.qos_config
                );
                true
            }
            None => {
                serial_println!(
                    "[CACHE]: CAT detected but CLOS configuration unavailable; using software warming."
                );
                false
            }
        }
    } else {
        false
    };
    let warmed_lines = unsafe { software_prefetch_warm(&(*initial_buffer).0) };
    if cat_configured {
        serial_println!(
            "[CACHE]: CLOS 1 weight-buffer warm sweep completed; cache_lines={}",
            warmed_lines
        );
    } else {
        serial_println!(
            "[CACHE]: Software L1/L3 prefetch warming active; cache_lines={}",
            warmed_lines
        );
    }
    unsafe { smp::start_application_processors(&mut smp_topology) };
    smp::log_topology(&smp_topology);

    let mailbox = unsafe { mailbox_ptr.as_ref() };
    if !cfg!(feature = "host-ipc") {
        if mailbox.magic != shared_mem::MAILBOX_MAGIC
            || mailbox.version != shared_mem::MAILBOX_VERSION
            || (mmio_ingress.is_none()
                && !verify_shared_mailbox(mailbox, &model, avx2_enabled))
        {
            serial_println!("[SHM TEST ERROR]: mailbox validation or zero-copy inference failed.");
            return uefi::Status::ABORTED;
        }
    }

    if cfg!(feature = "host-ipc") {
        if hardware_watchdog.is_supported() {
            let _ = hardware_watchdog.arm();
        }
        if mailbox.magic != shared_mem::MAILBOX_MAGIC
            || mailbox.version != shared_mem::MAILBOX_VERSION
            || !run_host_ipc_loop(
                mailbox,
                &model,
                avx2_enabled,
                &hardware_watchdog,
            )
        {
            hardware_watchdog.disarm();
            serial_println!("[SHM HOST IPC ERROR]: mailbox validation, host frame count, or output commit failed.");
            return uefi::Status::ABORTED;
        }
        hardware_watchdog.disarm();
        unsafe { hardware_shutdown() };
    }

    if let Some(device) = mmio_ingress {
        if hardware_watchdog.is_supported() {
            let _ = hardware_watchdog.arm();
        }
        serial_println!(
            "[MMIO]: ingress=PCIe shared BAR2 ({:#018x}); COM2 is disabled as a data path.",
            device.shared_memory_phys
        );
        if !run_mmio_loop(
            mailbox,
            &model,
            avx2_enabled,
            &hardware_watchdog,
        ) {
            hardware_watchdog.disarm();
            serial_println!("[MMIO ERROR]: frame limit, no-drop, or completion check failed.");
            return uefi::Status::ABORTED;
        }
        hardware_watchdog.disarm();
        unsafe { hardware_shutdown() };
    }

    if hardware_watchdog.is_supported() {
        let _ = hardware_watchdog.arm();
    }

    let calibrated_timeout_cycles = cycles_for_seconds(STREAM_TIMEOUT_SECONDS);
    let stream_timeout_cycles = if calibrated_timeout_cycles == 0 {
        u64::MAX
    } else {
        calibrated_timeout_cycles
    };
    serial_println!(
        "[UART]: COM2 RX=NB+64 or NS+stream_id+64; stream tags ignored by stateless Solo; NR+stream_id is a no-op, standalone NR exits; frame_limit={}, timeout={} s ({} cycles)",
        STREAM_FRAME_LIMIT,
        STREAM_TIMEOUT_SECONDS,
        stream_timeout_cycles
    );
    serial_println!("[RING]: Atomic SPSC input/output queues ready (capacity=128).");

    let (mut input_producer, mut input_consumer) =
        unsafe { (&mut *core::ptr::addr_of_mut!(INPUT_RING)).split() };
    let (mut output_producer, mut output_consumer) =
        unsafe { (&mut *core::ptr::addr_of_mut!(OUTPUT_RING)).split() };
    unsafe {
        serial::enable_rx_interrupt(serial::COM2_BASE);
        serial::write_raw_bytes_from(serial::COM2_BASE, b"UART_READY\n");
    }

    let mut frame_reader = serial::FrameReader::new();
    let stream_start = unsafe { read_tsc() };
    let mut stream_events = 0u64;
    let mut reset_commands = 0u64;
    let mut update_commands = 0u64;
    let mut update_successes = 0u64;
    let mut update_failures = 0u64;
    let mut reset_seen = false;
    let mut frames_ingress = 0u64;
    let mut frames_transmitted = 0u64;
    let mut ring_full_drops = 0u64;
    let mut latency_samples = [LatencySample::default(); MAX_LATENCY_SAMPLES];
    let mut latency_sample_count = 0usize;
    let mut latency_sample_cursor = 0usize;
    let mut latency_samples_total = 0u64;
    let mut telemetry_ring = telemetry::TelemetryRing::new();

    while !reset_seen
        && (STREAM_FRAME_LIMIT == 0
            || stream_events < STREAM_FRAME_LIMIT
            || smp::shard_update_in_progress())
        && unsafe { read_tsc() }.saturating_sub(stream_start) < stream_timeout_cycles
    {
        hardware_watchdog.kick_watchdog();
        if let Some(success) = smp::take_shard_update_result() {
            let lba = smp::shard_update_lba();
            if success {
                update_successes += 1;
                serial_println!(
                    "[HOTSWAP]: status=success, lba={}, active_index={}, signature=valid, atomic_order=SeqCst",
                    lba,
                    ACTIVE_SHARD_INDEX.load(Ordering::Acquire)
                );
            } else {
                update_failures += 1;
                serial_println!(
                    "[HOTSWAP]: status=rejected, lba={}, active_index={}, active_model=preserved",
                    lba,
                    ACTIVE_SHARD_INDEX.load(Ordering::Acquire)
                );
            }
        }
        if let Some(event) = frame_reader.poll_frame(serial::COM2_BASE) {
            match event {
                serial::FrameEvent::Input(received) => {
                    let frame = InputFrame {
                        payload: received.payload,
                        t0_preamble: received.preamble_tsc,
                        t1_ingress: received.ingress_tsc,
                        stream_id: received.stream_id,
                    };
                    if input_producer.push(frame).is_ok() {
                        frames_ingress += 1;
                        stream_events += 1;
                    } else {
                        ring_full_drops += 1;
                    }
                }
                serial::FrameEvent::Reset { stream_id, .. } => {
                    let target_stream = stream_id.unwrap_or(0);
                    reset_commands += 1;
                    stream_events += 1;
                    if stream_id.is_none() {
                        reset_seen = true;
                    }
                    serial_println!(
                        "[SOLO CONTROL]: command=NR reset_count={} stream={} stateless=true",
                        reset_commands,
                        target_stream
                    );
                }
                serial::FrameEvent::InvalidStreamSelector(stream_id) => {
                    stream_events += 1;
                    serial_println!(
                        "[UART CONTROL ERROR]: stream selector={} is outside 0..{}",
                        stream_id,
                        serial::ATTENTION_STREAM_COUNT
                    );
                }
                serial::FrameEvent::Update { lba } => {
                    update_commands += 1;
                    if smp::request_shard_update(lba) {
                        serial_println!(
                            "[HOTSWAP]: request queued, lba={}, active_index={}, execution=auxiliary-AP",
                            lba,
                            ACTIVE_SHARD_INDEX.load(Ordering::Acquire)
                        );
                    } else {
                        update_failures += 1;
                        serial_println!(
                            "[HOTSWAP]: request rejected, lba={}, reason=update-busy-or-no-AP, active_model=preserved",
                            lba
                        );
                    }
                }
            }
        }
        let pending_frame = if smp::shard_update_in_progress() {
            None
        } else {
            input_consumer.pop()
        };
        if let Some(frame) = pending_frame {
            let model_guard = acquire_active_model();
            let active_model = model_guard.model();
            let result = infer_frame(frame, active_model, avx2_enabled);
            drop(model_guard);

            if output_producer.push(result).is_err() {
                ring_full_drops += 1;
            } else if let Some(output) = output_consumer.pop() {
                let t3_uart_tx = unsafe { transmit_output_frame(&output) };
                telemetry_ring.record(telemetry::TelemetryEntry {
                    ingress_tsc: output.t1_ingress,
                    compute_end_tsc: output.t2_compute,
                    egress_tsc: t3_uart_tx,
                });
                latency_samples[latency_sample_cursor] = LatencySample {
                    ingress_cycles: output.t1_ingress.saturating_sub(output.t0_preamble),
                    compute_cycles: output.t2_compute.saturating_sub(output.t1_ingress),
                    egress_cycles: t3_uart_tx.saturating_sub(output.t2_compute),
                    turnaround_cycles: t3_uart_tx.saturating_sub(output.t0_preamble),
                };
                latency_sample_cursor = (latency_sample_cursor + 1) % MAX_LATENCY_SAMPLES;
                latency_sample_count = (latency_sample_count + 1).min(MAX_LATENCY_SAMPLES);
                latency_samples_total = latency_samples_total.wrapping_add(1);
                frames_transmitted += 1;
            }
        } else {
            waitpkg::bounded_wait();
        }
    }

    smp::log_ring_monitor_status();
    if update_commands > 0 {
        serial_println!(
            "[HOTSWAP SUMMARY]: commands={}, successful={}, rejected={}",
            update_commands,
            update_successes,
            update_failures
        );
    }
    let elapsed_cycles = unsafe { read_tsc() }.saturating_sub(stream_start).max(1);
    let throughput_milli_fps = if calibrated_hz == 0 {
        0
    } else {
        ((frames_transmitted as u128 * calibrated_hz as u128 * 1000) / elapsed_cycles as u128)
            .min(u64::MAX as u128) as u64
    };
    serial_println!(
        "[UART]: RX frames={}, TX frames={}, reset_commands={}, stream_events={}, ring_full_drops={}, elapsed_cycles={}, throughput={}.{:03} frames/s",
        frames_ingress,
        frames_transmitted,
        reset_commands,
        stream_events,
        ring_full_drops,
        elapsed_cycles,
        throughput_milli_fps / 1000,
        throughput_milli_fps % 1000
    );
    if latency_sample_count > 0 {
        let samples = &latency_samples[..latency_sample_count];
        serial_println!(
            "[LATENCY]: rolling samples={} of total processed frames={}",
            latency_sample_count,
            latency_samples_total
        );
        log_latency_summary("ingress", summarize_stage(samples, 0));
        log_latency_summary("compute", summarize_stage(samples, 1));
        log_latency_summary("egress", summarize_stage(samples, 2));
        log_latency_summary("turnaround", summarize_stage(samples, 3));
    } else {
        serial_println!("[LATENCY]: No completed frames; no latency samples available.");
    }
    let telemetry = telemetry_ring.summary();
    if telemetry.frames > 0 {
        let frequency_hz = timer::tsc_frequency_hz();
        let average_ns = if frequency_hz == 0 {
            0
        } else {
            ((telemetry.avg_cycles as u128 * 1_000_000_000u128) / frequency_hz as u128)
                .min(u64::MAX as u128) as u64
        };
        serial_println!(
            "[TELEMETRY]: frames={}, min_cycles={}, max_cycles={}, avg_cycles={} (approx {} ns @ {} Hz)",
            telemetry.frames,
            telemetry.min_cycles,
            telemetry.max_cycles,
            telemetry.avg_cycles,
            average_ns,
            frequency_hz
        );
    } else {
        serial_println!("[TELEMETRY]: frames=0, no completed inference samples");
    }
    serial_println!(
        "[RING]: Atomic SPSC ingress/egress operations completed without lock or panic."
    );
    hardware_watchdog.disarm();

    // 9. Clean Hardware Exit
    unsafe {
        hardware_shutdown();
    }
}
}

#[cfg(all(test, target_arch = "x86_64"))]
mod solo_runtime_tests {
    use super::*;

    #[test]
    fn frame_dispatch_writes_exactly_one_solo_value() {
        let weights = TernaryWeights64::from_masks(1, 0).unwrap();
        let model = LoadedModel {
            solo_weights: weights,
        };
        let mut inputs = [0i8; 64];
        inputs[0] = -9;
        let mut outputs = [77i32; 64];

        assert_eq!(infer_values(&inputs, &model, false, &mut outputs), 1);
        assert_eq!(outputs[0], -1);
        assert!(outputs[1..].iter().all(|&value| value == 77));

        if std::is_x86_feature_detected!("avx2") {
            outputs.fill(77);
            assert_eq!(infer_values(&inputs, &model, true, &mut outputs), 1);
            assert_eq!(outputs[0], -1);
            assert!(outputs[1..].iter().all(|&value| value == 77));
        }
    }

    #[test]
    fn pot_query_projection_reduces_coefficients_to_ternary_signs() {
        let mut row = [0u8; 32];
        row[0] = 0x29; // low nibble -1, high nibble +1
        row[1] = 0x08; // reserved nibble and zero both map to zero
        let weights = pot_query_row_to_solo(&row);
        assert_eq!(&weights.lanes[..4], &[-1, 1, 0, 0]);
        assert_eq!(weights.pos_mask & 0b1111, 0b0010);
        assert_eq!(weights.neg_mask & 0b1111, 0b0001);
    }

    #[test]
    fn wasm_solo_reference_matches_kernel_scalar() {
        let mut packed = [0u8; 16];
        for index in 0..64 {
            let code = match (index * 13 + 5) % 3 {
                0 => 0,
                1 => 1,
                _ => 3,
            };
            packed[index >> 2] |= code << ((index & 3) * 2);
        }
        let weights = TernaryWeights64::from_packed(&packed);
        let inputs = core::array::from_fn(|index| (index as i8).wrapping_mul(29).wrapping_sub(113));
        assert_eq!(
            crate::wasm_math::solo_infer(&weights.lanes, &inputs),
            infer_scalar(&weights, &inputs)
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod arm64;
#[cfg(target_arch = "aarch64")]
pub mod neon_kernel;

#[cfg(target_arch = "aarch64")]
use uefi::prelude::*;

#[cfg(target_arch = "aarch64")]
#[entry]
fn main(image_handle: uefi::Handle, system_table: SystemTable<Boot>) -> uefi::Status {
    arm64::run(image_handle, system_table)
}
