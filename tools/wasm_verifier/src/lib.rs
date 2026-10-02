#![cfg_attr(not(feature = "std"), no_std)]

#[allow(dead_code)]
mod pure;
pub use pure::solo_infer;

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn neur_solo(inputs: *const i8, weights: *const i8, output: *mut i32) -> i32 {
    if inputs.is_null() || weights.is_null() || output.is_null() {
        return -1;
    }
    let inputs = &*(inputs as *const [i8; 64]);
    let weights = &*(weights as *const [i8; 64]);
    *output = solo_infer(weights, inputs);
    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn neur_alloc(size: usize, alignment: usize) -> *mut u8 {
    extern "C" {
        static __heap_base: u8;
    }
    static mut NEXT: usize = 0;
    let alignment = alignment.max(1).next_power_of_two();
    unsafe {
        let start = if NEXT == 0 {
            &__heap_base as *const u8 as usize
        } else {
            NEXT
        };
        let aligned = (start + alignment - 1) & !(alignment - 1);
        let end = aligned.saturating_add(size);
        let current_pages = core::arch::wasm32::memory_size(0);
        let required_pages = end.div_ceil(65536);
        if required_pages > current_pages {
            let delta = required_pages - current_pages;
            if core::arch::wasm32::memory_grow(0, delta) == usize::MAX {
                return core::ptr::null_mut();
            }
        }
        NEXT = end;
        aligned as *mut u8
    }
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::solo_infer;

    #[test]
    fn solo_matches_independent_integer_reference() {
        let weights = core::array::from_fn(|index| match index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        });
        let inputs = core::array::from_fn(|index| (index as i8).wrapping_mul(37).wrapping_sub(91));
        let sum = inputs
            .iter()
            .zip(weights.iter())
            .map(|(&input, &weight)| input as i32 * weight as i32)
            .sum::<i32>();
        let expected = (sum > 0) as i32 - (sum < 0) as i32;
        assert_eq!(solo_infer(&weights, &inputs), expected);
    }

    #[test]
    fn solo_is_deterministic_and_retains_no_state() {
        let weights = [1i8; 64];
        let positive = [1i8; 64];
        let negative = [-1i8; 64];
        let first = solo_infer(&weights, &positive);
        assert_eq!(solo_infer(&weights, &negative), -1);
        assert_eq!(solo_infer(&weights, &positive), first);
        assert_eq!(first, 1);
        assert_eq!(solo_infer(&[0; 64], &positive), 0);
    }
}
