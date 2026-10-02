# Sovereign-Solo: Sub-10ns Deterministic Bare-Metal Scalar Inference

Sovereign-Solo is a `no_std` x86-64 UEFI appliance with a stateless, integer-only inference path. Each 64-byte signed input frame produces exactly one ternary scalar action: `-1`, `0`, or `+1`.

## Solo inference architecture

- **Zero-state execution:** each frame is independent; there is no recurrent attention memory, stream bank, or per-frame model state.
- **Predecoded weights:** after signature and payload validation, the loader converts the selected 64-weight projection once into a cache-line-aligned `TernaryWeights64`. The structure holds positive/negative masks and two AVX2-sized coefficient lanes. Hot-swaps perform the same conversion before the atomic model-slot swap.
- **AVX2 fast path:** an inline-assembly integer dot product operates on two YMM vectors and returns one branchless sign decision. Runtime dispatch checks CPU AVX2 support and enabled YMM state; `infer_scalar` is the portable x86 fallback.
- **No floating point or heap:** inference uses fixed-size integer storage and performs no allocation or floating-point arithmetic.
- **Measured latency:** the host RDTSC microbenchmark has measured a 14-cycle minimum (about 4.3 ns at a 3.26 GHz TSC). Results depend on CPU, firmware, virtualization, and measurement conditions; this is a benchmark observation, not a universal latency guarantee. The Wasm verifier includes browser and JS/Wasm boundary overhead and is not a bare-metal measurement.

Existing signed NEUR v2 formats remain accepted. The appliance projects MLP shards to layer-1 row 0, ternary attention shards to Q row 0, and signed PoT Q weights to ternary signs. This makes the inference output a single scalar rather than the original model's multi-output result.

## Bare-metal foundation

Solo preserves the appliance infrastructure around the compute path:

- UEFI boot and `ExitBootServices` memory-map handoff.
- Custom CR3 identity mappings with 1 GiB huge pages where available (2 MiB fallback).
- ACPI processor discovery and xAPIC SIPI startup for secondary processors.
- Polled-mode NVMe reads, DMA-aligned model buffers, and Ed25519 shard verification.
- MMIO/IVSHMEM mailbox and SPSC ring-buffer ingress/egress.
- Signed shadow-model hot-swapping without stopping frame ingress.

## Build and test

Install the nightly Rust toolchain and the UEFI target from `rust-toolchain.toml`.

```powershell
cargo +nightly check --target x86_64-unknown-uefi
cargo +nightly build --target x86_64-unknown-uefi --release
cargo +nightly test --bin neural_box_core --target x86_64-pc-windows-msvc -- --nocapture
```

The host test suite checks scalar/AVX2 parity, stateless determinism, frame dispatch, PoT projection, and prints the RDTSC Solo-kernel benchmark. AVX2 execution is exercised only when the host advertises AVX2.

## Solo WebAssembly verifier

The verifier calls the same stateless ternary scalar reference used for parity checks. It displays the 64 signed input lanes, one scalar decision, and a test table with zero-delta assertions. It has no attention matrix, stream selector, or recurrent state.

Build the standalone Wasm module and refresh the checked-in artifact:

```powershell
cargo +nightly build --locked --manifest-path tools/wasm_verifier/Cargo.toml --no-default-features --target wasm32-unknown-unknown --release
Copy-Item tools/wasm_verifier/target/wasm32-unknown-unknown/release/wasm_verifier.wasm tools/wasm_verifier/wasm_verifier.wasm -Force
node tools/wasm_verifier/serve.js
```

Open `http://127.0.0.1:8000/` and run the scalar parity suite. `tools/wasm_verifier/src/pure.rs` provides the no-std scalar implementation; a root host test verifies its result against `kernel::infer_scalar`.

## Appliance protocol

- Input: `NB` followed by 64 raw signed `i8` bytes, or `NS`, a stream tag byte, and 64 bytes. Stream tags are transport metadata only.
- Output: `NR`, a dimension byte of `1`, then one little-endian `i32` action (`-1`, `0`, or `1`).
- A selected-stream `NR` control is a stateless no-op; a standalone `NR` ends the UART session.
- `NU` plus an 8-byte little-endian NVMe LBA requests an authenticated hot-swap.

The stream and control bytes do not introduce inference state. The scalar result depends only on the active predecoded weights and the current 64-byte input.

## Repository hygiene

Generated `dist/`, Cargo `target/`, intermediate `build/`, Python caches, and local secrets are excluded by `.gitignore`. The checked-in `tools/wasm_verifier/wasm_verifier.wasm` is the intentional browser artifact.

## License

Licensed under either the MIT License (`LICENSE-MIT`) or Apache License, Version 2.0 (`LICENSE-APACHE`).
