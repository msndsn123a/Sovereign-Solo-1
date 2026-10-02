> [!IMPORTANT]
> **Live Proof & Playground:** [https://msndsn123a.github.io/Sovereign-Solo-1/](https://msndsn123a.github.io/Sovereign-Solo-1/)
>
> Client-side verification of deterministic scalar inference: **64 signed inputs → 1 scalar decision**, with parity delta `0`. The lab uses browser-native WebAssembly and JavaScript only—no CDN, installed package, external service, or server-side inference dependency.

# Sovereign-Solo: Sub-5ns Deterministic Bare-Metal Neural Inference Appliance

Sovereign-Solo is a standalone x86-64 appliance for deterministic, integer-only neural decisions under UEFI. The firmware runs without an operating system or Rust heap allocator (`#![no_std]`) and evaluates each input frame independently.

## Architectural Overview

### Core concept

A bare-metal, zero-state inference engine for low-latency decision-making directly on x86-64 silicon. Every call is independent: there is no recurrent memory, stream-specific state, or hidden history carried between frames.

### Compute contract

The model-facing operation is pure feed-forward evaluation:

$$
f(\mathbf{x}, \mathbf{W}) \rightarrow y \in \{-1, 0, +1\}
$$

- **Input:** 64 quantized signed bytes, `[i8; 64]`.
- **Weights:** 64 pre-decoded ternary coefficients in `TernaryWeights64`, a 64-byte-cache-line-aligned representation. Positive and negative masks are retained alongside two 32-byte AVX2 lane vectors.
- **Output:** one deterministic, branchless `i32` decision: `-1`, `0`, or `+1`.
- **State and allocation:** no inference state is retained across calls; the compute path uses fixed storage and performs no heap allocation or floating-point arithmetic.

Signed NEUR v2 shards are authenticated and validated before projection to the Solo row. MLP shards use layer-1 row 0; ternary attention shards use Q row 0; signed power-of-two Q coefficients are mapped to their ternary sign. The original shard format and Ed25519 verification remain in the loader; the runtime compute result is always one scalar.

### Microarchitectural path

- **Measured execution:** the host RDTSC microbenchmark has observed a 13–14 TSC-cycle minimum per inference. At a 3.0 GHz TSC this corresponds to approximately 4.3–4.7 ns. This is a machine- and measurement-specific microbenchmark, not a guaranteed latency on every processor or firmware configuration.
- **Inline AVX2 reduction:** the fast path uses inline assembly and two YMM loads, `vpabsb`/`vpsignb` byte operations, pairwise multiply-adds, and horizontal additions to reduce 64 input/weight products to one accumulator. The fixed register sequence avoids an out-of-line kernel call in the hot path.
- **Branchless decision:** sign extraction uses comparison-to-integer arithmetic to produce `-1/0/+1` without conditional branches. CPUs without available AVX2/YMM state use the scalar `infer_scalar` fallback.
- **Cache behavior:** weights are aligned to a 64-byte cache line and the loader warms the model data. This favors cache residency; ordinary cache placement is not a guarantee that a line remains pinned in L1.

## Latency Profiling: Bare-Metal RDTSC vs. WebAssembly

The appliance's RDTSC microbenchmark observes the hardware time-stamp counter around repeated Solo calls and reports cycles per inference. The observed 13–14 cycle minimum corresponds to roughly 4.3–4.7 ns when interpreted at a 3.0 GHz TSC. RDTSC provides hardware counter ticks without browser timer quantization, although serialization, harness work, CPU frequency behavior, virtualization, and firmware still affect measurements.

The browser verifier uses `performance.now()`, which is deliberately quantized or coarsened by browser privacy and side-channel mitigations. A short isolated call may therefore display as `0.00 µs`, or jump in timer steps such as `0.781 µs`; that is the clock's resolution, not proof that the code took no time. The verifier batches **10,000 Wasm calls**, converts the elapsed milliseconds to nanoseconds per call, and reports the batch average to reduce timer quantization error.

The browser figure measures the Wasm scalar reference plus browser scheduling and JS/Wasm boundary overhead. Batching makes that average more readable, but it does **not** turn the Wasm implementation into the AVX2 inline-assembly appliance kernel or establish a sub-10 ns bare-metal timing result. Use the serialized RDTSC host test for the kernel cycle measurement; treat both values as measurements tied to their respective environments.

## Bare-Metal Silicon and Appliance Foundation

The inference kernel runs within a fixed-storage UEFI application. The appliance foundation includes:

- **UEFI and memory map:** firmware boot, `ExitBootServices`, and a custom x86-64 CR3 identity map. The paging code uses 1 GiB huge pages where supported and falls back to 2 MiB pages. Fixed buffers and no demand-paged runtime avoid intentional paging activity in the inference path; hardware or firmware faults are not claimed to be impossible.
- **Core isolation and scheduling:** ACPI processor discovery and xAPIC Startup IPIs (SIPIs) can bring up secondary processors. Intel L3 Cache Allocation Technology (CAT) is configured when supported; CAT partitions cache allocation but does not pin individual lines.
- **Storage and security:** bounded polled-mode NVMe reads, DMA-aligned shard buffers, and Ed25519 verification of signed model payloads. Intel TME and AMD SME capabilities are detected and reported; strict memory-encryption policy is optional and platform-dependent.
- **Transport and I/O:** fixed-capacity SPSC rings, PCIe MMIO/IVSHMEM mailbox support, and binary UART framing. The appliance operates without interrupt-driven inference scheduling.
- **Model lifecycle:** authenticated shadow-shard hot-swap preserves the active model until the candidate is validated and atomically activated.

## Developer Guide and Operational Workflows

### Prerequisites

- Rust Nightly toolchain.
- `x86_64-unknown-uefi` target for the appliance.
- `wasm32-unknown-unknown` target for the browser verifier.
- Windows MSVC target/host runner for the documented host tests.
- Optional: QEMU and OVMF firmware for appliance integration tests.

Install the targets for Nightly if they are not already available:

```powershell
rustup +nightly target add x86_64-unknown-uefi wasm32-unknown-unknown
```

### Host unit tests and cycle benchmark

```powershell
cargo +nightly test --bin neural_box_core --target x86_64-pc-windows-msvc -- --nocapture
```

The tests cover scalar/AVX2 parity, deterministic zero-state calls, scalar frame dispatch, and ternary projection. On an AVX2-capable host, the RDTSC test prints a minimum cycles-per-inference measurement; host measurements can vary with virtualization, scheduling, and processor frequency.

### Bare-metal UEFI compilation

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
```

For a fast compile check without producing the optimized EFI artifact:

```powershell
cargo +nightly check --target x86_64-unknown-uefi
```

### Building the WebAssembly verification lab

```powershell
cargo +nightly build --locked --manifest-path tools/wasm_verifier/Cargo.toml --no-default-features --target wasm32-unknown-unknown --release
Copy-Item tools/wasm_verifier/target/wasm32-unknown-unknown/release/wasm_verifier.wasm tools/wasm_verifier/wasm_verifier.wasm -Force
node tools/wasm_verifier/serve.js
```

Open `http://127.0.0.1:8000/` and run **Run scalar parity suite**. The page exercises five vectors—including zero, signed-byte extremes, and deterministic patterns—and displays the input, scalar result, and a parity table. The browser reference and Wasm output are expected to have `delta = 0` for every case. Browser timing includes Wasm/JavaScript boundary overhead and is not the appliance RDTSC measurement.

### Appliance image and integration tests

With the release EFI binary and OVMF available, package and exercise the bare-metal image:

```powershell
python tools/package_image.py
.\tools\test_dual_volume.ps1
.\tools\test_uart_roundtrip.ps1
```

Additional scripts cover authenticated hot-swap, MMIO/IVSHMEM ingress, and repeated stateless UART decisions. See `DEPLOYMENT.md` for image layout, key handling, and deployment cautions.

## Hardware Specifications and Target Matrix

| Component | Minimum bring-up configuration | Peak/production target | Notes |
| --- | --- | --- | --- |
| CPU architecture | x86-64 CPU with UEFI support; scalar fallback available | x86-64 with AVX2 and operating-system/firmware-enabled XMM/YMM state | AVX2 is detected before dispatch; the scalar path remains available. |
| Cache and weight layout | 64-byte cache-line alignment for `TernaryWeights64` | L1 data cache available for hot weights/input; model warming and CAT partitioning where supported | Alignment and cache warming improve locality but do not lock cache lines. |
| Firmware | UEFI 2.x-compatible firmware with `ExitBootServices` support | Production firmware with reliable memory map, ACPI tables, and tested boot path | OVMF is suitable for development and QEMU tests. |
| Paging | x86-64 page tables and 2 MiB huge-page support | 1 GiB huge pages plus 2 MiB fallback | MMIO ranges receive the required device mappings. |
| Storage | PCIe NVMe namespace supported by the polled driver | NVMe with stable polling behavior and capacity for signed model shards | Block size and shard bounds are validated before reads. |
| Security | Ed25519 verification and a trusted embedded public key | Protected signing key management; optional active TME/SME per platform policy | TME/SME support and activation are hardware/firmware dependent. |
| Inter-core operation | Single BSP core is sufficient for Solo inference | ACPI-described SMP with xAPIC SIPI startup and optional CAT isolation | Secondary cores are used for supporting appliance tasks, not to create inference state. |
| Frame transport | UART binary frames | PCIe MMIO/IVSHMEM ring ingress plus UART diagnostics/control | Fixed-capacity rings avoid heap allocation in the appliance. |

## UART Scalar Protocol

- **Input:** `NB` followed by exactly 64 raw signed `i8` bytes. The tagged form is `NS`, one stream-tag byte, then 64 bytes; the tag does not select inference state.
- **Output:** `NR`, a dimension byte equal to `1`, followed by one little-endian `i32` action (`-1`, `0`, or `1`).
- **Control:** selected-tag `NR` is a stateless no-op; standalone `NR` ends the session. `NU` plus an 8-byte little-endian NVMe LBA queues a candidate shard update.

## Repository Hygiene

The repository ignores generated `dist/`, Cargo `target/`, intermediate `build/`, Python caches, local secrets, and firmware output copies. The checked-in `tools/wasm_verifier/wasm_verifier.wasm` is the intentional browser artifact.

## License

Licensed under either the MIT License (`LICENSE-MIT`) or Apache License, Version 2.0 (`LICENSE-APACHE`).
