[![Live Proof & Playground](https://img.shields.io/badge/Live-Proof%20%26%20Playground-1fb8a5?logo=webassembly&logoColor=white)](https://msndsn123a.github.io/Sovereign-Solo-1/)

# Sovereign-Solo: Sub-5ns Deterministic Bare-Metal Neural Inference Appliance

**Compute contract:** stateless scalar evaluation, `64 × i8 → {-1, 0, +1}`; the AVX2 path has measured 13–14 TSC cycles (about 4.3–4.7 ns when interpreted at a 3.0 GHz TSC) in the repository's RDTSC microbenchmark under x86-64.

The live WebAssembly lab runs client-side scalar parity checks: 64 signed inputs produce one scalar decision and the tested delta must be zero. It uses browser-native WebAssembly and JavaScript, with no CDN, installed package, external service, or server-side inference dependency. The lab verifies deterministic output; its browser timer does not measure bare-metal AVX2 latency.

## 1. System Overview

Sovereign-Solo is a standalone x86-64 UEFI appliance. Firmware is Rust `#![no_std]`, has no Rust heap allocator, and evaluates each input frame independently. The active runtime result is one branchless scalar decision. Fixed buffers, the NVMe loader, shard verifier, shared mailbox, UART framing, and hot-swap machinery provide the appliance environment.

### Inference data path

1. Read a signed model shard from UEFI `SimpleFileSystem`, or probe the supported raw-NVMe fallback locations if no file is found.
2. Validate metadata and payload bounds, verify the Ed25519 signature, and reject invalid ternary/PoT codes.
3. Decode the selected 64-input coefficient row once into `TernaryWeights64`, a 64-byte-aligned structure containing positive/negative masks and two 32-byte AVX2 lane vectors.
4. For each frame, calculate the dot product of the 64 signed-byte inputs and the predecoded ternary row.
5. Convert the accumulator to `-1`, `0`, or `+1` without retaining state between frames; publish one `i32` result.

The AVX2 implementation uses inline assembly (`vpabsb`, `vpsignb`, multiply-add and horizontal-add instructions) over two YMM vectors. CPU and enabled-vector-state checks gate that path; `infer_scalar` is the fallback. The measured host RDTSC minimum has been 13–14 cycles. At a nominal 3.0 GHz TSC this is about 4.3–4.7 ns, but it is a measurement of a particular host/test setup—not a guaranteed physical-device latency. See [Latency Profiling](#latency-profiling).

### Model-to-scalar projection

The firmware currently consumes **signed NEUR v2** shards. It does not currently accept a native `NEUR-SOLO` v3 or an independently signed 16-byte row-only shard. After validating a full supported payload, it projects:

- MLP: layer-1 row 0 (64 ternary coefficients).
- Ternary attention: Q row 0 (64 ternary coefficients).
- PoT attention: Q row 0 mapped to ternary signs; coefficient magnitudes and attention scale are not used by the scalar kernel.

Consequently, the exporter and ONNX converter produce full legacy-format models, while the appliance runtime computes only from the selected row. Training accuracy for the full 64→32→16 network does not by itself establish accuracy of this row-projection decision rule.

## 2. Model Training and Quantization (`ml/`)

### Current trainer and objective

`ml/train_pure_mlp.py` is a deterministic, CPU-only PyTorch QAT trainer for a **64→32→16** ternary MLP. With no external dataset argument, it creates seeded synthetic training data: a seeded random ternary teacher creates labels, and the trainable model learns those labels. It is not an ONNX ingestion tool.

The model has two bias-free linear layers. Weights pass through a ternary straight-through estimator (STE): values above `+0.5` map to `+1`, values below `-0.5` map to `-1`, and the interval between maps to zero. The hidden activation is integer-style hard sign: positive to `+1`, negative to `-1`, and zero to `0`. Gradients use an STE mask; training itself uses floating-point PyTorch arithmetic and is an offline operation.

The objective is cross-entropy plus a teacher-weight anchor:

$$
L = L_{CE} + \lambda\left(\|W_1-W_{teacher,1}\|_2^2 + \|W_2-W_{teacher,2}\|_2^2\right)
$$

This implementation does not apply a separate clipping pass. The forward quantizer and anchor term are the constraints used by this trainer. The appliance inference path is integer-only and stateless.

Defaults: seed `2026`, 12 epochs, 2,048 generated training samples, 512 test samples, batch size 128, learning rate `0.001`, and anchor weight `10.0`. It writes a checkpoint and JSON metrics.

```powershell
python ml/train_pure_mlp.py --checkpoint dist/pure_mlp_qat.pt --metrics dist/pure_mlp_qat_metrics.json
```

The trainer also accepts `--seed`, `--epochs`, `--train-samples`, `--test-samples`, `--batch-size`, `--learning-rate`, and `--anchor-weight`. PyTorch must be installed in the selected Python environment.

### Quantization and storage encoding

- Runtime inputs are signed bytes in the full `i8` range `[-128, 127]`. The synthetic trainer currently trains with values in `[-4, 4]`; that is a training-data choice, not a runtime input restriction.
- Each ternary coefficient is encoded in two bits, four coefficients per byte, low index first: `00` = 0, `01` = +1, `11` = −1. `10` is invalid and rejected by payload validation.
- A 64-element row occupies 16 packed bytes on disk. `TernaryWeights64::from_packed` decodes it once during shard loading/hot-swap into a 64-byte-aligned runtime representation. The complete in-memory struct also stores positive/negative masks; it is not merely a 64-byte file record.
- `ml/export_pure_shard.py` quantizes both network layers using the trainer's ternary STE, validates packed weights, and writes a full MLP shard plus deterministic input/output sidecars. The sidecar records `test_batch_output_i32` for full-model outputs and `test_batch_solo_i32` for row-0 scalar projections.

Exporter command and defaults:

```powershell
python ml/export_pure_shard.py --checkpoint dist/pure_mlp_qat.pt --output dist/trained_pure_mlp_shard.bin --expected dist/trained_pure_mlp_expected.json --block-size 512
```

Options are `--checkpoint`, `--output`, `--expected`, `--block-size` (`512` or `4096`), `--version` (must be `2`), and optional `--key-file` (raw 32-byte Ed25519 seed). Export requires a compatible `PureTernaryMLP` PyTorch state dict. It checks quantized full-MLP PyTorch outputs against an integer reference, then invokes the Rust payload signer. The Solo row reference is generated separately; full MLP parity is not the Solo inference result.

### ONNX conversion

`tools/onnx2neur/` is a separate native Rust converter. It accepts only Gemm/MatMul linear graphs meeting strict shape/dtype/bias constraints; it does not accept arbitrary ONNX operators. Ternary conversion supports a connected 64→32→16 MLP; PoT conversion supports the specified attention matrices. It emits signed NEUR v2, not a native Solo row shard.

```powershell
cargo +nightly build --manifest-path tools/onnx2neur/Cargo.toml --target x86_64-pc-windows-msvc --release
cargo +nightly run --manifest-path tools/onnx2neur/Cargo.toml --target x86_64-pc-windows-msvc --release -- --input model.onnx --output dist/model.neur --key-file signing.seed --quant ternary --threshold 0.25 --block-size 512
```

`--key-file` is required by this converter. `--threshold` defaults to `0.25`; `--pot-scale` accepts `0..6`; `--block-size` accepts powers of two from 512 through 4096. Its own README documents accepted graph shapes and restrictions.

## 3. Cryptographic Packaging and Shard Ingestion (`tools/`)

### Current signed shard layout

The loader currently recognizes **NEUR version 2**, not a version-3 `NEUR-SOLO` header. The 16-byte metadata prefix is followed by a 64-byte Ed25519 signature and then the model payload. Sector/block padding follows the signed bytes and is not part of the signature.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `NEUR` |
| 4 | 4 | Version, little-endian (`2` accepted by firmware) |
| 8 | 4 | Input dimension, little-endian (`64`) |
| 12 | 1 | Model type (`0` MLP, `1` causal attention) |
| 13 | 2 | Output metadata, little-endian: low 12 bits output dimension (`16`); high bits carry PoT scale and activation-LUT flag |
| 15 | 1 | Hidden/attention dimension and model-specific flags |
| 16 | 64 | Ed25519 signature |
| 80 | variable | Model payload |

The signature message is `metadata_prefix[0..16] || payload`; the signature field and LBA padding are excluded. Supported payloads are 640 bytes for dense MLP, 688 for sparse MLP, 832 for ternary attention, and 1,664 for PoT attention. A signed dense MLP record is 720 bytes before block padding (normally 1,024 bytes at 512-byte sectors).

The DMA destination is a 4 KiB-aligned buffer. Shards are padded by the builder/exporter to the selected logical-block size (512 or 4096 bytes); padding does not change logical payload size. The loader rejects records whose recognized header and payload do not fit in its 4 KiB transfer buffer. The parser does not accept a signed 16-byte-payload `NEUR-SOLO` record.

### Ed25519 signing and key handling

`src/crypto.rs` verifies Ed25519 over metadata plus payload before a candidate model is promoted. It embeds the RFC 8032 test-vector public key. The Rust payload builder defaults to the matching deterministic test seed, which is public and **must not be treated as a production secret**. Use it only for tests. A custom `--key-file` seed is 32 raw bytes, but firmware will reject its shard until the corresponding public key is installed in source and the EFI image is rebuilt. Protect signing seeds outside the repository; `.gitignore` excludes common key formats.

Build a signed development MLP shard with the existing wrapper:

```powershell
& (Join-Path $PWD 'tools/payload_builder/build_payload.ps1') -outPath dist/production_shard.bin -blockSize 512 -model mlp -variant pattern
```

The wrapper accepts `-outPath`, `-blockSize`, `-model mlp|attention`, `-variant pattern|zero`, `-quant ternary|pot`, `-potScale 0..6`, `-multiStream`, `-pruneBlocks`, `-activationLut`, and `-keyFile`. Invoke its Python frontend directly with:

```powershell
python tools/payload_builder/build_payload.py dist/production_shard.bin --model mlp --variant pattern --quant ternary --block-size 512
```

The Rust CLI also accepts `--unsigned-input` and signs a 16-byte metadata prefix plus the recognized payload size. Firmware remains the authoritative dimension, payload, encoding, and signature validator; the signer does not replace those checks.

### GPT image provisioning and storage

`tools/package_image.py` creates a 512-byte-sector GPT image with two FAT32 partitions and an optional raw update region:

| Region | LBA range (512-byte sectors) | Purpose |
|---|---:|---|
| ESP | 2048–133119 | UEFI application and startup script |
| `NEURAL_DATA` | 133120–264191 | `weights.bin` file read by UEFI |
| Reserved raw update area | begins at 264192 | Optional pre-staged signed hot-swap shard |

Each FAT32 partition is 64 MiB; the image is 130 MiB. The raw update area is not a second model partition or A/B filesystem slot. Hot-swap uses two in-RAM model buffers/slots and an NVMe LBA update request. The packager writes an optional update shard at LBA 264192 in 512-byte units; firmware interprets the request in device logical-block units. QEMU tests use 512-byte logical blocks. For 4 KiB logical-block NVMe, convert the byte offset to device block units rather than reusing the 512-byte sector number.

Package an image after building the EFI binary and a valid signed shard:

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
python tools/package_image.py --efi-path target/x86_64-unknown-uefi/release/neural_box_core.efi --shard-path dist/production_shard.bin --output dist/neural_box_appliance.img
```

`tools/package_image.ps1` exposes `-efiPath`, `-shardPath`, and `-outPath`. `tools/package_image.py` also accepts `--hot-swap-shard <signed-file>` to write an update record into the reserved raw-LBA region. The packager checks basic magic/version/minimum size; firmware signature and payload validation are authoritative.

### Loader behavior and runtime verification

Before `ExitBootServices`, the loader searches UEFI `SimpleFileSystem` handles for `\weights.bin` and `\NEURAL_WEIGHTS\weights.bin` using fixed buffers, then validates a found shard. A present but invalid file is rejected to the safe identity Solo model; it does not then try raw-NVMe fallback. Only if no file is found does fallback probe fixed byte offsets (133120, 34816, 67584, and 0) multiplied by 512; it is not a general GPT/partition scanner.

For hot-swap, UART `NU` supplies an 8-byte little-endian raw NVMe LBA. An auxiliary AP reads and validates the candidate in the inactive DMA/model buffer. The active slot changes only after signature/payload verification and reader coordination; failure preserves the active model. This is an in-memory shadow swap, not disk Slot A/Slot B selection, and it requires a suitable secondary AP.

## 4. Hardware Injection and Test Harnesses (`tools/`)

### UART protocol

The current binary UART frame protocol is **not** `SO`/`SR`; it is defined by `src/serial.rs`:

| Direction | Bytes |
|---|---|
| Input | `NB` (`0x4E 0x42`) + 64 raw signed bytes = 66 bytes |
| Tagged input | `NS` + stream tag (0–7) + 64 raw signed bytes = 67 bytes; the tag does not select inference state |
| Output | `NR` + dimension byte `1` + one little-endian `i32` = 7 bytes |
| Reset/exit | `NR` followed by a selector is a stateless no-op; standalone `NR` exits after the parser ambiguity timeout |
| Hot-swap request | `NU` + 8-byte little-endian raw NVMe LBA = 10 bytes |

The fixed-length UART parser implements no checksum, escaping, or per-frame sequence number. COM2's startup handshake is `UART_READY\n`. `tools/test_uart_roundtrip.ps1` validates the NEUR header/hash, sends the exporter's eight vectors, and compares scalar responses byte-for-byte.

### Host IPC and MMIO mailbox transports

Host injectors do **not** send SO/SR frames. They use the version-1 shared-mailbox ABI: the producer publishes raw `[i8; 64]` input in a slot; the guest publishes output metadata with `output_dim = 1` and the scalar in `values[0]`. The mailbox magic is `SHMB`, capacity 16, with atomic head/tail counters, 128-byte input slots, and 384-byte output slots. The output array remains `[i32; 64]` for ABI stability; only its first value is valid for Solo.

- `tools/host_injector.rs` is Windows-only. It maps an existing 64 KiB backing file, reads `test_batch_i8` and `test_batch_solo_i32` from expected JSON, sends 1–16 raw inputs, and validates one scalar per response.
- `tools/mmio_injector.rs` is Windows-only and maps a backing file. It currently sends all-zero 64-byte inputs and requires dimension one and output zero.
- Neither injector creates the QEMU device. Use the matching bridge script. MMIO integration requires QEMU `memory-backend-file` and `ivshmem-plain` support.

Build and invoke the Windows injectors directly (the backing files must already be mapped by the matching QEMU setup):

```powershell
rustc --edition=2021 -O tools/host_injector.rs -o dist/host_injector.exe
& (Join-Path $PWD 'dist/host_injector.exe') dist/shm_mailbox.bin dist/trained_pure_mlp_expected.json
rustc --edition=2021 -O tools/mmio_injector.rs -o dist/mmio_injector.exe
& (Join-Path $PWD 'dist/mmio_injector.exe') dist/ivshmem_mailbox.bin 8
```

### PowerShell/QEMU script catalog

Run examples from the repository root. QEMU harnesses require `assets/OVMF.fd`, the target EFI build, and their listed inputs.

| Script | Purpose and example |
|---|---|
| `ml/run_pure_pipeline.ps1` | Train → export/sign → EFI build → UART roundtrip: `& (Join-Path $PWD 'ml/run_pure_pipeline.ps1') -Python C:/path/to/python.exe -Port 5558 -QemuAccel whpx -BlockSize 512`. The checked-in default Python path is machine-specific; pass Python with PyTorch installed. |
| `tools/test_uart_roundtrip.ps1` | Send the exporter's eight signed-byte vectors and compare scalar responses: `& (Join-Path $PWD 'tools/test_uart_roundtrip.ps1') -Port 5556 -QemuAccel whpx -ShardPath dist/trained_pure_mlp_shard.bin -ExpectedPath dist/trained_pure_mlp_expected.json`. Requires shard, sidecar JSON, and EFI file. |
| `tools/test_dual_volume.ps1` | Boot GPT image, test FAT model load and eight zero-input scalar responses: `& (Join-Path $PWD 'tools/test_dual_volume.ps1') -CpuCount 1 -ImagePath dist/neural_box_appliance.img`. `-TamperWeights` tests signature rejection; `-RequireSparseActivation` tests sparse-shard validation/logging. Requires a prebuilt image and EFI. |
| `tools/test_hot_swap.ps1` | Build a signed zero MLP update, package it at a raw LBA, stream frames, and check post-swap zero decisions: `& (Join-Path $PWD 'tools/test_hot_swap.ps1') -CpuCount 2 -UpdateLba 264192`. Defaults are `dist/neural_box_hot_swap.img`, `dist/hot_swap_zero_shard.bin`, and LBA 264192; this script runs its own build/package steps. |
| `tools/test_attention_sequence.ps1` | Despite its legacy filename, verify repeated Solo output across `NS` tags and reset controls: `& (Join-Path $PWD 'tools/test_attention_sequence.ps1') -CpuCount 2 -Port 5569`. Builds a signed PoT-attention shard and checks the Q-row projection path. |
| `tools/test_mmio_pipeline.ps1` | Build/package, start IVSHMEM, run all-zero MMIO injection, and check zero-loss telemetry: `& (Join-Path $PWD 'tools/test_mmio_pipeline.ps1') -FrameCount 8 -MailboxPath dist/ivshmem_mailbox.bin`. Eight frames is coherent because the guest exits the MMIO loop at its fixed eight-frame limit. Requires QEMU file-backed memory and IVSHMEM support. |
| `tools/run_shm_bridge.ps1` | Build with `host-ipc`, create a 64 KiB host-backed guest RAM window, start QEMU, launch host injector: `& (Join-Path $PWD 'tools/run_shm_bridge.ps1') -ShardPath dist/trained_pure_mlp_shard.bin -ExpectedPath dist/trained_pure_mlp_expected.json -MailboxPath dist/shm_mailbox.bin -QemuAccel whpx`. Its declared `-Port` parameter is currently unused. |
| `tools/package_image.ps1` | Package GPT/FAT32 image: `& (Join-Path $PWD 'tools/package_image.ps1') -efiPath target/x86_64-unknown-uefi/release/neural_box_core.efi -shardPath dist/production_shard.bin -outPath dist/neural_box_appliance.img`. |
| `tools/payload_builder/build_payload.ps1` | Build a signed development MLP shard: `& (Join-Path $PWD 'tools/payload_builder/build_payload.ps1') -outPath dist/production_shard.bin -blockSize 512 -model mlp -variant pattern -quant ternary`. It accepts `-keyFile` for a raw 32-byte seed; firmware trust still depends on the embedded public key. |

Default QEMU CPU in several scripts is `Skylake-Server,+avx512f,+avx512dq`; individual scripts may override CPU/accelerator/port. `test_uart_roundtrip.ps1` defaults to port 5556, `test_dual_volume.ps1` to 5570, `test_hot_swap.ps1` to 5571, and `test_attention_sequence.ps1` to 5569. Use distinct ports for concurrent harnesses.

### Host IPC mailbox ABI details

The mailbox header occupies bytes `0..63`: magic/version at 0/4 and four `u64` ring cursors at 8, 16, 24, and 32. Input slots start at byte 64; there are 16 × 128-byte slots. Each input slot has state at +0, T0 at +8, and payload at +64. Output slots start at byte 2112; there are 16 × 384-byte slots. Each output slot has state at +0, values at +64, metadata at +320 (`output_dim`, `t0_ready`, `t1_ingest`, `t2_compute`), and `t3_commit` at +352. Total mailbox size is 8,256 bytes within the 64 KiB host-backed window. Host round-trip time is wall-clock timing; the guest records its own TSC stage values.

## 5. Developer Operations and Build Recipes

### Prerequisites

- Rust Nightly and `x86_64-unknown-uefi` (`rustup +nightly target add x86_64-unknown-uefi`).
- Windows MSVC host target/runner for the documented host test.
- Python 3 and PyTorch for training under `ml/`; neither is required by firmware runtime.
- `wasm32-unknown-unknown` for the browser lab (`rustup +nightly target add wasm32-unknown-unknown`).
- QEMU and OVMF for QEMU harnesses. MMIO additionally needs QEMU `memory-backend-file` and `ivshmem-plain`.

### Host unit tests and RDTSC benchmark

```powershell
cargo +nightly test --bin neural_box_core --target x86_64-pc-windows-msvc -- --nocapture
```

On a host advertising AVX2 and enabled YMM state, tests compare scalar/AVX2 decisions, check deterministic stateless calls and frame dispatch, and print a minimum cycles-per-call RDTSC result. This is host evidence, not a target-hardware worst-case guarantee.

### UEFI build

```powershell
cargo +nightly check --target x86_64-unknown-uefi
cargo +nightly build --target x86_64-unknown-uefi --release
```

### Wasm verifier build and browser check

```powershell
cargo +nightly build --locked --manifest-path tools/wasm_verifier/Cargo.toml --no-default-features --target wasm32-unknown-unknown --release
Copy-Item tools/wasm_verifier/target/wasm32-unknown-unknown/release/wasm_verifier.wasm tools/wasm_verifier/wasm_verifier.wasm -Force
node tools/wasm_verifier/serve.js
```

Visit `http://127.0.0.1:8000/`, run **Run scalar parity suite**, and confirm five vectors have delta zero. Browser timing averages 10,000 calls because `performance.now()` may be coarsened; it includes browser/runtime overhead and is not the bare-metal RDTSC result.

### End-to-end training, signing, packaging, and injection

The pipeline wrapper runs training, export/signing, EFI build, and UART roundtrip:

```powershell
& (Join-Path $PWD 'ml/run_pure_pipeline.ps1') -Python C:/path/to/python.exe -Port 5558 -QemuAccel whpx -BlockSize 512
```

For separate artifact inspection, use the trainer/exporter commands in Section 2, inspect the signed shard header/hash, package with `tools/package_image.py`, then run the desired UART, host-IPC, MMIO, or hot-swap harness. Keep the development signing seed out of production workflows; replace the embedded verification key before accepting production signatures.

## 6. Operational Boundaries and Limitations

- The firmware accepts signed NEUR v2 full-model formats. It does not accept `NEUR-SOLO` v3, an independently signed 16-byte row-only shard, or SO/SR framing. Current UART markers are `NB`, `NS`, `NR`, and `NU`; host IPC/MMIO use the version-1 `SHMB` mailbox.
- The disk image contains an ESP and `NEURAL_DATA` FAT32 partition. Hot-swap uses a raw-LBA region and two in-RAM model slots; there are no disk Slot A/Slot B model partitions.
- Raw NVMe startup fallback probes fixed offsets; it is not a partition scanner. If a filesystem shard exists but is invalid, firmware selects the safe identity model rather than scanning fallback LBAs.
- The packager's hot-swap LBA is in 512-byte sectors; runtime `NU` is interpreted in NVMe logical blocks. QEMU tests use 512-byte blocks; convert byte offsets for a 4 KiB logical-block device.
- The built-in signing seed and firmware public key are a known development/test pair. A custom `--key-file` seed is not trusted until its matching public key is embedded in firmware and the EFI image is rebuilt.
- `tools/onnx2neur` accepts constrained Gemm/MatMul-only graphs, not arbitrary ONNX models. The PyTorch exporter expects its defined checkpoint layout.
- The 13–14 TSC-cycle observation is environment-specific. It does not establish fixed worst-case latency under QEMU, a browser, interrupts, cache/DRAM misses, thermal changes, or all UEFI platforms.

## License

Licensed under either the MIT License (`LICENSE-MIT`) or Apache License, Version 2.0 (`LICENSE-APACHE`).
