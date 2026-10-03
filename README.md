# Sovereign-Solo: Sub-5ns Deterministic Bare-Metal Scalar Inference Appliance

> **[Open the Live Wasm Proof and Interactive Playground](https://msndsn123a.github.io/Sovereign-Solo-1/)** — 64 signed input lanes, one deterministic scalar decision, and an independent integer-parity check.

Sovereign-Solo evaluates a pure feed-forward function, $x\in i8^{64}\rightarrow y\in\{-1,0,+1\}$, without recurrent state, a model server, a network connection, or a runtime heap allocator. The AVX2 compute kernel has reported a 13–14 TSC-tick minimum in host microbenchmarks (about 4.3–4.7 ns if interpreted at a nominal 3.0 GHz TSC). This is an observed compute-only result, not a physical-device guarantee or an end-to-end UART latency promise. The firmware is `no_std`; it still links its UEFI and Ed25519 Rust crates and is not literally free of code dependencies.

## 1. Microarchitecture & Compute Engine (`src/kernel.rs`)

### Function evaluated

The shard contains 64 ternary coefficients $w_i\in\{-1,0,+1\}$. For signed-byte inputs $x_i\in[-128,127]$, the engine computes

$$
s=\sum_{i=0}^{63}x_iw_i,\qquad y=\mathbf{1}_{s>0}-\mathbf{1}_{s<0}.
$$

The result is exactly `+1`, `0`, or `-1`; an accumulator tie produces zero. Since $|s|\le 64\cdot128=8192$, an `i32` accumulator is sufficient. There is no bias, activation stack, recurrent state, or second projection in the Solo inference path.

`TernaryWeights64` is declared `#[repr(C, align(64))]`. It holds 64 decoded `i8` lanes, a positive-weight mask, and a negative-weight mask. The signed shard payload itself is only 16 packed bytes; loading expands it once before inference. Inputs and lanes are evaluated as two 32-byte halves.

### AVX2 path and scalar fallback

`infer_avx2` is handwritten `core::arch::asm!` using fixed YMM/XMM registers and explicit clobbers. The sequence loads two input and two weight chunks (`ymm0`–`ymm3`), takes input magnitudes with `vpabsb`, applies input signs to ternary weights with `vpsignb`, forms pair products with `vpmaddubsw`, widens/combines with `vpmaddwd`, then accumulates and horizontally reduces to an `i32`. `vmovd` returns the accumulator and `vzeroupper` clears upper-vector state. The `-128` input bit pattern is handled by the unsigned/signed multiply-add sequence without saturating its pair sums.

The returned scalar uses the source-level branchless expression `(value > 0) as i32 - (value < 0) as i32`. This describes the Rust expression, not a guarantee about every compiler's emitted threshold instructions; no disassembly-level branchlessness claim is made. `infer_scalar` is the fallback.

**Dispatch detail:** firmware enables the vector state through its AVX-512 OS-state path, then selects AVX2 only when AVX2 CPUID is also present. That state path requires AVX512F/XSAVE support. Consequently, a CPU with AVX2 but without the firmware's required AVX-512-state prerequisites can use scalar inference. The host parity test separately detects AVX2 for its own test.

### Measured baseline and timer limits

The test `solo_kernel_rdtsc_cycle_benchmark` runs 1,024 calls per sample for seven samples and reports the minimum elapsed TSC ticks per call. The test currently enforces an AVX2 ceiling of 50 ticks; **13–14 ticks is a previously observed baseline, not a constant or asserted test result**. At 3.0 GHz, 13 ticks is about 4.33 ns and 14 is about 4.67 ns. TSC ticks need not equal current core-clock cycles; frequency and platform behavior vary. The microbenchmark uses fixed/hot input and weights, includes loop/checksum/measurement overhead, and does not represent cache misses, interrupts, UART time, or worst-case latency.

The browser proof is a separate scalar WebAssembly parity check. Modern browsers commonly quantize `performance.now()` to roughly 0.781 µs steps; one call can therefore display `0.00 µs` or otherwise be below timer resolution. The playground averages a 10,000-pass batch to show a per-call nanosecond profile. That interval includes browser scheduling and the JavaScript/Wasm boundary; it is not a bare-metal RDTSC measurement.

## 2. Dedicated Shard Architecture: Native `NEUR` v2 Solo

The appliance accepts the native Solo model type `MODEL_TYPE_SOLO = 2`, with input dimension 64 and output dimension 1. Legacy MLP/attention records may still be produced by separate conversion tooling, but the firmware shard loader rejects those model types.

### Signed envelope

The metadata prefix is exactly 16 bytes. Ed25519 signs that prefix concatenated with the 16-byte payload; the signature field and sector padding are excluded.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | ASCII magic `NEUR` |
| 4 | 4 | Version `2`, little-endian `u32` |
| 8 | 4 | Input dimension `64`, little-endian `u32` |
| 12 | 1 | Model type `2` (`MODEL_TYPE_SOLO`) |
| 13 | 2 | Output dimension `1`, little-endian `u16` |
| 15 | 1 | Solo flags/reserved byte; must be zero |
| 16 | 64 | Ed25519 signature |
| 80 | 16 | 64 packed ternary weights |
| 96+ | variable | Builder's zero padding to a target block boundary; unsigned |

Thus the **signed record is exactly 96 bytes**: 16 metadata + 64 signature + 16 payload. The payload encodes four weights per byte, low index first: `00` = zero, `01` = +1, `11` = −1; `10` is invalid. The builder supports power-of-two block sizes from 512 through 4096 (512, 1024, 2048, or 4096). The Python exporter intentionally offers 512 or 4096. Padding does not change the signed payload, and the firmware does not require padding bytes to be zero.

After validating the header, signature, and ternary encoding, `src/main.rs` converts the exact 16-byte slice to `[u8; 16]` and calls `TernaryWeights64::from_packed` directly. It does not parse a larger MLP/attention buffer or project row zero at runtime. The production trust key in `src/crypto.rs` is the RFC 8032 test-vector public key; the default signing seed is public test material and is not a production secret. A custom seed requires installing its matching public key in firmware and rebuilding the EFI image.

## 3. Wire & Transport Protocol: `SO` / `SR`

### UART frames

`src/serial.rs` uses fixed-length, unescaped binary frames on COM2:

| Direction | Exact layout | Total |
|---|---|---:|
| Ingress (`SO`) | `[0x53, 0x4F]` (`'S','O'`) followed by exactly 64 raw signed `i8` bytes | **66 bytes** |
| Egress (`SR`) | `[0x53, 0x52]` (`'S','R'`) followed by one little-endian `i32` scalar | **6 bytes** |
| Hot-swap (`NU`) | `[0x4E,0x55]` followed by a little-endian `u64` NVMe LBA | **10 bytes** |

The parser has no checksum, escaping, or sequence number. It recognizes `SO` inference ingress and `NU` update requests; old `NB`, `NS`, and `NR` frame/control handling is not supported. COM2 sends the text readiness token `UART_READY\n` before binary streaming. `NU` is a separate control frame, not an SR response.

### Internal SPSC rings

`src/io_ring.rs` defines separate input and output SPSC rings with capacity 128. Each slot type is 64-byte aligned. `InputFrame` contains the 64-byte vector and two timestamps and is 128 bytes after alignment; `OutputFrame` contains one `i32` plus three timestamps and occupies one 64-byte cache line. Ring publication uses Release/Acquire atomics. These are internal records, not UART wire frames.

### PCIe MMIO and Shared Mailbox ABI V2

MMIO and host-backed shared memory use the same logical 64-byte input / one-scalar output contract, but carry only those values—not the literal UART SO/SR preamble bytes. `SharedMailbox` has magic `SHMB` (`0x53484D42`), version 2, capacity 16, and 64-byte alignment. The header is 64 bytes; input slots start at offset 64 and output slots at offset 2112. Every slot is 128 bytes:

| Slot | Offset within slot | Contents |
|---|---:|---|
| Input | +0 | Atomic state |
| Input | +8 | `t0_ready` (`u64`) |
| Input | +64 | `[i8; 64]` payload |
| Output | +0 | Atomic state |
| Output | +64 | One `i32` scalar |
| Output | +72, +80, +88 | `t0_ready`, `t1_ingest`, `t2_compute` timestamps |
| Output | +96 | `t3_commit` timestamp |

Total mailbox size is 4,160 bytes. Producers write a slot, publish `READY` with Release ordering, and advance the head; consumers observe state/head with Acquire ordering, compute or read the result in place, then release the slot and advance the tail. The host IPC bridge maps a 64 KiB host-backed window at guest physical address `0x1_0000_0000`. MMIO uses the IVSHMEM BAR as the shared backing region. The inbox payload is read directly by the guest compute callback; the scalar is written into the output slot rather than a 64-value array.

## 4. Complete Tooling & Script Catalog

Commands below are PowerShell commands run from the repository root unless noted. Windows-only host injectors require the Windows MSVC Rust toolchain. QEMU integration scripts also need the EFI image, OVMF, and their input artifacts.

### Training and native Solo export

`ml/train_pure_mlp.py` trains a deterministic offline 64→32→16 ternary model on seeded synthetic data; PyTorch is required. Weight quantization maps values above `+0.5` to +1, below `−0.5` to −1, and the interval between to zero. Export selects the first 64-weight row for the native Solo appliance artifact; full-network metrics are not the appliance scalar metric.

```powershell
python ml/train_pure_mlp.py --checkpoint dist/pure_mlp_qat.pt --metrics dist/pure_mlp_qat_metrics.json
python ml/export_pure_shard.py --checkpoint dist/pure_mlp_qat.pt --output dist/trained_pure_solo_shard.bin --expected dist/trained_pure_solo_expected.json --block-size 512
```

Exporter options: `--checkpoint`, `--output`, `--expected`, `--block-size {512,4096}`, `--version 2`, and optional `--key-file` (exactly 32 raw seed bytes). It writes the signed native Solo shard and an expected JSON sidecar containing test vectors and `test_batch_solo_i32`. The exporter uses a temporary unsigned shard when calling the Rust signer and removes it afterwards.

Trainer options are `--checkpoint`, `--metrics`, `--seed` (2026), `--epochs` (12), `--train-samples` (2048), `--test-samples` (512), `--batch-size` (128), `--learning-rate` (0.001), and `--anchor-weight` (10.0). The exporter sidecar includes both `test_batch_output_i32` (offline MLP parity) and `test_batch_solo_i32` (the runtime's one-scalar reference); only the latter is compared as appliance output.

### Payload builder and signer

The native Solo defaults and recommended commands are:

```powershell
python tools/payload_builder/build_payload.py dist/production_shard.bin --model solo --block-size 512 --variant pattern
python tools/payload_builder/build_payload.py dist/production_shard_4kn.bin --model solo --block-size 4096 --variant pattern
& (Join-Path $PWD 'tools/payload_builder/build_payload.ps1') -outPath dist/production_shard.bin -blockSize 512 -model solo -variant pattern
cargo +nightly run --manifest-path tools/payload_builder/Cargo.toml --target x86_64-pc-windows-msvc --release -- --model solo --output dist/production_shard.bin --block-size 512 --variant pattern
```

Python CLI: optional positional output; `--model {solo,mlp,attention}` (default solo), `--block-size` (default 512), `--variant {pattern,zero}`, `--quant {ternary,pot}`, `--multi-stream`, `--pot-scale 0..6`, `--prune-blocks`, `--activation-lut`, and `--key-file`. The PowerShell wrapper exposes corresponding `-outPath`, `-blockSize`, `-model`, `-variant`, `-quant`, `-potScale`, switches `-multiStream`, `-pruneBlocks`, `-activationLut`, and `-keyFile`. Native Solo permits ternary weights only and rejects optional model flags. For Solo, choose 512 or 4096 bytes. The builder accepts powers of two between 512 and 4096; the exporter limits itself to 512/4096. Legacy builder modes do not make those formats loadable by this Solo-only firmware. The default deterministic signing seed is for development/tests only.

### GPT/FAT32 appliance image

`tools/package_image.py` creates a 512-byte-sector GPT image containing an ESP FAT32 partition (LBA 2048–133119), a `NEURAL_DATA` FAT32 partition (LBA 133120–264191) with `weights.bin`, and an optional raw update region beginning at LBA 264192. Each FAT32 partition is 64 MiB. It embeds the EFI file and signed shard; it checks Solo header fields but firmware remains authoritative for signature and payload validation. If the default production shard is absent or incompatible, the packager generates a native Solo default.

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
python tools/package_image.py --efi-path target/x86_64-unknown-uefi/release/neural_box_core.efi --shard-path dist/production_shard.bin --output dist/neural_box_appliance.img
python tools/payload_builder/build_payload.py dist/hot_swap_zero_shard.bin --model solo --block-size 512 --variant zero
python tools/package_image.py --efi-path target/x86_64-unknown-uefi/release/neural_box_core.efi --shard-path dist/production_shard.bin --output dist/neural_box_hot_swap.img --hot-swap-shard dist/hot_swap_zero_shard.bin
& (Join-Path $PWD 'tools/package_image.ps1') -efiPath target/x86_64-unknown-uefi/release/neural_box_core.efi -shardPath dist/production_shard.bin -outPath dist/neural_box_appliance.img
```

Python options are `--efi-path`, `--shard-path`, `--output`, and optional `--hot-swap-shard`. The PowerShell wrapper accepts `-efiPath`, `-shardPath`, `-outPath`. Image partition geometry is in 512-byte sectors; an NVMe `NU` request is in the device's logical block units. Convert the byte offset when using 4 KiB logical blocks.

### Host IPC and MMIO injectors

Both injectors are Windows-only. Compile them with `rustc` from the repository root. They map an already-created backing file; QEMU must be started with the matching host memory or IVSHMEM device first.

```powershell
rustc --edition=2021 -O tools/host_injector.rs -o dist/host_injector.exe
& (Join-Path $PWD 'dist/host_injector.exe') dist/shm_mailbox.bin dist/trained_pure_solo_expected.json
rustc --edition=2021 -O tools/mmio_injector.rs -o dist/mmio_injector.exe
& (Join-Path $PWD 'dist/mmio_injector.exe') dist/ivshmem_mailbox.bin 8
```

`host_injector.exe <shm_mailbox.bin> <expected.json>` reads `test_batch_i8` and `test_batch_solo_i32`, requires 1–16 vectors of exactly 64 signed values and one expected scalar each, and reports wall-clock host round-trip time. It constructs an SO-shaped frame but only the 64-byte body occupies the shared mailbox slot; it validates the one-i32 output as the SR scalar. `mmio_injector.exe <backing-file> [frame-count]` defaults to 8 frames and accepts 1–16; it injects zero vectors and expects zero decisions. Neither tool constructs the QEMU device.

### QEMU / PowerShell test scripts

These end-to-end scripts require a working QEMU/OVMF setup. `-QemuCpu` defaults to `Skylake-Server,+avx512f,+avx512dq` where available; ports should be distinct for simultaneous guests.

| Script | Command and behavior |
|---|---|
| `tools/test_uart_roundtrip.ps1` | `& .\tools\test_uart_roundtrip.ps1 -Port 5556 -QemuAccel whpx -ShardPath dist/trained_pure_solo_shard.bin -ExpectedPath dist/trained_pure_solo_expected.json` — validates the v2 Solo header/hash, sends the sidecar's vectors as 66-byte SO frames, and byte-compares each six-byte SR scalar. Params: `Port` 5556, `QemuAccel` empty, `QemuCpu` Skylake default, `ShardPath`, `ExpectedPath`, `EfiPath`. Requires exported shard/sidecar and EFI. |
| `tools/test_hot_swap.ps1` | `& .\tools\test_hot_swap.ps1 -CpuCount 2 -UpdateLba 264192` — builds a zero-weight native Solo update, packages the image, sends SO/SR requests, issues NU, and verifies post-swap zero decisions. Params: `Port` 5571, `CpuCount` 2 (range 2–64), `ImagePath`, `UpdateShardPath`, `UpdateLba`. The packaged shard is written at the configured location; keep the request LBA aligned with that location (default 264192). |
| `tools/test_dual_volume.ps1` | `& .\tools\test_dual_volume.ps1 -CpuCount 1 -ImagePath dist/neural_box_appliance.img` — boots GPT/FAT32, sends eight zero-input SO frames, validates SR scalars and loader telemetry. Params: `Port` 5570, `CpuCount` 1 (range 1–64), `-TamperWeights`, `QemuAccel`, `QemuCpu`, `ImagePath`. Tamper mode verifies signature rejection. Requires a prebuilt image and EFI. |
| `tools/test_attention_sequence.ps1` | `& .\tools\test_attention_sequence.ps1 -CpuCount 2 -Port 5569` — legacy filename, now builds a native Solo shard and runs the eight-frame SO/SR protocol test. Params: `Port` 5569, `CpuCount` 2 (range 1–64), `QemuAccel`, `QemuCpu`, `ShardPath`, `EfiPath`. |
| `tools/test_mmio_pipeline.ps1` | `& .\tools\test_mmio_pipeline.ps1 -FrameCount 8 -MailboxPath dist/ivshmem_mailbox.bin` — builds/packages, starts IVSHMEM, invokes the MMIO injector, and checks no-loss telemetry. Params: `QemuCpu`, `QemuAccel`, `FrameCount` (guest completion is fixed at eight), `ImagePath`, `MailboxPath`. QEMU must support `memory-backend-file` and `ivshmem-plain`; the script packages its default image path. |
| `tools/run_shm_bridge.ps1` | `& .\tools\run_shm_bridge.ps1 -ShardPath dist/trained_pure_solo_shard.bin -ExpectedPath dist/trained_pure_solo_expected.json -MailboxPath dist/shm_mailbox.bin -QemuAccel whpx` — builds with `host-ipc`, starts QEMU with host-backed guest RAM, then runs the host injector. Params: `Port` (declared, currently unused), `ShardPath`, `ExpectedPath`, `MailboxPath`, `QemuAccel`. Requires the exported shard and sidecar. |
| `ml/run_pure_pipeline.ps1` | `& .\ml\run_pure_pipeline.ps1 -Python C:/path/to/python.exe -Port 5558 -QemuAccel whpx -BlockSize 512` — runs training, export/sign, UEFI release build, and UART roundtrip. Params: `Python` (checked-in default is machine-specific), `Port`, `QemuAccel`, `BlockSize`. Python needs PyTorch. |
| `tools/package_image.ps1` | `& .\tools\package_image.ps1 -efiPath target/x86_64-unknown-uefi/release/neural_box_core.efi -shardPath dist/production_shard.bin -outPath dist/neural_box_appliance.img` — wrapper around the image packager. |
| `tools/payload_builder/build_payload.ps1` | `& .\tools\payload_builder\build_payload.ps1 -outPath dist/production_shard.bin -blockSize 512 -model solo -variant pattern` — wrapper around the native Rust signer; parameters described above. |
| `tools/wasm_verifier/build.ps1` | `& .\tools\wasm_verifier\build.ps1` — no parameters; builds the locked release Wasm verifier and copies the `.wasm` artifact into its web directory. |

### Archived scratch image diagnostics (`scratch/`)

These scripts are not the production GPT/FAT32 workflow. Several hard-code the old machine-specific `C:\Users\IMOE001\9\...` workspace, construct legacy unsigned NEUR v1 records, or depend on an external `qemu-img.exe`; do not use them to create shards for this firmware. They are listed for source-tree completeness:

| Script | Parameters and effect |
|---|---|
| `scratch/create_weights.ps1` | Empty file; no parameters or behavior. |
| `scratch/test_64k.ps1` | Empty file; no parameters or behavior. |
| `scratch/inspect_fat.ps1` | No parameters. Reads the hard-coded `C:\Users\IMOE001\9\scratch\qemu_fat16.img` and prints MBR/FAT geometry. |
| `scratch/test_builder.ps1` | Optional `-outPath` (hard-coded workspace default). Uses hard-coded `qemu-img.exe`, builds an experimental image and writes a legacy unsigned version-1 shard; not runtime-compatible. |
| `scratch/test_custom_fat16.ps1` | `-efiPath` and `-outPath`, both with hard-coded workspace defaults. Builds a test FAT16 image and embeds a legacy unsigned version-1 raw payload. |
| `scratch/test_fat32.ps1` | `-efiPath` and `-outPath`, both with hard-coded workspace defaults. Builds an experimental MBR/FAT32 image and embeds a legacy unsigned version-1 raw payload. |

MMIO additionally requires a QEMU build with file-backed memory and IVSHMEM support. Host injector timing is `std::time::Instant`; firmware timing is TSC-based and represents different intervals.

## 5. Appliance Build & Operational Recipes

### Host tests and microbenchmark

```powershell
cargo +nightly test --bin neural_box_core --target x86_64-pc-windows-msvc -- --nocapture
```

The suite covers Solo arithmetic/parity, SO/SR framing, v2 mailbox layout, signature/shard ingestion, and hardware-independent firmware helpers. The RDTSC test prints a host-specific minimum; it does not promise 13–14 ticks on another machine.

### Bare-metal UEFI

```powershell
cargo +nightly check --target x86_64-unknown-uefi
cargo +nightly build --target x86_64-unknown-uefi --release
```

The release output is `target/x86_64-unknown-uefi/release/neural_box_core.efi`. Build the release before packaging or starting a QEMU appliance scenario.

### Wasm verification engine and playground

```powershell
cargo +nightly build --locked --manifest-path tools/wasm_verifier/Cargo.toml --no-default-features --target wasm32-unknown-unknown --release
Copy-Item tools/wasm_verifier/target/wasm32-unknown-unknown/release/wasm_verifier.wasm tools/wasm_verifier/wasm_verifier.wasm -Force
node tools/wasm_verifier/serve.js
```

Open `http://127.0.0.1:8000/`, load the standalone Wasm kernel, run **Run scalar parity suite**, and verify all five test cases have delta zero. The live GitHub Pages proof is linked at the top of this README.

### Image boot and hot-swap operation

Generate a signed Solo shard, build the EFI, and package the image using the commands above. The firmware first searches UEFI `SimpleFileSystem` for `weights.bin`; if a file exists but fails validation it selects the safe identity model rather than falling through to raw NVMe. Raw fallback probes fixed byte offsets, not a general GPT scanner. For a hot-swap, `NU` supplies the candidate NVMe logical block address. An auxiliary AP reads/verifies the signed Solo record into the inactive model slot; only a fully validated candidate becomes active. This is an in-memory shadow swap, not disk A/B partitions, and requires a suitable secondary processor.

### Generated artifacts and repository hygiene

The repository ignores `target/`, `dist/`, nested Cargo targets, executables, PDB/Rust metadata, Python bytecode/caches, and local temp files. Cargo build products, signed test shards, trained checkpoints, disk images, QEMU logs, and injector binaries are local generated artifacts; do not add them to source control. Check the final source/docs tree with:

```powershell
git status --short
git check-ignore target dist
```

The built-in test signing seed is public and must never be used as a production secret. Replace the embedded verification key and protect the matching signing key outside this repository for production deployments.

## License

Licensed under either the MIT License (`LICENSE-MIT`) or Apache License, Version 2.0 (`LICENSE-APACHE`).