# Sovereign-Solo Wasm scalar verifier

This `no_std` WebAssembly module exposes one stateless inference call: `neur_solo(inputs, weights, output)`. It accepts 64 signed input bytes and 64 signed ternary coefficient bytes (`-1`, `0`, `+1`), then writes one branchless scalar decision (`-1`, `0`, `+1`). No recurrent state, attention projection, stream selection, floating point, or JavaScript inference arithmetic is used.

The pure integer reference lives in `src/pure.rs`. The appliance host tests compare it against `kernel::infer_scalar` across mixed signed-byte inputs.

## Build and serve

From the repository root:

```powershell
cargo +nightly build --locked --manifest-path tools/wasm_verifier/Cargo.toml --no-default-features --target wasm32-unknown-unknown --release
Copy-Item tools/wasm_verifier/target/wasm32-unknown-unknown/release/wasm_verifier.wasm tools/wasm_verifier/wasm_verifier.wasm -Force
node tools/wasm_verifier/serve.js
```

Open `http://127.0.0.1:8000/`, randomize the 64-byte vector or run the five-case parity suite, and verify that each reference/Wasm delta is zero. Browser timing includes JS/Wasm boundary overhead and is not the bare-metal TSC benchmark.
