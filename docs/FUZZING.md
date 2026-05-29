# Fuzzing

> `fuzz/` directory holds `cargo-fuzz` targets. Owner: @craton-co/security.

## Targets

The `[[bin]]` entries in `fuzz/Cargo.toml` (sources under `fuzz/fuzz_targets/`) define 15 targets. Each guards a host-trust-boundary parser, emitter, or codec; unless noted, the invariant is "no panic on arbitrary input; documented errors only."

| Target | Subsystem / entry point | Input fuzzed | Invariant / crash class guarded |
|---|---|---|---|
| `fuzz_wasm_compile` | `wasmtime::Module::from_binary` | Arbitrary bytes (≤ 64 KiB) as a Wasm module | Host process never crashes on arbitrary module bytes. |
| `fuzz_rewrite_wasm` | `tensor-wasm-jit` `rewrite::rewrite_wasm` | Bytes that pass `wasmparser::validate` | Rewriter preserves Wasm validity — a rewritten module always re-validates (else instantiation traps). |
| `fuzz_ptx_emit` | `tensor-wasm-jit` `ptx_emit::emit` | `arbitrary`-derived `TensorWasmKernelBlueprint` op stream + grid hints | PTX emitter never panics on arbitrary blueprints. |
| `lowering_driver` | `tensor-wasm-jit` `lowering_driver::lower_function` (needs `cuda-oxide-backend`) | `arbitrary`-generated well-formed Cranelift `Function` | Cranelift → `LoweredFunction` driver never panics; failures surface as `Err(LoweringError)`. |
| `fuzz_snapshot_restore` | `tensor-wasm-snapshot` `SnapshotReader::restore` | Arbitrary bytes (≤ 64 MiB), no HMAC key | Malformed input returns `Err`, never a panic. |
| `snapshot_restore` | same, v2 (unsigned) path | Arbitrary bytes, no key configured | v2 classification/parse never panics; surfaces `TensorWasmError::Serialization`. |
| `snapshot_restore_signed` | same, v3 (HMAC-SHA256) path | First 32 bytes → synthetic key, remainder → payload | "Authenticate then parse" rejects bad signatures (HMAC mismatch) without decoding the payload; never panics. |
| `fuzz_snapshot_restore_arbitrary` | same, v4 artifact-envelope path | 32-byte synthetic key + `ARTIFACT_MAGIC`-prefixed blob | Tampered/arbitrary v4 envelopes are rejected as `Err`, never a panic. |
| `fuzz_artifact_decode_envelope` | `tensor-wasm-artifacts` `decode_envelope_from_bytes` / `_with_cap` | 32-byte synthetic key + arbitrary envelope (magic, version, BLAKE3 hash, zstd body, HMAC) | Every malformed shape (bad magic/version, HMAC mismatch, zstd garbage, zip-bomb `TooLarge`, hash mismatch) returns `Err(ArtifactError)`, never a panic. |
| `fuzz_wasi_cuda_abi` | `tensor-wasm-wasi-gpu` host functions (`wasi:cuda/host@0.2.0`) | `arbitrary` op stream driving `(ptr, len)` into `load_ptx` / `launch` / `last_error_copy` via a wasmtime guest | Host never crashes (UAF, overflow in `read_bytes`) on arbitrary guest pointers; `MalformedPtx` / `InvalidPointer` are expected. |
| `fuzz_parse_argv` | `tensor-wasm-wasi-gpu` `kernel_args::parse_argv` | Arbitrary argv bytes against a fixed 4 KiB zeroed guest memory | Argv parser never panics; only `AbiError::{InvalidArgs, InvalidPointer, KernelArgsUnsupported}` are acceptable errors. |
| `parse_argv` | same, split-input variant | Input split in half: argv buffer + attacker-shaped `mem` slice | Same contract as `fuzz_parse_argv`, with the guest memory also fuzzer-controlled. |
| `fuzz_pool_allocate` | `tensor-wasm-mem` `pool::UnifiedMemoryPool::allocate` | `arbitrary` `(u32, u32)` size/align against a 4 MiB slab | Allocator never panics; zero size, bad alignment, exhaustion, and overflow all surface as `Err(UnifiedError)`. |
| `token_scope_parser` | `tensor-wasm-api` `token_scope::parse_tokens_env` | `arbitrary`-derived `String` (`$TENSOR_WASM_API_TOKENS` grammar) | Parser never panics on operator input; every accepted bearer is non-empty and each scope's variant agrees with `is_all()`. |
| `audit_json_round_trip` | `tensor-wasm-api` `AuditRecord` Serialize | `arbitrary` `AuditRecordFixture` mirroring the public JSON shape | Production JSON parses back into the documented wire-format shape (catches Serialize drift even though `AuditRecord` has no `Deserialize`). |

> The `snapshot_restore` / `snapshot_restore_signed` / `fuzz_snapshot_restore_arbitrary` targets cover the v2, v3, and v4 reader paths respectively; `fuzz_parse_argv` and `parse_argv` are fixed-memory and split-input variants of the same `parse_argv` contract. See `fuzz/README.md` for seed-corpus and triage details.

## Running locally
```
cd fuzz
cargo +nightly fuzz run <target> -- -max_total_time=60
```

## Corpus
Stored in `fuzz/corpus/<target>/`. Public corpora are gitignored beyond seed inputs at `fuzz/corpus_seed/`.

## Cron
- `fuzz.yml`: nightly, 10 minutes per target.
- `fuzz-long.yml`: weekly, 4 hours per target.

## Crash triage
On crash, the workflow uploads the reproducer to `fuzz/artifacts/<target>/`. Open a SEC issue and follow `docs/SECURITY.md` for embargo.
