# tensor-wasm-fuzz

`cargo-fuzz` targets for Craton TensorWasm. Each binary listed below runs
through `libfuzzer-sys` against the corresponding subsystem.

| Target | Subsystem | Invariant |
|---|---|---|
| `fuzz_wasm_compile` | `wasmtime::Module::from_binary` | host process never crashes on arbitrary bytes |
| `fuzz_ptx_emit` | `tensor-wasm-jit` `ptx_emit::emit` | emitter never panics on arbitrary blueprints |
| `fuzz_snapshot_restore` | `tensor-wasm-snapshot` `SnapshotReader::restore` | restore returns `Err`, not panic, on malformed input |
| `fuzz_snapshot_restore_arbitrary` | `tensor-wasm-snapshot` `SnapshotReader::restore` (v4 artifact-envelope path) | prepends `ARTIFACT_MAGIC` + a synthetic HMAC key so `restore` dispatches onto the v0.4 envelope decode; tampered/arbitrary v4 envelopes are rejected as `Err`, never a panic |
| `fuzz_wasi_cuda_abi` | `tensor-wasm-wasi-gpu` host functions | host never crashes on arbitrary `(ptr, len)` from Wasm guest |
| `token_scope_parser` | `tensor-wasm-api` `token_scope::parse_tokens_env` | parser never panics; every accepted bearer is non-empty; scope variant matches `is_all()` |
| `audit_json_round_trip` | `tensor-wasm-api` `AuditRecord` Serialize | production JSON parses back into the documented wire-format shape (catches Serialize drift even though the type doesn't derive Deserialize) |
| `fuzz_parse_argv` | `tensor-wasm-wasi-gpu` `kernel_args::parse_argv` | host-trust-boundary argv parser never panics; errors only as documented `AbiError::{InvalidArgs, InvalidPointer, KernelArgsUnsupported}` |
| `fuzz_rewrite_wasm` | `tensor-wasm-jit` `rewrite::rewrite_wasm` | for any input that `wasmparser::validate` accepts, the rewritten module also validates (rewriter preserves Wasm validity) |
| `fuzz_pool_allocate` | `tensor-wasm-mem` `pool::UnifiedMemoryPool::allocate` | bump-pointer pool never panics on arbitrary `(size, align)` — every failure mode (zero size, bad align, exhaustion, overflow) surfaces as `Err(UnifiedError)` |
| `lowering_driver` | `tensor-wasm-jit` `lowering_driver::lower_function` | Cranelift → `LoweredFunction` driver never panics; every failure surfaces as `Err(LoweringError::{UnsupportedOpcode,UnsupportedType,UndefinedValue,MalformedTerminator,Rejected,BadBlockReference})` |
| `fuzz_artifact_decode_envelope` | `tensor-wasm-artifacts` `decode_envelope_from_bytes` / `_with_cap` | splits a synthetic 32-byte HMAC key off the input; the envelope decoder rejects every malformed shape (bad magic/version, HMAC mismatch, zstd garbage, zip-bomb, hash mismatch) as `Err(ArtifactError)`, never a panic |

The targets above exercise the host-trust-boundary parsers and
emitters across the runtime: Wasm compilation, PTX emission, snapshot
restore (legacy v2 and the v0.4 artifact-envelope path), the artifact-store
envelope decoder, the wasi-cuda ABI, the token-scope and audit-JSON
surfaces, kernel-args argv lowering, JIT rewrite, the unified-memory bump
allocator, and the Cranelift lowering driver. Each one's invariant is
"no panic on arbitrary input; documented errors only."

## Running locally

```sh
cargo install cargo-fuzz
cd fuzz

# 5-minute smoke (matches CI cadence)
cargo +nightly fuzz run fuzz_wasm_compile -- -max_total_time=300
cargo +nightly fuzz run fuzz_ptx_emit -- -max_total_time=300
cargo +nightly fuzz run fuzz_snapshot_restore -- -max_total_time=300
cargo +nightly fuzz run fuzz_snapshot_restore_arbitrary -- -max_total_time=300
cargo +nightly fuzz run fuzz_artifact_decode_envelope -- -max_total_time=300
cargo +nightly fuzz run fuzz_wasi_cuda_abi -- -max_total_time=300
cargo +nightly fuzz run token_scope_parser -- -max_total_time=300
cargo +nightly fuzz run audit_json_round_trip -- -max_total_time=300
cargo +nightly fuzz run fuzz_parse_argv -- -max_total_time=300
cargo +nightly fuzz run fuzz_rewrite_wasm -- -max_total_time=300
cargo +nightly fuzz run fuzz_pool_allocate -- -max_total_time=300
cargo +nightly fuzz run lowering_driver -- -max_total_time=60

# 24-hour soak (per the v0.5 PATH-TO-V1 security workstream:
# "keep `fuzz/` targets running 24x7 on dedicated hardware")
cargo +nightly fuzz run token_scope_parser -- -max_total_time=86400
cargo +nightly fuzz run audit_json_round_trip -- -max_total_time=86400
```

The CI workflow runs each target for 5 minutes per nightly. Seed corpora
live under `corpus/<target>/` (committed) — add interesting inputs there
to keep coverage stable across runs.

The plan calls for ≥ 100k iterations per target. On a developer laptop the
default `cargo fuzz run` reaches ≥ 200k iterations/minute for the lighter
targets (`token_scope_parser`, `audit_json_round_trip`, `fuzz_ptx_emit`);
budget ≥ 500 ms per CI step to clear that bar. The wasmtime and wasi-cuda
targets are heavier (~10–30k iter/minute) — give them the full 5 minutes.

## Triaging findings

libFuzzer writes a `crash-<sha1>` file under
`fuzz/artifacts/<target>/` whenever it produces a panic, OOM, or
sanitizer report. To reproduce:

```sh
cd fuzz
cargo +nightly fuzz run <target> artifacts/<target>/crash-<sha1>
```

Steps:

1. Reproduce locally with the artifact path above (deterministic — no
   `-runs` or `-max_total_time` argument).
2. Minimise: `cargo +nightly fuzz tmin <target> artifacts/<target>/crash-<sha1>`
   reduces the input to the smallest panicking form.
3. File a `S-security` issue with the minimised artifact attached and
   the panic message + backtrace from step 1. If the finding is in
   `token_scope_parser` or `audit_json_round_trip`, also link the bug
   to the v0.5 PATH-TO-V1 security workstream ("Fuzz corpus growth").
4. Land the fix **without modifying the harness** — the harness exists
   to catch real bugs, so silencing it would mask the regression. If the
   property assertion itself is wrong, that is a separate fix and
   should be reviewed by the security workstream owner.
5. Commit the minimised artifact into `corpus/<target>/` so the
   regression is permanently fenced in future runs.

See `docs/SECURITY-AUDIT.md` for the audit checklist this work supports.
