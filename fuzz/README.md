# tensor-wasm-fuzz

`cargo-fuzz` targets for Craton TensorWasm. Each binary listed below runs
through `libfuzzer-sys` against the corresponding subsystem.

| Target | Subsystem | Invariant |
|---|---|---|
| `fuzz_wasm_compile` | `wasmtime::Module::from_binary` | host process never crashes on arbitrary bytes |
| `fuzz_ptx_emit` | `tensor-wasm-jit` `ptx_emit::emit` | emitter never panics on arbitrary blueprints |
| `fuzz_snapshot_restore` | `tensor-wasm-snapshot` `SnapshotReader::restore` | restore returns `Err`, not panic, on malformed input |
| `fuzz_wasi_cuda_abi` | `tensor-wasm-wasi-gpu` host functions | host never crashes on arbitrary `(ptr, len)` from Wasm guest |
| `token_scope_parser` | `tensor-wasm-api` `token_scope::parse_tokens_env` | parser never panics; every accepted bearer is non-empty; scope variant matches `is_all()` |
| `audit_json_round_trip` | `tensor-wasm-api` `AuditRecord` Serialize | production JSON parses back into the documented wire-format shape (catches Serialize drift even though the type doesn't derive Deserialize) |

## Running locally

```sh
cargo install cargo-fuzz
cd fuzz

# 5-minute smoke (matches CI cadence)
cargo +nightly fuzz run fuzz_wasm_compile -- -max_total_time=300
cargo +nightly fuzz run fuzz_ptx_emit -- -max_total_time=300
cargo +nightly fuzz run fuzz_snapshot_restore -- -max_total_time=300
cargo +nightly fuzz run fuzz_wasi_cuda_abi -- -max_total_time=300
cargo +nightly fuzz run token_scope_parser -- -max_total_time=300
cargo +nightly fuzz run audit_json_round_trip -- -max_total_time=300

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
