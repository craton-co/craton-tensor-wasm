# bali-fuzz

`cargo-fuzz` targets for Project Bali. Three targets, each running through
`libfuzzer-sys` against the corresponding subsystem.

| Target | Subsystem | Invariant |
|---|---|---|
| `fuzz_wasm_compile` | wasmtime::Module::from_binary | host process never crashes on arbitrary bytes |
| `fuzz_ptx_emit` | bali-jit ptx_emit::emit | emitter never panics on arbitrary blueprints |
| `fuzz_snapshot_restore` | bali-snapshot SnapshotReader::restore | restore returns Err, not panic, on malformed input |

## Running locally

```sh
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run fuzz_wasm_compile -- -max_total_time=300
cargo +nightly fuzz run fuzz_ptx_emit -- -max_total_time=300
cargo +nightly fuzz run fuzz_snapshot_restore -- -max_total_time=300
```

The CI workflow runs each target for 5 minutes per nightly. Seed corpora live
under `corpus/<target>/` (committed) — add interesting inputs there to keep
coverage stable across runs.

The plan calls for ≥ 100k iterations per target. On a developer laptop the
default `cargo fuzz run` reaches ≥ 200k iterations/minute for the lighter
targets; budget ≥ 500 ms per CI step to clear that bar.

See `docs/SECURITY-AUDIT.md` for the audit checklist this work supports.
