# bali-bench

Benchmark harness crate for Project Bali, intended to host Criterion micro-benchmarks plus custom end-to-end suites that measure cold-start, dispatch latency, JIT throughput, and tenant isolation overheads. The S1 scaffold is intentionally empty; the actual Criterion benches land in S9 and S19.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` (lib) — async runtime used to drive end-to-end benchmark scenarios.
- `criterion` (dev only) — statistical micro-benchmark harness.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
