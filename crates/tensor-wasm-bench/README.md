# tensor-wasm-bench

Benchmark harness crate for Craton TensorWasm. Houses the Criterion micro-benchmarks
plus the end-to-end suites that measure cold-start, dispatch latency, JIT
throughput, snapshot round-trips, and the axum router floor latency.

The crate's library half is intentionally empty (`src/lib.rs` only carries
the `#![deny(missing_docs)]` module doc that points readers at the bench
files); all the action lives in [`benches/`](./benches/).

## Bench files

| File | What it measures | CUDA-host behavior |
|---|---|---|
| `benches/kernel_dispatch.rs` | Back-pressure permit acquire/release + dispatch-future poll, serial and concurrent (cap=64). | Becomes real launch-to-completion latency once `DispatchFuture::ready` is replaced with a CUDA-event-backed future. |
| `benches/cold_start.rs` | Snapshot capture (bincode + zstd encode), in-memory restore, and full capture → `fs::write` → `fs::read` → restore round-trip at 1/16/128/512 MiB. | Adds UVM page-migration cost on first touch (20-200 ms depending on PCIe link). |
| `benches/memory_bandwidth.rs` | Host-side `copy_from_slice` over a `GuardedHostBuffer` in both sequential and fixed-stride access patterns. | Becomes `cudaMemcpyAsync` device-to-device throughput, bounded by HBM bandwidth. |
| `benches/jit_compile.rs` | PTX text-emit latency for representative kernels (vector_add, matmul, conv2d), blueprint fingerprint cost, and `KernelCache` hit-vs-miss latency. | Adds `ptxas` (10-50 ms / kernel) and, eventually, `nvrtc` once that path lands. |
| `benches/e2e_inference.rs` | Full axum router round-trip through `tensor-wasm-api` driven via `tower::ServiceExt::oneshot`. Healthz, POST /functions, and invoke-not-found. | Unchanged — this measures the HTTP/serde floor, independent of GPU work. |

On a non-CUDA host (developer laptop, local CI) the dispatch/memory/JIT
benches degenerate to host-side measurements: dispatch futures resolve
immediately, `GuardedHostBuffer` is plain `mmap`, and PTX emit stops at
text generation. The numbers are still useful as regression backstops —
PERFORMANCE.md documents what they mean and what they map to on a CUDA
host.

## Running

```sh
# All benches in this crate (slow — Criterion defaults):
cargo bench -p tensor-wasm-bench

# One bench file at a time:
cargo bench -p tensor-wasm-bench --bench cold_start
cargo bench -p tensor-wasm-bench --bench kernel_dispatch
cargo bench -p tensor-wasm-bench --bench memory_bandwidth
cargo bench -p tensor-wasm-bench --bench jit_compile
cargo bench -p tensor-wasm-bench --bench e2e_inference

# Compile-only sanity check:
cargo bench -p tensor-wasm-bench --no-run

# Save a labelled baseline (Criterion writes to `target/criterion/<baseline>/`):
cargo bench -p tensor-wasm-bench --bench cold_start -- --save-baseline mybaseline
```

After a run, open `target/criterion/report/index.html` for the full
report (P50/P95/P99, histograms, regression plots against the previous
run).

## Results, baseline, and the regression gate

See [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) for the published
bench inventory, reference numbers (host-only and CUDA-projected), and
the CI regression-gate policy. The committed baseline lives at
[`bench-results/baseline.json`](../../bench-results/baseline.json) and is
consumed by [`.github/workflows/bench.yml`](../../.github/workflows/bench.yml).

## Dependencies

Workspace-internal (path-only, never published — see `publish = false`
in `Cargo.toml`):

- `tensor-wasm-api` — axum router for the e2e bench.
- `tensor-wasm-core` — `TenantId` / `InstanceId` newtypes used by snapshot fixtures.
- `tensor-wasm-jit` — IR + PTX emitter + `KernelCache`.
- `tensor-wasm-mem` — `GuardedHostBuffer` (formerly `PinnedHostBuffer`, renamed in
  the Batch D wave; the deprecated alias is still exported).
- `tensor-wasm-snapshot` — `SnapshotReader` / `SnapshotWriter`.
- `tensor-wasm-wasi-gpu` — `BackPressure` / `DispatchFuture`.

External (pinned at the workspace root):

- `tokio` — async runtime for the dispatch and e2e benches.
- `criterion` (dev) — the benchmark harness itself; every `[[bench]]`
  declares `harness = false`.
- `axum`, `tower`, `http-body-util` (dev) — driving the in-process HTTP
  router in `e2e_inference.rs`.
- `base64`, `serde_json` (dev) — building the JSON payload + base64-encoded
  Wasm header used by the create-function bench.
- `tempfile` (dev) — scratch directory for the `disk_round_trip` group.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every
workspace configuration. See [`docs/BUILD.md`](../../docs/BUILD.md) for
the project-wide flag taxonomy and how CUDA-only paths are gated upstream.
