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
| `benches/tail_latency.rs` | Hand-rolled 10 000-sample loop capturing P50/P95/P99/**P99.9**/max for `dispatch/serial/100`, `dispatch/concurrent_cap64/100`, `e2e/healthz/get`, `e2e/invoke_not_found/post`. Sidesteps Criterion's statistical pipeline because Criterion does not surface P99.9 and its default sample count is too small to resolve it. | Same as the underlying benches — dispatch metrics become real launch-to-completion latency on a CUDA host; e2e metrics are HTTP/serde-floor regardless. |
| `benches/dispatch_future_backends.rs` | F3 scaffold — runs the existing busy-poll `DispatchFuture` path and a stubbed `cuda-async`-backed alternative through the same no-op dispatch loop and reports W4.6-shape percentiles per backend. Today writes one real `busy-poll` line plus one documented `cuda-async` skip line to `bench-results/dispatch-future-backends.json`. Answers [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md) Unresolved question #3 once the v0.4 cuda-oxide port lands. | Same as `kernel_dispatch.rs` for the busy-poll path; the cuda-async path will measure `cuda_async::Stream::synchronize`-equivalent latency once wired. |

The `dispatch_future_backends` bench is the F3 follow-up to W4.6: it
extends the same nearest-rank percentile harness with a backend axis so
that, when the v0.4 cuda-oxide port replaces the busy-poll
`Event::query` loop with `cuda-async`, the speedup (or regression) is
visible in one diff against `bench-results/dispatch-future-backends.json`.
The bench is feature-gated behind `tensor-wasm-bench`'s `cuda` feature
(forwards to `tensor-wasm-wasi-gpu/cuda`); the v0.4 wiring additionally
requires `--features cuda-oxide-backend` (forwards to
`tensor-wasm-mem/cuda-oxide-backend`). See
[`bench-results/README.md`](../../bench-results/README.md#dispatch-future-backend-comparison)
for the schema, the skip-line convention, and the exact regeneration
recipe under both today's stub configuration and the eventual v0.4
configuration.

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
cargo bench -p tensor-wasm-bench --bench tail_latency

# F3 backend-comparison bench (busy-poll vs cuda-async). Today emits real
# numbers for busy-poll and a skip line for cuda-async; the v0.4 cuda-oxide
# port lights up the second line. Requires `--features cuda` to compile
# the real body (without it both backends emit skip lines):
cargo bench -p tensor-wasm-bench --bench dispatch_future_backends --features cuda

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

Two Cargo features, both off by default so the regular
`cargo bench -p tensor-wasm-bench` pipeline stays cuda-toolkit-free:

- `cuda` — forwards to `tensor-wasm-wasi-gpu/cuda`. Required to compile
  the real body of `benches/dispatch_future_backends.rs`; without it the
  bench emits two skip lines and exits cleanly. None of the other bench
  files in this crate are affected by this feature.
- `cuda-oxide-backend` — forwards to `tensor-wasm-mem/cuda-oxide-backend`.
  Required (in addition to `cuda`) once the v0.4 cuda-oxide port wires
  the cuda-async path in `benches/dispatch_future_backends.rs`. v0.3.x
  builds do not need it because the scaffold deliberately carries no
  `cuda-async` use statement. See
  [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md) for the v0.5
  cust-successor rollout that this feature flag tracks.

See [`docs/BUILD.md`](../../docs/BUILD.md) for the project-wide flag
taxonomy and how CUDA-only paths are gated upstream.
