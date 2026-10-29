# tensor-wasm-bench

Benchmark harness crate for Craton TensorWasm. Houses the Criterion micro-benchmarks
plus the end-to-end suites that measure cold-start, dispatch latency, JIT
throughput, snapshot round-trips, and the axum router floor latency.

The crate's library half carries the shared percentile/JSON helpers used by
the hand-rolled benches plus the CI regression-gate comparator core
(`classify_metric` / `check_regression`); most of the action lives in
[`benches/`](./benches/), and the gate binary in
[`src/bin/bench_compare.rs`](./src/bin/bench_compare.rs).

## Bench files

| File | What it measures | CUDA-host behavior |
|---|---|---|
| `benches/kernel_dispatch.rs` | Back-pressure permit acquire/release + dispatch-future poll: `serial`, uncontended `concurrent_cap64`, `span_only` (isolates the per-dispatch `info_span!` cost), `contended` (cap≪n so the semaphore queues), and CUDA-only `real_sync` (records a `cust` event and drives `DispatchFuture::with_event`). | `real_sync` measures the real `event.query()` completion-poll floor; the `ready`-path groups stay the scheduling floor. |
| `benches/cold_start.rs` | Snapshot capture (bincode + zstd encode), warm steady-state in-memory restore (`restore_steady_state`), and full capture → `fs::write` → `fs::read` → restore round-trip at 1/16/128/512 MiB. Fixtures use a high-entropy `i % 251` counter pattern so zstd cannot crush them to nothing. | Adds UVM page-migration cost on first touch (20-200 ms depending on PCIe link). |
| `benches/memory_bandwidth.rs` | Host-side `copy_from_slice` over a `GuardedHostBuffer` in both sequential and fixed-stride access patterns. | Becomes `cudaMemcpyAsync` device-to-device throughput, bounded by HBM bandwidth. |
| `benches/jit_compile.rs` | PTX text-emit latency for representative kernels (vector_add, matmul, conv2d), blueprint fingerprint cost, and `KernelCache` hit-vs-miss latency. | Adds `ptxas` (10-50 ms / kernel) and, eventually, `nvrtc` once that path lands. |
| `benches/e2e_inference.rs` | Full axum router round-trip through `tensor-wasm-api` driven via `tower::ServiceExt::oneshot`. Healthz, POST /functions, invoke-not-found, and a `secured` variant that exercises the bearer-auth + tenant-scope + rate-limit middleware (auth/rate-limit ENABLED). | Unchanged — this measures the HTTP/serde floor, independent of GPU work. |
| `benches/call_export_args.rs` | `call_export_with_args` overhead: a `hot` group (instance spawned once, called in the loop) isolates the args-marshalling cost, and an `envelope` group (spawn + invoke + terminate per sample) matches the `/invoke` per-request shape. | Unchanged — executor/wasmtime floor. |
| `benches/streaming_invoke.rs` | `/invoke-stream` framing floor: `baseline_invoke` (sync `/invoke`), `sse` (`Accept: text/event-stream`), and `chunked` (octet-stream fallback), all against a pre-deployed `_start` no-op function. | Unchanged — HTTP/streaming floor. |
| `benches/tail_latency.rs` | Hand-rolled 10 000-sample loop capturing P50/P95/P99/**P99.9**/max for `dispatch/serial/100`, `dispatch/span_only/1`, `dispatch/concurrent_cap64/100`, `dispatch/contended_cap1/1000`, `e2e/healthz/get`, `e2e/invoke_not_found/post`, plus CUDA-only `dispatch/real_sync/1`. Sidesteps Criterion's statistical pipeline because Criterion does not surface P99.9 and its default sample count is too small to resolve it. | `real_sync` becomes real launch-to-completion latency on a CUDA host; e2e metrics are HTTP/serde-floor regardless. |
| `benches/dispatch_future_backends.rs` | F3 — runs two **real** GPU-sync strategies through the same no-op dispatch loop and reports W4.6-shape percentiles per backend: `busy-poll` (async `event.query()` poll + 50 µs yield) vs `cuda-async` (blocking `stream.synchronize()` on the blocking pool). Answers [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md) Unresolved question #3. Requires `--features cuda`; a skip line now means CUDA was unavailable at runtime, not unimplemented. | Both lines are the sync-machinery floor over an empty stream — the A/B is "busy-poll vs stream-synchronize wait strategy". |

The `dispatch_future_backends` bench is the F3 follow-up to W4.6: it
extends the same nearest-rank percentile harness with a backend axis so
the busy-poll (`event.query`) vs stream-synchronize speedup (or
regression) is visible in one diff against
`bench-results/dispatch-future-backends.json`. Both backends drive real
`cust` work, so the bench is feature-gated behind `tensor-wasm-bench`'s
`cuda` feature (forwards to `tensor-wasm-wasi-gpu/cuda`). The genuine
cuda-oxide async runtime is still a v0.5 port; until it lands the
`cuda-async` backend's mechanism is `cust`'s `stream.synchronize`, the
closest faithful stand-in, and only the `dispatch_once` body changes when
the real runtime ports in. See
[`bench-results/README.md`](../../bench-results/README.md#dispatch-future-backend-comparison)
for the schema and the regeneration recipe.

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

# F3 backend-comparison bench (busy-poll vs cuda-async). Both backends drive
# real `cust` work; requires `--features cuda` to compile the real body
# (without it both backends emit skip lines):
cargo bench -p tensor-wasm-bench --bench dispatch_future_backends --features cuda

# Compile-only sanity check:
cargo bench -p tensor-wasm-bench --no-run

# Save a labelled baseline (Criterion writes to `target/criterion/<baseline>/`):
cargo bench -p tensor-wasm-bench --bench cold_start -- --save-baseline mybaseline

# CI regression gate: compare a current tail-latency result against the
# committed baseline and fail (exit 1) on a >10% P99 regression of any GATED
# metric. Diagnostic metrics are printed but never fail. Requires the
# `regression-gate` feature (pulls in serde_json as a normal dep):
cargo run -p tensor-wasm-bench --features regression-gate --bin bench_compare -- \
    bench-results/baseline.json bench-results/tail-latency.json --threshold 0.10

# Flamegraph profiling (pprof). Off by default; with the feature on, each
# e2e_inference bench writes a flamegraph under target/criterion/<id>/profile/:
cargo bench -p tensor-wasm-bench --features profiling --bench e2e_inference
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

The gate itself is the `bench_compare` binary
([`src/bin/bench_compare.rs`](./src/bin/bench_compare.rs)), which diffs a
current result document against the baseline and fails on a gated-metric
regression. The gated-vs-diagnostic split is encoded in
`tensor_wasm_bench::classify_metric` (unit-tested in `src/lib.rs`):

- **Gated** (fail CI on regression): the headline CPU-bound, low-variance
  metrics — `dispatch/serial*`, `dispatch/concurrent*`, `e2e/healthz*`,
  `e2e/invoke_not_found*`, `jit_compile/*`, `cold_start/capture*`.
- **Diagnostic** (recorded + diffed by humans, never fails CI): GPU-,
  filesystem-, or A/B-dependent metrics — `dispatch/real_sync*`,
  `dispatch/contended*`, `dispatch/span_only*`, `*restore_steady_state*`,
  `disk_round_trip*`, `e2e/secured*`, `memory_bandwidth*`,
  `metrics_label_validation*`, and the `dispatch_future_backends` backend
  comparison. Any unrecognised metric defaults to diagnostic, so adding a
  new bench never breaks CI before its baseline is captured.

## Quiet-host protocol

The percentile and tail numbers (especially `tail_latency.rs`'s P99.9) are
only comparable run-to-run on a quiet, pinned host. Before capturing a
baseline or a gate input, follow this protocol:

1. **CPU governor**: set `performance` (Linux:
   `sudo cpupower frequency-set -g performance`) so turbo/throttle
   transitions don't inject tail latency. On Windows set the "High
   performance" power plan.
2. **Affinity**: pin the bench process to a fixed set of physical cores away
   from housekeeping (Linux: `taskset -c 2-5 cargo bench ...`; Windows:
   `start /affinity` or `Start-Process -ProcessorAffinity`). Use physical,
   not SMT-sibling, cores for the dispatch micro-benches.
3. **Isolation**: close background apps, disable the screensaver/indexer,
   and on Linux prefer an isolated cgroup or `isolcpus=` boot arg for the
   pinned cores on dedicated bench runners.
4. **Seed / determinism**: the benches use deterministic fixtures (the
   `i % 251` counter pattern, fixed UUIDs, a single accepted bench token) so
   there is no RNG to seed — but record the toolchain, `--features` set, and
   `BACKEND_LABEL` alongside the result so two JSON files are only ever
   compared within the same configuration. `bench_compare` compares by metric
   key and assumes the two files were produced under the same backend.
5. **Warm-up**: the hand-rolled benches already run `WARMUP = 1_000`
   un-timed iterations; do not reduce this when chasing a tighter wall-clock.

The same protocol is assumed by the `--features profiling` flamegraphs —
a noisy host produces flamegraphs dominated by scheduler artefacts.

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

All off by default so the regular `cargo bench -p tensor-wasm-bench`
pipeline stays cuda-toolkit-free and serde_json-dev-only:

- `cuda` — forwards to `tensor-wasm-wasi-gpu/cuda` and pulls in `cust` (via
  `dep:cust`). Required to compile the real GPU-event paths:
  `kernel_dispatch::bench_real_sync`, `tail_latency`'s `dispatch/real_sync/1`,
  the real-event `busy-poll` path, and the `cuda-async` `stream.synchronize`
  body in `benches/dispatch_future_backends.rs`. Without it those paths are
  `#[cfg]`-ed out (the backends bench emits skip lines).
- `cuda-oxide-backend` — forwards to `tensor-wasm-mem/cuda-oxide-backend`.
  Tracks the v0.5 cust-successor rollout; see
  [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md). The
  `dispatch_future_backends` bench already runs real `cust` work under
  `cuda` alone, so this flag is forward-looking for when the genuine
  cuda-oxide async runtime ports in.
- `cudarc-backend` — label-only flag for `tail_latency.rs`'s `BACKEND_LABEL`
  axis (does not forward to the mem crate; see the `Cargo.toml` comment).
- `regression-gate` — builds the `bench_compare` CI-gate binary, pulling in
  `serde_json` as a normal dep so the comparator can parse result documents.
- `profiling` — enables the pprof flamegraph profiler hook for the
  `e2e_inference` benches (via `dep:pprof`).

See [`docs/BUILD.md`](../../docs/BUILD.md) for the project-wide flag
taxonomy and how CUDA-only paths are gated upstream.
