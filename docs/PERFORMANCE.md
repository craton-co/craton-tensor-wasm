# Craton TensorWasm — Performance

This document describes how TensorWasm measures performance, what the current
reference numbers look like, and how the CI regression gate works. Two
reference points matter: (a) the **host-only path** that a developer laptop
and the local CI runner exercise (no CUDA libraries, the back-pressure and
snapshot machinery exercised against host memory only), and (b) the
**CUDA-host path** that a CUDA-equipped self-hosted runner will measure once
the deployment work in S22 lands. Until then, GPU-side numbers in this doc
are modeled estimates, clearly marked.

## How we measure

Every bench lives in [`crates/tensor-wasm-bench/benches/`](../crates/tensor-wasm-bench/benches/)
and is a [Criterion](https://bheisler.github.io/criterion.rs/book/) bench
declared with `harness = false` in `Cargo.toml`. The defaults we rely on:

- A warm-up phase before any sample is recorded (Criterion default, 3 s
  locally, 1 s in CI).
- At least **30 samples** per metric, so the reported P50 is meaningfully
  stable.
- Criterion's built-in outlier detection and coefficient-of-variation
  reporting. We aim for **CV < 5%**; benches that exceed that on the
  reference machines are flagged as noisy and either tightened (more
  samples, longer measurement window) or excluded from the regression gate.
- All numbers below are **P50 from local runs** unless noted otherwise.
  P95/P99 are visible in the Criterion HTML reports under
  `target/criterion/` after a run.

## Bench inventory

Every entry below is a Criterion `<group>/<id>` pair, written exactly as
Criterion emits it on stdout and as it appears under
`target/criterion/<group>/<id>/`. The "Source" column names the crate
hosting the bench — most live in `tensor-wasm-bench` but `tenant_registry/*`
lives in `tensor-wasm-tenant` because the `TenantRegistry` it exercises is
private to that crate.

| Bench id (Criterion `<group>/<id>`) | Source crate | Source file | What it measures | Throughput unit |
|---|---|---|---|---|
| `dispatch/serial/<N>` for N in {1, 10, 100, 1000} | `tensor-wasm-bench` | `benches/kernel_dispatch.rs` | Per-dispatch overhead (back-pressure permit acquire+release + future poll), serial. Setup hoisted via `iter_batched_ref`. | dispatches/sec |
| `dispatch/concurrent_cap64/<N>` for N in {1, 10, 100, 1000} | `tensor-wasm-bench` | `benches/kernel_dispatch.rs` | Same as above but cap=64 with 4 worker threads. | dispatches/sec |
| `cold_start/capture/<bytes>` for bytes in {1048576, 16777216, 134217728, 536870912} | `tensor-wasm-bench` | `benches/cold_start.rs` | Snapshot capture (bincode + zstd encode). | bytes/sec |
| `cold_start/restore/<bytes>` for bytes in {1048576, 16777216, 134217728, 536870912} | `tensor-wasm-bench` | `benches/cold_start.rs` | In-memory snapshot restore (zstd decode + bincode decode). Steady-state — see source-file caveat re: page-cache warmth. | bytes/sec |
| `cold_start/disk_round_trip/<bytes>` for bytes in {1048576, 16777216} | `tensor-wasm-bench` | `benches/cold_start.rs` | True cold-disk reference: capture + `fs::write` + `fs::read` + restore each iteration. Stops at 16 MiB; above that, IO dominates. | bytes/sec |
| `memory_bandwidth/sequential/<bytes>` for bytes in {4096, 65536, 1048576, 16777216} | `tensor-wasm-bench` | `benches/memory_bandwidth.rs` | Host-side `copy_from_slice` over `GuardedHostBuffer`. | bytes/sec |
| `memory_bandwidth/strided/<bytes>` for bytes in {65536, 1048576, 16777216} | `tensor-wasm-bench` | `benches/memory_bandwidth.rs` | Fixed-stride 64-byte copies (stride=4096). Renamed from `random_stride` — see `bench-results/baseline-notes.md`. | bytes/sec |
| `jit_compile/emit_text/<kernel>` for kernel in {`vector_add[4]`, `vector_add[16]`, `matmul[16x16x16]`, `conv2d[3x3]`} | `tensor-wasm-bench` | `benches/jit_compile.rs` | PTX text-emit latency. | iters/sec |
| `jit_compile/fingerprint/matmul_16x16x16` | `tensor-wasm-bench` | `benches/jit_compile.rs` | Blueprint hash latency. | iters/sec |
| `jit_compile/cache/cold_miss_then_insert` | `tensor-wasm-bench` | `benches/jit_compile.rs` | emit + `KernelCache::put` + `get`. Cache hoisted via `iter_batched_ref`. | iters/sec |
| `jit_compile/cache/warm_hit` | `tensor-wasm-bench` | `benches/jit_compile.rs` | Pre-populated `KernelCache::get` only. S13 done-when: <1ms. | iters/sec |
| `e2e/healthz/get` | `tensor-wasm-bench` | `benches/e2e_inference.rs` | Full axum router round-trip on GET `/healthz`. | requests/sec |
| `e2e/create_function/post` | `tensor-wasm-bench` | `benches/e2e_inference.rs` | POST `/functions` latency; fresh router per iter via `iter_batched`. | requests/sec |
| `e2e/invoke_not_found/post` | `tensor-wasm-bench` | `benches/e2e_inference.rs` | POST `/functions/<unknown>/invoke` error path. | requests/sec |
| `tenant_registry/lookup/<N>` for N in {1, 16, 256} | `tensor-wasm-tenant` | `benches/context_switch.rs` | `TenantRegistry::get` host-side lookup; CUDA equivalent is `cuCtxPushCurrent`/`cuCtxPopCurrent`. S16 done-when: <5µs. | iters/sec |
| `tenant_registry/consume_release/256KiB` | `tensor-wasm-tenant` | `benches/context_switch.rs` | `consume_bytes` + `release_bytes` quota round-trip. | iters/sec |

`kernel_dispatch` was added in S9 and `tenant_registry` in S16; the
remaining four bench files in `tensor-wasm-bench` (`cold_start`,
`memory_bandwidth`, `jit_compile`, `e2e_inference`) were introduced in
S19 alongside this document.

### Interpreting Criterion HTML

After any `cargo bench` invocation, Criterion writes a static-HTML
report tree under `target/criterion/`. The useful entry points:

- `target/criterion/report/index.html` — top-level summary across all
  groups in the run. Skim this to spot which bench moved.
- `target/criterion/<group>/<id>/report/index.html` — one full report
  per metric. P50, P95, P99 estimates with confidence intervals, the
  raw KDE/violin of sample times, an iteration-time scatterplot for
  spotting outliers, and a regression plot against the previous local
  run.
- `target/criterion/<group>/<id>/<baseline>/estimates.json` — machine-
  readable medians + CIs for the named baseline. The CI gate parses
  this style of output (via the `bencher`-format stdout lines) to
  decide pass/fail.

Example: after `cargo bench -p tensor-wasm-bench --bench cold_start`, the 1
MiB restore metric report is at
`target/criterion/cold_start/restore/1048576/report/index.html` and
the published baseline median lives in
[`bench-results/baseline.json`](../bench-results/baseline.json) under
the matching key.

## Reference numbers (host-only, modeled)

The figures below are **placeholders pending the S22 self-hosted CI runner**
and come from quick spot-checks on a developer laptop. Treat them as
ballpark, not contractual — see *Regression policy* for how the gate handles
drift.

| Bench | Input | P50 |
|---|---|---|
| kernel_dispatch/serial | 1000 dispatches | ~150 µs total (~150 ns / dispatch) |
| kernel_dispatch/concurrent_cap64 | 1000 dispatches, 4 worker threads | ~80 µs total |
| cold_start/capture | 1 MiB snapshot | ~5 ms |
| cold_start/capture | 16 MiB snapshot | ~75 ms |
| cold_start/capture | 128 MiB snapshot | ~600 ms |
| cold_start/capture | 512 MiB snapshot | ~2.4 s (modeled, linear extrapolation) |
| cold_start/restore | 1 MiB | ~3 ms |
| cold_start/restore | 16 MiB | ~50 ms |
| cold_start/restore | 128 MiB | ~400 ms |
| cold_start/restore | 512 MiB | ~1.6 s (modeled, linear extrapolation) |
| memory_bandwidth/sequential | 16 MiB | ~3 ms (5+ GB/s host RAM) |
| jit_compile/emit_text | matmul[16x16x16] | ~5-20 µs |
| e2e/healthz | — | ~30-60 µs |
| e2e/create_function | 9 byte payload | ~40-80 µs |
| e2e/invoke_not_found | — | ~30-60 µs |

The host-only `kernel_dispatch` number is essentially a Tokio semaphore
acquire+release round-trip — there is no GPU work in the loop, so the
number reflects scheduler overhead, not real launch latency.

## CUDA-host path (deferred to S22)

When the self-hosted runner with a real GPU is online, the numbers above
will shift in well-understood ways. Rough expectations:

- **`cold_start/restore`** picks up an additional 20-200 ms of UVM
  page-migration cost on first touch, depending on PCIe bandwidth. PCIe 4.0
  x16 is ~32 GB/s on paper, less in practice once you account for
  small-transfer overhead and contention with other workloads on the host.
- **`kernel_dispatch`** overhead drops to roughly **5-20 µs per dispatch**
  once the immediate-resolve stub is replaced with a CUDA Event-based sync
  in the runtime. The semaphore cost stays the same; what changes is that
  each permit is now backed by a real launch + event record.
- **`memory_bandwidth`** for device-resident buffers is dominated by HBM2
  or HBM3 bandwidth — 500-3000 GB/s for large sequential transfers,
  depending on the SKU. Strided patterns fall off the same way they do on
  host RAM, just at much higher absolute throughput.
- **`jit_compile`** is mostly host-side text emission today; once the
  nvrtc-backed path lands, expect a one-time ~10-100 ms hit per unique
  blueprint, amortized by the fingerprint cache.

This section will be replaced with measured ranges (not estimates) when
S22 completes.

### Wasm linear memory UVM wiring (v0.3.3)

The numbers above assume the property the v0.3.2 audit flagged as
unverified is actually true: that the wasm linear memory itself lives
in CUDA Unified Memory. As of v0.3.3 it does. `TensorWasmLinearMemory`
constructs a `UnifiedBuffer` whose feature-gated backing routes through
`cuMemAllocManaged` under `--features unified-memory` (and a heap
`Box<[u8]>` otherwise — see [`crates/tensor-wasm-mem/README.md`](../crates/tensor-wasm-mem/README.md)
for the wiring narrative). A guest pointer that flows through the W1.1
wasi-cuda kernel-args pipeline therefore resolves to a host pointer
that doubles as a device pointer, removing the `cudaMemcpy` that would
otherwise show up on every kernel launch. Memory growth is
pre-allocate-at-max (Wasmtime `static`-style); a v0.4 follow-up will
land in-place grow once `cuMemAddressReserve` / `cuMemMap` are wired
through. The build configuration is asserted in
`crates/tensor-wasm-mem/src/wasm_memory.rs` via
`TensorWasmLinearMemory::is_uvm_backed()`.

## Regression policy

The [`bench` workflow](../.github/workflows/bench.yml) runs the full bench
suite on pull requests that touch `crates/tensor-wasm-bench/**` or
`crates/*/src/**`, and compares the result against a committed baseline at
[`bench-results/baseline.json`](../bench-results/baseline.json). The CI step
parses Criterion's `--output-format bencher` lines, looks each tracked
metric up in the baseline, and fails the build when the measured median
exceeds `baseline.median_ns * (1 + (tolerance_pct + regress_pct_threshold) / 100)`.
See [`bench-results/README.md`](../bench-results/README.md) for the
metric-to-source-file map and the re-baseline procedure, and
[`bench-results/baseline-notes.md`](../bench-results/baseline-notes.md)
for the running log of bench-id renames and additions.

In the committed baseline today, `regress_pct_threshold` is **10%** and
per-metric `tolerance_pct` ranges from **30%** (cold-start, where each
sample is tens of milliseconds and noise is small relative to the mean)
to **100%** (sub-microsecond metrics where CV is naturally high). The
effective ceiling for a given metric is the sum of those two — e.g. a
30%-tolerance metric fails only if it regresses by more than 40% above
baseline. This is deliberately loose for the S19 scaffold; the numbers in
`baseline.json` are conservative hand-picked starting points, **not
measured medians**. S22 replaces them with values captured on the
self-hosted CUDA runner, at which point tolerances tighten.

New benches are added to the baseline in a separate commit, after a clean
run on the reference machine, so that adding a bench never lands together
with a code change in the same PR.

Re-baseline procedure:

1. On a clean `main`, run `make ci-bench` (defined in the project Makefile,
   matches the flags the workflow uses).
2. Inspect the diff between `target/criterion/*` and
   `bench-results/baseline.json`. The Criterion HTML reports are the
   easiest way to see what moved and why.
3. Commit the new baseline **only** once you've reviewed each metric's
   change and confirmed it's intentional. A re-baseline PR should explain
   what caused the shift (faster code, slower code, noisier host, etc.).

If a regression is real and expected (e.g. a feature trade-off), the
re-baseline commit and the feature commit should land back-to-back, with
the re-baseline commit message linking to the feature PR.

## How to run locally

```sh
# Full suite (slow — uses Criterion defaults):
cargo bench --workspace

# A single bench file:
cargo bench -p tensor-wasm-bench --bench cold_start

# Compile-only — CI step zero, useful as a fast sanity check:
cargo bench --workspace --no-run

# Match the CI flags exactly (shorter warm-up + measurement windows):
make ci-bench
```

After a run, open `target/criterion/report/index.html` for the full
Criterion report, including P95/P99, histograms, and regression plots
against the previous local run.

See [BUILD.md](BUILD.md) for the wider build-and-test workflow, and
[`crates/tensor-wasm-bench/benches/`](../crates/tensor-wasm-bench/benches/) for the
bench sources.

---

_Status: S19 scaffold. Numbers re-baseline once the S22 self-hosted CUDA runner is online._
