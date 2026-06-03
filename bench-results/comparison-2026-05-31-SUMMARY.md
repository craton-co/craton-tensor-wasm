# Competitor benchmark run — 2026-05-31

Indicative (not publication-grade) competitor comparison run on the dev box.
See `docs/BENCHMARKING.md` for the methodology this approximates and the
disclosures required before any of these numbers are quoted externally.

## Host / disclosures

- **Machine:** Intel 24c/32t, 64 GB RAM, Windows 11, RTX 2060 (SM_75), nightly-2026-04-03.
- **NOT publication-grade:** no CPU-governor pin, no Turbo-off, no core isolation,
  Defender + IDE + ambient load present. Measured CV ~3% on the warm hyperfine
  runs but the absolute floor drifts run-to-run with machine load (231 ms warm /
  563 ms under load on the same workload). Treat ratios, not absolutes, as the
  signal. Per `docs/BENCHMARKING.md` these are "indicative" numbers; the quiet-host
  / pinned-clock capture is still required for external publication.

## Dimension 1 — WASM execution overhead

- **Workload:** `bench-out/loop_sum.wasm` — a 500,000,000-iteration i32-add loop,
  `_start` export. Same `.wasm` on every runtime.
- **Harness:** `hyperfine --warmup 3 --min-runs 30`.
- **Artifacts:** `comparison-dim1-2026-05-31.{json,md}`.

**500M-iteration loop (`loop_sum.wasm`), JIT runtimes only** — WasmEdge's
interpreter cannot finish 500M iterations within 120 s, so this size is
JIT-vs-JIT:

| Runtime | Mean ± stddev | Relative | Notes |
|---|---|---|---|
| TensorWasm (CLI) | 563.4 ± 18.0 ms | 1.02 ± 0.04 | full engine + tenant spawn around Cranelift |
| Wasmtime 45.0.0  | 555.1 ± 16.9 ms | 1.00 (ref) | upstream, Cranelift |

**1M-iteration loop (`loop_1000000.wasm`), all three runtimes** — scaled down so
the WasmEdge interpreter completes (artifacts `comparison-dim1b-1M-2026-05-31.{json,md}`):

| Runtime | Mean ± stddev | Relative | Notes |
|---|---|---|---|
| TensorWasm (CLI)        | 20.8 ± 2.6 ms      | 1.00 (ref) | Cranelift JIT |
| Wasmtime 45.0.0         | 33.7 ± 5.3 ms      | 1.62× slower | Cranelift JIT |
| **WasmEdge 0.15.0 (interpreter)** | **15,936 ± 2,420 ms** | **~765× slower** | pure interpreter; no JIT; AOT segfaults (see below) |
| Wasmer 7.1.0            | — | n/a | could not benchmark (see below) |

**Verdicts:**

- **TensorWasm ≈ Wasmtime on pure compute.** At 500M iters they are statistically
  tied (1.02×, overlapping CIs) — the expected/correct result, since TensorWasm
  wraps Wasmtime+Cranelift, so the wrapper adds no measurable execution overhead.
  (At 1M iters TensorWasm shows a nominal 1.6× *advantage*, but at that tiny
  workload process-startup/cold-cache noise dominates the ~13 ms gap and the 500M
  run is the trustworthy compute comparison — the 1M table exists to give WasmEdge
  a completable workload, not to claim TensorWasm beats Wasmtime.)
- **WasmEdge is ~765× slower** on this tight integer loop. This is the textbook
  **interpreter-vs-JIT** gap, not a subtle finding: WasmEdge 0.15 runs in
  interpreter mode by default (~12–16 µs/iteration) while TensorWasm/Wasmtime
  JIT-compile to native (~0.02–0.05 µs/iteration). WasmEdge's AOT path (`wasmedgec`)
  is the fair-fight mode that would close most of this gap, but on this 0.15
  windows-msvc build the AOT artifact **segfaults at run time** (compile succeeds in
  321 ms; run exits with SIGSEGV / rc 139), so an AOT number could not be obtained.
  Per `docs/BENCHMARKING.md`, comparing a JIT against an interpreter is only fair if
  disclosed — hence the explicit "(interpreter)" label and this note.

### Why Wasmer has no number

- **Wasmer 7.1.0** (installed via `cargo install wasmer-cli`, with and without
  `--features cranelift`): every run of the loop fails with
  `error: No backends support the required features for the Wasm module`. This is
  a Wasmer 7.1 backend/feature-detection issue with the cargo-built CLI on this
  box, not a TensorWasm finding. Would need the standalone Wasmer installer or a
  different backend build to benchmark.

### Methodology note (Windows `cmd` path gotcha)

hyperfine on Windows runs each command through `cmd.exe`, which treats a leading
`forward/slash/path` as a flag (`"target" is not recognized...`). Commands must
use `back\slash\paths` (bare names like `wasmtime` are fine). An earlier run that
"dropped" WasmEdge was this quoting bug, not a runtime failure — fixed here.

## Dimension 3 — GPU kernel dispatch overhead (competitor floor)

- **Harness:** `comparison-harness/cuda/raw_launch.cu` — a tight
  `cuLaunchKernel` loop (CUDA **driver** API, matching TensorWasm's
  `host::launch` path) over the same `vector_add` kernel TensorWasm dispatches.
- **Run:** `raw_launch.exe 100000 65536` (100k launches, 65536-elem vectors), RTX 2060.
- **Artifact:** `comparison-dim3-gpu-2026-05-31.json`.

| Metric | raw `cuLaunchKernel` (ns) |
|---|---|
| mean | 10,540 |
| min  | 3,200 |
| p50  | 5,400 |
| p95  | 33,500 |
| p99  | 49,300 |
| p99.9| 129,000 |

This is the **competitor lower bound** for dimension 3 (no sandbox, no WASM, no
back-pressure). Per `docs/BENCHMARKING.md`, TensorWasm's `dispatch/serial` should
land within 2–5× of this once the wasi-cuda launch path is wired into a GPU-timed
bench. **TensorWasm's own GPU-dispatch number is not yet a turnkey bench** (the
existing `dispatch/*` criterion bench measures scheduling overhead with the
DispatchFuture resolving immediately on non-CUDA; it is not a real GPU
launch-to-completion timing), so the TensorWasm/raw ratio can't be computed from
in-repo benches today. The floor above is the reference for when it is.

## Tooling installed this session (kept for future runs)

- **Wasmtime 45.0.0** — `~/.cargo/bin/wasmtime` (works).
- **WasmEdge 0.15.0** — `tools/wasmedge/bin/wasmedge.exe` (+ `wasmedgec.exe` AOT).
- **Wasmer 7.1.0** — `~/.cargo/bin/wasmer` (broken on this module, see above).
- **GPU dispatch harness** — `comparison-harness/cuda/{raw_launch.cu,build.bat,raw_launch.exe}`
  (compiled with MSVC `cl.exe` + driver-API `cuda.lib`; `raw_launch.cu` is pure
  host code so it does not need nvcc).

## Not run

- Dimensions 2 (cold-start), 4 (multi-tenant), 5 (HTTP gateway) competitor
  comparisons — these need Spin/workerd/Triton/MPS setups not present here.
- The "S22 self-hosted CUDA runner" is GitHub Actions CI infrastructure
  (`docs/runbooks/self-hosted-cuda-runner.md`), not a competitor; GPU benches
  were run directly on this CUDA host instead.
