# Craton TensorWasm — Benchmarking Guide

How to compare TensorWasm against other WASM runtimes, FaaS platforms, and
GPU-dispatch stacks **honestly** — same workload, same hardware envelope,
same statistical bar, with the disclosures a reader needs to reproduce
the result.

This doc is the companion to [`PERFORMANCE.md`](PERFORMANCE.md). That one
covers TensorWasm's internal regression gate (committed `baseline.json`, CI
tolerance policy). This one covers external comparisons — the kind of
numbers you'd put in a blog post, a tech-talk slide, or a customer
evaluation.

If you only read one section: skip to [Anti-cheating checklist](#anti-cheating-checklist).

## Contents

1. [Scope and non-goals](#scope-and-non-goals)
2. [The five dimensions TensorWasm competes on](#the-five-dimensions-tensor-wasm-competes-on)
3. [Choosing your competitor set](#choosing-your-competitor-set)
4. [Hardware and OS normalization](#hardware-and-os-normalization)
5. [Methodology](#methodology) — includes [Tail latency](#tail-latency)
6. [Workload corpus](#workload-corpus)
7. [Per-competitor recipes](#per-competitor-recipes)
8. [Bench-ID to competitor-metric map](#bench-id-to-competitor-metric-map)
9. [Reporting format](#reporting-format)
10. [Anti-cheating checklist](#anti-cheating-checklist)
11. [Where TensorWasm wins, where it won't](#where-tensor-wasm-wins-where-it-wont)

---

## Scope and non-goals

**In scope.** Side-by-side measurements of TensorWasm's published bench groups
(`cold_start/*`, `dispatch/*`, `jit_compile/*`, `e2e/*`,
`memory_bandwidth/*`, `tenant_registry/*`, `call_export/*`,
`invoke_stream/*`) against equivalent metrics in other runtimes, using a
shared workload and hardware envelope.

**Not in scope.** "Vibes" benchmarks (one-shot `time ./run.sh`),
synthetic micro-benches that don't map to a real TensorWasm subsystem,
marketing comparisons that change the workload between runtimes. If the
number you want to publish doesn't fit one of the five dimensions below,
add a bench group to `tensor-wasm-bench` first.

**Hard rule.** Every external comparison number must be reproducible
from this repo at a pinned SHA with the commands in the report.
"Trust me, I ran it" is not a benchmark.

---

## The five dimensions TensorWasm competes on

Each dimension has a corresponding TensorWasm bench group, a class of
competitor, and a fair-fight constraint that makes the comparison
meaningful.

| Dimension | TensorWasm bench group | Competitor class | Fair-fight constraint |
|---|---|---|---|
| 1. WASM execution overhead | `e2e/*`, plus a custom microbench | Wasmtime, Wasmer, WasmEdge, V8/Node | Same `.wasm` file, same export, same input, same compile mode (Cranelift vs Singlepass etc. disclosed) |
| 2. Cold-start latency | `cold_start/capture`, `cold_start/restore`, `cold_start/disk_round_trip` | Spin, Fermyon Cloud, Lambda SnapStart, Cloudflare workerd, raw Wasmtime `Module::deserialize` | Same payload size, same persistence medium (in-memory vs disk vs network), warm vs cold disclosed |
| 3. Kernel dispatch overhead | `dispatch/serial`, `dispatch/concurrent_cap64` | Raw `cuLaunchKernel` from C++, Triton dispatcher, JAX/XLA dispatch, PyTorch eager launch | Same kernel, same arg layout, same launch grid, same stream/queue depth |
| 4. Multi-tenant context switching | `tenant_registry/lookup`, `tenant_registry/consume_release` | NVIDIA MPS bare-metal, Triton Inference Server, k8s GPU sharing (time-slicing), MIG partitions | Same tenant count, same isolation guarantee level (memory-isolated vs time-sliced), disclosed |
| 5. HTTP gateway floor | `e2e/healthz`, `e2e/create_function`, `e2e/invoke_not_found` | workerd, Spin gateway, AWS Lambda URL, Fermyon Cloud, raw axum/hyper/actix | Identical route shape, identical payload, single-host (no cross-AZ network), same concurrency |

These five dimensions are deliberately separate. Do not collapse them
into a single "TensorWasm vs X" number — there's no honest way to do that.
A Wasmtime cold-start comparison is meaningful; a "TensorWasm vs Wasmtime"
single number is not, because TensorWasm adds a snapshot subsystem and a
tenant registry on top.

---

## Choosing your competitor set

Pick by what claim you want to support:

- **"TensorWasm's WASM execution overhead is comparable to upstream Wasmtime."**
  → dimension 1 vs Wasmtime only. Same `.wasm`, same Cranelift settings.
  Expectation: TensorWasm should be within **±5%** of upstream Wasmtime on
  pure compute (we're a thin wrapper); larger gaps are a TensorWasm bug worth
  filing.

- **"TensorWasm's cold-start beats $COMPETITOR."**
  → dimension 2. Compare `cold_start/restore` against the competitor's
  warm-start-after-snapshot equivalent. Be explicit about what's loaded
  (Wasm module vs Wasm + GPU residency vs Wasm + GPU + tenant state) —
  these have very different costs.

- **"TensorWasm's GPU dispatch overhead is close to raw CUDA."**
  → dimension 3 vs a C++ harness that calls `cuLaunchKernel` directly.
  Expectation: TensorWasm's `dispatch/serial` should be within **2-5×** of raw
  `cuLaunchKernel` on a CUDA host once the immediate-resolve stub is
  replaced (S22). On a non-CUDA host the comparison is meaningless —
  don't try it.

- **"TensorWasm isolates tenants with less overhead than $COMPETITOR."**
  → dimension 4. Use the same tenant count and the same per-tenant
  workload. Be explicit about the isolation model — MPS gives you
  spatial sharing, MIG gives you hard partitioning, TensorWasm gives you
  per-context CUDA streams plus the `TenantRegistry` quota gate; they
  are not equivalent and the comparison must say so.

- **"TensorWasm's HTTP floor is competitive with $FAAS_PLATFORM."**
  → dimension 5. **Always single-host, always local** for the comparison;
  network latency dominates everything else and is not a runtime property.

Mixing dimensions in a single chart is the most common way to produce
a misleading benchmark. Don't do it. One chart per dimension.

---

## Hardware and OS normalization

Run every comparison run on the **same physical machine**, in the
**same boot session**, with the following pinned:

### CPU

- Disable Turbo Boost / CPB. On Linux:
  ```sh
  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # Intel
  echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost           # AMD
  ```
- Pin the CPU governor to `performance`:
  ```sh
  sudo cpupower frequency-set -g performance
  ```
- Isolate the cores you'll bench on (kernel boot param `isolcpus=2-5`)
  and pin the bench process with `taskset -c 2-5 cargo bench ...`.
- Disable SMT/HT on the bench cores if your competitor has different
  SMT behavior — keep both runtimes on physical-core-only or both on
  SMT-on, never mixed.
- Drop the page cache before any disk-touching bench:
  `sudo sysctl -w vm.drop_caches=3`. The `cold_start/disk_round_trip`
  group needs this to be honest.

### GPU (if comparison covers dimensions 3 or 4)

- Pin GPU clocks to a known SKU rate. On NVIDIA:
  ```sh
  sudo nvidia-smi -lgc <gpu_clock_mhz>     # lock graphics clock
  sudo nvidia-smi -lmc <mem_clock_mhz>     # lock memory clock
  sudo nvidia-smi -ac <mem_mhz>,<gfx_mhz>  # legacy combo form
  ```
- Set `nvidia-smi -pm 1` (persistence mode on) so the driver doesn't
  unload between runs.
- Set `nvidia-smi -c EXCLUSIVE_PROCESS` for single-tenant comparisons;
  reset to `DEFAULT` (and start `nvidia-cuda-mps-control -d`) for MPS
  comparisons. Disclose which mode in the report.
- Drain other workloads — confirm `nvidia-smi` shows 0% utilization and
  no processes before starting.
- For multi-GPU hosts, pin to one GPU with `CUDA_VISIBLE_DEVICES=0`
  and disclose which device (full `nvidia-smi -q` for the device id).
- If your bench depends on PCIe transfer (UVM cold-touch), disclose
  PCIe gen and width: `lspci -vvv -d 10de: | grep -E 'LnkSta|LnkCap'`.

### NUMA / memory

- If the host has > 1 NUMA node, pin the bench to one node:
  `numactl --cpunodebind=0 --membind=0 cargo bench ...`.
- Set `vm.swappiness=0` for the duration of the benchmark session.
- Confirm no other workload is pressuring memory: `free -h` and
  `vmstat 1 5` before each comparison batch.

### Process hygiene

- Stop unrelated daemons (mail, sync clients, IDE indexers, package
  managers, anti-virus scanners). On Linux `systemctl list-units
  --state=running` is a useful pre-bench audit.
- Run `cargo bench` from a TTY, not from VS Code's integrated
  terminal — IDE telemetry can perturb sub-millisecond metrics.
- Bench at the **same time of day** on systems with thermal limits;
  ambient temperature shifts P95 measurably on a Threadripper.

Record every one of these in the [reporting JSON](#reporting-format).
If you can't honestly check the box, don't publish the number.

---

## Methodology

TensorWasm's benches use Criterion with these defaults — apply the same to
your competitor harness or the comparison is meaningless:

- **Warm-up.** At least 3 s warm-up before any sample is recorded. For
  JIT-tiered competitors (V8, JS engines) extend to 10-30 s so all
  tiers stabilize.
- **Sample count.** Minimum **30 samples** per metric. Below 30 the
  median estimator is too noisy to claim a < 10 % difference.
- **Measurement window.** 5 s minimum per Criterion config. Sub-µs
  metrics need longer (Criterion auto-extends via iteration batching).
- **Outlier detection.** Criterion reports outliers (mild / severe);
  any bench with > 10 % severe outliers is noisy and the run should be
  retried on a quieter host before publishing.
- **CV target.** Coefficient of variation < 5 %. Above 5 %, fix the
  noise source (background process, thermal throttling, page-cache
  warmth) before publishing. The repo ships
  [`scripts/run-quiet-bench.sh`](../scripts/run-quiet-bench.sh)
  (+ a `.ps1` Windows equivalent) that raises the Criterion
  sample-count from 100 to 500, pins the CPU governor to performance,
  and drops the page cache between groups. It's the "usable middle
  ground" — full publishable noise reduction also requires the
  isolcpus / Turbo / SMT / Defender steps listed in "Hardware and OS
  normalization" below.

  **Disclosure for committed bench numbers (audit Problem #9).** The
  `bench-results/baseline.json`, `bench-results/tail-latency.json`,
  and `bench-results/dispatch-future-backends.json` numbers were
  captured on a developer Windows 11 host with Defender running, IDE
  open, and other ambient processes — **CV typically > 5 %** per
  metric. They are committed as **noise-floor measurements suitable
  for regression-gate tripping** (a 2× drift will fire even at this
  noise level) but NOT as the numbers to quote in an external
  comparison. The S22 self-hosted runner (audit Problem #8 →
  [`.github/workflows/cuda.yml`](../.github/workflows/cuda.yml) +
  [`docs/runbooks/self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md))
  will produce the publication-grade numbers once it lands; until
  then operators evaluating TensorWasm should run the quiet script
  on their own hardware and compare with their own production
  baseline.
- **Distribution shape.** Report P50, P95, P99 — **never just mean**.
  Latency distributions in async runtimes are long-tailed; the mean
  hides the tail.

### Statistical significance

Don't claim "X is faster than Y" without a confidence interval. Two
medians 2 % apart with overlapping 99 % CIs are the same number.

Criterion writes per-metric CIs into
`target/criterion/<group>/<id>/<baseline>/estimates.json` —
the `lower_bound` / `upper_bound` fields. Report them alongside the
median. A useful rule of thumb: if the 95 % CIs of two competitors
overlap, the comparison is **inconclusive** and should be stated as
such, not as a win.

### Tail latency

Criterion's default reporter publishes mean, std-dev and median per metric;
it does **not** publish P99 or P99.9 out of the box, and its default ~100
samples per metric is too coarse to resolve a stable P99.9 anyway. For the
v0.3 ["Production observability"](PATH-TO-V1.md#v030--production-observability)
milestone we need a long-tail floor on the dispatch path and the HTTP
gateway path, so the dedicated bench file
[`crates/tensor-wasm-bench/benches/tail_latency.rs`](../crates/tensor-wasm-bench/benches/tail_latency.rs)
runs a hand-rolled sampling loop alongside the Criterion suite.

- **Sample count:** 10 000 raw `Duration` observations per metric (warm-up
  1 000 iterations, un-counted). This places the P99.9 sample at sorted
  rank 9 990 (the tenth-worst observation) — large enough to sit inside
  the population tail rather than at the global max.
- **Percentile algorithm:** nearest-rank (`samples[ceil(p * n) - 1]`),
  matching `hdrhistogram` and the Tigerbeetle / Datadog tail-tracking
  references. Linear interpolation would change the numbers by at most one
  inter-sample gap, which is well inside the per-sample noise floor on a
  µs-scale dispatch metric.
- **Metrics covered:** `dispatch/serial/100`, `dispatch/concurrent_cap64/100`,
  `e2e/healthz/get`, `e2e/invoke_not_found/post`. Each metric re-uses the
  same setup helpers as the corresponding Criterion bench so the P50
  numbers line up by construction.
- **Tracing overhead:** the W4.1 OpenTelemetry spans add a constant
  ~50-150 ns to every e2e request even with no subscriber attached. This
  raises the floor (P50 and P99.9 by the same amount) but does **not**
  distort the `p99_9 - p50` tail gap, so the published numbers are kept
  raw — they reflect what an operator actually sees. To isolate the
  tracing tax, set `TENSOR_WASM_TRACING=off` and re-run; the delta is
  the cost.
- **Output:** one JSON line per metric to stdout (CI-grep prefix
  `TAIL_LATENCY `) plus a sidecar at `bench-results/tail-latency.json`
  when the bench is run from the workspace root. The file is **not**
  consumed by the regression gate — see
  [`bench-results/README.md#tail-latency-artefact`](../bench-results/README.md#tail-latency-artefact).
- **Backend axis (W4.4 — [RFC 0001](../rfcs/0001-cuda-oxide-integration.md)
  Unresolved questions extension).** The bench carries a compile-time
  `BACKEND_LABEL` that flows into the Criterion group name
  (`tail_latency_<backend>`), every `TAIL_LATENCY` JSON line as a
  `"backend":` field, and the rendered result file's top-level `backend`
  field plus a per-metric `backend` field. The label is selected by
  feature flag at bench-build time:

  | `cargo bench --features ...` | `BACKEND_LABEL` | `bench-results/tail-latency.json` `backend` |
  |---|---|---|
  | (none, default)              | `unified-memory` | `"unified-memory"` |
  | `cudarc-backend`             | `cudarc`         | `"cudarc"` |
  | `cuda-oxide-backend`         | `cuda-oxide`     | `"cuda-oxide"` |

  Per-backend regressions become visible only by running the bench
  **three times**, once per backend flag, and diffing the three result
  files. The bench file does not perform the multi-run itself — the
  CI matrix wiring is wave-4 ops work. Manual operator recipe:

  ```sh
  # 1. unified-memory (the cust-backed historical default; on v0.4
  #    deprecation watch per RFC 0001 Unresolved questions).
  cargo bench -p tensor-wasm-bench --bench tail_latency
  mv bench-results/tail-latency.json bench-results/tail-latency-unified-memory.json

  # 2. cudarc (the cust → cudarc spike). The bench-layer flag is
  #    label-only and does NOT pull in cust — see the comment in
  #    crates/tensor-wasm-bench/Cargo.toml for why.
  cargo bench -p tensor-wasm-bench --bench tail_latency --features cudarc-backend
  mv bench-results/tail-latency.json bench-results/tail-latency-cudarc.json

  # 3. cuda-oxide (the v0.5 cust successor scaffold from RFC 0001).
  #    Requires libclang available to cuda-bindings' build script
  #    (set LIBCLANG_PATH on Windows; install libclang-dev on Linux).
  cargo bench -p tensor-wasm-bench --bench tail_latency --features cuda-oxide-backend
  mv bench-results/tail-latency.json bench-results/tail-latency-cuda-oxide.json

  # 4. Diff. Any per-backend regression shows up as a p99_9_ns drift
  #    between two of the three files at the same `metric` key.
  diff <(jq .metrics bench-results/tail-latency-unified-memory.json) \
       <(jq .metrics bench-results/tail-latency-cudarc.json)
  ```

  The three files differ only in the top-level `backend` discriminator
  and the per-metric `backend` field today — the dispatch-loop and e2e
  router code paths under measurement are identical across labels in
  v0.3.x. The per-backend split exists so that **once** the cuda-oxide
  port lands at v0.4 and the per-backend dispatch surfaces are real, the
  bench harness is already capturing them under the right label
  without a parallel rewrite.

  Enabling multiple backend flags at once is permitted (the mem crate
  accepts the combination) but the bench picks one label by priority
  (`cuda-oxide` > `cudarc` > `unified-memory`) and announces the choice
  on stderr at bench startup; see the `BACKEND_LABEL` docs in
  [`crates/tensor-wasm-bench/benches/tail_latency.rs`](../crates/tensor-wasm-bench/benches/tail_latency.rs)
  for the rationale.

```sh
cargo bench -p tensor-wasm-bench --bench tail_latency
```

### Typed-args call_export (`call_export/*`)

Batch 6 introduced [`TensorWasmExecutor::call_export_with_args`], a
slice-of-`WasmArg` entrypoint that replaces the legacy
`typed::<(), ()>`-shaped `call_export` shim for any guest export with a
non-trivial signature. The bench file
[`crates/tensor-wasm-bench/benches/call_export_args.rs`](../crates/tensor-wasm-bench/benches/call_export_args.rs)
pins the overhead of the new path so a future args-marshalling regression
trips the gate before it ships:

- **`call_export/noargs/call_export_with_args_empty`** — drives the
  typed-args entrypoint with an empty arg slice on a `() -> ()` export
  (`noop`). Compared against the legacy no-args shim this isolates the
  per-call slice-iteration + signature-reflection cost.
- **`call_export/args/two_i32`** — drives `call_export_with_args` with
  `[WasmArg::I32(1), WasmArg::I32(2)]` against an `(i32, i32) -> i32`
  export (`add`). Measures the actual marshalling work: enum → wasmtime
  `Val`, slice-length check against the typed export, and the
  `Val`-array-to-`Results` conversion on return.

Both groups spawn + terminate the instance inside the timed loop so the
absolute numbers are anchored to the same envelope as the `/invoke` HTTP
path; the cross-group delta is the args-path overhead in isolation. The
two `baseline.json` entries land with `regression_check: false` and zero
medians as stubs — the first quiet-host capture (`run-quiet-bench.sh`)
must populate real medians before the gate is flipped on.

```sh
cargo bench -p tensor-wasm-bench --bench call_export_args
```

### Streaming invoke (`invoke_stream/*`)

Batch 7 restores the `/invoke-stream` route (B7.1 — in flight on a
parallel branch at time of writing). The bench file
[`crates/tensor-wasm-bench/benches/streaming_invoke.rs`](../crates/tensor-wasm-bench/benches/streaming_invoke.rs)
pins the floor of the streaming path against the synchronous `/invoke`
baseline so v0.4's actual chunk-emitter has a regression target:

- **`invoke_stream/baseline_invoke`** — synchronous `/invoke` reference
  number. Same handler depth, same registry lookup, same body drain.
- **`invoke_stream/sse`** — `/invoke-stream` with
  `Accept: text/event-stream`. Measures the SSE framing floor (v0.3.7
  emits a single `event: scaffold` frame).
- **`invoke_stream/chunked`** — `/invoke-stream` with the default
  `Accept`. Measures the chunked-transfer-encoding fallback floor.

The bench file currently ships as a `todo!()`-bodied placeholder because
the `/invoke-stream` route is not yet on `build_router` in this worktree;
running it surfaces a clear "B7.1: route not yet wired" panic rather than
emitting misleading numbers against the legacy `/invoke` path. The
`baseline.json` stubs carry `regression_check: false` until B7.1 merges
and the placeholders are replaced with real router-driven sample loops
(pattern after `tail_latency.rs::measure_invoke_not_found`).

```sh
cargo bench -p tensor-wasm-bench --bench streaming_invoke
```

### Cold vs warm

State which one you're measuring, every time. TensorWasm's three cold-start
metrics are deliberately distinct:

| Metric | What's cold | What's warm |
|---|---|---|
| `cold_start/capture` | Nothing (steady-state) | All host caches, GPU contexts |
| `cold_start/restore` | Nothing (steady-state, page cache hot) | zstd dictionary, OS page cache |
| `cold_start/disk_round_trip` | Disk read + zstd decode | Nothing (forces a true file round-trip per sample) |

For a true cold comparison against another runtime, mirror this
three-way split — measure their warm-restore, their warm-deserialize,
and their cold-from-disk separately.

### Apples-to-apples binding

When a competitor exposes multiple compilation modes (Wasmer's
Cranelift / Singlepass / LLVM; Wasmtime's Cranelift / Winch), run them
**all** and publish all three. Picking the slowest competitor mode to
make TensorWasm look good is dishonest. Picking the fastest is fine if you
disclose. The safe path is publishing every available mode.

---

## Workload corpus

Use the same workload across competitors. TensorWasm ships these fixtures:

| Path | Format | Use for |
|---|---|---|
| `tests/wasm-fixtures/matrix_multiply.wat` | WAT | Dimensions 1, 5 — small, deterministic, easy to compile in any runtime |
| `kernels/vector_add.ptx` | PTX | Dimension 3 — direct GPU dispatch comparison against raw `cuLaunchKernel` |

For most public comparisons these are too small. Augment with:

- **`vector_add` at sizes 2^10, 2^16, 2^20, 2^24 elements.** Tests
  dispatch overhead vs. throughput crossover. Use the same kernel
  source across runtimes (the PTX in `kernels/` is the reference).
- **`matmul` at 256x256, 1024x1024, 4096x4096 f32.** Tests JIT quality
  for dimension 1; tests memory-bandwidth-bound throughput for
  dimension 3.
- **`conv2d` 3x3 stencil at 1024x1024 f32.** Tests JIT quality for
  patterns the auto-offload pipeline (`tensor-wasm-jit`) actually recognizes.
- **A "do-nothing" wasm export** (empty function, returns `i32(0)`).
  Isolates dispatch overhead from compute.
- **A small ONNX inference model** (e.g. MobileNetV2 at 224x224 f32)
  for end-to-end inference comparisons. Convert with the competitor's
  preferred toolchain; for TensorWasm use `tensor-wasm-cli run`.

Workload files used for any published comparison **must** be committed
or linked to a permanent URL (HuggingFace hash, ONNX zoo SHA, etc.).
"I used MobileNet" is not a workload spec.

---

## Per-competitor recipes

Each recipe gives: the install command, the equivalent metric, the
exact invocation, and the pitfall to watch for.

### vs Wasmtime (upstream)

The most important comparison — TensorWasm wraps Wasmtime, so any large gap
on dimension 1 is a TensorWasm regression.

```sh
# Install matching Wasmtime version
cargo install wasmtime-cli --version <pin matching Cargo.lock>

# TensorWasm side (dimension 1):
cargo bench -p tensor-wasm-bench --bench e2e_inference -- --save-baseline tensor-wasm

# Wasmtime side — wrap the same .wasm in a minimal harness:
# (see comparison-harness/wasmtime/main.rs — write it once, commit it)
cargo run --release -p wasmtime-comparison -- --wasm tests/wasm-fixtures/matrix_multiply.wat
```

**Pitfall.** TensorWasm defaults to Cranelift; Wasmtime defaults to
Cranelift; both should match. If you set `TENSOR_WASM_COMPILER=winch`
(if/when we expose it) the comparison shifts and you must disclose.
Also: Wasmtime's `Module::deserialize` skips parsing — if you compare
that against TensorWasm's `cold_start/restore` you're comparing the wrong
layer; restore does parse + tenant-state restore on top.

### vs Wasmer

```sh
cargo install wasmer-cli                 # or per-backend variant
wasmer compile --backend cranelift tests/wasm-fixtures/matrix_multiply.wat -o mm.wasmu
hyperfine --warmup 5 -m 30 'wasmer run mm.wasmu'
```

**Pitfall.** Wasmer supports three backends (Cranelift, Singlepass,
LLVM) with very different compile-vs-runtime tradeoffs. Publish all
three. Don't compare TensorWasm (Cranelift) against Wasmer (LLVM) without
disclosure — LLVM compiles 5-20× slower but runs faster.

### vs WasmEdge

```sh
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash
wasmedgec mm.wasm mm.so       # AOT compile
hyperfine --warmup 5 -m 30 'wasmedge mm.so'
```

**Pitfall.** WasmEdge has an AOT path (`wasmedgec`) and an interpreter
path; running the interpreter and calling it "WasmEdge" is unfair.
Publish AOT numbers when the runtime supports AOT.

### vs Spin / Fermyon

Spin is a higher-level runtime built on Wasmtime. Compare on dimension
5 (HTTP gateway floor) and dimension 2 (cold-start of a Spin component
vs TensorWasm's `cold_start/restore`).

```sh
# Install spin
curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash
spin new -t http-rust mycomponent
spin build && spin up &
hyperfine --warmup 5 -m 30 'curl -s http://localhost:3000/'
```

**Pitfall.** Spin's cold-start includes component instantiation +
WASI setup; TensorWasm's `cold_start/restore` includes snapshot decode +
tenant registry repopulation. These are not the same thing — describe
both pipelines in the report so the reader can judge.

### vs workerd (Cloudflare)

```sh
npm install -g workerd
# Write a minimal worker that returns 200 immediately
workerd serve config.capnp &
hyperfine --warmup 5 -m 30 'curl -s http://localhost:8080/'
```

**Pitfall.** workerd runs JS, not Wasm-by-default. If your worker
loads a `.wasm`, the comparison is on dimension 1 + dimension 5 mixed;
if it's pure JS it's dimension 5 only. State which.

### vs raw CUDA (dimension 3 upper bound)

Write a 50-line C++ harness that calls `cuLaunchKernel` in a loop
with the same kernel TensorWasm uses:

```cpp
// comparison-harness/cuda/raw_launch.cu — commit this
for (int i = 0; i < N; ++i) {
    cuLaunchKernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0, stream,
                   args, nullptr);
}
cuStreamSynchronize(stream);
```

Run it under the same fixed clocks as TensorWasm's `dispatch/serial`. The
ratio `tensor_wasm_dispatch_ns / raw_cuda_dispatch_ns` is the TensorWasm GPU
dispatch overhead — publish it directly. Once the v0.2 dynamic-argv
work lands (currently returns `KernelArgsUnsupported`, see
[RISKS.md](RISKS.md)), the gap should close to 2-5×.

### vs Triton Inference Server (dimension 4)

TensorWasm competes with Triton on the "many tenants on one GPU" axis. Set
up Triton with `N` model instances of the same model, hit it with
matched concurrent load, compare per-request P95 against TensorWasm serving
`N` tenants of the same workload.

```sh
# Triton side
docker run --gpus=1 -p 8000:8000 \
  -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
# TensorWasm side — register N tenants, invoke same workload
```

**Pitfall.** Triton has its own request batcher; TensorWasm doesn't (yet).
Compare with batching disabled on Triton or with matched batching
behavior on both. A Triton run with dynamic batching against a TensorWasm
run without is not a comparison, it's a batcher demo.

### vs native (dimension 1 lower bound)

Compile the same algorithm as a native Rust binary and bench it under
hyperfine. This is the "no VM" floor. TensorWasm's overhead vs native is
the cost of running in WebAssembly at all — useful context for any
dimension-1 chart.

---

## Bench-ID to competitor-metric map

For each TensorWasm bench in `baseline.json`, here's the equivalent metric
to measure on the competitor side. Use this when writing a comparison
report — every TensorWasm row needs a matched competitor row, never an
unmatched one.

| TensorWasm bench id | Competitor | Equivalent metric on competitor |
|---|---|---|
| `cold_start/capture/<N>` | Wasmtime | `Module::serialize` of equivalent payload |
| `cold_start/capture/<N>` | Spin | `spin build` step (note: includes more) |
| `cold_start/restore/<N>` | Wasmtime | `Module::deserialize` |
| `cold_start/restore/<N>` | Wasmer | `Module::deserialize_from_file` (with matched backend) |
| `cold_start/restore/<N>` | workerd | First-request latency after worker upload |
| `cold_start/restore/<N>` | Lambda | SnapStart "restore" duration from CloudWatch |
| `cold_start/disk_round_trip/<N>` | Wasmtime | `serialize` + `fs::write` + `fs::read` + `deserialize` round-trip |
| `dispatch/serial/<N>` | raw CUDA | `cuLaunchKernel` loop, same kernel, single stream |
| `dispatch/serial/<N>` | Triton | C-API direct dispatch via Triton's backend SDK |
| `dispatch/concurrent_cap64/<N>` | raw CUDA | N concurrent streams, same kernel |
| `dispatch/concurrent_cap64/<N>` | MPS | N-client load against single GPU with MPS daemon |
| `memory_bandwidth/sequential/<N>` | C `memcpy` | `BUF_SIZE` `memcpy` loop |
| `memory_bandwidth/sequential/<N>` | raw CUDA | `cudaMemcpyAsync` D2D, same size |
| `memory_bandwidth/strided/<N>` | C strided copy | Same stride, same buffer size |
| `jit_compile/emit_text/matmul[...]` | Triton | `triton.compile` (note: Triton emits LLVM IR + PTX) |
| `jit_compile/emit_text/matmul[...]` | nvrtc | `nvrtcCompileProgram` of equivalent CUDA C++ |
| `jit_compile/cache/warm_hit` | Triton | second-call `triton.compile` (cached) |
| `tenant_registry/lookup/<N>` | MPS | `cuCtxPushCurrent`/`cuCtxPopCurrent` round-trip |
| `tenant_registry/consume_release/256KiB` | k8s GPU sharing | scheduler request+release round-trip |
| `e2e/healthz/get` | workerd | matched healthcheck route latency |
| `e2e/healthz/get` | Spin | matched healthcheck route latency |
| `e2e/create_function/post` | workerd | matched worker upload latency |
| `e2e/invoke_not_found/post` | workerd | matched 404 latency |

If your comparison covers a TensorWasm metric not in this table, add a row
to this doc in the same PR that publishes the comparison.

---

## Reporting format

### Required JSON sidecar

Every external comparison publishes a `comparison.json` alongside
the chart or table. Schema:

```json
{
  "report_id": "tensor-wasm-vs-wasmtime-2026-06-01",
  "tensor_wasm_sha": "abcd123...",
  "tensor_wasm_toolchain": "nightly-2026-04-03",
  "hardware": {
    "cpu": "AMD Threadripper 7980X (64-core)",
    "cpu_governor": "performance",
    "turbo": false,
    "smt": "off",
    "isolated_cores": "8-15",
    "ram_gb": 256,
    "numa_nodes": 1,
    "gpu": "NVIDIA H100 80GB PCIe",
    "gpu_driver": "550.54.15",
    "cuda": "12.4.1",
    "gpu_clocks_locked": { "graphics_mhz": 1755, "memory_mhz": 2619 },
    "gpu_compute_mode": "EXCLUSIVE_PROCESS",
    "mps": "off",
    "pcie": "gen5 x16"
  },
  "os": {
    "kernel": "Linux 6.6.32",
    "distro": "Ubuntu 24.04 LTS",
    "page_cache_dropped_per_sample": true
  },
  "methodology": {
    "warmup_s": 5,
    "measurement_s": 10,
    "samples_per_metric": 50,
    "harness": "criterion 0.5",
    "cv_target_pct": 5,
    "outlier_policy": "rerun if severe outliers > 10%"
  },
  "competitors": [
    {
      "name": "wasmtime",
      "version": "25.0.3",
      "config": "cranelift, async, epoch-interruption-on"
    },
    {
      "name": "wasmer",
      "version": "5.0.0",
      "config": "backend=cranelift"
    }
  ],
  "results": [
    {
      "tensor_wasm_metric": "cold_start/restore/16777216",
      "tensor_wasm_p50_ns": 50000000,
      "tensor_wasm_p95_ns": 55000000,
      "tensor_wasm_p99_ns": 62000000,
      "tensor_wasm_ci95_lower_ns": 49100000,
      "tensor_wasm_ci95_upper_ns": 51200000,
      "competitor": "wasmtime",
      "competitor_metric": "Module::deserialize(16 MiB)",
      "competitor_p50_ns": 18000000,
      "competitor_p95_ns": 19500000,
      "competitor_ci95_lower_ns": 17800000,
      "competitor_ci95_upper_ns": 18300000,
      "verdict": "TensorWasm ~2.8x slower; expected — TensorWasm restore includes tenant-state replay and is not a pure deserialize",
      "raw_criterion_json": "artifacts/tensor-wasm-restore-16M.json"
    }
  ]
}
```

`verdict` is mandatory and **must** be honest. "Comparable", "slower
by X", "faster by X with overlapping CIs (inconclusive)" are all
acceptable. "TensorWasm wins" without numbers backing it is not.

### Markdown table template

```
| Metric (TensorWasm bench id)           | TensorWasm P50       | wasmtime 25.0  | wasmer 5.0 (cl)| Δ vs wasmtime |
|----------------------------------|----------------|----------------|----------------|---------------|
| cold_start/restore/16777216      | 50.0 ± 1.1 ms  | 18.0 ± 0.2 ms  | n/a            | 2.78x slower  |
| dispatch/serial/100              | 12.3 ± 0.3 µs  | n/a            | n/a            | n/a           |
| ...                              |                |                |                |               |
```

`±` value is the half-width of the 95 % CI. `n/a` is fine and
**preferable** to a made-up number when the competitor doesn't expose
that metric.

### Required artifacts in the report

- `comparison.json` (above schema)
- Raw Criterion JSON for each TensorWasm metric (`target/criterion/.../estimates.json`)
- Raw competitor output (hyperfine JSON, nsys report, custom harness CSV)
- The competitor harness source (commit it under `comparison-harness/`)
- The exact commands run, in order, copy-pasteable

If any of these is missing, the comparison is unverifiable and should
not be published under the TensorWasm name.

---

## Anti-cheating checklist

A benchmark is dishonest if **any** of these are true. Re-run before
publishing if you can't tick every box.

- [ ] Same workload across all runtimes (same `.wasm`, same input, same export, same kernel source).
- [ ] Same hardware envelope, same boot session.
- [ ] CPU governor and GPU clocks pinned and disclosed.
- [ ] Background load audited and quiet (`top`, `nvidia-smi`, `iotop`).
- [ ] At least 30 samples per metric.
- [ ] P50, P95, P99 all reported (not just mean, not just P50).
- [ ] 95 % CIs reported and overlapping-CI cases marked "inconclusive".
- [ ] CV < 5 % per metric, or noise source documented.
- [ ] TensorWasm built with `--release` and `RUSTFLAGS` defaults. No `target-cpu=native` unless every competitor was built the same way.
- [ ] Competitor built with **its** release config (AOT mode for WasmEdge, LLVM backend disclosed for Wasmer, etc.).
- [ ] Warm-up sufficient for the slowest-tiering competitor (V8 needs 10+ s; Wasmtime Cranelift needs ~3 s).
- [ ] Cold vs warm explicitly stated per metric. Page cache state disclosed for disk-touching benches.
- [ ] No cross-dimension collapses ("TensorWasm vs X overall" is forbidden).
- [ ] No selective omission of competitor backends/modes you ran.
- [ ] Repo SHA pinned; competitor versions pinned; `comparison.json` committed.
- [ ] Every chart and number is reproducible from the published commands.

If a maintainer can't reproduce your number from the artifacts, the
number doesn't exist.

---

## Where TensorWasm wins, where it won't

Honesty as marketing strategy: publish where TensorWasm loses too. Readers
trust comparisons that include losses.

### TensorWasm should win on

- **`cold_start/restore` with snapshot reuse vs cold Wasmtime
  `Module::new`** — we ship the snapshot subsystem, upstream Wasmtime
  doesn't bundle one. Expect 5-20× on warm restore.
- **Multi-tenant GPU isolation vs naive `cudaSetDevice` sharing** —
  the `TenantRegistry` quota path is < 5 µs and gives real isolation,
  not best-effort.
- **Sandboxed GPU dispatch vs no-sandbox CUDA wrappers** — TensorWasm pays
  for the WASM sandbox and still gets within target overhead of raw
  CUDA on dimension 3 (target: 2-5× of `cuLaunchKernel` post-S22).

### TensorWasm will lose on

- **Pure WASM execution speed vs hand-tuned LLVM-backed Wasmer** —
  Wasmer LLVM compiles slower but runs faster on tight loops. If your
  workload is "I have one Wasm module I call a million times,"
  Wasmer-LLVM is probably faster. TensorWasm's wins are elsewhere.
- **Cold-start vs in-process Wasmtime `deserialize`** — TensorWasm's restore
  is a superset (snapshot decode + tenant state); a pure
  `Module::deserialize` is always going to be a tighter loop.
- **Raw GPU dispatch latency vs C++ CUDA** — we add WASI-GPU bounds
  checks and a back-pressure semaphore that raw C++ doesn't. We
  expect 2-5× post-S22; never claim parity.
- **Pre-built FaaS platforms with CDNs** — workerd at the edge with
  Cloudflare's network is unbeatable on user-perceived latency for
  cross-region requests. TensorWasm is a runtime; "vs Cloudflare Workers
  on a self-hosted box" is fair, "vs Cloudflare Workers on Cloudflare"
  is not.

Saying these out loud in the same report as TensorWasm's wins is what
makes the wins credible.

---

## Related docs

- [PERFORMANCE.md](PERFORMANCE.md) — internal regression-gate policy
  and committed baseline.
- [`bench-results/README.md`](../bench-results/README.md) —
  metric-to-source-file map and re-baselining procedure.
- [`crates/tensor-wasm-bench/README.md`](../crates/tensor-wasm-bench/README.md) —
  bench harness crate layout.
- [RISKS.md](RISKS.md) — known v0.1.0 limitations relevant to
  GPU-dispatch comparisons (kernel-args marshalling stub, `cust` EOL).
- [BUILD.md](BUILD.md) — feature-flag taxonomy; matters for
  apples-to-apples comparison of TensorWasm configurations.

---

_Status: written alongside the v0.1.0 fix wave. Update when the S22
self-hosted CUDA runner produces real (measured, not modeled) numbers
for the GPU-bound dimensions._
