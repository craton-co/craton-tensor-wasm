# bench-results/

Committed bench artifacts consumed by the CI regression gate. Anything
here is read by [`.github/workflows/bench.yml`](../.github/workflows/bench.yml)
on PRs that touch `crates/tensor-wasm-bench/**` or `crates/*/src/**`, and the
build fails when measured medians exceed the per-metric ceiling.

## Files

- **`baseline.json`** — committed median + tolerance per tracked Criterion
  metric. One entry per `(group, id)` pair. See "Metric inventory" below
  for what each entry maps back to.
- **`baseline-notes.md`** — running log of bench-id renames, additions,
  removals, and other things the regression-gate Python parser needs to
  know about across releases. Read this before editing `baseline.json`.

## Metric inventory

Every entry in `baseline.json` is `<group>/<id>`, exactly as Criterion
emits it on stdout / writes it into `target/criterion/<group>/<id>/`.
The source crate and file for each metric:

| Baseline key | Source crate | Source file | Bench group | Bench id |
|---|---|---|---|---|
| `cold_start/capture/1048576` | `tensor-wasm-bench` | `benches/cold_start.rs` | `cold_start/capture` | `1048576` (1 MiB) |
| `cold_start/capture/16777216` | `tensor-wasm-bench` | `benches/cold_start.rs` | `cold_start/capture` | `16777216` (16 MiB) |
| `cold_start/restore/1048576` | `tensor-wasm-bench` | `benches/cold_start.rs` | `cold_start/restore` | `1048576` (1 MiB) |
| `cold_start/restore/16777216` | `tensor-wasm-bench` | `benches/cold_start.rs` | `cold_start/restore` | `16777216` (16 MiB) |
| `dispatch/serial/10` | `tensor-wasm-bench` | `benches/kernel_dispatch.rs` | `dispatch/serial` | `10` |
| `dispatch/concurrent_cap64/100` | `tensor-wasm-bench` | `benches/kernel_dispatch.rs` | `dispatch/concurrent_cap64` | `100` |
| `jit_compile/emit_text/matmul[16x16x16]` | `tensor-wasm-bench` | `benches/jit_compile.rs` | `jit_compile/emit_text` | `matmul[16x16x16]` |
| `jit_compile/cache/warm_hit` | `tensor-wasm-bench` | `benches/jit_compile.rs` | `jit_compile/cache` | `warm_hit` |
| `tenant_registry/lookup/16` | `tensor-wasm-tenant` | `benches/context_switch.rs` | `tenant_registry/lookup` | `16` |

## Re-baselining

Run a clean local pass and save it as a named baseline:

```sh
# Per-bench (recommended — easier to spot per-file regressions):
cargo bench -p tensor-wasm-bench --bench cold_start       -- --save-baseline new
cargo bench -p tensor-wasm-bench --bench kernel_dispatch  -- --save-baseline new
cargo bench -p tensor-wasm-bench --bench memory_bandwidth -- --save-baseline new
cargo bench -p tensor-wasm-bench --bench jit_compile      -- --save-baseline new
cargo bench -p tensor-wasm-bench --bench e2e_inference    -- --save-baseline new
cargo bench -p tensor-wasm-tenant --bench context_switch  -- --save-baseline new

# Or whole-workspace if you trust the host:
cargo bench --workspace -- --save-baseline new
```

Then inspect `target/criterion/<group>/<id>/new/estimates.json` (or the
HTML at `target/criterion/<group>/<id>/report/index.html`) and copy each
median into `baseline.json`. Update `baseline-notes.md` with any
group/id renames or coverage changes. Submit the re-baseline as its own
commit, separate from any behavioral change, so future bisects can
attribute regressions cleanly.

## How CI consumes this

The `bench` workflow at [`.github/workflows/bench.yml`](../.github/workflows/bench.yml)
parses Criterion's `--output-format bencher` lines, joins them against
`baseline.json` by `<group>/<id>`, and fails the build when
`measured_median > baseline.median_ns * (1 + (tolerance_pct + regress_pct_threshold) / 100)`.
The published policy is in [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).
