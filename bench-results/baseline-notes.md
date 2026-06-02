# baseline-notes.md

Append-only log of bench-id changes that the CI regression-gate parser
(in `.github/workflows/bench.yml`) must reconcile against `baseline.json`.
Every rename, deletion, or coverage change goes here so future bisects
can match a commit to a baseline-key change.

Format: one section per dated change, with `Affected baseline key(s)` so
the parser patch is mechanical.

## 2026-05-24 — `memory_bandwidth/random_stride` → `memory_bandwidth/strided`

The `bench_random_strided_copy` function in
`crates/tensor-wasm-bench/benches/memory_bandwidth.rs` is a fixed-stride
sequential walk, not a random walk — there is no permutation, just a
constant `STRIDE` increment. To keep the metric name honest the group
was renamed from `memory_bandwidth/random_stride` to
`memory_bandwidth/strided`.

- **Affected baseline key(s):** none today. The current `baseline.json`
  did not include any `memory_bandwidth/*` entry, so no edit is needed.
  If a future re-baseline adds one, the key should be
  `memory_bandwidth/strided/<size>` (e.g. `memory_bandwidth/strided/1048576`).
- **Parser action:** none required for the existing baseline. The note
  is here in case anyone bisects across this commit and sees a
  Criterion HTML report under `target/criterion/memory_bandwidth/strided/`
  while expecting the old name.
- **Follow-up:** a future PR may add a separate
  `memory_bandwidth/random_indexed` group that uses
  `StdRng::seed_from_u64(0xBA11)` to pre-build a permutation, giving us
  a true random-access metric alongside the honest strided one.

## 2026-05-24 — `PinnedHostBuffer` → `GuardedHostBuffer` (no key change)

Batch D renamed `tensor_wasm_mem::PinnedHostBuffer` to `GuardedHostBuffer`
(deprecated alias still exported). The
`crates/tensor-wasm-bench/benches/memory_bandwidth.rs` import was updated to
the new name. No baseline keys are affected — the rename is at the Rust
type level only.
