<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Differential JIT correctness oracle

Roadmap feature #6 — the lowest-cost, highest-credibility security
item on the v0.5 external-audit pre-flight checklist. The oracle runs
every `auto_offload` candidate on both the Wasmtime CPU interpreter
(the ground truth) and the JIT PTX path, then asserts bit-identity
between the two outputs. Any divergence is an audit-blocking finding.

Scaffold lives in [`crates/tensor-wasm-jit/src/differential.rs`](../crates/tensor-wasm-jit/src/differential.rs)
behind the `differential-oracle` feature flag.
[`docs/PATH-TO-V1.md#post-v036-strategic-features`](PATH-TO-V1.md#post-v036-strategic-features)
tracks this as item #6.

## Motivation

The v0.5 external audit will probe two questions about the JIT pipeline:

1. **Soundness** — can the JIT-emitted PTX produce a different result
   than the original Wasm body, in any condition?
2. **Trust boundary** — what stops a guest from observing JIT-internal
   state (timing, error parity, etc.) through the difference between
   the two paths?

A differential oracle answers both. Soundness reduces to "did the two
paths agree?" and is a property the runtime can self-assert without
asking the auditor to read every PTX instruction. Trust boundary
reduces to "are the trap and error codes parity-equivalent?" which
the harness checks alongside output bytes. The oracle is the cheapest
pre-deploy gate that turns a class of "I'd need to read the lowering
code to convince myself" objections into "the CI gate already failed
on this in PR #X if it could have happened".

## v0.4 implementation plan

The v0.3.6 scaffold compiles harness types only. v0.4 wires the
two-path runner:

1. **CPU path** — re-use the existing `tensor-wasm-exec` Wasmtime
   instance from the unit-test harness. Run the original Wasm body
   (not the JIT-rewritten one) over a fixed guest-memory snapshot.
2. **GPU path** — drive the JIT pipeline through to the
   `KernelCache::get` + `cudarc` launch surface. Read back the
   contiguous output region after the launch finishes.
3. **Compare** — `memcmp` the two regions; produce
   `OracleVerdict::Match` or `OracleVerdict::Divergence` with the
   `first_diff_offset` of the first non-matching byte.
4. **CI gate** — a new job in [`.github/workflows/`](.github/workflows/)
   runs the oracle over the blueprint corpus on the self-hosted S22
   runner (see [`docs/CUDA-SETUP.md`](CUDA-SETUP.md)). Divergences
   block the PR; the workflow uploads the offending blueprint
   fingerprint as a build artifact for triage.

S22 runner integration is the only piece that requires hardware the
default CI lacks. Hosts without CUDA continue to receive
`OracleVerdict::Skipped("no-cuda; ...")` so the harness compiles
green on Tier-1 Linux x86_64, Windows MSVC, and macOS.

## Sample blueprint corpus

The oracle ingests blueprints from two sources:

- **Hand-curated** — copies of the `auto_offload_e2e.rs` and
  `auto_offload_pipeline.rs` fixtures. These cover matmul,
  vector_add, and conv2d — the v0.1.0 supported shapes.
- **Proptest-generated** — a `proptest` strategy that emits random
  `TensorWasmKernelBlueprint` instances inside the
  [`crates/tensor-wasm-jit/src/reject_list.rs`](../crates/tensor-wasm-jit/src/reject_list.rs)
  acceptance window. Corpus lives at
  `crates/tensor-wasm-jit/corpus/differential/` and is checked in
  alongside the seeds so a divergence is reproducible.

The blueprint fingerprint (BLAKE3 over the canonical byte form, per
`TensorWasmKernelBlueprint::fingerprint`) is the stable id the CI
gate uses to deduplicate divergence reports across runs.

## What counts as a divergence

The oracle compares three dimensions; any inequality is a divergence:

1. **Output bytes** — the contiguous output region in guest memory
   after both paths return. Compared byte-for-byte. Length mismatch
   is reported as a divergence with `first_diff_offset: None`.
2. **Side-effects** — none, by construction. The
   `reject_list` (see [`crates/tensor-wasm-jit/src/reject_list.rs`](../crates/tensor-wasm-jit/src/reject_list.rs))
   refuses to offload any blueprint with observable side-effects
   other than the output region (no host calls, no memory writes
   outside the declared output range, no traps mid-kernel). The
   oracle therefore asserts the absence of side-effects as a
   precondition; the v0.4 runner asserts no spurious writes outside
   the declared output range as a regression check on the reject
   list.
3. **Trap / error parity** — if one path traps (Wasm trap or CUDA
   launch failure) and the other does not, that is a divergence.
   If both trap, the trap reason codes must match. The harness maps
   CUDA launch failures into the closest Wasm trap code (out-of-
   bounds, integer overflow, unreachable) per the table in
   [`crates/tensor-wasm-jit/src/deopt.rs`](../crates/tensor-wasm-jit/src/deopt.rs).

## Tolerance table (v0.3.7)

Per-blueprint floating-point tolerance the oracle accepts between the
CPU reference and the JIT-emitted GPU path. Values mirror
[`ToleranceTable::default_table`](../crates/tensor-wasm-jit/src/differential.rs)
and are the contract the proptest harness in
[`crates/tensor-wasm-jit/tests/differential_proptest.rs`](../crates/tensor-wasm-jit/tests/differential_proptest.rs)
asserts.

| Blueprint    | dtype | ULPs | abs    | rel    | Rationale                                            |
| ------------ | ----- | ---- | ------ | ------ | ---------------------------------------------------- |
| vector_add   | f32   | 1    | 0      | 0      | Element-wise; round-to-nearest-even on both paths    |
| vector_mul   | f32   | 1    | 0      | 0      | Element-wise; same rounding contract                 |
| vector_fma   | f32   | 1    | 0      | 0      | `f32::mul_add` ↔ `fma.rn.f32`, single rounding       |
| matmul       | f32   | 2    | 1e-6   | 1e-6   | Reduction order is implementation-defined            |
| conv2d       | f32   | 2    | 1e-6   | 1e-6   | Sliding-window MAC; same shape as matmul             |
| (any)        | f16   | x4   | x4     | x4     | 10-bit mantissa; ULP budget scales by the multiplier |
| (any)        | int   | 0    | 0      | 0      | Bit-identical required                               |

The `ToleranceTable::strict()` constructor zeroes every entry — used
by the CI gate that asserts bit-identity for ops the oracle pins as
exact (today: integer kernels only). [`Tolerance::approx_eq_f32`](../crates/tensor-wasm-jit/src/differential.rs)
implements the standard `|g - c| <= max(abs, rel * |c|) OR ulp_dist
<= ulps` check; NaN ↔ NaN is treated as equal (mirrors `==` with the
`total_cmp` semantics PTX uses on reductions).

## Related docs

- [SECURITY-AUDIT.md](SECURITY-AUDIT.md) — internal pre-audit posture
- [AUTO-OFFLOAD.md](AUTO-OFFLOAD.md) — pipeline this oracle covers
- [RISKS.md](RISKS.md) — current "differential testing limited to
  unit tests" callout (item to close once v0.4 wires this end-to-end)
- [PATH-TO-V1.md](PATH-TO-V1.md#post-v036-strategic-features) — the
  roadmap section that frames this as item #6
