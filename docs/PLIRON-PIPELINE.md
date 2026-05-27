<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Pliron Pipeline — The Four-Wave Auto-Offload Plan

_Status: **wave 1 in flight** at the time of writing. The upstream decision
is recorded in [RFC 0001 — cuda-oxide as the v0.5 cust successor](../rfcs/0001-cuda-oxide-integration.md);
specifically the "Pliron lever and the auto-offload pipeline" section and
the "Future possibilities" entry on a Pliron-based auto-offload pipeline.
This document is the implementation companion: it captures the four-wave
plan that takes [`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md) from the v0.1.0
blueprint-only pipeline to arbitrary pure-compute auto-offload via
cuda-oxide._

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Why the interim LoweredOp IR](#2-why-the-interim-loweredop-ir)
3. [The four waves](#3-the-four-waves)
4. [Mapping table reference](#4-mapping-table-reference)
5. [Unsupported (deferred or hard-rejected)](#5-unsupported-deferred-or-hard-rejected)
6. [Status notes](#6-status-notes)
7. [Cross-references](#7-cross-references)

---

## 1. Pipeline overview

The end-to-end pipeline TensorWasm is moving toward — from a guest Wasm
module to a launched PTX kernel — is the following chain. The first three
stages exist today; the rest land across waves 1 through 4.

```
   +-------------+    +----------------+    +-----------+
   |  Wasm bytes |--->|  wasmparser    |--->|  BlockIR  |   (exists today)
   +-------------+    +----------------+    +-----------+
                                                  |
                                                  v
                                          +-----------------+
                                          |  Cranelift IR   |   (exists today)
                                          +-----------------+
                                                  |
                                                  v
                                       +---------------------+
                                       |  LoweredOp (wave 1) |   (in flight)
                                       +---------------------+
                                                  |
                                                  v
                                       +------------------------+
                                       |  pliron::Operation     |   (wave 3)
                                       +------------------------+
                                                  |
                                                  v
                                       +------------------------+
                                       |  mem2reg               |   (wave 3, cuda-oxide)
                                       +------------------------+
                                                  |
                                                  v
                                       +------------------------+
                                       |  dialect-llvm          |   (wave 3, cuda-oxide)
                                       +------------------------+
                                                  |
                                                  v
                                       +------------------------+
                                       |  LLVM IR -> PTX        |   (wave 3, cuda-oxide)
                                       +------------------------+
                                                  |
                                                  v
                                       +------------------------+
                                       |  cuda-oxide runtime    |   (wave 4 backend trait)
                                       +------------------------+
```

The boxes above the dashed line (Wasm → wasmparser → BlockIR → Cranelift
IR) are the existing v0.1.0 frontend used by
[`tensor_wasm_jit::detector::classify`](../crates/tensor-wasm-jit/src/detector.rs).
Everything from **LoweredOp** downward is new pipeline surface tracked
under the wave plan in Section 3.

## 2. Why the interim LoweredOp IR

A literal reading of RFC 0001 would have the Cranelift IR lower straight
into Pliron's `Operation` type. We are not doing that yet, for two reasons.

**(a) Pliron is git-pinned alpha and the API is moving.** RFC 0001's
"Drawbacks" section calls this out: Pliron is pinned to a `git` revision
in cuda-oxide's `Cargo.toml` and is not yet on crates.io. cuda-oxide
itself is v0.1.0 alpha and explicitly warns of API breakage; the contingent
v0.5 default flip in RFC 0001 is gated on cuda-oxide reaching ≥ 0.2.0
with a stable host API. Importing Pliron and `cuda-oxide` directly into
the lowering passes today would mean every upstream Pliron rev forces a
full re-port of those passes — and we would be doing that work before any
caller can actually run the produced kernels. The interim
[`LoweredOp`](../crates/tensor-wasm-jit/src/lowered_ir.rs) IR is a
pure-Rust shield: the families lower into `LoweredOp`, and only the
**LoweredOp → pliron::Operation** converter (wave 3) takes the upstream
churn hit.

**(b) Per-family lowering passes are independently unit-testable without
a GPU toolchain.** The cuda-oxide pipeline requires the
`nightly-2026-04-03` toolchain override, an NVIDIA driver, and a host
with a supported GPU SKU even to round-trip a single op end-to-end. By
making the wave-1 families produce `LoweredOp` instead of
`pliron::Operation`, every lowering pass (arith, float, memory, control
flow, vector, conversion) has a unit-test surface that runs on a stock
contributor laptop in milliseconds, with zero CUDA setup. The wave-3
converter is a single component to integration-test against the real
Pliron dep; the families themselves stay testable forever.

The cost of the interim IR is one extra translation step
(`LoweredOp → pliron::Operation`) and a small amount of duplicated type
machinery (`LoweredType`, `LoweredSignature`). That cost is bounded; the
cost of chasing alpha API churn through six lowering passes simultaneously
is not.

## 3. The four waves

### Wave 1 — pure-Rust LoweredOp scaffold (in flight)

The current wave. Nine parallel tasks land the interim IR and the six
lowering families without taking any cuda-oxide or Pliron dependency.

| Task | Scope |
|---|---|
| **F1** | `LoweredOp` enum, `LoweredFunction`, `LoweredBlock`, `LoweredType`, `LoweredSignature` — the IR scaffold. Lives in [`crates/tensor-wasm-jit/src/lowered_ir.rs`](../crates/tensor-wasm-jit/src/lowered_ir.rs). |
| **F2** | Trait surface upgrade on the [`pliron_dialect`](../crates/tensor-wasm-jit/src/pliron_dialect.rs) module so the v0.4 entry point takes `LoweredFunction` instead of `&str` placeholder. |
| **L1** | Arithmetic family lowering (`iadd`, `isub`, `imul`, `idiv`/`udiv`, integer compare, `select`). |
| **L2** | Float family lowering (`fadd`, `fsub`, `fmul`, `fdiv`, `fma`, float compare). |
| **L3** | Memory family lowering (`load`, `store`, `stack_load`, `stack_store`). |
| **L4** | Control-flow family lowering (`jump`, `brif`/`brz`/`brnz`, `br_table`, block successors, SSA phi/block-param shape). |
| **L5** | Vector family lowering (`vmin`, `vmax`, `vsplat`, `vselect`, `vall_true`/`vany_true`, FMA over v128). |
| **L6** | Conversion family lowering (`bitcast`, `breduce`, `bextend`/`sextend`, sign-vs-zero choice). |
| **T1** | Test-fixture builders: hand-rolled `LoweredFunction` constructors and golden-file inputs for each family's unit tests. |
| **D1** | This document. |

All nine tasks are docs- or pure-Rust-only and ride the workspace default
`nightly-2026-03-15` toolchain. No new external dependencies land in
wave 1.

### Wave 2 — wire the families together

Wave 2 takes the six wave-1 families from "compiles + unit-tested" to
"produces a `LoweredFunction` end-to-end from a real Wasm guest." The
work is sequential because every step depends on the wave-1 IR being
stable.

- **Reject-list detector pass.** Augment
  [`detector::classify`](../crates/tensor-wasm-jit/src/detector.rs) to
  reject candidates that contain atomics, `table.get`/`table.set`,
  `ref.func`/GC ops, `memory.grow`, or large `memory.copy`/`memory.fill`
  before any lowering runs. The canonical list is in the
  [`pliron_dialect` rustdoc "Unsupported"](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
  section; the detector enforces it.
- **Function signature lowering.** Translate Cranelift `Signature` →
  `LoweredSignature`, including kernel-args base-pointer threading per
  the W1.1 typed-argv wire format (see
  [`CUDA-KERNELS.md`](CUDA-KERNELS.md) Section 6).
- **SSA value tracking across blocks.** Lift Cranelift's per-block SSA
  numbering into module-level `LoweredOp` operands so the wave-3
  converter sees a coherent value graph.
- **Module-level lowering driver.** The free function that ties the
  six families together and produces a `LoweredFunction` from a
  `cranelift_codegen::ir::Function`. Lives next to the family modules.
- **Integration with `detector.rs`.** Wire the driver into the existing
  detector → lowering call site behind the `auto-offload` feature; the
  output is still consumed only by tests in wave 2.
- **Snapshot compat.** The W1.3 cross-version snapshot suite gains a
  `LoweredFunction` axis so the on-disk format stays backend-independent
  (cf. [`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md)).
- **Fuzz harness.** A `cargo-fuzz` target driving arbitrary
  Cranelift-IR-shaped inputs through the families, asserting the driver
  never panics. Catches malformed inputs the detector should have
  rejected.
- **End-to-end smoke test.** A wasm guest → wasmparser → Cranelift →
  `LoweredFunction` round-trip with a fixture comparison, running on
  the default toolchain with no CUDA toolkit installed.

### Wave 3 — add the real Pliron dep and emit PTX

Wave 3 introduces the first git-pinned upstream dependency in this
pipeline. It is gated behind the opt-in `cuda-oxide-backend` feature
flag introduced in v0.3.1 (RFC 0001 "Rollout").

- **Add Pliron + cuda-oxide as deps behind `cuda-oxide-backend`.**
  Cargo features stay mutually compatible per RFC 0001 "Feature-flag
  layout"; the default workspace build is unaffected.
- **`LoweredOp → pliron::Operation` converter.** A single module under
  `crates/tensor-wasm-jit/src/` that consumes a `LoweredFunction` and
  emits a Pliron `Module` containing `dialect-mir` ops. This is the
  one place upstream API churn lands.
- **Wire `mem2reg` + `dialect-llvm` + PTX emission via cuda-oxide.**
  Use the cuda-oxide compiler pipeline (`dialect-mir → mem2reg →
  dialect-llvm → LLVM IR → PTX`) to produce a PTX module suitable for
  loading via `cust::module::Module::from_ptx` (or its `cudarc` /
  `cuda-host` equivalents — the choice is the wave-4 backend trait's
  problem).
- **`cargo-deny` allowlist entry.** RFC 0001 "Drawbacks" calls this
  out: Pliron is `git`-pinned and `cargo-deny` flags such pins by
  default. The allowlist entry is documented in
  [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md).
- **Reproducible-build doc update.** Add the cuda-oxide-backend build
  recipe to [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md), pinned
  to the cuda-oxide nightly per
  [`CUDA-SETUP.md`](CUDA-SETUP.md).
- **CI matrix entry.** One additional matrix row exercising
  `RUSTUP_TOOLCHAIN=nightly-2026-04-03 cargo build --features
  cuda-oxide-backend` against the smoke test from wave 2.

### Wave 4 — backend trait impl and runtime integration

Wave 4 makes the PTX produced by wave 3 actually launchable through
TensorWasm's runtime, and adds the operator surface.

- **`CudaOxideBackend` trait impl in `tensor-wasm-mem`.** Implements
  the same `CudaBackend` trait as the `unified-memory`/cust and
  `cudarc-backend` paths (RFC 0001 "Feature-flag layout"). The wave-3
  PTX flows through this backend to a real device launch.
- **Conformance suite.** The W1.2 cudarc smoke test gets its wave-3
  twin per RFC 0001 "Test plan"; the auto-offload conformance suite
  runs the same fixture set across all three backends and asserts
  bitwise-identical results within the documented float tolerance.
- **Helm chart toggle.** [`deploy/helm/tensor-wasm/`](../deploy/helm/tensor-wasm/README.md)
  gains a `values.yaml` knob for the backend choice — `unified-memory`
  (default), `cudarc-backend`, or `cuda-oxide-backend` — so operators
  can flip backends without rebuilding from source. RFC 0001
  "Unresolved questions" puts this on the v0.4 parity work; wave 4 is
  where it lands in practice.

## 4. Mapping table reference

The canonical Cranelift → `dialect-mir` mapping table lives in the
[`pliron_dialect` rustdoc](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
"Mapping table" section. That table is the single source of truth and
the v0.4-port author should walk it top-to-bottom. The summary below is
for orientation only; **always consult the rustdoc for the load-bearing
detail** (FMA-rounding contract, device-pointer translation, host-call
prohibition, sign-vs-zero extend choice, etc.).

| Family | Cranelift ops | dialect-mir family |
|---|---|---|
| Integer arith | `iadd`, `isub`, `imul`, `idiv`/`udiv` | `arith.addi`/`subi`/`muli`/`divsi`/`divui` |
| Float arith | `fadd`, `fsub`, `fmul`, `fdiv`, `fma` | `arith.addf`/`subf`/`mulf`/`divf`/`fma` |
| Memory | `load`, `store`, `stack_load`, `stack_store` | `memref.load`/`store` (with device-pointer translation) |
| Control flow | `jump`, `brif`/`brz`/`brnz`, `br_table` | `cf.br`/`cond_br`/`switch` |
| Vector | `vmin`, `vmax`, `vsplat`, `vselect`, `vall_true`/`vany_true` | `vector.minimum`/`maximum`/`splat`/`select`/`reduce_and`/`reduce_or` |
| Conversion | `bitcast`, `breduce`, `bextend`/`sextend` | `arith.bitcast`/`trunci`/`extui`/`extsi` |
| Select | `select` | `arith.select` |
| Calls | `call`, `call_indirect` | `func.call`/`call_indirect` (device-only) |

## 5. Unsupported (deferred or hard-rejected)

The wave-2 detector pass rejects candidates that contain any of the
following before reaching the wave-1 families. The
[`pliron_dialect` rustdoc "Unsupported in v0.4"](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
section is the canonical list with per-op rationale; this document only
names the categories so a reader can decide whether to dig into the
rustdoc.

- **Atomics.** Wasm threads + GPU atomics is a memory-model alignment
  problem larger than the wave-3 scope. Deferred to a future RFC.
- **Strict-FP exception bits.** PTX default rounding + flush-to-zero
  diverges from Wasm-strict FP; candidates that depend on strict-FP
  guarantees are rejected.
- **`table.get` / `table.set`.** Tables live host-side; device-resident
  table mirrors are out of scope.
- **`ref.func` and GC ops.** No device-side representation;
  hard-rejected.
- **`memory.grow` / `memory.size`.** Linear-memory resizing requires a
  host round-trip; kernels run with a fixed memory snapshot.
- **Large `memory.copy` / `memory.fill`.** Small copies inline; copies
  larger than 4 KiB fall back to a host bounce (PTX `cp.async.bulk` is
  sm_90+ and the v0.4 baseline is sm_80).

This list is intentionally not duplicated here — when categories evolve,
the rustdoc moves and this document points at the new location.

## 6. Status notes

- **Wave 1 is in flight.** Nine parallel tasks (F1, F2, L1–L6, T1, D1)
  land in the same milestone. F1 is the interim IR scaffold; this
  document is D1. None of the wave-1 tasks introduce an external
  dependency.
- **Upstream decision recorded in RFC 0001.** The choice to ride
  cuda-oxide rather than stay on `cust` or jump straight to a hand-rolled
  PTX emitter is documented in
  [RFC 0001](../rfcs/0001-cuda-oxide-integration.md) — see Summary,
  "Pliron lever and the auto-offload pipeline", and "Future
  possibilities". This document is the implementation companion to that
  RFC; it does not re-litigate the decision.
- **Pliron is git-pinned, not yet on crates.io.** RFC 0001 "Drawbacks"
  flags this as a supply-chain hazard for the
  [reproducible-builds](REPRODUCIBLE-BUILDS.md) and `cargo-deny` work.
  The interim `LoweredOp` IR exists in part so that hazard does not land
  in the workspace until wave 3, when `cuda-oxide-backend` is the
  feature flag carrying the cost.
- **Toolchain split is real and intentional.** The workspace default
  stays on `nightly-2026-03-15`; wave 3 adds an opt-in
  `nightly-2026-04-03` override for the `cuda-oxide-backend` feature.
  Wave 1 and wave 2 do not touch the toolchain.
- **v0.5 default flip is contingent.** If cuda-oxide ≥ 0.2.0 has not
  shipped by the v0.5 freeze, the default backend flips to
  `cudarc-backend` instead and `cuda-oxide-backend` stays opt-in for
  one more release. The wave plan in this document is unchanged by
  that contingency — the families and the converter still ship; only
  the wave-4 default-backend selection moves.

## 7. Cross-references

- [RFC 0001 — cuda-oxide as the v0.5 cust successor](../rfcs/0001-cuda-oxide-integration.md):
  the upstream decision, the option matrix, the rollout sequencing,
  and the "Future possibilities" entry that this document implements.
- [AUTO-OFFLOAD.md](AUTO-OFFLOAD.md): user-facing reference for which
  Wasm patterns the v0.1.0 detector recognises today. The wave-4
  conformance suite extends that surface to arbitrary pure-compute
  loops.
- [CUDA-KERNELS.md](CUDA-KERNELS.md): the three kernel-authoring paths
  (hand-PTX, nvcc, cuda-oxide `#[cuda_module]`). The wave-3 PTX output
  from this pipeline is loaded via the same machinery.
- [CUDA-SETUP.md](CUDA-SETUP.md): toolchain and driver expectations,
  including the wave-3 `nightly-2026-04-03` opt-in override.
- [REPRODUCIBLE-BUILDS.md](REPRODUCIBLE-BUILDS.md): supply-chain story
  for the wave-3 Pliron `git` pin and the `cargo-deny` allowlist entry.
- [SNAPSHOT-COMPATIBILITY.md](SNAPSHOT-COMPATIBILITY.md): the wave-2
  `LoweredFunction` axis added to the cross-version snapshot suite.
- Wave-1 source files:
  - [`crates/tensor-wasm-jit/src/lowered_ir.rs`](../crates/tensor-wasm-jit/src/lowered_ir.rs)
    — the interim `LoweredOp` IR (F1).
  - [`crates/tensor-wasm-jit/src/pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
    — the canonical Cranelift → `dialect-mir` mapping table and the
    final trait signature (F2).
  - [`crates/tensor-wasm-jit/src/detector.rs`](../crates/tensor-wasm-jit/src/detector.rs)
    — the existing detector the wave-2 reject-list extends.

---

_Status: wave 1 of the four-wave plan. Update the per-wave status line
as each wave lands. The next-wave checklist lives at the top of each
section above; if a wave needs more detail it grows into its own
sub-doc and this document keeps only the one-paragraph summary._
