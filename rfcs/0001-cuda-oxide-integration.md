<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# RFC 0001: cuda-oxide as the v0.5 cust successor

- **Author(s):** Maintainers
- **Status:** Draft
- **Created:** 2026-05-25
- **Discussion PR:** TBD
- **Related:**
  - [`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md) — Open Decision #1
    (`cust` successor) and Open Decision #8 (toolchain pin cadence).
  - [`docs/RISKS.md`](../docs/RISKS.md) — "CUDA `cust` 0.3.x EOL" row.
  - [`docs/CUDARC-SPIKE.md`](../docs/CUDARC-SPIKE.md) — W1.2 spike
    findings, including the cudarc 0.13.9 symbol-path frictions.
  - [`docs/CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md) — author-facing
    surface for writing kernels today.
  - [`docs/CUDA-SETUP.md`](../docs/CUDA-SETUP.md) — toolkit, driver,
    and runner expectations (will gain a cuda-oxide section).
  - [`GOVERNANCE.md`](../GOVERNANCE.md) — RFC amendment procedure if
    this design changes after acceptance.

## Summary

Adopt **cuda-oxide** (NVIDIA Labs' new Rust-to-CUDA compiler + host
runtime, v0.1.0 alpha released 2026-05-09) as the v0.5 target for
[`PATH-TO-V1.md`](../docs/PATH-TO-V1.md) Open Decision #1 — the
`cust` successor question. Ship a parallel `cuda-oxide-backend`
feature flag at v0.3.1 alongside the existing W1.2
[`cudarc-backend`](../docs/CUDARC-SPIKE.md) spike, so all three
backends (`unified-memory`/cust default, `cudarc-backend`,
`cuda-oxide-backend`) coexist for at least one minor release. At v0.5,
promote `cuda-oxide-backend` to default **contingent on cuda-oxide
reaching v0.2.0 or later** with a stable host API; if cuda-oxide is
still on v0.1.x at the v0.5 freeze, fall back to `cudarc-backend` as
default and keep `cuda-oxide-backend` opt-in. Either way, `cust`
leaves the workspace at v0.5.

## Motivation

Three forces converge on this decision now, not later.

**`cust` is EOL upstream.** [`RISKS.md`](../docs/RISKS.md) has tracked
the "CUDA `cust` 0.3.x EOL" row since v0.1.0; no security or
compatibility patches are landing, and the W1.2 spike confirmed that
`cust 0.3.2` itself no longer compiles cleanly against
`nightly-2026-03-15` because of a removed `bytemuck::PodCastError`
reference. Staying on cust through another release cycle is a
security-patch hazard with no path back, not a conservative choice.

**The W1.2 cudarc spike landed, but cudarc has its own frictions.**
[`CUDARC-SPIKE.md`](../docs/CUDARC-SPIKE.md) documents six concrete
symbol-path mismatches in cudarc 0.13.9 (`cuMemAllocManaged`,
`cuMemPrefetchAsync`, `cuMemFree_v2`, `cuMemAdvise`,
`CUmem_advise_enum`, `CUresult::CUDA_SUCCESS`) plus the gap that
neither backend currently compiles on the workspace's pinned nightly.
The W5.9 build attempt sharpened these findings. cudarc is a fine
clean-room implementation — `candle`, `burn`, and `dfdx` all use it —
but it is not NVIDIA's direction.

**cuda-oxide is NVIDIA's direction.** NVIDIA Labs published cuda-oxide
v0.1.0 on 2026-05-09 (repo: `github.com/NVlabs/cuda-oxide`, docs:
`nvlabs.github.io/cuda-oxide`). It is a custom `rustc` codegen backend
that compiles standard Rust source directly to PTX through a
Pliron-based MLIR-style pipeline (Rust → `rustc_public` Stable MIR →
dialect-mir → mem2reg → dialect-llvm → LLVM IR → PTX), and ships its
own host-side crates (`cuda-host`, `cuda-core`, `cuda-device`,
`cuda-macros`, `cuda-async`). The cuda-oxide ecosystem doc explicitly
positions it as compatible with cudarc as a host runtime, and
distinguishes its scope from `rust-cuda` ("CUDA into Rust" vs "Rust to
NVIDIA GPUs"). For a project that wants to be on the NVIDIA-blessed
Rust-on-CUDA stack long-term, ignoring cuda-oxide means optimising
against a clean-room reimplementation while NVIDIA tunes their own.

The v0.5 release is the API-freeze checkpoint
([`PATH-TO-V1.md`](../docs/PATH-TO-V1.md) — "v0.5.0-beta — External
validation"). Whatever GPU backend ships at v0.5 is the one external
auditors review, design partners deploy against, and v1.0 inherits.
Open Decision #1 must resolve before that freeze, and the proposal
needs maintainer alignment in writing — that is this RFC.

## Detailed design

### The option matrix today

| Concern | Today (cust) | Option: cudarc | Option: cuda-oxide |
|---|---|---|---|
| Host runtime (alloc, stream, event, launch) | `cust` 0.3.x, EOL | `cudarc` 0.13.x, maintained, clean-room | `cuda-host` 0.1.x, NVIDIA-shipped, alpha |
| Kernel authoring | Hand-written PTX or out-of-tree `nvcc` | Same as today | Rust source via `#[cuda_module]` (cuda-oxide compiles to PTX) |
| Upstream owner | None | Single maintainer, active | NVIDIA Labs |
| Crates.io presence | Yes (frozen) | Yes (regular releases) | Partial — Pliron pinned via `git` rev, not on crates.io |
| Compatible with the W1.2 spike | N/A (status quo) | Spike landed | cuda-oxide ecosystem doc lists cudarc as a compatible host alternative; the two coexist |
| Toolchain pin | `nightly-2026-03-15` (workspace default) | Same | `nightly-2026-04-03` (cuda-oxide's own pin) |
| Workspace edition | Edition 2021 | Same | Edition 2024 (consumable from 2021 callers, but worth flagging) |

The two axes are independent: **host runtime** (who manages the
context, allocations, streams, events, launches) and **kernel
authoring** (how kernels get written and shipped as PTX). cuda-oxide
is the first option that has a credible answer for both axes from a
single project.

### Feature-flag layout

Three Cargo features in `tensor-wasm-mem`, mutually compatible:

- `unified-memory` (default for CUDA builds) — today's `cust` path.
  Stays the default for v0.3.x and v0.4.x.
- `cudarc-backend` — the W1.2 spike, already landed. Stays opt-in for
  v0.3.x and v0.4.x.
- `cuda-oxide-backend` — new in v0.3.1. Opt-in for v0.3.x and v0.4.x.
  Promoted to default at v0.5 contingent on cuda-oxide ≥ 0.2.0.

All three resolve to the same `Backing::Cuda` enum variant from the
caller's perspective; they differ only in which FFI shim sits behind
the variant. The intent is that downstream crates
(`tensor-wasm-wasi-gpu`, `tensor-wasm-tenant`, `tensor-wasm-jit`)
write to one abstraction (`crate::backend::CudaBackend` trait, to be
introduced in the v0.3.1 scaffold PR) and pick a concrete backend at
compile time.

### Toolchain plan

cuda-oxide pins `nightly-2026-04-03`. TensorWasm pins
`nightly-2026-03-15`
([`rust-toolchain.toml`](../rust-toolchain.toml)). This is a real
conflict: enabling `--features cuda-oxide-backend` on the workspace
default toolchain will not build.

The proposal:

1. **Workspace default stays on `nightly-2026-03-15` for v0.3.x.** No
   surprise bumps for contributors who are not touching the new
   feature.
2. **`cuda-oxide-backend` requires an opt-in toolchain override.**
   Documented in [`CUDA-SETUP.md`](../docs/CUDA-SETUP.md) as a
   `RUSTUP_TOOLCHAIN=nightly-2026-04-03 cargo build --features
   cuda-oxide-backend ...` invocation. CI grows one matrix entry
   exercising the override; the existing matrix is unchanged.
3. **At v0.4, the workspace default bumps** to a nightly that
   satisfies both cuda-oxide's pin and the W2.9 Wasmtime cadence
   policy. The quarterly cadence already committed in
   [`PATH-TO-V1.md`](../docs/PATH-TO-V1.md) Open Decision #8 lines up
   naturally — the v0.4 bump is one of the planned quarters.
4. **At v0.5, the toolchain bump and the default-backend flip happen
   in the same release.** If cuda-oxide v0.2 is not out by then, the
   default flips to `cudarc-backend` instead and `cuda-oxide-backend`
   stays opt-in for one more release.

### Pliron lever and the auto-offload pipeline

cuda-oxide is built on Pliron, an MLIR-like Rust-native IR framework.
This unlocks a second-order opportunity: a future
`tensor-wasm-jit::pliron_dialect` module can lower Wasm-derived IR to
cuda-oxide's `dialect-mir` directly, instead of generating PTX
text-templates from the three hand-written blueprints (matmul,
vector_add, conv2d 3×3 — see [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md)).

That would expand auto-offload coverage from the current blueprint set
to arbitrary pure-compute loops the detector can prove safe. It is
**not** in scope for this RFC — this RFC only proposes the host-side
adoption that makes the lever available later. See "Future
possibilities".

### Kernel-authoring surface

Today, kernels are hand-written PTX or out-of-tree `nvcc` artifacts
loaded via `cust::module::Module::from_ptx`. cudarc keeps the same
shape with `CudaDevice::load_ptx`. cuda-oxide adds a third option:
Rust source compiled in-tree by the `#[cuda_module]` macro from
`cuda-macros`, producing PTX as a build artifact.

The proposal does **not** make `#[cuda_module]` the default kernel
authoring surface. [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md) gains
a new "Path C: Rust kernels via cuda-oxide" section alongside the
existing PTX and nvcc paths; the three coexist, the operator chooses
per kernel. The author-side migration is opt-in and unrelated to the
host-side backend flip.

### Rollout (PR sequencing)

1. **v0.3.1 (this RFC + scaffold).** Land this RFC. Land
   `cuda-oxide-backend` as an empty feature flag with a stub module
   under `crates/tensor-wasm-mem/src/cuda_oxide_backend.rs` that
   compiles with the cuda-oxide nightly and exposes a
   `CudaOxideBackend` type implementing the same `CudaBackend` trait
   as the cust + cudarc paths. No call sites use it; the smoke test
   confirms it compiles. CI gains the one toolchain-override matrix
   entry. [`CUDA-SETUP.md`](../docs/CUDA-SETUP.md) and
   [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md) gain the new sections.
2. **v0.4 (parity).** Port the unified-memory, advise, prefetch, and
   stream/event/launch operations to `CudaOxideBackend` so all three
   backends pass the same conformance test suite under the S22 runner.
   Promote the W2.7 Helm chart to expose a `values.yaml` toggle for
   the backend choice (see Unresolved questions).
3. **v0.5 (default flip).** Promote `cuda-oxide-backend` to default
   contingent on cuda-oxide ≥ 0.2.0 with a stable host API. Drop
   `cust` from the workspace and remove the `unified-memory` feature
   alias. Update [`RISKS.md`](../docs/RISKS.md) to mark the cust EOL
   row **Resolved**. `cudarc-backend` stays available as a supported
   alternative.

### Test plan

- The W1.2 cudarc smoke test
  (`crates/tensor-wasm-mem/tests/cudarc_smoke.rs`) gets a cuda-oxide
  twin: `tests/cuda_oxide_smoke.rs`, same shape, gated behind
  `--features cuda-oxide-backend`.
- The W1.3 cross-version snapshot compat suite already validates that
  the on-disk format is backend-independent; no new tests needed
  there, but the suite runs under all three backends in CI once the
  scaffold lands.
- The W4.6 P99.9 latency bench extension grows a backend axis so
  performance regressions per backend are visible.
- The W4.5 "Writing CUDA kernels" guide gains an executable example
  under `examples/` for the `#[cuda_module]` path.

### Compatibility implications

No public API change at v0.3.1 (scaffold only). The
`Backing::Cuda` enum variant is unchanged; the WIT surface
(`wit/wasi-cuda.wit`) is unchanged; the HTTP API is unchanged. At v0.5
the default backend flip is a **build-time** change, not a runtime
one — guests see no behavioural difference. Operators who explicitly
opt out of the new default (by setting `--features cudarc-backend`
without `cuda-oxide-backend`) get the cudarc path; nothing forces the
flip on a downstream that has reasons to stay.

## Drawbacks

- **cuda-oxide is v0.1.0 alpha.** The project explicitly warns of API
  breakage. Anything we build against the v0.1 surface may need
  rework at v0.2. The contingent-default approach in the Summary is
  the mitigation, but it does not erase the maintenance cost of
  chasing alpha churn through v0.3.x and v0.4.x.
- **Pliron is not on crates.io.** cuda-oxide's `Cargo.toml` pins
  Pliron to a `git` revision. Dependabot is blind to git pins
  (see `.github/dependabot.yml`), `cargo-deny` flags them, and
  reproducible-builds work (W3.6) needs an explicit allowlist entry.
  Until Pliron publishes, the supply-chain story is weaker than for
  cudarc.
- **Toolchain split.** Two pinned nightlies in the workspace — one
  for the default build, one for `cuda-oxide-backend` — is a
  contributor-onboarding tax. New contributors hit it the first time
  they enable the feature and the build fails with a cryptic
  "nightly-2026-03-15 does not have feature X" error. Mitigated by
  the CUDA-SETUP.md section but not eliminated.
- **Edition 2024 vs Edition 2021.** cuda-oxide's workspace is Edition
  2024; TensorWasm is Edition 2021. Consuming an Edition 2024 crate
  from an Edition 2021 caller is fine in practice, but the review
  cost of "did this PR accidentally rely on an Edition 2024 feature"
  is non-zero until the workspace migrates.
- **One more axis in the support matrix.** Three backends, two
  toolchains, the existing platform tier matrix — the cross-product
  of things CI can fail on grows. The W5.7 macOS compile-test work
  did not anticipate a backend that requires a different nightly; the
  matrix may need to skip cuda-oxide on macOS until upstream decides
  whether that platform is supported at all.

## Rationale and alternatives

The four options [`PATH-TO-V1.md`](../docs/PATH-TO-V1.md) Open
Decision #1 enumerates: cudarc, bespoke FFI, rust-cuda fork, and the
implicit fourth option of staying on cust + vendoring it. cuda-oxide
is a fifth option that did not exist when PATH-TO-V1 was written.

### Option A: cudarc-only

**What it is.** Promote the W1.2 spike to default at v0.4 or v0.5,
remove cust, do not add cuda-oxide. The path
[`CUDARC-SPIKE.md`](../docs/CUDARC-SPIKE.md) recommends.

**Why rejected.** Stable and clean, but stops at "we are on a
clean-room reimplementation of NVIDIA's Driver API." Long-term, NVIDIA
will optimise their own stack and a clean-room consumer follows.
Forecloses the cuda-oxide kernel-authoring lever — `#[cuda_module]`
and the Pliron-based auto-offload future would have to be re-added
later under a third backend anyway.

**What would change the calculus.** cuda-oxide stalling at v0.1.x
through 2026, NVIDIA discontinuing the project, or Pliron failing to
publish to crates.io. If any of those happen, the contingent fallback
in this RFC's Summary lands us here.

### Option B: cuda-oxide-only (skip cudarc)

**What it is.** Drop the W1.2 cudarc spike, go straight from cust to
cuda-oxide at v0.4 or v0.5.

**Why rejected.** cuda-oxide is alpha; betting the only GPU backend
on a project that may API-break for months is the wrong risk profile
for the v0.5 freeze. The W1.2 spike already exists and works around
the cudarc 0.13.9 frictions documented in
[`CUDARC-SPIKE.md`](../docs/CUDARC-SPIKE.md); throwing it away to
reduce option count is a false economy. cudarc as a fallback is what
makes the contingent v0.5 default flip safe.

**What would change the calculus.** cuda-oxide shipping a v1.0 stable
host API before v0.5 freeze. In that world, the cudarc backend
becomes vestigial and a future RFC retires it.

### Option C: both side-by-side, decide default at v0.5 (the proposal)

**What it is.** The Summary and Detailed design above.

**Why proposed.** Maximum optionality at the cost of one extra
feature flag and one CI matrix entry. The cost of carrying an extra
backend through two releases is bounded and concrete; the cost of
guessing wrong at v0.5 freeze is unbounded.

**What would change the calculus.** Maintainer review surfacing a
cost we have not anticipated (e.g., the toolchain-split tax turning
out to break a CI invariant we depend on). The "Unresolved questions"
section lists the specific items that could change this.

### Option D: vendor cust 0.3.x

**What it is.** Fork cust into the workspace and patch it ourselves.
Maximum control over the FFI surface, maximum maintenance burden.

**Why rejected.** [`PATH-TO-V1.md`](../docs/PATH-TO-V1.md) Open
Decision #1 explicitly listed and rejected this. Re-rejecting here
for completeness: a 2-4 person maintainer team should not run a
hand-vendored CUDA Driver API binding when two upstream alternatives
exist. The W1.2 spike removed the "but what if neither alternative
works" objection.

**What would change the calculus.** cudarc *and* cuda-oxide both
becoming unmaintained. Not a scenario worth designing for today;
documented for the record.

### Option E: do nothing (stay on cust)

**What it is.** Treat the cust EOL row in
[`RISKS.md`](../docs/RISKS.md) as an accepted risk; ship v1.0 on
cust 0.3.x.

**Why rejected.** The W1.2 spike confirmed cust no longer compiles on
the workspace nightly. "Do nothing" is no longer a free option — it
is a commitment to either keep the workspace on an older nightly
indefinitely or vendor cust ourselves (Option D).

## Unresolved questions

- **Will cuda-oxide v0.2 ship before the v0.5 freeze?** Proposed
  answer: assume yes for planning, design the fallback to
  `cudarc-backend` for if no. Open until cuda-oxide publishes a v0.2
  release schedule, or until the v0.4 release cycle, whichever comes
  first.
- **How does Pliron pin to a stable release vs git?** Proposed
  answer: track Pliron upstream and re-evaluate once it publishes
  to crates.io; if it has not by v0.5, document the git pin in
  [`docs/REPRODUCIBLE-BUILDS.md`](../docs/REPRODUCIBLE-BUILDS.md) and
  add an explicit `cargo-deny` allowlist entry. Open until Pliron
  publishes.
- **Does cuda-async's Tokio integration outperform our hand-rolled
  `DispatchFuture` busy-poll?** Proposed answer: unknown; benchmark
  during the v0.4 parity work. The W4.6 P99.9 bench extension is the
  right harness for this measurement. Open until that benchmark
  lands.
- **Should the W2.7 Helm chart and the W5.6 Nomad manifests track
  the `cuda-oxide-backend` feature with a `values.yaml` toggle?**
  Proposed answer: yes, but the toggle landing waits for the v0.4
  parity work — there is no point exposing a knob that selects a
  scaffold-only backend. Open until v0.4 PR sequencing.
- **Does the `#[cuda_module]` author-side path need its own RFC?**
  Proposed answer: no for v0.5 (it is opt-in and per-kernel); yes for
  v1.0 if it becomes the documented default in
  [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md). Open until v0.5
  feedback.
- **What is the deprecation timeline for the `unified-memory`
  feature alias?** Proposed answer: deprecation warning at v0.4,
  removal at v0.5 with the cust drop. Open until v0.4 release-notes
  drafting.

## Prior art

- **`rust-cuda`** (`rust-gpu/rust-cuda`). Older project, predates
  Pliron and `rustc_public`. The cuda-oxide ecosystem doc explicitly
  contrasts its scope ("Rust to NVIDIA GPUs") with cuda-oxide's
  ("CUDA into Rust"). Useful precedent for "Rust source → PTX is
  possible"; not the same design. We take the validation that the
  approach works; we leave the older codegen architecture on the
  table.
- **CubeCL** (`tracel-ai/cubecl`). Cross-vendor GPU DSL embedded in
  Rust, not a `rustc` backend. Generates CUDA, ROCm, WGPU from the
  same source. Out of scope for v1.0 because v1.0 is NVIDIA-only
  (anti-goal in [`PATH-TO-V1.md`](../docs/PATH-TO-V1.md)), but the
  cross-vendor abstraction shape is informative for any v2 ROCm /
  Metal work.
- **`rust-gpu`** (`EmbarkStudios/rust-gpu`). Rust to SPIR-V for
  graphics shaders. Different target, different problem domain.
  Informative for the "custom rustc codegen backend" engineering
  pattern only.
- **Wasmer WASIX GPU experiments.** Different runtime, different
  WASI proposal lineage. Mentioned for completeness; nothing portable
  back into TensorWasm's WIT-based surface.
- **The W1.2 cudarc spike itself.** The closest prior art is in this
  repository. [`CUDARC-SPIKE.md`](../docs/CUDARC-SPIKE.md) is the
  template for how this RFC's cuda-oxide scaffold should land —
  parallel implementation behind a feature flag, smoke test gated on
  hardware, no impact on the default build until cutover.

## Future possibilities

- **Drop `cust` entirely at v0.5.** Already in scope; mentioned here
  to flag the dependency-graph cleanup that follows.
- **Pliron-based auto-offload pipeline** (Wasm → Cranelift IR →
  Pliron `dialect-mir` → cuda-oxide → PTX) as a v0.6+ research goal.
  Would expand auto-offload coverage beyond the three blueprints in
  [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md) to arbitrary
  pure-compute loops. Needs its own RFC if it materialises.
- **`cuda-async`-backed `DispatchFuture` replacement.** Removes the
  busy-poll in the back-pressure / future-sync path. Subject to the
  benchmark in Unresolved questions; if cuda-async wins, a follow-up
  RFC retires the hand-rolled future.
- **`#[cuda_module]` as the documented default kernel-authoring
  surface.** If the v0.5 design-partner feedback is that Rust kernels
  are strictly easier to maintain than hand-written PTX, a v1.0 or
  v1.1 RFC could flip the [`CUDA-KERNELS.md`](../docs/CUDA-KERNELS.md)
  recommendation.
- **Retire `cudarc-backend`.** If cuda-oxide reaches stable v1.0 and
  the cudarc backend stops earning its keep, a future RFC removes
  it. Not on the v1.0 path.
- **Cross-vendor abstraction via Pliron dialects.** Pliron's
  MLIR-style multi-dialect design is what would make a future ROCm /
  Metal backend tractable inside the same compiler pipeline. Out of
  scope for v1.0 (anti-goal in
  [`PATH-TO-V1.md`](../docs/PATH-TO-V1.md)), but noted so the v0.5
  feature-flag layout does not foreclose it.
