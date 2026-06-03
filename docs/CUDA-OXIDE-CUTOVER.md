<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# `cuda-oxide` v0.2 cutover — procedure runbook

Procedure runbook for the day NVlabs ships `cuda-oxide` v0.2.0 and the
maintainer flips `cuda-oxide-backend` from the v0.3.1 opt-in scaffold to
the v0.5 default. This is the contingent-yes path of
[RFC 0001](../rfcs/0001-cuda-oxide-integration.md) Option C
("Both side-by-side, decide default at v0.5"). The contingent-no path —
cuda-oxide still on v0.1.x at the v0.5 freeze — keeps `cudarc-backend`
as the v0.5 default and shelves this runbook; the W1.2
[`CUDARC-SPIKE.md`](CUDARC-SPIKE.md) recommendation is the fallback.

Procedure runbook, not an alert runbook; follow the
[`runbooks/README.md`](runbooks/README.md) "Procedure runbooks" contract
and mirror the voice of C1
[`self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md). It
is **executable**: every step names a file path, a command, or a
concrete edit. Status: **gated on cuda-oxide v0.2 release** — do not
start until the four preconditions below all go green.

## When to run this

All four must be true. Concrete checks, not judgement calls:

- [ ] **cuda-oxide v0.2.0 tag exists.** Verify with
  `git ls-remote --tags https://github.com/NVlabs/cuda-oxide | grep -E
  'refs/tags/v0\.2\.0$'`. Pre-release tags (`-alpha.1`, `-rc1`) do
  **not** count — RFC 0001 names "v0.2.0 or later with a stable host
  API"; `-alpha` / `-rc` is upstream saying the API is not stable.
- [ ] **The v0.1 → v0.2 CHANGELOG enumerates the wire/API breakage.**
  Read `https://github.com/NVlabs/cuda-oxide/blob/v0.2.0/CHANGELOG.md`
  and confirm it lists at least the rename / signature changes for
  `cuda_host::DeviceBuffer`, `cuda_host::CudaDevice::alloc_managed`,
  `cuda_host::Stream`, and the `cuda_async::register_callback` waker
  entry point. CHANGELOG silent on the host API the v0.4 port assumed?
  Hold and open a clarification issue upstream.
- [ ] **Pliron is on crates.io OR the v0.2 cuda-oxide pins a Pliron rev
  that has stopped moving.** "Stopped moving" = the Pliron commit
  cuda-oxide v0.2 pins matches Pliron `main` HEAD seven days before
  the cuda-oxide v0.2 tag. The check protects our F2
  [`deny.toml`](../deny.toml) allowlist stability for one release.
- [ ] **The S22 self-hosted CUDA runner from C1 is registered and green
  on `cuda.yml`.** Settings → Actions → Runners shows "Idle" for
  `self-hosted,cuda`, and the most recent `cuda` run on `dev` is
  green for all four jobs (`cust-unified-memory`, `wasi-gpu-cuda`,
  `cudarc-backend`, `cuda-oxide-backend`). Fix via
  [`runbooks/self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md)
  first if not.

Any box unchecked? **Stop here.** The cudarc fallback stays the v0.5
default and the runbook is shelved until the gate clears.

## Pre-flight

Capture the pre-cutover state so [Rollback](#rollback-procedure) has a
known-good reference.

### Read the v0.1 → v0.2 CHANGELOG end to end

Take notes in a scratch `notes-v0.2-cutover.md` (gitignored). Concrete
questions to answer:

- Did `cuda_host::DeviceBuffer<u8>` (the type our O2 scaffold's TODO
  names as the most-likely v0.4 inner field) rename? If so, update
  every occurrence in
  [`cuda_oxide_backend.rs`](../crates/tensor-wasm-mem/src/cuda_oxide_backend.rs)
  plus the [`Cargo.toml`](../Cargo.toml) workspace deps comment.
- Did the managed-allocation entry point keep its v0.1 name
  (`CudaDevice::alloc_managed`) or rename (e.g. `alloc_uvm`)?
- Did `cuda_async::register_callback` (the waker hook B1's audit notes
  presumed as the proper replacement for the 50 µs tokio-sleep
  busy-poll) survive with the same shape?
- Did Pliron's `dialect-mir` opcodes drift relative to the 23-row
  mapping table the O3 scaffold pinned in
  [`pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs)?
  Yes → [Step 4](#step-4--pliron-dialect-lowering-implementation)
  expands proportionally.

### Check whether the cuda-oxide host crates have hit crates.io

The v0.3.1 [`Cargo.toml`](../Cargo.toml) carries a TODO on
`cuda-host` / `cuda-core` / `cuda-async`: "v0.4 should switch to
crates.io versions once NVlabs publishes". Verify:

```sh
cargo search cuda-host
cargo search cuda-core
cargo search cuda-async
```

Published under those names at v0.2.0 → switch to `version = "0.2"`.
Published under different names or not at all → keep the git pin, bump
the rev to the v0.2.0 tag SHA. Decision captured in
[Step 1](#step-1--dependency-bump).

### Confirm the toolchain pin

cuda-oxide v0.1 pins `nightly-2026-04-03` and F4 already bumped
[`rust-toolchain.toml`](../rust-toolchain.toml) to match. Verify v0.2
against `https://github.com/NVlabs/cuda-oxide/blob/v0.2.0/rust-toolchain.toml`:

- **Same pin** → no toolchain work; proceed.
- **Newer pin within the W2.9 quarterly window** (e.g.
  `nightly-2026-05-10`) → bump
  [`rust-toolchain.toml`](../rust-toolchain.toml) in
  [Step 1](#step-1--dependency-bump); CHANGELOG-notify contributors.
- **Older pin** → hold. A downward bump on a v0.2 release is a smell;
  cudarc fallback preferable to chasing instability.

### Branch from `dev` and baseline tests

```sh
git fetch origin
git switch -c cutover/cuda-oxide-v0.2 origin/dev
cargo test --workspace --release > pre-cutover-tests.log 2>&1
cargo test --workspace --release --features cudarc-backend     >> pre-cutover-tests.log 2>&1
cargo test --workspace --release --features cuda-oxide-backend >> pre-cutover-tests.log 2>&1
```

The branch name is load-bearing — the rollback procedure greps for it.
Run baseline on the C1 S22 runner; the WDDM dev box documented in
[self-hosted-cuda-runner.md](runbooks/self-hosted-cuda-runner.md)
"Platform caveats" misses 22 tests for unrelated reasons.

## Step 1 — dependency bump

Edit [`Cargo.toml`](../Cargo.toml) workspace deps section.

**If cuda-oxide host crates ARE on crates.io** — replace the three
git-pinned dep lines with:

```toml
cuda-host  = { version = "0.2", default-features = false }
cuda-core  = { version = "0.2", default-features = false }
cuda-async = { version = "0.2", default-features = false }
```

Drop the long TODO comment block; one-line replacement pointing at
[Step 8](#step-8--documentation-update).

**If they are NOT on crates.io yet** — resolve the v0.2.0 tag SHA
with `git ls-remote --tags https://github.com/NVlabs/cuda-oxide v0.2.0`
and paste it as the new `rev` in all three lines. Keep the long TODO
block; replace "v0.1.0" with "v0.2.0" and update the SHA; re-target
the TODO at v0.3 of cuda-oxide for the next cycle.

**Update `deny.toml`.** [`deny.toml`](../deny.toml) F2 allowlists the
Pliron commit pinned by cuda-oxide v0.1. If v0.2 pins a different
Pliron rev, extend (do not replace) the `allow-git` entry — other
consumers may still be on the old rev. Verify with `cargo deny check
sources`.

**Cargo update + reproducibility.**

```sh
cargo update -p cuda-host -p cuda-core -p cuda-async
cargo update          # pull transitive pliron / dialect deps
```

Lockfile churn should be limited to the cuda-oxide crates, Pliron, and
its downstream dialects. Anything else moving is a smell.

Update [`docs/REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md)
§"Git-pinned sources": delete the three rows if crates.io migration
happened, otherwise update the SHA column with a v0.2.0 tag footnote.

## Step 2 — API drift fixes in `cuda_oxide_backend.rs`

Edit
[`crates/tensor-wasm-mem/src/cuda_oxide_backend.rs`](../crates/tensor-wasm-mem/src/cuda_oxide_backend.rs).
Every "v0.4 port" / "scaffold stub" marker lands real code now.

**Replace the `PhantomData` placeholder.** The
`_todo_inner: PhantomData<*mut u8>` field becomes the real owned
handle (v0.1 README suggests `cuda_host::DeviceBuffer<u8>`; verify
against v0.2 CHANGELOG):

```rust
pub struct CudaOxideUnifiedBuffer {
    size: usize,
    inner: cuda_host::DeviceBuffer<u8>,
}
```

Drop the long `// TODO(v0.4 port)` block. The `unsafe impl Send +
Sync` impls stay; rewrite the SAFETY comment to cite the cuda-oxide
v0.2 guarantee that `DeviceBuffer<u8>` is `Send`-safe (or wrap in
`Arc` if it is not).

**Replace the `NOT_YET_WIRED` sentinel.** Today's body always returns
`Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`; the cutover replaces
it with a real allocation:

```rust
pub fn allocate(size: usize) -> Result<Self, UnifiedError> {
    let device = CudaDevice::current().map_err(UnifiedError::cuda)?;
    let inner = device.alloc_managed(size).map_err(UnifiedError::cuda)?;
    Ok(Self { size, inner })
}
```

Verify the exact API name (`alloc_managed`, `alloc_uvm`, whatever
v0.2 settled on). Apply the same swap to the `apply_advice` free
function (real `cuda_host::DeviceBuffer::advise`). Delete the
`pub(crate) const NOT_YET_WIRED` constant — it was a grep-able
landmark that should not survive cutover.

**Wire `Drop` against the real free.** The v0.1 scaffold's `Drop`
only emits a `tracing::warn!`. Replace with a body that mirrors the
cudarc-backend
[`drop`](../crates/tensor-wasm-mem/src/cudarc_backend.rs) shape — if
cuda_host's own `DeviceBuffer<u8>::Drop` already calls `cuMemFree_v2`
and logs failures, our impl becomes an empty body; otherwise restore
an explicit warn-on-failure call matching
[`CUDARC-SPIKE.md`](CUDARC-SPIKE.md) gap #6.

**Unignore the deeper smoke test.**
[`tests/cuda_oxide_smoke.rs`](../crates/tensor-wasm-mem/tests/cuda_oxide_smoke.rs)
carries one `#[ignore]`d test `cuda_oxide_round_trip_on_device_v0_4`.
Flip the ignore off and flesh the body — the docstring already lists
the four assertions (`len`, write, read, `apply_advice`). Drop the
two `_returns_not_yet_wired` / `_is_exported` tests; they assert the
scaffold sentinel that no longer exists.

## Step 3 — backing wiring in `unified.rs`

Edit
[`unified.rs`](../crates/tensor-wasm-mem/src/unified.rs). D2 laid down
the three-branch `backing_impl` cfg structure; cutover adds the fourth
branch and expands the precedence table from 4 rows to 8.

**Add the fourth `backing_impl` cfg branch.** Below the three existing
ones, the explicit "only cuda-oxide-backend is on" case:

```rust
#[cfg(all(
    not(feature = "unified-memory"),
    not(feature = "cudarc-backend"),
    feature = "cuda-oxide-backend",
))]
mod backing_impl {
    use super::*;
    use crate::cuda_oxide_backend::CudaOxideUnifiedBuffer;

    pub(crate) const IS_UVM_BACKED: bool = true;
    pub(crate) enum Backing { CudaOxide(CudaOxideUnifiedBuffer) }

    impl Backing {
        pub(crate) fn allocate(size: usize)
            -> Result<(*mut u8, Self), UnifiedError>
        {
            let buf = CudaOxideUnifiedBuffer::allocate(size)?;
            let ptr = buf.as_ptr() as *mut u8;
            Ok((ptr, Backing::CudaOxide(buf)))
        }
    }
}
```

Verify `CudaOxideUnifiedBuffer` exposes `as_ptr()` returning the
managed pointer; if not, add it in
[Step 2](#step-2--api-drift-fixes-in-cuda_oxide_backendrs).

**Expand the precedence table.** The module-level rustdoc table grows
from 4 rows to 8 (the 2³ matrix of `unified-memory` × `cudarc-backend`
× `cuda-oxide-backend`). Ordering: `unified-memory` precedes
`cudarc-backend` precedes `cuda-oxide-backend`, mirroring cfg order.
Document each row's `IS_UVM_BACKED`.

## Step 4 — Pliron-dialect lowering implementation

**Largest single step. Budget ~3 days for a careful first pass.** The
O3 scaffold trait `WasmToPliron` and the 23-row mapping table in
[`pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
module rustdoc were the load-bearing artifacts the v0.4 port waited
on; they go executable here.

**Add the Pliron crate dep** in `tensor-wasm-jit/Cargo.toml`, gated
behind `cuda-oxide-backend`:

```toml
[dependencies]
pliron = { version = "0.2", optional = true }

[features]
cuda-oxide-backend = ["dep:pliron"]
```

Pliron still git-pinned? Mirror the [Step 1](#step-1--dependency-bump)
git-rev approach.

**Replace `StubLowerer::lower`** with a real Cranelift IR → Pliron
`Operation` translator. The trait signature also changes: the
string-to-string placeholder becomes a real Cranelift module type in
and Pliron `Module` out. Public-API change to `tensor-wasm-jit`; v0.5
SemVer bump covers it.

Start with the cleanest four mapping-table rows:

| Cranelift op | Pliron `dialect-mir` op |
|---|---|
| `iadd` | `arith.addi` |
| `isub` | `arith.subi` |
| `imul` | `arith.muli` |
| `idiv` / `udiv` | `arith.divsi` / `arith.divui` |

Pure integer arithmetic, width carried in result type, 1:1 mapping,
no device-pointer translation, no `mem2reg` or branch lowering. They
exercise the trait shape end-to-end.

**Add a lowering test.** New file
`crates/tensor-wasm-jit/tests/pliron_lowering_smoke.rs`. Build a
4-line Cranelift function `iadd; imul; isub; idiv`, run
`cranelift_to_dialect_mir`, assert the emitted Pliron module passes
Pliron's `Module::verify()`. Mark hardware-dependent assertions
`#[ignore = "requires nightly-2026-04-03 + cuda-oxide v0.2"]` so the
default workspace build skips them.

**Defer the hard rows.** Do NOT land in v0.5 cutover; document the
deferral in the module rustdoc and file follow-up issues:

- `call` / `call_indirect` — device-vs-host call distinction is a
  detector-contract change bigger than v0.5. Defer to v0.5.1 / v0.6.
- `atomic_*` (load, store, rmw, cas) — Wasm threads + GPU atomics
  memory model. O3 already lists as hard-rejected; keep rejected.
- Vectorised SIMD (`vmin`, `vmax`, `vsplat`, `vselect`, `vall_true`,
  `vany_true`) — per-warp-lane mapping needs warp-shuffle intrinsics;
  W1.1 does not yet thread warp size through. Defer.
- `load` / `store` — device-pointer translation needs the W1.1 base
  pointer threaded through. v0.5.1 follow-up; v0.5 first pass works
  on operations with no memory operand.

The detector
([`detector.rs`](../crates/tensor-wasm-jit/src/detector.rs)) gets a
one-line gate: candidates containing any deferred opcode fall back to
the blueprint path. The O3-pre-declared `PlironLoweringError::UnsupportedOp`
variant is the signal the detector filters on.

## Step 5 — cuda-async DispatchFuture wiring

Resolves audit Problem #1 (the B1 50 µs tokio-sleep workaround) the
right way. Today
[`async_dispatch.rs`](../crates/tensor-wasm-wasi-gpu/src/async_dispatch.rs)
polls the CUDA event every 50 µs via `tokio::time::sleep(Duration::
from_micros(50))` after registering the waker. cuda-async exposes a
real callback hook the CUDA driver invokes when the event signals —
fires the waker directly, no poll cycle.

**Replace the busy-poll body.** Grep for `Duration::from_micros(50)`:

```rust
// Before (B1 workaround):
let waker = cx.waker().clone();
tokio::spawn(async move {
    loop {
        if event_signaled() { waker.wake(); break; }
        tokio::time::sleep(std::time::Duration::from_micros(50)).await;
    }
});

// After (cuda-async waker):
let waker = cx.waker().clone();
cuda_async::register_callback(event_handle, move || waker.wake())
    .map_err(DispatchError::from)?;
```

Verify the entry-point name against v0.2 docs (`register_callback`,
`on_complete`, etc.). Shape: callback closure invoked once on event
signal.

**Un-stub the F3 cuda-async bench backend** in
[`dispatch_future_backends.rs`](../crates/tensor-wasm-bench/benches/dispatch_future_backends.rs).
The `CudaAsyncBackend` impl currently emits a `"status":"skipped"`
line; wire it to exercise the new `cuda_async::register_callback`
path the same way `BusyPollBackend` exercises `DispatchFuture::ready`.

**Rerun and compare numbers:**

```sh
cargo bench -p tensor-wasm-bench --bench dispatch_future_backends \
    --features cuda-oxide-backend
```

Expected (from RFC 0001 Unresolved Question on cuda-async vs
busy-poll):

- **Short kernels (≤ 50 µs):** roughly tied. Busy-poll's 50 µs poll
  shorter than kernel itself; wakes effectively immediately.
- **Long kernels (≥ 1 ms):** cuda-async wins ~5-10%. Busy-poll wastes
  ≥ 20 wake-and-check cycles per kernel; callback fires once at
  completion.

Record numbers in `bench-results/dispatch-future-backends-v0.5.txt` for
the cutover PR. If cuda-async **loses** the benchmark, do not flip the
default — busy-poll is robust, callback path may have a subtle bug,
and a v0.5 cutover should not ship a regression to fix a theoretical
waste.

## Step 6 — default flip

The moment "cuda-oxide-backend becomes the v0.5 default" lands in
code rather than plans.

**`crates/tensor-wasm-mem/Cargo.toml`:**

```toml
[features]
default = ["cuda-oxide-backend"]
# Was: default = ["unified-memory"]
unified-memory = ["dep:cust", "dep:ptx-builder"]  # kept; deprecated
cudarc-backend = ["dep:cudarc"]
cuda-oxide-backend = ["dep:cuda-host", "dep:cuda-core", "dep:cuda-async"]
```

**Add `#[deprecated]`** to the `unified-memory`-gated `cust`-backed
module in
[`crates/tensor-wasm-mem/src/lib.rs`](../crates/tensor-wasm-mem/src/lib.rs):

```rust
#[cfg(feature = "unified-memory")]
#[deprecated(
    since = "0.5.0",
    note = "the `unified-memory` cust path is scheduled for removal in v0.6 \
            -- migrate to `cuda-oxide-backend` (default) or `cudarc-backend`. \
            See docs/MIGRATION-v0-to-v1.md."
)]
pub mod cust_backed { /* ... */ }
```

Match the wording to the CHANGELOG entry — operators read both.

**Update [`CHANGELOG.md`](../CHANGELOG.md)** with a v0.5.0 entry:

```markdown
## [0.5.0] — YYYY-MM-DD

### Changed (BREAKING)
- Default GPU backend flipped from `cust` (`unified-memory`) to
  `cuda-oxide` (`cuda-oxide-backend`) per RFC 0001.
- Workspace toolchain `nightly-2026-04-03` (in place since v0.3.4 per F4).

### Deprecated
- `unified-memory` feature flag — deprecated alias for the v0.4-era
  cust path. Removal in v0.6.

### Added
- `tensor-wasm-jit::pliron_dialect::cranelift_to_dialect_mir` —
  Cranelift → Pliron `dialect-mir` lowering (4 of 23 mapping rows;
  remaining tracked in v0.5.1).
- cuda-async waker integration for `DispatchFuture`; B1 50 µs
  tokio-sleep busy-poll removed.
```

**Update [`MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md)** with a
"v0.4 → v0.5" section: feature-flag rename + deprecation timeline,
toolchain expectation (no change for F4-updated contributors), Helm
chart `image.backend` default change (empty default now resolves to
the cuda-oxide image variant, not host-only — operators relying on
empty-default must opt into the host-only tag explicitly).

**Update [`RISKS.md`](RISKS.md)** — the "CUDA `cust` 0.3.x EOL" row
flips to **Resolved (v0.5)** with a forward-reference to v0.6 removal.

## Step 7 — validation

Run on the C1 S22 runner — the only host that exercises the full
matrix correctly. Not the WDDM dev box.

```sh
# Default build (now cuda-oxide-backend).
cargo test --workspace --release

# Each backend explicitly, on a no-default-features base.
cargo test --workspace --release --no-default-features --features cuda-oxide-backend
cargo test --workspace --release --no-default-features --features cudarc-backend
cargo test --workspace --release --no-default-features --features unified-memory

# B2 end-to-end PTX dispatch test against cuda-oxide.
cargo test -p tensor-wasm-wasi-gpu --release \
    --no-default-features --features cuda-oxide-backend \
    --test vector_add_end_to_end -- --ignored
```

All four invocations must pass. Coexistence is the load-bearing
property — the three-backend live evaluation RFC 0001 "Rollout —
v0.4 (parity)" promised is what makes the v0.5 default flip safe.

**Pass criteria:**

- Full test suite passes under `--features cuda-oxide-backend`.
- cudarc-backend smoke still passes (fallback stays viable).
- cust smoke still passes (one release of deprecated-but-working
  coexistence).
- B2 `vector_add_end_to_end` passes through cuda-oxide just like it
  does through cust today (same
  [`kernels/vector_add.ptx`](../kernels/vector_add.ptx) fixture, same
  expected output).
- F3 `dispatch_future_backends` shows real cuda-async numbers (no
  `"status":"skipped"` line).

**Failure modes:**

- Compare against `pre-cutover-tests.log` from Pre-flight. Pre-existing
  failure = not a cutover regression; document and proceed.
- New failure under `--features cuda-oxide-backend` → v0.2 API drift
  under-estimated; re-read CHANGELOG and rework
  [Step 2](#step-2--api-drift-fixes-in-cuda_oxide_backendrs).
- New failure under `--features cudarc-backend` or
  `--features unified-memory` → cutover broke coexistence; revert
  breaking edit and retry.

## Step 8 — documentation update

The cutover is real only after documentation reflects it.

**RFC 0001 → accepted.**

```sh
git mv rfcs/0001-cuda-oxide-integration.md rfcs/accepted/
```

Front-matter edits: status `Accepted (cutover commit YYYY-MM-DD)`,
`Implemented in: v0.5.0`. Update every cross-link in the workspace:

```sh
grep -rln "rfcs/0001-cuda-oxide-integration" \
    --include="*.md" --include="*.rs" --include="*.toml" .
```

Hits include the O2/O3 scaffolds, [`Cargo.toml`](../Cargo.toml),
[`CUDARC-SPIKE.md`](CUDARC-SPIKE.md),
[`CUDA-KERNELS.md`](CUDA-KERNELS.md), [`RISKS.md`](RISKS.md),
[`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md),
[`PATH-TO-V1.md`](PATH-TO-V1.md), and a few more — let grep speak.

**CUDARC-SPIKE.md downgrade.** Prepend a banner to the
"Recommendation: cutover plan" section:

> **Update YYYY-MM-DD (v0.5 cutover):** cuda-oxide v0.2 shipped; the
> v0.5 default flipped to `cuda-oxide-backend` per RFC 0001 (accepted
> at [`rfcs/accepted/`](../rfcs/accepted/)). `cudarc-backend` stays a
> supported alternative and the documented fallback; the rest of this
> spike is preserved as the design rationale for the fallback path.

**CUDA-KERNELS.md "Path C".** Strip the v0.1.0 alpha caveats
("alpha", "v0.1 surface may break", "wait for v0.2"). Path C is now
the canonical kernel-authoring path; Path A (hand-PTX) and Path B
(out-of-tree nvcc) become legacy alternatives. Do **not** delete A/B —
they describe what existing v0.3.x/v0.4.x kernel files look like.

**PATH-TO-V1.md Open Decision #1** flips to status **Resolved
YYYY-MM-DD**, pointing at the accepted-RFC path, naming
`cuda-oxide-backend` as v0.5 default, `cudarc-backend` as fallback,
`cust` deprecated v0.5 / removed v0.6. If Open Decision #8
(toolchain pin cadence) also got resolved by F4 + this cutover, mark
it in the same edit.

**Helm chart README.**
[`deploy/helm/tensor-wasm/README.md`](../deploy/helm/tensor-wasm/README.md)
§"Backend selection" — empty `image.backend` now resolves to the
cuda-oxide variant, not host-only. No Helm code change; the C8
[`Dockerfile`](../Dockerfile) already builds all four variants and
the F1 [`values.yaml`](../deploy/helm/tensor-wasm/values.yaml) already
documents the toggle. README narrative only.

## Rollback procedure

If validation in [Step 7](#step-7--validation) fails in a way that
cannot be patched in the same PR.

**Revert the cutover commit on `dev`:**

```sh
git switch dev
git revert <CUTOVER_COMMIT_SHA> --no-edit
```

Push the revert through normal review, not directly. Do not
force-push or amend `cutover/cuda-oxide-v0.2`; the branch name is the
audit trail and other contributors may have rebased onto it.

**Re-pin to v0.1.0 SHA** if the revert did not cleanly restore
[`Cargo.toml`](../Cargo.toml):

```toml
cuda-host  = { git = "https://github.com/NVlabs/cuda-oxide", rev = "4a56e4220aab8ce5d085a411e7f806cebb647d14", default-features = false }
cuda-core  = { git = "https://github.com/NVlabs/cuda-oxide", rev = "4a56e4220aab8ce5d085a411e7f806cebb647d14", default-features = false }
cuda-async = { git = "https://github.com/NVlabs/cuda-oxide", rev = "4a56e4220aab8ce5d085a411e7f806cebb647d14", default-features = false }
```

That is the v0.1.0 tag SHA the v0.3.1 scaffold was built against — the
known-good rollback target documented in this runbook's preamble.

**Restore `unified-memory` as the workspace default** and drop the
`#[deprecated]` attribute on the cust path from
[Step 6](#step-6--default-flip).

**Keep `cudarc-backend` as the documented fallback.** Already in tree
from W1.2; no edit required. It was the v0.5 default in the
contingent-no branch of RFC 0001 and remains the recommendation when
the next cuda-oxide stable release is not imminent.

**File an upstream issue against `NVlabs/cuda-oxide`** with the
cuda-oxide commit SHA, the `cargo test` failure output, the S22
runner's CUDA toolkit + driver (via
`nvidia-smi --query-gpu=driver_version,compute_cap --format=csv`),
and a link to the reverted PR. The TensorWasm cutover may move; the
upstream regression report stays useful regardless.

## Time budget

One maintainer, uninterrupted, S22 runner available. Pad ~25% for
review turnaround.

| Step | Effort | Notes |
|---|---|---|
| Pre-flight | 0.5 day | CHANGELOG read + baseline + scratch notes |
| [Step 1](#step-1--dependency-bump) | 0.5 day | mechanical on crates.io; +0.5 day if `deny.toml` Pliron rev moved |
| [Step 2](#step-2--api-drift-fixes-in-cuda_oxide_backendrs) | 1 day | scales with v0.1 → v0.2 rename churn |
| [Step 3](#step-3--backing-wiring-in-unifiedrs) | 0.5 day | mirrors the cudarc cfg branch |
| [Step 4](#step-4--pliron-dialect-lowering-implementation) | **3 days** | **largest step**; first 4 mapping rows + smoke test |
| [Step 5](#step-5--cuda-async-dispatchfuture-wiring) | 1 day | includes bench run + numbers archive |
| [Step 6](#step-6--default-flip) | 0.5 day | Cargo.toml + CHANGELOG + `#[deprecated]` |
| [Step 7](#step-7--validation) | 1 day | four-invocation matrix on the S22 runner |
| [Step 8](#step-8--documentation-update) | 1 day | RFC move + grep-and-update across ~10 docs |
| **Total** | **5-10 working days** | lower = clean release, no surprises; upper = drift from scaffold assumptions |

If any step over-runs its budget by 2x, stop and treat it as a
[Rollback procedure](#rollback-procedure) trigger rather than pushing
through. The cudarc fallback exists precisely so "rollback and ship
cudarc as v0.5 default" stays viable through the entire window.

## Related

- [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md)
  — v0.5 decision rationale, Option C side-by-side strategy,
  toolchain bump plan. Becomes `rfcs/accepted/` after
  [Step 8](#step-8--documentation-update).
- [`docs/CUDARC-SPIKE.md`](CUDARC-SPIKE.md) — W1.2 spike + W5.9
  friction notes; same shape of doc for the cudarc fallback decision.
  Carries the recommendation that becomes the v0.5 fallback if the
  preconditions here never go green.
- [`docs/runbooks/self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md)
  — C1 runner registration. [Step 7](#step-7--validation) requires it.
- [`deny.toml`](../deny.toml) — F2 Pliron rev allowlist; touched in
  [Step 1](#step-1--dependency-bump) if the Pliron pin moved.
- [`crates/tensor-wasm-bench/benches/dispatch_future_backends.rs`](../crates/tensor-wasm-bench/benches/dispatch_future_backends.rs)
  — F3 bench; un-stubbed in
  [Step 5](#step-5--cuda-async-dispatchfuture-wiring).
- [`crates/tensor-wasm-mem/src/cuda_oxide_backend.rs`](../crates/tensor-wasm-mem/src/cuda_oxide_backend.rs)
  — O2 scaffold; the `NOT_YET_WIRED` sentinel + `PhantomData` field
  are what [Step 2](#step-2--api-drift-fixes-in-cuda_oxide_backendrs)
  replaces.
- [`crates/tensor-wasm-jit/src/pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
  — O3 scaffold; trait signature + 23-row mapping table that
  [Step 4](#step-4--pliron-dialect-lowering-implementation) starts
  implementing.
- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — Open Decision #1 (cust
  successor) resolves at cutover; #8 (toolchain pin cadence) intersects
  via F4.
- [`docs/RISKS.md`](RISKS.md) — cust EOL row flips to **Resolved** in
  [Step 6](#step-6--default-flip).
- [`docs/MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) — v0.4 → v0.5
  section grows in [Step 6](#step-6--default-flip).
- [`Dockerfile`](../Dockerfile) +
  [`deploy/helm/tensor-wasm/values.yaml`](../deploy/helm/tensor-wasm/values.yaml)
  — C8 + F1 backend-tag selection; no code change, only README
  narrative in [Step 8](#step-8--documentation-update).
- [`docs/runbooks/README.md`](runbooks/README.md) — runbook contract
  this document follows (procedure-runbook variant).

---

_Status: dormant until cuda-oxide v0.2.0 ships. Until then, the
v0.3.x + v0.4.x three-backend live evaluation from RFC 0001 Option C
runs its course, the cudarc fallback stays viable, and this runbook
sits ready for the day the
[preconditions](#when-to-run-this) all go green._
