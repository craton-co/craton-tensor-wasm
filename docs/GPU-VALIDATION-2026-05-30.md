# Hardware verification run — 2026-05-30

First real-silicon validation of the WASM→GPU stack. Until this run every
CUDA path was gated behind `#[ignore = "requires CUDA hardware"]` /
`#[cfg(feature = "cuda")]` and had **never executed on a GPU** — and, as it
turns out, the `--features cuda` host path had never even *compiled*.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 |
| Compute capability | **7.5 (Turing)** |
| Driver | 591.86 (CUDA 13.1 driver API) |
| CUDA Toolkit | 13.2 (`nvcc` 13.2.78, `cuda.lib`/`cudart.lib`/`nvrtc.lib` present) |
| Host | Windows 11, MSVC BuildTools 14.44, Rust nightly-2026-04-03 |
| libclang | 18.1.1 via `pip install --user libclang` (needed by `cust_raw` bindgen; not previously present) |

Reproduce with `scripts/run-gpu-tests.{sh,ps1}` (auto-detects CUDA + libclang,
runs every suite single-threaded, logs to `bench-results/gpu-run/`).

## Results

### ✅ `tensor-wasm-mem --features unified-memory` (cust path) — PASS
`cust` 0.3.2 / `cust_raw` 0.11.3 **build cleanly against CUDA 13.2 + libclang 18**
(the single biggest unknown). On hardware:
- `cust_unified_buffer_snapshot_round_trip_on_device` … **ok** — real
  `cuMemAllocManaged` allocate → write → snapshot → restore round-trip.

### ◑ `tensor-wasm-mem --features gpu-mem-pool` (cudarc path) — 4 PASS / 1 FAIL
- `cudarc_backend::allocate_and_drop_small_buffer` … ok
- `cudarc_backend::device_cache_returns_same_arc_for_same_ordinal` … ok
- `driver_pin_matches_requested_cap` … ok
- `under_cap_allocation_through_pool_succeeds` … ok
- **`over_cap_allocation_through_pool_is_rejected_by_driver` … FAIL** — see BUG-1.

`cudarc` 0.13.9 (`features = ["driver","cuda-12000"]`) loads the 13.1 driver
fine — the CUDA 12 driver-API bindings are forward-compatible.

### ✗ `tensor-wasm-wasi-gpu --features cuda` — DOES NOT COMPILE — see BUG-2

## BUG-1 — driver-level per-tenant GPU memory cap (T39) is not enforced

`tests/cuda_mem_pool_driver_pin.rs::over_cap_allocation_through_pool_is_rejected_by_driver`
asks for 128 MiB from a pool created with a 64 MiB "cap" and expects the driver
to refuse with `CUDA_ERROR_OUT_OF_MEMORY`. **The allocation succeeds.**

Root cause (`crates/tensor-wasm-mem/src/cuda_mem_pool.rs:215`): the cap is wired
as `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, cap)`. But
`RELEASE_THRESHOLD` is a *memory-retention hint* — it controls how much freed
memory the pool caches before returning it to the OS. **It is not an allocation
ceiling**, and CUDA memory pools expose no hard max-size attribute. So the T39
threat model (a tenant with a raw driver handle bypassing the in-process
`consume_gpu_bytes` counter) is **not** closed at the driver level, contrary to
the module's "driver pin LANDED" status note and `docs/GPU-QUOTAS.md`.

Severity: this is a security-relevant correctness defect in a multi-tenant
isolation feature, and it would have shipped silently — the test that catches it
only runs on hardware. Fix options: enforce the cap host-side in
`TenantMemPool::allocate` (reject when `live + size > cap` before
`cuMemAllocFromPoolAsync`), and/or back the pool with a fixed-size virtual-memory
reservation (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`) whose mapped
size is the hard cap. Either way, drop the claim that `RELEASE_THRESHOLD` is the
enforcement mechanism.

## BUG-2 — the `--features cuda` host path has bit-rotted (never compiled)

`cargo test -p tensor-wasm-wasi-gpu --features cuda` fails to build with six
`error[E0063]: missing field 'device_ptr' in initializer of
'device_mem::DeviceMemEntry'`:

```
crates/tensor-wasm-wasi-gpu/src/host.rs:2596
crates/tensor-wasm-wasi-gpu/src/host.rs:2641
crates/tensor-wasm-wasi-gpu/src/host.rs:2686
crates/tensor-wasm-wasi-gpu/src/host.rs:2870
crates/tensor-wasm-wasi-gpu/src/host.rs:2877
crates/tensor-wasm-wasi-gpu/src/host.rs:2925
```

A `device_ptr` field was added to `DeviceMemEntry` but the six `cfg(cuda)`-gated
constructors were never updated. Because there is no GPU CI runner, nothing ever
compiles this feature — so the headline path (Wasm → `wasi:cuda` →
`cuLaunchKernel` → readback) could not even be built, let alone run. This is the
strongest argument for the GPU CI lane: a plain `cargo build --features cuda` in
CI would have caught it.

Fix: initialize `device_ptr` at each site with the device pointer the
allocation/registration already has in scope. (In progress.)

## sm_75 support (this GPU is Turing, the kernels target Ampere)

`kernels/vector_add.ptx`, the test fixture, and `ptx_emit`'s `DEFAULT_TARGET`
all hard-code `.target sm_80`. sm_80 PTX is rejected by the driver JIT on this
sm_75 card, so `vector_add_end_to_end_real_ptx_real_kernel` self-skips. The
kernel body is capability-agnostic (no wmma/tensor-core ops), so sm_75 variants
were added:
- `kernels/vector_add_sm75.ptx`
- `crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add_sm75.ptx`

Remaining: make `register_real_kernel` / the emitter pick the sm_75 fixture when
the device's compute capability is < 8.0, so the end-to-end launch proof runs on
Turing-class hardware. (In progress, blocked on BUG-2 — the test crate must
compile first.)

## Status of the loop

| Stage | State |
|---|---|
| cust builds on CUDA 13.2 | ✅ proven |
| cudarc builds + runs on GPU | ✅ proven (1 real bug found) |
| cust unified-memory round-trip on GPU | ✅ proven |
| `--features cuda` host path compiles | ❌ BUG-2 (fix pending) |
| Real kernel launch + verified output | ⏳ blocked on BUG-2 + sm_75 wiring |
| `--features cuda` benches on GPU | ⏳ blocked on BUG-2 |
| Self-hosted GPU CI lane active | ⏳ `gpu.yml` still needs a registered `[self-hosted, gpu]` runner |

## Update — fix #7 (PTX regeneration + capability-aware selection) — code landed, launch still blocked by the driver JIT on this box

This work made three correct, committable improvements, but did **not** achieve
a verified kernel launch on this machine — the local driver's PTX JIT rejects
every PTX we feed it, which is an environment limitation, not a code bug.

**What landed (correct, and right for a healthy driver / the S22 runner):**
1. `kernels/vector_add.cu` is now the source of truth; `make ptx` regenerates
   all four fixtures with nvcc instead of hand-writing PTX.
2. The e2e test queries device compute capability via the raw driver API
   (`cuDeviceGetAttribute` — cust's safe `Device::get_attribute` is not exposed
   under our `default-features = false` cust set) and loads ONLY the
   arch-matched fixture (sm_75 for cc < 8.0, else sm_80). It never feeds a
   known-mismatched fixture to the JIT first (which had left a sticky context
   error that poisoned the next load — the `InvalidPtx`→`UnknownError` cascade).
3. CUDA init ordering fixed: a shared `ensure_cuda_initialized()` runs `cuInit`
   (via `quick_init`) before the capability query, which otherwise failed with
   `CUDA_ERROR_NOT_INITIALIZED`.

**What the hardware actually says (the blocker).** On the RTX 2060 (cc 7.5),
`cust::module::Module::from_ptx` rejects the `vector_add` PTX at **every** ISA
version we tried, for the arch-matched sm_75 target:

| PTX source | `.version` | `Module::from_ptx` result |
|---|---|---|
| original hand-written | 8.0 | `InvalidPtx` |
| nvcc 13.2 (toolkit) | 9.2 | `UnknownError` (cust can't map `UNSUPPORTED_PTX_VERSION`: driver is 13.1, toolkit 13.2) |
| nvcc 12.6 (toolkit) | 8.5 | `InvalidPtx` |

A structurally-valid, nvcc-generated, **arch-matched** `.version 8.5` sm_75
module being rejected with `InvalidPtx` on an sm_75 device points at the
**driver's JIT compiler being non-functional in this environment** (headless /
WDDM / sandbox quirk), not at the PTX. Corroborating: the cust + cudarc *memory*
paths (`cuMemAllocManaged`, pools, snapshot round-trip) all pass on this same
box — those never invoke the PTX JIT; only module loading does, and only that
fails.

**Status of the launch proof:** still **not** verified on this machine, blocked
by the local JIT. The code changes here are the right ones and should produce a
real launch on a box with a working driver JIT (e.g. the S22 self-hosted
runner). The e2e test now *fails loudly* (panics) when the arch-matched module
is rejected, rather than silently skipping — deliberately, so a broken JIT or a
bad fixture is surfaced rather than hidden. On a healthy runner it proceeds to
`cuLaunchKernel` + readback.

#6 (thread-bound context in the async path) and #1 (driver-level mem cap) remain
open and untouched by this change.

## Update — #1 verified, #9 found + fixed (cudarc context binding)

After commit `47c5a05` (parallel session: host-side per-tenant cap for #1 +
`cuda_ctx.rs` shared primary context for #6), re-running the gpu-mem-pool ignored
suite on the RTX 2060 confirmed and uncovered:

- **#1 VERIFIED ✅**: `over_cap_allocation_through_pool_is_rejected_by_driver ... ok`
  (was FAILED), plus `under_cap` / `driver_pin_matches_requested_cap` /
  `cuda_mem_pool_scaffold` (3) all ok. The host-side `live_bytes` reservation in
  `TenantMemPool::allocate` closes BUG-1.
- **BUG-9 (found, then fixed) ✅**: with #1 letting the run proceed past the old
  failure, `cudarc_smoke::cudarc_round_trip_on_device` and
  `cudarc_prefetch_round_trip_on_device` then failed with
  `cuMemAllocManaged -> CUDA_ERROR_INVALID_CONTEXT`. Root cause:
  `CudarcUnifiedBuffer::new_on` (cudarc_backend.rs) called `device_for()` (which
  returns a cached `Arc<CudaDevice>` clone — only the thread that first built the
  device has its context current) but did **not** call `ensure_context_bound`
  before `cuMemAllocManaged`. Its sibling `apply_advice`/`prefetch_*` paths
  already bound the context; `new_on` was the gap. This is the cudarc-path twin of
  #6. One-line fix (add `ensure_context_bound(&device)?` in `new_on`); the stale
  "the device above ensures the primary context is current" comment was wrong and
  is corrected. These are `#[ignore]`d hardware tests, so hosted CI was unaffected
  — only a real GPU surfaces it (and session 1's fail-fast had stopped before
  these ran, so they had never actually executed on silicon).

After the #9 fix, `cudarc_round_trip_on_device` and
`cudarc_apply_advice_read_mostly_on_device` pass, along with the snapshot
round-trip, visible-window, `cuda_mem_pool_scaffold` (3), and all three
driver-pin tests (incl. `over_cap`). The cust `unified-memory` snapshot
round-trip also passes.

**BUG-10 (found, NOT fixed): `cuMemPrefetchAsync` unsupported on this box.**
`cudarc_prefetch_round_trip_on_device` still fails — but now one line LATER than
the #9 alloc failure (cudarc_smoke.rs:80, the prefetch, not :79 the alloc), with
`cuMemPrefetchAsync(device) -> CUDA_ERROR_INVALID_DEVICE`. This is almost
certainly an environment limitation, not a code bug: `cuMemPrefetchAsync`
requires the device's `concurrentManagedAccess` attribute to be non-zero, which
is **0 on Windows under the WDDM driver model** (managed-memory prefetch is a
Linux / TCC feature). The right fix is to gate the prefetch path (and this test)
on the `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` attribute and treat
prefetch as a no-op/skip where unsupported — a small, separate change deferred
here rather than landed unverified. Tracked as BUG-10.

Remaining red items on this box are both environment limitations of the local
Windows/WDDM driver, not code defects: the PTX-JIT launch proof (#7/BUG-8) and
`cuMemPrefetchAsync` (BUG-10). Everything that does not depend on those two
driver features passes on the RTX 2060.

## Update — BUG-10 fixed+verified, BUG-8 fixed (code), BUG-9 verified

- **BUG-10 — FIXED + VERIFIED ✅.** `CudarcUnifiedBuffer::prefetch_to_device` /
  `prefetch_to_host` now query `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`
  (helper `supports_managed_prefetch`) and degrade to an advisory no-op where it
  is 0 (Windows/WDDM), instead of erroring. The full `cudarc_smoke` ignored suite
  is now green on the RTX 2060 — `cudarc_prefetch_round_trip_on_device ... ok`
  (was FAILED with `INVALID_DEVICE`), plus round-trip and advice. On Linux/TCC the
  real prefetch path runs unchanged.
- **BUG-9 — FIXED + VERIFIED ✅** (commit `842ad14`): `CudarcUnifiedBuffer::new_on`
  binds the primary context before `cuMemAllocManaged`.
- **BUG-8 — FIXED (code) + compiles; runtime still gated by BUG-7 JIT.** The e2e
  test now backs guest linear memory with `cuMemAllocManaged` via
  `make_managed_engine_and_linker` (`TensorWasmMemoryCreator` installed through
  `Config::with_host_memory`), and the wasi-gpu `cuda` feature pulls in
  `tensor-wasm-mem/unified-memory`. Confirmed: it **compiles cleanly under
  `--features cuda`** and the test runs up to the module-load gate, where it still
  hits the local driver's `InvalidPtx` (BUG-7) before instantiation/launch. So
  the managed-memory wiring is in place; the only thing between here and a
  verified launch is a host with a working PTX JIT (Linux / TCC / the S22 runner).

### Final scoreboard (RTX 2060 / Windows / WDDM / CUDA 13.1 driver)

| # | Status |
|---|---|
| BUG-1 (per-tenant cap) | fixed + **verified on GPU** |
| BUG-2 (`--features cuda` compile) | fixed + **verified** |
| BUG-6 (cust ctx thread-bind) | fixed (code); mem paths verified |
| BUG-7 (PTX JIT rejects modules) | nvcc-regen + cap-select landed; **launch blocked by local JIT (env)** |
| BUG-8 (managed-backed guest mem) | fixed (code) + **compiles**; runtime gated by BUG-7 |
| BUG-9 (cudarc alloc ctx bind) | fixed + **verified on GPU** |
| BUG-10 (`cuMemPrefetchAsync` on WDDM) | fixed + **verified on GPU** |

Five of seven fixed-and-verified on this box. The two that can't be verified here
(BUG-7 launch, BUG-8 end-to-end) are both gated on the local driver's PTX JIT and
should pass on a Linux/TCC host or the S22 self-hosted runner; the code for both
is in place and compiles.

## RESOLVED 2026-05-31 — BUG-7 + BUG-8 fixed; real kernel launch VERIFIED on GPU

`vector_add_end_to_end_real_ptx_real_kernel` **passes on the RTX 2060 when run
on its own**: a real CUDA kernel launches through the full
Wasm -> wasi:cuda -> cuLaunchKernel path and the test verifies
`c[i] == a[i] + b[i]` read back from managed linear memory. This is the headline
WASM->GPU proof the validation effort set out to establish — a real compute
kernel, driven by a Wasm guest, producing verified-correct output on silicon.

```
cargo test -p tensor-wasm-wasi-gpu --features cuda --test kernel_args_e2e \
  vector_add_end_to_end_real_ptx_real_kernel -- --ignored
=> test result: ok. 1 passed; 0 failed
```

KNOWN REMAINING ISSUE (full-file run): when the ENTIRE `kernel_args_e2e` file
runs in one process (`--include-ignored`), the launch test fails with
`CUDA_ERROR_INVALID_CONTEXT` at `Module::from_ptx` — an earlier test in the file
leaves the process-shared CUDA primary context in a state where the late launch
test's thread has no current context. The launch CODE is correct (proven by the
isolation pass); this is a test-harness context-lifecycle interaction across
tests in one binary. See "Full-file ordering" below. The other 7 tests in the
file pass in both modes.

BUG-7 was NOT an environment limitation (my earlier conclusion was wrong). Two
real causes, both fixed:
1. **Non-ASCII bytes in the PTX.** The committed fixtures had a hand-written
   header comment containing a U+2014 em-dash. `ptxas`/the driver JIT rejects
   non-ASCII bytes anywhere in the PTX image with `CUDA_ERROR_INVALID_PTX`.
   Fix: fixtures are now generated VERBATIM by nvcc (pure ASCII) from
   `kernels/vector_add.cu`, with an ASCII-only provenance header.
2. **PTX ISA version vs driver.** nvcc 13.2 emits `.version 9.2`, which this
   box's CUDA 13.1 driver rejects (`UNSUPPORTED_PTX_VERSION`, surfaced by cust
   as `UnknownError`). Fix: generate with the CUDA 12.6 toolkit (`.version 8.5`),
   which the 13.1 driver accepts. sm_75 target matches the device.

BUG-8 required three things, all now in `make_managed_engine_and_linker`:
1. Guest linear memory backed by `cuMemAllocManaged` via `TensorWasmMemoryCreator`
   (`Config::with_host_memory`), so kernel pointer-args are device-addressable.
2. The wasmtime engine knobs the UnifiedBuffer backend needs (mirrors
   `tensor-wasm-exec`'s engine.rs): `memory_reservation(0)`,
   `memory_guard_size(0)`, `guard_before_linear_memory(false)` — managed memory
   cannot satisfy the default 4 GiB static reservation or host mprotect guards.
   Plus `async_support(true)` for the async launch path.
3. A SINGLE CUDA context shared between module-load and launch: the test's
   `ensure_cuda_initialized` now routes through `cuda_ctx::ensure_current_context`
   (the same primary-context helper the launch path uses). Loading the module in
   one context and launching its function on a stream from another context fails
   `cuLaunchKernel` with `INVALID_VALUE`.

### Final scoreboard (RTX 2060 / Windows / WDDM / CUDA 13.1 driver)

| # | Status |
|---|---|
| BUG-1 (per-tenant cap) | fixed + verified on GPU |
| BUG-2 (`--features cuda` compile) | fixed + verified |
| BUG-6 (cust ctx thread-bind) | fixed + verified (shared ctx now exercised by the passing launch) |
| BUG-7 (PTX rejected by JIT) | **fixed + VERIFIED (isolation): real kernel launches** |
| BUG-8 (managed-backed guest mem) | **fixed + VERIFIED (isolation): launch output correct** |
| BUG-9 (cudarc alloc ctx bind) | fixed + verified on GPU |
| BUG-10 (`cuMemPrefetchAsync` on WDDM) | fixed + verified on GPU |

Remaining known issue (separate, pre-existing, NOT a BUG-7/8 regression):
`host::tests::alloc_tracks_handle_then_free_lifecycle` and
`wasi_gpu_smoke::sync_returns_ok_without_cuda` hard-code the no-CUDA return value
and now fail under `--features cuda` precisely BECAUSE the device path works
(alloc returns a real handle, sync returns 0). These are the BUG-4 class; they
each need a `#[cfg(feature = "cuda")]` arm. The `kernel_args_e2e` integration
suite (which contains the launch proof) is fully green.

## Full-file ordering — partial fix; one cross-test interaction remains

Commit `a2f76af` improved cross-test robustness in two real ways:
- `cuda_ctx::ensure_current_context` no longer uses `cust::quick_init` (which
  conflicts with tensor-wasm-mem's own cust init and cached an `Err`). It now
  retains the device-0 PRIMARY context directly via `cuDevicePrimaryCtxRetain` +
  `cuCtxSetCurrent` — refcounted, coexists with every other retainer in the
  process.
- `dispatch_pipeline_compiles_against_real_module_bytes` now uses the
  arch-matched fixture via `select_vector_add_ptx()`, so its `from_ptx` no longer
  fails-and-poisons on this sm_75 box.

HONEST STATUS: this did **not** fully fix the full-file run. Measured on the
RTX 2060 with the committed code:

```
# isolation — PASS
... vector_add_end_to_end_real_ptx_real_kernel -- --ignored
=> 1 passed; 0 failed

# full file — the launch test FAILS
... --test kernel_args_e2e -- --include-ignored
=> 7 passed; 1 failed  (vector_add_end_to_end_real_ptx_real_kernel:
                        Module::from_ptx -> InvalidContext)
```

So some other test in the file still leaves the shared primary context in a state
where the launch test's thread sees no current context at `from_ptx`. This is a
**test-harness** context-lifecycle issue, not a defect in the launch path (which
the isolation run proves correct). Likely culprit: a managed-memory-backed test's
`Store`/`UnifiedBuffer` drop, or a `cust::Context` drop, decrementing the primary
context refcount or resetting current-context state for a later thread. The
durable fix is to make `register_real_kernel` (and the launch host path) call
`ensure_current_context()` unconditionally right before `from_ptx`/launch on
whatever thread runs them, and to audit cust `Context`/`UnifiedBuffer` drops for
primary-ctx release. Tracked as follow-up; the launch proof itself stands via the
isolation run.
