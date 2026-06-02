<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Hardware-Gated Work — Written But Unverified Without a CUDA GPU

This is the single authoritative inventory of code paths that are **written
and committed but have never executed against real NVIDIA silicon**. They
compile (against the CUDA stub libraries on hosted CI, or against a real
toolkit on a contributor box), their unignored unit tests pass, and their
hardware tests exist — but those hardware tests are marked
`#[ignore = "requires CUDA hardware"]` and are skipped everywhere except a
self-hosted GPU runner that does not yet exist.

A reviewer or operator reading this list should treat every item below as
**unproven on hardware**: the logic is reviewed, the types check, the stub
path runs, but the assertion that the CUDA driver actually does what the code
asks for has not been observed. The gap closes the moment the GPU CI lane in
[`.github/workflows/gpu.yml`](../.github/workflows/gpu.yml) runs against a
registered `[self-hosted, gpu]` runner (see
[`docs/runbooks/self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md)).

This document is the answer to "what hasn't been run on a GPU yet?" — it does
not duplicate the *how-to-build* matrix (that is
[`CUDA-SETUP.md`](CUDA-SETUP.md)) or the *risk-tracking* register (that is
[`RISKS.md`](RISKS.md)); both cross-link here.

## Contents

1. [Why these can't be CI-verified](#why-these-cant-be-ci-verified)
2. [How the GPU CI lane validates them](#how-the-gpu-ci-lane-validates-them)
3. [Work items](#work-items)
   1. [cust unified-memory allocation + advise + prefetch](#1-cust-unified-memory-allocation--advise--prefetch)
   2. [cudarc backend allocation + prefetch](#2-cudarc-backend-allocation--prefetch)
   3. [cuda-oxide scaffold allocation + prefetch](#3-cuda-oxide-scaffold-allocation--prefetch)
   4. [cuStreamAddCallback-based async dispatch](#4-custreamaddcallback-based-async-dispatch)
   5. [Explicit device-memory host functions (cuMemAlloc / memcpy)](#5-explicit-device-memory-host-functions-cumemalloc--memcpy)
   6. [try_grow_in_place via cuMemAddressReserve](#6-try_grow_in_place-via-cumemaddressreserve)
   7. [Experimental wmma MatMul lowering (opt-in)](#7-experimental-wmma-matmul-lowering-opt-in)
   8. [cuda-oxide host backend (compile_error-guarded)](#8-cuda-oxide-host-backend-compile_error-guarded)
4. [Cross-references](#cross-references)

---

## Why these can't be CI-verified

GitHub-hosted runners have no GPU. Per [`CUDA-SETUP.md` — "Stub libraries for
CI"](CUDA-SETUP.md#stub-libraries-for-ci), the hosted `ci` workflow links
against a directory of no-op stub `.so` files at `/usr/local/cuda/lib64/`.
That stub satisfies the linker (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`, …
resolve) so the workspace builds and the unignored unit tests run — but the
moment a test actually calls into the driver (allocate managed memory, prefetch
a page, launch a kernel, copy bytes host↔device), the stub does nothing useful
and the assertion is meaningless. Those tests are therefore marked
`#[ignore = "requires CUDA hardware"]` and skipped on hosted CI.

The result: the *control flow* around each driver call is exercised on every
PR, but the *driver call itself* — and whatever invariant depends on the
driver doing the right thing — is not. Everything below is in that second
category.

## How the GPU CI lane validates them

[`.github/workflows/gpu.yml`](../.github/workflows/gpu.yml) is the
hardware-verification lane. It is **dormant** until a runner registers with
both the `self-hosted` and `gpu` labels, and it triggers only on the `gpu-ci`
PR label or a manual `workflow_dispatch` (never on push — there is no hosted
GPU). When it runs, it re-runs the `#[ignore]`d suite with `-- --ignored`
across `tensor-wasm-mem`, `tensor-wasm-wasi-gpu`, and `tensor-wasm-tenant`,
plus the `--features cuda` benches. Each work item below names the exact job
that covers it.

This lane is deliberately distinct from
[`.github/workflows/cuda.yml`](../.github/workflows/cuda.yml): `cuda.yml` is
the per-push build + unignored-smoke lane on `[self-hosted, cuda]`; `gpu.yml`
is the on-demand, expensive, ignored-test + bench lane on `[self-hosted, gpu]`.

---

## Work items

### 1. cust unified-memory allocation + advise + prefetch

**What it does.** Under `--features unified-memory`, `tensor-wasm-mem`'s
`UnifiedBuffer` (the default cust 0.3 backing) allocates Unified Memory via
`cuMemAllocManaged`, applies access hints with `cuMemAdvise`
(`READ_MOSTLY` et al.), and migrates pages with `cuMemPrefetchAsync`. This is
the production allocation path every GPU-resident Wasm guest uses today.

**Why it can't be CI-verified without hardware.** `cuMemAllocManaged` and the
advise/prefetch family are real driver calls; the CI stub returns success
without allocating anything, so a "allocate, write from device, read back the
right bytes" assertion proves nothing on hosted runners. The round-trip tests
(`tests/cust_snapshot_conformance.rs`, and the cust-backed cases in
`tests/common`) are `#[ignore = "requires CUDA hardware"]`.

**How the GPU CI lane validates it.** The `mem-hardware-tests` job runs
`cargo test -p tensor-wasm-mem --release --features unified-memory --
--ignored --test-threads=1`. (Single-threaded because cust 0.3's
primary-context model does not survive parallel test execution.)

### 2. cudarc backend allocation + prefetch

**What it does.** Under `--features cudarc-backend` (and its strict-superset
`gpu-mem-pool`), `tensor-wasm-mem::cudarc_backend` provides a parallel
`UnifiedBuffer` implementation over the maintained `cudarc` crate — the
cust → cudarc migration spike (see [`CUDARC-SPIKE.md`](CUDARC-SPIKE.md)). It
covers the same allocate / advise / prefetch surface, the visible-window-only
zeroing guard, and — under `gpu-mem-pool` — the driver-level per-tenant cap
via `cuMemPool*` (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`).

**Why it can't be CI-verified without hardware.** Allocating a cudarc-backed
slab calls `cuMemAllocManaged`; the device-cache test calls `CudaDevice::new`,
which `dlopen`s `libcuda.so` / `nvcuda.dll`; the pool-cap tests need the driver
to actually reject an over-cap allocation. None of that is meaningful against
the stub. The relevant tests
(`tests/cudarc_smoke.rs`, `tests/cudarc_visible_window_only.rs`,
`tests/cudarc_snapshot_conformance.rs`, `tests/cuda_mem_pool_driver_pin.rs`,
the `cudarc_backend` unit tests) are all `#[ignore]`.

**How the GPU CI lane validates it.** The `mem-hardware-tests` job runs
`cargo test -p tensor-wasm-mem --release --features gpu-mem-pool --
--ignored --test-threads=1`. `gpu-mem-pool` is the strict-superset of
`cudarc-backend`, so this one invocation exercises both the cudarc allocation
path and the `cuMemPool` driver pin.

### 3. cuda-oxide scaffold allocation + prefetch

**What it does.** Under `--features cuda-oxide-backend` (the **dep-less** v0.5
cust-successor scaffold per RFC 0001), `tensor-wasm-mem::cuda_oxide_backend`
exposes `CudaOxideUnifiedBuffer`. Today the non-hardware path returns the
documented `NOT_YET_WIRED` sentinel; the ignored hardware tests are the v0.4
round-trip targets that the cutover will make real.

**Why it can't be CI-verified without hardware.** The ignored tests in
`tests/cuda_oxide_smoke.rs` (`cuda_oxide_round_trip_on_device`,
`cuda_oxide_apply_advice_read_mostly_on_device`,
`cuda_oxide_prefetch_round_trip_on_device`) and
`tests/cuda_oxide_snapshot_conformance.rs` allocate and migrate real managed
memory once the host backend is wired; they cannot run on the stub.

**How the GPU CI lane validates it.** The `mem-hardware-tests` job runs
`cargo test -p tensor-wasm-mem --release --features cuda-oxide-backend --
--ignored --test-threads=1`. (Note: this stays on the dep-less
`cuda-oxide-backend` scaffold — the strict-superset
`experimental-cuda-oxide-host-backend` is item 8 below and does not build
yet.)

### 4. cuStreamAddCallback-based async dispatch

**What it does.** `tensor-wasm-wasi-gpu::async_dispatch` resolves a kernel
launch's completion future. The intended design wakes the Tokio task from a
`cuStreamAddCallback` callback fired by the driver when the stream drains.

**Why it can't be CI-verified without hardware.** The current implementation
is a **sleep-poll fallback**: the future clones its waker, spawns a 50 µs
`tokio::time::sleep`, and re-polls `cust::event::Event::query()` on wake (see
the `poll` impl in `async_dispatch.rs`, around the "until we have a proper
`cuStreamAddCallback`-driven waker" comment). The *only* way to observe that a
`cuStreamAddCallback`-driven waker behaves correctly — that the callback fires
exactly once, on the right thread, after the stream actually drains, without a
busy-poll or a lost wakeup — is to record an event on a real stream and launch
a real kernel. On the stub there is no stream to drain and no callback to fire.

**How the GPU CI lane validates it.** When the `cuStreamAddCallback` waker
replaces the sleep-poll, its correctness is asserted by the kernel-launch
end-to-end tests in the `wasi-gpu-hardware-tests` job
(`cargo test -p tensor-wasm-wasi-gpu --release --features cuda -- --ignored
--test-threads=1`): a real `cuLaunchKernel` whose completion future must
resolve with the correct readback proves the waker fired correctly.

### 5. Explicit device-memory host functions (cuMemAlloc / memcpy)

**What it does.** `tensor-wasm-wasi-gpu::device_mem` implements the
`wasi:cuda` device-buffer surface — `alloc` / `free` / `memcpy-h2d` /
`memcpy-d2h` — backed by `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, and
`cuMemcpyDtoH`. This is the explicit device-only allocation path (as opposed to
the UVM path that pointer kernel-args use today), called out as out-of-scope
for v0.2 in [`RISKS.md` — "Kernel-args marshalling"](RISKS.md#kernel-args-marshalling-v02).

**Why it can't be CI-verified without hardware.** On no-CUDA / stub builds the
host functions only validate arguments and return a handle — **no real
`cuMemAlloc` runs and no bytes move**. Proving that a host-to-device copy
followed by a device-to-host copy round-trips the same bytes requires the
driver to actually allocate device memory and perform the DMA, which the stub
cannot do. The end-to-end assertions are `#[ignore]`.

**How the GPU CI lane validates it.** The `wasi-gpu-hardware-tests` job
(`--features cuda -- --ignored`) drives the `alloc → memcpy-h2d → kernel →
memcpy-d2h → readback` path against the real driver and asserts the bytes
survive the round-trip.

### 6. try_grow_in_place via cuMemAddressReserve

**What it does.** `UnifiedBuffer::try_grow_in_place` is the intended
zero-copy grow path: reserve a virtual range up front with
`cuMemAddressReserve`, then `cuMemMap` additional physical pages into the
reserved range as the Wasm linear memory grows — so a `memory.grow` does not
force a realloc-and-copy of the whole buffer.

**Why it can't be CI-verified without hardware.** The method is currently a
**scaffold that returns a documented sentinel error** (`try_grow_in_place`
errors with "in-place grow not yet wired", and `supports_grow_in_place()`
reports `false`); the only unignored test asserts that sentinel. The
`cuMemAddressReserve` + `cuMemMap` virtual-memory API can only be observed to
actually map pages into a reserved range — and to keep the existing data
valid across the grow — on a real driver. The host-only test deliberately
checks the *not-wired* sentinel, not the real behaviour.

**How the GPU CI lane validates it.** Once the `cuMemAddressReserve` path
lands, its real-grow assertion becomes an `#[ignore]`d test under
`tensor-wasm-mem` and is picked up by the `mem-hardware-tests` job's
`--ignored` run (under whichever backend feature the implementation targets,
e.g. `unified-memory` or `gpu-mem-pool`).

### 7. Experimental wmma MatMul lowering (opt-in)

**What it does.** `tensor-wasm-jit::ptx_emit` can lower a
`TensorWasmOp::MatMul { m: 16, n: 16, k: 16 }` to a Tensor-Core
`wmma.mma.sync.aligned.row.col.m16n16k16` fragment-load → `wmma.mma.sync` →
fragment-store sequence (sm_80+). It is **opt-in**: by default `MatMul` is
refused with `EmitError::NotYetImplemented`; the lowering only fires when
`EmitConfig::enable_experimental_matmul` is set to `true`.

**Why it can't be CI-verified without hardware.** The emitter is gated off by
default precisely because a semantically-broken wmma block would silently
corrupt GPU state, and the fragment-load / accumulator-chaining / leading-
dimension contract is only checkable by *running* the emitted PTX on a
Tensor-Core GPU and comparing the result against the CPU reference. Host CI can
assert the PTX *text* is well-formed, but not that it computes the right matrix
product. (Note the dev-box caveat: the RTX 2060 is SM_75 and cannot exercise
the sm_80 wmma blueprint at all — per
[`CUDA-SETUP.md` — "The SM_75 caveat"](CUDA-SETUP.md#the-sm_75-caveat-in-detail),
wmma kernels must be validated against the SM_89 runner, not a dev box.)

**How the GPU CI lane validates it.** On the SM_89 self-hosted runner, the
experimental-matmul end-to-end test (an `#[ignore]`d test that emits the wmma
PTX, launches it, and asserts bit-/tolerance-correctness against the CPU
oracle) runs under the `wasi-gpu-hardware-tests` job's `--features cuda
--ignored` pass. This is the only configuration that can confirm the lowering
is correct, not merely well-formed.

### 8. cuda-oxide host backend (compile_error-guarded)

**What it does.** `experimental-cuda-oxide-host-backend` is the
**strict-superset** sibling of `cuda-oxide-backend` (the W3.3
`pliron-llvm-backend` pattern). It pulls in the four cuda-oxide host crates
(`cuda-host`, `cuda-core`, `cuda-device`, `cuda-macros`) and the
`host_backend` module that maps `CudaOxideUnifiedBuffer` onto real
`cuMemAllocManaged` / `cuMemPrefetchAsync` / `cuMemAdvise` / `cuMemFree_v2`
calls — the v0.4 cuda-oxide parity port.

**Why it can't be CI-verified without hardware.** The module
**intentionally does not build**: it opens with a `compile_error!` because
the host FFI surface is *inferred from cuda-oxide docs and unverified*. Beyond
the toolchain requirements (a CUDA Toolkit and a `libclang` for the
`cuda-bindings` bindgen step, which the default contributor box lacks), the
`compile_error!` is a deliberate tripwire: the inferred FFI signatures must be
checked against the real headers before the code is allowed to compile. The
guard is lifted only once the self-hosted runner has actually compiled and
validated the port.

**How the GPU CI lane validates it.** This item is **two steps away** from the
current `gpu.yml`. First, the `compile_error!` is lifted on the self-hosted
box once the inferred FFI is verified; then the `mem-hardware-tests` job's
cuda-oxide step is switched from `--features cuda-oxide-backend` to
`--features experimental-cuda-oxide-host-backend` so its `#[ignore]`d
round-trip tests (in `tests/cuda_oxide_smoke.rs` /
`tests/cuda_oxide_snapshot_conformance.rs`) execute against the real host
runtime. Until then it is the least-verified item on this list: it does not
even compile on hosted CI.

---

## Cross-references

- [`.github/workflows/gpu.yml`](../.github/workflows/gpu.yml) — the
  hardware-gated GPU CI lane that validates every item above.
- [`.github/workflows/cuda.yml`](../.github/workflows/cuda.yml) — the
  per-push self-hosted build + unignored-smoke lane (sibling, not a
  replacement).
- [`CUDA-SETUP.md`](CUDA-SETUP.md) — toolkit / driver / feature-flag matrix
  and the "Stub libraries for CI" explanation of why these tests are ignored
  on hosted runners.
- [`RISKS.md`](RISKS.md) — the risk register; the `cust` EOL, kernel-args, and
  GPU-quota rows reference items here.
- [`CUDARC-SPIKE.md`](CUDARC-SPIKE.md) — the cust → cudarc migration spike
  behind items 2 and 3.
- [`GPU-QUOTAS.md`](GPU-QUOTAS.md) — the per-tenant quota work whose v0.4
  driver-level enforcement is hardware-gated.
- [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md)
  — the cuda-oxide adoption RFC behind items 3 and 8.
- [`runbooks/self-hosted-cuda-runner.md`](runbooks/self-hosted-cuda-runner.md)
  — how to register the runner that makes the lane non-dormant.
