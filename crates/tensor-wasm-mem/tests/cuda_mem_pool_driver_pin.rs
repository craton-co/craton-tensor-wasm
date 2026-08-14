// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T39 per-tenant memory-cap tests for [`tensor_wasm_mem::cuda_mem_pool`].
//!
//! These tests prove the v0.4 deliverable for PATH-TO-V1.md §3.1 #8 — that the
//! per-tenant GPU memory cap is enforced for allocations routed through the
//! tenant pool, independently of the in-process
//! `TenantContext::consume_gpu_bytes` counter.
//!
//! NOTE (fix #1, see `docs/GPU-VALIDATION-2026-05-30.md` BUG-1): enforcement is
//! HOST-SIDE, inside [`TenantMemPool::allocate`], NOT driver-level.
//! `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, cap)` is only a
//! retention hint — it does not bound allocation size, and a hardware run
//! confirmed a 128 MiB request against a 64 MiB-"capped" pool succeeded when
//! the cap relied on the threshold alone. `allocate` now reserves against the
//! cap before `cuMemAllocFromPoolAsync`, so an over-cap request — including one
//! that bypasses the in-process counter by calling the pool directly, as these
//! tests do — is refused with a `CUDA_ERROR_OUT_OF_MEMORY`-shaped error. The
//! test names retain `_by_driver` for continuity; read them as "refused at the
//! pool layer".
//!
//! Tests in this file:
//!
//! * `over_cap_allocation_through_pool_is_rejected_by_driver` —
//!   construct a pool with `cap = 64 MiB`, ask the driver for 128 MiB
//!   through the pool, assert OOM. The in-process counter is not
//!   touched, so a successful allocation here would prove the v0.3.7
//!   bypass is closed at the driver level.
//! * `under_cap_allocation_through_pool_succeeds` — the symmetric
//!   "happy path" — same pool, ask for 16 MiB, assert success and
//!   `len() == 16 MiB`.
//! * `driver_pin_matches_requested_cap` — verifies the requested cap
//!   round-trips through `TenantMemPool::cap_bytes()`; documents the
//!   v0.4 contract that the driver MAY round internally but our getter
//!   returns the value the operator asked for.
//!
//! Every test is `#[ignore]`d because it requires a working CUDA driver AND
//! the non-default `gpu-mem-pool` feature flag. Run on a hardware-equipped
//! host with:
//!
//! ```text
//! cargo test --features gpu-mem-pool -- --ignored cuda_mem_pool_driver_pin
//! ```
//!
//! HARDWARE NOTE (mem finding 7): this box now HAS a CUDA GPU (RTX 2060,
//! sm_75, CUDA 13.2), so these tests are hardware-runnable today via the
//! command above. They stay `#[ignore]`d by default because they still need
//! the `gpu-mem-pool` feature and a driver, so plain `cargo test` must not
//! attempt them. The host-reachable half of the cap contract (the
//! reservation arithmetic, the `DriverMemPool` trait wiring) is additionally
//! covered WITHOUT hardware by `src/cuda_mem_pool.rs`'s `reserve_step_*` unit
//! tests and the `mock-cuda` `MockDriverMemPool` integration tests.

#![cfg(feature = "gpu-mem-pool")]

use std::sync::Arc;

use tensor_wasm_mem::cuda_mem_pool::TenantMemPool;
use tensor_wasm_mem::unified::{DeviceId, UnifiedBuffer, UnifiedError};

const MIB: u64 = 1024 * 1024;
const CAP_64_MIB: u64 = 64 * MIB;

/// T39 driver-pin: a 128 MiB allocation against a 64 MiB-capped pool
/// must be rejected by the CUDA driver itself.
///
/// This is the v0.4 acceptance test for PATH-TO-V1.md §3.1 #8. We
/// deliberately do NOT route through the in-process `consume_gpu_bytes`
/// counter here — the whole point is to prove the driver itself
/// refuses the over-cap allocation. A pre-T39 build (where the cap
/// lived only in `bytes_in_use` and `cuMemPoolSetAttribute` was never
/// called) would happily allocate the 128 MiB and this assertion would
/// fail, signalling the bypass is still open.
#[test]
#[ignore = "requires CUDA hardware"]
fn over_cap_allocation_through_pool_is_rejected_by_driver() {
    let pool = TenantMemPool::new(0, CAP_64_MIB)
        .expect("cuMemPoolCreate + SetAttribute must succeed on a CUDA host");
    let pool = Arc::new(pool);
    // 128 MiB > 64 MiB cap. `TenantMemPool::allocate` must reject this at the
    // host-side reservation (before `cuMemAllocFromPoolAsync`) with a
    // `CUDA_ERROR_OUT_OF_MEMORY`-shaped `UnifiedError::Cuda(...)`.
    let res = UnifiedBuffer::new_in_tenant_pool(
        Arc::clone(&pool),
        (128 * MIB) as usize,
        DeviceId::default(),
    );
    match res {
        Err(UnifiedError::Cuda(msg)) => {
            // Be lenient on the exact string: the driver's
            // `CUDA_ERROR_OUT_OF_MEMORY` Debug representation differs
            // between cudarc minor versions. Match on the substring
            // most likely to remain stable across upgrades.
            assert!(
                msg.contains("OUT_OF_MEMORY")
                    || msg.contains("OutOfMemory")
                    || msg.contains("CUDA_ERROR"),
                "expected driver OOM-style error, got: {msg}",
            );
        }
        Err(other) => panic!(
            "expected UnifiedError::Cuda(OOM) for an over-cap allocation; \
             got a different error variant: {other:?}. T39 driver pin \
             must surface the driver-level refusal as Cuda(...), not \
             collapse it into Allocation or NotSupported.",
        ),
        Ok(buf) => panic!(
            "T39 REGRESSION: 128 MiB allocation succeeded against a \
             64 MiB-capped pool (len={}). The per-tenant cap is NOT being \
             enforced — the host-side reservation in TenantMemPool::allocate \
             (live_bytes vs cap_bytes) is missing or wrong. NB: \
             CU_MEMPOOL_ATTR_RELEASE_THRESHOLD alone does NOT enforce this \
             (it is a retention hint). Check TenantMemPool::allocate in \
             crates/tensor-wasm-mem/src/cuda_mem_pool.rs.",
            buf.len(),
        ),
    }
}

/// T39 happy-path: a 16 MiB allocation against a 64 MiB-capped pool
/// must succeed and return a buffer of the requested length.
///
/// Belt-and-braces: documents that the cap rejection above is a true
/// cap, not a "the pool refuses all allocations" failure mode (which
/// would also trip the over-cap assertion but mean something
/// completely different).
#[test]
#[ignore = "requires CUDA hardware"]
fn under_cap_allocation_through_pool_succeeds() {
    let pool = TenantMemPool::new(0, CAP_64_MIB)
        .expect("cuMemPoolCreate + SetAttribute must succeed on a CUDA host");
    let pool = Arc::new(pool);
    let buf = UnifiedBuffer::new_in_tenant_pool(
        Arc::clone(&pool),
        (16 * MIB) as usize,
        DeviceId::default(),
    )
    .expect("16 MiB allocation under the 64 MiB cap must succeed");
    assert_eq!(
        buf.len(),
        (16 * MIB) as usize,
        "UnifiedBuffer::new_in_tenant_pool must report the requested \
         size; a mismatch here would mean the buffer's `size` field \
         drifted from the cuMemAllocFromPoolAsync allocation",
    );
    // Drop the buffer; the TenantPoolBacking Drop frees through
    // cuMemFreeAsync, which we cannot directly observe here but a
    // failure would log at error! and (on a tight-budget host) starve
    // the next allocation. The pool itself drops at end-of-scope via
    // cuMemPoolDestroy.
}

/// T39 contract: the cap the operator passed to
/// [`TenantMemPool::new`] is the value
/// [`TenantMemPool::cap_bytes`] reports back.
///
/// The driver may round the threshold internally (per CUDA 12.x
/// docs); we deliberately surface the *requested* value through this
/// getter so the operator's monitoring dashboards align with the
/// configured cap. [`TenantMemPool::effective_cap_bytes`] (mem finding 5)
/// round-trips through `cuMemPoolGetAttribute` for the driver's actual,
/// possibly-rounded, value.
#[test]
#[ignore = "requires CUDA hardware + --features gpu-mem-pool (GPU now present; run with -- --ignored)"]
fn driver_pin_matches_requested_cap() {
    let pool = TenantMemPool::new(0, CAP_64_MIB)
        .expect("cuMemPoolCreate + SetAttribute must succeed on a CUDA host");
    assert_eq!(
        pool.cap_bytes(),
        CAP_64_MIB,
        "TenantMemPool::cap_bytes must round-trip the requested value; \
         a mismatch here means the builder dropped the cap or the \
         getter is reading the wrong field",
    );
    assert_eq!(
        pool.device_ordinal(),
        0,
        "TenantMemPool::device_ordinal must round-trip the requested ordinal",
    );
    assert!(
        !pool.raw_handle().is_null(),
        "raw CUmemoryPool handle must be non-null after successful new()",
    );
    // mem finding 5: the driver-reported effective cap must be readable and
    // at least the requested value (the driver may round UP, never silently
    // below the configured retention threshold).
    let effective = pool
        .effective_cap_bytes()
        .expect("cuMemPoolGetAttribute(RELEASE_THRESHOLD) must succeed on a CUDA host");
    assert!(
        effective >= CAP_64_MIB,
        "driver effective cap {effective} must be >= the requested {CAP_64_MIB}; \
         the driver may round up but must not report below the configured threshold",
    );
}
