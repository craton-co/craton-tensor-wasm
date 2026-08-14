// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Hardware-free test doubles for the CUDA backings (`mock-cuda` feature).
//!
//! # Why this module exists
//!
//! Three logic paths in this crate only ever ran behind `#[ignore]`'d
//! GPU-hardware tests, so CI never exercised them:
//!
//! 1. **Tenant-accounting rollback** — the `Err` arm of
//!    [`crate::unified::UnifiedBuffer::new_with_visible_window_on_with_tenant_context`]
//!    that calls [`tensor_wasm_tenant::TenantContext::release_gpu_bytes`]
//!    to undo a `consume_gpu_bytes` reservation when the underlying
//!    driver allocation fails. Real `cuMemAllocManaged` never fails in
//!    host-only CI (there is no driver), so the rollback branch was
//!    dead in coverage.
//! 2. **Leak-recording / tenant-release `Drop`** — the `Drop` impl on
//!    `UnifiedBuffer` that returns `size` bytes to the tenant, and the
//!    free-on-drop discipline mirrored from
//!    [`crate::cudarc_backend::CudarcUnifiedBuffer::drop`].
//! 3. **Tenant-pool free path** — `TenantPoolBacking::drop` →
//!    `TenantMemPool::deallocate` (`cuMemFreeAsync`), the symmetric free
//!    for a `cuMemAllocFromPoolAsync` allocation.
//!
//! This module provides:
//!
//! - [`MockUnifiedBacking`] — a [`crate::unified::UnifiedBacking`]
//!   implementor backed by a host `Vec<u8>`, with an injectable
//!   allocation failure ([`MockUnifiedBacking::try_alloc`]) and a `Drop`
//!   that records the free into a per-instance [`FreeLog`] so a test can
//!   assert the buffer was freed exactly once.
//! - [`MockDriverMemPool`] — a [`tensor_wasm_core::mem_pool::DriverMemPool`]
//!   implementor that simulates `cuMemAllocFromPoolAsync` /
//!   `cuMemFreeAsync` with an injectable allocation failure and a free
//!   log, so the tenant-pool free-on-drop path can run without a driver.
//! - [`mock_alloc_with_tenant_context`] — mirrors the production
//!   consume → allocate → (rollback on failure) control flow against the
//!   real [`tensor_wasm_tenant::TenantContext`] so the rollback branch is
//!   exercised deterministically.
//! - [`MockTenantPoolBuffer`] — a buffer-shaped wrapper whose `Drop`
//!   frees through a [`MockDriverMemPool`], modelling
//!   `TenantPoolBacking::drop`.
//!
//! Everything here is gated behind `#[cfg(feature = "mock-cuda")]` and is
//! off by default; it pulls in no new dependencies. The feature exists
//! purely to make the above branches CI-runnable on a host with no GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tensor_wasm_core::mem_pool::{DriverMemPool, MemPoolError};
use tensor_wasm_tenant::TenantContext;

use crate::unified::{UnifiedBacking, UnifiedError, UvmAdvice};

/// Records free events for a mock allocation so a test can assert the
/// buffer was freed exactly once on `Drop`.
///
/// Cheap to clone (it is an `Arc<AtomicUsize>` inside): a test keeps one
/// clone and hands another to the mock, then reads [`FreeLog::frees`]
/// after dropping the mock.
#[derive(Debug, Clone, Default)]
pub struct FreeLog {
    frees: Arc<AtomicUsize>,
    /// Count of FAILED frees (leaks). Models the real backings'
    /// [`crate::cudarc_backend::cudarc_free_failures`] counter: a free that
    /// the driver rejected leaves the allocation orphaned, a
    /// security-relevant event the audit surface must count. Incremented by
    /// the mock `Drop` paths when a free failure is injected (finding 7).
    leaks: Arc<AtomicUsize>,
}

impl FreeLog {
    /// A fresh log with a zero free count.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of free events recorded so far.
    pub fn frees(&self) -> usize {
        self.frees.load(Ordering::Acquire)
    }

    /// Number of FAILED frees (leaks) recorded so far. Mirrors
    /// [`crate::cudarc_backend::cudarc_free_failures`] for the mock seam.
    pub fn leaks(&self) -> usize {
        self.leaks.load(Ordering::Acquire)
    }

    /// Record one successful free event. Called by the mock backings' `Drop`
    /// impls.
    fn record_free(&self) {
        self.frees.fetch_add(1, Ordering::AcqRel);
    }

    /// Record one FAILED free (a leak). Called by a mock `Drop` when a free
    /// failure is injected, modelling the real backings' leak accounting.
    fn record_leak(&self) {
        self.leaks.fetch_add(1, Ordering::AcqRel);
    }
}

/// A [`UnifiedBacking`] test double backed by a host `Vec<u8>`.
///
/// Behaves like a successful UVM allocation for the read/write surface,
/// but its `Drop` records a free into the supplied [`FreeLog`] (modelling
/// the leak-recording / free-on-drop discipline of the real CUDA
/// backings) and its construction can be made to FAIL on demand
/// ([`Self::try_alloc`]) so the allocation-failure rollback branch can be
/// driven without a GPU.
#[derive(Debug)]
pub struct MockUnifiedBacking {
    bytes: Vec<u8>,
    free_log: FreeLog,
    /// When `true`, the `Drop` impl simulates a FAILED `cuMemFree_v2`:
    /// instead of recording a successful free it records a leak (finding 7),
    /// modelling the real backings' free-failure / leak-recording path.
    fail_free: bool,
}

impl MockUnifiedBacking {
    /// Allocate a mock backing of `size` bytes, zero-initialised, that
    /// records its free into `free_log` on `Drop`.
    ///
    /// Mirrors a successful `cuMemAllocManaged`: returns `Ok`.
    pub fn alloc(size: usize, free_log: FreeLog) -> Self {
        Self {
            bytes: vec![0u8; size],
            free_log,
            fail_free: false,
        }
    }

    /// Like [`Self::alloc`], but the buffer's `Drop` will simulate a FAILED
    /// free (`cuMemFree_v2 -> error`): it records a leak into the `FreeLog`
    /// rather than a successful free (finding 7). Used to drive the
    /// drop-failure-observability branch without a GPU.
    pub fn alloc_with_failing_free(size: usize, free_log: FreeLog) -> Self {
        Self {
            bytes: vec![0u8; size],
            free_log,
            fail_free: true,
        }
    }

    /// Allocate, or simulate a driver allocation failure.
    ///
    /// When `fail` is `true` this returns [`UnifiedError::Allocation`]
    /// WITHOUT recording anything in `free_log` — exactly the shape a
    /// failing `cuMemAllocManaged` takes, so the caller's rollback branch
    /// runs. When `fail` is `false` it behaves like [`Self::alloc`].
    pub fn try_alloc(size: usize, free_log: FreeLog, fail: bool) -> Result<Self, UnifiedError> {
        if fail {
            return Err(UnifiedError::Allocation(
                "mock-cuda: injected allocation failure".into(),
            ));
        }
        Ok(Self::alloc(size, free_log))
    }
}

impl Drop for MockUnifiedBacking {
    fn drop(&mut self) {
        // Models the free-on-drop discipline of the real backings. On the
        // happy path a single free is recorded per live allocation (a test
        // asserts `free_log.frees() == 1`). When a free failure is injected
        // (finding 7) the buffer instead records a LEAK — mirroring
        // `CudarcUnifiedBuffer::drop`, which on a failed `cuMemFree_v2` bumps
        // `cudarc_free_failures` and retains the orphaned VA — so a test can
        // assert the leak was observed and counted without a GPU.
        if self.fail_free {
            self.free_log.record_leak();
        } else {
            self.free_log.record_free();
        }
    }
}

impl UnifiedBacking for MockUnifiedBacking {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    fn apply_advice(&self, _hint: UvmAdvice) -> Result<(), UnifiedError> {
        // The mock backing has no driver to advise; mirror the legacy
        // `UnifiedBuffer` no-op contract rather than escalating to
        // `NotSupported`.
        Ok(())
    }

    fn prefetch_to_device(&self, _device_ord: u32) -> Result<(), UnifiedError> {
        Ok(())
    }

    fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        Ok(())
    }
}

/// Mirror of the production consume → allocate → (rollback on failure)
/// control flow in
/// [`crate::unified::UnifiedBuffer::new_with_visible_window_on_with_tenant_context`],
/// driven through a [`MockUnifiedBacking`] so the **rollback branch runs
/// deterministically without a GPU**.
///
/// The steps match production exactly:
///
/// 1. Reject zero-byte requests before touching the tenant counter.
/// 2. Reserve `size` bytes against the tenant cap via
///    [`TenantContext::consume_gpu_bytes`]; propagate
///    `GpuMemoryExhausted` untouched.
/// 3. Attempt the (mock) allocation. On success, hand back a
///    [`MockUnifiedBacking`] whose `Drop` will both record the free and
///    (via the returned wrapper's own drop in the real path) release the
///    tenant bytes. On failure, **roll back the reservation** with
///    [`TenantContext::release_gpu_bytes`] before returning the error —
///    this is the previously-uncovered branch.
///
/// Set `inject_alloc_failure` to `true` to exercise the rollback path.
pub fn mock_alloc_with_tenant_context(
    size: usize,
    tenant_ctx: Arc<TenantContext>,
    free_log: FreeLog,
    inject_alloc_failure: bool,
) -> Result<MockUnifiedBacking, tensor_wasm_core::error::TensorWasmError> {
    if size == 0 {
        return Err(UnifiedError::ZeroSize.into());
    }
    // Step 1: reserve against the cap (or counter-only when no cap).
    tenant_ctx.consume_gpu_bytes(size as u64)?;
    // Step 2: allocate; roll back the reservation on driver failure so
    // the tenant counter does not drift above true utilisation.
    match MockUnifiedBacking::try_alloc(size, free_log, inject_alloc_failure) {
        Ok(backing) => Ok(backing),
        Err(e) => {
            tenant_ctx.release_gpu_bytes(size as u64);
            Err(e.into())
        }
    }
}

/// A [`DriverMemPool`] test double that simulates
/// `cuMemAllocFromPoolAsync` / `cuMemFreeAsync` without a driver.
///
/// - [`Self::allocate`] returns a fresh opaque handle, or an injected
///   [`UnifiedError`] when [`Self::set_alloc_failure`] armed a failure
///   (modelling an over-cap `CUDA_ERROR_OUT_OF_MEMORY`).
/// - [`Self::deallocate`] records a free into the pool's [`FreeLog`],
///   modelling the symmetric `cuMemFreeAsync`.
/// - [`DriverMemPool::set_release_threshold`] records the requested cap
///   so the tenant driver-cap path has a real implementor to push
///   through.
#[derive(Debug)]
pub struct MockDriverMemPool {
    cap_bytes: std::sync::atomic::AtomicU64,
    free_log: FreeLog,
    next_handle: AtomicUsize,
    fail_alloc: std::sync::atomic::AtomicBool,
    /// When set, [`Self::deallocate`] simulates a FAILED `cuMemFreeAsync`:
    /// it records a leak (not a free) into the `FreeLog` and returns an
    /// error (finding 7).
    fail_free: std::sync::atomic::AtomicBool,
}

impl MockDriverMemPool {
    /// A pool with no release threshold pinned yet and an empty free log.
    pub fn new(free_log: FreeLog) -> Self {
        Self {
            cap_bytes: std::sync::atomic::AtomicU64::new(0),
            free_log,
            // Start at 1 so a handle is never the null-equivalent 0.
            next_handle: AtomicUsize::new(1),
            fail_alloc: std::sync::atomic::AtomicBool::new(false),
            fail_free: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Arm or disarm an injected allocation failure for the next
    /// [`Self::allocate`] call(s).
    pub fn set_alloc_failure(&self, fail: bool) {
        self.fail_alloc.store(fail, Ordering::Release);
    }

    /// Arm or disarm an injected FREE failure for the next
    /// [`Self::deallocate`] call(s) (finding 7). A failed free records a
    /// leak rather than a free and returns an error, modelling a failed
    /// `cuMemFreeAsync` that orphans the allocation.
    pub fn set_free_failure(&self, fail: bool) {
        self.fail_free.store(fail, Ordering::Release);
    }

    /// Simulate `cuMemAllocFromPoolAsync`: returns an opaque non-zero
    /// handle, or an injected over-cap failure.
    pub fn allocate(&self, _size: usize) -> Result<usize, UnifiedError> {
        if self.fail_alloc.load(Ordering::Acquire) {
            return Err(UnifiedError::Cuda(
                "mock-cuda: cuMemAllocFromPoolAsync -> CUDA_ERROR_OUT_OF_MEMORY".into(),
            ));
        }
        Ok(self.next_handle.fetch_add(1, Ordering::AcqRel))
    }

    /// Simulate `cuMemFreeAsync`: records a successful free into the pool's
    /// log, or — when a free failure is armed (finding 7) — records a leak
    /// and returns an error, modelling a failed free that orphans the
    /// allocation.
    pub fn deallocate(&self, _handle: usize) -> Result<(), UnifiedError> {
        if self.fail_free.load(Ordering::Acquire) {
            self.free_log.record_leak();
            return Err(UnifiedError::Cuda(
                "mock-cuda: cuMemFreeAsync -> CUDA_ERROR_INVALID_VALUE (leaked)".into(),
            ));
        }
        self.free_log.record_free();
        Ok(())
    }
}

impl DriverMemPool for MockDriverMemPool {
    fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError> {
        self.cap_bytes.store(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn release_threshold(&self) -> Option<u64> {
        Some(self.cap_bytes.load(Ordering::Relaxed))
    }
}

/// Buffer-shaped wrapper modelling `TenantPoolBacking`: holds an
/// `Arc<MockDriverMemPool>` and a handle, and frees through the pool on
/// `Drop` (modelling `TenantPoolBacking::drop` →
/// `TenantMemPool::deallocate`).
///
/// A `cuMemFreeAsync` failure cannot be returned from `Drop`, so — like
/// the production path — a failing `deallocate` is swallowed (the real
/// path logs at `error!`); the [`FreeLog`] still reflects only successful
/// frees, which is what a test asserts.
#[derive(Debug)]
pub struct MockTenantPoolBuffer {
    pool: Arc<MockDriverMemPool>,
    handle: usize,
    size: usize,
}

impl MockTenantPoolBuffer {
    /// Allocate `size` bytes from `pool`, surfacing an injected
    /// over-cap failure through [`UnifiedError`].
    pub fn new(pool: Arc<MockDriverMemPool>, size: usize) -> Result<Self, UnifiedError> {
        let handle = pool.allocate(size)?;
        Ok(Self { pool, handle, size })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for MockTenantPoolBuffer {
    fn drop(&mut self) {
        // Mirrors `TenantPoolBacking::drop`: free through the pool and
        // swallow (log, in production) any failure since `Drop` cannot
        // return an error.
        if let Err(e) = self.pool.deallocate(self.handle) {
            tracing::error!(
                target: "tensor_wasm_mem::mock_cuda",
                error = ?e,
                "mock cuMemFreeAsync failed in MockTenantPoolBuffer::drop",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_wasm_core::types::TenantId;
    use tensor_wasm_tenant::TenantContext;

    fn capped_ctx(cap: u64) -> Arc<TenantContext> {
        Arc::new(
            TenantContext::builder(TenantId(1))
                .with_gpu_memory_bytes_cap(cap)
                .build(),
        )
    }

    #[test]
    fn mock_backing_records_free_once_on_drop() {
        let log = FreeLog::new();
        {
            let mut b = MockUnifiedBacking::alloc(64, log.clone());
            assert_eq!(b.len(), 64);
            b.as_mut_slice().fill(7);
            assert!(b.as_slice().iter().all(|&v| v == 7));
            assert_eq!(log.frees(), 0, "no free before drop");
        }
        assert_eq!(log.frees(), 1, "exactly one free recorded on drop");
    }

    #[test]
    fn rollback_restores_tenant_counter_on_alloc_failure() {
        let ctx = capped_ctx(4096);
        let log = FreeLog::new();
        let err = mock_alloc_with_tenant_context(1024, ctx.clone(), log.clone(), true)
            .expect_err("injected alloc failure must surface");
        // Rollback branch ran: the consume was undone.
        assert_eq!(
            ctx.gpu_bytes_in_use(),
            0,
            "tenant GPU counter must be rolled back after alloc failure"
        );
        // No backing was constructed, so no free was recorded.
        assert_eq!(log.frees(), 0);
        assert!(matches!(
            err,
            tensor_wasm_core::error::TensorWasmError::Serialization(_)
        ));
    }

    #[test]
    fn success_path_consumes_then_drop_does_not_double_free() {
        let ctx = capped_ctx(4096);
        let log = FreeLog::new();
        let backing = mock_alloc_with_tenant_context(1024, ctx.clone(), log.clone(), false)
            .expect("alloc should succeed");
        assert_eq!(ctx.gpu_bytes_in_use(), 1024, "consume recorded");
        drop(backing);
        assert_eq!(log.frees(), 1, "backing freed exactly once");
    }

    #[test]
    fn tenant_pool_frees_on_drop() {
        let log = FreeLog::new();
        let pool = Arc::new(MockDriverMemPool::new(log.clone()));
        {
            let buf = MockTenantPoolBuffer::new(pool.clone(), 256).expect("pool alloc");
            assert_eq!(buf.len(), 256);
            assert_eq!(log.frees(), 0, "no free before drop");
        }
        assert_eq!(log.frees(), 1, "tenant-pool buffer freed on drop");
    }

    #[test]
    fn tenant_pool_alloc_failure_surfaces_and_records_no_free() {
        let log = FreeLog::new();
        let pool = Arc::new(MockDriverMemPool::new(log.clone()));
        pool.set_alloc_failure(true);
        let err = MockTenantPoolBuffer::new(pool.clone(), 256)
            .expect_err("injected over-cap failure must surface");
        assert!(matches!(err, UnifiedError::Cuda(_)));
        assert_eq!(log.frees(), 0, "failed alloc records no free");
    }

    #[test]
    fn failing_free_records_leak_not_free() {
        // finding 7: a backing whose free fails on drop must record a LEAK
        // (mirroring `cudarc_free_failures`) and NOT a successful free, so
        // the drop-failure-observability gap is closed in CI without a GPU.
        let log = FreeLog::new();
        {
            let b = MockUnifiedBacking::alloc_with_failing_free(128, log.clone());
            assert_eq!(b.len(), 128);
            assert_eq!(log.frees(), 0);
            assert_eq!(log.leaks(), 0, "no leak before drop");
        }
        assert_eq!(log.frees(), 0, "a failed free must NOT count as a free");
        assert_eq!(log.leaks(), 1, "the leaked allocation must be counted");
    }

    #[test]
    fn tenant_pool_failing_free_records_leak_on_drop() {
        // finding 7: the tenant-pool free path's failure branch records a
        // leak when `cuMemFreeAsync` fails. `Drop` swallows the error (it
        // cannot return one), but the leak counter still reflects it.
        let log = FreeLog::new();
        let pool = Arc::new(MockDriverMemPool::new(log.clone()));
        {
            let _buf = MockTenantPoolBuffer::new(pool.clone(), 256).expect("pool alloc");
            pool.set_free_failure(true);
            assert_eq!(log.leaks(), 0, "no leak before drop");
        }
        assert_eq!(log.frees(), 0, "the free failed, so no successful free");
        assert_eq!(log.leaks(), 1, "the failed free was counted as a leak");
    }

    #[test]
    fn mock_pool_is_a_driver_mem_pool() {
        let log = FreeLog::new();
        let pool: Arc<dyn DriverMemPool> = Arc::new(MockDriverMemPool::new(log));
        pool.set_release_threshold(2048).unwrap();
        assert_eq!(pool.release_threshold(), Some(2048));
    }
}
