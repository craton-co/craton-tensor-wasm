// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! CI-runnable coverage for the CUDA backends' previously hardware-only
//! logic paths, via the `mock-cuda` test-double seam.
//!
//! Gated on `--features mock-cuda` (wired through a `[[test]]` with
//! `required-features = ["mock-cuda"]` in `Cargo.toml`). These tests run
//! on a host with NO GPU: they drive the `tensor_wasm_mem::mock_cuda`
//! doubles to exercise three branches that real `cuMemAllocManaged` /
//! `cuMemAllocFromPoolAsync` never trip in host-only CI:
//!
//! 1. Allocation-failure ROLLBACK — the `release_gpu_bytes` undo branch
//!    of `UnifiedBuffer::new_with_visible_window_on_with_tenant_context`.
//! 2. Free-on-drop / leak-recording `Drop` — the discipline mirrored
//!    from `CudarcUnifiedBuffer::drop`.
//! 3. Tenant-pool free-on-drop — `TenantPoolBacking::drop` ->
//!    `TenantMemPool::deallocate`.
#![cfg(feature = "mock-cuda")]

use std::sync::Arc;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::mem_pool::DriverMemPool;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_mem::mock_cuda::{
    mock_alloc_with_tenant_context, FreeLog, MockDriverMemPool, MockTenantPoolBuffer,
    MockUnifiedBacking,
};
use tensor_wasm_mem::unified::{UnifiedBacking, UnifiedError};
use tensor_wasm_tenant::TenantContext;

fn capped_ctx(cap: u64) -> Arc<TenantContext> {
    Arc::new(
        TenantContext::builder(TenantId(99))
            .with_gpu_memory_bytes_cap(cap)
            .build(),
    )
}

/// Branch 1: allocation-failure rollback. A failed driver allocation must
/// undo the `consume_gpu_bytes` reservation so the tenant GPU counter
/// returns to its pre-allocation value.
#[test]
fn alloc_failure_rolls_back_tenant_gpu_counter() {
    let ctx = capped_ctx(8192);
    let log = FreeLog::new();

    // First, a successful allocation to put the counter above zero, so a
    // failed-then-rolled-back second allocation is visibly distinct from
    // "never consumed".
    let ok = mock_alloc_with_tenant_context(2048, ctx.clone(), log.clone(), false)
        .expect("first alloc succeeds");
    assert_eq!(ctx.gpu_bytes_in_use(), 2048);

    // Now inject a failure. The rollback branch must fire.
    let err = mock_alloc_with_tenant_context(4096, ctx.clone(), log.clone(), true)
        .expect_err("injected failure must surface as error");
    assert!(matches!(err, TensorWasmError::Serialization(_)));

    // Counter is back to just the first (live) allocation — the failed
    // 4096 was rolled back, not left ratcheting the counter up.
    assert_eq!(
        ctx.gpu_bytes_in_use(),
        2048,
        "failed allocation must be rolled back off the tenant GPU counter"
    );

    // The failed allocation constructed no backing, so no free recorded.
    assert_eq!(log.frees(), 1, "only the successful backing exists so far");
    drop(ok);
    assert_eq!(log.frees(), 2, "the successful backing freed on drop");
}

/// Branch 2: free-on-drop / leak-recording `Drop`. The mock backing must
/// record exactly one free when it drops, and zero before.
#[test]
fn drop_records_exactly_one_free() {
    let log = FreeLog::new();
    let mut backing = MockUnifiedBacking::alloc(512, log.clone());
    backing.as_mut_slice().fill(0xAB);
    assert!(backing.as_slice().iter().all(|&b| b == 0xAB));
    assert_eq!(log.frees(), 0);
    drop(backing);
    assert_eq!(log.frees(), 1, "drop must record exactly one free");
}

/// Branch 3: tenant-pool free-on-drop. A buffer allocated from a mock
/// driver mem-pool must free back through the pool on drop.
#[test]
fn tenant_pool_buffer_frees_through_pool_on_drop() {
    let log = FreeLog::new();
    let pool = Arc::new(MockDriverMemPool::new(log.clone()));
    {
        let buf = MockTenantPoolBuffer::new(pool.clone(), 1024).expect("pool alloc");
        assert_eq!(buf.len(), 1024);
        assert!(!buf.is_empty());
        assert_eq!(log.frees(), 0, "no free while buffer is live");
    }
    assert_eq!(log.frees(), 1, "tenant-pool buffer freed on drop");
}

/// Tenant-pool over-cap allocation failure surfaces as `UnifiedError`
/// and records no free (the symmetric guard for branch 3's free path).
#[test]
fn tenant_pool_over_cap_alloc_fails_without_free() {
    let log = FreeLog::new();
    let pool = Arc::new(MockDriverMemPool::new(log.clone()));
    pool.set_alloc_failure(true);
    let err =
        MockTenantPoolBuffer::new(pool.clone(), 4096).expect_err("over-cap alloc must fail");
    assert!(matches!(err, UnifiedError::Cuda(_)));
    assert_eq!(log.frees(), 0);
}

/// The mock pool is a real `DriverMemPool` so the tenant driver-cap path
/// (`with_driver_enforced_gpu_cap`) can drive it end-to-end. Pinning a
/// cap and reading it back guards the trait wiring.
#[test]
fn mock_pool_drives_driver_enforced_cap_path() {
    let log = FreeLog::new();
    let pool = Arc::new(MockDriverMemPool::new(log));
    let ctx = TenantContext::builder(TenantId(100))
        .with_gpu_memory_bytes_cap(4096)
        .with_driver_enforced_gpu_cap(pool.clone())
        .build();

    // Consuming under the cap pins the driver release threshold to the
    // cap via the `DriverMemPool` trait — exercising the
    // `consume_gpu_bytes` -> `set_release_threshold` integration with a
    // hardware-free pool.
    ctx.consume_gpu_bytes(1024).expect("under cap");
    let pool_dyn: Arc<dyn DriverMemPool> = pool;
    assert_eq!(
        pool_dyn.release_threshold(),
        Some(4096),
        "driver cap pinned to the tenant cap on consume"
    );
}
