// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//
//! Integration tests for the per-tenant GPU memory quota (roadmap
//! feature #8 / step 4 in the scaffold patch).
//!
//! The tests pin the v0.3.7 record-only contract:
//!
//! * `consume_gpu_bytes` succeeds up to and including the cap.
//! * `consume_gpu_bytes` past the cap returns
//!   [`tensor_wasm_core::error::TensorWasmError::GpuMemoryExhausted`]
//!   with the documented `requested` / `limit` / `current` triple.
//! * `release_gpu_bytes` restores headroom proportionally to the bytes
//!   released, and a subsequent `consume_gpu_bytes` of the freed
//!   amount succeeds.
//! * `gpu_memory_bytes_cap == None` ("no cap") allows arbitrary
//!   reservation; the counter is still updated so dashboards see real
//!   utilisation but the request is never refused.
//! * The CAS-loop atomic discipline matches the CPU sibling counter:
//!   a 32-thread stress with balanced consume/release leaves the
//!   counter at the algebraically expected value, never wrapping.
//!
//! These tests deliberately exercise *only* the in-process counter —
//! the CUDA driver-level enforcement via
//! `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`
//! lands in v0.4 (see `docs/GPU-QUOTAS.md`).

use std::sync::Arc;
use std::thread;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::TenantContext;

#[test]
fn consume_at_cap_succeeds() {
    // Allocating exactly the cap is the boundary case; the CAS loop
    // uses `<=` against the limit so this MUST succeed and leave the
    // counter pinned at the cap.
    let ctx = TenantContext::builder(TenantId(1))
        .with_gpu_memory_bytes_cap(1024)
        .build();
    ctx.consume_gpu_bytes(1024)
        .expect("exactly-at-cap allocation must succeed");
    assert_eq!(ctx.gpu_bytes_in_use(), 1024);
}

#[test]
fn consume_past_cap_returns_exhausted() {
    let ctx = TenantContext::builder(TenantId(2))
        .with_gpu_memory_bytes_cap(1024)
        .build();
    // Pre-load the counter with most of the cap so the second
    // allocation is the one that trips the bound. This pins that the
    // failure carries the *current* in-use figure (1000), not zero.
    ctx.consume_gpu_bytes(1000)
        .expect("pre-load below cap must succeed");

    let err = ctx
        .consume_gpu_bytes(100)
        .expect_err("100 + 1000 > 1024 must be refused");
    match err {
        TensorWasmError::GpuMemoryExhausted {
            requested,
            limit,
            current,
        } => {
            assert_eq!(requested, 100, "wrong requested in error: {err:?}");
            assert_eq!(limit, 1024, "wrong limit in error: {err:?}");
            assert_eq!(
                current, 1000,
                "current must reflect pre-rejection state: {err:?}"
            );
        }
        other => panic!("expected GpuMemoryExhausted, got {other:?}"),
    }
    // A failed allocation must NOT move the counter — the CAS would
    // otherwise leave the tenant in a state where the next legitimate
    // allocation is rejected for inflated accounting.
    assert_eq!(ctx.gpu_bytes_in_use(), 1000);
}

#[test]
fn release_restores_headroom() {
    // Consume to the cap, release half, allocate the freed half. This
    // pins the round-trip contract: release on the GPU counter is
    // accounting-symmetric with consume, and headroom is restored
    // immediately (no deferred release in v0.3.7).
    let ctx = TenantContext::builder(TenantId(3))
        .with_gpu_memory_bytes_cap(1024)
        .build();
    ctx.consume_gpu_bytes(1024).unwrap();
    assert!(
        ctx.consume_gpu_bytes(1).is_err(),
        "sanity: cap is hit before release"
    );

    ctx.release_gpu_bytes(512);
    assert_eq!(ctx.gpu_bytes_in_use(), 512);

    ctx.consume_gpu_bytes(512)
        .expect("freed half must be reservable again");
    assert_eq!(ctx.gpu_bytes_in_use(), 1024);
}

#[test]
fn no_cap_allows_unbounded() {
    // The v0.3.7 record-only contract requires `gpu_memory_bytes_cap
    // == None` to be the "operator trust" mode: the counter is bumped
    // (so the per-tenant gauge reports real utilisation on
    // dashboards), but the request is never refused.
    let ctx = TenantContext::builder(TenantId(4)).build();
    assert_eq!(
        ctx.gpu_memory_bytes_cap(),
        None,
        "default builder must leave cap unset"
    );

    // 1 GiB then 8 GiB are both far above any reasonable
    // single-allocation budget; under "no cap" both must succeed.
    ctx.consume_gpu_bytes(1024 * 1024 * 1024).unwrap();
    ctx.consume_gpu_bytes(8 * 1024 * 1024 * 1024).unwrap();
    let total = (1024u64 * 1024 * 1024) + (8u64 * 1024 * 1024 * 1024);
    assert_eq!(ctx.gpu_bytes_in_use(), total);

    // u64::MAX overflow is still rejected — the counter must not
    // wrap silently even under "no cap". This is the same
    // `checked_add` guard the CPU path has on `consume_bytes_inner`.
    let err = ctx
        .consume_gpu_bytes(u64::MAX)
        .expect_err("checked_add overflow must always refuse, even with no cap");
    assert!(matches!(err, TensorWasmError::GpuMemoryExhausted { .. }));
    assert_eq!(
        ctx.gpu_bytes_in_use(),
        total,
        "rejected u64::MAX add must not move the counter"
    );
}

#[test]
fn concurrent_consume_release_no_drift() {
    // 32-thread stress test for the CAS-loop atomic discipline.
    //
    // Each worker thread performs `ITERATIONS` matched consume/release
    // pairs of `STEP` bytes. After all threads join the counter MUST
    // be exactly zero — any drift signals an atomic-ordering bug in
    // the CAS loop (e.g. a `Release` where `AcqRel` was required).
    //
    // The thread count is intentionally higher than the typical
    // CAS-contention sweet spot (4-8 cores) so the test exercises the
    // `compare_exchange_weak` retry loop rather than always landing on
    // the first attempt.
    const THREADS: usize = 32;
    const ITERATIONS: u64 = 1_000;
    const STEP: u64 = 64;

    // Cap is large enough that all 32 threads holding `STEP` bytes
    // each simultaneously stay below it; otherwise consume_gpu_bytes
    // would legitimately refuse under interleaving and the test would
    // be flaky.
    let cap = (THREADS as u64) * STEP * 2;
    let ctx = Arc::new(
        TenantContext::builder(TenantId(0xC0FFEE))
            .with_gpu_memory_bytes_cap(cap)
            .build(),
    );

    let mut handles = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        let ctx = Arc::clone(&ctx);
        handles.push(thread::spawn(move || {
            for _ in 0..ITERATIONS {
                ctx.consume_gpu_bytes(STEP)
                    .expect("matched consume/release must always fit under cap");
                ctx.release_gpu_bytes(STEP);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    assert_eq!(
        ctx.gpu_bytes_in_use(),
        0,
        "balanced consume/release across {THREADS} threads must leave counter at zero",
    );
}
