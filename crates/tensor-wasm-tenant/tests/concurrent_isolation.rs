// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S16 done-when: 10 tenants × 100 MiB + concurrent "kernels" — verify zero
//! cross-tenant data contamination.
//!
//! Without CUDA hardware we model "kernels" as Rust tasks that consume bytes
//! against per-tenant quotas. The invariant under test is the same one the
//! CUDA-hosted version exercises: a tenant's quota counter is mutated by
//! that tenant alone, never by another's concurrent work.

#![allow(deprecated)]
// This concurrency stress test pre-dates the capability gate. It uses
// the unchecked variants to keep the kernel-model code terse; the
// type-system enforcement of the cap gate is covered in the registry
// inline tests. Once v0.4 removes the unchecked variants, this file
// will be rewritten to thread the cap through the task closures.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{IsolationKind, TenantContext, TenantRegistry};

const TENANTS: u64 = 10;
const QUOTA_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB per tenant
const KERNELS_PER_TENANT: u64 = 64;
const BYTES_PER_KERNEL: u64 = 256 * 1024; // 256 KiB

fn make_ctx(id: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_isolation(IsolationKind::StreamIsolated)
        .with_stream_id(id)
        .with_memory_quota_bytes(QUOTA_BYTES)
        .build()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ten_tenants_concurrent_kernels_have_no_contamination() {
    let reg = TenantRegistry::new();
    for i in 0..TENANTS {
        reg.register(make_ctx(i)).expect("register");
    }

    // Per-tenant counter of "kernels launched" for the cross-check below.
    let launched: Vec<Arc<AtomicU64>> = (0..TENANTS).map(|_| Arc::new(AtomicU64::new(0))).collect();

    let mut handles = Vec::new();
    for tenant_id in 0..TENANTS {
        let reg = reg.clone();
        let launched = Arc::clone(&launched[tenant_id as usize]);
        handles.push(tokio::spawn(async move {
            let ctx = reg.get(TenantId(tenant_id)).expect("ctx");
            for _ in 0..KERNELS_PER_TENANT {
                ctx.consume_bytes(BYTES_PER_KERNEL).expect("consume");
                launched.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                ctx.release_bytes(BYTES_PER_KERNEL);
            }
        }));
    }
    for h in handles {
        h.await.expect("task");
    }

    // 1. Every tenant launched the expected number of kernels.
    for (i, count) in launched.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::Relaxed),
            KERNELS_PER_TENANT,
            "tenant {i} kernel count",
        );
    }

    // 2. Each tenant's quota counter is back to zero. If any of the launches
    //    had been mis-routed to another tenant we'd see a non-zero residual
    //    in the wrong place.
    for i in 0..TENANTS {
        let ctx = reg.get(TenantId(i)).expect("ctx");
        assert_eq!(
            ctx.bytes_in_use(),
            0,
            "tenant {i} should have released all bytes; cross-tenant contamination?",
        );
        assert_eq!(ctx.quota(), QUOTA_BYTES, "tenant {i} quota");
    }

    // 3. Registry size is still TENANTS — no spurious registrations.
    assert_eq!(reg.len() as u64, TENANTS);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_overrun_is_contained_to_one_tenant() {
    // One tenant intentionally exceeds its quota; the others must complete
    // their kernels unaffected. Proves the contamination boundary holds even
    // when one tenant is misbehaving.
    let reg = TenantRegistry::new();
    for i in 0..TENANTS {
        reg.register(make_ctx(i)).expect("register");
    }

    // The offender tries to consume 2× its quota; all others stay well below.
    let offender = 3u64;
    let offender_ctx = reg.get(TenantId(offender)).unwrap();
    let offender_handle = tokio::spawn(async move {
        // First fill the quota exactly.
        offender_ctx.consume_bytes(QUOTA_BYTES).unwrap();
        // Next allocation must fail.
        let err = offender_ctx.consume_bytes(1).unwrap_err();
        // Borrow-test: it's `TensorWasmError::MemoryExhausted { .. }`.
        assert!(format!("{err}").contains("memory exhausted"));
    });

    let mut other_handles = Vec::new();
    for tenant_id in 0..TENANTS {
        if tenant_id == offender {
            continue;
        }
        let reg = reg.clone();
        other_handles.push(tokio::spawn(async move {
            let ctx = reg.get(TenantId(tenant_id)).expect("ctx");
            for _ in 0..KERNELS_PER_TENANT {
                ctx.consume_bytes(BYTES_PER_KERNEL).expect("consume");
                tokio::task::yield_now().await;
                ctx.release_bytes(BYTES_PER_KERNEL);
            }
            assert_eq!(ctx.bytes_in_use(), 0);
        }));
    }

    offender_handle.await.expect("offender");
    for h in other_handles {
        h.await.expect("other");
    }

    // Offender is still at quota (it failed to release after overrun).
    let off = reg.get(TenantId(offender)).unwrap();
    assert_eq!(off.bytes_in_use(), QUOTA_BYTES);
    // Everyone else is clean.
    for i in 0..TENANTS {
        if i == offender {
            continue;
        }
        let ctx = reg.get(TenantId(i)).unwrap();
        assert_eq!(ctx.bytes_in_use(), 0, "tenant {i} contaminated");
    }
}
