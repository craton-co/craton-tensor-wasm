// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Capability-gated GPU byte accounting
//! (`TenantContext::consume_gpu_bytes_with_capability` /
//! `release_gpu_bytes_with_capability`): the GPU counter now gets the same
//! cross-tenant isolation guarantee the CPU path has. A `TenantCapability`
//! minted for a foreign tenant is rejected with `TenantIsolationViolation` and
//! leaves the counter untouched; the owning tenant's capability is accepted.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{TenantContext, TenantRegistry};

#[test]
fn gpu_consume_accepts_owner_rejects_foreign() {
    let (reg, _admin) = TenantRegistry::new();

    // Two tenants, each with its own capability, registered in the same
    // registry so the strict-cap-binding token (when enabled) matches.
    let (ctx_a, cap_a) = reg
        .register_with_capability(
            TenantContext::builder(TenantId(1))
                .with_gpu_memory_bytes_cap(1 << 20)
                .build(),
        )
        .unwrap();
    let (_ctx_b, cap_b) = reg
        .register_with_capability(
            TenantContext::builder(TenantId(2))
                .with_gpu_memory_bytes_cap(1 << 20)
                .build(),
        )
        .unwrap();

    // Owner's capability is accepted and moves the counter.
    ctx_a
        .consume_gpu_bytes_with_capability(&cap_a, 4096)
        .unwrap();
    assert_eq!(ctx_a.gpu_bytes_in_use(), 4096);

    // Tenant B's capability presented against tenant A's context is rejected
    // and must NOT move A's counter.
    let err = ctx_a
        .consume_gpu_bytes_with_capability(&cap_b, 1024)
        .expect_err("foreign capability must be rejected");
    match err {
        TensorWasmError::TenantIsolationViolation { tenant_id, .. } => {
            // The offender reported is the capability's tenant (B).
            assert_eq!(tenant_id, TenantId(2));
        }
        other => panic!("expected TenantIsolationViolation, got {other:?}"),
    }
    assert_eq!(
        ctx_a.gpu_bytes_in_use(),
        4096,
        "rejected foreign consume must not perturb the counter"
    );
}

#[test]
fn gpu_release_accepts_owner_rejects_foreign() {
    let (reg, _admin) = TenantRegistry::new();
    let (ctx_a, cap_a) = reg
        .register_with_capability(TenantContext::builder(TenantId(10)).build())
        .unwrap();
    let (_ctx_b, cap_b) = reg
        .register_with_capability(TenantContext::builder(TenantId(11)).build())
        .unwrap();

    ctx_a
        .consume_gpu_bytes_with_capability(&cap_a, 8192)
        .unwrap();
    assert_eq!(ctx_a.gpu_bytes_in_use(), 8192);

    // Foreign capability cannot release A's GPU bytes.
    let err = ctx_a
        .release_gpu_bytes_with_capability(&cap_b, 4096)
        .expect_err("foreign capability must be rejected on release");
    assert!(matches!(
        err,
        TensorWasmError::TenantIsolationViolation { .. }
    ));
    assert_eq!(
        ctx_a.gpu_bytes_in_use(),
        8192,
        "rejected foreign release must not move the counter"
    );

    // Owner's capability releases as expected.
    ctx_a
        .release_gpu_bytes_with_capability(&cap_a, 4096)
        .unwrap();
    assert_eq!(ctx_a.gpu_bytes_in_use(), 4096);
}

#[test]
fn gpu_consume_with_capability_still_enforces_cap() {
    // The capability gate is orthogonal to the byte cap: an over-cap consume
    // by the rightful owner is still refused with GpuMemoryExhausted.
    let (reg, _admin) = TenantRegistry::new();
    let (ctx, cap) = reg
        .register_with_capability(
            TenantContext::builder(TenantId(20))
                .with_gpu_memory_bytes_cap(1024)
                .build(),
        )
        .unwrap();
    let err = ctx
        .consume_gpu_bytes_with_capability(&cap, 2048)
        .expect_err("over-cap consume must be refused even with a valid capability");
    assert!(matches!(err, TensorWasmError::GpuMemoryExhausted { .. }));
    assert_eq!(ctx.gpu_bytes_in_use(), 0);
}
