// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Admission-control regression tests for the per-tenant fairness cap
//! [`EngineConfig::max_instances_per_tenant`].
//!
//! Three invariants are pinned here:
//!   1. Tenant A hitting its per-tenant cap is refused with the dedicated
//!      [`ExecError::TenantCapacityExhausted`] (carrying the offending tenant;
//!      distinct from the engine-wide [`ExecError::CapacityExhausted`], so the
//!      API layer can surface a 429 `tenant_capacity_exhausted` instead of a
//!      generic 503) — WITHOUT affecting tenant B, who can still spawn freely
//!      up to the same per-tenant ceiling.
//!   2. A *failed* spawn (invalid wasm) for a tenant must roll BOTH the
//!      engine-wide and the per-tenant count back, so the tenant is not
//!      silently denied its own cap by submitting garbage in a loop.
//!   3. Terminating a tenant's instance frees its per-tenant slot, re-opening
//!      admission for that tenant.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// Trivial valid module with a single no-op export.
fn trivial_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).expect("valid wat")
}

/// Build an executor with a per-tenant cap (and a generous engine-wide cap so
/// the engine-wide ceiling never masks the per-tenant one).
fn make_executor(max_per_tenant: usize) -> TensorWasmExecutor {
    let cfg = EngineConfig {
        max_instances: Some(1_000),
        max_instances_per_tenant: Some(max_per_tenant),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    TensorWasmExecutor::new(engine)
}

#[tokio::test]
async fn tenant_a_at_cap_does_not_block_tenant_b() {
    // Per-tenant cap of 2.
    let exec = make_executor(2);
    let wasm = trivial_wasm();
    let a = TenantId(1);
    let b = TenantId(2);

    // Tenant A fills its cap (2 instances).
    let a1 = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect("A first spawn admitted");
    let a2 = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect("A second spawn admitted");
    assert_eq!(exec.tenant_instance_count(a), 2, "A holds two slots");

    // A's third spawn is refused at its per-tenant cap. The per-tenant case
    // surfaces the dedicated `TenantCapacityExhausted` variant (carrying the
    // offending tenant) — distinct from the engine-wide `CapacityExhausted`.
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect_err("A third spawn must be refused at its per-tenant cap");
    match err {
        ExecError::TenantCapacityExhausted {
            tenant,
            active,
            limit,
        } => {
            assert_eq!(tenant, a, "rejection must name the offending tenant");
            assert_eq!(limit, 2, "limit must be A's per-tenant cap");
            assert!(active > limit, "active={active} must exceed limit={limit}");
        }
        other => panic!("expected TenantCapacityExhausted, got {other:?}"),
    }
    // The refused spawn must not have leaked A's count or the engine-wide one.
    assert_eq!(exec.tenant_instance_count(a), 2, "refused spawn leaks no A slot");
    assert_eq!(exec.instances_len(), 2, "refused spawn leaks no engine slot");

    // CRUCIAL: tenant B is completely unaffected — it can still spawn up to
    // its own per-tenant cap even though A is saturated.
    let b1 = exec
        .spawn_instance(SpawnConfig::for_tenant(b), &wasm)
        .await
        .expect("B first spawn admitted despite A at cap");
    let b2 = exec
        .spawn_instance(SpawnConfig::for_tenant(b), &wasm)
        .await
        .expect("B second spawn admitted despite A at cap");
    assert_eq!(exec.tenant_instance_count(b), 2, "B holds two slots");

    // Terminating one of A's instances re-opens A's admission.
    exec.terminate(a1).await.expect("terminate a1");
    assert_eq!(exec.tenant_instance_count(a), 1, "A slot freed on terminate");
    let a3 = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect("A spawn admitted again after freeing a slot");

    // Clean up.
    for id in [a2, a3, b1, b2] {
        exec.terminate(id).await.expect("terminate");
    }
    assert_eq!(exec.instances_len(), 0);
    assert_eq!(exec.tenant_instance_count(a), 0);
    assert_eq!(exec.tenant_instance_count(b), 0);
}

#[tokio::test]
async fn failed_spawn_rolls_back_per_tenant_count() {
    // Per-tenant cap of 2. One valid spawn, then a FAILED spawn (garbage
    // wasm) for the same tenant must roll its per-tenant slot back so a
    // subsequent valid spawn is still admitted under the cap.
    let exec = make_executor(2);
    let wasm = trivial_wasm();
    let a = TenantId(7);

    let a1 = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect("first valid spawn admitted");
    assert_eq!(exec.tenant_instance_count(a), 1);

    // Garbage wasm: compilation fails AFTER the admission slot (both
    // engine-wide and per-tenant) is charged. Both must roll back.
    let garbage: &[u8] = b"\x00not-a-wasm-module";
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(a), garbage)
        .await
        .expect_err("invalid wasm must fail to spawn");
    assert!(
        matches!(err, ExecError::Wasmtime(_)),
        "expected a wasmtime compile error, got {err:?}",
    );

    // The failed spawn must have released A's per-tenant slot — back to 1.
    assert_eq!(
        exec.tenant_instance_count(a),
        1,
        "failed spawn must roll back its per-tenant slot",
    );
    assert_eq!(exec.instances_len(), 1, "failed spawn must roll back engine slot");

    // Proof the per-tenant cap was not leaked: a SECOND valid spawn (filling
    // A's last slot) must still be admitted. If the failed spawn had leaked a
    // slot, A would be at its cap of 2 and this would be refused.
    let a2 = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect("valid spawn after a failed one must still be admitted");
    assert_eq!(exec.tenant_instance_count(a), 2);

    exec.terminate(a1).await.expect("terminate a1");
    exec.terminate(a2).await.expect("terminate a2");
    assert_eq!(exec.tenant_instance_count(a), 0);
    assert_eq!(exec.instances_len(), 0);
}

#[tokio::test]
async fn no_per_tenant_cap_means_unlimited_per_tenant() {
    // With `max_instances_per_tenant = None`, a single tenant may spawn up to
    // the engine-wide cap with no per-tenant interference, and the per-tenant
    // counter stays at 0 (the map is never touched).
    let cfg = EngineConfig {
        max_instances: Some(5),
        max_instances_per_tenant: None,
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let wasm = trivial_wasm();
    let a = TenantId(1);

    let mut ids = Vec::new();
    for _ in 0..5 {
        ids.push(
            exec.spawn_instance(SpawnConfig::for_tenant(a), &wasm)
                .await
                .expect("spawn admitted under engine-wide cap"),
        );
    }
    // Per-tenant counter is untouched when the cap is disabled.
    assert_eq!(exec.tenant_instance_count(a), 0);
    // The engine-wide cap still bites.
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(a), &wasm)
        .await
        .expect_err("sixth spawn must hit the engine-wide cap");
    assert!(matches!(err, ExecError::CapacityExhausted { .. }));

    for id in ids {
        exec.terminate(id).await.expect("terminate");
    }
}
