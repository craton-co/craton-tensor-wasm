// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Strict capability-binding semantics under `--features strict-cap-binding`.
//!
//! In the default (non-strict) build, capabilities minted by
//! [`TenantRegistry::new`] / [`TenantRegistry::register_with_capability`]
//! are interchangeable across registries — a `RegistryAdminCapability`
//! minted by registry A unlocks registry B's admin methods, and the
//! same goes for `TenantCapability` against same-id contexts. That is
//! the surface invariant pinned by
//! `tests/admin_cap_required.rs::independent_constructions_yield_independent_caps`
//! and was the documented v0.3 behaviour.
//!
//! Audit follow-up: deployments that host two trust domains in the same
//! process must be able to refuse cross-registry cap use, because a
//! subsystem that happens to hold `Arc<TenantRegistry>` for the "wrong"
//! registry and a cap from the "right" one could otherwise drive admin
//! ops against the wrong tenant. The `strict-cap-binding` feature flips
//! the semantics: every cap carries an `Arc::ptr_eq`-comparable token
//! pointing at its minting registry's per-instance allocation, and
//! admin / quota methods reject foreign caps explicitly.
//!
//! This test pins the strict-mode contract. It only compiles under the
//! feature; without it, the file is a no-op.

#![cfg(feature = "strict-cap-binding")]

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{
    registry::RegistryError, TenantCapability, TenantContext, TenantRegistry,
};

fn make_ctx(id: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_memory_quota_bytes(4096)
        .build()
}

#[test]
fn admin_cap_from_foreign_registry_is_rejected_via_strict_api() {
    // Two independent registries. Under strict binding, each cap is
    // tied to its registry by `Arc::ptr_eq` on a per-instance allocation;
    // using A's cap against B must surface
    // [`RegistryError::CapabilityFromForeignRegistry`] from every
    // `*_strict` admin method.
    let (reg_a, cap_a) = TenantRegistry::new();
    let (reg_b, cap_b) = TenantRegistry::new();

    reg_a.register(make_ctx(100)).unwrap();
    reg_b.register(make_ctx(200)).unwrap();

    // A's cap against B: rejected.
    assert_eq!(
        reg_b.len_strict(&cap_a).unwrap_err(),
        RegistryError::CapabilityFromForeignRegistry,
    );
    assert_eq!(
        reg_b.get_strict(TenantId(200), &cap_a).unwrap_err(),
        RegistryError::CapabilityFromForeignRegistry,
    );
    assert_eq!(
        reg_b.tenants_strict(&cap_a).unwrap_err(),
        RegistryError::CapabilityFromForeignRegistry,
    );
    assert_eq!(
        reg_b.unregister_strict(TenantId(200), &cap_a).unwrap_err(),
        RegistryError::CapabilityFromForeignRegistry,
    );

    // A's cap on A: every strict variant succeeds.
    assert_eq!(reg_a.len_strict(&cap_a).unwrap(), 1);
    assert!(reg_a.get_strict(TenantId(100), &cap_a).unwrap().is_some());
    let snap = reg_a.tenants_strict(&cap_a).unwrap();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].id(), TenantId(100));

    // Symmetric: B's cap on A is also rejected.
    assert_eq!(
        reg_a.len_strict(&cap_b).unwrap_err(),
        RegistryError::CapabilityFromForeignRegistry,
    );
}

#[test]
fn admin_cap_legacy_apis_silently_no_op_on_foreign_cap() {
    // The `Option`/`usize`-returning legacy admin methods preserve their
    // signatures under strict binding: a foreign cap looks like "no
    // matching tenant" (`None`) or "empty registry" (`0`). Tests that
    // want the typed error should call the `*_strict` variants.
    let (reg_a, cap_a) = TenantRegistry::new();
    let (reg_b, cap_b) = TenantRegistry::new();

    reg_a.register(make_ctx(11)).unwrap();
    reg_b.register(make_ctx(22)).unwrap();

    // Using B's cap against A: legacy `get` collapses to `None`, legacy
    // `len` collapses to `0`, legacy `tenants` returns an empty Vec.
    assert!(reg_a.get(TenantId(11), &cap_b).is_none());
    assert_eq!(reg_a.len(&cap_b), 0);
    assert!(reg_a.tenants(&cap_b).is_empty());
    // And `unregister` does NOT actually evict the tenant — the legacy
    // signature collapses the refusal to `None`, but the underlying
    // DashMap entry is untouched.
    assert!(reg_a.unregister(TenantId(11), &cap_b).is_none());
    // Sanity: A's own cap still sees tenant 11 — the rejected
    // `unregister` above did not mutate A's state.
    assert_eq!(reg_a.len(&cap_a), 1);
    assert!(reg_a.get(TenantId(11), &cap_a).is_some());
}

#[test]
fn tenant_cap_from_foreign_registry_cannot_drive_quota() {
    // The per-tenant `TenantCapability` is the strict-binding sibling of
    // `RegistryAdminCapability`. A cap minted by registry A naming
    // `TenantId(7)` must NOT be accepted by registry B's `TenantId(7)`
    // context — even though both contexts share the same numeric id.
    // Without strict binding this attack works (see audit note on
    // `TenantCapability`); strict binding rejects it via the
    // `registry_token` comparison inside `check_capability`.
    let (reg_a, _cap_a) = TenantRegistry::new();
    let (reg_b, _cap_b) = TenantRegistry::new();

    let (ctx_a, tcap_a) = reg_a.register_with_capability(make_ctx(7)).unwrap();
    let (ctx_b, tcap_b) = reg_b.register_with_capability(make_ctx(7)).unwrap();

    // Baseline: each context starts empty and each cap drives its own.
    assert_eq!(ctx_a.bytes_in_use(), 0);
    assert_eq!(ctx_b.bytes_in_use(), 0);
    ctx_a.consume_bytes_with_capability(&tcap_a, 128).unwrap();
    ctx_b.consume_bytes_with_capability(&tcap_b, 256).unwrap();

    // Cross-registry attempt: A's cap on B's context. Strict binding
    // surfaces this as `TenantIsolationViolation` (the same error class
    // used by cross-tenant attempts inside one registry — the audit's
    // recommendation), with the cap's tenant id labelled as the
    // offender.
    let err = ctx_b
        .consume_bytes_with_capability(&tcap_a, 64)
        .expect_err("foreign-registry cap must be refused under strict binding");
    match err {
        tensor_wasm_core::error::TensorWasmError::TenantIsolationViolation {
            tenant_id, ..
        } => {
            assert_eq!(tenant_id, TenantId(7));
        }
        other => panic!("expected TenantIsolationViolation, got {other:?}"),
    }
    // Counter unchanged by the rejected attempt.
    assert_eq!(ctx_b.bytes_in_use(), 256);

    // Symmetric: B's cap on A's context.
    ctx_a
        .release_bytes_with_capability(&tcap_b, 1)
        .expect_err("foreign-registry cap must be refused under strict binding");
    assert_eq!(ctx_a.bytes_in_use(), 128);

    // And same-registry use continues to work.
    ctx_a.consume_bytes_with_capability(&tcap_a, 16).unwrap();
    assert_eq!(ctx_a.bytes_in_use(), 144);
}

#[test]
fn clone_of_registry_shares_token_so_caps_remain_valid() {
    // Cloning a `TenantRegistry` is the documented way to hand the same
    // registry to a subsystem. Under strict binding the clone must
    // share the registry token (caps minted against either handle must
    // continue to work against the other) — otherwise legitimate
    // multi-component embedders would break.
    let (reg, cap) = TenantRegistry::new();
    let reg_clone = reg.clone();
    reg.register(make_ctx(1)).unwrap();

    // `cap` was minted by `reg`. After cloning, it must still work on
    // `reg_clone` because the clone shares the same token allocation.
    assert_eq!(reg_clone.len_strict(&cap).unwrap(), 1);
    assert!(reg_clone.get_strict(TenantId(1), &cap).unwrap().is_some());
}

#[test]
fn tenant_cap_is_clone_safe_and_continues_to_bind() {
    // `TenantCapability: Clone` — strict binding must survive a clone of
    // the cap (the registry_token field is `Arc<()>`, so cloning the
    // cap clones the Arc and `ptr_eq` still holds).
    let (reg, _) = TenantRegistry::new();
    let (ctx, tcap) = reg.register_with_capability(make_ctx(99)).unwrap();
    let tcap2: TenantCapability = tcap.clone();
    ctx.consume_bytes_with_capability(&tcap2, 32).unwrap();
    assert_eq!(ctx.bytes_in_use(), 32);
}
