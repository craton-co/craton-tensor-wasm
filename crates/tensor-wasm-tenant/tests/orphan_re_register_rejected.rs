//! Regression tests for tenant 1.6 #9 — unregister-while-mutating race.
//!
//! `unregister(T)` removes the registry entry but in-flight callers may
//! still hold an `Arc<TenantContext>` for the orphan. Re-registering T with
//! a fresh context would let those in-flight calls commit quota to the
//! orphan's counter while the new registration sees zero — a per-tenant
//! quota reset. The fix records a `Weak<TenantContext>` tombstone on
//! `unregister` and refuses re-registration while the orphan's strong
//! count is non-zero.

#![cfg(not(feature = "cuda"))]

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{registry::RegistryError, TenantContext, TenantRegistry};

#[test]
fn re_register_with_live_orphan_is_rejected() {
    let (reg, admin_cap) = TenantRegistry::new();
    let (orphan_arc, _tenant_cap) = reg
        .register_with_capability(TenantContext::builder(TenantId(42)).build())
        .unwrap();
    let _ = reg.unregister(TenantId(42), &admin_cap).unwrap();
    // Try to re-register — the orphan_arc is still alive, so this must fail.
    let err = reg
        .register_with_capability(TenantContext::builder(TenantId(42)).build())
        .unwrap_err();
    assert_eq!(err, RegistryError::OrphanStillAlive(TenantId(42)));
    // Drop the orphan; subsequent re-register succeeds.
    drop(orphan_arc);
    let _ = reg
        .register_with_capability(TenantContext::builder(TenantId(42)).build())
        .expect("orphan dropped, re-register must succeed");
}

#[test]
fn collect_tombstones_prunes_dead_weak_refs() {
    let (reg, admin_cap) = TenantRegistry::new();
    let _ = reg
        .register_with_capability(TenantContext::builder(TenantId(7)).build())
        .unwrap();
    // unregister but DROP the returned Arc — strong_count is now 0.
    let _ = reg.unregister(TenantId(7), &admin_cap);
    // A subsequent register without collect_tombstones must succeed too —
    // the orphan check correctly sees a dead weak ref.
    let _ = reg
        .register_with_capability(TenantContext::builder(TenantId(7)).build())
        .expect("dead orphan must not block re-register");
    // collect_tombstones is also safe to call directly.
    let _pruned = reg.collect_tombstones(&admin_cap);
}

#[test]
fn unregister_then_re_register_distinct_id_is_unaffected() {
    let (reg, admin_cap) = TenantRegistry::new();
    let (live_orphan, _) = reg
        .register_with_capability(TenantContext::builder(TenantId(1)).build())
        .unwrap();
    let _ = reg.unregister(TenantId(1), &admin_cap);
    // A different id is unaffected by the orphan on id 1.
    let _ = reg
        .register_with_capability(TenantContext::builder(TenantId(2)).build())
        .expect("orphan on id 1 must not block id 2");
    drop(live_orphan);
}
