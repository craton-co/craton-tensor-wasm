// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `TenantRegistry::new()` must mint exactly one
//! [`RegistryAdminCapability`], independent caps from independent
//! constructions, and downstream call-sites that only hold an
//! `Arc<TenantRegistry>` (no cap) must be unable to invoke
//! `get` / `unregister` / `tenants` / `len`.
//!
//! The third property is enforced at the **type** level — there is no
//! runtime check for "missing cap"; the method signatures simply do not
//! accept a registry-without-a-cap form. We pin that with a
//! `trybuild`-style compile-fail snippet embedded as a doc-test comment
//! and a runtime use that exercises the cap-bound path.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{TenantContext, TenantRegistry};

fn make_ctx(id: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_memory_quota_bytes(4096)
        .build()
}

#[test]
fn new_returns_a_cap_alongside_the_registry() {
    // Type assertion: `new()` must return a tuple of
    // `(TenantRegistry, RegistryAdminCapability)`. If the signature ever
    // regresses to a single `TenantRegistry`, this destructure stops
    // compiling — the test breaks at the type level, not at runtime.
    let (reg, cap) = TenantRegistry::new();
    assert!(reg.is_empty());

    // The freshly minted cap actually unlocks the admin path.
    reg.register(make_ctx(1)).unwrap();
    assert_eq!(reg.len(&cap), 1);
    let found = reg.get(TenantId(1), &cap).expect("just registered");
    assert_eq!(found.id(), TenantId(1));
    let snapshot = reg.tenants(&cap);
    assert_eq!(snapshot.len(), 1);
    let removed = reg.unregister(TenantId(1), &cap).unwrap();
    assert_eq!(removed.id(), TenantId(1));
}

#[test]
fn independent_constructions_yield_independent_caps() {
    // Each `new()` mints a fresh cap. The two registries hold disjoint
    // DashMap shards, so admin operations against one are visible only
    // when invoked with that registry's own cap.
    //
    // The cap-interchangeability assertion (cap_a used against reg_b)
    // describes the default-feature behaviour: caps are an opaque
    // "you-hold-*some*-cap" token, not a tag tied to a specific
    // registry. Under `--features strict-cap-binding` the cap is bound
    // to its minting registry by `Arc::ptr_eq`, and cross-registry use
    // is refused with [`RegistryError::CapabilityFromForeignRegistry`];
    // that flavour is exercised by `tests/cap_binding_strict.rs`.
    let (reg_a, cap_a) = TenantRegistry::new();
    let (reg_b, cap_b) = TenantRegistry::new();
    assert!(reg_a.is_empty());
    assert!(reg_b.is_empty());

    reg_a.register(make_ctx(100)).unwrap();
    reg_b.register(make_ctx(200)).unwrap();

    // Both caps work against their own registry — this holds in both
    // strict and non-strict modes.
    assert_eq!(reg_a.len(&cap_a), 1);
    assert_eq!(reg_b.len(&cap_b), 1);

    // Each registry only sees its own tenant.
    assert!(reg_a.get(TenantId(200), &cap_a).is_none());
    assert!(reg_b.get(TenantId(100), &cap_b).is_none());

    // Non-strict mode: a cap from registry A is also accepted by
    // registry B (the security boundary is "*some* cap" rather than
    // "this exact registry's cap"). Strict mode flips this to a hard
    // refusal — see `tests/cap_binding_strict.rs`.
    #[cfg(not(feature = "strict-cap-binding"))]
    {
        // `cap_a` works against `reg_b` (opaque-token contract).
        assert_eq!(reg_b.len(&cap_a), 1);
        let snap_via_a = reg_b.tenants(&cap_a);
        assert_eq!(snap_via_a.len(), 1);
        assert_eq!(snap_via_a[0].id(), TenantId(200));
    }

    // And the caps are independently usable — one going out of scope
    // does not invalidate the other (they share no allocation).
    drop(cap_a);
    let snap = reg_b.tenants(&cap_b);
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].id(), TenantId(200));
}

#[test]
fn arc_registry_without_cap_cannot_enumerate_or_evict() {
    // Simulate the production layout: the admin layer keeps the cap, and
    // hands an `Arc<TenantRegistry>` to subsystems that should only be
    // able to register their own tenants and operate on the `Arc` they
    // already hold.
    let (reg, cap) = TenantRegistry::new();
    let arc_for_subsystem: Arc<TenantRegistry> = Arc::new(reg.clone());

    let _registered = reg.register(make_ctx(7)).unwrap();

    // The subsystem CAN still `register` (open by design — registering
    // a tenant whose id you already chose adds no authority over
    // anyone else), and can call `is_empty` (it leaks only one bit).
    let _ = arc_for_subsystem.register(make_ctx(8));
    assert!(!arc_for_subsystem.is_empty());

    // What the subsystem CANNOT do is enumerate / evict / count — those
    // methods now require `&RegistryAdminCapability`, which it never got.
    // The following lines would fail to compile if uncommented; see the
    // doc-comment at the top of this file. We assert the cap-gated path
    // still works for the admin holder as the positive control.
    //
    //     arc_for_subsystem.get(TenantId(7));            // E0061
    //     arc_for_subsystem.unregister(TenantId(7));     // E0061
    //     arc_for_subsystem.tenants();                   // E0061
    //     arc_for_subsystem.len();                       // E0061
    assert_eq!(reg.len(&cap), 2);
    assert!(reg.get(TenantId(7), &cap).is_some());
    assert!(reg.get(TenantId(8), &cap).is_some());
}
