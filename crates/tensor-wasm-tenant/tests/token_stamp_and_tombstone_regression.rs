// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Regressions for two tenant-registry fixes:
//!
//! B(2) — `tombstone_count` is now a non-test `pub` operator gauge.
//!   `tombstone_count_is_observable` pins that visibility: it is callable from
//!   an integration test (i.e. an external crate) WITHOUT a
//!   `RegistryAdminCapability`, and it tracks the tombstone-map size across an
//!   unregister (which records a tombstone) and a `collect_tombstones` reclaim
//!   (which prunes the dead weak ref back to zero). If `tombstone_count`
//!   regressed to `#[cfg(test)]`/private, this file would fail to compile.
//!
//! B(1) — `register_with_capability` stamps `ctx.registry_token` ONLY on the
//!   success path. `rejected_registration_does_not_stamp_token` exercises the
//!   rejection paths (`AlreadyRegistered` and `OrphanStillAlive`) and asserts
//!   the expected typed errors, plus the observable security property that the
//!   surviving registration's capability still drives quota correctly — the
//!   registry is left in a consistent state by the rejected call.
//!
//!   Visibility limitation: `TenantContext::registry_token` is `pub(crate)`
//!   with no public accessor, and a rejected `ctx` is dropped (never returned),
//!   so an external integration test cannot directly inspect the rejected
//!   context's token field. We therefore assert the closest observable
//!   property instead — see the per-test comment below. The "token stamped only
//!   on success" invariant proper is unit-tested inside the crate where the
//!   field is reachable; this file pins the externally-observable behaviour.
//!
//! Gated off `cuda` to match `tests/orphan_re_register_rejected.rs` (the
//! sibling tombstone test), since the CUDA build pulls in a different context
//! construction surface. The tenant crate's default features include
//! `strict-cap-binding`, so no extra gate is needed for the cap machinery.

#![cfg(not(feature = "cuda"))]

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{registry::RegistryError, TenantContext, TenantRegistry};

fn make_ctx(id: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_memory_quota_bytes(4096)
        .build()
}

#[test]
fn tombstone_count_is_observable() {
    // `new()` returns `(TenantRegistry, RegistryAdminCapability)`.
    let (reg, cap) = TenantRegistry::new();

    // Non-test `pub` gauge: callable from an external crate without a
    // capability. A fresh registry has no tombstones.
    assert_eq!(
        reg.tombstone_count(),
        0,
        "a fresh registry holds no tombstones",
    );

    // Register a tenant, keep the returned Arc, then unregister it. The
    // unregister records a `Weak<TenantContext>` tombstone for the id
    // (inserted unconditionally before the inner entry is removed), so the
    // tombstone map grows by one.
    let (orphan_arc, _tenant_cap) = reg
        .register_with_capability(make_ctx(42))
        .expect("register must succeed on a fresh id");
    let removed = reg
        .unregister(TenantId(42), &cap)
        .expect("unregister of a live id must return the Arc");
    assert_eq!(removed.id(), TenantId(42));

    // A tombstone now exists for id 42. The opportunistic prune on
    // `unregister` deliberately skips the just-touched id, so the count is
    // observably >= 1 here regardless of whether the orphan Arc is alive.
    assert!(
        reg.tombstone_count() >= 1,
        "unregister must record a tombstone observable via the public gauge",
    );

    // Drop both strong references so the tombstone's weak ref becomes
    // dead and `collect_tombstones` can reclaim it.
    drop(orphan_arc);
    drop(removed);

    // `collect_tombstones` is admin-cap-gated and mutates the map, pruning
    // the now-dead weak ref. The public gauge must drop back to zero.
    let pruned = reg.collect_tombstones(&cap);
    assert!(
        pruned >= 1,
        "collect_tombstones must reclaim the dead tombstone",
    );
    assert_eq!(
        reg.tombstone_count(),
        0,
        "tombstone_count must return to zero after the dead entry is reclaimed",
    );
}

#[test]
fn rejected_registration_does_not_stamp_token() {
    // --- Rejection path 1: AlreadyRegistered ---
    //
    // Registering the same id twice (while the first registration is live)
    // is rejected. Pre-fix, `register_with_capability` stamped the rejected
    // `ctx.registry_token` before the fallible `entry()` match resolved,
    // leaving the dropped ctx carrying this registry's token. Post-fix the
    // stamp happens ONLY on the success arm, so the rejected ctx never
    // carries a token. The ctx is dropped (not returned) on rejection, so
    // the externally-observable evidence is (a) the typed error and (b) that
    // the registry is left consistent.
    let (reg, _admin_cap) = TenantRegistry::new();

    let (live_ctx, live_cap) = reg
        .register_with_capability(make_ctx(7))
        .expect("first registration of id 7 must succeed");

    let err = reg
        .register_with_capability(make_ctx(7))
        .expect_err("re-registering a live id must be rejected");
    assert_eq!(
        err,
        RegistryError::AlreadyRegistered(TenantId(7)),
        "duplicate registration must surface AlreadyRegistered",
    );

    // Observable security property (closest assertable to "no foreign token
    // stamped on the rejected ctx"): the rejected registration left the
    // registry consistent, and the surviving registration's capability still
    // drives quota against the live context. The strict registry-token
    // binding inside `check_capability` would reject this consume if the
    // success-path stamping had been disturbed by the rejected attempt.
    live_ctx
        .consume_bytes_with_capability(&live_cap, 128)
        .expect("the surviving registration's cap must still drive quota");
    assert_eq!(live_ctx.bytes_in_use(), 128);

    // --- Rejection path 2: OrphanStillAlive ---
    //
    // Unregister an id while holding its Arc (an "orphan"), then attempt to
    // re-register the same id. The orphan is still alive, so re-registration
    // is rejected. The rejected ctx is again dropped without a token stamp.
    let (orphan_arc, _orphan_cap) = reg
        .register_with_capability(make_ctx(99))
        .expect("register id 99 must succeed");
    // unregister needs an admin cap from THIS registry.
    let _ = reg
        .unregister(TenantId(99), &_admin_cap)
        .expect("unregister id 99 must succeed");
    let err = reg
        .register_with_capability(make_ctx(99))
        .expect_err("re-register while orphan is alive must be rejected");
    assert_eq!(
        err,
        RegistryError::OrphanStillAlive(TenantId(99)),
        "re-register with a live orphan must surface OrphanStillAlive",
    );

    // Visibility limitation (documented at file top): we cannot read the
    // rejected ctx's `registry_token` (it is `pub(crate)` and the ctx is
    // dropped on rejection). The observable property we CAN pin is that once
    // the orphan drops, a fresh registration of the same id succeeds and its
    // new capability works — i.e. the rejected attempts left no stale
    // token-bearing state that would corrupt the eventual success path.
    drop(orphan_arc);
    let (fresh_ctx, fresh_cap) = reg
        .register_with_capability(make_ctx(99))
        .expect("orphan dropped — re-register of id 99 must now succeed");
    fresh_ctx
        .consume_bytes_with_capability(&fresh_cap, 64)
        .expect("the fresh registration's cap must drive quota");
    assert_eq!(fresh_ctx.bytes_in_use(), 64);
}
