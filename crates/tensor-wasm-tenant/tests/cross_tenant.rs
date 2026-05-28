// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Cross-tenant isolation: two `TenantContext`s in the same registry must
//! account for memory independently, and `unregister` must return an
//! `Option<Arc<TenantContext>>` so callers can recover the handle for an
//! orderly shutdown.

#![allow(deprecated)]
// Exercises the unchecked quota-mutation shim. The capability-gate
// upgrade is covered by the inline `registry` test
// `capability_from_one_tenant_cannot_mutate_another`; this file keeps
// pinning the per-tenant counter independence until the unchecked
// methods come out in v0.4.

use std::sync::Arc;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{IsolationKind, TenantContext, TenantRegistry};

fn make_ctx(id: u64, quota: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_isolation(IsolationKind::StreamIsolated)
        .with_stream_id(id)
        .with_memory_quota_bytes(quota)
        .build()
}

#[test]
fn distinct_tenants_have_independent_quotas() {
    let (reg, _cap) = TenantRegistry::new();
    let a: Arc<TenantContext> = reg.register(make_ctx(1, 4096)).unwrap();
    let b: Arc<TenantContext> = reg.register(make_ctx(2, 8192)).unwrap();

    // Quotas reported separately.
    assert_eq!(a.quota(), 4096);
    assert_eq!(b.quota(), 8192);

    // Consuming bytes in tenant A must not bleed into B.
    a.consume_bytes(2048).unwrap();
    assert_eq!(a.bytes_in_use(), 2048);
    assert_eq!(b.bytes_in_use(), 0);

    b.consume_bytes(7000).unwrap();
    assert_eq!(a.bytes_in_use(), 2048);
    assert_eq!(b.bytes_in_use(), 7000);

    // Push A to its limit and confirm B is unaffected by A's MemoryExhausted.
    a.consume_bytes(2048).unwrap();
    assert_eq!(a.bytes_in_use(), 4096);
    let err = a.consume_bytes(1).unwrap_err();
    assert!(matches!(
        err,
        TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 4096
        }
    ));
    assert_eq!(b.bytes_in_use(), 7000);

    // B can still consume up to its own quota.
    b.consume_bytes(1192).unwrap();
    assert_eq!(b.bytes_in_use(), 8192);
}

#[test]
fn unregister_returns_arc_option() {
    let (reg, cap) = TenantRegistry::new();
    reg.register(make_ctx(11, 1024)).unwrap();
    reg.register(make_ctx(12, 1024)).unwrap();
    assert_eq!(reg.len(&cap), 2);

    // Type assertion: unregister must return Option<Arc<TenantContext>>.
    let removed: Option<Arc<TenantContext>> = reg.unregister(TenantId(11), &cap);
    let removed = removed.expect("tenant 11 was registered");
    assert_eq!(removed.id(), TenantId(11));
    assert_eq!(reg.len(&cap), 1);

    // Removing a tenant that was never present is a clean None.
    let missing: Option<Arc<TenantContext>> = reg.unregister(TenantId(999), &cap);
    assert!(missing.is_none());

    // The remaining tenant is still reachable and its quota intact.
    let twelve = reg
        .get(TenantId(12), &cap)
        .expect("tenant 12 still present");
    assert_eq!(twelve.id(), TenantId(12));
    assert_eq!(twelve.quota(), 1024);
}

#[test]
fn arc_clones_share_quota_counter() {
    // Sanity check that the `Arc<TenantContext>` semantics actually share a
    // single counter — if the registry accidentally cloned the context on
    // lookup we would silently lose quota accounting under concurrent use.
    let (reg, cap) = TenantRegistry::new();
    reg.register(make_ctx(21, 4096)).unwrap();
    let a = reg.get(TenantId(21), &cap).unwrap();
    let b = reg.get(TenantId(21), &cap).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    a.consume_bytes(1000).unwrap();
    assert_eq!(b.bytes_in_use(), 1000);
}
