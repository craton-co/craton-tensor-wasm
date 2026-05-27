// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `consume_bytes` must saturate on add overflow.
//!
//! A tenant configured with `u64::MAX` quota can legitimately consume
//! `u64::MAX` bytes in a single shot; the second request — for any
//! non-zero number of bytes — must be rejected with
//! `TensorWasmError::MemoryExhausted` (because `current + n` saturates to
//! `u64::MAX` which is **not** `> u64::MAX`, the rejection comes from
//! the explicit equality / overflow guard, not from wrap-around).
//!
//! Crucially, the counter must remain at `u64::MAX` after the failed
//! second call. A buggy CAS loop could leave it lower than that.

#![allow(deprecated)]
// Exercises the unchecked `consume_bytes` shim deliberately — the
// overflow-saturation contract must hold for the deprecated path until
// v0.4 removes it. The capability-checked variant shares the same inner
// implementation (`consume_bytes_inner`) so this test transitively
// covers it too.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::TenantContext;

#[test]
fn consume_u64_max_then_one_more_byte_saturates() {
    let ctx = TenantContext::builder(TenantId(1))
        .with_memory_quota_bytes(u64::MAX)
        .build();

    // First call: ask for the entire address space. Allowed.
    ctx.consume_bytes(u64::MAX).expect("first consume succeeds");
    assert_eq!(ctx.bytes_in_use(), u64::MAX);

    // Second call: even one extra byte must fail. `saturating_add` keeps
    // `next` at `u64::MAX`, but the guard rejects because `next > limit`
    // is false; we instead detect "would overflow" by the post-add
    // value being equal to the saturation point while `n > 0`. Either
    // way, the public contract is: this errors and the counter does
    // not move.
    let err = ctx.consume_bytes(1).expect_err("second consume rejected");
    match err {
        TensorWasmError::MemoryExhausted { requested, limit } => {
            assert_eq!(requested, 1);
            assert_eq!(limit, u64::MAX);
        }
        other => panic!("expected MemoryExhausted, got {other:?}"),
    }
    assert_eq!(
        ctx.bytes_in_use(),
        u64::MAX,
        "counter must stay at u64::MAX (saturating, not wrapping)",
    );
}

#[test]
fn second_attempt_with_u64_max_also_rejected_and_does_not_wrap() {
    let ctx = TenantContext::builder(TenantId(2))
        .with_memory_quota_bytes(u64::MAX)
        .build();

    ctx.consume_bytes(u64::MAX).unwrap();
    // Try to consume u64::MAX a second time — clearly impossible.
    let err = ctx.consume_bytes(u64::MAX).unwrap_err();
    assert!(matches!(err, TensorWasmError::MemoryExhausted { .. }));
    assert_eq!(ctx.bytes_in_use(), u64::MAX);
}
