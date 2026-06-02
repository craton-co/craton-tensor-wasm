// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Audit follow-up: two `WasiCudaContext`s must not share `last_error`.
//!
//! The `last_error` slot is per-`WasiCudaContext` by construction (it
//! lives in a `Mutex<Option<String>>` field), but no existing test
//! asserts the cross-instance invariant — the wasi-gpu audit flagged
//! this as a gap because the slot's *visibility* via `last_error_len` /
//! `last_error_copy` is the only channel a guest has into host-side
//! per-instance state, and an accidental sharing (e.g. a future
//! refactor that promoted the field to a `static` or moved it into the
//! shared `KernelRegistry`) would be a cross-instance information leak.
//!
//! This test pins the contract: when instance A records an error,
//! instance B's `last_error()` must still return `None`, and vice
//! versa.

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::host::WasiCudaContext;

/// Recording an error on instance A must not be observable on
/// instance B's `last_error()`, and vice versa.
///
/// We exercise both directions explicitly: an accidentally-shared
/// `Mutex` would surface the same value on both sides regardless of
/// which one wrote, so testing just one direction is insufficient.
#[test]
fn last_error_is_per_instance() {
    let ctx_a = WasiCudaContext::new(InstanceId(1001));
    let ctx_b = WasiCudaContext::new(InstanceId(1002));

    // Baseline: both contexts start with no recorded error.
    assert!(
        ctx_a.last_error().is_none(),
        "fresh ctx A must have no last_error",
    );
    assert!(
        ctx_b.last_error().is_none(),
        "fresh ctx B must have no last_error",
    );

    // Record on A. B must remain pristine.
    ctx_a.record_error_for_test("instance-A failure");
    assert_eq!(
        ctx_a.last_error().as_deref(),
        Some("instance-A failure"),
        "ctx A must see its own recorded error",
    );
    assert!(
        ctx_b.last_error().is_none(),
        "ctx B must NOT see ctx A's error — last_error is per-instance state, \
         got: {:?}",
        ctx_b.last_error(),
    );

    // Now record a distinct message on B. Both must report their own.
    ctx_b.record_error_for_test("instance-B failure");
    assert_eq!(
        ctx_a.last_error().as_deref(),
        Some("instance-A failure"),
        "ctx A's slot must not be overwritten by a record on ctx B",
    );
    assert_eq!(
        ctx_b.last_error().as_deref(),
        Some("instance-B failure"),
        "ctx B must see its own recorded error",
    );
}
