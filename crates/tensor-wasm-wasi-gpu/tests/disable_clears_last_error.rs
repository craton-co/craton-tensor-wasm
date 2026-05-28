// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Audit follow-up: `disable_wasi_cuda` must clear any prior `last_error`.
//!
//! The contract is documented on `WasiCudaContext::disable_wasi_cuda`:
//!
//! > Also clears any previously-recorded `last_error` so flipping the
//! > capability cannot let a guest read state recorded while the
//! > capability was disabled (wasi-gpu 1.5 follow-up).
//!
//! The audit flagged that this clearing behaviour had no test coverage.
//! Without it, a guest could:
//!   1. trigger a host-side error while the capability is enabled,
//!   2. wait for the embedder to call `disable_wasi_cuda` for policy
//!      reasons, then
//!   3. read the residual error through `last_error_len` /
//!      `last_error_copy` — which themselves now short-circuit on the
//!      disabled-capability path, so step 3 is also gated, but the
//!      defence-in-depth is to make sure there's *nothing left to leak*
//!      regardless of any future regression on the host functions
//!      themselves.
//!
//! Also covers the "re-enable does not resurrect" property: after
//! `disable_wasi_cuda` clears the slot, a follow-up `enable_wasi_cuda`
//! must not bring the cleared message back.

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::host::WasiCudaContext;

/// Recording an error then calling `disable_wasi_cuda` must wipe
/// `last_error` to `None`.
///
/// We enable the capability, record an error, then disable. The
/// disable path must clear the slot. The capability flag itself is
/// also asserted (separate from the slot clearing — both are part of
/// `disable_wasi_cuda`'s public contract).
#[test]
fn disable_wasi_cuda_clears_last_error() {
    let mut ctx = WasiCudaContext::new(InstanceId(2001));

    // Enable, record, then disable.
    ctx.enable_wasi_cuda();
    assert!(ctx.wasi_cuda_enabled(), "enable_wasi_cuda must flip the flag");
    ctx.record_error_for_test("transient failure while enabled");
    assert_eq!(
        ctx.last_error().as_deref(),
        Some("transient failure while enabled"),
        "precondition: error must be recorded before we test the clear",
    );

    // The contract under test.
    ctx.disable_wasi_cuda();

    assert!(
        !ctx.wasi_cuda_enabled(),
        "disable_wasi_cuda must clear the capability flag",
    );
    assert!(
        ctx.last_error().is_none(),
        "disable_wasi_cuda must clear last_error; got: {:?}",
        ctx.last_error(),
    );
}

/// After `disable_wasi_cuda` clears the slot, re-enabling the
/// capability must NOT resurrect the cleared message. Guards against a
/// regression where the disable path is changed to "stash and restore"
/// rather than "clear outright".
#[test]
fn reenable_after_disable_does_not_restore_last_error() {
    let mut ctx = WasiCudaContext::new(InstanceId(2002));

    ctx.enable_wasi_cuda();
    ctx.record_error_for_test("would-be ghost message");
    ctx.disable_wasi_cuda();
    // Sanity-check the immediate clear (the dedicated test above pins
    // this too; we re-check here so a failure mode is attributable).
    assert!(ctx.last_error().is_none());

    ctx.enable_wasi_cuda();
    assert!(
        ctx.last_error().is_none(),
        "re-enabling wasi-cuda must not resurrect a cleared error; got: {:?}",
        ctx.last_error(),
    );
}

/// `disable_wasi_cuda` on a context that never recorded an error must
/// be a no-op for `last_error` (idempotent clear). Guards against a
/// regression where the disable path panics on `None` or sets the slot
/// to `Some("")` instead of leaving it `None`.
#[test]
fn disable_wasi_cuda_on_clean_context_is_noop_for_last_error() {
    let mut ctx = WasiCudaContext::new(InstanceId(2003));
    assert!(ctx.last_error().is_none());
    ctx.enable_wasi_cuda();
    ctx.disable_wasi_cuda();
    assert!(
        ctx.last_error().is_none(),
        "disable_wasi_cuda on a never-errored context must keep last_error None",
    );
}
