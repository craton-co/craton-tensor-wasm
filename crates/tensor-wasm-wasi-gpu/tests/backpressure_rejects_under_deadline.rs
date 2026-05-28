// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T36 — back-pressure deadline-rejection integration tests.
//!
//! Confirms that a `BackPressure` with a deadline installed via
//! [`BackPressure::with_deadline_hint`] refuses NEW acquires once the
//! deadline enters the `DEADLINE_NEAR_WINDOW` while still letting
//! in-flight permits drain.

use std::time::{Duration, Instant};

use tensor_wasm_wasi_gpu::abi::AbiError;
use tensor_wasm_wasi_gpu::async_dispatch::{
    BackPressure, BackPressureError, DEADLINE_NEAR_WINDOW,
};

#[tokio::test]
async fn cap2_near_deadline_rejects_new_acquires_with_typed_error() {
    // BackPressure cap=2 with a deadline well inside the NEAR window:
    // every new acquire must be refused with the typed
    // `DeadlineNear` variant (the production wire collapses this to
    // QuotaExceeded on the AbiError side).
    let deadline = Instant::now() + DEADLINE_NEAR_WINDOW / 2;
    let bp = BackPressure::with_cap(2).with_deadline_hint(Some(deadline));
    let err = bp
        .acquire_with_deadline()
        .await
        .expect_err("near-deadline acquire must be refused");
    assert_eq!(err, BackPressureError::DeadlineNear);
    assert_eq!(bp.active(), 0);
}

#[tokio::test]
async fn cap2_near_deadline_rejects_via_acquire_borrowed_too() {
    // The production launch path uses `acquire_borrowed` which
    // returns the AbiError shape. New acquires must surface
    // QuotaExceeded (the collapsed form of `DeadlineNear`).
    let deadline = Instant::now() + DEADLINE_NEAR_WINDOW / 2;
    let bp = BackPressure::with_cap(2).with_deadline_hint(Some(deadline));
    let err = bp
        .acquire_borrowed()
        .await
        .expect_err("near-deadline acquire must surface AbiError");
    assert_eq!(err, AbiError::QuotaExceeded);
}

#[tokio::test]
async fn in_flight_drains_under_near_deadline() {
    // Deadline 100 ms out — outside the 50 ms NEAR window at the
    // moment we acquire. The in-flight permit is unaffected when the
    // deadline crosses into the window; only NEW acquires are
    // refused. Drop the in-flight last and confirm `active()` falls
    // to 0 — i.e. the existing cohort drained normally.
    let deadline = Instant::now() + Duration::from_millis(100);
    let bp = BackPressure::with_cap(2).with_deadline_hint(Some(deadline));
    let in_flight = bp.acquire_borrowed().await.expect("first acquire fits");
    assert_eq!(bp.active(), 1);

    // Cross into the NEAR window (50 ms remaining requires the
    // deadline to be ≤ 50 ms out; we sleep 60 ms so ≤ 40 ms
    // remains).
    tokio::time::sleep(Duration::from_millis(60)).await;
    let err = bp
        .acquire_borrowed()
        .await
        .expect_err("near-deadline must refuse new acquires");
    assert_eq!(err, AbiError::QuotaExceeded);
    // In-flight is still held.
    assert_eq!(bp.active(), 1);
    drop(in_flight);
    assert_eq!(bp.active(), 0);
}

#[tokio::test]
async fn no_deadline_clone_passes_acquire_through() {
    // Two clones of the same BackPressure: one carries a near
    // deadline, the other carries none. The no-deadline clone must
    // still acquire successfully even though the shared semaphore
    // pool is the same.
    let bp_base = BackPressure::with_cap(2);
    let near = Instant::now() + DEADLINE_NEAR_WINDOW / 2;
    let bp_near = bp_base.clone().with_deadline_hint(Some(near));
    let bp_none = bp_base.clone();
    // Sanity: shared pool — active() reflects whichever clone holds
    // a permit because the inner Arc is shared.
    assert_eq!(bp_base.active(), 0);
    // bp_none must succeed; bp_near must refuse.
    let permit = bp_none.acquire_borrowed().await.expect("no-deadline succeeds");
    assert_eq!(bp_base.active(), 1);
    let err = bp_near
        .acquire_borrowed()
        .await
        .expect_err("near-deadline clone must refuse");
    assert_eq!(err, AbiError::QuotaExceeded);
    drop(permit);
    assert_eq!(bp_base.active(), 0);
}

#[tokio::test]
async fn elapsed_deadline_typed_variant() {
    // A deadline already in the past must surface `DeadlineElapsed`
    // (not `DeadlineNear`) on the typed surface.
    let bp = BackPressure::with_cap(2)
        .with_deadline_hint(Some(Instant::now() - Duration::from_millis(10)));
    let err = bp
        .acquire_with_deadline()
        .await
        .expect_err("elapsed-deadline must refuse");
    assert_eq!(err, BackPressureError::DeadlineElapsed);
}
