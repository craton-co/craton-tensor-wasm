// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for core M5: TOCTOU race in
//! `tensor_wasm_core::telemetry::init_with_otlp`.
//!
//! The bug: `init_with_otlp` used to call
//! `opentelemetry::global::set_tracer_provider(...)` and
//! `set_text_map_propagator(...)` *before* `tracing_subscriber::try_init`.
//! If `try_init` then failed (another subscriber raced in between the
//! `INIT.is_completed()` check at the top of the function and the
//! `try_init` call inside the `Once` closure), the function returned
//! `Err(AlreadyInitialized)` but the OTel globals had already been
//! mutated. Subsequent calls saw the `Once` as completed, skipped the
//! re-init, and silently returned `Ok(false)` — masking the inconsistency.
//!
//! The fix moves `try_init` ahead of the OTel global mutations and caches
//! the first call's outcome in a `OnceLock<Result<(), OtlpInitError>>` so
//! every subsequent call replays the same result.
//!
//! This is hard to test without an injection point into either
//! `tracing_subscriber::try_init` or `opentelemetry::global::*`. The two
//! invariants we can pin down with a black-box integration test are:
//!
//! 1. **Signature stability.** The new `init_with_otlp` still has the
//!    public signature
//!    `fn(LogLevel, bool, &str) -> Result<bool, OtlpInitError>`. The
//!    compile-time `let _: fn(...) -> _ = init_with_otlp;` binding below
//!    fails to compile if anyone changes that shape.
//! 2. **Outcome stability across calls.** Whatever the first call returns
//!    (`Ok(true)`, `Ok(false)`, or any `Err(_)` variant), the second call
//!    must return a "compatible" outcome — never the opposite kind. In
//!    particular, the function must never return `Ok(true)` on a *second*
//!    call (that would mean it did a second initialisation), and must
//!    never flip from `Err(_)` to `Ok(_)` (that would mean the cached
//!    failure was silently swallowed).
//!
//! Both invariants together rule out the original M5 inconsistency.

#![cfg(feature = "otlp")]

use tensor_wasm_core::telemetry::{init_with_otlp, LogLevel, OtlpInitError};

/// Classify an outcome so we can compare the first and second call without
/// having to clone an `Err` we don't own (the function only hands out an
/// owned `Result` and the second call may legitimately produce a different
/// `Err` payload if it lost the race differently). What we care about is
/// the *kind* of outcome.
#[derive(Debug, PartialEq, Eq)]
enum Kind {
    OkTrue,
    OkFalse,
    ErrExporter,
    ErrAlreadyInitialized,
}

fn classify(r: &Result<bool, OtlpInitError>) -> Kind {
    match r {
        Ok(true) => Kind::OkTrue,
        Ok(false) => Kind::OkFalse,
        Err(OtlpInitError::Exporter(_)) => Kind::ErrExporter,
        Err(OtlpInitError::AlreadyInitialized) => Kind::ErrAlreadyInitialized,
    }
}

#[test]
fn signature_is_stable() {
    // If the public signature of `init_with_otlp` ever changes, this
    // binding stops type-checking — i.e. the regression test fails at
    // compile time, which is the strongest possible signal.
    let _: fn(LogLevel, bool, &str) -> Result<bool, OtlpInitError> = init_with_otlp;
}

#[test]
fn calling_twice_returns_stable_outcome() {
    // Run the function back-to-back in the same process. We can't predict
    // whether an OTLP collector is running on `localhost:4317`, so the
    // first call may return any of the four `Kind`s. What we DO know is:
    //
    //   * If the first call returned `Ok(true)`, the second MUST return
    //     `Ok(false)` (cached "we succeeded once already") — never
    //     `Ok(true)` again, and never an `Err(_)` (the cached `Ok(())`
    //     must replay as `Ok(false)`).
    //
    //   * If the first call returned `Ok(false)`, the second MUST also
    //     return `Ok(false)`. (`Ok(false)` on first call only happens if
    //     *another test* in this same binary already initialised the
    //     OTLP pipeline; in that case the cached `Ok(())` keeps replaying
    //     as `Ok(false)`.)
    //
    //   * If the first call returned `Err(_)`, the second MUST return the
    //     same `Err` *variant*. The cached error must NOT silently turn
    //     into `Ok(false)` (that was the M5 bug: a stale `Once` masking a
    //     half-initialised pipeline).
    //
    // All of these are summarised by the table:
    //
    //   first        | required second
    //   -------------+----------------
    //   Ok(true)     | Ok(false)
    //   Ok(false)    | Ok(false)
    //   Err(kind)    | Err(same kind)
    let first = init_with_otlp(LogLevel::Info, false, "TENSOR_WASM_TEST_OTLP_NO_LEAK_ENDPOINT");
    let second = init_with_otlp(LogLevel::Info, false, "TENSOR_WASM_TEST_OTLP_NO_LEAK_ENDPOINT");

    let k1 = classify(&first);
    let k2 = classify(&second);

    // Second call must never be `Ok(true)` — that would mean a second
    // initialisation slipped past the `Once`.
    assert_ne!(
        k2,
        Kind::OkTrue,
        "init_with_otlp returned Ok(true) on its SECOND call; the `Once` guard is broken. first={first:?}, second={second:?}",
    );

    match k1 {
        Kind::OkTrue | Kind::OkFalse => {
            // Cached success: second call must report a (different-but-
            // success-compatible) `Ok(false)`.
            assert_eq!(
                k2,
                Kind::OkFalse,
                "successful first call must replay as Ok(false), but second was {second:?}",
            );
        }
        Kind::ErrExporter | Kind::ErrAlreadyInitialized => {
            // Cached failure: second call must NOT silently turn into
            // `Ok(_)`. It must keep returning the same error kind so
            // operators can diagnose the failure from any call site.
            assert_eq!(
                k1, k2,
                "init_with_otlp flipped outcome between first={first:?} and second={second:?}; cached error must replay verbatim",
            );
        }
    }
}
