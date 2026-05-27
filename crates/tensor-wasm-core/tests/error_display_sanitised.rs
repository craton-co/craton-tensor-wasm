// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression tests for `TensorWasmError` display sanitisation (core M2).
//!
//! The `CudaError`, `WasmTrap`, `WasmCompile`, and `Serialization` variants
//! historically formatted their inner vendor string verbatim via `Display`.
//! Those strings come from third-party crates (cust, cuda-oxide, wasmtime,
//! serde) and routinely contain host filesystem paths, raw pointer addresses,
//! or fragments of tenant-supplied bytes — none of which is safe to surface in
//! an HTTP response body or end-user log line.
//!
//! These tests construct each affected variant with a sentinel string that
//! deliberately mixes all three leak categories (`SECRET`, `/usr/...`,
//! `0x7f...`) and assert:
//!
//! 1. `Display` (`{e}`) and alternate-Display (`{e:#}`) never echo any of
//!    those tokens.
//! 2. `Debug` (`{e:?}`) — which is what `tracing` / server-side operator logs
//!    use — DOES still surface the inner sentinel, so diagnostic context is
//!    preserved for the human running the host.
//! 3. `inner()` returns the original string for trusted-side consumers.
//!
//! If any of these regress, an inner-error leak has been reintroduced.

use tensor_wasm_core::error::TensorWasmError;

/// A sentinel that mixes the three leak categories we guard against: a
/// fake-secret marker, a host path prefix, and a host-address-shaped hex
/// pointer. Reused across every variant so any failure is easy to grep.
const SENTINEL: &str = "/usr/lib/secret 0x7f000000 SECRET_HERE";

/// Tokens that must NOT appear in any tenant-facing rendering of the error
/// (`Display` or alternate `Display`). Listed individually so the assertion
/// failure tells you exactly which category leaked.
const FORBIDDEN_TOKENS: &[&str] = &["SECRET", "/usr/", "0x7f"];

fn assert_display_is_sanitised(e: &TensorWasmError, label: &str) {
    let d = format!("{e}");
    let d_alt = format!("{e:#}");
    for tok in FORBIDDEN_TOKENS {
        assert!(
            !d.contains(tok),
            "{label}: Display leaked forbidden token {tok:?}: {d:?}",
        );
        assert!(
            !d_alt.contains(tok),
            "{label}: alternate Display leaked forbidden token {tok:?}: {d_alt:?}",
        );
    }
    // The sanitised label must still be non-empty so callers / clients see
    // *some* class information.
    assert!(!d.is_empty(), "{label}: Display string was empty");
}

fn assert_debug_still_carries_sentinel(e: &TensorWasmError, label: &str) {
    let dbg = format!("{e:?}");
    assert!(
        dbg.contains("SECRET_HERE"),
        "{label}: Debug should still surface the inner sentinel for operator \
         logs, but did not: {dbg:?}",
    );
}

fn assert_inner_round_trips(e: &TensorWasmError, label: &str) {
    assert_eq!(
        e.inner(),
        Some(SENTINEL),
        "{label}: inner() must return the original sentinel for trusted-side \
         consumers (server logs, alert payloads)",
    );
}

#[test]
fn cuda_error_display_does_not_leak() {
    let e = TensorWasmError::CudaError(Box::from(SENTINEL));
    assert_display_is_sanitised(&e, "CudaError");
    assert_debug_still_carries_sentinel(&e, "CudaError");
    assert_inner_round_trips(&e, "CudaError");
    // The opaque label is part of the public contract — pin it so future
    // refactors that drop the label entirely (or accidentally rewrap with the
    // inner string) get caught here.
    assert_eq!(format!("{e}"), "cuda driver call failed");
}

#[test]
fn wasm_trap_display_does_not_leak() {
    let e = TensorWasmError::WasmTrap(Box::from(SENTINEL));
    assert_display_is_sanitised(&e, "WasmTrap");
    assert_debug_still_carries_sentinel(&e, "WasmTrap");
    assert_inner_round_trips(&e, "WasmTrap");
    assert_eq!(format!("{e}"), "wasm trap");
}

#[test]
fn wasm_compile_display_does_not_leak() {
    let e = TensorWasmError::WasmCompile(Box::from(SENTINEL));
    assert_display_is_sanitised(&e, "WasmCompile");
    assert_debug_still_carries_sentinel(&e, "WasmCompile");
    assert_inner_round_trips(&e, "WasmCompile");
    assert_eq!(format!("{e}"), "wasm compile failed");
}

#[test]
fn serialization_display_does_not_leak() {
    let e = TensorWasmError::Serialization(Box::from(SENTINEL));
    assert_display_is_sanitised(&e, "Serialization");
    assert_debug_still_carries_sentinel(&e, "Serialization");
    assert_inner_round_trips(&e, "Serialization");
    assert_eq!(format!("{e}"), "serialization error");
}

#[test]
fn kind_and_is_retryable_unchanged_after_sanitisation() {
    // `kind()` and `is_retryable()` classify on the variant itself, not on
    // the inner string. Sanitising `Display` must not have altered either —
    // metrics labels and the API layer's 5xx routing both depend on them.
    let cuda = TensorWasmError::CudaError(Box::from(SENTINEL));
    let trap = TensorWasmError::WasmTrap(Box::from(SENTINEL));
    let compile = TensorWasmError::WasmCompile(Box::from(SENTINEL));
    let ser = TensorWasmError::Serialization(Box::from(SENTINEL));

    assert_eq!(cuda.kind(), "cuda");
    assert_eq!(trap.kind(), "wasm_trap");
    assert_eq!(compile.kind(), "wasm_compile");
    assert_eq!(ser.kind(), "serialization");

    // None of the four sanitised variants are retryable — CUDA/Wasm failures
    // are not transient, and a serialisation error on identical bytes will
    // fail identically. (`KernelTimeout`, `Io`, `MemoryExhausted` carry the
    // retryable bit; those are covered by the inline unit tests.)
    assert!(!cuda.is_retryable(), "CudaError must not be retryable");
    assert!(!trap.is_retryable(), "WasmTrap must not be retryable");
    assert!(!compile.is_retryable(), "WasmCompile must not be retryable");
    assert!(!ser.is_retryable(), "Serialization must not be retryable");
}
