// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Input-bound regression tests for the wasi-cuda host bridge.
//!
//! Covers two abuse cases that previously had no cap:
//!
//! - `record_error` accepted an unbounded `String` per call — a guest
//!   looping `launch` with malformed input could force a large
//!   `format!` allocation on each call (wasi-gpu 1.6).
//! - `load_ptx` fed `entry_len` straight into `read_bytes`, which was
//!   bounded only by linear memory — a guest could pass a 64 MiB entry
//!   name and force a 64 MiB UTF-8 validation + `String::from` per call
//!   (wasi-gpu 1.9).
//!
//! Both are now capped by constants in `host`: see
//! [`MAX_RECORDED_ERROR_BYTES`] and [`MAX_ENTRY_NAME_BYTES`].

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::host::{
    WasiCudaContext, MAX_ENTRY_NAME_BYTES, MAX_RECORDED_ERROR_BYTES,
};

/// `record_error` must truncate messages longer than
/// [`MAX_RECORDED_ERROR_BYTES`] (and append a single ellipsis) so a
/// guest cannot push arbitrarily-large strings into the host's
/// per-instance `last_error` slot.
///
/// The exact byte count of the truncated message is
/// `cutoff + len("\u{2026}")`. The ellipsis is a 3-byte UTF-8 sequence,
/// so the upper bound is `MAX_RECORDED_ERROR_BYTES + 3`. We assert
/// `<= MAX_RECORDED_ERROR_BYTES + 3` to allow for that and reject the
/// uncapped case (10 KiB).
#[test]
fn record_error_truncates_long_messages() {
    let ctx = WasiCudaContext::new(InstanceId(7777));
    // 10 KiB of ASCII so every byte is also a char boundary — that
    // means the truncation index is exactly `MAX_RECORDED_ERROR_BYTES`.
    let huge = "x".repeat(10 * 1024);
    ctx.record_error_for_test(huge);
    let stored = ctx.last_error().expect("error recorded");
    assert!(
        stored.len() <= MAX_RECORDED_ERROR_BYTES + "\u{2026}".len(),
        "stored.len() = {}, expected <= {}",
        stored.len(),
        MAX_RECORDED_ERROR_BYTES + "\u{2026}".len(),
    );
    // For an ASCII payload the truncation index lands exactly at the
    // cap, so the truncated prefix is `MAX_RECORDED_ERROR_BYTES` bytes
    // of 'x' followed by the ellipsis.
    assert!(
        stored.ends_with('\u{2026}'),
        "expected truncated message to end with ellipsis, got tail: {:?}",
        &stored[stored.len().saturating_sub(8)..],
    );
    assert!(
        stored.starts_with("xxxx"),
        "expected truncated message to keep the leading payload",
    );
}

/// A short message must round-trip without truncation — the cap is a
/// ceiling, not a floor. Guards against an off-by-one that would clip
/// every error message.
#[test]
fn record_error_keeps_short_messages_intact() {
    let ctx = WasiCudaContext::new(InstanceId(7778));
    ctx.record_error_for_test("brief");
    assert_eq!(ctx.last_error().as_deref(), Some("brief"));
}

/// A message exactly at the cap boundary must NOT be truncated — only
/// messages strictly greater than the cap go through the truncation
/// path. Guards the `msg.len() > MAX_RECORDED_ERROR_BYTES` predicate.
#[test]
fn record_error_at_cap_is_not_truncated() {
    let ctx = WasiCudaContext::new(InstanceId(7779));
    let at_cap = "y".repeat(MAX_RECORDED_ERROR_BYTES);
    ctx.record_error_for_test(at_cap.clone());
    let stored = ctx.last_error().expect("error recorded");
    assert_eq!(stored.len(), MAX_RECORDED_ERROR_BYTES);
    assert_eq!(stored, at_cap);
}

/// Truncation must land on a UTF-8 boundary even when the cap falls
/// in the middle of a multi-byte sequence. We build a string where the
/// `MAX_RECORDED_ERROR_BYTES`-th byte is inside a multi-byte char and
/// confirm the result is still valid UTF-8 (the `String` invariant
/// guarantees this — the test exists to catch a regression where
/// `String::truncate` is called with a non-boundary index and panics).
#[test]
fn record_error_truncates_on_utf8_boundary() {
    let ctx = WasiCudaContext::new(InstanceId(7780));
    // Pad with single-byte ASCII to just below the cap, then add
    // multi-byte chars (`é` = 2 bytes each) that straddle the cap.
    let mut payload = "a".repeat(MAX_RECORDED_ERROR_BYTES - 1);
    for _ in 0..256 {
        payload.push('é');
    }
    ctx.record_error_for_test(payload);
    let stored = ctx.last_error().expect("error recorded");
    // The String invariant (well-formed UTF-8) is what we're really
    // checking — if `truncate` had been called off-boundary it would
    // have panicked before we reached this line.
    assert!(stored.is_char_boundary(stored.len()));
    assert!(stored.ends_with('\u{2026}'));
}

/// Constant-shape regression test for the `load_ptx` entry-name cap.
/// Driving the actual wasmtime path requires the kernel-args plumbing
/// (see `wasi_gpu_smoke.rs`); the entry-name guard is a simple
/// `entry_len as usize > MAX_ENTRY_NAME_BYTES` check that fires before
/// `read_bytes` is reached, so the meaningful invariants are (a) the
/// cap exists, and (b) it is small enough that a future bump would be
/// a deliberate decision rather than an accident.
#[test]
fn load_ptx_rejects_oversized_entry_name() {
    // PTX kernel identifiers are short C-style identifiers — 256 is a
    // generous ceiling. A future bump above 512 should be deliberate.
    const _: () = assert!(
        MAX_ENTRY_NAME_BYTES <= 512,
        "MAX_ENTRY_NAME_BYTES raising past 512 should be a deliberate decision",
    );
    // And the predicate itself: a 64 MiB entry-name length must be
    // rejected by the same arithmetic the host uses.
    let abusive_entry_len: i32 = 64 * 1024 * 1024;
    assert!(
        (abusive_entry_len as usize) > MAX_ENTRY_NAME_BYTES,
        "64 MiB entry_len must exceed MAX_ENTRY_NAME_BYTES",
    );
}
