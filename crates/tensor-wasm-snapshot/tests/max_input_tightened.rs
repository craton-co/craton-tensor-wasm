// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T9: pin `limits::MAX_INPUT_BYTES` at the tightened 1 GiB value.
//!
//! The pre-T9 expression evaluated to 4 GiB (`64 * 1024 * 1024 * 64`),
//! which on a multi-tenant API surface was effectively no cap on the
//! attacker's pre-decompression memory footprint. T9 lowers the cap to
//! 1 GiB to bound the worst-case allocation requested by the reader
//! before zstd ever runs.
//!
//! Companion to `tests/max_input_rejected.rs`, which exercises the
//! reader's runtime rejection path by allocating `MAX_INPUT_BYTES + 1`
//! bytes. That test passes regardless of the constant's value because
//! it reads the constant at run time; this test pins the constant
//! itself, so any regression that raises the cap back toward 4 GiB
//! fails here without needing a multi-GiB host.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_snapshot::limits;
use tensor_wasm_snapshot::reader::SnapshotReader;

#[test]
fn max_input_bytes_is_one_gib() {
    // 1 GiB exactly — the T9 cap. Pinning the literal value here means
    // a future refactor that touches `MAX_INPUT_BYTES` has to also
    // update this test, which is the intended forcing function.
    assert_eq!(limits::MAX_INPUT_BYTES, 1024 * 1024 * 1024);
    // And confirm it shrank from the pre-T9 4 GiB value — a regression
    // that accidentally re-introduces `64 * 1024 * 1024 * 64` would
    // re-inflate the cap to 4 GiB without changing the existing
    // `max_input_rejected.rs` test outcome.
    const _: () = assert!(limits::MAX_INPUT_BYTES < 2 * 1024 * 1024 * 1024);
}

/// Targeted variant of `max_input_rejected.rs` that probes the rejection
/// path against the *new* (tightened) cap with a 1.5 GiB buffer — the
/// prompt's "attempting to write 1.5 GiB" scenario. Under the pre-T9 4
/// GiB cap this buffer would slip through; under the T9 1 GiB cap it
/// must be rejected with the "input too large" message.
///
/// Skipped on hosts that cannot allocate 1.5 GiB, matching the pattern
/// in `max_input_rejected.rs`. Skipping is safe because the same
/// `bytes.len() > limits::MAX_INPUT_BYTES` branch runs on every input
/// — coverage is exercised by the existing test on smaller hosts via
/// the `MAX_INPUT_BYTES + 1` shape.
#[test]
fn one_and_a_half_gib_input_is_rejected() {
    let oversize: usize = (3 * 1024 * 1024 * 1024) / 2; // 1.5 GiB

    let mut buf: Vec<u8> = Vec::new();
    if buf.try_reserve_exact(oversize).is_err() {
        eprintln!(
            "skipping max_input_tightened::one_and_a_half_gib_input_is_rejected: \
             host lacks {oversize} bytes of free RAM"
        );
        return;
    }
    buf.resize(oversize, 0);

    let err = SnapshotReader::new()
        .restore(&buf)
        .expect_err("1.5 GiB input must be rejected under the T9 1 GiB cap");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("input too large"),
                "expected input-cap rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
