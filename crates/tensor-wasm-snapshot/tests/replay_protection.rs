// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Behavioural coverage for the reader-side replay/rollback defences exercised
//! by the legacy (v2) `SnapshotReader::restore` path:
//!
//! * **Nonce** (`SnapshotReader::with_expected_nonce`, enforced in
//!   `reader.rs`'s `check_replay`): match → accept, mismatch → reject, missing
//!   → reject. This is the previously-uncovered `Some(actual) == expected` /
//!   `Some(_)` / `None` arm set.
//! * **Sequence floor** (`SnapshotReader::with_min_sequence_no`): a snapshot at
//!   or above the floor is accepted; one below is rejected as a
//!   rollback/replay.
//!
//! These run against a *default* (unsigned, no-HMAC-key) writer, so the bytes
//! travel the legacy v2 `restore` path end-to-end — the same path an operator
//! using `with_min_sequence_no` / `with_expected_nonce` without signing would
//! hit. The replay fields live inside the bincode payload, so even on the
//! unsigned path the reader reads them back faithfully; on a signed v3/v4 blob
//! the signature additionally certifies them (covered elsewhere).

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// Capture a tiny unsigned v2 snapshot, optionally stamping a sequence number
/// and/or nonce. Returns the framed bytes.
fn capture(seq: Option<u64>, nonce: Option<[u8; 16]>) -> Vec<u8> {
    let mut writer = SnapshotWriter::new();
    if let Some(s) = seq {
        writer = writer.with_sequence_no(s);
    }
    if let Some(n) = nonce {
        writer = writer.with_nonce(n);
    }
    writer
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &[1, 2, 3, 4],
            gpu_memory: &[9, 9, 9, 9],
            registers: &[0xAB; 8],
        })
        .expect("capture")
}

/// Helper: assert that `restore` failed with a `Serialization` error whose
/// message contains `needle`.
fn assert_rejected_with(err: TensorWasmError, needle: &str) {
    match err {
        TensorWasmError::Serialization(m) => {
            assert!(
                m.contains(needle),
                "expected message containing {needle:?}, got: {m}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}

// ----- nonce: match / mismatch / missing -----

#[test]
fn nonce_match_is_accepted() {
    let nonce = [0x5Au8; 16];
    let bytes = capture(None, Some(nonce));
    let restored = SnapshotReader::new()
        .with_expected_nonce(nonce)
        .restore(&bytes)
        .expect("matching nonce must be accepted on the legacy restore path");
    assert_eq!(restored.metadata.nonce, Some(nonce));
}

#[test]
fn nonce_mismatch_is_rejected() {
    let bytes = capture(None, Some([0x11u8; 16]));
    let err = SnapshotReader::new()
        .with_expected_nonce([0x22u8; 16])
        .restore(&bytes)
        .expect_err("a different nonce must be rejected");
    assert_rejected_with(err, "nonce mismatch");
}

#[test]
fn nonce_missing_but_required_is_rejected() {
    // Writer stamped no nonce; reader demands one.
    let bytes = capture(None, None);
    let err = SnapshotReader::new()
        .with_expected_nonce([0x33u8; 16])
        .restore(&bytes)
        .expect_err("a blob without a nonce must be rejected when one is required");
    assert_rejected_with(err, "nonce missing");
}

#[test]
fn nonce_present_but_unchecked_is_accepted() {
    // Backward-compat: a reader that does NOT call `with_expected_nonce`
    // accepts a blob that carries a nonce (the check is opt-in).
    let nonce = [0x77u8; 16];
    let bytes = capture(None, Some(nonce));
    let restored = SnapshotReader::new()
        .restore(&bytes)
        .expect("nonce-carrying blob must round-trip when the check is disabled");
    assert_eq!(restored.metadata.nonce, Some(nonce));
}

// ----- sequence_no floor: accept / reject on the legacy restore path -----

#[test]
fn sequence_no_at_floor_is_accepted() {
    let bytes = capture(Some(10), None);
    let restored = SnapshotReader::new()
        .with_min_sequence_no(10)
        .restore(&bytes)
        .expect("sequence_no == floor must be accepted");
    assert_eq!(restored.metadata.sequence_no, 10);
}

#[test]
fn sequence_no_above_floor_is_accepted() {
    let bytes = capture(Some(11), None);
    let restored = SnapshotReader::new()
        .with_min_sequence_no(10)
        .restore(&bytes)
        .expect("sequence_no > floor must be accepted");
    assert_eq!(restored.metadata.sequence_no, 11);
}

#[test]
fn sequence_no_below_floor_is_rejected() {
    let bytes = capture(Some(9), None);
    let err = SnapshotReader::new()
        .with_min_sequence_no(10)
        .restore(&bytes)
        .expect_err("sequence_no < floor must be rejected (rollback/replay)");
    assert_rejected_with(err, "below floor");
}

#[test]
fn sequence_no_floor_disabled_by_default() {
    // Backward-compat: with no floor set, even sequence_no 0 is accepted.
    let bytes = capture(Some(0), None);
    let restored = SnapshotReader::new()
        .restore(&bytes)
        .expect("default reader must accept any sequence_no");
    assert_eq!(restored.metadata.sequence_no, 0);
}

#[test]
fn nonce_and_sequence_floor_compose() {
    // Both checks active and both satisfied → accepted.
    let nonce = [0xEEu8; 16];
    let bytes = capture(Some(42), Some(nonce));
    SnapshotReader::new()
        .with_min_sequence_no(42)
        .with_expected_nonce(nonce)
        .restore(&bytes)
        .expect("both replay checks satisfied must accept");

    // Sequence floor fails first (it is checked before the nonce) → rejected.
    let below = capture(Some(41), Some(nonce));
    let err = SnapshotReader::new()
        .with_min_sequence_no(42)
        .with_expected_nonce(nonce)
        .restore(&below)
        .expect_err("a below-floor sequence must be rejected even with a matching nonce");
    assert_rejected_with(err, "below floor");
}
