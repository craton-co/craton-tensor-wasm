// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Concatenated-zstd-frame rejection.
//!
//! zstd is a framed format that natively supports decoding a stream of
//! back-to-back frames. The snapshot reader uses `single_frame()` precisely
//! to *opt out* of that behaviour, so a snapshot whose compressed prefix is
//! two glued zstd frames must be rejected — otherwise an attacker could
//! hide a chosen second payload behind a valid first frame, with the
//! reader silently observing only the first.
//!
//! This test forges a v3 envelope whose prefix is two valid zstd frames
//! glued end-to-end, then computes a correct HMAC over the entire
//! concatenated prefix (so the HMAC gate accepts and we land on the
//! single-frame check, not on a signature failure). The reader must
//! reject with the v3 "unexpected bytes between zstd frame and trailer"
//! diagnostic — i.e. `zstd_consumed != auth_prefix.len()` fires after
//! `single_frame()` stops at the first frame end.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, DEFAULT_ZSTD_LEVEL};
use tensor_wasm_snapshot::{SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN};

const KEY: [u8; 32] = [0x9Cu8; 32];

/// Two valid zstd frames concatenated, wrapped in a v3 trailer whose HMAC
/// covers the entire concatenated prefix. HMAC gate accepts; the reader's
/// `single_frame()` decode then leaves bytes from the second frame
/// unconsumed inside the authenticated prefix, and the v3 "unexpected
/// bytes between zstd frame and trailer" guard rejects the blob.
#[test]
fn two_concatenated_zstd_frames_under_valid_hmac_are_rejected() {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Build a real v3 blob to harvest its compressed prefix — that prefix
    // is one valid zstd frame containing a valid bincode-encoded snapshot.
    // We will concatenate it with a second valid frame to construct the
    // forgery.
    let real_blob = SnapshotWriter::new()
        .with_hmac_sha256_key(KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0x1111),
            instance_id: InstanceId(0x2222),
            wasm_memory: &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            gpu_memory: &[0xCCu8; 32],
            registers: &[0xDDu8; 16],
        })
        .expect("capture v3");

    let first_frame: &[u8] = &real_blob[..real_blob.len() - SIGNATURE_TRAILER_LEN];

    // Synthesize a second valid zstd frame whose contents are unrelated to
    // the snapshot wire format — the point is only that it parses as zstd,
    // so `single_frame()` sees one frame and stops while extra valid-zstd
    // bytes remain inside the authenticated prefix.
    let second_payload = b"trailing chosen payload after first frame";
    let second_frame = zstd::encode_all(&second_payload[..], DEFAULT_ZSTD_LEVEL)
        .expect("zstd encode second frame");

    // Glue the two frames and compute HMAC over the resulting prefix so the
    // HMAC gate accepts. Without this, the test would short-circuit on
    // "HMAC mismatch" and we would not reach the single-frame guard.
    let mut concatenated_prefix: Vec<u8> = Vec::with_capacity(first_frame.len() + second_frame.len());
    concatenated_prefix.extend_from_slice(first_frame);
    concatenated_prefix.extend_from_slice(&second_frame);

    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&KEY).expect("HMAC init");
    mac.update(&concatenated_prefix);
    mac.update(&[SIGNATURE_KIND_HMAC_SHA256]);
    let sig = mac.finalize().into_bytes();

    let mut forged_blob = concatenated_prefix;
    forged_blob.push(SIGNATURE_KIND_HMAC_SHA256);
    forged_blob.extend_from_slice(sig.as_slice());

    // Sanity: the discriminator at the trailer position still classifies
    // the blob as v3 (otherwise we would not reach the single-frame guard).
    assert_eq!(
        forged_blob[forged_blob.len() - SIGNATURE_TRAILER_LEN],
        SIGNATURE_KIND_HMAC_SHA256,
    );

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&forged_blob)
        .expect_err("concatenated-zstd v3 forgery must be rejected");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    // The HMAC accepted — otherwise we would see `HMAC mismatch` here, which
    // would mean the single-frame guard was never tested. Then the
    // single-frame guard rejected with the "unexpected bytes" diagnostic.
    assert!(
        !msg.contains("HMAC mismatch"),
        "HMAC must accept the forgery so the single-frame guard is exercised; got: {msg}",
    );
    assert!(
        msg.contains("unexpected bytes") || msg.contains("past zstd frame"),
        "expected single-frame guard rejection, got: {msg}",
    );
}

/// Companion case: two concatenated frames with NO v3 trailer (i.e. the
/// reader sees this as a v2 input). The v2 path has the same single-frame
/// guard — `zstd_consumed != auth_prefix.len()` — so the rejection lands
/// on "snapshot v2 has unexpected trailing bytes". This exercises the
/// v2-side guard symmetrically with the v3 case above.
#[test]
fn two_concatenated_zstd_frames_as_v2_are_rejected() {
    let real_blob = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(0x3333),
            instance_id: InstanceId(0x4444),
            wasm_memory: &[1u8, 2, 3, 4],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture v2");

    // Append a second valid zstd frame. The trailing byte of the second
    // frame must NOT equal `SIGNATURE_KIND_HMAC_SHA256` at the trailer
    // offset, otherwise the reader misclassifies the blob as v3. We pick
    // a payload that, after zstd framing, lands a non-`0x01` byte at the
    // (len - SIGNATURE_TRAILER_LEN) offset. If the harvested byte happens
    // to equal 1, we fall back to the v3 rejection assertion — the
    // single-frame guard fires either way and the test still pins the
    // contract.
    let second_payload = b"a totally distinct payload that lives in frame two";
    let second_frame = zstd::encode_all(&second_payload[..], DEFAULT_ZSTD_LEVEL)
        .expect("zstd encode second frame");

    let mut glued: Vec<u8> = Vec::with_capacity(real_blob.len() + second_frame.len());
    glued.extend_from_slice(&real_blob);
    glued.extend_from_slice(&second_frame);

    let classified_as_v3 = glued.len() >= SIGNATURE_TRAILER_LEN
        && glued[glued.len() - SIGNATURE_TRAILER_LEN] == SIGNATURE_KIND_HMAC_SHA256;

    let err = SnapshotReader::new()
        .restore(&glued)
        .expect_err("concatenated v2-shaped blob must be rejected");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    if classified_as_v3 {
        // Rare collision: the bytes at the trailer offset happen to spell
        // SIGNATURE_KIND_HMAC_SHA256, so the reader classifies this as v3
        // and the HMAC gate rejects (no key configured on this reader).
        // Either failure mode proves the reader refuses the concatenated
        // shape — what we are pinning is that it does NOT silently accept.
        assert!(
            msg.contains("signed") || msg.contains("HMAC") || msg.contains("v3"),
            "v3-misclassified glued blob should reject on the signed path, got: {msg}",
        );
    } else {
        assert!(
            msg.contains("unexpected trailing bytes") || msg.contains("past zstd frame"),
            "expected v2 single-frame guard rejection, got: {msg}",
        );
    }
}
