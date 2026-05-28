// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! HMAC is verified *before* zstd or bincode are exercised on v3 input.
//!
//! Hoisting HMAC to "authenticate then parse" position means a forged or
//! tampered v3 blob must fail with the signature-mismatch error path —
//! never with a `bincode decode:` or `zstd decode:` error — because the
//! reader returns from the HMAC check before either decoder is touched.
//! Mirror property: a v3 blob whose HMAC is *valid* but whose inner
//! payload bytes are garbage must reach the decoder errors (proving the
//! accept-path doesn't short-circuit the rest of the validation pipeline).
//!
//! These two assertions together pin down the ordering contract that
//! makes the change a real security improvement and not just a code
//! reshuffle.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, DEFAULT_ZSTD_LEVEL};
use tensor_wasm_snapshot::{
    SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN, V3_TRAILER_MAGIC, V3_TRAILER_MAGIC_LEN,
};

/// Fixed HMAC key used by both capture and restore in this test file. Stable
/// across runs so any HMAC-keying regression surfaces deterministically.
const KEY: [u8; 32] = [0x5Au8; 32];

/// (1) Reject path: a v3 blob whose trailer has been mutated must fail with
/// the HMAC-mismatch error — and crucially, the error message must NOT name
/// `bincode` or `zstd`, proving the reader returned before decompression or
/// deserialisation ran.
#[test]
fn tampered_trailer_fails_with_hmac_error_before_decoders_run() {
    // Build a valid v3 blob.
    let mut blob = SnapshotWriter::new()
        .with_hmac_sha256_key(KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234),
            wasm_memory: &[1u8, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[0x99u8; 32],
            registers: &[0xAA; 4],
        })
        .expect("capture v3");

    assert!(
        blob.len() > SIGNATURE_TRAILER_LEN,
        "expected v3 blob to be > {} bytes (SIGNATURE_TRAILER_LEN), got {}",
        SIGNATURE_TRAILER_LEN,
        blob.len(),
    );

    // T8 sanity: the last `SIGNATURE_TRAILER_LEN` bytes are the trailer,
    // laid out as `[V3_TRAILER_MAGIC: 4][kind: 1][sig: 32]`. The magic
    // sits at offset `-SIGNATURE_TRAILER_LEN` and the kind byte sits
    // immediately after the magic.
    let trailer_start = blob.len() - SIGNATURE_TRAILER_LEN;
    assert_eq!(
        &blob[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN],
        &V3_TRAILER_MAGIC,
        "writer must emit V3_TRAILER_MAGIC at the trailer position",
    );
    assert_eq!(
        blob[trailer_start + V3_TRAILER_MAGIC_LEN],
        SIGNATURE_KIND_HMAC_SHA256,
        "writer must emit the HMAC-SHA256 kind byte after the trailer magic",
    );

    // Flip the last byte of the HMAC trailer (which is part of the 32-byte
    // signature — tampering this falsifies the HMAC without touching the
    // signature-kind byte, so the reader still classifies the blob as v3).
    let last = blob.len() - 1;
    blob[last] ^= 0x01;

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&blob)
        .expect_err("tampered HMAC trailer must be rejected");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    // The error must explicitly name the HMAC mismatch.
    assert!(
        msg.contains("HMAC mismatch"),
        "expected 'snapshot HMAC mismatch', got: {msg}",
    );

    // And — the load-bearing assertion for this ordering change — the
    // reader must NOT have reached the bincode or zstd decoders. If either
    // of those substrings appears in the message, the HMAC check ran AFTER
    // decoding (the bug we are fixing).
    let lower = msg.to_ascii_lowercase();
    assert!(
        !lower.contains("bincode"),
        "HMAC must reject before bincode runs; got: {msg}",
    );
    assert!(
        !lower.contains("zstd"),
        "HMAC must reject before zstd runs; got: {msg}",
    );
}

/// (2) Accept path: a v3 blob whose HMAC is *valid* over a corrupted inner
/// payload must still surface the inner decode failure. This proves the
/// reorder did not short-circuit the rest of the validation pipeline after
/// HMAC accepts — it really continues to decompress, decode, and check the
/// structural invariants.
#[test]
fn valid_hmac_over_corrupted_payload_still_reaches_bincode() {
    // Build a "compressed prefix" that the reader will treat as v3 but that
    // is deliberately garbage at the bincode layer. We zstd-encode 16 bytes
    // of `0xFF`, which decompresses to bytes that do NOT form a valid
    // `Snapshot` (the magic field will not match `SNAPSHOT_MAGIC`, and even
    // before that, the bincode decoder will fail trying to interpret the
    // wrong magic / version layout).
    //
    // We then compute HMAC-SHA256(KEY, compressed_prefix || V3_TRAILER_MAGIC
    // || [SIGNATURE_KIND_HMAC_SHA256]) over those exact bytes and append
    // `[V3_TRAILER_MAGIC][SIGNATURE_KIND_HMAC_SHA256][32-byte sig]`. The
    // result is a blob whose HMAC verifies cleanly with KEY — i.e. the
    // reader's "authenticate" step accepts it — but whose post-decompression
    // bincode decode (or magic check) must fail. That failure proves the
    // accept-path continues to the rest of the pipeline rather than
    // returning Ok prematurely.
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let garbage_uncompressed = vec![0xFFu8; 16];
    let compressed_prefix = zstd::encode_all(garbage_uncompressed.as_slice(), DEFAULT_ZSTD_LEVEL)
        .expect("zstd encode");

    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&KEY).expect("HMAC init");
    mac.update(&compressed_prefix);
    // T8: writer mixes V3_TRAILER_MAGIC into the HMAC input as well, so
    // a forged blob must match that ordering or the reader rejects it
    // with "HMAC mismatch" and the test cannot reach the bincode path
    // it is meant to exercise.
    mac.update(&V3_TRAILER_MAGIC);
    // snapshot 1.1: the writer also authenticates the signature-kind byte
    // so the verifier expects it in the HMAC input. Mirror here so the
    // forged blob actually verifies and the test reaches the bincode path.
    mac.update(&[SIGNATURE_KIND_HMAC_SHA256]);
    let sig = mac.finalize().into_bytes();

    let mut blob = compressed_prefix;
    blob.extend_from_slice(&V3_TRAILER_MAGIC);
    blob.push(SIGNATURE_KIND_HMAC_SHA256);
    blob.extend_from_slice(sig.as_slice());

    // Sanity: the trailer we just appended classifies the blob as v3 — the
    // magic prefix sits at `len - SIGNATURE_TRAILER_LEN` and the kind byte
    // sits one V3_TRAILER_MAGIC_LEN further in.
    let trailer_start = blob.len() - SIGNATURE_TRAILER_LEN;
    assert_eq!(
        &blob[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN],
        &V3_TRAILER_MAGIC,
    );
    assert_eq!(
        blob[trailer_start + V3_TRAILER_MAGIC_LEN],
        SIGNATURE_KIND_HMAC_SHA256,
    );

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&blob)
        .expect_err("garbage payload under valid HMAC must still be rejected");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    // The HMAC accept-path completed successfully (otherwise we would see
    // "HMAC mismatch" here). Instead we expect the bincode decoder or the
    // magic check to surface the corruption — that is the proof the rest
    // of the pipeline still runs after HMAC accepts.
    assert!(
        !msg.contains("HMAC mismatch"),
        "HMAC over compressed_prefix was valid; got HMAC error anyway: {msg}",
    );
    let lower = msg.to_ascii_lowercase();
    assert!(
        lower.contains("bincode") || lower.contains("magic") || lower.contains("version"),
        "expected bincode/magic/version rejection from the post-HMAC pipeline, got: {msg}",
    );
}
