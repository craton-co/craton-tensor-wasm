// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pins commit baf379c: the v3 HMAC input MUST mix the `signature_kind` byte
//! into the MAC computation. This test forges a v3 trailer by computing the
//! HMAC over the compressed prefix WITHOUT the kind byte and proves the
//! reader rejects it. This is the asymmetric direction of the pinned test
//! at `hmac_verified_before_decode.rs:128-131` (writer-side mix-in): there
//! we proved a forgery that DOES include the kind byte in the MAC input
//! reaches the post-HMAC pipeline; here we prove a forgery that OMITS the
//! kind byte never gets past the HMAC gate.
//!
//! Together the two tests pin both directions of the snapshot-1.1 contract:
//! the kind byte is part of the authenticated input, and a verifier that
//! forgot to mix it in would silently accept this blob — which is exactly
//! the downgrade primitive the commit was added to prevent.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, DEFAULT_ZSTD_LEVEL};
use tensor_wasm_snapshot::{SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN};

/// Fixed HMAC key — stable across runs so any regression surfaces
/// deterministically rather than flaking on a particular seed.
const KEY: [u8; 32] = [0x73u8; 32];

/// Forge a v3 trailer whose HMAC was computed over the compressed prefix
/// **without** the trailing signature-kind byte mixed in. A verifier that
/// also omits the kind byte from its MAC input would accept this blob; the
/// real verifier (which mixes it in, per writer.rs:452 and reader.rs:577)
/// must reject with `HMAC mismatch`.
#[test]
fn forged_trailer_without_kind_byte_in_mac_input_is_rejected() {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Capture a real v3 blob first so we get a legitimate zstd+bincode prefix
    // (i.e. one that would deserialise cleanly if the HMAC accepted). Using a
    // genuine prefix means a regression in the verifier — accepting this
    // forgery — would let the test fall through to the post-HMAC pipeline
    // and observably produce an Ok(_), making the failure mode loud.
    let real_blob = SnapshotWriter::new()
        .with_hmac_sha256_key(KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xCAFE),
            instance_id: InstanceId(0xBEEF),
            wasm_memory: &[1u8, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[0x42u8; 64],
            registers: &[0xAA; 16],
        })
        .expect("capture v3");

    // Split off the writer-emitted trailer; we are about to replace it with
    // a forged one that authenticates only the prefix bytes (no kind byte).
    let prefix_len = real_blob.len() - SIGNATURE_TRAILER_LEN;
    let compressed_prefix = &real_blob[..prefix_len];

    // Forgery: HMAC over the prefix ONLY — the kind byte is deliberately
    // omitted from the MAC input. The real writer mixes it in at
    // `writer.rs:452` (snapshot 1.1). A verifier that also omitted it would
    // accept this blob; the real verifier (reader.rs:577) must reject.
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&KEY).expect("HMAC init");
    mac.update(compressed_prefix);
    // NOTE: no `mac.update(&[SIGNATURE_KIND_HMAC_SHA256])` here — that omission
    // is the entire point of this test.
    let forged_sig = mac.finalize().into_bytes();

    let mut forged_blob: Vec<u8> = compressed_prefix.to_vec();
    forged_blob.push(SIGNATURE_KIND_HMAC_SHA256);
    forged_blob.extend_from_slice(forged_sig.as_slice());

    // Sanity: the forged blob is the same total length and the trailer
    // discriminator still classifies it as v3.
    assert_eq!(forged_blob.len(), real_blob.len());
    assert_eq!(
        forged_blob[forged_blob.len() - SIGNATURE_TRAILER_LEN],
        SIGNATURE_KIND_HMAC_SHA256,
        "forged blob must still classify as v3",
    );
    // And the prefix is byte-identical to the real one.
    assert_eq!(&forged_blob[..prefix_len], compressed_prefix);

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&forged_blob)
        .expect_err("forged trailer (no kind byte in MAC input) must be rejected");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    assert!(
        msg.contains("HMAC mismatch"),
        "expected 'HMAC mismatch', got: {msg}",
    );

    // Defence in depth: the reject must come from the HMAC gate, not from
    // the post-HMAC pipeline. If we ever see `bincode` or `zstd` in the
    // error message, the verifier accepted the forgery and the failure is
    // surfacing downstream — which is exactly the regression this test
    // pins against.
    let lower = msg.to_ascii_lowercase();
    assert!(
        !lower.contains("bincode"),
        "HMAC must reject forgery before bincode runs; got: {msg}",
    );
    assert!(
        !lower.contains("zstd"),
        "HMAC must reject forgery before zstd runs; got: {msg}",
    );
}

/// Companion sanity check: the *symmetric* forgery — HMAC computed over
/// `[prefix || kind_byte]` — IS accepted by the verifier (i.e. crosses the
/// HMAC gate and surfaces a downstream error). This is the positive control
/// that proves the negative test above is not just rejecting every forged
/// blob regardless of MAC input. Mirrors the accept-path test in
/// `hmac_verified_before_decode.rs` but over an arbitrary garbage payload
/// rather than a real prefix, so the downstream rejection is unambiguous.
#[test]
fn forged_trailer_with_kind_byte_in_mac_input_reaches_post_hmac_pipeline() {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Garbage uncompressed → garbage decompressed, so the post-HMAC pipeline
    // will fail at the magic / bincode layer. The point is to land *past*
    // the HMAC gate, proving the verifier accepted the trailer.
    let garbage = vec![0u8; 32];
    let compressed_prefix = zstd::encode_all(garbage.as_slice(), DEFAULT_ZSTD_LEVEL)
        .expect("zstd encode garbage");

    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&KEY).expect("HMAC init");
    mac.update(&compressed_prefix);
    // INCLUDING the kind byte — this is what the real writer does, and what
    // the real verifier expects. The blob crosses the HMAC gate.
    mac.update(&[SIGNATURE_KIND_HMAC_SHA256]);
    let sig = mac.finalize().into_bytes();

    let mut blob = compressed_prefix;
    blob.push(SIGNATURE_KIND_HMAC_SHA256);
    blob.extend_from_slice(sig.as_slice());

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&blob)
        .expect_err("garbage payload under valid HMAC must still surface a downstream error");

    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };

    // The HMAC gate accepted (otherwise we'd see "HMAC mismatch"); the
    // post-HMAC pipeline rejects the garbage. This is the proof that
    // mixing the kind byte into the MAC input is the actual difference
    // between the accepted and rejected forgeries above.
    assert!(
        !msg.contains("HMAC mismatch"),
        "control-case forgery (with kind byte in MAC) was unexpectedly rejected at HMAC gate: {msg}",
    );
}
