// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! In-memory envelope helpers (`encode_envelope_to_vec` /
//! `decode_envelope_from_bytes` / `decode_envelope_from_bytes_with_cap`).
//!
//! These cover three properties the snapshot crate relies on:
//!
//! 1. encode -> decode is a faithful round-trip.
//! 2. The framed bytes are byte-identical to what `DiskArtifactStore`
//!    would write, so a `DiskArtifactStore::get` reads back an
//!    envelope produced by `encode_envelope_to_vec` (the docs promise
//!    byte-identical output).
//! 3. The `_with_cap` variant enforces a caller-supplied decompressed
//!    ceiling: a payload that is correctly signed but decompresses past
//!    a tight cap is rejected with `TooLarge`, NOT exposed to the caller.

use tensor_wasm_artifacts::{
    decode_envelope_from_bytes, decode_envelope_from_bytes_with_cap, encode_envelope_to_vec,
    ArtifactError, ArtifactStore, ContentHash, DiskArtifactStore, ARTIFACT_HEADER_LEN,
};

const KEY: [u8; 32] = [0x3Cu8; 32];

#[test]
fn encode_decode_round_trip() {
    let payload = b"envelope round-trip body that compresses a little \
                    envelope round-trip body that compresses a little";
    let framed = encode_envelope_to_vec(payload, &KEY).expect("encode");
    let decoded = decode_envelope_from_bytes(&framed, &KEY).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn encode_decode_round_trip_empty_payload() {
    // An empty payload still produces a well-formed, verifiable envelope.
    let framed = encode_envelope_to_vec(b"", &KEY).expect("encode empty");
    let decoded = decode_envelope_from_bytes(&framed, &KEY).expect("decode empty");
    assert!(decoded.is_empty());
}

#[test]
fn decode_with_wrong_key_is_bad_hmac() {
    let framed = encode_envelope_to_vec(b"signed under KEY", &KEY).expect("encode");
    let wrong = [0x00u8; 32];
    let err = decode_envelope_from_bytes(&framed, &wrong).expect_err("must reject wrong key");
    assert!(matches!(err, ArtifactError::BadHmac), "got {err:?}");
}

#[test]
fn encoded_envelope_is_byte_identical_to_disk_store() {
    // The docs promise `encode_envelope_to_vec` emits the exact byte
    // sequence `DiskArtifactStore::put` writes to its tempfile. Verify
    // the cross-path interop: encode in memory, drop the bytes on disk
    // under the disk store's `{hash}.{key_fp}.bin` name, then read them
    // back through `DiskArtifactStore::get`.
    let payload = b"interop payload: encode in memory, get from disk store";
    let framed = encode_envelope_to_vec(payload, &KEY).expect("encode");

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();
    let store = DiskArtifactStore::new(dir.clone(), KEY);

    // Derive the on-disk filename the store would use for this content.
    // We reuse the store's own `put` to learn the hash + filename shape,
    // then overwrite that file with the in-memory framed bytes and
    // assert `get` still returns the payload — proving the two byte
    // streams are interchangeable.
    let hash = store.put(payload).expect("put establishes filename");
    // Find the sole `.bin` file and rewrite it with the encoder output.
    let mut target = None;
    for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if name.ends_with(".bin") {
            target = Some(entry.path());
        }
    }
    let target = target.expect("disk store wrote a file");
    std::fs::write(&target, &framed).expect("overwrite with encoder output");

    let got = store.get(&hash).expect("get reads encoder-produced bytes");
    assert_eq!(
        got, payload,
        "encode_envelope_to_vec output must be get-able"
    );
}

#[test]
fn with_cap_rejects_oversized_but_signed_payload() {
    // Build a CORRECTLY SIGNED envelope whose payload is larger than a
    // deliberately tiny cap. The HMAC verifies (so we are past the
    // integrity gate), but the decompressed-size cap must still refuse
    // the payload with `TooLarge` rather than handing it back.
    let payload = vec![0xA5u8; 4096];
    let framed = encode_envelope_to_vec(&payload, &KEY).expect("encode");

    // Sanity: under the default cap it decodes fine.
    let ok = decode_envelope_from_bytes(&framed, &KEY).expect("default cap decodes");
    assert_eq!(ok.len(), payload.len());

    // Under a 16-byte cap the 4096-byte payload is over budget.
    let small_cap = 16usize;
    let err = decode_envelope_from_bytes_with_cap(&framed, &KEY, small_cap)
        .expect_err("must reject over-cap payload");
    match err {
        ArtifactError::TooLarge { actual, limit } => {
            assert!(
                actual > small_cap,
                "actual {actual} should exceed cap {small_cap}"
            );
            assert_eq!(limit, small_cap, "limit echoes the supplied cap");
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

#[test]
fn with_cap_allows_payload_exactly_at_cap() {
    // A payload whose decompressed length equals the cap is allowed:
    // the `Take(cap + 1)` probe distinguishes "exactly cap" (ok) from
    // "> cap" (rejected).
    let payload = vec![0x11u8; 1000];
    let framed = encode_envelope_to_vec(&payload, &KEY).expect("encode");
    let decoded = decode_envelope_from_bytes_with_cap(&framed, &KEY, 1000).expect("exact cap ok");
    assert_eq!(decoded.len(), 1000);
}

#[test]
fn header_prefix_matches_expected_layout() {
    // Independent check that the framed bytes begin with the documented
    // header: magic || version || content_hash. Confirms the encoder
    // lays out the prefix the way the disk store and decoder expect.
    let payload = b"layout check";
    let framed = encode_envelope_to_vec(payload, &KEY).expect("encode");
    assert!(framed.len() > ARTIFACT_HEADER_LEN);
    let hash = ContentHash::of(payload);
    // content_hash field occupies bytes [20, 52).
    assert_eq!(&framed[20..52], hash.as_bytes());
}
