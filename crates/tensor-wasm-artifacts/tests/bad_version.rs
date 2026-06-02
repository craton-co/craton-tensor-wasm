// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Version rejection: write a valid blob, flip the version field in the
//! header, assert `get` (disk) and `decode_envelope_from_bytes` (memory)
//! both reject with `BadVersion`.
//!
//! The version field lives at bytes [16, 20) of the header as a
//! little-endian `u32`, immediately after the 16-byte magic. Flipping
//! it past the magic check forces the version branch to be the
//! rejection — not `BadMagic`, not `BadHmac` (the version check runs
//! before any HMAC work).

use tensor_wasm_artifacts::{
    decode_envelope_from_bytes, encode_envelope_to_vec, ArtifactError, ArtifactStore,
    DiskArtifactStore, ARTIFACT_VERSION,
};

fn sole_artifact_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir").flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if name.ends_with(".bin") {
            hits.push(entry.path());
        }
    }
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one artifact file, found {hits:?}"
    );
    hits.into_iter().next().unwrap()
}

#[test]
fn disk_flipped_version_byte_returns_bad_version() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x9A; 32]);
    let payload = b"valid body before version corruption";
    let hash = store.put(payload).expect("put");

    // Sanity: un-tampered read works.
    assert_eq!(store.get(&hash).expect("get"), payload);

    // Bump the low byte of the version field to something unsupported.
    let path = sole_artifact_file(tmp.path());
    let mut bytes = std::fs::read(&path).expect("read");
    // Byte 16 is the LSB of the LE u32 version. ARTIFACT_VERSION is 1,
    // so setting it to 0xFF yields version 0xFF (unsupported).
    assert_eq!(
        bytes[16], ARTIFACT_VERSION as u8,
        "precondition: version LSB"
    );
    bytes[16] = 0xFF;
    std::fs::write(&path, &bytes).expect("write tampered");

    let err = store.get(&hash).expect_err("must reject bad version");
    match err {
        ArtifactError::BadVersion(v) => assert_ne!(v, ARTIFACT_VERSION, "got unsupported version"),
        other => panic!("expected BadVersion, got {other:?}"),
    }
}

#[test]
fn envelope_flipped_version_byte_returns_bad_version() {
    let key = [0x9A; 32];
    let mut framed = encode_envelope_to_vec(b"in-memory body", &key).expect("encode");
    assert_eq!(
        framed[16], ARTIFACT_VERSION as u8,
        "precondition: version LSB"
    );
    framed[16] = 0x7F;
    let err = decode_envelope_from_bytes(&framed, &key).expect_err("must reject bad version");
    assert!(matches!(err, ArtifactError::BadVersion(_)), "got {err:?}");
}
