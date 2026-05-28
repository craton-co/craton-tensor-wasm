// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tampering rejection: write a valid artifact, flip a byte inside the
//! signed prefix, assert `get` returns `BadHmac`.

use std::io::{Read, Write};

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactStore, DiskArtifactStore, ARTIFACT_HEADER_LEN, ARTIFACT_HMAC_LEN,
};

/// Locate the single `.bin` file the store wrote under `dir` and return
/// its path. The disk store names entries `{hash_hex}.{key_fp_hex}.bin`
/// so a fresh tempdir will hold exactly one match.
fn sole_artifact_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir").flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if name.ends_with(".bin") {
            hits.push(entry.path());
        }
    }
    assert_eq!(hits.len(), 1, "expected exactly one artifact file, found {hits:?}");
    hits.into_iter().next().unwrap()
}

#[test]
fn tampered_byte_in_zstd_body_returns_bad_hmac() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0xAB; 32]);
    let payload = b"the quick brown fox jumps over the lazy dog repeatedly enough \
                    to give zstd something to chew on so the body is not trivial";
    let hash = store.put(payload).expect("put");

    // Sanity: the un-tampered read works.
    assert_eq!(store.get(&hash).expect("get"), payload);

    // Locate the file and flip a byte inside the zstd body — past the
    // 52-byte header, before the trailing 32-byte HMAC. This avoids the
    // BadMagic / BadVersion short-circuits and forces the HMAC check to
    // be the rejection.
    let path = sole_artifact_file(tmp.path());
    let mut bytes = Vec::new();
    std::fs::OpenOptions::new()
        .read(true)
        .open(&path)
        .expect("open r")
        .read_to_end(&mut bytes)
        .expect("read");
    assert!(bytes.len() > ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN);
    let flip_at = ARTIFACT_HEADER_LEN + 1;
    bytes[flip_at] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("write tampered");

    let err = store.get(&hash).expect_err("must reject tampered artifact");
    assert!(
        matches!(err, ArtifactError::BadHmac),
        "expected BadHmac, got {err:?}"
    );
}

#[test]
fn tampered_hmac_tag_returns_bad_hmac() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0xCD; 32]);
    let payload = b"another payload to sign";
    let hash = store.put(payload).expect("put");

    let path = sole_artifact_file(tmp.path());
    let mut bytes = Vec::new();
    std::fs::OpenOptions::new()
        .read(true)
        .open(&path)
        .expect("open r")
        .read_to_end(&mut bytes)
        .expect("read");
    // Flip the very last byte (inside the HMAC tag).
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&path)
        .expect("open w")
        .write_all(&bytes)
        .expect("write");

    let err = store.get(&hash).expect_err("must reject");
    assert!(matches!(err, ArtifactError::BadHmac), "got {err:?}");
}

#[test]
fn tampered_magic_returns_bad_magic() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0xEF; 32]);
    let payload = b"x";
    let hash = store.put(payload).expect("put");

    let path = sole_artifact_file(tmp.path());
    let mut bytes = std::fs::read(&path).expect("read");
    bytes[0] ^= 0xFF; // corrupt the magic byte 0
    std::fs::write(&path, &bytes).expect("write");

    let err = store.get(&hash).expect_err("must reject");
    assert!(matches!(err, ArtifactError::BadMagic), "got {err:?}");
}
