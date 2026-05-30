// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `ArtifactMetadata` sidecar: `put_with_metadata` / `metadata` round-trip,
//! and the guarantee that a plain `put` writes no sidecar.

use tensor_wasm_artifacts::{ArtifactError, ArtifactMetadata, ArtifactStore, DiskArtifactStore};

#[test]
fn metadata_put_get_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x21; 32]);

    let payload = b"artifact bytes with provenance";
    let meta = ArtifactMetadata {
        created_unix_ms: 1_716_900_000_000,
        original_len: payload.len() as u64,
        source_tier: "jit-l2".to_string(),
    };

    let hash = store
        .put_with_metadata(payload, &meta)
        .expect("put_with_metadata");

    // The blob round-trips exactly like a plain put.
    assert_eq!(store.get(&hash).expect("get"), payload);

    // The sidecar comes back byte-for-byte.
    let got = store.metadata(&hash).expect("metadata");
    assert_eq!(got, meta);

    // The metadata sidecar is NOT surfaced by `list` (which only matches
    // `.bin` blobs): exactly one logical entry.
    assert_eq!(store.list().expect("list").len(), 1);
}

#[test]
fn plain_put_has_no_metadata() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x43; 32]);
    let hash = store.put(b"no sidecar here").expect("put");
    let err = store
        .metadata(&hash)
        .expect_err("plain put has no metadata");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}

#[test]
fn metadata_for_missing_blob_is_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x65; 32]);
    let hash = tensor_wasm_artifacts::ContentHash::from_bytes([0x07; 32]);
    let err = store.metadata(&hash).expect_err("missing");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}

#[test]
fn remove_drops_metadata_sidecar() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x87; 32]);
    let meta = ArtifactMetadata {
        created_unix_ms: 42,
        original_len: 3,
        source_tier: "snapshot".to_string(),
    };
    let hash = store.put_with_metadata(b"abc", &meta).expect("put");

    assert!(store.remove(&hash).expect("remove"), "blob removed");
    // The sidecar must not outlive the blob.
    let err = store.metadata(&hash).expect_err("sidecar gone");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}
