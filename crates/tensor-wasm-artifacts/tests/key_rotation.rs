// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pluggable key provider + rotation: a store under a rotating provider
//! reads a blob written under an older, still-accepted key, and `list`
//! enumerates across the rotation boundary.

use std::sync::Arc;

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactStore, DiskArtifactStore, RotatingKeyProvider,
};

const OLD_KEY: [u8; 32] = [0x0A; 32];
const NEW_KEY: [u8; 32] = [0x0B; 32];

#[test]
fn rotating_provider_reads_blob_written_under_old_key() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    // Phase 1: write under the OLD key via the legacy single-key ctor.
    let old_store = DiskArtifactStore::new(dir.clone(), OLD_KEY);
    let payload = b"data written before rotation";
    let hash = old_store.put(payload).expect("put under old key");

    // Phase 2: rotate — active key is NEW, OLD is still accepted for read.
    let provider = Arc::new(RotatingKeyProvider::new(NEW_KEY, [OLD_KEY]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);

    // The rotated store can still read the old blob (tries NEW first,
    // falls through to OLD which holds the file).
    assert_eq!(
        rotated.get(&hash).expect("read old blob after rotation"),
        payload,
        "rotating provider must read blobs signed by an accepted older key"
    );

    // And it streams via get_to too.
    let mut sink: Vec<u8> = Vec::new();
    rotated.get_to(&hash, &mut sink).expect("get_to old blob");
    assert_eq!(sink, payload);
}

#[test]
fn single_key_store_cannot_read_other_keys_blob() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let old_store = DiskArtifactStore::new(dir.clone(), OLD_KEY);
    let hash = old_store.put(b"old").expect("put");

    // A plain single-key store under NEW does NOT accept OLD for reads,
    // so the partitioned namespace yields NotFound.
    let new_only = DiskArtifactStore::new(dir.clone(), NEW_KEY);
    let err = new_only.get(&hash).expect_err("must miss");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}

#[test]
fn list_spans_rotation_boundary() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    // Old blob under OLD key.
    let old_store = DiskArtifactStore::new(dir.clone(), OLD_KEY);
    let old_hash = old_store.put(b"old-era artifact").expect("put old");

    // New blob under NEW key, written through the rotated store.
    let provider = Arc::new(RotatingKeyProvider::new(NEW_KEY, [OLD_KEY]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);
    let new_hash = rotated.put(b"new-era artifact").expect("put new");

    // `list` unions across both accepted keys.
    let mut listed = rotated.list().expect("list");
    listed.sort_by_key(|h| *h.as_bytes());
    assert_eq!(listed.len(), 2, "list must span the rotation boundary");
    assert!(listed.contains(&old_hash));
    assert!(listed.contains(&new_hash));
}

#[test]
fn rotated_put_signs_under_active_key() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let provider = Arc::new(RotatingKeyProvider::new(NEW_KEY, [OLD_KEY]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);
    let hash = rotated.put(b"fresh").expect("put");

    // A single-key store under NEW (the active key) can read it; one
    // under OLD cannot — proving the put signed under the active key.
    let new_only = DiskArtifactStore::new(dir.clone(), NEW_KEY);
    assert_eq!(new_only.get(&hash).expect("new reads"), b"fresh");

    let old_only = DiskArtifactStore::new(dir.clone(), OLD_KEY);
    let err = old_only.get(&hash).expect_err("old must miss new put");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}
