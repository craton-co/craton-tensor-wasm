// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `DiskArtifactStore` happy-path: put/get/list under a tempdir.

use tensor_wasm_artifacts::{ArtifactStore, ContentHash, DiskArtifactStore};

#[test]
fn disk_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x55; 32]);

    let payload = b"some opaque artifact bytes that compress nicely\
                    some opaque artifact bytes that compress nicely\
                    some opaque artifact bytes that compress nicely";
    let hash = store.put(payload).expect("put");
    assert_eq!(hash, ContentHash::from_bytes(blake3::hash(payload).into()));

    let got = store.get(&hash).expect("get");
    assert_eq!(got, payload);

    let listed = store.list().expect("list");
    assert_eq!(listed.len(), 1, "list must surface the single put");
    assert_eq!(listed[0], hash);
}

#[test]
fn disk_multiple_payloads_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x66; 32]);

    let payloads: Vec<Vec<u8>> = (0..8)
        .map(|i| (0..512).map(|n| ((n + i) & 0xFF) as u8).collect())
        .collect();
    let mut hashes = Vec::new();
    for p in &payloads {
        hashes.push(store.put(p).expect("put"));
    }

    // All eight come back intact.
    for (h, p) in hashes.iter().zip(payloads.iter()) {
        let got = store.get(h).expect("get");
        assert_eq!(&got, p);
    }

    let mut listed = store.list().expect("list");
    listed.sort_by_key(|h| *h.as_bytes());
    assert_eq!(listed.len(), 8);
}

#[test]
fn disk_get_missing_is_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x77; 32]);
    let hash = ContentHash::from_bytes([0xAA; 32]);
    let err = store.get(&hash).expect_err("must miss");
    assert!(matches!(
        err,
        tensor_wasm_artifacts::ArtifactError::NotFound(_)
    ));
}

#[test]
fn disk_put_is_idempotent_on_same_payload() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x88; 32]);
    let p = b"deterministic body";
    let h1 = store.put(p).unwrap();
    let h2 = store.put(p).unwrap();
    assert_eq!(h1, h2);
    assert_eq!(
        store.list().expect("list").len(),
        1,
        "second put overwrites in place"
    );
    assert_eq!(store.get(&h1).unwrap(), p);
}
