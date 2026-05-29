// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `InMemoryArtifactStore` happy-path: put/get/list.

use tensor_wasm_artifacts::{ArtifactStore, ContentHash, InMemoryArtifactStore};

#[test]
fn in_memory_round_trip() {
    let store = InMemoryArtifactStore::new([0x11; 32]);
    let payload = b"the rain in spain falls mainly on the plain";
    let hash = store.put(payload).expect("put");
    // Hash matches the BLAKE3 of the payload.
    let expected: [u8; 32] = blake3::hash(payload).into();
    assert_eq!(hash, ContentHash::from_bytes(expected));

    let got = store.get(&hash).expect("get");
    assert_eq!(got, payload);

    let listed = store.list().expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0], hash);
}

#[test]
fn in_memory_put_is_idempotent() {
    let store = InMemoryArtifactStore::new([0x22; 32]);
    let payload = b"abc";
    let h1 = store.put(payload).unwrap();
    let h2 = store.put(payload).unwrap();
    assert_eq!(h1, h2);
    assert_eq!(store.list().expect("list").len(), 1);
}

#[test]
fn in_memory_get_missing_is_not_found() {
    let store = InMemoryArtifactStore::new([0x33; 32]);
    let hash = ContentHash::from_bytes([0u8; 32]);
    let err = store.get(&hash).expect_err("must miss");
    assert!(matches!(
        err,
        tensor_wasm_artifacts::ArtifactError::NotFound(_)
    ));
}
