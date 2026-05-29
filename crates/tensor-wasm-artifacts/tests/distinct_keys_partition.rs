// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Two `DiskArtifactStore`s in the same directory under different HMAC
//! keys must produce distinct files (filename includes the 8-byte key
//! fingerprint) and must NOT each see the other's entries via `list`.

use tensor_wasm_artifacts::{ArtifactError, ArtifactStore, DiskArtifactStore};

#[test]
fn distinct_keys_produce_distinct_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let store_a = DiskArtifactStore::new(dir.clone(), [0xAA; 32]);
    let store_b = DiskArtifactStore::new(dir.clone(), [0xBB; 32]);

    // Same payload through both stores — same content hash, but the
    // on-disk filename diverges via the key-fingerprint suffix.
    let payload = b"identical payload across keys";
    let h_a = store_a.put(payload).expect("put A");
    let h_b = store_b.put(payload).expect("put B");
    assert_eq!(h_a, h_b, "content hash is key-independent");

    // The directory now holds two files (one per key).
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .flatten()
        .map(|e| e.file_name().into_string().unwrap_or_default())
        .filter(|n| n.ends_with(".bin"))
        .collect();
    assert_eq!(
        entries.len(),
        2,
        "two stores under same dir + same payload but different keys must yield two files; got {entries:?}"
    );

    // Each store sees only its own entry.
    let list_a = store_a.list().expect("list A");
    let list_b = store_b.list().expect("list B");
    assert_eq!(list_a.len(), 1, "store A sees only its own file");
    assert_eq!(list_b.len(), 1, "store B sees only its own file");
}

#[test]
fn distinct_keys_cannot_read_each_others_blobs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let store_a = DiskArtifactStore::new(dir.clone(), [0x01; 32]);
    let store_b = DiskArtifactStore::new(dir.clone(), [0x02; 32]);

    let payload_a = b"alpha";
    let hash_a = store_a.put(payload_a).expect("put A");

    // Store B asking for hash_a sees NotFound, not BadHmac — the
    // filename's key-fingerprint segment partitions the namespace
    // before any HMAC check runs, so B's `get` never opens A's file.
    let err = store_b.get(&hash_a).expect_err("must miss");
    assert!(
        matches!(err, ArtifactError::NotFound(_)),
        "expected NotFound from partitioned namespace, got {err:?}"
    );

    // A still reads its own entry normally.
    assert_eq!(store_a.get(&hash_a).expect("get A"), payload_a);
}
