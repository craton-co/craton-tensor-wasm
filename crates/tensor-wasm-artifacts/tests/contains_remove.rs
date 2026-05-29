// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `contains` / `remove` lifecycle for both store implementations.
//!
//! The contract under test:
//!   put -> contains == true
//!       -> remove == true  (an entry was deleted)
//!       -> contains == false
//!       -> remove == false (nothing left to delete; no-op, not an error)

use tensor_wasm_artifacts::{
    ArtifactStore, ContentHash, DiskArtifactStore, InMemoryArtifactStore,
};

fn exercise<S: ArtifactStore>(store: &S) {
    let payload = b"contains/remove lifecycle payload";
    let hash = store.put(payload).expect("put");

    assert!(store.contains(&hash).expect("contains after put"), "must exist after put");

    assert!(store.remove(&hash).expect("first remove"), "first remove deletes -> true");

    assert!(
        !store.contains(&hash).expect("contains after remove"),
        "must not exist after remove"
    );

    assert!(
        !store.remove(&hash).expect("second remove"),
        "second remove is a no-op -> false"
    );

    // After removal, `get` is a genuine miss.
    let err = store.get(&hash).expect_err("get after remove must miss");
    assert!(
        matches!(err, tensor_wasm_artifacts::ArtifactError::NotFound(_)),
        "got {err:?}"
    );
}

#[test]
fn in_memory_contains_remove_round_trip() {
    let store = InMemoryArtifactStore::new([0x21; 32]);
    exercise(&store);
}

#[test]
fn disk_contains_remove_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x21; 32]);
    exercise(&store);
}

#[test]
fn contains_absent_hash_is_false_without_error() {
    // A probe for a hash that was never stored returns Ok(false), not
    // an error — `contains` is a cheap existence check, and "absent"
    // is a normal answer.
    let mem = InMemoryArtifactStore::new([0x31; 32]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let disk = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x31; 32]);
    let absent = ContentHash::from_bytes([0xFF; 32]);

    assert!(!mem.contains(&absent).expect("mem contains absent"));
    // Disk dir does not even exist yet; probe must still answer false.
    assert!(!disk.contains(&absent).expect("disk contains absent"));
    assert!(!disk.remove(&absent).expect("disk remove absent"));
}

#[test]
fn disk_remove_unlinks_only_the_targeted_entry() {
    // Removing one entry leaves the others intact and listable.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x41; 32]);
    let h1 = store.put(b"first").expect("put 1");
    let h2 = store.put(b"second").expect("put 2");

    assert!(store.remove(&h1).expect("remove h1"));
    assert!(!store.contains(&h1).expect("h1 gone"));
    assert!(store.contains(&h2).expect("h2 stays"));
    assert_eq!(store.get(&h2).expect("get h2"), b"second");

    let listed = store.list().expect("list");
    assert_eq!(listed.len(), 1, "only h2 remains");
    assert_eq!(listed[0], h2);
}
