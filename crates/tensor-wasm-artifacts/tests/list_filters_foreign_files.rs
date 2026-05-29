// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `DiskArtifactStore::list` foreign-file filtering.
//!
//! `list` enumerates the store directory and reconstructs a
//! `ContentHash` only for files whose name matches
//! `{64 hex chars}.{16-hex key fingerprint}.bin`. Anything else — a
//! wrong suffix, a wrong-length hash segment, a non-hex hash segment, or
//! a file written under a different key's fingerprint — must be skipped
//! silently so a junk file dropped in the cache dir cannot corrupt a
//! GC/audit listing.

use tensor_wasm_artifacts::{ArtifactStore, DiskArtifactStore};

const KEY: [u8; 32] = [0x5Eu8; 32];

/// 8-byte hex fingerprint of the key — matches `lib.rs::key_fingerprint_hex`.
fn key_fp_hex(key: &[u8; 32]) -> String {
    let h = blake3::hash(&key[..]);
    h.as_bytes()[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

#[test]
fn list_skips_foreign_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();
    let store = DiskArtifactStore::new(dir.clone(), KEY);

    // One genuine entry so `list` has something real to surface.
    let real_hash = store.put(b"the one real artifact").expect("put");

    let fp = key_fp_hex(&KEY);
    let valid_hash_hex = "ab".repeat(32); // 64 hex chars

    // Drop several decoys alongside it. None should appear in `list`.
    let decoys = [
        // Wrong suffix (right hash + key fp shape, but not `.bin`).
        format!("{valid_hash_hex}.{fp}.txt"),
        // Right suffix, but the hash segment is too short (63 chars).
        format!("{}.{fp}.bin", "cd".repeat(31) + "e"),
        // Right suffix + correct length, but the hash segment is non-hex
        // (contains 'z' / 'g' which are not hex digits).
        format!("{}.{fp}.bin", "zz".repeat(32)),
        // Correct hash shape but a DIFFERENT key fingerprint — belongs
        // to another store sharing the dir, must be partitioned out.
        format!("{valid_hash_hex}.{}.bin", "00".repeat(8)),
        // A totally unrelated file.
        "README.txt".to_string(),
    ];
    for name in &decoys {
        std::fs::write(dir.join(name), b"junk").expect("write decoy");
    }

    let listed = store.list().expect("list");
    assert_eq!(listed.len(), 1, "list must skip every foreign file, got {listed:?}");
    assert_eq!(listed[0], real_hash, "the sole listed entry is the genuine one");
}

#[test]
fn list_on_empty_or_missing_dir_is_empty() {
    // A store whose directory was never created (no `put` yet) lists
    // empty rather than erroring — the dir is created lazily.
    let tmp = tempfile::tempdir().expect("tempdir");
    let sub = tmp.path().join("never-created");
    let store = DiskArtifactStore::new(sub, KEY);
    assert!(store.list().expect("list missing dir").is_empty());
}
