// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Concurrent `put` of the same payload from N threads must be
//! idempotent and panic-free.
//!
//! The trait documents that implementations are safe under concurrent
//! `put` for the same payload — a put after a put for the same content
//! hash is idempotent. `InMemoryArtifactStore` backs its map with a
//! `parking_lot::Mutex`, so racing inserts of identical bytes must
//! converge on a single entry whose hash every thread agrees on.

use std::sync::Arc;

use tensor_wasm_artifacts::{ArtifactStore, ContentHash, InMemoryArtifactStore};

#[test]
fn concurrent_same_payload_put_is_idempotent() {
    let store = InMemoryArtifactStore::new([0x5C; 32]);
    let payload = b"the very same bytes hammered from every thread at once";
    let expected = ContentHash::of(payload);

    const N: usize = 16;
    // `thread::scope` lets the borrowed `&store` cross into the threads
    // without `Arc`; each thread returns the hash it observed so we can
    // assert they all agree.
    let hashes: Vec<ContentHash> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..N)
            .map(|_| s.spawn(|| store.put(payload).expect("concurrent put")))
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread join")).collect()
    });

    // Every thread computed the same content hash.
    for h in &hashes {
        assert_eq!(*h, expected, "all threads agree on the content hash");
    }

    // The store holds exactly one entry despite N concurrent puts.
    let listed = store.list().expect("list");
    assert_eq!(listed.len(), 1, "concurrent identical puts collapse to one entry");
    assert_eq!(listed[0], expected);

    // And it reads back intact.
    assert_eq!(store.get(&expected).expect("get"), payload);
}

#[test]
fn concurrent_put_via_arc_shared_handle() {
    // Same property, exercised through the `Arc<dyn ArtifactStore>`
    // sharing shape the trait's `Send + Sync` bound exists to support.
    let store: Arc<dyn ArtifactStore> = Arc::new(InMemoryArtifactStore::new([0x6D; 32]));
    let payload = b"arc-shared concurrent payload";
    let expected = ContentHash::of(payload);

    const N: usize = 8;
    let mut handles = Vec::new();
    for _ in 0..N {
        let s = Arc::clone(&store);
        let p = payload.to_vec();
        handles.push(std::thread::spawn(move || s.put(&p).expect("put")));
    }
    for h in handles {
        assert_eq!(h.join().expect("join"), expected);
    }

    assert_eq!(store.list().expect("list").len(), 1);
    assert_eq!(store.get(&expected).expect("get"), payload);
}
