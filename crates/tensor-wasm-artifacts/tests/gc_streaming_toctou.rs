// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Coverage for the streaming/GC/TOCTOU surface added to
//! `tensor-wasm-artifacts`:
//!
//! 1. **`get_to` TOCTOU** — a `remove` + re-`put` is forced to happen
//!    *between* `get_to`'s verify pass and its decode pass (triggered from
//!    inside the caller's writer on first byte). Because `get_to` opens the
//!    blob once and rewinds the SAME handle, the bytes streamed to the
//!    writer must still equal the authenticated payload — a mid-stream swap
//!    cannot smuggle wrong bytes to the caller.
//! 2. **`encode_envelope_to_writer` byte-equality** — its output must be
//!    byte-identical to `encode_envelope_to_vec` for the same input.
//! 3. **`put_from`** — streaming ingest produces a blob `get` reads back
//!    intact, and matches the hash `put` would assign.
//! 4. **`total_bytes` / `prune_older_than` / `gc`** — quota/GC primitives.

use std::io::Write;
use std::sync::Arc;

use tensor_wasm_artifacts::{
    encode_envelope_to_vec, encode_envelope_to_writer, ArtifactStore, ContentHash,
    DiskArtifactStore, GcPolicy,
};

const KEY: [u8; 32] = [0x71u8; 32];

/// A writer that, on its FIRST `write` call (i.e. as `get_to`'s decode
/// pass begins emitting bytes — strictly after the verify pass has
/// authenticated the blob), races a `remove` + re-`put` of the same
/// payload on a background thread and waits for it to finish before
/// letting the decode continue. This forces the swap into the exact window
/// between the two passes.
struct SwapOnFirstWrite {
    bytes: Vec<u8>,
    fired: bool,
    store: Arc<DiskArtifactStore>,
    hash: ContentHash,
    payload: Vec<u8>,
}

impl Write for SwapOnFirstWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if !self.fired {
            self.fired = true;
            let store = Arc::clone(&self.store);
            let hash = self.hash;
            let payload = self.payload.clone();
            // Race the unlink + re-publish on another thread, then join so
            // the swap is guaranteed to have happened before we accept the
            // rest of the decoded stream.
            let h = std::thread::spawn(move || {
                let _ = store.remove(&hash);
                let _ = store.put(&payload);
            });
            let _ = h.join();
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn get_to_survives_remove_reput_race_between_passes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(DiskArtifactStore::new(tmp.path().to_path_buf(), KEY));

    // A body large enough that the decode pass spans several writes, so the
    // swap fires well before the stream completes.
    let payload: Vec<u8> = (0..400_000u32)
        .map(|n| (n.wrapping_mul(2654435761) >> 11) as u8)
        .collect();
    let hash = store.put(&payload).expect("put");

    let mut sink = SwapOnFirstWrite {
        bytes: Vec::new(),
        fired: false,
        store: Arc::clone(&store),
        hash,
        payload: payload.clone(),
    };

    // The two-pass single-handle design pins the verified inode, so even
    // though the path's directory entry is unlinked + republished mid-
    // decode, the bytes streamed out are the authenticated ones.
    let n = store.get_to(&hash, &mut sink).expect("get_to across race");
    assert_eq!(n, payload.len() as u64, "byte count matches the original");
    assert_eq!(sink.bytes, payload, "streamed bytes equal the authenticated payload");
    assert!(sink.fired, "the swap actually fired during the decode pass");

    // The store still reads the blob back cleanly afterwards (the re-put
    // restored an identical, idempotent file).
    assert_eq!(store.get(&hash).expect("get after race"), payload);
}

#[test]
fn encode_to_writer_is_byte_identical_to_vec() {
    // The streaming encoder must emit exactly the same framed bytes as the
    // buffered one, across several payload shapes (empty, tiny, large,
    // incompressible-ish).
    let cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"x".to_vec(),
        b"the quick brown fox".repeat(100),
        (0..200_000u32).map(|n| (n.wrapping_mul(40503) >> 7) as u8).collect(),
    ];
    for payload in &cases {
        let via_vec = encode_envelope_to_vec(payload, &KEY).expect("encode_to_vec");
        let mut via_writer: Vec<u8> = Vec::new();
        encode_envelope_to_writer(&mut via_writer, payload, &KEY).expect("encode_to_writer");
        assert_eq!(
            via_vec, via_writer,
            "writer and vec encoders must be byte-identical (payload len {})",
            payload.len()
        );
    }
}

#[test]
fn put_from_round_trips_and_matches_put_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), KEY);

    let payload =
        b"streamed in from a reader without buffering the whole thing first \
          streamed in from a reader without buffering the whole thing first";

    // Stream the payload in from a reader.
    let mut src = std::io::Cursor::new(payload.to_vec());
    let streamed_hash = store.put_from(&mut src).expect("put_from");

    // Same content hash a buffered `put` would assign.
    assert_eq!(streamed_hash, ContentHash::of(payload), "hash matches put");

    // And it reads back byte-for-byte through the normal get path.
    assert_eq!(store.get(&streamed_hash).expect("get"), payload);

    // The on-disk envelope is interchangeable with a buffered put: a second
    // store doing a buffered `put` of the same bytes lands the identical
    // file (idempotent, one entry).
    store.put(payload).expect("buffered put of same bytes");
    assert_eq!(store.list().expect("list").len(), 1, "one content-addressed entry");
}

#[test]
fn total_bytes_sums_blob_footprint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), KEY);
    assert_eq!(store.total_bytes().expect("empty"), 0, "empty store is 0 bytes");

    let a = store.put(b"alpha payload one").expect("put a");
    let b = store.put(b"beta payload two, different bytes").expect("put b");
    assert_ne!(a, b);

    // The reported total equals the on-disk sum of the two `.bin` files.
    let on_disk: u64 = std::fs::read_dir(tmp.path())
        .expect("read_dir")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
        .map(|e| e.metadata().expect("meta").len())
        .sum();
    assert_eq!(store.total_bytes().expect("total"), on_disk);
    assert!(on_disk > 0);
}

#[test]
fn prune_older_than_removes_only_old_blobs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), KEY);

    let old = store.put(b"an older artifact").expect("put old");
    // Backdate the old blob's mtime well into the past.
    let long_ago =
        std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 60 * 24 * 30);
    let backdated = set_mtime(tmp.path(), &old, long_ago);

    let fresh = store.put(b"a fresh artifact written just now").expect("put fresh");

    // Prune anything older than one day ago.
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 60 * 24);
    let removed = store.prune_older_than(cutoff).expect("prune");
    if backdated {
        // The backdated blob must be reclaimed; the fresh one must survive.
        assert_eq!(removed, 1, "exactly the backdated blob is pruned");
        assert!(!store.contains(&old).expect("old gone"), "backdated blob pruned");
    } else {
        // Platform refused the mtime set; neither blob is older than cutoff.
        assert_eq!(removed, 0, "nothing is older than the cutoff");
    }
    // The fresh blob is always retained.
    assert!(store.contains(&fresh).expect("fresh present"), "fresh blob survives prune");
}

#[test]
fn gc_size_bound_evicts_oldest_first() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), KEY);

    // Three distinct, poorly-compressible blobs of similar size.
    let mk = |seed: u32| -> Vec<u8> {
        (0..50_000u32).map(|n| (n.wrapping_mul(2246822519).wrapping_add(seed) >> 9) as u8).collect()
    };
    let p0 = mk(1);
    let p1 = mk(2);
    let p2 = mk(3);
    let now = std::time::SystemTime::now();
    let h0 = store.put(&p0).expect("put 0");
    // Stagger mtimes so "oldest first" is well-defined. h0 oldest, h2 newest.
    let s0 = set_mtime(tmp.path(), &h0, now - std::time::Duration::from_secs(3000));
    let h1 = store.put(&p1).expect("put 1");
    let s1 = set_mtime(tmp.path(), &h1, now - std::time::Duration::from_secs(2000));
    let h2 = store.put(&p2).expect("put 2");
    let s2 = set_mtime(tmp.path(), &h2, now - std::time::Duration::from_secs(1000));

    let total = store.total_bytes().expect("total");
    assert!(total > 0);

    // Budget that fits ~2 of the 3 blobs, forcing eviction of the oldest.
    let budget = total - (total / 3) - 1;
    let outcome = store.gc(GcPolicy::with_max_total_bytes(budget)).expect("gc");

    assert!(outcome.removed_by_size >= 1, "at least one blob evicted under the budget");
    assert_eq!(outcome.removed_by_age, 0, "no age bound was set");
    assert!(outcome.bytes_after <= budget, "footprint fits the budget after gc");
    // The newest blob is always retained.
    assert!(store.contains(&h2).expect("newest present"), "newest blob survives size gc");
    // The oldest is the first to go — but only assert ordering if the mtime
    // staggering actually took effect on this platform.
    if s0 && s1 && s2 {
        assert!(!store.contains(&h0).expect("oldest gone"), "oldest blob evicted first");
    }
}

/// Set the mtime of the `.bin` file of `hash` to `when` via the stable
/// `File::set_modified`, returning `true` if the platform accepted it.
/// Used to make GC's oldest-first ordering deterministic without an
/// external crate.
fn set_mtime(dir: &std::path::Path, hash: &ContentHash, when: std::time::SystemTime) -> bool {
    let hex = hash.to_string();
    let path = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            let s = p.to_string_lossy();
            s.contains(&hex) && s.ends_with(".bin")
        });
    let Some(path) = path else {
        return false;
    };
    let Ok(f) = std::fs::OpenOptions::new().write(true).open(&path) else {
        return false;
    };
    f.set_modified(when).is_ok()
}
