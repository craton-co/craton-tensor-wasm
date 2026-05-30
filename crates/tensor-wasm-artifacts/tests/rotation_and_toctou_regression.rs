// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression coverage for two recently fixed defects in
//! [`tensor_wasm_artifacts::DiskArtifactStore`]:
//!
//! 1. **Rotation-aware `contains` / `remove` / `metadata`.** Under a
//!    [`RotatingKeyProvider`] (one active key plus accepted retired keys),
//!    a blob written under a now-retired key used to be invisible to
//!    `contains`, un-removable by `remove`, and its sidecar unreadable by
//!    `metadata` — all three only probed the *active* key's namespace. That
//!    let GC leak retired-key blobs (they round-tripped via `get`/`list`,
//!    which were already rotation-aware, but could never be detected or
//!    reclaimed). These tests assert the post-fix behaviour: all three
//!    resolve against every accepted read key.
//!
//! 2. **`get_to` single-handle two-pass (TOCTOU close).** `get_to` now opens
//!    the blob file ONCE and rewinds between the verify pass and the decode
//!    pass, so the bytes streamed to the caller's writer are exactly the
//!    bytes the HMAC authenticated. These tests assert the happy-path stream
//!    equals the payload and that a tampered on-disk blob surfaces as
//!    `BadHmac` with NOT ONE unverified byte reaching the writer.

use std::io::Write;
use std::sync::Arc;

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactMetadata, ArtifactStore, ContentHash, DiskArtifactStore,
    RotatingKeyProvider, ARTIFACT_HEADER_LEN, ARTIFACT_HMAC_LEN,
};

const K1: [u8; 32] = [0x0A; 32];
const K2: [u8; 32] = [0x0B; 32];

/// Locate the single `.bin` blob file the store wrote under `dir`. The disk
/// store names entries `{hash_hex}.{key_fp_hex}.bin`, so a fresh tempdir
/// holds exactly one match. Mirrors the helper in `tamper_rejected.rs` /
/// `get_to_streaming.rs`.
fn sole_blob_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir").flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if name.ends_with(".bin") {
            hits.push(entry.path());
        }
    }
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one blob file, found {hits:?}"
    );
    hits.into_iter().next().unwrap()
}

/// A writer that records every byte handed to it, so tests can prove a
/// tampered blob yields `BadHmac` *without* any decoded byte reaching the
/// sink. Mirrors `RecordingWriter` in `get_to_streaming.rs`.
struct RecordingWriter {
    bytes: Vec<u8>,
}

impl Write for RecordingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// =====================================================================
// Rotation-aware contains / remove / metadata
// =====================================================================

/// A blob written under K1 (the legacy single-key ctor) must be reported
/// present by `contains` after rotating to a provider whose active key is
/// K2 and whose accepted-read set still includes K1.
///
/// Pre-fix: `contains` only probed the active (K2) namespace, so this
/// returned `Ok(false)` even though `get` could read the blob — a GC leak.
#[test]
fn contains_finds_retired_key_blob() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    // Phase 1: write under the retired key K1 via the legacy ctor.
    let old_store = DiskArtifactStore::new(dir.clone(), K1);
    let payload = b"blob written under the soon-to-be-retired key";
    let hash = old_store.put(payload).expect("put under K1");

    // Phase 2: rotate — active key is K2, K1 is still accepted for reads.
    let provider = Arc::new(RotatingKeyProvider::new(K2, [K1]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);

    // The fix: `contains` resolves against every accepted read key, so the
    // retired-key blob is reported present (pre-fix this was false).
    assert!(
        rotated.contains(&hash).expect("contains across rotation"),
        "contains must find a blob written under an accepted retired key"
    );

    // And it is genuinely readable — `contains` did not lie.
    assert_eq!(
        rotated.get(&hash).expect("get retired-key blob"),
        payload,
        "the blob contains reported present must actually round-trip"
    );
}

/// A retired-key blob must be `remove`-able through the rotated store, and
/// afterward both `contains` and `get` must report it gone.
///
/// Pre-fix: `remove` only unlinked under the active (K2) namespace, where
/// no file exists, so it returned `Ok(false)` and the retired-key file
/// leaked on disk forever.
#[test]
fn remove_unlinks_retired_key_blob() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let old_store = DiskArtifactStore::new(dir.clone(), K1);
    let hash = old_store
        .put(b"retire me and reclaim me")
        .expect("put under K1");

    let provider = Arc::new(RotatingKeyProvider::new(K2, [K1]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);

    // Sanity: present before removal (also exercises the contains fix).
    assert!(rotated.contains(&hash).expect("present pre-remove"));

    // The fix: `remove` resolves the holding key (K1) and unlinks there,
    // so it reports a real deletion (pre-fix this was false — a no-op).
    assert!(
        rotated.remove(&hash).expect("remove retired-key blob"),
        "remove must unlink a blob written under an accepted retired key"
    );

    // After removal the blob is gone by every measure.
    assert!(
        !rotated.contains(&hash).expect("absent post-remove"),
        "contains must report the removed retired-key blob gone"
    );
    let err = rotated.get(&hash).expect_err("get after remove must miss");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");

    // The on-disk file is actually unlinked, not merely hidden.
    let remaining: Vec<_> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .flatten()
        .map(|e| e.file_name().into_string().unwrap_or_default())
        .filter(|n| n.ends_with(".bin"))
        .collect();
    assert!(
        remaining.is_empty(),
        "retired-key blob file must be unlinked, found {remaining:?}"
    );

    // A second remove is a no-op (mirrors the contains_remove contract).
    assert!(
        !rotated.remove(&hash).expect("second remove is a no-op"),
        "removing an already-gone blob returns false, not an error"
    );
}

/// `metadata` must read the sidecar of a blob whose `.bin` lives under an
/// accepted retired key.
///
/// Pre-fix: `metadata` derived the sidecar path from the active (K2) key's
/// fingerprint only, so a K1-written sidecar surfaced as `NotFound` after
/// rotation even though the blob and its sidecar were both on disk.
#[test]
fn metadata_reads_retired_key_sidecar() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let payload = b"artifact with provenance, written pre-rotation";
    let meta = ArtifactMetadata {
        created_unix_ms: 1_716_900_000_000,
        original_len: payload.len() as u64,
        source_tier: "jit-l2".to_string(),
    };

    // Write blob + sidecar under the retired key K1.
    let old_store = DiskArtifactStore::new(dir.clone(), K1);
    let hash = old_store
        .put_with_metadata(payload, &meta)
        .expect("put_with_metadata under K1");

    // Rotate to K2-active / K1-accepted.
    let provider = Arc::new(RotatingKeyProvider::new(K2, [K1]));
    let rotated = DiskArtifactStore::with_key_provider(dir.clone(), provider);

    // The fix: `metadata` resolves the holding key (K1) and reads its
    // sidecar (pre-fix this returned NotFound).
    let got = rotated
        .metadata(&hash)
        .expect("metadata across rotation boundary");
    assert_eq!(
        got, meta,
        "retired-key sidecar must come back byte-for-byte"
    );

    // And `remove` then takes the sidecar with the blob, even across
    // rotation (the sidecar must not outlive the blob).
    assert!(rotated.remove(&hash).expect("remove"), "blob removed");
    let err = rotated
        .metadata(&hash)
        .expect_err("sidecar gone after remove");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
}

// =====================================================================
// get_to single-handle two-pass / TOCTOU integrity
// =====================================================================

/// Happy path: `get_to` streams exactly the original payload, byte-for-byte
/// and in the right count, for a payload large enough to span several stream
/// buffers across both the verify pass and the decode pass.
#[test]
fn get_to_streams_only_authenticated_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x3C; 32]);

    // Pseudo-random-ish, compressible-but-not-trivial body that forces the
    // two passes (verify, then decode) to each iterate several buffers.
    let payload: Vec<u8> = (0..300_000u32)
        .map(|n| (n.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let hash = store.put(&payload).expect("put");

    let mut sink: Vec<u8> = Vec::new();
    let written = store.get_to(&hash, &mut sink).expect("get_to");

    assert_eq!(written, payload.len() as u64, "reported byte count matches");
    assert_eq!(sink, payload, "streamed bytes equal the original payload");

    // The single-handle two-pass must agree with the owned-buffer `get`.
    assert_eq!(store.get(&hash).expect("get"), sink, "get_to and get agree");
}

/// A byte flipped inside the zstd body must make `get_to` fail with
/// `BadHmac` and leave the caller's writer EMPTY: the single-handle two-pass
/// authenticates the whole blob on pass 1 before pass 2 decodes anything to
/// `out`, so no unverified byte is ever exposed.
#[test]
fn get_to_tampered_body_emits_no_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x7A; 32]);
    let payload = b"the quick brown fox jumps over the lazy dog, repeated enough \
                    times that zstd has a real body to chew through and the HMAC \
                    covers more than one stream buffer of compressed output here";
    let hash = store.put(payload).expect("put");

    // Sanity: the clean read streams the payload through the recorder.
    let mut clean = RecordingWriter { bytes: Vec::new() };
    store.get_to(&hash, &mut clean).expect("clean get_to");
    assert_eq!(clean.bytes, payload);

    // Flip a byte inside the zstd body: past the header, before the tag.
    let path = sole_blob_file(tmp.path());
    let mut bytes = std::fs::read(&path).expect("read blob");
    assert!(bytes.len() > ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN);
    bytes[ARTIFACT_HEADER_LEN + 2] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("write tampered");

    // The verify pass authenticates the whole blob BEFORE the decode pass
    // streams anything, so the tampered body surfaces as BadHmac with an
    // empty sink — no unverified bytes leaked to the writer.
    let mut sink = RecordingWriter { bytes: Vec::new() };
    let err = store
        .get_to(&hash, &mut sink)
        .expect_err("must reject tamper");
    assert!(matches!(err, ArtifactError::BadHmac), "got {err:?}");
    assert!(
        sink.bytes.is_empty(),
        "no unverified bytes may reach the writer on tamper; got {} bytes",
        sink.bytes.len()
    );
}

/// `get_to` for a hash that was never stored must miss with `NotFound` and
/// write nothing — the two-pass open finds no file on pass 1.
#[test]
fn get_to_missing_is_not_found_with_empty_sink() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x56; 32]);
    let hash = ContentHash::from_bytes([0x99; 32]);

    let mut sink: Vec<u8> = Vec::new();
    let err = store.get_to(&hash, &mut sink).expect_err("must miss");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
    assert!(sink.is_empty(), "no bytes written on a miss");
}
