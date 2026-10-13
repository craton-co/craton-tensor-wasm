// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integrity-failure coverage that goes *past* the HMAC gate, using
//! hand-crafted, correctly-signed blobs:
//!
//! 1. **`HashMismatch`** — a blob whose HMAC is valid (so we are past the
//!    signature check) but whose header content-hash field disagrees with
//!    what the body decompresses to. The defence-in-depth content-hash
//!    recheck (`lib.rs` `get` / `decode_to_writer`) must reject it.
//! 2. **`Decompression` (post-HMAC)** — a correctly-signed blob whose body
//!    is not a valid zstd frame. The MAC verifies first, then the decode
//!    fault surfaces as `Decompression` (not `BadHmac`).
//! 3. **Truncated file (`< min_len`)** — a `.bin` file shorter than
//!    header+HMAC must be rejected with `BadMagic` before any keyed work.
//!
//! Each test re-implements the fixed HMAC-SHA256-over-prefix signing and
//! the key fingerprint locally (the in-crate helpers are private); the
//! algorithm is frozen by the on-disk format so a parallel copy is safe.

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactStore, ContentHash, DiskArtifactStore, ARTIFACT_HEADER_LEN,
    ARTIFACT_HMAC_LEN, ARTIFACT_MAGIC, ARTIFACT_VERSION, DEFAULT_ZSTD_LEVEL,
};

const KEY: [u8; 32] = [0x42u8; 32];

/// HMAC-SHA256 over `bytes` under `key` — matches `lib.rs`'s `new_mac` +
/// `finalize_into_tag`.
fn sign(key: &[u8; 32], bytes: &[u8]) -> [u8; ARTIFACT_HMAC_LEN] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key[..]).expect("hmac key");
    mac.update(bytes);
    let out = mac.finalize().into_bytes();
    let mut tag = [0u8; ARTIFACT_HMAC_LEN];
    tag.copy_from_slice(out.as_slice());
    tag
}

/// 8-byte hex fingerprint of the key — matches `lib.rs::key_fingerprint_hex`.
fn key_fp_hex(key: &[u8; 32]) -> String {
    let h = blake3::hash(&key[..]);
    h.as_bytes()[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Write `framed` to the store dir under the `{hash}.{fp}.bin` name `get`
/// will look up for `hash`, returning the store rooted there.
fn store_with_blob(dir: &std::path::Path, hash: &ContentHash, framed: &[u8]) -> DiskArtifactStore {
    let fname = format!("{}.{}.bin", hash, key_fp_hex(&KEY));
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(dir.join(&fname), framed).expect("write crafted blob");
    DiskArtifactStore::new(dir.to_path_buf(), KEY)
}

#[test]
fn valid_hmac_but_wrong_header_hash_is_hash_mismatch() {
    // Body decompresses to `payload`, but the header records a DIFFERENT
    // 32-byte content-hash field. We then sign the (header || body) prefix
    // correctly, so the HMAC passes and the recheck is what rejects.
    let payload = b"defence-in-depth: header hash must match the decoded body";
    let zstd_body = zstd::encode_all(&payload[..], DEFAULT_ZSTD_LEVEL).expect("zstd encode");

    // A header hash that is deliberately NOT blake3(payload).
    let wrong_hash_field = [0xAAu8; 32];
    let true_hash = ContentHash::of(payload);
    assert_ne!(
        wrong_hash_field,
        *true_hash.as_bytes(),
        "precondition: header hash differs from the real one"
    );

    let mut framed = Vec::new();
    framed.extend_from_slice(&ARTIFACT_MAGIC);
    framed.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    framed.extend_from_slice(&wrong_hash_field);
    framed.extend_from_slice(&zstd_body);
    let tag = sign(&KEY, &framed);
    framed.extend_from_slice(&tag);

    // Look the blob up under the WRONG header hash (that is the filename
    // the crafted header claims). `get` will: pass the HMAC, decode the
    // body, recompute blake3 = true_hash, and find it disagrees with the
    // header's `wrong_hash_field` -> HashMismatch.
    let lookup = ContentHash::from_bytes(wrong_hash_field);
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = store_with_blob(tmp.path(), &lookup, &framed);

    let err = store
        .get(&lookup)
        .expect_err("must reject header/body hash disagreement");
    match err {
        ArtifactError::HashMismatch { .. } => {}
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

#[test]
fn valid_hmac_but_garbage_body_is_decompression() {
    // A correctly-signed blob whose body is NOT a zstd frame. The MAC
    // verifies (we sign it), so the failure must be `Decompression`, not
    // `BadHmac` — proving the decode runs only after authentication.
    let garbage_body = b"this is plainly not a zstd frame at all, just text bytes";
    // The header hash is irrelevant here (decode fails before the hash
    // recheck), so any 32 bytes work.
    let hash_field = [0u8; 32];

    let mut framed = Vec::new();
    framed.extend_from_slice(&ARTIFACT_MAGIC);
    framed.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    framed.extend_from_slice(&hash_field);
    framed.extend_from_slice(&garbage_body[..]);
    let tag = sign(&KEY, &framed);
    framed.extend_from_slice(&tag);

    let lookup = ContentHash::from_bytes(hash_field);
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = store_with_blob(tmp.path(), &lookup, &framed);

    let err = store
        .get(&lookup)
        .expect_err("must reject undecodable body");
    match err {
        ArtifactError::Decompression(_) => {}
        other => panic!("expected Decompression (post-HMAC), got {other:?}"),
    }
}

#[test]
fn truncated_file_below_min_len_is_bad_magic() {
    // A `.bin` file shorter than header+HMAC cannot hold a real envelope;
    // the length gate rejects it with `BadMagic` before any keyed work.
    let hash = ContentHash::from_bytes([0x33u8; 32]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let min_len = ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN;
    let stub = vec![0u8; min_len - 1]; // one byte short of the minimum
    let store = store_with_blob(tmp.path(), &hash, &stub);

    let err = store.get(&hash).expect_err("must reject truncated file");
    assert!(matches!(err, ArtifactError::BadMagic), "got {err:?}");

    // `get_to` shares the same gate (in `verify_blob`).
    let mut sink: Vec<u8> = Vec::new();
    let err2 = store
        .get_to(&hash, &mut sink)
        .expect_err("get_to must reject too");
    assert!(matches!(err2, ArtifactError::BadMagic), "got {err2:?}");
    assert!(sink.is_empty(), "no bytes emitted for a truncated file");
}
