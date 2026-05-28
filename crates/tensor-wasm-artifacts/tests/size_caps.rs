// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Size-cap defences (T10): reject oversized uploads at `put` time and
//! reject zstd-bomb inputs at `get` time.
//!
//! Two scenarios:
//!
//! 1. `put` with `payload.len() > MAX_PAYLOAD_LEN` returns
//!    [`ArtifactError::TooLarge`] before the framed-buffer allocation
//!    runs. Without the cap, an attacker who can drive `put` could
//!    request a multi-GB internal `Vec` and OOM the process.
//!
//! 2. `get` over a hand-built envelope whose zstd body decompresses past
//!    [`MAX_DECOMPRESSED_LEN`] returns [`ArtifactError::TooLarge`]. The
//!    streaming decoder is driven through `Read::take(cap + 1)` so a
//!    high-ratio "zip bomb" cannot grow the destination buffer past the
//!    cap regardless of how large the logical payload claims to be.
//!    Marked `#[ignore]` because the bomb has to actually expand past
//!    1 GiB to hit the cap, which is too heavy for the default test
//!    suite — run with `cargo test -p tensor-wasm-artifacts -- --ignored`.

use std::io::Write as _;

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactStore, ContentHash, DiskArtifactStore, ARTIFACT_HEADER_LEN,
    ARTIFACT_HMAC_LEN, ARTIFACT_MAGIC, ARTIFACT_VERSION, DEFAULT_ZSTD_LEVEL, MAX_DECOMPRESSED_LEN,
    MAX_PAYLOAD_LEN,
};

/// Shared HMAC key for the crafted-blob test. Must match what we sign
/// the synthetic envelope with below so the `get` path reaches the
/// decompressor instead of bailing on `BadHmac`.
const TEST_KEY: [u8; 32] = [0x42u8; 32];

#[test]
fn put_rejects_payload_over_cap() {
    // Build a payload one byte past the cap. `MAX_PAYLOAD_LEN` is 256
    // MiB so this allocation is non-trivial but well within a CI box's
    // budget; what we're verifying is that the check runs *before* the
    // store's own internal framed-buffer allocation, which would double
    // that footprint and is exactly what the cap exists to prevent.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x11u8; 32]);
    let payload = vec![0u8; MAX_PAYLOAD_LEN + 1];

    let err = store.put(&payload).expect_err("must reject oversized payload");
    match err {
        ArtifactError::TooLarge { actual, limit } => {
            assert_eq!(actual, MAX_PAYLOAD_LEN + 1, "actual size in error");
            assert_eq!(limit, MAX_PAYLOAD_LEN, "limit in error");
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }

    // No file was written: the rejection happened before `create_dir_all`
    // or any tempfile activity, so the store dir is empty.
    let entries: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("read_dir")
        .flatten()
        .collect();
    assert!(
        entries.is_empty(),
        "no file should be persisted for rejected put, found {entries:?}"
    );
}

/// HMAC-sign a synthetic envelope using the same key as the disk store.
/// Duplicates the helper inside `lib.rs::hmac_tag` because that function
/// is private; the algorithm is fixed (HMAC-SHA256 over the prefix) so
/// keeping a parallel copy here is safe and avoids exposing the helper
/// publicly purely for tests.
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
/// Re-implemented here for the same reason `sign` above is.
fn key_fp_hex(key: &[u8; 32]) -> String {
    let h = blake3::hash(&key[..]);
    h.as_bytes()[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

#[test]
#[ignore = "allocates >1 GiB to actually trigger the decompressed-size cap; \
            run with `cargo test -p tensor-wasm-artifacts -- --ignored`"]
fn get_rejects_zstd_bomb_over_cap() {
    // Build a payload that, when decompressed, exceeds
    // `MAX_DECOMPRESSED_LEN` by exactly one byte. Using `cap + 1`
    // instead of the spec's suggested `2 * cap` halves the test's peak
    // RSS while still hitting the same code path (the decoder's `Take`
    // adapter rejects strictly past `cap`, so `cap + 1` is the minimum
    // payload that trips it).
    let plaintext_len = MAX_DECOMPRESSED_LEN + 1;
    let plaintext = vec![0u8; plaintext_len];
    // Level 1 is plenty: zeros compress to a tiny ratio regardless of
    // level, and a lower level keeps test wall-time down.
    let zstd_body = zstd::encode_all(plaintext.as_slice(), DEFAULT_ZSTD_LEVEL)
        .expect("zstd encode");
    // Drop the huge plaintext immediately so peak RSS is bounded by
    // `plaintext_len + zstd_body.len()` during encode and by
    // `zstd_body.len() + decompressed_buffer` during the get below.
    drop(plaintext);

    // The content hash recorded on disk is `blake3(plaintext)`. We
    // can't keep the plaintext around to hash it, but we don't actually
    // need its true hash: the decompressed-cap check fires *before* the
    // content-hash comparison, so any 32-byte placeholder in the
    // header is fine. Use zeros.
    let content_hash_field = [0u8; 32];

    // Assemble: magic || version || content_hash || zstd(body)
    let mut framed: Vec<u8> = Vec::with_capacity(
        ARTIFACT_HEADER_LEN + zstd_body.len() + ARTIFACT_HMAC_LEN,
    );
    framed.write_all(&ARTIFACT_MAGIC).unwrap();
    framed.write_all(&ARTIFACT_VERSION.to_le_bytes()).unwrap();
    framed.write_all(&content_hash_field).unwrap();
    framed.write_all(&zstd_body).unwrap();

    // Sign the prefix with the test key and append the tag.
    let tag = sign(&TEST_KEY, &framed);
    framed.extend_from_slice(&tag);

    // Drop the encoded body now that it's been absorbed into `framed`.
    drop(zstd_body);

    // Write the file at the path the store will look up. We feed `get`
    // a fake `ContentHash` that matches the file's name suffix; the
    // value of the lookup key itself doesn't matter once the file
    // opens because the decompressed-cap check intervenes well before
    // the hash comparison.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), TEST_KEY);
    let fake_hash = ContentHash(content_hash_field);
    let fname = format!("{}.{}.bin", fake_hash, key_fp_hex(&TEST_KEY));
    let target = tmp.path().join(&fname);
    std::fs::create_dir_all(tmp.path()).expect("mkdir");
    std::fs::write(&target, &framed).expect("write crafted blob");
    drop(framed);

    let err = store
        .get(&fake_hash)
        .expect_err("must reject zstd-bomb payload");
    match err {
        ArtifactError::TooLarge { actual, limit } => {
            assert!(
                actual > MAX_DECOMPRESSED_LEN,
                "actual {actual} should exceed cap {MAX_DECOMPRESSED_LEN}",
            );
            assert_eq!(limit, MAX_DECOMPRESSED_LEN, "limit in error");
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}
