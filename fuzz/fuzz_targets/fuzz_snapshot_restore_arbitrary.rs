// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `SnapshotReader::restore` on the v0.4 artifact-envelope path.
//!
//! Complementary to `fuzz_snapshot_restore.rs` (which drives the v2 reader with
//! no key configured). This target steers the fuzzer onto the *v4* dispatch:
//! `SnapshotReader::restore` first classifies its input by the leading 16 bytes
//! and, when they match `tensor_wasm_artifacts::ARTIFACT_MAGIC`, hands the blob
//! to the artifact-store envelope decode path
//! (`decode_envelope_from_bytes_with_cap` — magic + version + HMAC + zstd +
//! BLAKE3 content-hash, then the bincode `Snapshot` inner-payload checks). That
//! path only runs when a reader HMAC key is configured, so we split the first
//! 32 bytes of fuzzer input into a synthetic key and prepend the artifact magic
//! to the remaining bytes before calling `restore`.
//!
//! The overwhelmingly common outcome is an HMAC mismatch — exactly the property
//! under test: a tampered or arbitrary v4 envelope must be rejected (as a
//! structured `TensorWasmError::Serialization`) without panicking or decoding
//! the inner payload. Occasionally the mutator lands a key+blob pair that
//! survives the trailer and drives the zstd / bincode path; either way,
//! `restore` must never panic. A panic from this target is a hard failure.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_artifacts::ARTIFACT_MAGIC;
use tensor_wasm_snapshot::reader::SnapshotReader;

fuzz_target!(|data: &[u8]| {
    // Need 32 bytes for the synthetic HMAC key prefix. Anything shorter can't
    // exercise the keyed v4 path, so skip it to keep the mutator productive.
    if data.len() < 32 {
        return;
    }
    // Cap the remaining payload length to the same 64 MiB ceiling the other
    // snapshot targets use; the reader's own input check rejects anything
    // bigger immediately.
    if data.len() - 32 > 64 * 1024 * 1024 {
        return;
    }
    let (key_bytes, rest) = data.split_at(32);
    let key: [u8; 32] = key_bytes
        .try_into()
        .expect("split_at(32) yields exactly 32 bytes");

    // Prepend the artifact magic so `restore`'s classifier dispatches onto the
    // v4 artifact-envelope decode rather than falling through to v3/v2. The
    // fuzzer mutates everything after the magic (header tail, zstd body, HMAC
    // tag), so it explores the full envelope-rejection surface.
    let mut blob = Vec::with_capacity(ARTIFACT_MAGIC.len() + rest.len());
    blob.extend_from_slice(&ARTIFACT_MAGIC);
    blob.extend_from_slice(rest);

    let _ = SnapshotReader::default()
        .with_hmac_sha256_key(key)
        .restore(&blob);
});
