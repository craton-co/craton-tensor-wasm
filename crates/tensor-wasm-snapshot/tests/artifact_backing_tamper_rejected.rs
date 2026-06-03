// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! In-place byte tampering of an artifact-backed snapshot must be rejected.
//!
//! Captures a snapshot via the opt-in artifact-store path, locates the
//! single resulting `.bin` file on disk, flips one byte in the middle of it,
//! then asserts `restore_from_artifact_store` returns an error. The artifact
//! store's HMAC-SHA256 trailer (computed over `magic || version ||
//! content_hash || zstd(payload)` — see `crates/tensor-wasm-artifacts/src/lib.rs`)
//! is what catches the mutation; the snapshot crate's own CRC32 and inner
//! magic are not involved because the tamper does not survive the artifact
//! store's first integrity gate.
//!
//! Gated on the `artifact-backing` cargo feature so the file is a no-op when
//! the feature is off.

#![cfg(feature = "artifact-backing")]

use std::fs;

use tempfile::tempdir;
use tensor_wasm_artifacts::DiskArtifactStore;
use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// Fixed key for the capture step. The restore step uses the *same* key so
/// the rejection comes from the HMAC-over-tampered-bytes mismatch rather
/// than from a key swap (which is covered by the distinct-keys test).
const KEY: [u8; 32] = [0xAB; 32];

#[test]
fn artifact_backing_tamper_rejected() {
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), KEY);

    // Use a deliberately non-trivial payload so the on-disk file is large
    // enough for the mid-file byte flip to land inside the signed prefix
    // (not in the trailing HMAC tag — the HMAC tag is the last 32 bytes,
    // and flipping a byte there would also fail verification but for a
    // less interesting reason).
    let wasm: Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..2048)
        .map(|i| ((i.wrapping_mul(11)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = vec![0xCD; 256];

    let hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture");

    // Locate the resulting artifact file. There should be exactly one — the
    // disk store's `put` emits `{content_hash_hex}.{key_fp_hex}.bin`.
    let entries: Vec<_> = fs::read_dir(dir.path())
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "expected exactly one artifact file, found {} ({:?})",
        entries.len(),
        entries.iter().map(|e| e.file_name()).collect::<Vec<_>>(),
    );
    let path = entries[0].path();

    // Flip a byte in the middle of the file. We deliberately stay clear of
    // the trailing 32-byte HMAC tag so the tamper lands inside the signed
    // prefix — the property under test is that the HMAC catches mutation
    // of the AUTHENTICATED bytes, not of the tag itself.
    let mut bytes = fs::read(&path).expect("read artifact");
    assert!(
        bytes.len() > 96,
        "artifact file unexpectedly small: {} bytes",
        bytes.len(),
    );
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    fs::write(&path, &bytes).expect("rewrite artifact");

    // Restore must fail. The artifact store surfaces `ArtifactError::BadHmac`
    // (or `Decompression` if the flip landed inside the zstd frame and
    // corrupted the framing bytes before the HMAC check); the reader maps
    // either into the generic `Serialization` variant. Both outcomes
    // satisfy the integrity property — what matters is that the reader
    // does NOT return a valid `Snapshot` for tampered bytes.
    let err = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash)
        .expect_err("tampered artifact must be rejected");
    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            // The forwarded `ArtifactError` `Display` text contains
            // "hmac" for the BadHmac case or "zstd"/"decompression" if the
            // flip happened to land inside the zstd frame epilogue. Either
            // is an acceptable rejection mode.
            assert!(
                lower.contains("hmac")
                    || lower.contains("zstd")
                    || lower.contains("decompression")
                    || lower.contains("hash mismatch")
                    || lower.contains("bad magic")
                    || lower.contains("bad version"),
                "expected an artifact-store integrity rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
