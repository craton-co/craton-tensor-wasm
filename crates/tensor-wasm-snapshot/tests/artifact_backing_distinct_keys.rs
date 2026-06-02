// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! An artifact-backed snapshot written under key A is invisible to a reader
//! that constructs the store under key B.
//!
//! The `DiskArtifactStore` partitions on-disk filenames by the first 8 bytes
//! of `blake3(hmac_key)` (see `crates/tensor-wasm-artifacts/src/lib.rs`).
//! Two stores rooted at the same directory under different keys therefore
//! produce two distinct files for an identical payload, and a `get` under
//! the wrong key returns `NotFound` rather than a silent HMAC failure or a
//! cross-key oracle.
//!
//! This test exercises that property end-to-end through the snapshot
//! crate's opt-in adapter: write through key A's store, then attempt to
//! read through key B's store sharing the same backing directory. The
//! reader must surface a `NotFound`-class error.
//!
//! Gated on the `artifact-backing` cargo feature so the file is a no-op when
//! the feature is off.

#![cfg(feature = "artifact-backing")]

use tempfile::tempdir;
use tensor_wasm_artifacts::DiskArtifactStore;
use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

const KEY_A: [u8; 32] = [0x01; 32];
const KEY_B: [u8; 32] = [0x02; 32];

#[test]
fn artifact_backing_distinct_keys_partition_namespace() {
    let dir = tempdir().expect("tempdir");

    // Construct two stores sharing the backing directory but signing under
    // distinct keys. The key fingerprint is part of the on-disk filename,
    // so both can coexist in the same directory without collision.
    let store_a = DiskArtifactStore::new(dir.path().to_path_buf(), KEY_A);
    let store_b = DiskArtifactStore::new(dir.path().to_path_buf(), KEY_B);

    let wasm = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let gpu = vec![9u8; 64];
    let regs = vec![0xCC; 16];

    // Write under key A.
    let hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store_a,
        )
        .expect("capture under key A");

    // Sanity: store A can read its own write back.
    let restored = SnapshotReader::new()
        .restore_from_artifact_store(&store_a, &hash)
        .expect("restore under key A");
    assert_eq!(restored.wasm_memory, wasm);

    // Now attempt to read the same content hash through store B. The
    // disk filename is `{hash}.{key_fp(B)}.bin`, which does not exist on
    // disk — only `{hash}.{key_fp(A)}.bin` does. The store therefore
    // returns `ArtifactError::NotFound`, which the snapshot reader maps
    // into the generic `Serialization` variant. The error message must
    // mention "not found" to distinguish this case from an integrity
    // failure.
    let err = SnapshotReader::new()
        .restore_from_artifact_store(&store_b, &hash)
        .expect_err("read under wrong key must be rejected");
    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("not found"),
                "expected a NotFound rejection (key-fingerprinted filename \
                 partitioning), got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }

    // Defence in depth: store B's `list()` (via the artifact crate)
    // returns no entries because the on-disk file's filename does not
    // carry B's key fingerprint. We can verify this indirectly: writing
    // an unrelated payload through B produces a second file, and
    // reading store_a's hash through store_b still misses.
    let other_hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(2),
                instance_id: InstanceId(2),
                wasm_memory: &[10, 20, 30],
                gpu_memory: &[],
                registers: &[],
            },
            &store_b,
        )
        .expect("capture under key B");

    // The two captures produced distinct content hashes (different inputs +
    // different timestamps), so neither cross-resolves.
    let err_b_lookup_a = SnapshotReader::new()
        .restore_from_artifact_store(&store_b, &hash)
        .expect_err("B still cannot resolve A's hash");
    match err_b_lookup_a {
        TensorWasmError::Serialization(msg) => assert!(
            msg.to_ascii_lowercase().contains("not found"),
            "unexpected message: {msg}",
        ),
        other => panic!("expected Serialization error, got {other:?}"),
    }
    // And B can still read its own.
    let restored_b = SnapshotReader::new()
        .restore_from_artifact_store(&store_b, &other_hash)
        .expect("restore under key B");
    assert_eq!(restored_b.wasm_memory, vec![10u8, 20, 30]);
}
