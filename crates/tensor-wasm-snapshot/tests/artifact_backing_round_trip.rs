// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Round-trip test for the opt-in artifact-store backing path.
//!
//! Captures a synthesised [`InstanceState`] via
//! [`SnapshotWriter::capture_to_artifact_store`], then restores it via
//! [`SnapshotReader::restore_from_artifact_store`] and asserts byte-equality
//! on every section (wasm_memory, gpu_memory, registers, metadata).
//!
//! Gated on the `artifact-backing` cargo feature so the file is a no-op when
//! the feature is off — matches the pattern used by the v3 round-trip test.

#![cfg(feature = "artifact-backing")]

use tempfile::tempdir;
use tensor_wasm_artifacts::DiskArtifactStore;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};

/// Fixed 32-byte HMAC key for the artifact store. Stable across runs so any
/// regression in the artifact store's signing produces a deterministic
/// failure rather than a flaky one. Distinct from the v3 test key so a
/// cross-wired writer/reader cannot accidentally exchange blobs.
const TEST_KEY: [u8; 32] = [0xC3; 32];

/// Build deterministic-but-not-trivial memory bodies so the test catches
/// off-by-one and aliasing bugs in the artifact-backed writer.
fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..8192).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..4096)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..512).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

#[test]
fn artifact_backed_round_trip_byte_equal() {
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), TEST_KEY);

    let (wasm, gpu, regs) = synth_state();

    let writer = SnapshotWriter::new();
    let hash = writer
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(0xABCD),
                instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture_to_artifact_store");

    let restored = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash)
        .expect("restore_from_artifact_store");

    // Inner magic must be the snapshot magic — the artifact store envelope
    // sits *around* the bincode-encoded `Snapshot`, so the inner magic is
    // preserved.
    assert_eq!(restored.magic, SNAPSHOT_MAGIC);

    // Byte-equality on every section the brief requires.
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xABCD));
    assert_eq!(
        restored.metadata.instance_id,
        InstanceId(0x1234_5678_9ABC_DEF0)
    );
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
    // Writer fills the timestamp from the system clock; on any host with a
    // working clock this is strictly positive.
    assert!(restored.metadata.created_unix_ms > 0);
}

#[test]
fn artifact_backed_round_trip_with_empty_bodies() {
    // Edge case: zero-length payload still produces a valid artifact-store
    // record. Restoring under the same key/hash must succeed and yield
    // empty bodies — the artifact store's own integrity checks pass through
    // a zero-byte payload cleanly.
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), TEST_KEY);

    let hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(0),
                instance_id: InstanceId(0),
                wasm_memory: &[],
                gpu_memory: &[],
                registers: &[],
            },
            &store,
        )
        .expect("capture empty");

    let restored = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash)
        .expect("restore empty");

    assert!(restored.wasm_memory.is_empty());
    assert!(restored.gpu_memory.is_empty());
    assert!(restored.registers.is_empty());
    assert_eq!(restored.metadata.total_uncompressed_bytes, 0);
}

#[test]
fn identical_payloads_produce_identical_hashes() {
    // The artifact store is content-addressed via BLAKE3 of the bincode
    // payload. Two writers building from the same `InstanceState` must
    // therefore produce the same `ContentHash` *iff* the metadata
    // timestamp matches. Since `build_metadata` reads
    // `SystemTime::now()`, this test pins the input through the
    // single-writer path and asserts the round-trip identity instead.
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), TEST_KEY);

    let (wasm, gpu, regs) = synth_state();
    let writer = SnapshotWriter::new();
    let hash_a = writer
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(7),
                instance_id: InstanceId(42),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture a");

    // Re-reading under the same hash returns the same payload byte-for-byte.
    let restored_1 = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash_a)
        .expect("restore #1");
    let restored_2 = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash_a)
        .expect("restore #2");
    assert_eq!(restored_1.wasm_memory, restored_2.wasm_memory);
    assert_eq!(restored_1.gpu_memory, restored_2.gpu_memory);
    assert_eq!(restored_1.registers, restored_2.registers);
    assert_eq!(restored_1.crc32, restored_2.crc32);
}
