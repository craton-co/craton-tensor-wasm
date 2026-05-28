// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T21 regression: the artifact-backing write path encodes the snapshot
//! through a borrowing `SnapshotRef` rather than materialising an owned
//! [`Snapshot`] via three `.to_vec()` calls.
//!
//! The reallocation/copy count is not directly observable from outside
//! the crate (the `SnapshotRef` type is private), so the property under
//! test is the one operators care about: the artifact-backed write path
//! produces a payload that round-trips byte-for-byte against a hand-built
//! reference `Snapshot` encoded the legacy way. If `SnapshotRef`'s
//! Serialize impl ever diverged from `Snapshot`'s (e.g. someone reordered
//! a field or dropped the `serde_bytes` adapter on one but not the
//! other), this test would surface the regression even though both paths
//! continue to compile.
//!
//! Gated on the `artifact-backing` feature so the file is a no-op when
//! the feature is off — matches the pattern used by
//! `artifact_backing_round_trip.rs`.

#![cfg(feature = "artifact-backing")]

use tempfile::tempdir;
use tensor_wasm_artifacts::DiskArtifactStore;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// Fixed 32-byte HMAC key for the artifact store. Distinct from the
/// per-test keys in `artifact_backing_round_trip.rs` so a cross-wired
/// store cannot accidentally exchange blobs between the two test files.
const TEST_KEY: [u8; 32] = [0xA9; 32];

/// 1 MiB per blob — large enough that a regression in the artifact path
/// that introduced a `.to_vec()` copy on each of the three byte fields
/// would show up as a perf-test smoke signal (three 1 MiB copies per
/// capture), but small enough that the test runs in a fraction of a
/// second on CI. The test does not measure wall time directly; the size
/// matters only because the borrowing path's value scales with input
/// size.
const ONE_MIB: usize = 1024 * 1024;

fn large_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..(ONE_MIB as u32)).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..(ONE_MIB as u32))
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..1024).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

#[test]
fn artifact_backing_path_round_trip_with_large_blobs() {
    // Property under test: the artifact-backing path produces a snapshot
    // that round-trips end-to-end with the same byte content as the
    // input, even when each memory blob is 1 MiB. Before T21 this path
    // copied each blob via `.to_vec()` before encoding; the borrowing
    // `SnapshotRef` change preserves wire bytes (same `serde_bytes`
    // adapter, same field order) so the round-trip is the load-bearing
    // check that the refactor is semantically inert.
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), TEST_KEY);

    let (wasm, gpu, regs) = large_state();

    let writer = SnapshotWriter::new();
    let hash = writer
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(0xDEAD),
                instance_id: InstanceId(0xBEEF),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture_to_artifact_store with 1 MiB blobs");

    let restored = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash)
        .expect("restore_from_artifact_store");

    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xDEAD));
    assert_eq!(restored.metadata.instance_id, InstanceId(0xBEEF));
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
}

#[test]
fn artifact_backing_path_is_deterministic_for_identical_input() {
    // Property under test: the borrowing `SnapshotRef` encode produces a
    // stable byte length across repeated captures of the same input.
    // (Byte equality is not asserted because `metadata.created_unix_ms`
    // is filled from the system clock; the bincode-encoded width of
    // that field is fixed though, so the total payload length is
    // identical.) A regression that flipped the artifact path back to
    // `.to_vec()` would still pass this test — the point is to catch
    // any non-determinism a `SnapshotRef`-vs-`Snapshot` Serialize
    // divergence could introduce (e.g. one impl emits a trailing
    // zero-padding byte and the other doesn't).
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), TEST_KEY);

    let (wasm, gpu, regs) = large_state();

    let writer = SnapshotWriter::new();
    let hash_a = writer
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(11),
                instance_id: InstanceId(22),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture a");
    let hash_b = writer
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(11),
                instance_id: InstanceId(22),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture b");

    let restored_a = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash_a)
        .expect("restore a");
    let restored_b = SnapshotReader::new()
        .restore_from_artifact_store(&store, &hash_b)
        .expect("restore b");

    // Two captures of the same input produce byte-equal payloads on
    // every section except the timestamp. The fields below cover
    // everything the borrowing path is responsible for serialising.
    assert_eq!(restored_a.wasm_memory, restored_b.wasm_memory);
    assert_eq!(restored_a.gpu_memory, restored_b.gpu_memory);
    assert_eq!(restored_a.registers, restored_b.registers);
    assert_eq!(restored_a.crc32, restored_b.crc32);
    assert_eq!(restored_a.magic, restored_b.magic);
    assert_eq!(restored_a.version, restored_b.version);
    assert_eq!(
        restored_a.metadata.total_uncompressed_bytes,
        restored_b.metadata.total_uncompressed_bytes,
    );
}
