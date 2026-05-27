// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Round-trip test for v3 HMAC-SHA256 signed snapshots.
//!
//! Captures an [`InstanceState`] with a 32-byte HMAC key, persists the
//! resulting v3 blob to a temp file, restores it through a reader configured
//! with the same key, and verifies every payload field is byte-for-byte
//! identical to the input. Mirrors `tests/round_trip.rs` shape, gated on the
//! `signed-snapshots` cargo feature so the file is a no-op when the feature
//! is off.

#![cfg(feature = "signed-snapshots")]

use std::fs;

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};
use tensor_wasm_snapshot::SNAPSHOT_VERSION_V3;
use tempfile::NamedTempFile;

/// Fixed 32-byte test key. Stable across runs so a regression in HMAC keying
/// surfaces as a deterministic failure rather than a flaky one.
const TEST_KEY: [u8; 32] = [0xAB; 32];

/// Build deterministic-but-not-trivial memory bodies so the test catches
/// off-by-one and aliasing bugs in the signed-snapshot writer.
fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..8192).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..4096)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..512).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

#[test]
fn signed_round_trip_through_temp_file() {
    let (wasm, gpu, regs) = synth_state();

    let writer = SnapshotWriter::new().with_hmac_sha256_key(TEST_KEY);
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture");

    // Persist to a temp file and read back via the filesystem to mimic a real
    // cold-start path. The signed blob must survive a write+read cycle with
    // no in-flight mutation.
    let tmp = NamedTempFile::new().expect("temp file");
    fs::write(tmp.path(), &blob).expect("write blob");
    let on_disk = fs::read(tmp.path()).expect("read blob");
    assert_eq!(on_disk, blob);

    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&on_disk)
        .expect("restore");
    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.version, SNAPSHOT_VERSION_V3);
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
    // Metadata timestamp is filled in by the writer; just sanity-check it's
    // non-zero on a host with a working clock.
    assert!(restored.metadata.created_unix_ms > 0);
}

#[test]
fn signed_round_trip_with_empty_bodies() {
    // Edge case: zero-length payload still produces a valid HMAC. Restoring
    // with the same key must succeed and yield empty bodies.
    let writer = SnapshotWriter::new().with_hmac_sha256_key(TEST_KEY);
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(0),
            instance_id: InstanceId(0),
            wasm_memory: &[],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture empty");
    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&blob)
        .expect("restore empty");
    assert!(restored.wasm_memory.is_empty());
    assert!(restored.gpu_memory.is_empty());
    assert!(restored.registers.is_empty());
    assert_eq!(restored.metadata.total_uncompressed_bytes, 0);
    assert_eq!(restored.version, SNAPSHOT_VERSION_V3);
}
