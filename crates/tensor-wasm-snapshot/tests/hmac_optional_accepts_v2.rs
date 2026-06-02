// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Backward-compatibility: a default reader (no `require_signature`) must
//! still accept v2 unsigned snapshots after the v3 format ships.
//!
//! This is the smoke test behind the v3 rollout contract: existing v2 blobs
//! on disk continue to load through `SnapshotReader::new().restore(...)`.
//! If this test ever fails the team has accidentally made signing mandatory,
//! which is a breaking change for unsigned-tenant deployments.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};
use tensor_wasm_snapshot::SNAPSHOT_VERSION_V2;

#[test]
fn default_reader_accepts_unsigned_v2_blob() {
    let wasm: Vec<u8> = (0u8..64).collect();
    let gpu: Vec<u8> = vec![0x55; 128];
    let regs: Vec<u8> = vec![0xCC; 32];

    // Capture as v2 — no key on the writer.
    let blob = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(7),
            instance_id: InstanceId(42),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v2");

    // Default reader: no `require_signature`, no key. Must accept the v2 blob
    // and reproduce every byte.
    let restored = SnapshotReader::new()
        .restore(&blob)
        .expect("default reader must accept unsigned v2");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.version, SNAPSHOT_VERSION_V2);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(7));
    assert_eq!(restored.metadata.instance_id, InstanceId(42));
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
}
