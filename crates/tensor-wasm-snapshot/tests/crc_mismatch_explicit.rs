// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Explicit CRC32 mismatch rejection.
//!
//! Hand-builds a deterministic `Snapshot`, mutates the wasm payload after the
//! `crc32` field has been computed (so the stored checksum is now wrong),
//! re-encodes via bincode+zstd, and asserts the reader rejects it specifically
//! because of the CRC mismatch — not because zstd or bincode framing broke
//! incidentally.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{Snapshot, SnapshotMetadata, DEFAULT_ZSTD_LEVEL, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};
use tensor_wasm_snapshot::payload_crc32;

#[test]
fn crc_mismatch_after_payload_mutation_is_rejected() {
    let wasm = vec![0x10u8, 0x20, 0x30, 0x40, 0x50];
    let gpu = vec![0xAAu8, 0xBB, 0xCC];
    let regs = vec![0x77u8; 8];

    // Compute the checksum from the ORIGINAL bytes; the snapshot stores this
    // value, then we deliberately corrupt one byte of `wasm_memory` so the
    // stored `crc32` no longer matches what the reader recomputes.
    let good_crc = payload_crc32(&wasm, &gpu, &regs);

    let mut snap = Snapshot {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
        wasm_memory: wasm.clone(),
        gpu_memory: gpu.clone(),
        registers: regs.clone(),
        metadata: SnapshotMetadata {
            tenant_id: TenantId(0xC0FFEE),
            instance_id: InstanceId(0xDEADBEEF),
            created_unix_ms: 42,
            total_uncompressed_bytes: (wasm.len() + gpu.len() + regs.len()) as u64,
        },
        crc32: good_crc,
    };

    // Mutate exactly one byte of the wasm payload AFTER setting the crc32
    // field so the stored checksum is now stale relative to the actual bytes.
    snap.wasm_memory[2] ^= 0xFF;

    let encoded = bincode::serialize(&snap).expect("bincode encode");
    let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd encode");

    let err = SnapshotReader::new()
        .restore(&compressed)
        .expect_err("CRC mismatch must be rejected");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("crc32"),
                "expected CRC mismatch rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
