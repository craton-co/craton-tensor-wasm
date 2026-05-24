// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Older-than-current snapshot versions are rejected.
//!
//! The existing reader tests only cover `SNAPSHOT_VERSION + 1` (future version);
//! this test covers `SNAPSHOT_VERSION - 1` (legacy version) to prove the reader
//! refuses both directions of skew instead of silently accepting a stale
//! schema and producing a half-restored instance.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{Snapshot, SnapshotMetadata, DEFAULT_ZSTD_LEVEL, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};
use tensor_wasm_snapshot::payload_crc32;

#[test]
fn snapshot_version_minus_one_is_rejected() {
    assert!(
        SNAPSHOT_VERSION >= 1,
        "test assumes there is a representable previous version",
    );

    let snap = Snapshot {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION - 1,
        wasm_memory: vec![],
        gpu_memory: vec![],
        registers: vec![],
        metadata: SnapshotMetadata {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            created_unix_ms: 0,
            total_uncompressed_bytes: 0,
        },
        crc32: payload_crc32(&[], &[], &[]),
    };

    let cfg = bincode::config::legacy();
    let encoded = bincode::serde::encode_to_vec(&snap, cfg).expect("bincode encode");
    let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd encode");

    let err = SnapshotReader::new()
        .restore(&compressed)
        .expect_err("older-than-current version must be rejected");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("version"),
                "expected version rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
