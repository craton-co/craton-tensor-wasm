// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Decompressed-stream cap (zip-bomb defence).
//!
//! Compresses a payload of repeated zeros that fits in a handful of bytes once
//! zstd is done with it, but blows up to hundreds of kilobytes on decode. The
//! reader is configured with a tight `with_max_decompressed` ceiling so the
//! test stays fast and the assertion is unambiguous about *which* limit fired.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_snapshot::reader::SnapshotReader;

/// Highly-compressible payload size: well past the reader's cap below so the
/// streaming decoder is forced to stop short and surface a rejection.
const BOMB_DECOMPRESSED_BYTES: usize = 256 * 1024;
const READER_CAP: usize = 1024;

#[test]
fn payload_decompressing_past_cap_is_rejected() {
    let zeros = vec![0u8; BOMB_DECOMPRESSED_BYTES];
    let compressed = zstd::encode_all(zeros.as_slice(), 22).expect("zstd encode");
    assert!(
        compressed.len() < READER_CAP,
        "compressed bomb should be tiny ({} bytes); test setup invariant violated",
        compressed.len(),
    );

    let reader = SnapshotReader::new().with_max_decompressed(READER_CAP);
    let err = reader
        .restore(&compressed)
        .expect_err("zip-bomb payload must be rejected by decompressed cap");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("decompressed payload too large"),
                "expected decompressed-cap rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}

#[test]
fn payload_under_cap_still_round_trips_through_default_reader() {
    // Sanity: with the default 256 MiB cap, a real captured snapshot still
    // restores fine — the new streaming-decoder code path did not break the
    // happy path.
    use tensor_wasm_core::types::{InstanceId, TenantId};
    use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(7),
            instance_id: InstanceId(7),
            wasm_memory: &vec![0u8; 64 * 1024],
            gpu_memory: &vec![1u8; 32 * 1024],
            registers: &[0xCDu8; 64],
        })
        .expect("capture");
    let restored = SnapshotReader::new().restore(&bytes).expect("restore");
    assert_eq!(restored.wasm_memory.len(), 64 * 1024);
    assert_eq!(restored.gpu_memory.len(), 32 * 1024);
}
