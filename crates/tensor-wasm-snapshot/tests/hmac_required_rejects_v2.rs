// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `require_signature()` on the reader must refuse v2 (unsigned) snapshots.
//!
//! Production callers in regulated tenants opt into strict mode by chaining
//! `SnapshotReader::new().require_signature().restore(...)`. That call must
//! reject any blob whose envelope is missing the v3 HMAC trailer, regardless
//! of whether the inner bincode payload itself round-trips. This test exercises
//! exactly the strict-mode path against a freshly-captured v2 blob (no key on
//! the writer) and asserts the error message names "signature required" or
//! "unsigned".

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

#[test]
fn require_signature_rejects_unsigned_v2_blob() {
    // Capture as v2 (default writer, no `with_hmac_sha256_key`).
    let blob = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[9, 9, 9, 9],
            registers: &[0xAB; 16],
        })
        .expect("capture v2");

    let err = SnapshotReader::new()
        .require_signature()
        .restore(&blob)
        .expect_err("strict reader must reject unsigned v2");

    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("unsigned")
                    || lower.contains("signature is required")
                    || lower.contains("signature required"),
                "expected unsigned/signature-required rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
