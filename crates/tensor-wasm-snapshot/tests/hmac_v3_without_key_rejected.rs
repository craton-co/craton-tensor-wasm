// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! A signed v3 snapshot cannot be restored by a reader that has no HMAC key.
//!
//! Conceptually the reader has three signature configurations:
//!   1. No key, no `require_signature` — accepts v2, rejects v3 (this test).
//!   2. `require_signature` set — rejects v2 (see `hmac_required_rejects_v2.rs`).
//!   3. Key set — accepts v3 signed with that key (see `hmac_round_trip.rs`).
//!
//! Path 1 is the safety net for misconfigured deployments: a reader that
//! cannot verify the signature must never silently strip the trailer and
//! accept the payload as if it were v2 — that would defeat the entire point
//! of signing. The error message must name the missing key so operators can
//! diagnose the misconfiguration.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

const KEY: [u8; 32] = [0xAB; 32];

#[test]
fn signed_v3_restore_without_key_is_rejected() {
    let blob = SnapshotWriter::new()
        .with_hmac_sha256_key(KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234),
            wasm_memory: &[1, 2, 3, 4],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture signed v3");

    // Default reader: no `with_hmac_sha256_key`. It must refuse the v3 blob
    // rather than silently strip the trailer.
    let err = SnapshotReader::new()
        .restore(&blob)
        .expect_err("v3 restore without key must be rejected");

    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("no hmac key configured")
                    || lower.contains("no key configured")
                    || lower.contains("hmac key")
                    || lower.contains("missing key"),
                "expected missing-HMAC-key rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
