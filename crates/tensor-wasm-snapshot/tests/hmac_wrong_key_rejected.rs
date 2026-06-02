// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Restore with a different HMAC key than the one used at capture must fail.
//!
//! The HMAC trailer is the only thing standing between a tenant's snapshot and
//! a foreign-key-restore attempt. This test asserts that a key mismatch is
//! surfaced as `TensorWasmError::Serialization` with a message that names the
//! HMAC/signature failure — not silently accepted, not surfaced as a generic
//! decode error.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// Key used at capture time.
const K1: [u8; 32] = [0xAB; 32];
/// Different key used at restore time. Differs from `K1` in every byte so the
/// HMAC output is guaranteed to diverge.
const K2: [u8; 32] = [0xCD; 32];

#[test]
fn restore_with_wrong_key_is_rejected() {
    let wasm: Vec<u8> = (0u8..128).collect();
    let gpu: Vec<u8> = vec![0x77; 256];
    let regs: Vec<u8> = vec![0x33; 16];

    let blob = SnapshotWriter::new()
        .with_hmac_sha256_key(K1)
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(2),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture with K1");

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(K2)
        .restore(&blob)
        .expect_err("restore with K2 must reject");

    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            // Accept either "HMAC mismatch" or "signature mismatch" wording —
            // M8.1 picks the exact phrasing; we only require that the reader
            // surfaces it as a signature-class rejection, not a generic decode
            // failure.
            assert!(
                lower.contains("hmac")
                    || lower.contains("signature mismatch")
                    || lower.contains("signature is invalid")
                    || lower.contains("invalid signature"),
                "expected HMAC/signature mismatch rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
