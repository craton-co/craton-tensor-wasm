// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Hardening tests for `tensor-wasm-snapshot` capture and restore.
//!
//! Each test targets a single class of malicious or accidentally-corrupted
//! input documented in `docs/SECURITY-AUDIT.md`. They run the real public API
//! end-to-end (no internal mocking) so a regression in writer-side validation
//! or reader-side checksumming surfaces here, not in production.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{
    limits, InstanceState, Snapshot, SnapshotMetadata, SnapshotWriter, DEFAULT_ZSTD_LEVEL,
    SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// Convenience: capture a tiny valid snapshot for tests that need a baseline blob.
fn capture_tiny() -> Vec<u8> {
    SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[9, 9, 9, 9],
            registers: &[0xAB; 16],
        })
        .expect("baseline capture")
}

#[test]
fn restore_rejects_oversized_wasm_memory() {
    // Capture itself must refuse a Wasm memory blob larger than the cap so we
    // never even produce bytes for the reader to see. Use vec![0; cap+1] —
    // allocator pressure is acceptable for this test (1 GiB + 1) only if the
    // host has the memory; gate behind cfg(not(miri)) and skip in CI without
    // 4 GiB of RAM by checking the const directly.
    //
    // To stay portable we exercise the path with a smaller cap mirror: build
    // a vector exactly one byte over the cap. On constrained CI nodes the
    // allocator may refuse; treat that as test-skipped via `try_reserve`.
    let oversize = limits::MAX_WASM_MEMORY_BYTES.saturating_add(1);
    let mut wasm: Vec<u8> = Vec::new();
    if wasm.try_reserve_exact(oversize).is_err() {
        // Not enough RAM to materialise the oversized buffer on this host;
        // we cannot run the capture-side check here. The reader-side mirror
        // below still exercises the validation logic.
        return;
    }
    wasm.resize(oversize, 0);

    let err = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &wasm,
            gpu_memory: &[],
            registers: &[],
        })
        .expect_err("capture must reject oversized wasm memory");
    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("wasm_memory") && msg.contains("too large"),
                "unexpected error message: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}

#[test]
fn restore_rejects_oversized_registers_via_handcrafted_blob() {
    // Bypass the writer to exercise the reader-side cap without allocating a
    // full gigabyte. Hand-build a Snapshot with `registers.len()` just over
    // the registers cap, serialise it, and confirm the reader rejects it.
    let registers = vec![0u8; limits::MAX_REGISTERS_BYTES + 1];
    let crc32 = tensor_wasm_snapshot::payload_crc32(&[], &[], &registers);
    let snap = Snapshot {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
        wasm_memory: vec![],
        gpu_memory: vec![],
        registers,
        metadata: SnapshotMetadata {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            created_unix_ms: 0,
            total_uncompressed_bytes: (limits::MAX_REGISTERS_BYTES + 1) as u64,
        },
        crc32,
    };
    let cfg = bincode::config::legacy();
    let encoded = bincode::serde::encode_to_vec(&snap, cfg).expect("bincode");
    let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd");

    let err = SnapshotReader::new()
        .restore(&compressed)
        .expect_err("restore must reject oversized registers");
    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("registers") && msg.contains("too large"),
                "unexpected error message: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}

#[test]
fn restore_rejects_tampered_crc32() {
    let mut bytes = capture_tiny();
    // Flip a byte well past the magic/version header. We do not know the exact
    // offset of the wasm_memory bytes inside the compressed payload, but any
    // bit-flip in the middle of the zstd frame will either fail to decode (and
    // be rejected as a zstd error) or yield mutated bytes whose CRC no longer
    // matches the stored value. Either branch is an acceptable rejection.
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;

    let err = SnapshotReader::new()
        .restore(&bytes)
        .expect_err("tampered blob must be rejected");
    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };
    assert!(
        msg.contains("crc32")
            || msg.contains("zstd")
            || msg.contains("bincode")
            || msg.contains("magic")
            || msg.contains("version"),
        "expected zstd / bincode / crc32 / header rejection, got: {msg}",
    );
}

#[test]
fn restore_rejects_truncated_input() {
    let bytes = capture_tiny();
    // Snip the tail of the compressed payload mid-frame.
    let truncated = &bytes[..bytes.len().saturating_sub(8)];
    let err = SnapshotReader::new()
        .restore(truncated)
        .expect_err("truncated input must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));
}

#[test]
fn restore_rejects_magic_mismatch() {
    let reader = SnapshotReader::new();
    let err = reader
        .restore(b"NOTSNAPSHOT00000")
        .expect_err("non-snapshot bytes must be rejected");
    // The literal bytes are not a valid zstd frame, so we expect any
    // Serialization rejection — magic, zstd, or bincode all qualify.
    assert!(matches!(err, TensorWasmError::Serialization(_)));

    // Hand-craft a *valid* zstd-wrapped bincode blob with a corrupted magic to
    // exercise the magic check directly.
    let snap = Snapshot {
        magic: 0xDEAD_BEEF,
        version: SNAPSHOT_VERSION,
        wasm_memory: vec![],
        gpu_memory: vec![],
        registers: vec![],
        metadata: SnapshotMetadata {
            tenant_id: TenantId(0),
            instance_id: InstanceId(0),
            created_unix_ms: 0,
            total_uncompressed_bytes: 0,
        },
        crc32: tensor_wasm_snapshot::payload_crc32(&[], &[], &[]),
    };
    let cfg = bincode::config::legacy();
    let encoded = bincode::serde::encode_to_vec(&snap, cfg).expect("bincode");
    let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd");
    let err = reader
        .restore(&compressed)
        .expect_err("bad magic must be rejected");
    let msg = match err {
        TensorWasmError::Serialization(m) => m,
        other => panic!("expected Serialization error, got {other:?}"),
    };
    assert!(
        msg.contains("magic"),
        "expected magic rejection, got: {msg}"
    );
}

#[test]
fn valid_snapshot_still_round_trips() {
    // Regression guard: the new validation must not block legitimate captures.
    let bytes = capture_tiny();
    let restored = SnapshotReader::new().restore(&bytes).expect("restore");
    assert_eq!(restored.wasm_memory, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(restored.gpu_memory, vec![9, 9, 9, 9]);
    assert_eq!(restored.registers.len(), 16);
    assert_eq!(restored.version, SNAPSHOT_VERSION);
}
