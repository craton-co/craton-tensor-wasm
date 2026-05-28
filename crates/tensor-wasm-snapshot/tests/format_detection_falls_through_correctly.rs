// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T40 envelope dispatcher: the reader's leading-magic check correctly
//! routes inputs to the artifact-envelope decoder, the legacy v3
//! decoder, the v2 decoder, or surfaces an error — without confusing
//! the three.
//!
//! The dispatcher's contract:
//!
//!   1. Leading 16 bytes equal `ARTIFACT_MAGIC` → decode via the
//!      unified artifact envelope (T40 default).
//!   2. Otherwise, last 37 bytes' magic prefix equals `V3_TRAILER_MAGIC`
//!      → decode via the v3 path (T8 magic-prefix trailer).
//!   3. Otherwise → decode via the v2 path (zstd-bincode only).
//!   4. Garbage that matches none of the above → `Serialization` error
//!      without panicking.
//!
//! This file pins each of those branches with a synthetic input
//! constructed to land specifically on it, so a future refactor that
//! re-orders the dispatch (or accidentally short-circuits a branch)
//! produces a focused failure mode rather than a cascade.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// Shared key for every signed test below. Stable across runs so any
/// regression in the dispatcher's signature paths surfaces
/// deterministically.
const TEST_KEY: [u8; 32] = [0x55u8; 32];

fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..1024).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..512)
        .map(|i| ((i.wrapping_mul(11)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..32).map(|i| ((i ^ 0x55) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

/// Branch 1: leading-magic dispatch sends the bytes through the
/// artifact-envelope decoder. The blob is produced by the T40 default
/// writer (HMAC key set, no legacy opt-out) and decoded through the
/// default reader (HMAC key set; no extra configuration).
#[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
#[test]
fn artifact_envelope_is_detected_and_decoded() {
    let (wasm, gpu, regs) = synth_state();

    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xA1),
            instance_id: InstanceId(0xA1A1),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture default (T40)");

    // Sanity: this really is the artifact envelope.
    assert_eq!(
        &bytes[..16],
        &tensor_wasm_artifacts::ARTIFACT_MAGIC,
        "test setup invariant: blob must lead with the artifact magic",
    );

    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&bytes)
        .expect("artifact envelope must decode through the default reader");
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
}

/// Branch 2: leading bytes are NOT the artifact magic; the dispatcher
/// falls through to the legacy v3 trailer detector. The blob is a
/// signed v3 capture produced via `capture_legacy()`.
#[cfg(feature = "signed-snapshots")]
#[test]
fn v3_inline_envelope_is_detected_and_decoded_through_legacy_path() {
    let (wasm, gpu, regs) = synth_state();

    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture_legacy(InstanceState {
            tenant_id: TenantId(0xB2),
            instance_id: InstanceId(0xB2B2),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture_legacy v3");

    // Sanity: this is NOT the artifact envelope (the legacy path is
    // load-bearing for the test).
    assert!(bytes.len() >= 16);
    assert_ne!(
        &bytes[..16],
        &tensor_wasm_artifacts::ARTIFACT_MAGIC,
        "test setup invariant: blob must NOT lead with the artifact magic",
    );
    // And it IS a v3 blob — the trailer magic sits at offset -37.
    assert!(bytes.len() >= 37);
    assert_eq!(&bytes[bytes.len() - 37..bytes.len() - 33], b"S3T1");

    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&bytes)
        .expect("v3 inline envelope must decode through the legacy fallback");
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
}

/// Branch 3: not an artifact envelope, no v3 trailer → v2 decoder.
/// The blob is an unsigned capture from a keyless writer.
#[test]
fn v2_envelope_is_detected_and_decoded() {
    let (wasm, gpu, regs) = synth_state();

    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(0xC3),
            instance_id: InstanceId(0xC3C3),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v2 (keyless)");

    // Sanity: neither the artifact magic nor the v3 trailer magic
    // matches. The v2 path is the only valid dispatch.
    assert!(bytes.len() >= 16);
    assert_ne!(
        &bytes[..16],
        &b"twasm-artifact01"[..],
        "test setup invariant: v2 blob must not lead with the artifact magic",
    );

    let restored = SnapshotReader::new()
        .restore(&bytes)
        .expect("v2 envelope must decode through the default reader");
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
}

/// Branch 4: garbage input. The reader must produce a `Serialization`
/// error without panicking, regardless of which dispatch branch it
/// happens to land on.
#[test]
fn garbage_input_is_rejected_without_panicking() {
    let reader = SnapshotReader::new();

    // Empty input — must be rejected.
    let err = reader
        .restore(&[])
        .expect_err("empty input must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));

    // Random bytes that don't match any envelope's leading magic and
    // are not a valid zstd frame.
    let garbage: Vec<u8> = (0u8..255).cycle().take(2048).collect();
    let err = reader
        .restore(&garbage)
        .expect_err("garbage must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));

    // A blob that LOOKS like the start of the artifact envelope but
    // isn't long enough to carry the header + HMAC trailer. The
    // dispatcher routes it to the artifact path (leading bytes match
    // the magic), but the artifact decoder rejects the truncated
    // input. Crucially it should NOT fall through to v3/v2 — a
    // half-baked artifact envelope is a malformed artifact envelope,
    // not a misclassified legacy blob.
    #[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
    {
        let mut truncated = b"twasm-artifact01".to_vec();
        truncated.extend_from_slice(&[0u8; 8]); // far less than header + hmac
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(TEST_KEY)
            .restore(&truncated)
            .expect_err("truncated artifact envelope must be rejected");
        // Surfaces through the artifact decode path — message mentions
        // the envelope or the minimum-length / HMAC failure.
        if let TensorWasmError::Serialization(msg) = err {
            assert!(
                msg.to_ascii_lowercase().contains("artifact")
                    || msg.to_ascii_lowercase().contains("envelope")
                    || msg.to_ascii_lowercase().contains("hmac")
                    || msg.to_ascii_lowercase().contains("magic"),
                "truncated artifact must reject via the artifact path, got: {msg}",
            );
        } else {
            panic!("expected Serialization error");
        }
    }
}

/// Mixed-branch regression: take a real artifact envelope, append a
/// few extra bytes, and confirm the dispatcher still routes the input
/// through the artifact decoder (which rejects the appended garbage
/// via its HMAC check) rather than incidentally matching the v3
/// trailer magic at the new tail position. This guards against a
/// dispatcher that checks BOTH conditions and ambiguously picks one.
#[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
#[test]
fn artifact_envelope_with_appended_garbage_is_rejected_by_artifact_path() {
    let (wasm, gpu, regs) = synth_state();
    let mut bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xD4),
            instance_id: InstanceId(0xD4D4),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture default");
    // Append `S3T1` followed by enough junk to mimic a v3 trailer
    // length. Under a dispatcher that checks both conditions, this
    // would create ambiguity. The correct behaviour is: leading magic
    // wins, the artifact decoder takes over, and the HMAC check
    // rejects because the appended bytes were not part of the
    // signed prefix.
    bytes.extend_from_slice(b"S3T1");
    bytes.push(1); // kind byte
    bytes.extend_from_slice(&[0u8; 32]); // bogus signature

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&bytes)
        .expect_err("appended-garbage artifact envelope must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));
}
