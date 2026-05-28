// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T40 backward-compat guarantee: the default `SnapshotReader::restore`
//! continues to parse legacy v2 (unsigned) and v3 (T8 magic-prefix trailer)
//! blobs indefinitely.
//!
//! The v0.4 default-cutover flipped the *writer* to emit the unified
//! artifact-store envelope; the *reader* still accepts every legacy shape
//! so existing on-disk snapshots survive the upgrade. This file pins that
//! fall-through behaviour against three inputs:
//!
//!   1. An unsigned v2 capture (no HMAC key, no legacy opt-out — the
//!      writer's keyless fallback path that keeps emitting v2).
//!   2. A signed v3 capture produced via `capture_legacy()` — the
//!      explicit per-call opt-out for tooling that still expects the
//!      magic-prefix v3 trailer on the wire.
//!   3. A signed v3 capture produced via the writer-builder opt-out
//!      `with_legacy_envelope()` — same wire format as (2), different
//!      call shape, so a regression that only broke one path wouldn't
//!      silently slip past the other.
//!
//! All three are decoded through the default `SnapshotReader::restore`
//! (no extra configuration), proving the reader's
//! artifact-envelope-first dispatcher correctly falls through to the v3
//! / v2 paths when the leading bytes are not the artifact magic.

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};

/// Build a deterministic synthetic state used by every test below. Same
/// shape as the conformance suites elsewhere in the crate so a failure
/// mode (oversized blob, wrong bincode encoding) surfaces the same way
/// across files.
fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..2048).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..1024).map(|i| ((i.wrapping_mul(17)) % 253) as u8).collect();
    let regs: Vec<u8> = (0u32..64).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

/// (1) Unsigned v2 capture — the keyless path that v0.4 deliberately
/// preserves so every existing in-tree caller (tests, benches, mem
/// conformance) continues to work without changes.
#[test]
fn unsigned_v2_capture_round_trips_through_default_reader() {
    let (wasm, gpu, regs) = synth_state();

    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(0xCAFE),
            instance_id: InstanceId(0xBABE),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v2 (keyless)");

    // The default reader detects the (lack of) artifact magic, falls
    // through to the v3 trailer detector (which also doesn't fire on
    // an unsigned v2 input), and decodes the bincode payload through
    // the v2 path.
    let restored = SnapshotReader::new()
        .restore(&bytes)
        .expect("default reader must accept unsigned v2");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xCAFE));
    assert_eq!(restored.metadata.instance_id, InstanceId(0xBABE));
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
}

/// (2) Signed v3 capture via the explicit `capture_legacy()` per-call
/// opt-out. The resulting bytes are the inline v3 envelope (T8
/// magic-prefix trailer). The default reader auto-detects the trailer
/// magic at `len - 37..len - 33` and decodes via the v3 path.
#[cfg(feature = "signed-snapshots")]
#[test]
fn signed_v3_capture_via_capture_legacy_round_trips_through_default_reader() {
    let (wasm, gpu, regs) = synth_state();
    let key = [0x33u8; 32];

    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(key)
        .capture_legacy(InstanceState {
            tenant_id: TenantId(0xC0DE),
            instance_id: InstanceId(0xD00D),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture_legacy v3");

    // Sanity: the trailer magic `S3T1` sits at offset
    // `len - SIGNATURE_TRAILER_LEN` (= `len - 37`). If this assertion
    // fails, the writer didn't actually produce a legacy v3 blob and
    // the rest of the test would be testing the wrong invariant.
    assert!(bytes.len() >= 37);
    assert_eq!(&bytes[bytes.len() - 37..bytes.len() - 33], b"S3T1");

    // Reader must have the same key for the trailer verification to
    // pass. The default reader is otherwise unmodified — it dispatches
    // through the legacy v3 path because the leading bytes are zstd,
    // not the artifact magic.
    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(key)
        .restore(&bytes)
        .expect("default reader must accept v3 via legacy fallback");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xC0DE));
    assert_eq!(restored.metadata.instance_id, InstanceId(0xD00D));
}

/// (3) Signed v3 capture via the writer-builder opt-out
/// `with_legacy_envelope()`. Identical wire format to (2) but different
/// call shape; pinned separately so a future refactor that decouples
/// the two paths (e.g. only routing one through the artifact envelope)
/// trips this test independently of (2).
#[cfg(feature = "signed-snapshots")]
#[test]
fn signed_v3_capture_via_with_legacy_envelope_round_trips_through_default_reader() {
    let (wasm, gpu, regs) = synth_state();
    let key = [0x44u8; 32];

    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(key)
        .with_legacy_envelope()
        .capture(InstanceState {
            tenant_id: TenantId(0x1234),
            instance_id: InstanceId(0x5678),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("with_legacy_envelope v3");

    // Same trailer-magic sanity check as (2). If both (2) and (3)
    // simultaneously regress to NOT emit the trailer magic, the writer
    // builder fell back to the v0.4 envelope despite the opt-out — a
    // bug that needs to surface loudly.
    assert!(bytes.len() >= 37);
    assert_eq!(&bytes[bytes.len() - 37..bytes.len() - 33], b"S3T1");

    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(key)
        .restore(&bytes)
        .expect("default reader must accept v3 via legacy fallback");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
}
