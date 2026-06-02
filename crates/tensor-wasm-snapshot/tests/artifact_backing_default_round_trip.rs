// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T40 default-cutover: the v0.4 `SnapshotWriter::capture` path emits the
//! unified artifact-store envelope, not the legacy v3 inline trailer.
//!
//! Without an explicit `--features` flag (the `artifact-backing` feature is
//! now a default cargo feature on this crate), constructing a writer with an
//! HMAC key configured and calling `capture` produces bytes whose first 16
//! bytes are `b"twasm-artifact01"` — the unified envelope's magic. The blob
//! round-trips through the default `SnapshotReader::restore`, which detects
//! the envelope by its leading magic and decodes it without ever touching
//! the legacy v3 trailer detector.
//!
//! Gated on `artifact-backing` so the file is a no-op when a downstream
//! consumer has built with `--no-default-features --features signed-snapshots`
//! (the legacy opt-out shape).

#![cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]

use tensor_wasm_artifacts::ARTIFACT_MAGIC;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};
use tensor_wasm_snapshot::SNAPSHOT_VERSION_V3;

/// 32-byte HMAC key for the default-path capture. Stable across runs so any
/// regression in the artifact-envelope signing surfaces deterministically.
const TEST_KEY: [u8; 32] = [0x40; 32];

#[test]
fn default_capture_emits_artifact_envelope_magic() {
    // Build a small but non-trivial state — the goal is to verify the
    // envelope shape, not throughput. Any input that exercises the
    // bincode+zstd pipeline is sufficient.
    let wasm: Vec<u8> = (0u32..512).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..256)
        .map(|i| ((i.wrapping_mul(13)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..32).map(|i| ((i ^ 0x40) & 0xFF) as u8).collect();

    // Default path: feature is enabled by the cargo defaults, writer has
    // an HMAC key, no `with_legacy_envelope` opt-out. T40 dispatches
    // through `capture_via_artifact_envelope`.
    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0x40),
            instance_id: InstanceId(0x4040),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture via T40 default");

    // The load-bearing assertion: the first 16 bytes are the artifact
    // store's magic, NOT the zstd frame header (which would be the
    // legacy v2/v3 shape's leading bytes).
    assert!(
        bytes.len() >= ARTIFACT_MAGIC.len(),
        "envelope must be at least the magic length",
    );
    assert_eq!(
        &bytes[..ARTIFACT_MAGIC.len()],
        &ARTIFACT_MAGIC,
        "T40 default capture must emit the unified artifact-store magic",
    );

    // Defence in depth: a legacy v3 blob ends with the `S3T1` trailer
    // magic. The artifact envelope ends with a 32-byte HMAC tag whose
    // bytes are pseudorandom — vanishingly unlikely to spell `S3T1`.
    let v3_trailer_magic = b"S3T1";
    // Trailer would sit 37 bytes from the end if this were a v3 blob.
    if bytes.len() >= 37 {
        let candidate = &bytes[bytes.len() - 37..bytes.len() - 33];
        assert_ne!(
            candidate, v3_trailer_magic,
            "T40 envelope must NOT carry the legacy v3 trailer magic at offset -37..-33",
        );
    }
}

#[test]
fn default_capture_round_trips_through_default_reader() {
    let wasm: Vec<u8> = (0u32..2048).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..1024)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..128).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();

    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture default");

    // Confirm the envelope magic.
    assert_eq!(&bytes[..ARTIFACT_MAGIC.len()], &ARTIFACT_MAGIC);

    // The default reader auto-detects the artifact envelope and
    // decodes it. The inner Snapshot must round-trip byte-for-byte.
    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&bytes)
        .expect("default reader must decode T40 envelope");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    // Artifact-envelope writes stamp the inner version as v2 — the
    // outer envelope owns authentication, so the inner v3 trailer
    // would be redundant. This is also documented in
    // `SnapshotWriter::build_snapshot_ref`. Pin the discriminant so a
    // future regression that inadvertently bumps the inner version
    // surfaces here.
    assert_ne!(
        restored.version, SNAPSHOT_VERSION_V3,
        "T40 envelope must NOT carry inner version v3 (the outer envelope already signs)",
    );
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xABCD));
    assert_eq!(
        restored.metadata.instance_id,
        InstanceId(0x1234_5678_9ABC_DEF0),
    );
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
    assert!(restored.metadata.created_unix_ms > 0);
}

#[test]
fn default_capture_with_legacy_opt_out_still_emits_v3() {
    // Companion: `with_legacy_envelope()` opts out of the v0.4 default
    // even when the feature is on. The resulting blob must NOT start
    // with the artifact magic — it must be a legacy zstd frame.
    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .with_legacy_envelope()
        .capture(InstanceState {
            tenant_id: TenantId(7),
            instance_id: InstanceId(77),
            wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture legacy via opt-out");

    assert!(
        bytes.len() >= 4,
        "legacy blob must be at least the zstd magic length",
    );
    // zstd frames start with the magic `0xFD2FB528` (LE). The
    // artifact envelope starts with `b"twasm-artifact01"`. The two are
    // disjoint, so checking that the leading bytes are NOT the
    // artifact magic is the load-bearing assertion.
    assert_ne!(
        &bytes[..ARTIFACT_MAGIC.len().min(bytes.len())],
        &ARTIFACT_MAGIC[..ARTIFACT_MAGIC.len().min(bytes.len())],
        "with_legacy_envelope() must NOT emit the artifact magic",
    );
}

#[test]
fn default_capture_without_key_falls_back_to_v2() {
    // T40 graceful-fallback: a writer that has no HMAC key configured
    // continues to emit the legacy unsigned v2 envelope. The v0.4
    // artifact envelope needs an HMAC key by construction, so this is
    // the only honest answer for keyless callers — and it preserves
    // the v0.3.x contract for every in-tree test, bench, and
    // conformance suite that constructs `SnapshotWriter::new()` with
    // no further configuration.
    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &[1, 2, 3, 4],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture keyless");

    assert!(bytes.len() >= ARTIFACT_MAGIC.len());
    assert_ne!(
        &bytes[..ARTIFACT_MAGIC.len()],
        &ARTIFACT_MAGIC,
        "keyless capture must remain on legacy v2 (no envelope)",
    );
}
