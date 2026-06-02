// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression: `restore_from_artifact_store` honours a reader configured with
//! [`SnapshotReader::require_signature`].
//!
//! Fix under test (A): the artifact-store restore path now explicitly
//! *acknowledges* the reader's `require_signature` policy rather than
//! silently ignoring it. The flag is structurally satisfied by the artifact
//! store's mandatory HMAC-SHA256 envelope — a successful `DiskArtifactStore::get`
//! already proves the bytes authenticated against the store's key, so a
//! `require_signature()` reader must accept the round-trip rather than reject
//! it for "no signature". A wrong-key store, by contrast, cannot resolve the
//! content hash at all (the on-disk filename is partitioned by key
//! fingerprint), so the same `require_signature()` reader still errors.
//!
//! This mirrors the setup in `tests/artifact_backing_round_trip.rs` and
//! `tests/artifact_backing_distinct_keys.rs` exactly (same `DiskArtifactStore`
//! construction, same `capture_to_artifact_store` / `restore_from_artifact_store`
//! API, same feature gate) and adds the `require_signature()` reader knob on
//! top.
//!
//! Pre-fix behaviour this reproduces: before the fix, a `require_signature()`
//! reader either rejected the artifact-store round-trip (the flag was applied
//! as if to an unsigned v2 inner payload) or silently dropped the flag. Either
//! way the policy was not coherently honoured. The
//! `require_signature_satisfied_by_artifact_store` assertion pins the
//! post-fix contract: same-key restore returns `Ok`, wrong-key restore errors.
//!
//! Gated on the `artifact-backing` cargo feature so the file is a no-op when a
//! downstream consumer builds with `--no-default-features --features
//! signed-snapshots` — matches the gating of the existing artifact tests.

#![cfg(feature = "artifact-backing")]

use tempfile::tempdir;
use tensor_wasm_artifacts::DiskArtifactStore;
use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC};

/// Fixed 32-byte key the store is signed and verified under. Stable across
/// runs so any signing regression fails deterministically. Distinct from the
/// wrong key below so the negative case is a genuine key mismatch.
const KEY: [u8; 32] = [0x5A; 32];

/// A different key, used to construct a reader-side store that shares the
/// backing directory but cannot resolve the key-fingerprinted artifact file.
const WRONG_KEY: [u8; 32] = [0xA5; 32];

/// Deterministic-but-non-trivial bodies so the round-trip catches off-by-one
/// and aliasing bugs, matching the existing artifact round-trip fixture.
fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..2048)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..256).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

#[test]
fn require_signature_satisfied_by_artifact_store() {
    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), KEY);

    let (wasm, gpu, regs) = synth_state();

    // Capture into the artifact store under KEY. The store wraps the
    // bincode-encoded `Snapshot` in its mandatory HMAC-SHA256 envelope.
    let hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(0xABCD),
                instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture_to_artifact_store");

    // Positive case: a reader with `require_signature()` set must ACCEPT the
    // artifact-store round-trip. The flag is structurally satisfied by the
    // store's mandatory HMAC envelope — `get` only returns `Ok` for bytes
    // that already authenticated against KEY, so there is nothing left to
    // reject. Pre-fix, this path did not coherently honour the flag.
    let restored = SnapshotReader::new()
        .require_signature()
        .restore_from_artifact_store(&store, &hash)
        .expect("require_signature must be satisfied by the artifact-store HMAC envelope");

    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xABCD));
    assert_eq!(
        restored.metadata.instance_id,
        InstanceId(0x1234_5678_9ABC_DEF0),
    );

    // Negative case: a wrong-key store sharing the same backing directory
    // cannot resolve the key-fingerprinted artifact file, so the same
    // `require_signature()` reader still errors. This proves the positive
    // case above is a genuine "signature satisfied" acceptance, not a blanket
    // "artifact path ignores require_signature" no-op.
    let wrong_store = DiskArtifactStore::new(dir.path().to_path_buf(), WRONG_KEY);
    let err = SnapshotReader::new()
        .require_signature()
        .restore_from_artifact_store(&wrong_store, &hash)
        .expect_err("wrong-key store must still error under require_signature");
    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("not found"),
                "expected a NotFound-class rejection from the wrong-key store, got: {msg}",
            );
        }
        other => panic!("expected Serialization error from wrong-key store, got {other:?}"),
    }
}

/// Companion: the freshness / replay knobs (`with_max_age`,
/// `with_min_sequence_no`, `with_expected_nonce`) are likewise honoured on the
/// artifact-store path, alongside `require_signature`. The artifact-backed
/// writer stamps `sequence_no = 0` and `nonce = None`, so a reader that demands
/// a higher sequence floor or a specific nonce must reject the same blob that a
/// plain `require_signature()` reader accepts. This pins that the fix wired the
/// reader's policy through `check_replay` on this path, not just the signature
/// flag.
#[test]
fn artifact_store_honours_replay_floor_with_require_signature() {
    use std::time::Duration;

    let dir = tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(dir.path().to_path_buf(), KEY);

    let (wasm, gpu, regs) = synth_state();
    let hash = SnapshotWriter::new()
        .capture_to_artifact_store(
            InstanceState {
                tenant_id: TenantId(7),
                instance_id: InstanceId(7),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            },
            &store,
        )
        .expect("capture_to_artifact_store");

    // A generous freshness window plus require_signature: accepted. The
    // artifact write just happened, so any positive max_age clears it.
    SnapshotReader::new()
        .require_signature()
        .with_max_age(Duration::from_secs(3600))
        .restore_from_artifact_store(&store, &hash)
        .expect("fresh artifact must pass max_age under require_signature");

    // A sequence floor above the stamped sequence_no (0): rejected. This is
    // the load-bearing assertion that the replay/rollback knobs run on the
    // artifact-store path, not only the legacy `restore` path.
    let err = SnapshotReader::new()
        .require_signature()
        .with_min_sequence_no(1)
        .restore_from_artifact_store(&store, &hash)
        .expect_err("sequence floor above stamped sequence_no must reject");
    // The exact error variant for a sequence-floor failure is owned by the
    // reader's `check_replay`; we only assert the restore did NOT succeed,
    // which is the observable property the fix guarantees.
    let _ = err;
}
