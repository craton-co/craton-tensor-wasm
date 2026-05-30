// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Streaming / lower-peak-memory restore for the v4 artifact path.
//!
//! `SnapshotReader::restore_streaming` and `restore_to_writer` reduce the
//! reader-side peak by handing each memory blob to a sink and dropping the
//! reader's own copy immediately, rather than returning the whole `Snapshot`
//! (three owned `Vec<u8>`s) by value. This file pins the two invariants that
//! matter:
//!
//! 1. **Identical result.** A streaming restore reconstructs exactly the same
//!    bytes as the buffered `restore`, on both the v4 artifact-envelope path
//!    (writer has an HMAC key) and the legacy keyless v2 path.
//! 2. **Verify-before-expose.** A tampered blob is rejected and the sink is
//!    never invoked — no unverified byte reaches the caller.
//!
//! The `mmap` feature, when enabled, adds `restore_from_path_mmap`; its
//! round-trip and tamper-rejection are covered by the `#[cfg(feature =
//! "mmap")]` tests at the bottom.

use std::cell::RefCell;

use tensor_wasm_core::error::{Result, TensorWasmError};
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::{SnapshotReader, SnapshotSink};
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

const TEST_KEY: [u8; 32] = [0x5A; 32];

/// Collecting sink that records each blob it is handed, in order. Used to
/// assert the streamed bytes equal what `restore` would have returned.
#[derive(Default)]
struct CollectSink {
    wasm: Vec<u8>,
    gpu: Vec<u8>,
    regs: Vec<u8>,
    calls: Vec<&'static str>,
}

impl SnapshotSink for CollectSink {
    fn wasm_memory(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.calls.push("wasm");
        self.wasm = bytes;
        Ok(())
    }
    fn gpu_memory(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.calls.push("gpu");
        self.gpu = bytes;
        Ok(())
    }
    fn registers(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.calls.push("regs");
        self.regs = bytes;
        Ok(())
    }
}

/// Sink that errors before storing anything — used to prove the sink is
/// *never* invoked when authentication fails (the error variant should be the
/// reader's `Serialization`, not this sentinel).
struct PoisonSink {
    invoked: RefCell<bool>,
}

impl SnapshotSink for PoisonSink {
    fn wasm_memory(&mut self, _bytes: Vec<u8>) -> Result<()> {
        *self.invoked.borrow_mut() = true;
        Err(TensorWasmError::Serialization("POISON SINK INVOKED".into()))
    }
    fn gpu_memory(&mut self, _bytes: Vec<u8>) -> Result<()> {
        *self.invoked.borrow_mut() = true;
        Err(TensorWasmError::Serialization("POISON SINK INVOKED".into()))
    }
    fn registers(&mut self, _bytes: Vec<u8>) -> Result<()> {
        *self.invoked.borrow_mut() = true;
        Err(TensorWasmError::Serialization("POISON SINK INVOKED".into()))
    }
}

fn sample_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..8192)
        .map(|i| ((i.wrapping_mul(31)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..256).map(|i| ((i ^ 0x5A) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

/// v4 artifact-envelope path: streaming restore yields byte-identical blobs
/// and metadata to the buffered `restore`.
#[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
#[test]
fn streaming_matches_buffered_v4() {
    let (wasm, gpu, regs) = sample_state();
    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v4");

    let buffered = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore(&bytes)
        .expect("buffered restore");

    let mut sink = CollectSink::default();
    let meta = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore_streaming(&bytes, &mut sink)
        .expect("streaming restore");

    assert_eq!(
        sink.calls,
        ["wasm", "gpu", "regs"],
        "canonical order + once each"
    );
    assert_eq!(sink.wasm, buffered.wasm_memory);
    assert_eq!(sink.gpu, buffered.gpu_memory);
    assert_eq!(sink.regs, buffered.registers);
    assert_eq!(sink.wasm, wasm);
    assert_eq!(sink.gpu, gpu);
    assert_eq!(sink.regs, regs);
    assert_eq!(meta.tenant_id, buffered.metadata.tenant_id);
    assert_eq!(meta.instance_id, buffered.metadata.instance_id);
    assert_eq!(
        meta.total_uncompressed_bytes,
        buffered.metadata.total_uncompressed_bytes,
    );
}

/// Keyless legacy v2 path: streaming restore matches buffered restore too
/// (the streaming entry point is format-agnostic — it delegates to `restore`).
#[test]
fn streaming_matches_buffered_v2() {
    let (wasm, gpu, regs) = sample_state();
    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(7),
            instance_id: InstanceId(77),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v2");

    let buffered = SnapshotReader::new().restore(&bytes).expect("buffered");

    let mut sink = CollectSink::default();
    SnapshotReader::new()
        .restore_streaming(&bytes, &mut sink)
        .expect("streaming");

    assert_eq!(sink.wasm, buffered.wasm_memory);
    assert_eq!(sink.gpu, buffered.gpu_memory);
    assert_eq!(sink.regs, buffered.registers);
}

/// `restore_to_writer` concatenates the three blobs in canonical order; the
/// result equals `wasm || gpu || regs` from a buffered restore.
#[test]
fn restore_to_writer_concatenates_blobs() {
    let (wasm, gpu, regs) = sample_state();
    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(2),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture");

    let mut out: Vec<u8> = Vec::new();
    let meta = SnapshotReader::new()
        .restore_to_writer(&bytes, &mut out)
        .expect("restore_to_writer");

    let mut expected = Vec::new();
    expected.extend_from_slice(&wasm);
    expected.extend_from_slice(&gpu);
    expected.extend_from_slice(&regs);
    assert_eq!(out, expected);
    assert_eq!(
        meta.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
}

/// Verify-before-expose: a v4 blob with a flipped byte must be rejected and
/// the sink must never be invoked.
#[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
#[test]
fn streaming_tamper_rejected_before_any_output_v4() {
    let (wasm, gpu, regs) = sample_state();
    let mut bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(3),
            instance_id: InstanceId(4),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v4");

    // Flip a byte inside the compressed body (past the 52-byte artifact
    // header, before the trailing 32-byte HMAC tag). The HMAC must reject it.
    let flip_at = bytes.len() / 2;
    bytes[flip_at] ^= 0xFF;

    let mut sink = PoisonSink {
        invoked: RefCell::new(false),
    };
    let err = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore_streaming(&bytes, &mut sink)
        .expect_err("tampered v4 blob must be rejected");

    assert!(
        matches!(err, TensorWasmError::Serialization(_)),
        "expected Serialization error, got {err:?}",
    );
    assert!(
        !*sink.invoked.borrow(),
        "sink must NOT be invoked on a tampered blob (verify-before-expose)",
    );
    // The error must come from the reader's authentication, not the poison
    // sink — a defence against a regression that streamed before verifying.
    if let TensorWasmError::Serialization(m) = &err {
        assert!(
            !m.contains("POISON SINK INVOKED"),
            "tamper must be caught before the sink runs, got: {m}",
        );
    }
}

/// Verify-before-expose on the legacy v2 path: a truncated blob is rejected
/// and the sink is never invoked.
#[test]
fn streaming_tamper_rejected_before_any_output_v2() {
    let (wasm, gpu, regs) = sample_state();
    let bytes = SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(5),
            instance_id: InstanceId(6),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v2");
    let truncated = &bytes[..bytes.len() / 2];

    let mut sink = PoisonSink {
        invoked: RefCell::new(false),
    };
    let err = SnapshotReader::new()
        .restore_streaming(truncated, &mut sink)
        .expect_err("truncated blob must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));
    assert!(
        !*sink.invoked.borrow(),
        "sink must NOT be invoked on a malformed blob",
    );
}

// ---------------------------------------------------------------------------
// mmap input-side path (non-default `mmap` feature).
// ---------------------------------------------------------------------------

/// `restore_from_path_mmap` maps the snapshot file and round-trips identically
/// to the in-memory `restore`.
#[cfg(all(
    feature = "mmap",
    feature = "artifact-backing",
    feature = "signed-snapshots"
))]
#[test]
fn mmap_round_trips_v4() {
    use std::io::Write;

    let (wasm, gpu, regs) = sample_state();
    let bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(0xBEEF),
            instance_id: InstanceId(0xF00D),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v4");

    let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
    tmp.write_all(&bytes).expect("write blob");
    tmp.flush().expect("flush");

    let restored = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore_from_path_mmap(tmp.path())
        .expect("mmap restore");

    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xBEEF));

    // And the streaming-from-mmap combo yields the same blobs.
    let mut sink = CollectSink::default();
    SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore_streaming_from_path_mmap(tmp.path(), &mut sink)
        .expect("mmap streaming restore");
    assert_eq!(sink.wasm, wasm);
    assert_eq!(sink.gpu, gpu);
    assert_eq!(sink.regs, regs);
}

/// mmap path preserves verify-before-expose: a tampered file is rejected.
#[cfg(all(
    feature = "mmap",
    feature = "artifact-backing",
    feature = "signed-snapshots"
))]
#[test]
fn mmap_tamper_rejected() {
    use std::io::Write;

    let (wasm, gpu, regs) = sample_state();
    let mut bytes = SnapshotWriter::new()
        .with_hmac_sha256_key(TEST_KEY)
        .capture(InstanceState {
            tenant_id: TenantId(9),
            instance_id: InstanceId(10),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture v4");
    let flip_at = bytes.len() / 2;
    bytes[flip_at] ^= 0xFF;

    let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
    tmp.write_all(&bytes).expect("write");
    tmp.flush().expect("flush");

    let err = SnapshotReader::new()
        .with_hmac_sha256_key(TEST_KEY)
        .restore_from_path_mmap(tmp.path())
        .expect_err("tampered file must be rejected");
    assert!(matches!(err, TensorWasmError::Serialization(_)));
}
