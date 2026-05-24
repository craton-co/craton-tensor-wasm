// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `SnapshotWriter`: capture instance state into a self-describing byte blob.
//!
//! A snapshot is a bincode-encoded [`Snapshot`] header (containing wasm linear
//! memory, GPU memory, register file, and metadata) wrapped in a zstd-compressed
//! container. The on-disk magic value `0xBA11_5407` and a `version` field let
//! the reader reject stale or foreign blobs without panicking. The writer is
//! deliberately synchronous — tokio is brought in by upstream callers that
//! stream the resulting `Vec<u8>` to disk or to object storage.

use std::time::{SystemTime, UNIX_EPOCH};

use tensor_wasm_core::error::{TensorWasmError, Result};
use tensor_wasm_core::types::{InstanceId, TenantId};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Magic bytes that identify a TensorWasm snapshot blob.
///
/// Stored at the head of the bincode payload. Read by [`crate::reader::SnapshotReader`]
/// to refuse blobs that were not produced by this crate. The literal spells
/// `BA11 5407` ("BALI SHOT") in leet hex — a frozen legacy magic from the
/// Project Bali era. Do not change without a snapshot-format version bump,
/// or every existing snapshot becomes unreadable.
pub const SNAPSHOT_MAGIC: u32 = 0xBA11_5407;

/// On-disk schema version for [`Snapshot`].
///
/// Bumped whenever the serialised layout of `Snapshot` changes in a way that is
/// not bincode-compatible with previous releases. The reader will reject any
/// blob whose `version` does not equal this constant.
///
/// Version `2` adds the [`Snapshot::crc32`] payload checksum and enforces the
/// size limits defined in [`limits`]. Snapshots produced by version `1` writers
/// are not accepted by version `2` readers.
pub const SNAPSHOT_VERSION: u32 = 2;

/// Default zstd compression level used by [`SnapshotWriter::capture`].
///
/// Level 3 is the upstream default — a good speed/ratio trade-off for the
/// mostly-incompressible memory payloads typical of GPU workloads.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Hard upper bounds on the payload sizes accepted by capture and restore.
///
/// These constants gate every memory blob carried by a snapshot. They are
/// deliberately generous — large enough to fit any plausible workload TensorWasm
/// targets — and exist solely to bound the allocator's exposure to malicious
/// or accidentally-corrupted inputs. Both [`SnapshotWriter::capture`] and
/// [`crate::reader::SnapshotReader::restore`] return an error if any blob
/// exceeds its limit.
pub mod limits {
    /// Maximum Wasm linear memory accepted in a single snapshot (1 GiB).
    ///
    /// Sized at the Wasm32 address-space ceiling so legitimate instances are
    /// never rejected, but a corrupted `wasm_memory.len()` field cannot drive
    /// the restore-side allocator into terabyte territory.
    pub const MAX_WASM_MEMORY_BYTES: usize = 1024 * 1024 * 1024;

    /// Maximum GPU device-side memory accepted in a single snapshot (4 GiB).
    ///
    /// GPU buffers can dwarf Wasm memory on inference workloads, so the cap is
    /// looser. This is the value the restore path will trust when sizing the
    /// uncompressed staging buffer, so it directly bounds restore-time memory
    /// pressure for a hostile snapshot.
    pub const MAX_GPU_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

    /// Maximum register-file payload accepted in a single snapshot (1 MiB).
    ///
    /// Generous for any plausible Wasm/PTX register state — real captures are
    /// in the kilobytes — but kept finite so a tampered length cannot trigger
    /// a runaway allocation.
    pub const MAX_REGISTERS_BYTES: usize = 1024 * 1024;

    /// Aggregate cap covering all three memory blobs plus envelope overhead.
    ///
    /// Exposed so callers can pre-size buffers without re-summing the per-blob
    /// limits. The 64 KiB slack accounts for the bincode/zstd envelope and the
    /// metadata struct.
    pub const MAX_TOTAL_PAYLOAD_BYTES: usize =
        MAX_WASM_MEMORY_BYTES + MAX_GPU_MEMORY_BYTES + MAX_REGISTERS_BYTES + 65536;

    /// Hard ceiling on the compressed byte slice accepted by the reader (~4 GiB).
    ///
    /// Snapshots arriving over the network may have been crafted to look small
    /// but decompress into terabytes. The reader rejects any input larger than
    /// this before calling zstd to bound the attacker's memory budget.
    ///
    /// On a 32-bit target this expression would overflow `usize`; the static
    /// assertion at the crate root ensures we only ever compile on 64-bit.
    pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024 * 64;

    /// Default ceiling on bytes produced by zstd decompression before bincode
    /// runs (256 MiB). Bounds memory pressure from a zip-bomb that streams
    /// indefinitely past the bincode-visible struct sizes.
    ///
    /// Callers can override via
    /// [`SnapshotReader::with_max_decompressed`](crate::reader::SnapshotReader::with_max_decompressed)
    /// when restoring trusted snapshots that legitimately exceed this default.
    pub const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;
}

/// Metadata captured alongside the memory blobs in every snapshot.
///
/// All fields are required and serialised in declaration order; new fields
/// must be appended (never reordered) and accompanied by a [`SNAPSHOT_VERSION`]
/// bump.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Tenant that owned the instance at snapshot time.
    pub tenant_id: TenantId,
    /// Instance whose state is captured.
    pub instance_id: InstanceId,
    /// Wall-clock time the snapshot was taken, in milliseconds since the Unix epoch.
    pub created_unix_ms: u64,
    /// Sum of `wasm_memory.len() + gpu_memory.len() + registers.len()` before zstd.
    ///
    /// Recorded so callers can size pre-allocated buffers and report compression
    /// ratios without re-decompressing.
    pub total_uncompressed_bytes: u64,
}

/// In-memory representation of a captured instance.
///
/// Produced by [`SnapshotWriter::capture`] and reconstituted by
/// [`crate::reader::SnapshotReader::restore`]. The bytes inside the three memory
/// vectors are opaque to this crate — upper layers (`tensor-wasm-mem`, `tensor-wasm-exec`)
/// own the schema of what they put in.
///
/// The three byte-blob fields are serialised through `serde_bytes` so bincode
/// emits a single length-prefixed byte string per blob (instead of a sequence
/// of one-byte elements). The on-disk encoding matches what
/// [`SnapshotWriter::capture`] writes via the borrowing `SnapshotRef` helper —
/// no host-side `.to_vec()` copy is needed on the write path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Magic identifying this blob as a TensorWasm snapshot. Must equal [`SNAPSHOT_MAGIC`].
    pub magic: u32,
    /// On-disk schema version. Must equal [`SNAPSHOT_VERSION`].
    pub version: u32,
    /// Raw bytes of the Wasm linear memory at capture time.
    #[serde(with = "serde_bytes")]
    pub wasm_memory: Vec<u8>,
    /// Raw bytes of the GPU device-side memory at capture time.
    #[serde(with = "serde_bytes")]
    pub gpu_memory: Vec<u8>,
    /// Register-file snapshot (PTX-level state captured by the JIT).
    #[serde(with = "serde_bytes")]
    pub registers: Vec<u8>,
    /// Free-form metadata describing the snapshot's provenance.
    pub metadata: SnapshotMetadata,
    /// CRC32 checksum over `wasm_memory`, `gpu_memory`, and `registers` (in that
    /// order), computed with the IEEE polynomial via [`crc32fast`]. The reader
    /// recomputes this value and rejects the snapshot if it does not match.
    pub crc32: u32,
}

/// Borrowing mirror of [`Snapshot`] used only on the write path so capture does
/// not have to clone the caller's byte slices.
///
/// Field order, types (post-`serde_bytes` adaptation), and bincode encoding are
/// identical to [`Snapshot`], so a blob produced from `SnapshotRef` round-trips
/// through [`crate::reader::SnapshotReader::restore`] into the owned form.
#[derive(Debug, Serialize)]
struct SnapshotRef<'a> {
    magic: u32,
    version: u32,
    #[serde(with = "serde_bytes")]
    wasm_memory: &'a [u8],
    #[serde(with = "serde_bytes")]
    gpu_memory: &'a [u8],
    #[serde(with = "serde_bytes")]
    registers: &'a [u8],
    metadata: SnapshotMetadata,
    crc32: u32,
}

/// Compute the payload CRC32 over the three memory blobs in their canonical
/// order (`wasm_memory`, then `gpu_memory`, then `registers`).
///
/// Exposed so callers that build a [`Snapshot`] by hand (tests, fuzzers) can
/// produce a value that survives [`crate::reader::SnapshotReader::restore`].
#[must_use]
pub fn payload_crc32(wasm_memory: &[u8], gpu_memory: &[u8], registers: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(wasm_memory);
    hasher.update(gpu_memory);
    hasher.update(registers);
    hasher.finalize()
}

/// Input collected by the caller and handed to [`SnapshotWriter::capture`].
///
/// Kept as a borrowed view so callers do not have to clone large memory blobs
/// just to call into this crate — the writer serialises the byte slices in
/// place via [`SnapshotRef`] and never materialises a host-side copy.
#[derive(Clone, Copy, Debug)]
pub struct InstanceState<'a> {
    /// Tenant that owns the instance.
    pub tenant_id: TenantId,
    /// Instance identifier.
    pub instance_id: InstanceId,
    /// Wasm linear memory bytes.
    pub wasm_memory: &'a [u8],
    /// GPU device memory bytes.
    pub gpu_memory: &'a [u8],
    /// Register-file bytes.
    pub registers: &'a [u8],
}

/// Captures live [`InstanceState`] into a portable, zstd-compressed byte vector.
///
/// Stateless: no internal buffers are reused between calls, so the writer is
/// `Send + Sync` and trivially shareable across worker threads.
#[derive(Clone, Copy, Debug, Default)]
pub struct SnapshotWriter {
    /// zstd compression level to use. Defaults to [`DEFAULT_ZSTD_LEVEL`].
    pub zstd_level: i32,
}

/// Validate `len` against `max` and return a descriptive error if it overflows.
///
/// Used by both the writer (to reject oversized inputs at capture time) and
/// the reader (to refuse oversized blobs after deserialisation). Shared so the
/// two sides emit identical error messages for the same condition.
pub(crate) fn check_blob_size(kind: &'static str, len: usize, max: usize) -> Result<()> {
    if len > max {
        Err(TensorWasmError::Serialization(
            format!("snapshot {kind} too large: {len} bytes exceeds cap of {max} bytes").into(),
        ))
    } else {
        Ok(())
    }
}

impl SnapshotWriter {
    /// Construct a writer using [`DEFAULT_ZSTD_LEVEL`].
    pub const fn new() -> Self {
        Self {
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }

    /// Construct a writer with an explicit zstd compression level.
    ///
    /// Valid range is `1..=22`; out-of-range values are clamped by the zstd
    /// library at compression time.
    pub const fn with_level(zstd_level: i32) -> Self {
        Self { zstd_level }
    }

    /// Encode and compress `state` into a snapshot blob.
    ///
    /// The returned bytes are self-describing: the magic and version are
    /// embedded in the bincode payload, so a caller only needs to persist the
    /// `Vec<u8>` as-is. Returns [`TensorWasmError::Serialization`] if bincode encoding
    /// fails, the zstd encoder reports a system error, or any input blob exceeds
    /// the caps in [`limits`]. Capture is the first line of defence — oversized
    /// inputs are rejected here so the writer never produces bytes that the
    /// reader would have to reject.
    #[instrument(skip(self, state), fields(
        tenant = %state.tenant_id,
        instance = %state.instance_id,
    ))]
    pub fn capture(&self, state: InstanceState<'_>) -> Result<Vec<u8>> {
        check_blob_size(
            "wasm_memory",
            state.wasm_memory.len(),
            limits::MAX_WASM_MEMORY_BYTES,
        )?;
        check_blob_size(
            "gpu_memory",
            state.gpu_memory.len(),
            limits::MAX_GPU_MEMORY_BYTES,
        )?;
        check_blob_size(
            "registers",
            state.registers.len(),
            limits::MAX_REGISTERS_BYTES,
        )?;

        let total_uncompressed_bytes =
            (state.wasm_memory.len() + state.gpu_memory.len() + state.registers.len()) as u64;
        let created_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);

        let crc32 = payload_crc32(state.wasm_memory, state.gpu_memory, state.registers);

        // Build the on-wire view by borrowing the caller's slices — no
        // host-side `.to_vec()` for the three memory blobs. bincode serialises
        // the borrowed `&[u8]` (via `serde_bytes`) identically to how it
        // deserialises into `Vec<u8>` (also via `serde_bytes`), so the wire
        // format is unchanged.
        let snapshot_ref = SnapshotRef {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            wasm_memory: state.wasm_memory,
            gpu_memory: state.gpu_memory,
            registers: state.registers,
            metadata: SnapshotMetadata {
                tenant_id: state.tenant_id,
                instance_id: state.instance_id,
                created_unix_ms,
                total_uncompressed_bytes,
            },
            crc32,
        };

        // `bincode::config::legacy()` produces the same wire format as bincode 1.x's
        // `DefaultOptions::new().with_fixint_encoding().with_little_endian()` —
        // little-endian, fixed-width integers, no length limit on the encoder.
        // Documented byte-compatible with the v1 default config.
        let cfg = bincode::config::legacy();
        let encoded = bincode::serde::encode_to_vec(&snapshot_ref, cfg)
            .map_err(|e| TensorWasmError::Serialization(format!("bincode encode: {e}").into()))?;
        let compressed = zstd::encode_all(encoded.as_slice(), self.zstd_level)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd encode: {e}").into()))?;

        debug!(
            uncompressed = encoded.len(),
            compressed = compressed.len(),
            "snapshot captured",
        );
        Ok(compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::SnapshotReader;

    fn sample_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let wasm = (0u8..=255).cycle().take(4096).collect::<Vec<u8>>();
        let gpu = (0u8..=255).cycle().skip(7).take(2048).collect::<Vec<u8>>();
        let regs = vec![0xAB; 256];
        (wasm, gpu, regs)
    }

    #[test]
    fn round_trip_writer_to_reader() {
        let (wasm, gpu, regs) = sample_state();
        let writer = SnapshotWriter::new();
        let bytes = writer
            .capture(InstanceState {
                tenant_id: TenantId(7),
                instance_id: InstanceId(42),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");
        let restored = SnapshotReader::new().restore(&bytes).expect("restore");
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
        assert_eq!(restored.metadata.tenant_id, TenantId(7));
        assert_eq!(restored.metadata.instance_id, InstanceId(42));
        assert_eq!(
            restored.metadata.total_uncompressed_bytes,
            (wasm.len() + gpu.len() + regs.len()) as u64,
        );
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let mut s = Snapshot {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION + 1,
            wasm_memory: vec![],
            gpu_memory: vec![],
            registers: vec![],
            metadata: SnapshotMetadata {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                created_unix_ms: 0,
                total_uncompressed_bytes: 0,
            },
            crc32: payload_crc32(&[], &[], &[]),
        };
        // Build a hand-rolled blob with a bad version field.
        let cfg = bincode::config::legacy();
        let encoded = bincode::serde::encode_to_vec(&s, cfg).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).unwrap();
        let err = SnapshotReader::new()
            .restore(&compressed)
            .expect_err("must reject");
        assert!(matches!(err, TensorWasmError::Serialization(_)));
        // Sanity: a corrected version round-trips.
        s.version = SNAPSHOT_VERSION;
        let encoded = bincode::serde::encode_to_vec(&s, cfg).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).unwrap();
        SnapshotReader::new().restore(&compressed).expect("ok");
    }

    #[test]
    fn magic_mismatch_is_rejected() {
        let s = Snapshot {
            magic: 0xDEAD_BEEF,
            version: SNAPSHOT_VERSION,
            wasm_memory: vec![],
            gpu_memory: vec![],
            registers: vec![],
            metadata: SnapshotMetadata {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                created_unix_ms: 0,
                total_uncompressed_bytes: 0,
            },
            crc32: payload_crc32(&[], &[], &[]),
        };
        let cfg = bincode::config::legacy();
        let encoded = bincode::serde::encode_to_vec(&s, cfg).unwrap();
        let compressed = zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).unwrap();
        let err = SnapshotReader::new()
            .restore(&compressed)
            .expect_err("must reject");
        assert!(matches!(err, TensorWasmError::Serialization(_)));
    }

    #[test]
    fn empty_bodies_round_trip() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(0),
                instance_id: InstanceId(0),
                wasm_memory: &[],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let r = SnapshotReader::new().restore(&bytes).expect("restore");
        assert!(r.wasm_memory.is_empty());
        assert!(r.gpu_memory.is_empty());
        assert!(r.registers.is_empty());
        assert_eq!(r.metadata.total_uncompressed_bytes, 0);
    }
}
