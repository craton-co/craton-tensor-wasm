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

#[cfg(feature = "signed-snapshots")]
use crate::format::{SIGNATURE_KIND_HMAC_SHA256, SNAPSHOT_VERSION_V3, V3_TRAILER_MAGIC};
#[cfg(feature = "artifact-backing")]
use crate::format::SNAPSHOT_VERSION_V2;
#[cfg(feature = "signed-snapshots")]
use zeroize::Zeroizing;

/// Magic bytes that identify a TensorWasm snapshot blob.
///
/// Stored at the head of the bincode payload. Read by [`crate::reader::SnapshotReader`]
/// to refuse blobs that were not produced by this crate. The literal spells
/// `BA11 5407` ("BALI SHOT") in leet hex — a frozen legacy magic from the
/// Project Bali era. Do not change without a snapshot-format version bump,
/// or every existing snapshot becomes unreadable.
pub const SNAPSHOT_MAGIC: u32 = 0xBA11_5407;

/// On-disk schema version emitted by default by [`SnapshotWriter::capture`].
///
/// Bumped whenever the serialised layout of `Snapshot` changes in a way that is
/// not bincode-compatible with previous releases. The reader accepts both this
/// (`v2`, unsigned) and [`crate::format::SNAPSHOT_VERSION_V3`] (HMAC-signed,
/// produced when the writer has had
/// [`SnapshotWriter::with_hmac_sha256_key`] called on it) — see `FORMAT.md`.
///
/// Version `2` adds the [`Snapshot::crc32`] payload checksum and enforces the
/// size limits defined in [`limits`]. Snapshots produced by version `1` writers
/// are not accepted.
///
/// The default stays bound to v2 through v0.3.x for backward compatibility;
/// the v0.4 release will flip the default to v3 (signed) and require operators
/// to opt out for unsigned writes.
pub const SNAPSHOT_VERSION: u32 = crate::format::SNAPSHOT_VERSION_V2;

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

    /// Hard ceiling on the compressed byte slice accepted by the reader (1 GiB).
    ///
    /// Snapshots arriving over the network may have been crafted to look small
    /// but decompress into terabytes. The reader rejects any input larger than
    /// this before calling zstd to bound the attacker's memory budget.
    ///
    /// Tightened to 1 GiB in T9 to reduce attacker pre-decompression memory
    /// footprint. Was 4 GiB. Adjust upward if a legitimate workload's
    /// snapshot exceeds this.
    ///
    /// On a 32-bit target this expression would overflow `usize`; the static
    /// assertion at the crate root ensures we only ever compile on 64-bit.
    pub const MAX_INPUT_BYTES: usize = 1024 * 1024 * 1024;

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
///
/// **Replay protection (v0.4 design landing).** The trailing
/// [`SnapshotMetadata::sequence_no`] and [`SnapshotMetadata::nonce`] fields
/// reserve room in the wire format for the freshness check described in
/// `FORMAT.md` § "Replay protection — v0.4". The fields are serialised today
/// but neither the reader nor the writer enforce them yet; that lands in
/// v0.4 together with `SnapshotReader::with_expected_nonce` and the
/// operator-side `last_seen` bookkeeping. Callers that rely on
/// bincode-exact compatibility with v0.3.x blobs must regenerate captures
/// after upgrading — the on-disk byte layout grows by 9 bytes per metadata
/// record (a `u64` plus the `Option<[u8;16]>` discriminant; +16 more when
/// the nonce is `Some`).
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
    /// Monotonic counter generated by the writer. Operators that want strict
    /// replay protection should track the highest seen `sequence_no` per
    /// signing key and refuse `restore` for any blob with `sequence_no <=
    /// last_seen`. The format itself does not enforce this — see
    /// `docs/SNAPSHOT-COMPATIBILITY.md#replay-protection` (v0.4).
    ///
    /// Defaults to `0` for snapshots produced by writers that have not
    /// opted in. Today every writer in this crate emits `0`; the v0.4
    /// release will switch to a writer-supplied monotonic counter.
    ///
    /// **v0.4 follow-up: enforced replay protection via this field.**
    /// Today: timestamp-based freshness (via
    /// [`crate::reader::SnapshotReader::with_max_age`]) is the only
    /// guard. The field is reserved on the wire so the v0.4 cutover can
    /// land without another format bump.
    #[cfg_attr(feature = "signed-snapshots", serde(default))]
    pub sequence_no: u64,
    /// Optional caller-supplied nonce. When set, the reader requires the
    /// caller to supply a matching expected nonce via
    /// `SnapshotReader::with_expected_nonce`. Defaults to None — backward
    /// compatible with operators that have not opted into the v0.4
    /// freshness check.
    ///
    /// **v0.4 follow-up: enforced replay protection via this field.**
    /// Today: timestamp-based freshness (via
    /// [`crate::reader::SnapshotReader::with_max_age`]) is the only
    /// guard. The field is reserved on the wire so the v0.4 cutover can
    /// land without another format bump.
    #[cfg_attr(feature = "signed-snapshots", serde(default))]
    pub nonce: Option<[u8; 16]>,
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
///
/// `Debug` is implemented manually to print byte-length placeholders rather
/// than the full byte vectors — a derived `Debug` over multi-GiB blobs would
/// spool gigabytes to logs on any `tracing::debug!(?snapshot)`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("magic", &format_args!("0x{:08x}", self.magic))
            .field("version", &self.version)
            .field("wasm_memory_len", &self.wasm_memory.len())
            .field("gpu_memory_len", &self.gpu_memory.len())
            .field("registers_len", &self.registers.len())
            .field("metadata", &self.metadata)
            .field("crc32", &format_args!("0x{:08x}", self.crc32))
            .finish()
    }
}

/// Borrowing mirror of [`Snapshot`] used only on the write path so capture does
/// not have to clone the caller's byte slices. Drives both the legacy v2/v3
/// envelope ([`SnapshotWriter::capture`]) and the artifact-store envelope
/// ([`SnapshotWriter::capture_to_artifact_store`]) — the only difference
/// between the two callers is which `version` field they fill in.
///
/// Field order, types (post-`serde_bytes` adaptation), and bincode encoding are
/// identical to [`Snapshot`], so a blob produced from `SnapshotRef` round-trips
/// through [`crate::reader::SnapshotReader::restore`] (or, under the
/// `artifact-backing` feature, through
/// [`crate::reader::SnapshotReader::restore_from_artifact_store`]) into the
/// owned form.
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
///
/// By default a writer emits the unsigned v2 envelope. Call
/// [`SnapshotWriter::with_hmac_sha256_key`] (requires the `signed-snapshots`
/// feature) to switch to the HMAC-SHA256-signed v3 envelope.
///
/// `Debug` is implemented manually to redact `hmac_key` — a derived `Debug`
/// would print all 32 key bytes via `{:?}` and expose the signing secret
/// any time a caller writes `tracing::debug!(?writer)` or similar.
///
/// `Copy` is intentionally NOT derived when the `signed-snapshots` feature
/// is enabled: the HMAC key is wrapped in [`zeroize::Zeroizing`] so its
/// backing bytes are scrubbed on drop, and `Zeroizing<T>` is never `Copy`
/// (a `Copy` would silently duplicate the secret and skip the scrub on
/// the original). The no-feature build keeps `Copy` for backward
/// compatibility — no secret material is present there.
#[cfg_attr(not(feature = "signed-snapshots"), derive(Copy))]
#[derive(Clone, Default)]
pub struct SnapshotWriter {
    /// zstd compression level to use. Defaults to [`DEFAULT_ZSTD_LEVEL`].
    pub zstd_level: i32,
    /// HMAC-SHA256 key used to sign v3 snapshots. `None` -> emit v2.
    ///
    /// Stored only when the `signed-snapshots` feature is enabled so the
    /// no-feature build keeps the legacy two-field struct layout and does
    /// not pay 33 bytes of state per writer instance. The field is `pub(crate)`
    /// rather than `pub` so callers cannot read the key back out by name
    /// once they have configured it.
    ///
    /// Wrapped in [`zeroize::Zeroizing`] so the 32 key bytes are
    /// overwritten when the writer is dropped — best-effort defence
    /// against the key surviving in swap-backed memory or in the
    /// allocator's freelist after the writer has gone out of scope.
    #[cfg(feature = "signed-snapshots")]
    pub(crate) hmac_key: Option<Zeroizing<[u8; 32]>>,
}

impl std::fmt::Debug for SnapshotWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("SnapshotWriter");
        d.field("zstd_level", &self.zstd_level);
        #[cfg(feature = "signed-snapshots")]
        d.field(
            "hmac_key",
            &self.hmac_key.as_ref().map(|_| "<REDACTED 32-byte HMAC key>"),
        );
        d.finish()
    }
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
    /// Construct a writer using [`DEFAULT_ZSTD_LEVEL`] and no HMAC key.
    pub const fn new() -> Self {
        Self {
            zstd_level: DEFAULT_ZSTD_LEVEL,
            #[cfg(feature = "signed-snapshots")]
            hmac_key: None,
        }
    }

    /// Construct a writer with an explicit zstd compression level and no HMAC key.
    ///
    /// Valid range is `1..=22`; out-of-range values are clamped by the zstd
    /// library at compression time.
    pub const fn with_level(zstd_level: i32) -> Self {
        Self {
            zstd_level,
            #[cfg(feature = "signed-snapshots")]
            hmac_key: None,
        }
    }

    /// Configure HMAC-SHA256 signing with a 32-byte key.
    ///
    /// Once configured, [`SnapshotWriter::capture`] emits the v3 envelope:
    /// the usual zstd(bincode(...)) blob with the inner `version` field bumped
    /// from `2` to `3`, followed by a one-byte signature-kind discriminant
    /// (`1` = HMAC-SHA256) and the 32-byte HMAC computed over the entire
    /// compressed prefix. See `FORMAT.md` for the exact byte layout.
    ///
    /// The key is held by value inside the writer and never logged or surfaced
    /// in error messages. Treat the writer as secret once this method has been
    /// called: it is `Clone + Copy`, so any copy carries the same key.
    ///
    /// Without this call (or without the `signed-snapshots` feature compiled
    /// in) the writer continues to emit the unsigned v2 envelope, preserving
    /// the v0.3.x default behaviour.
    #[cfg(feature = "signed-snapshots")]
    #[cfg_attr(docsrs, doc(cfg(feature = "signed-snapshots")))]
    #[must_use]
    pub fn with_hmac_sha256_key(mut self, key: [u8; 32]) -> Self {
        // Wrap the key in `Zeroizing` so the 32 bytes are scrubbed when the
        // writer drops. `Zeroizing::new` is not `const`, so this constructor
        // is no longer `const fn`. Existing callers were all runtime sites
        // (no `const` callers exist in-tree); see the `with_hmac_sha256_key`
        // grep audit in the commit that introduced the wrapper.
        self.hmac_key = Some(Zeroizing::new(key));
        self
    }

    /// Validate `state`'s blob sizes against the `limits` caps and build the
    /// metadata struct that any envelope (legacy v2/v3 or the v0.4 unified
    /// artifact store) needs to carry. Shared by [`SnapshotWriter::capture`]
    /// and (under the `artifact-backing` feature) by
    /// [`SnapshotWriter::capture_to_artifact_store`] so the two write paths
    /// cannot drift on which input is "too large" or on which metadata
    /// fields are populated.
    ///
    /// Returns `(metadata, crc32)`. Kept private because both consumers
    /// borrow the byte slices into a `SnapshotRef` to avoid a host-side
    /// copy of the (potentially multi-GiB) memory blobs: `capture`
    /// streams the `SnapshotRef` straight through bincode into the zstd
    /// encoder, while `capture_to_artifact_store` bincode-encodes a
    /// `SnapshotRef` into a `Vec<u8>` and hands the encoded bytes to
    /// the artifact store. The shared work stops at the metadata so the
    /// two paths can pick their own `SnapshotRef` `version` field
    /// (legacy v2/v3 on `capture`, fixed v2 on `capture_to_artifact_store`).
    fn build_metadata(&self, state: &InstanceState<'_>) -> Result<(SnapshotMetadata, u32)> {
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
        // T9: propagate a `SystemTime` failure as a Serialization error rather
        // than silently defaulting `created_unix_ms` to `0`. The reader's
        // freshness check (`SnapshotReader::with_max_age`) treats `0` as
        // "epoch", so a silent default would let a clock-broken host emit
        // snapshots that the reader would either always reject (any opted-in
        // `max_age`) or — worse, if the operator concludes the timestamps are
        // unreliable and disables the check — accept forever. Surfacing the
        // failure here forces the operator to fix the clock before captures
        // can proceed.
        let created_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                TensorWasmError::Serialization(
                    format!("snapshot created_unix_ms: SystemTime::now() before UNIX_EPOCH: {e}")
                        .into(),
                )
            })
            .and_then(|d| {
                u64::try_from(d.as_millis()).map_err(|_| {
                    TensorWasmError::Serialization(
                        "snapshot created_unix_ms: milliseconds since epoch overflows u64".into(),
                    )
                })
            })?;

        let crc32 = payload_crc32(state.wasm_memory, state.gpu_memory, state.registers);
        let metadata = SnapshotMetadata {
            tenant_id: state.tenant_id,
            instance_id: state.instance_id,
            created_unix_ms,
            total_uncompressed_bytes,
            // Replay-protection fields are sketched in the format today
            // but not yet driven by the writer. Both default values
            // ("no sequence", "no nonce") preserve the v0.3.x semantics:
            // operators that have not opted into the v0.4 freshness
            // check continue to see snapshots they captured before the
            // field existed as logically-equivalent (0, None) records.
            sequence_no: 0,
            nonce: None,
        };
        Ok((metadata, crc32))
    }

    /// Build a borrowing [`SnapshotRef`] view over `state` for the
    /// artifact-store write path. Mirrors the same size checks and metadata
    /// population as [`SnapshotWriter::capture`] (via
    /// [`SnapshotWriter::build_metadata`]) but produces a `SnapshotRef`
    /// instead of an owned [`Snapshot`] so the three potentially-GiB byte
    /// blobs are not host-side copied just to be handed to bincode.
    ///
    /// PERF (audit T21): pre-T21 this returned an owned `Snapshot` built
    /// from three `.to_vec()` calls. On a multi-GiB GPU capture that meant
    /// three full-payload copies before bincode even ran. The borrowing
    /// `SnapshotRef` serialises to byte-identical bincode (same field
    /// order, same `serde_bytes` adapter on the three byte fields) so the
    /// wire format and the reader path are unchanged.
    ///
    /// Returns the (`SnapshotRef`, `total_uncompressed_bytes`) pair so the
    /// caller can log the uncompressed size without redoing the sum.
    #[cfg(feature = "artifact-backing")]
    fn build_snapshot_ref<'a>(
        &self,
        state: InstanceState<'a>,
    ) -> Result<(SnapshotRef<'a>, u64)> {
        let (metadata, crc32) = self.build_metadata(&state)?;
        let total_uncompressed_bytes = metadata.total_uncompressed_bytes;
        Ok((
            SnapshotRef {
                magic: SNAPSHOT_MAGIC,
                // Artifact-backed snapshots reuse the v2 inner-version
                // discriminant: the outer envelope is the artifact store's
                // own (magic + content-hash + HMAC) frame, so the inline v3
                // HMAC trailer is redundant — we do not bump to v3 just
                // because the writer happens to have an HMAC key configured
                // for the legacy path. v0.4 may collapse this field
                // entirely once the artifact-store envelope is the only
                // shape on the wire.
                version: SNAPSHOT_VERSION_V2,
                wasm_memory: state.wasm_memory,
                gpu_memory: state.gpu_memory,
                registers: state.registers,
                metadata,
                crc32,
            },
            total_uncompressed_bytes,
        ))
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
        let (metadata, crc32) = self.build_metadata(&state)?;
        let total_uncompressed_bytes = metadata.total_uncompressed_bytes;

        // Pick the on-wire version. Without an HMAC key configured we emit the
        // legacy unsigned v2 envelope; with a key we bump to v3 so the reader
        // (which keys signature handling off the version field) knows to look
        // for the trailing signature kind + 32-byte HMAC.
        #[cfg(feature = "signed-snapshots")]
        let on_wire_version = if self.hmac_key.is_some() {
            SNAPSHOT_VERSION_V3
        } else {
            SNAPSHOT_VERSION
        };
        #[cfg(not(feature = "signed-snapshots"))]
        let on_wire_version = SNAPSHOT_VERSION;

        // Build the on-wire view by borrowing the caller's slices — no
        // host-side `.to_vec()` for the three memory blobs. bincode serialises
        // the borrowed `&[u8]` (via `serde_bytes`) identically to how it
        // deserialises into `Vec<u8>` (also via `serde_bytes`), so the wire
        // format is unchanged.
        let snapshot_ref = SnapshotRef {
            magic: SNAPSHOT_MAGIC,
            version: on_wire_version,
            wasm_memory: state.wasm_memory,
            gpu_memory: state.gpu_memory,
            registers: state.registers,
            metadata,
            crc32,
        };

        // Stream bincode directly into the zstd encoder so we never materialise
        // the full uncompressed payload as an intermediate `Vec<u8>`. For large
        // captures (hundreds of MiB) this eliminates a peak-resident copy and a
        // redundant pass over the buffer. `bincode::config::legacy()` keeps the
        // wire format byte-identical to bincode 1.x's default fixint+LE config.
        let cfg = bincode::config::legacy();
        // PERF (audit T21): pre-size `compressed` using a 4:1 ratio heuristic
        // over the input's total uncompressed size. The old constant 8 KiB
        // capacity forced ~10+ reallocations on every multi-MiB GPU capture
        // because the streaming zstd encoder writes in increments of a few
        // tens of KiB. 4:1 is a reasonable expected ratio for fp16/fp32
        // tensor memory (which dominates GPU snapshots) — overshooting wastes
        // a one-off allocation we'd grow into anyway, while undershooting
        // costs amortised O(N log N) reallocations. The clamp avoids
        // pathological behaviour: the 8 KiB floor keeps tiny captures from
        // allocating a zero-byte Vec, and the `MAX_INPUT_BYTES / 4` ceiling
        // (256 MiB at today's 1 GiB MAX_INPUT_BYTES) protects against a
        // hostile-but-passing-size_check input pushing the writer's
        // peak-resident footprint past the reader's hard cap. The output
        // bytes are unchanged — this is purely about reducing reallocation
        // count on the write path.
        let estimated_compressed = (total_uncompressed_bytes as usize / 4)
            .clamp(8 * 1024, limits::MAX_INPUT_BYTES / 4);
        let mut compressed: Vec<u8> = Vec::with_capacity(estimated_compressed);
        let mut encoder = zstd::stream::write::Encoder::new(&mut compressed, self.zstd_level)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd init: {e}").into()))?;
        bincode::serde::encode_into_std_write(&snapshot_ref, &mut encoder, cfg)
            .map_err(|e| TensorWasmError::Serialization(format!("bincode encode: {e}").into()))?;
        encoder
            .finish()
            .map_err(|e| TensorWasmError::Serialization(format!("zstd finish: {e}").into()))?;

        // v3: append [magic = V3_TRAILER_MAGIC][signature_kind = 1]
        // [HMAC-SHA256(key, compressed_prefix || magic || kind)].
        // The HMAC covers the entire v2-shaped prefix (i.e. every byte
        // written so far, which transitively covers magic, version, payload,
        // and CRC32 because they are all encoded inside the bincode/zstd
        // frame) **plus** the new 4-byte trailer magic and the
        // signature-kind discriminant. Including the trailer magic in the
        // HMAC input means an attacker cannot rewrite the magic bytes to
        // disguise a v3 blob as something else without invalidating the
        // signature. The key never leaves this scope and is never written
        // to the trace span.
        //
        // T8 (this commit) bumps the trailer from `[kind][sig]` (33 bytes)
        // to `[magic][kind][sig]` (37 bytes). The change is BREAKING for
        // any v3 blob produced by a pre-T8 writer.
        #[cfg(feature = "signed-snapshots")]
        if let Some(ref key) = self.hmac_key {
            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            // `key` is `&Zeroizing<[u8; 32]>`; `Zeroizing` is
            // `Deref<Target = [u8; 32]>`, so `&key[..]` borrows the full
            // 32-byte slice without copying. The zeroizing wrapper still
            // owns the bytes — they're scrubbed when `self` drops.
            let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key[..]).map_err(|_| {
                // `new_from_slice` only errors on invalid key length; ours is a
                // fixed [u8; 32], so this branch is unreachable in practice.
                // We still translate the error rather than panic, and keep the
                // message key-free.
                TensorWasmError::Serialization("HMAC init failed".into())
            })?;
            mac.update(&compressed);
            // T8: authenticate the trailer magic too. A reader that observes
            // `V3_TRAILER_MAGIC` and then verifies HMAC must see the same
            // bytes the writer signed, otherwise a magic-rewrite would
            // become a downgrade primitive once a future trailer revision
            // exists.
            mac.update(&V3_TRAILER_MAGIC);
            // snapshot 1.1: include the signature-kind discriminant byte in
            // the HMAC input so a future second variant (e.g. Ed25519)
            // cannot be substituted via trailer rewrite — that would
            // otherwise be a downgrade primitive while two kinds coexist.
            // Today there is only one kind, but mixing it in now means the
            // reader can validate it without a wire-format break later
            // (the reader's symmetric `mac.update(&[kind_byte])` lands in
            // the same commit).
            mac.update(&[SIGNATURE_KIND_HMAC_SHA256]);
            let sig = mac.finalize().into_bytes();
            // Reserve once for the full trailer (magic + kind + signature)
            // so the trailing extend never reallocates and the writer's
            // peak memory footprint is unchanged from pre-T8.
            compressed.reserve(V3_TRAILER_MAGIC.len() + 1 + sig.len());
            compressed.extend_from_slice(&V3_TRAILER_MAGIC);
            compressed.push(SIGNATURE_KIND_HMAC_SHA256);
            compressed.extend_from_slice(sig.as_slice());
        }

        debug!(
            uncompressed = total_uncompressed_bytes,
            compressed = compressed.len(),
            version = on_wire_version,
            "snapshot captured",
        );
        Ok(compressed)
    }

    /// Write the snapshot via [`tensor_wasm_artifacts::DiskArtifactStore`]
    /// rather than the inline v2/v3 envelope.
    ///
    /// The bincode-encoded [`Snapshot`] (NOT zstd-wrapped — the artifact
    /// store handles compression and HMAC itself) is handed to
    /// [`tensor_wasm_artifacts::DiskArtifactStore::put`], which wraps it in
    /// the unified envelope:
    ///
    /// ```text
    /// twasm-artifact01 || version(4) || blake3(payload) || zstd(payload) || hmac_sha256(prefix)
    /// ```
    ///
    /// This is the **v0.4 convergence path**. Today's invocation continues
    /// to use the inline envelope when this method is not called — the
    /// existing [`SnapshotWriter::capture`] is byte-for-byte unchanged so
    /// snapshots already on disk under any previous v0.3.x build remain
    /// readable through [`crate::reader::SnapshotReader::restore`].
    ///
    /// The writer's `zstd_level` and `hmac_key` fields are intentionally
    /// **not** consulted on this path: the artifact store owns its own
    /// compression level and HMAC key (passed in at construction time).
    /// Callers that want operator-tunable HMAC keys for snapshots can
    /// continue to use [`SnapshotWriter::capture`] under the v3 envelope
    /// until the v0.4 default cutover.
    ///
    /// Returns the [`tensor_wasm_artifacts::ContentHash`] under which the
    /// snapshot was stored. Pair it with
    /// [`crate::reader::SnapshotReader::restore_from_artifact_store`] to
    /// read the snapshot back.
    #[cfg(feature = "artifact-backing")]
    #[cfg_attr(docsrs, doc(cfg(feature = "artifact-backing")))]
    #[instrument(skip(self, state, store), fields(
        tenant = %state.tenant_id,
        instance = %state.instance_id,
    ))]
    pub fn capture_to_artifact_store(
        &self,
        state: InstanceState<'_>,
        store: &tensor_wasm_artifacts::DiskArtifactStore,
    ) -> Result<tensor_wasm_artifacts::ContentHash> {
        // Build the bincode-encoded Snapshot (NOT zstd-wrapped) — the
        // artifact store handles compression itself. The size caps from
        // `build_snapshot_ref`'s `build_metadata` call still apply, so an
        // oversized capture is rejected here before bincode runs.
        //
        // PERF (audit T21): encode from a borrowing `SnapshotRef` rather
        // than an owned `Snapshot`. The pre-T21 path called `.to_vec()`
        // on each of `wasm_memory`, `gpu_memory`, and `registers` —
        // three full-payload copies before bincode even ran. The
        // borrowing view serialises to byte-identical bincode (same
        // field order, same `serde_bytes` adapter) so the artifact
        // store sees the same payload and the reader's existing
        // `decode_from_slice::<Snapshot>` path keeps working without
        // change.
        let (snapshot_ref, total_uncompressed_bytes) = self.build_snapshot_ref(state)?;
        let bytes = bincode::serde::encode_to_vec(&snapshot_ref, bincode::config::legacy())
            .map_err(|e| {
                TensorWasmError::Serialization(format!("bincode encode: {e}").into())
            })?;
        let hash = store.put(&bytes).map_err(|e| {
            // `ArtifactError` is not part of the `TensorWasmError` enum
            // (it lives in a leaf crate that does not depend on
            // `tensor-wasm-core`), so we surface its `Display`
            // representation through the generic Serialization variant.
            // The artifact store's error messages are already
            // operator-facing (no key bytes, no secret material) so the
            // forward is safe.
            TensorWasmError::Serialization(
                format!("artifact store put: {e}").into(),
            )
        })?;
        debug!(
            uncompressed = total_uncompressed_bytes,
            encoded = bytes.len(),
            content_hash = %hash,
            "snapshot captured to artifact store",
        );
        Ok(hash)
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
        // Use a version that is neither v2 (unsigned) nor v3 (signed) so the
        // rejection is unambiguously about an unknown schema rather than
        // about a missing HMAC key.
        const UNKNOWN_VERSION: u32 = 99;
        let mut s = Snapshot {
            magic: SNAPSHOT_MAGIC,
            version: UNKNOWN_VERSION,
            wasm_memory: vec![],
            gpu_memory: vec![],
            registers: vec![],
            metadata: SnapshotMetadata {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                created_unix_ms: 0,
                total_uncompressed_bytes: 0,
                sequence_no: 0,
                nonce: None,
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
                sequence_no: 0,
                nonce: None,
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

    /// Without the HMAC key configured, the writer must keep emitting v2
    /// regardless of the `signed-snapshots` feature being compiled in. This
    /// is the v0.3.x backward-compat contract.
    #[test]
    fn default_writer_still_emits_v2() {
        let (wasm, gpu, regs) = sample_state();
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");
        // Decompress and peek at the version field (LE u32 at offset 4..8 of
        // the bincode payload). It must read as v2 for an unsigned writer.
        let decompressed = zstd::decode_all(bytes.as_slice()).expect("decode");
        let version = u32::from_le_bytes([
            decompressed[4],
            decompressed[5],
            decompressed[6],
            decompressed[7],
        ]);
        assert_eq!(version, crate::format::SNAPSHOT_VERSION_V2);
    }

    /// Configuring an HMAC key bumps the on-wire version to v3 and appends a
    /// 33-byte trailer (`[kind=1][32-byte signature]`). A v3 blob round-trips
    /// through a reader configured with the same key.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn signed_writer_emits_v3_and_round_trips() {
        let key = [0x42u8; 32];
        let (wasm, gpu, regs) = sample_state();
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(9),
                instance_id: InstanceId(99),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");

        // T8: trailer is now `[magic: 4][kind: 1][sig: 32]` = 37 bytes.
        // The magic sits at `len - SIGNATURE_TRAILER_LEN` and the kind
        // byte sits at `len - SIGNATURE_TRAILER_LEN + V3_TRAILER_MAGIC_LEN`.
        assert!(bytes.len() >= crate::format::SIGNATURE_TRAILER_LEN + 4);
        let trailer_start = bytes.len() - crate::format::SIGNATURE_TRAILER_LEN;
        assert_eq!(
            &bytes[trailer_start..trailer_start + crate::format::V3_TRAILER_MAGIC_LEN],
            &crate::format::V3_TRAILER_MAGIC,
        );
        assert_eq!(
            bytes[trailer_start + crate::format::V3_TRAILER_MAGIC_LEN],
            crate::format::SIGNATURE_KIND_HMAC_SHA256,
        );

        // Decompress just the prefix to confirm the inner version reads as v3.
        let compressed_prefix = &bytes[..bytes.len() - crate::format::SIGNATURE_TRAILER_LEN];
        let decompressed = zstd::decode_all(compressed_prefix).expect("decode prefix");
        let version = u32::from_le_bytes([
            decompressed[4],
            decompressed[5],
            decompressed[6],
            decompressed[7],
        ]);
        assert_eq!(version, crate::format::SNAPSHOT_VERSION_V3);

        let restored = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect("restore v3");
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
        assert_eq!(restored.version, crate::format::SNAPSHOT_VERSION_V3);
    }
}
