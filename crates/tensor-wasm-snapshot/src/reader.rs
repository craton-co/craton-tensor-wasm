// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `SnapshotReader`: restore a [`Snapshot`] from a compressed byte blob.
//!
//! The reader is the strict counterpart to [`crate::writer::SnapshotWriter`]:
//! it never panics on malformed input. Magic and version mismatches, truncated
//! zstd frames, broken bincode payloads, oversized decompressed streams, and
//! oversized declared `Vec<u8>` lengths are all surfaced as
//! [`TensorWasmError::Serialization`]. The hot path is a streaming zstd decode
//! capped at [`limits::MAX_DECOMPRESSED_BYTES`] followed by a size-limited
//! bincode deserialise; no I/O is performed.
//!
//! NOTE: cuda-feature code in this file is compile-tested on CUDA hosts only;
//! on no-CUDA hosts only the `#[cfg(not(feature = "cuda"))]` branches are
//! exercised. The cuda branches use the `cust` 0.3.x unified-memory APIs.

use std::io::Read;

use tensor_wasm_core::error::{TensorWasmError, Result};
use tracing::{debug, instrument};

use crate::format::{SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3};
use crate::writer::{
    check_blob_size, limits, payload_crc32, Snapshot, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

#[cfg(feature = "signed-snapshots")]
use crate::format::{SignatureKind, HMAC_SHA256_SIG_LEN, SIGNATURE_TRAILER_LEN};

/// Reverse of [`SnapshotWriter`](crate::writer::SnapshotWriter) — turns a
/// compressed byte blob back into an in-memory [`Snapshot`].
///
/// Stateless (the fields are configuration, not buffers) and `Send + Sync`;
/// one instance can serve many concurrent restores. Construction is `const`,
/// so this is suitable for `static` singletons.
///
/// The default `max_decompressed` is [`limits::MAX_DECOMPRESSED_BYTES`] (256
/// MiB). Override via [`SnapshotReader::with_max_decompressed`] for trusted
/// snapshots that legitimately exceed the default.
///
/// By default the reader accepts both v2 (unsigned) and v3 (HMAC-SHA256
/// signed) blobs:
/// - A v2 blob is accepted as-is (legacy behaviour).
/// - A v3 blob is **rejected** unless [`SnapshotReader::with_hmac_sha256_key`]
///   has been called to install the verification key.
///
/// Call [`SnapshotReader::require_signature`] to also refuse unsigned v2
/// blobs (defence-in-depth for production deployments).
///
/// `Debug` is implemented manually to redact `hmac_key` — a derived `Debug`
/// would print all 32 key bytes via `{:?}` and expose the signing secret
/// any time a caller writes `tracing::debug!(?reader)` or similar.
#[derive(Clone, Copy)]
pub struct SnapshotReader {
    /// Hard ceiling on bytes the streaming zstd decoder is allowed to emit
    /// before being aborted. Bounds the attacker's memory budget independent
    /// of what the bincode `Vec<u8>` length fields claim.
    max_decompressed: usize,
    /// HMAC-SHA256 key used to verify v3 signatures. `None` -> v3 inputs
    /// are rejected.
    #[cfg(feature = "signed-snapshots")]
    hmac_key: Option<[u8; 32]>,
    /// When `true`, v2 (unsigned) inputs are rejected even if otherwise
    /// well-formed. Allows operators to enforce signature-only restores
    /// without compiling a separate binary.
    require_signature: bool,
}

impl Default for SnapshotReader {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SnapshotReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("SnapshotReader");
        d.field("max_decompressed", &self.max_decompressed);
        #[cfg(feature = "signed-snapshots")]
        d.field(
            "hmac_key",
            &self.hmac_key.as_ref().map(|_| "<REDACTED 32-byte HMAC key>"),
        );
        d.field("require_signature", &self.require_signature);
        d.finish()
    }
}

impl SnapshotReader {
    /// Construct a fresh reader with the default decompressed-size cap
    /// ([`limits::MAX_DECOMPRESSED_BYTES`]). Equivalent to
    /// [`SnapshotReader::default`] but `const`.
    pub const fn new() -> Self {
        Self {
            max_decompressed: limits::MAX_DECOMPRESSED_BYTES,
            #[cfg(feature = "signed-snapshots")]
            hmac_key: None,
            require_signature: false,
        }
    }

    /// Configure HMAC-SHA256 verification with a 32-byte key.
    ///
    /// Required before the reader can accept v3 blobs. Without a key, any v3
    /// input is rejected with a `Serialization` error ("snapshot is signed
    /// (v3) but reader has no HMAC key"). v2 inputs continue to be accepted
    /// unless [`SnapshotReader::require_signature`] has also been called.
    ///
    /// The key is held by value and never logged or surfaced in error
    /// messages. Treat the reader as secret once this method has been
    /// called: it is `Clone + Copy`.
    #[cfg(feature = "signed-snapshots")]
    #[cfg_attr(docsrs, doc(cfg(feature = "signed-snapshots")))]
    #[must_use]
    pub const fn with_hmac_sha256_key(mut self, key: [u8; 32]) -> Self {
        self.hmac_key = Some(key);
        self
    }

    /// Refuse v2 (unsigned) snapshots, accepting only v3 with a valid HMAC.
    ///
    /// Combined with [`SnapshotReader::with_hmac_sha256_key`], this is the
    /// production-grade defaults-against-downgrade configuration: an attacker
    /// cannot bypass signature verification by stripping the v3 trailer and
    /// re-encoding the inner payload as v2, because the reader refuses v2
    /// outright. Without an HMAC key configured, this reader will reject
    /// *every* well-formed input (v2 by this flag, v3 by the missing key) —
    /// callers must set both for a usable configuration.
    #[must_use]
    pub const fn require_signature(mut self) -> Self {
        self.require_signature = true;
        self
    }

    /// Override the decompressed-stream cap.
    ///
    /// Use when restoring trusted snapshots that legitimately decompress past
    /// the 256 MiB default. The reader still refuses inputs whose compressed
    /// size exceeds [`limits::MAX_INPUT_BYTES`].
    ///
    /// Semantic note (bincode 2.x migration): in the 1.x era this knob also
    /// bounded the bincode allocator via `Options::with_limit(max)` — that
    /// limit was a runtime value. bincode 2.x's allocator limit is a
    /// *compile-time* `const` generic, so it can no longer be tied to a
    /// per-instance runtime override. The reader instead uses a static
    /// allocator ceiling of [`limits::MAX_TOTAL_PAYLOAD_BYTES`] (the sum of
    /// the per-blob caps plus envelope slack). This runtime knob continues
    /// to bound the *decompressed buffer* via `Read::take`, and the per-blob
    /// caps in [`limits`] still reject any oversized declared length after
    /// deserialisation — so the practical guarantees (no zip-bomb, no
    /// length-prefix abuse) are unchanged. Only the layer that catches the
    /// length-prefix abuse has shifted from "bincode allocator cap" to
    /// "static bincode allocator cap + per-blob check".
    #[must_use]
    pub const fn with_max_decompressed(mut self, max: usize) -> Self {
        self.max_decompressed = max;
        self
    }

    /// Maximum decompressed-stream size this reader will accept, in bytes.
    #[must_use]
    pub const fn max_decompressed(&self) -> usize {
        self.max_decompressed
    }

    /// Decode `bytes` previously produced by
    /// [`SnapshotWriter::capture`](crate::writer::SnapshotWriter::capture).
    ///
    /// Returns [`TensorWasmError::Serialization`] for every malformed input — bad
    /// zstd frame, bad bincode bytes, wrong magic, wrong version, oversized
    /// input, oversized decompressed stream, oversized declared `Vec<u8>`
    /// length, or CRC32 mismatch. The function never panics, so callers can
    /// safely feed it untrusted bytes from disk or the network.
    ///
    /// Validation order is intentional: cheap header checks (input size)
    /// happen before any expensive decompression. Decompression is streamed
    /// through a hard byte cap so a "zip bomb" payload cannot allocate past
    /// [`SnapshotReader::max_decompressed`] even if its compressed footprint
    /// fits under [`limits::MAX_INPUT_BYTES`].
    #[instrument(skip(self, bytes), fields(input_len = bytes.len()))]
    pub fn restore(&self, bytes: &[u8]) -> Result<Snapshot> {
        // Cap the raw input first to bound the attacker's memory budget before
        // zstd ever runs. A snapshot that decompresses to gigabytes is fine, as
        // long as the *compressed* slice itself is below this ceiling.
        if bytes.len() > limits::MAX_INPUT_BYTES {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot input too large: {} bytes exceeds cap of {} bytes",
                    bytes.len(),
                    limits::MAX_INPUT_BYTES,
                )
                .into(),
            ));
        }

        // Streaming zstd decode with a hard ceiling. `Read::take` aborts the
        // decoder once `max_decompressed + 1` bytes are emitted, so a zip-bomb
        // payload cannot grow the destination buffer past the cap. We probe
        // one byte past the cap so we can distinguish "decompresses to exactly
        // cap bytes" (allowed) from "decompresses to >cap bytes" (rejected).
        //
        // The decoder is wrapped around a `Cursor<&[u8]>` (which is itself a
        // `BufRead`) via `with_buffer`, bypassing zstd's default `BufReader`
        // wrap. That matters for v3: after decoding we read `cursor.position()`
        // to find the exact byte offset where the zstd frame ended. Any bytes
        // past that point are the v3 signature trailer. `single_frame()` stops
        // the decoder at the first frame end rather than treating the trailer
        // bytes as the start of a second concatenated frame (which would
        // otherwise fail with a misleading "zstd init" error).
        let cap = self.max_decompressed;
        let probe_limit = u64::try_from(cap)
            .ok()
            .and_then(|c| c.checked_add(1))
            .unwrap_or(u64::MAX);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut decoder = zstd::stream::read::Decoder::with_buffer(&mut cursor)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd init: {e}").into()))?
            .single_frame();
        // Pre-size the buffer to a small constant (1 MiB, capped by `cap`) to
        // avoid grow-by-doubling reallocs. We refuse to trust an
        // attacker-supplied frame-size hint, and the `Take` ceiling guarantees
        // we cannot grow past `cap + 1` bytes regardless of the input.
        let initial_capacity = cap.min(1024 * 1024);
        let mut decompressed: Vec<u8> = Vec::with_capacity(initial_capacity);
        (&mut decoder)
            .take(probe_limit)
            .read_to_end(&mut decompressed)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd decode: {e}").into()))?;
        drop(decoder);
        // Bytes the zstd decoder consumed from the input. The remainder
        // (`bytes[zstd_consumed..]`) is either empty (v2) or the v3 trailer.
        let zstd_consumed = usize::try_from(cursor.position()).unwrap_or(usize::MAX);
        if decompressed.len() > cap {
            return Err(TensorWasmError::Serialization(
                format!("snapshot decompressed payload too large: > {cap} bytes exceeds cap")
                    .into(),
            ));
        }

        // Bound the bincode allocator separately: even within `cap` bytes of
        // decompressed input, a malicious `Vec<u8>` length prefix could ask
        // for a much larger allocation. In bincode 2.x the allocator limit is
        // a const generic, so we use a static upper bound — the sum of every
        // per-blob cap plus envelope slack. Any single allocation that would
        // push past this static ceiling is refused by bincode before the
        // backing buffer is touched. The per-blob `check_blob_size` calls
        // below catch anything that survives this gate.
        //
        // `legacy()` keeps the on-wire encoding (LE, fixint) byte-identical
        // to bincode 1.x's `DefaultOptions::new().with_fixint_encoding().with_little_endian()`.
        // `decode_from_slice` returns `(T, consumed_bytes)` and ignores any
        // trailing bytes by default, replacing the explicit `.allow_trailing_bytes()`
        // opt-in from 1.x.
        let cfg = bincode::config::legacy()
            .with_limit::<{ crate::writer::limits::MAX_TOTAL_PAYLOAD_BYTES }>();
        let (snapshot, _read): (Snapshot, usize) =
            bincode::serde::decode_from_slice(decompressed.as_slice(), cfg)
                .map_err(|e| TensorWasmError::Serialization(format!("bincode decode: {e}").into()))?;

        if snapshot.magic != SNAPSHOT_MAGIC {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot magic mismatch: expected {:#X}, got {:#X}",
                    SNAPSHOT_MAGIC, snapshot.magic,
                )
                .into(),
            ));
        }

        // Version dispatch. v2 is the unsigned legacy envelope (no trailer);
        // v3 carries the HMAC-SHA256 trailer right after the zstd frame.
        // Anything else is a hard reject, just like v2 readers refused
        // future versions.
        match snapshot.version {
            SNAPSHOT_VERSION_V2 => {
                if self.require_signature {
                    return Err(TensorWasmError::Serialization(
                        "snapshot is unsigned (v2) but signature is required".into(),
                    ));
                }
                // v2 forbids any trailing bytes after the zstd frame —
                // otherwise an attacker could append a chosen 33-byte tail
                // and observe the reader's reaction.
                if zstd_consumed != bytes.len() {
                    return Err(TensorWasmError::Serialization(
                        format!(
                            "snapshot v2 has unexpected trailing bytes: {} byte(s) past zstd frame",
                            bytes.len().saturating_sub(zstd_consumed),
                        )
                        .into(),
                    ));
                }
            }
            SNAPSHOT_VERSION_V3 => {
                #[cfg(feature = "signed-snapshots")]
                {
                    self.verify_v3_trailer(bytes, zstd_consumed)?;
                }
                #[cfg(not(feature = "signed-snapshots"))]
                {
                    return Err(TensorWasmError::Serialization(
                        "snapshot is signed (v3) but the `signed-snapshots` feature is not compiled in"
                            .into(),
                    ));
                }
            }
            other => {
                return Err(TensorWasmError::Serialization(
                    format!(
                        "snapshot version mismatch: expected {} or {}, got {}",
                        SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3, other,
                    )
                    .into(),
                ));
            }
        }

        // Per-blob caps catch a tampered `Vec<u8>` length that survived bincode.
        check_blob_size(
            "wasm_memory",
            snapshot.wasm_memory.len(),
            limits::MAX_WASM_MEMORY_BYTES,
        )?;
        check_blob_size(
            "gpu_memory",
            snapshot.gpu_memory.len(),
            limits::MAX_GPU_MEMORY_BYTES,
        )?;
        check_blob_size(
            "registers",
            snapshot.registers.len(),
            limits::MAX_REGISTERS_BYTES,
        )?;

        // CRC32 is the integrity check — it catches in-place byte flips that
        // happen to survive zstd and bincode framing.
        let expected = payload_crc32(
            &snapshot.wasm_memory,
            &snapshot.gpu_memory,
            &snapshot.registers,
        );
        if snapshot.crc32 != expected {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot crc32 mismatch: expected {:#010X}, got {:#010X}",
                    expected, snapshot.crc32,
                )
                .into(),
            ));
        }

        // Cross-check the declared `total_uncompressed_bytes` against the
        // actual blob sums. Each `.len()` is already known to fit under its
        // per-blob cap (checked above), so the `checked_add`s here cannot
        // realistically overflow on a 64-bit host — but we use them anyway to
        // keep the arithmetic robust against future cap changes, and surface
        // any overflow as a format error rather than a wrap-around. Callers
        // that trust the metadata to size buffers or report compression ratios
        // would otherwise be misled by a tampered field.
        let actual_total = snapshot
            .wasm_memory
            .len()
            .checked_add(snapshot.gpu_memory.len())
            .and_then(|s| s.checked_add(snapshot.registers.len()))
            .ok_or_else(|| {
                TensorWasmError::Serialization(
                    "snapshot blob length sum overflowed usize".into(),
                )
            })?;
        if (actual_total as u64) != snapshot.metadata.total_uncompressed_bytes {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot metadata.total_uncompressed_bytes mismatch: \
                     expected {} (wasm_memory={} + gpu_memory={} + registers={}), got {}",
                    actual_total,
                    snapshot.wasm_memory.len(),
                    snapshot.gpu_memory.len(),
                    snapshot.registers.len(),
                    snapshot.metadata.total_uncompressed_bytes,
                )
                .into(),
            ));
        }

        debug!(
            decompressed = decompressed.len(),
            wasm = snapshot.wasm_memory.len(),
            gpu = snapshot.gpu_memory.len(),
            regs = snapshot.registers.len(),
            version = snapshot.version,
            "snapshot restored",
        );
        Ok(snapshot)
    }

    /// Validate the trailing `[signature_kind][signature]` bytes of a v3 blob.
    ///
    /// Called after [`SnapshotReader::restore`] has confirmed the inner
    /// `version` field is [`SNAPSHOT_VERSION_V3`]. `prefix_len` is the
    /// number of bytes the zstd decoder consumed — anything past that point
    /// in `bytes` is the trailer.
    ///
    /// Errors are deliberately generic: we never include the expected or
    /// observed signature bytes in the error message, since either could
    /// leak information about the secret key under a side-channel attacker.
    /// The constant-time `ct_eq` from `subtle` is used to compare the
    /// recomputed HMAC against the stored bytes so a timing oracle cannot
    /// recover the signature byte-by-byte.
    #[cfg(feature = "signed-snapshots")]
    fn verify_v3_trailer(&self, bytes: &[u8], prefix_len: usize) -> Result<()> {
        let key = self.hmac_key.as_ref().ok_or_else(|| {
            TensorWasmError::Serialization(
                "snapshot is signed (v3) but reader has no HMAC key".into(),
            )
        })?;

        // The trailer must be exactly `[kind: u8][32-byte sig]`. Anything
        // shorter is a truncation; anything longer is junk after the
        // signature (which we refuse rather than silently accept).
        let trailer = bytes
            .get(prefix_len..)
            .ok_or_else(|| TensorWasmError::Serialization("snapshot v3 trailer missing".into()))?;
        if trailer.len() != SIGNATURE_TRAILER_LEN {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot v3 trailer length mismatch: expected {} bytes, got {}",
                    SIGNATURE_TRAILER_LEN,
                    trailer.len(),
                )
                .into(),
            ));
        }
        let kind_byte = trailer[0];
        let sig_bytes = &trailer[1..];
        let kind = SignatureKind::from_byte(kind_byte).ok_or_else(|| {
            TensorWasmError::Serialization(
                format!("unknown signature_kind: {kind_byte}").into(),
            )
        })?;
        debug_assert_eq!(sig_bytes.len(), kind.signature_len());

        match kind {
            SignatureKind::HmacSha256 => {
                use hmac::{Hmac, Mac};
                use sha2::Sha256;
                use subtle::ConstantTimeEq;

                let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).map_err(|_| {
                    // [u8; 32] is always a valid HMAC-SHA256 key length; this
                    // branch is unreachable but we translate rather than
                    // panic and keep the key out of the message.
                    TensorWasmError::Serialization("HMAC init failed".into())
                })?;
                // HMAC covers every byte the zstd decoder consumed (the v2-shaped
                // prefix), i.e. transitively magic, version, payload, and CRC32.
                mac.update(&bytes[..prefix_len]);
                let expected = mac.finalize().into_bytes();
                // ConstantTimeEq returns `Choice` (0 or 1) without short-circuiting.
                let ok: bool = expected
                    .as_slice()
                    .ct_eq(sig_bytes)
                    .unwrap_u8()
                    == 1;
                if !ok {
                    return Err(TensorWasmError::Serialization(
                        "snapshot HMAC mismatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// In-memory representation of a snapshot whose `gpu_memory` blob has been
/// materialised into a CUDA `UnifiedBuffer` and prefetched to the target
/// device. Only available when the `cuda` feature is enabled.
///
/// The wasm-memory, registers, and metadata fields are owned exactly like
/// the corresponding fields on [`Snapshot`]; the GPU buffer is a fresh
/// allocation backed by managed memory, ready to be handed to a kernel.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub struct RestoredOnGpu {
    /// Raw bytes of the Wasm linear memory at capture time (host-side copy).
    pub wasm_memory: Vec<u8>,
    /// GPU device-side memory blob, now resident in unified memory.
    pub gpu_memory: cust::memory::UnifiedBuffer<u8>,
    /// Register-file snapshot (PTX-level state captured by the JIT).
    pub registers: Vec<u8>,
    /// Free-form metadata describing the snapshot's provenance.
    pub metadata: crate::writer::SnapshotMetadata,
}

/// Restore `bytes` and stage the `gpu_memory` payload onto the GPU at
/// `device_index` via `cuMemPrefetchAsync` on a fresh non-blocking stream.
///
/// On success the returned [`RestoredOnGpu`] owns a populated
/// `UnifiedBuffer<u8>` whose pages have been requested to migrate to the
/// target device. The stream is synchronised before return so the buffer
/// is observably ready (no half-prefetched state leaks to the caller).
///
/// Requires the `cuda` feature; on no-CUDA builds this symbol does not
/// exist, and callers should fall back to [`SnapshotReader::restore`]
/// followed by a manual host-to-device copy.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
#[instrument(skip(bytes), fields(input_len = bytes.len(), device_index = device_index))]
pub fn restore_to_gpu(bytes: &[u8], device_index: u32) -> Result<RestoredOnGpu> {
    use cust::memory::UnifiedBuffer;
    use cust::stream::{Stream, StreamFlags};

    let snapshot = SnapshotReader::new().restore(bytes)?;

    // UnifiedBuffer::new requires a non-zero capacity to actually allocate;
    // a zero-length snapshot is allowed — we just produce an empty buffer.
    let mut gpu_buf: UnifiedBuffer<u8> = if snapshot.gpu_memory.is_empty() {
        // SAFETY: capacity 0 -> no allocation, no uninitialised reads possible.
        unsafe { UnifiedBuffer::uninitialized(0) }
            .map_err(|e| TensorWasmError::CudaError(format!("UnifiedBuffer::uninitialized(0): {e:?}")))?
    } else {
        UnifiedBuffer::new(&0u8, snapshot.gpu_memory.len())
            .map_err(|e| TensorWasmError::CudaError(format!("UnifiedBuffer::new: {e:?}")))?
    };

    if !snapshot.gpu_memory.is_empty() {
        gpu_buf.as_mut_slice().copy_from_slice(&snapshot.gpu_memory);

        let device = cust::device::Device::get_device(device_index as i32).map_err(|e| {
            TensorWasmError::CudaError(format!("Device::get_device({device_index}): {e:?}"))
        })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| TensorWasmError::CudaError(format!("Stream::new: {e:?}")))?;
        gpu_buf
            .prefetch_to_device(&stream, &device)
            .map_err(|e| TensorWasmError::CudaError(format!("prefetch_to_device: {e:?}")))?;
        stream
            .synchronize()
            .map_err(|e| TensorWasmError::CudaError(format!("stream.synchronize: {e:?}")))?;
    }

    Ok(RestoredOnGpu {
        wasm_memory: snapshot.wasm_memory,
        gpu_memory: gpu_buf,
        registers: snapshot.registers,
        metadata: snapshot.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{InstanceState, SnapshotWriter};
    use tensor_wasm_core::types::{InstanceId, TenantId};

    #[test]
    fn malformed_bytes_return_error_without_panicking() {
        let reader = SnapshotReader::new();
        for bad in [
            &b""[..],
            &b"not zstd"[..],
            &[0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00][..], // zstd magic with truncated frame
            &[0xFF; 64][..],
        ] {
            let err = reader.restore(bad).expect_err("must reject");
            assert!(matches!(err, TensorWasmError::Serialization(_)));
        }
    }

    #[test]
    fn valid_round_trip_succeeds() {
        let wasm = vec![1u8, 2, 3, 4, 5];
        let gpu = vec![9u8; 1024];
        let regs = vec![0x42u8; 16];
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(11),
                instance_id: InstanceId(22),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");
        let restored = SnapshotReader::new().restore(&bytes).expect("restore");
        assert_eq!(restored.magic, SNAPSHOT_MAGIC);
        assert_eq!(restored.version, SNAPSHOT_VERSION);
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
    }

    #[test]
    fn truncated_blob_is_error() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Chop off the trailing half of the compressed payload.
        let truncated = &bytes[..bytes.len() / 2];
        let err = SnapshotReader::new()
            .restore(truncated)
            .expect_err("must reject");
        assert!(matches!(err, TensorWasmError::Serialization(_)));
    }

    #[test]
    fn with_max_decompressed_overrides_default() {
        let reader = SnapshotReader::new().with_max_decompressed(1024);
        assert_eq!(reader.max_decompressed(), 1024);
    }

    /// `require_signature` causes well-formed v2 blobs to be rejected with a
    /// message that mentions the unsigned envelope.
    #[test]
    fn require_signature_rejects_v2() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .require_signature()
            .restore(&bytes)
            .expect_err("v2 must be rejected when signature required");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("unsigned") && m.contains("required"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A reader with no HMAC key configured refuses every v3 blob, even one
    /// that was produced by a writer using the same key (the reader has no
    /// way to verify the signature so the only safe answer is to reject).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_without_key_is_rejected() {
        let key = [0x11u8; 32];
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(2),
                instance_id: InstanceId(2),
                wasm_memory: &[7; 64],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .restore(&bytes)
            .expect_err("v3 must be rejected when reader has no key");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("signed") && m.contains("HMAC key"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Round-trip a v3 blob with a matching key, then flip a byte in the
    /// stored signature and confirm the reader rejects it with the generic
    /// "HMAC mismatch" message (no key material leaks into the error).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_tampered_signature_is_rejected() {
        let key = [0xA5u8; 32];
        let mut bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(3),
                instance_id: InstanceId(3),
                wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
                gpu_memory: &[0xCC; 32],
                registers: &[0xDD; 8],
            })
            .expect("capture");
        // Flip the last byte of the signature.
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect_err("tampered signature must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("HMAC mismatch"), "unexpected message: {m}");
                // Defence in depth: confirm the key never leaks into the
                // error string (no hex characters from the 0xA5 pattern).
                assert!(!m.contains("A5"), "error must not leak key bytes: {m}");
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A v3 blob signed with key A is rejected by a reader configured with
    /// key B — the classic wrong-key case.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_wrong_key_is_rejected() {
        let key_a = [0x01u8; 32];
        let key_b = [0x02u8; 32];
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key_a)
            .capture(InstanceState {
                tenant_id: TenantId(4),
                instance_id: InstanceId(4),
                wasm_memory: &[42; 16],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key_b)
            .restore(&bytes)
            .expect_err("wrong key must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("HMAC mismatch"), "unexpected message: {m}");
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// An unknown signature_kind byte (anything other than 1 in v0.3.x) is
    /// rejected before the HMAC comparison runs.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_unknown_signature_kind_is_rejected() {
        let key = [0x77u8; 32];
        let mut bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(5),
                instance_id: InstanceId(5),
                wasm_memory: &[],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Trailer layout: `[kind][32-byte sig]`. The kind byte sits at
        // `len - 33` (i.e. `len - SIGNATURE_TRAILER_LEN`).
        let kind_pos = bytes.len() - SIGNATURE_TRAILER_LEN;
        bytes[kind_pos] = 0xFE; // not a known SignatureKind
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect_err("unknown signature_kind must be rejected");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("unknown signature_kind"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Sanity: the HMAC trailer length equals exactly 1 + the SHA-256 digest
    /// size, so the trailer offset arithmetic above is correct.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn trailer_constants_are_consistent() {
        assert_eq!(SIGNATURE_TRAILER_LEN, 1 + HMAC_SHA256_SIG_LEN);
    }
}
