//! `SnapshotReader`: restore a [`Snapshot`] from a compressed byte blob.
//!
//! The reader is the strict counterpart to [`crate::writer::SnapshotWriter`]:
//! it never panics on malformed input. Magic and version mismatches, truncated
//! zstd frames, and broken bincode payloads are all surfaced as
//! [`BaliError::Serialization`]. The hot path is a single zstd decode followed
//! by a bincode deserialise; no I/O is performed.
//!
//! NOTE: cuda-feature code in this file is compile-tested on CUDA hosts only;
//! on no-CUDA hosts only the `#[cfg(not(feature = "cuda"))]` branches are
//! exercised. The cuda branches use the `cust` 0.3.x unified-memory APIs.

use bali_core::error::{BaliError, Result};
use tracing::{debug, instrument};

use crate::writer::{
    check_blob_size, limits, payload_crc32, Snapshot, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// Reverse of [`SnapshotWriter`](crate::writer::SnapshotWriter) — turns a
/// compressed byte blob back into an in-memory [`Snapshot`].
///
/// Stateless and `Send + Sync`; one instance can serve many concurrent
/// restores. Construction is `const`, so this is suitable for `static`
/// singletons.
#[derive(Clone, Copy, Debug, Default)]
pub struct SnapshotReader;

impl SnapshotReader {
    /// Construct a fresh reader. Equivalent to [`SnapshotReader::default`] but `const`.
    pub const fn new() -> Self {
        Self
    }

    /// Decode `bytes` previously produced by
    /// [`SnapshotWriter::capture`](crate::writer::SnapshotWriter::capture).
    ///
    /// Returns [`BaliError::Serialization`] for every malformed input — bad
    /// zstd frame, bad bincode bytes, wrong magic, wrong version, oversized
    /// input, oversized payload, or CRC32 mismatch. The function never panics,
    /// so callers can safely feed it untrusted bytes from disk or the network.
    ///
    /// Validation order is intentional: cheap header checks (input size, magic,
    /// version) happen before any expensive decompression or hashing.
    #[instrument(skip(self, bytes), fields(input_len = bytes.len()))]
    pub fn restore(&self, bytes: &[u8]) -> Result<Snapshot> {
        // Cap the raw input first to bound the attacker's memory budget before
        // zstd ever runs. A snapshot that decompresses to gigabytes is fine, as
        // long as the *compressed* slice itself is below this ceiling.
        if bytes.len() > limits::MAX_INPUT_BYTES {
            return Err(BaliError::Serialization(format!(
                "snapshot input too large: {} bytes exceeds cap of {} bytes",
                bytes.len(),
                limits::MAX_INPUT_BYTES,
            )));
        }

        let decompressed = zstd::decode_all(bytes)
            .map_err(|e| BaliError::Serialization(format!("zstd decode: {e}")))?;
        let snapshot: Snapshot = bincode::deserialize(&decompressed)
            .map_err(|e| BaliError::Serialization(format!("bincode decode: {e}")))?;

        if snapshot.magic != SNAPSHOT_MAGIC {
            return Err(BaliError::Serialization(format!(
                "snapshot magic mismatch: expected {:#X}, got {:#X}",
                SNAPSHOT_MAGIC, snapshot.magic,
            )));
        }
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(BaliError::Serialization(format!(
                "snapshot version mismatch: expected {}, got {}",
                SNAPSHOT_VERSION, snapshot.version,
            )));
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
            return Err(BaliError::Serialization(format!(
                "snapshot crc32 mismatch: expected {:#010X}, got {:#010X}",
                expected, snapshot.crc32,
            )));
        }

        debug!(
            decompressed = decompressed.len(),
            wasm = snapshot.wasm_memory.len(),
            gpu = snapshot.gpu_memory.len(),
            regs = snapshot.registers.len(),
            "snapshot restored",
        );
        Ok(snapshot)
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
pub fn restore_to_gpu(bytes: &[u8], device_index: u32) -> Result<RestoredOnGpu> {
    use cust::memory::MemoryAdvise;
    use cust::memory::UnifiedBuffer;
    use cust::stream::{Stream, StreamFlags};

    let snapshot = SnapshotReader::new().restore(bytes)?;

    // UnifiedBuffer::new requires a non-zero capacity to actually allocate;
    // a zero-length snapshot is allowed — we just produce an empty buffer.
    let mut gpu_buf: UnifiedBuffer<u8> = if snapshot.gpu_memory.is_empty() {
        // SAFETY: capacity 0 -> no allocation, no uninitialised reads possible.
        unsafe { UnifiedBuffer::uninitialized(0) }
            .map_err(|e| BaliError::CudaError(format!("UnifiedBuffer::uninitialized(0): {e:?}")))?
    } else {
        UnifiedBuffer::new(&0u8, snapshot.gpu_memory.len())
            .map_err(|e| BaliError::CudaError(format!("UnifiedBuffer::new: {e:?}")))?
    };

    if !snapshot.gpu_memory.is_empty() {
        gpu_buf.as_mut_slice().copy_from_slice(&snapshot.gpu_memory);

        let device = cust::device::Device::get_device(device_index as i32).map_err(|e| {
            BaliError::CudaError(format!("Device::get_device({device_index}): {e:?}"))
        })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| BaliError::CudaError(format!("Stream::new: {e:?}")))?;
        gpu_buf
            .prefetch_to_device(&stream, &device)
            .map_err(|e| BaliError::CudaError(format!("prefetch_to_device: {e:?}")))?;
        stream
            .synchronize()
            .map_err(|e| BaliError::CudaError(format!("stream.synchronize: {e:?}")))?;
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
    use bali_core::types::{InstanceId, TenantId};

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
            assert!(matches!(err, BaliError::Serialization(_)));
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
        assert!(matches!(err, BaliError::Serialization(_)));
    }
}
