# tensor-wasm-snapshot wire format

This document specifies the on-disk byte layout produced by
`SnapshotWriter::capture` and consumed by `SnapshotReader::restore`.

## Envelope

```
zstd(
  bincode(
    Snapshot {
      magic:        u32 = SNAPSHOT_MAGIC (0xBA11_5407),
      version:      u32 = SNAPSHOT_VERSION (currently 2),
      wasm_memory:  Vec<u8>,    // serde_bytes — length-prefixed byte string
      gpu_memory:   Vec<u8>,    // serde_bytes — length-prefixed byte string
      registers:    Vec<u8>,    // serde_bytes — length-prefixed byte string
      metadata: SnapshotMetadata {
        tenant_id:                 TenantId(u64),
        instance_id:               InstanceId(u64),
        created_unix_ms:           u64,
        total_uncompressed_bytes:  u64,
      },
      crc32:        u32,        // IEEE polynomial; covers wasm_memory ++
                                //   gpu_memory ++ registers in that order
    }
  )
)
```

## Encoding

| Layer | Library | Settings |
|-------|---------|----------|
| Outer | [`zstd`](https://docs.rs/zstd) | Default compression level (`DEFAULT_ZSTD_LEVEL = 3`), single frame. The reader streams via `zstd::stream::read::Decoder` capped with `Read::take`. |
| Inner | [`bincode`](https://docs.rs/bincode) 1.x | Default config: little-endian, fixint, reject-trailing-bytes on encode. The reader uses `bincode::Options::with_limit(max_decompressed)` plus `.allow_trailing_bytes()` to refuse oversized `Vec<u8>` length-prefix abuse without crashing on benign envelope padding. |
| Byte blobs | [`serde_bytes`](https://docs.rs/serde_bytes) | Each of `wasm_memory`, `gpu_memory`, `registers` is serialised as a single length-prefixed byte string (`u64` LE length, then raw bytes). On the write path the bytes are borrowed via a `SnapshotRef<'a>` mirror struct so no host-side copy is made; on the read path they deserialise into owned `Vec<u8>`. The wire bytes are identical to the pre-`serde_bytes` `Vec<u8>` encoding (bincode emits `len: u64 LE` + raw bytes in both cases). |

## Size caps

All caps live in the `limits` module. The reader rejects any input that
violates one of them; `TensorWasmError::Serialization` is returned in every case.

| Constant | Value | Enforced where |
|----------|-------|----------------|
| `MAX_INPUT_BYTES` | `64 * 1024 * 1024 * 64` (~4 GiB) | Before zstd is invoked, against the raw compressed slice. |
| `MAX_DECOMPRESSED_BYTES` | `256 * 1024 * 1024` (256 MiB) | Streaming zstd cap (`Read::take`) **and** bincode allocator cap (`Options::with_limit`). Overridable per reader via `SnapshotReader::with_max_decompressed`. |
| `MAX_WASM_MEMORY_BYTES` | `1024 * 1024 * 1024` (1 GiB) | Capture (writer) and restore (reader) checks against `wasm_memory.len()`. |
| `MAX_GPU_MEMORY_BYTES` | `4 * 1024 * 1024 * 1024` (4 GiB) | Capture and restore checks against `gpu_memory.len()`. |
| `MAX_REGISTERS_BYTES` | `1024 * 1024` (1 MiB) | Capture and restore checks against `registers.len()`. |
| `MAX_TOTAL_PAYLOAD_BYTES` | sum of the three blob caps + 64 KiB | Exported for callers that pre-size buffers; not enforced directly (the per-blob caps already bound it). |

The compressed-input and decompressed-stream caps together bound the
allocator exposure of a single restore call. The per-blob caps catch the
case where a single legitimate-looking blob has been resized inside the
bincode payload.

## CRC32

`crc32` is computed with the IEEE polynomial (via the `crc32fast` crate)
over the concatenation of `wasm_memory`, `gpu_memory`, and `registers`
**in that order, with no separators**. The reader recomputes it after
deserialisation and rejects the snapshot if it does not match.

CRC32 is an integrity check, not a security primitive — it catches bit
flips in storage and transit but does not authenticate the source. Pair
with a transport-layer signature (mTLS, signed manifest) when restoring
from untrusted peers.

## Platform

`tensor-wasm-snapshot` only compiles on 64-bit targets. A `const _: () = assert!`
at the crate root enforces this; the size caps would silently truncate
`usize` on 32-bit hosts and break the bounding guarantees the reader
documents.

## Version history

| `SNAPSHOT_VERSION` | Changes | Compatible with |
|--------------------|---------|-----------------|
| `1` | Initial release. Fields: `magic`, `version`, `wasm_memory`, `gpu_memory`, `registers`, `metadata`. No `crc32`, no enforced size caps. | Itself only. |
| `2` *(current)* | Added `Snapshot::crc32` (computed as described above). Added the `limits` size caps and reader enforcement. Switched byte blobs to `serde_bytes` for zero-copy write — wire-identical to the prior `Vec<u8>` encoding. | Itself only. The reader refuses any blob whose `version` field is not exactly `2`. |

A `version` mismatch is a hard error; the reader does not attempt to
migrate older snapshots in place. Re-capture from the live instance is
the supported upgrade path.
