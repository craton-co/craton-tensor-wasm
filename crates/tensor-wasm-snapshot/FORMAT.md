# tensor-wasm-snapshot wire format

This document specifies the on-disk byte layout produced by
`SnapshotWriter::capture` and consumed by `SnapshotReader::restore`.

## Envelope

```
zstd(
  bincode(
    Snapshot {
      magic:        u32 = SNAPSHOT_MAGIC (0xBA11_5407),
      version:      u32 = SNAPSHOT_VERSION (currently 2; v3 readers also accept 3, see below),
      wasm_memory:  Vec<u8>,    // serde_bytes — length-prefixed byte string
      gpu_memory:   Vec<u8>,    // serde_bytes — length-prefixed byte string
      registers:    Vec<u8>,    // serde_bytes — length-prefixed byte string
      metadata: SnapshotMetadata {
        tenant_id:                 TenantId(u64),
        instance_id:               InstanceId(u128),
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
| Inner | [`bincode`](https://docs.rs/bincode) 2.x | `bincode::config::legacy()`: little-endian, fixint — byte-identical wire format to bincode 1.x's `DefaultOptions::new().with_fixint_encoding().with_little_endian()`. The reader uses `legacy().with_limit::<{ MAX_TOTAL_PAYLOAD_BYTES }>()` (a compile-time `const` generic ceiling, ~5 GiB) to refuse oversized `Vec<u8>` length-prefix abuse; `bincode::serde::decode_from_slice` ignores trailing bytes by default, replacing the explicit `.allow_trailing_bytes()` opt-in from 1.x. |
| Byte blobs | [`serde_bytes`](https://docs.rs/serde_bytes) | Each of `wasm_memory`, `gpu_memory`, `registers` is serialised as a single length-prefixed byte string (`u64` LE length, then raw bytes). On the write path the bytes are borrowed via a `SnapshotRef<'a>` mirror struct so no host-side copy is made; on the read path they deserialise into owned `Vec<u8>`. The wire bytes are identical to the pre-`serde_bytes` `Vec<u8>` encoding (bincode emits `len: u64 LE` + raw bytes in both cases). |

## Size caps

All caps live in the `limits` module. The reader rejects any input that
violates one of them; `TensorWasmError::Serialization` is returned in every case.

| Constant | Value | Enforced where |
|----------|-------|----------------|
| `MAX_INPUT_BYTES` | `64 * 1024 * 1024 * 64` (~4 GiB) | Before zstd is invoked, against the raw compressed slice. |
| `MAX_DECOMPRESSED_BYTES` | `256 * 1024 * 1024` (256 MiB) | Streaming zstd cap (`Read::take`). Overridable per reader via `SnapshotReader::with_max_decompressed`. |
| `MAX_WASM_MEMORY_BYTES` | `1024 * 1024 * 1024` (1 GiB) | Capture (writer) and restore (reader) checks against `wasm_memory.len()`. |
| `MAX_GPU_MEMORY_BYTES` | `4 * 1024 * 1024 * 1024` (4 GiB) | Capture and restore checks against `gpu_memory.len()`. |
| `MAX_REGISTERS_BYTES` | `1024 * 1024` (1 MiB) | Capture and restore checks against `registers.len()`. |
| `MAX_TOTAL_PAYLOAD_BYTES` | sum of the three blob caps + 64 KiB | Bincode 2.x allocator ceiling — passed as the `with_limit::<N>` const generic so the deserialiser refuses any single allocation past this static bound before reading the body. The per-blob caps still validate after decode. |

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
flips in storage and transit but does not authenticate the source. The
v3 wire format below adds an authenticator on top of the v2 envelope;
use it (or pair v2 with a transport-layer signature: mTLS, signed
manifest) when restoring from untrusted peers.

## v3 wire format (signed-snapshots feature)

The v3 envelope **wraps** the v2 envelope, it does not replace it. A v3
blob is the bytes of a v2 capture (with the inner `version` field bumped
to `3`) followed by a single signature-kind discriminant and a
fixed-length signature. The HMAC covers the entire v2-shaped prefix
(equivalently: every byte from `magic` through `crc32` inclusive, *as
encoded on the wire* — the entire zstd frame).

### Byte layout

```
+----------------------------------------------------------+
| zstd(                                                    |
|   bincode(                                               |
|     Snapshot {                                           |
|       magic:        u32 = SNAPSHOT_MAGIC,                |
|       version:      u32 = SNAPSHOT_VERSION_V3 (= 3),     |
|       wasm_memory:  Vec<u8>,                             |
|       gpu_memory:   Vec<u8>,                             |
|       registers:    Vec<u8>,                             |
|       metadata:     SnapshotMetadata,                    |
|       crc32:        u32,                                 |
|     }                                                    |
|   )                                                      |
| )                                                        |  <- end of zstd frame
+----------------------------------------------------------+
| signature_kind: u8 = SIGNATURE_KIND_HMAC_SHA256 (= 1)    |
| signature:      [u8; 32]  -- HMAC-SHA256 output          |
+----------------------------------------------------------+
```

The trailer is **not** zstd-compressed. It sits *after* the zstd frame so
that:

1. Generic zstd tooling that runs the decoder in single-frame mode (which
   the reader does) sees only the authenticated prefix and ignores the
   trailer cleanly — there is no "garbage at end of frame" error path.
2. The reader can detect the trailer's location purely by asking the
   zstd decoder how many input bytes it consumed: anything past that
   offset is the trailer. No length prefix is needed.

### HMAC inputs

```
HMAC-SHA256(key, prefix_bytes)
```

where `prefix_bytes` is `input[..zstd_consumed]` — i.e. the complete v2-
shaped envelope, magic and version and CRC32 included. Because every
byte of the v2 envelope is authenticated, an attacker cannot strip the
trailer and re-encode the payload as v2 without the reader noticing
(provided the operator has called `SnapshotReader::require_signature` to
refuse the unsigned envelope).

### Key length

The key is exactly **32 bytes** (256 bits), the natural block size for
HMAC-SHA256. Shorter keys are not accepted; longer keys would have to be
hashed down to 32 bytes by HMAC, which we refuse for forward compatibility.

### Algorithm rationale

HMAC-SHA256 was chosen because:

- It is the modern Rust ecosystem default for symmetric message
  authentication (`hmac` + `sha2` are the canonical RustCrypto crates,
  pre-`ring` and pre-`aws-lc-rs` in dependency footprint).
- The reader already uses `subtle::ConstantTimeEq` elsewhere in the
  workspace (see `tensor-wasm-api`) for constant-time comparisons, so
  pulling in `subtle` here matches existing precedent.
- The 32-byte output is the same size as a SHA-256 digest, so storage and
  network overhead are 33 bytes per snapshot regardless of compressed
  size.
- Symmetric authentication is sufficient for the threat model — operators
  control both the writer and the reader. An asymmetric signature (e.g.
  Ed25519) would let third parties verify snapshots, which is a non-goal
  for v0.3.x. The `SignatureKind` enum is `#[non_exhaustive]` so a
  future variant can be added without a wire-format break.

### Defaults

The writer **defaults to v2** in v0.3.x: operators opt into v3 by calling
`SnapshotWriter::with_hmac_sha256_key(key)`. The reader **accepts both**
v2 and v3 by default; calling `SnapshotReader::require_signature()`
flips the reader into v3-only mode. The default will switch to "v3 on
write" in v0.4 — see `docs/SNAPSHOT-COMPATIBILITY.md` for the migration
plan.

### Constants

| Constant | Value | Defined in |
|----------|-------|------------|
| `SNAPSHOT_VERSION_V2` | `2` | `format.rs` |
| `SNAPSHOT_VERSION_V3` | `3` | `format.rs` |
| `SIGNATURE_KIND_HMAC_SHA256` | `1` | `format.rs` |
| `HMAC_SHA256_SIG_LEN` | `32` | `format.rs` |
| `SIGNATURE_TRAILER_LEN` | `33` (= `1 + HMAC_SHA256_SIG_LEN`) | `format.rs` |

## Platform

`tensor-wasm-snapshot` only compiles on 64-bit targets. A `const _: () = assert!`
at the crate root enforces this; the size caps would silently truncate
`usize` on 32-bit hosts and break the bounding guarantees the reader
documents.

## Version history

| `SNAPSHOT_VERSION` | Changes | Compatible with |
|--------------------|---------|-----------------|
| `1` | Initial release. Fields: `magic`, `version`, `wasm_memory`, `gpu_memory`, `registers`, `metadata`. No `crc32`, no enforced size caps. | Itself only. |
| `2` *(current writer default)* | Added `Snapshot::crc32` (computed as described above). Added the `limits` size caps and reader enforcement. Switched byte blobs to `serde_bytes` for zero-copy write — wire-identical to the prior `Vec<u8>` encoding. | Itself, and v3 readers (which accept both). |
| `3` | Adds an HMAC-SHA256 trailer after the zstd frame (`[signature_kind: u8][32-byte signature]`). The inner `version` field is `3`; the inner bincode payload is otherwise identical to v2. The HMAC covers the full v2-shaped prefix. Produced by writers that have had `SnapshotWriter::with_hmac_sha256_key` called; refused by readers without an HMAC key. | Itself only. v2 readers (pre-v0.3.5) refuse v3 because the `version` field is unknown to them. |

A `version` mismatch (anything other than `2` or `3`) is a hard error;
the reader does not attempt to migrate older snapshots in place. Re-
capture from the live instance is the supported upgrade path.
