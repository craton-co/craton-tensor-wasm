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
        sequence_no:               u64,            // monotonic counter; 0 = unset. LIVE: written by SnapshotWriter::with_sequence_no, enforced by SnapshotReader::with_min_sequence_no.
        nonce:                     Option<[u8;16]>, // bincode: 1-byte tag (0=None, 1=Some) then 16 bytes when Some. LIVE: written by SnapshotWriter::with_nonce, enforced by SnapshotReader::with_expected_nonce.
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
| `MAX_INPUT_BYTES` | `1024 * 1024 * 1024` (1 GiB) | Before zstd is invoked, against the raw compressed slice. |
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
| trailer_magic:  [u8; 4] = V3_TRAILER_MAGIC = b"S3T1"     |
| signature_kind: u8      = SIGNATURE_KIND_HMAC_SHA256 (1) |
| signature:      [u8; 32] -- HMAC-SHA256 output           |
+----------------------------------------------------------+
```

The trailer is **not** zstd-compressed. It sits *after* the zstd frame so
that:

1. Generic zstd tooling that runs the decoder in single-frame mode (which
   the reader does) sees only the authenticated prefix and ignores the
   trailer cleanly — there is no "garbage at end of frame" error path.
2. The reader can detect the trailer's location with a single cheap
   constant-offset read: a v3 blob is classified by checking that
   `input[input.len() - SIGNATURE_TRAILER_LEN..input.len() - SIGNATURE_TRAILER_LEN + 4]
   == V3_TRAILER_MAGIC`. This lets the reader **authenticate before
   decoding** — HMAC verifies the compressed prefix before any zstd or
   bincode work runs on it. No length prefix is needed; the trailer is
   fixed-length (37 bytes today: 4-byte magic + 1-byte kind + 32-byte
   signature) and the `SignatureKind` discriminant is forward-compatible.

#### T8 — magic-prefix detector (BREAKING)

Prior to T8 the v3 detector was a **single-byte** sniff:
`input[input.len() - 33] == SIGNATURE_KIND_HMAC_SHA256` (i.e. `== 1`).
Because that byte sat inside the zstd frame epilogue of a legitimate v2
blob with ~1/256 probability, a v2 capture could be misclassified as v3
and then rejected by the HMAC check. The mis-classified blob never
parsed — but the downgrade-shaped error message and the wasted HMAC
work were both observable side channels. T8 prepends a 4-byte
`V3_TRAILER_MAGIC` (`b"S3T1"`, "Snapshot 3 Trailer v1") to the trailer
and uses *that* as the classifier, shrinking the false-positive rate to
~1/2^32. The `SNAPSHOT_VERSION_V3` revision number is **not** bumped —
the magic prefix is itself the discriminator and the inner bincode
payload is byte-identical to the pre-T8 v3 shape.

The new trailer layout is **not** backward-compatible with v3 blobs
produced by pre-T8 writers (whose 33-byte trailer lacks the magic
prefix and so fails the classifier). Operators with archived pre-T8 v3
captures must re-sign them with a current writer; v2 snapshots are
unaffected.

### HMAC inputs

```
HMAC-SHA256(key, prefix_bytes || V3_TRAILER_MAGIC || [signature_kind])
```

where `prefix_bytes` is `input[..input.len() - SIGNATURE_TRAILER_LEN]` —
i.e. the complete v2-shaped envelope, magic and version and CRC32
included, every byte the zstd decoder would later see. The HMAC input
also covers the 4-byte `V3_TRAILER_MAGIC` and the 1-byte
`signature_kind` discriminant so that an attacker who rewrites the
trailer header (e.g. to disguise the blob or to substitute a future
signature kind) is caught at verification time. Because every byte of
the v2 envelope and the trailer header is authenticated, an attacker
cannot strip the trailer and re-encode the payload as v2 without the
reader noticing (provided the operator has called
`SnapshotReader::require_signature` to refuse the unsigned envelope).

### Validation order

The reader runs "authenticate then parse" — HMAC verification is the
**second** step, before any expensive or attacker-shaped work:

1. Reject inputs larger than `MAX_INPUT_BYTES` (cheap length check).
2. **Classify by trailer and verify HMAC** over the compressed prefix.
   On HMAC failure the reader returns `Serialization("snapshot HMAC
   mismatch")` immediately — zstd and bincode never see the bytes.
3. Stream-decompress the authenticated prefix (capped by
   `max_decompressed`), then bincode-decode into a `Snapshot`.
4. Check magic, version-consistency (the inner `version` must agree with
   the trailer-derived classification), per-blob caps, CRC32, and the
   `total_uncompressed_bytes` cross-check.

This order means that a forged or tampered v3 blob cannot drive the
zstd or bincode decoders as a side channel — the entire downstream
pipeline only ever runs on bytes whose HMAC matches the configured
key.

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
  network overhead are 37 bytes per snapshot (4-byte magic + 1-byte kind +
  32-byte signature) regardless of compressed size.
- Symmetric authentication is sufficient when operators control both the
  writer and the reader. When verifiers should *not* be trusted with signing
  capability (many readers, single publisher), the asymmetric Ed25519 trailer
  (`signature_kind = 2`, see "Ed25519 asymmetric signatures" above) is the
  better fit: verifiers hold only the public key. The `SignatureKind` enum is
  `#[non_exhaustive]`, which is what let Ed25519 be added without a
  wire-format break.

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
| `V3_TRAILER_MAGIC` | `b"S3T1"` (4 bytes) | `format.rs` |
| `V3_TRAILER_MAGIC_LEN` | `4` | `format.rs` |
| `SIGNATURE_KIND_HMAC_SHA256` | `1` | `format.rs` |
| `SIGNATURE_KIND_ED25519` | `2` | `format.rs` |
| `HMAC_SHA256_SIG_LEN` | `32` | `format.rs` |
| `ED25519_SIG_LEN` | `64` | `format.rs` |
| `SIGNATURE_TRAILER_LEN` | `37` (HMAC; = `V3_TRAILER_MAGIC_LEN + 1 + HMAC_SHA256_SIG_LEN`) | `format.rs` |
| `ED25519_TRAILER_LEN` | `69` (= `V3_TRAILER_MAGIC_LEN + 1 + ED25519_SIG_LEN`) | `format.rs` |

## Ed25519 asymmetric signatures (`signature_kind = 2`)

The v3 envelope also supports an **asymmetric** trailer. It reuses the same
`V3_TRAILER_MAGIC` and inner `version = 3` as the HMAC trailer; only the
`signature_kind` byte (`2`) and the signature length (64 bytes) differ:

```
+----------------------------------------------------------+
| zstd(bincode(Snapshot { version = 3, ... }))             |  <- end of zstd frame
+----------------------------------------------------------+
| trailer_magic:  [u8; 4] = V3_TRAILER_MAGIC = b"S3T1"     |
| signature_kind: u8      = SIGNATURE_KIND_ED25519 (2)     |
| signature:      [u8; 64] -- Ed25519 (RFC 8032) signature |
+----------------------------------------------------------+
```

### Signed message

```
Ed25519-sign(signing_key, prefix_bytes || V3_TRAILER_MAGIC || [signature_kind = 2])
```

— byte-identical in *shape* to the HMAC input, so the prefix (the entire
v2-shaped zstd frame), the trailer magic, and the kind byte are all
authenticated. Because the kind byte is signed, an attacker cannot rewrite
it to `1` (HMAC) to attempt a cross-scheme downgrade without invalidating the
signature; the reader's classifier additionally refuses a trailer whose kind
byte is inconsistent with the trailer length it was detected at.

### Why asymmetric

HMAC is symmetric: every verifier must hold the same secret, which can also
**forge** snapshots. Ed25519 lets a single publisher hold the private
`ed25519_dalek::SigningKey` while arbitrarily many verifiers hold only the
public `ed25519_dalek::VerifyingKey` — a compromised verifier cannot mint new
snapshots. Configure via `SnapshotWriter::with_ed25519_signing_key(signing_key)`
on the write side and `SnapshotReader::with_ed25519_verifying_key(verifying_key)`
on the read side. If both an Ed25519 key and an HMAC key are configured on a
writer, the Ed25519 trailer takes precedence (the writer never double-signs).

### Reader detection of variable-length trailers

The trailer detector probes both candidate lengths — 69 (Ed25519) then 37
(HMAC) — and accepts the first whose 4-byte magic *and* immediately-following
kind byte are self-consistent with the probed length. This disambiguates the
two trailer sizes while keeping the ~1/2^32 magic false-positive guarantee.
An HMAC blob is never emitted through the v0.4 artifact envelope when an
Ed25519 key is configured; the writer routes Ed25519 captures through the
inline (legacy) v3 envelope because the artifact envelope authenticates with
HMAC only.

## Replay protection (LANDED) — `sequence_no` and `nonce`

The `SnapshotMetadata::sequence_no` (`u64`) and `nonce` (`Option<[u8; 16]>`)
fields are now **live** (previously reserved-but-inert):

- **Writer** — `SnapshotWriter::with_sequence_no(n)` and
  `SnapshotWriter::with_nonce(bytes)` populate the fields. Defaults remain
  `0` / `None` (the v0.3.x "unset" semantics) for writers that do not opt in.
- **Reader** — `SnapshotReader::with_min_sequence_no(floor)` rejects any blob
  whose `sequence_no < floor`; `SnapshotReader::with_expected_nonce(bytes)`
  rejects any blob whose `nonce != Some(bytes)` (including a `None` nonce).
  Both checks are opt-in and run **after** authentication and the structural
  checks, so for a v3 blob the signature has already certified both fields —
  an attacker cannot rewrite them to slip past the checks without re-signing.

### Operator pattern: per-key highest-seen sequence_no

Track a persistent `last_seen` sequence number **per signing key**. On
restore, construct the reader with `with_min_sequence_no(last_seen)` (or
`last_seen + 1` to forbid replays of the exact last value), and after a
successful restore update `last_seen = max(last_seen, restored.metadata.sequence_no)`.
A replayed older capture then carries a `sequence_no` below the floor and is
rejected — closing the rollback window that timestamp freshness
(`with_max_age`) alone cannot: an attacker who replays a once-valid capture
inside the `max_age` window still trips the sequence floor.

Rejections surface as `TensorWasmError::Serialization` with a distinct,
greppable message per check (`"snapshot sequence_no ... below floor ..."`,
`"snapshot nonce mismatch ..."`, `"snapshot nonce missing ..."`).

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
| `3` | Adds an HMAC-SHA256 trailer after the zstd frame (`[V3_TRAILER_MAGIC: 4][signature_kind: u8][32-byte signature]` = 37 bytes; pre-T8 = `[signature_kind: u8][32-byte signature]` = 33 bytes, see T8 note below). The inner `version` field is `3`; the inner bincode payload is otherwise identical to v2. The HMAC covers the full v2-shaped prefix concatenated with the trailer magic and the kind byte. Produced by writers that have had `SnapshotWriter::with_hmac_sha256_key` called; refused by readers without an HMAC key. **T8 (BREAKING):** the trailer was bumped from 33 to 37 bytes by prepending `V3_TRAILER_MAGIC` (`b"S3T1"`) to replace the ~1/256 false-positive single-byte trailer sniff with a ~1/2^32 magic-prefix check. The `SNAPSHOT_VERSION_V3` number is unchanged; the magic prefix is itself the discriminator. Pre-T8 v3 captures no longer parse and must be re-signed with a current writer. | Itself only (post-T8 trailer shape). v2 readers (pre-v0.3.6) refuse v3 because the `version` field is unknown to them. |

A `version` mismatch (anything other than `2` or `3`) is a hard error;
the reader does not attempt to migrate older snapshots in place. Re-
capture from the live instance is the supported upgrade path.

## v4 (artifact-store-backed, default in v0.4 — T40)

The `artifact-backing` cargo feature is **on by default** as of v0.4
(T40). New writes from `SnapshotWriter::capture` route through the
unified `tensor-wasm-artifacts` envelope when the writer has an HMAC
key configured via `SnapshotWriter::with_hmac_sha256_key`; reads
auto-detect the envelope by its leading 16-byte magic and decode it
without ever entering the v3 / v2 paths. The legacy v2 and v3
envelopes documented above remain accepted on the **read** side
indefinitely — every snapshot already on disk continues to load
byte-for-byte unchanged.

### Detection ordering (reader)

`SnapshotReader::restore` dispatches by the leading-magic test, in
this order:

1. **`bytes[..16] == b"twasm-artifact01"`** → decode via the artifact
   envelope path (see "On-disk shape" below). Returns
   `Serialization` for any artifact-envelope failure other than the
   leading-magic mismatch; in particular, a tampered or wrong-key v4
   blob is rejected here and is **not** retried as a legacy blob.
2. **Trailer-magic prefix at `bytes[len - 37..len - 33] == b"S3T1"`**
   → decode via the legacy v3 path (T8 magic-prefix trailer; see
   "v3 wire format" above). Requires the `signed-snapshots` feature
   and an HMAC key configured on the reader.
3. **Otherwise** → decode via the legacy v2 path (the bare
   zstd(bincode(Snapshot)) shape).

Detection is purely magic-based; no version field is consulted until
after the envelope decoder has authenticated the bytes. This is what
keeps "authenticate then parse" honest on the v4 path (the artifact
crate's `decode_envelope_from_bytes` verifies HMAC before any
decompressed bytes reach the snapshot decoder) and what lets the
legacy v3 path keep its T8 magic-prefix guarantee.

### Opt-out (writer)

Two equivalent ways to keep emitting the legacy v3 wire format:

1. Per-call: `SnapshotWriter::capture_legacy(state)` — always emits
   v2/v3, regardless of feature flag or builder state.
2. Per-writer: `SnapshotWriter::with_legacy_envelope()` — flips this
   writer into legacy-only mode; subsequent `capture(...)` calls
   behave like `capture_legacy(...)`.

Both keep compiling on a build that disabled `artifact-backing` (the
`with_legacy_envelope` builder becomes an idempotency no-op).
Downstream consumers can also build with
`--no-default-features --features signed-snapshots` to disable v4
emission at compile time; that's the path for tooling whose entire
config surface predates v0.4.

### Behaviour without an HMAC key (graceful fallback)

If `SnapshotWriter::with_hmac_sha256_key` has not been called, the
v0.4 default `capture` cannot emit the v4 envelope (the outer
envelope mandates HMAC by construction). The writer falls back to
the legacy unsigned v2 envelope — same wire bytes the v0.3.x keyless
default produced. This is the load-bearing compatibility hook for
every in-tree caller (proptest, mem-conformance, bench) that
constructs `SnapshotWriter::new()` without further configuration:
their captures continue to be v2 on the wire and the default reader
parses them through the v2 fall-through above.

## Artifact-store backing (managed disk store, also under `artifact-backing`)

Behind the same `artifact-backing` cargo feature, the writer and
reader also expose two methods that route snapshots through the
persistent `tensor-wasm-artifacts::DiskArtifactStore` (atomic-rename
writes, content-addressed filenames, key-fingerprinted partitions)
instead of the in-memory envelope shape `capture` emits:

```rust
// Writer side (gated on the `artifact-backing` feature):
let hash: tensor_wasm_artifacts::ContentHash =
    writer.capture_to_artifact_store(state, &store)?;

// Reader side (same feature):
let snapshot: Snapshot =
    reader.restore_from_artifact_store(&store, &hash)?;
```

### On-disk shape

When a snapshot is captured through `capture_to_artifact_store`, the
bytes the disk store writes are **not** the v2/v3 envelope — they are
the artifact store's own envelope wrapping a bincode-encoded
[`Snapshot`] payload (zstd compression and HMAC are owned by the
artifact store, not by this crate):

```
twasm-artifact01(16) || version(4)=1 || blake3(payload)(32)
                    || zstd(bincode(Snapshot))
                    || hmac_sha256(prefix)(32)
```

The inner `Snapshot::version` field is `SNAPSHOT_VERSION_V2 = 2` (the
outer envelope already supplies authentication, so the v3 trailer would
be redundant). The reader accepts an inner `version` of `2` **only** on
this path (M1): no v3 inner trailer is written or verified inside the
artifact envelope, so a v3 discriminant would assert a signed-inner
format the reader cannot actually check. Until a signed-inner-v4 format
exists with its own inner-trailer verification, an inner `version` of
`3` is rejected here rather than waved through.

### Behaviour differences from the inline envelope

| Concern | Inline v2/v3 envelope | Artifact-store envelope |
|---|---|---|
| Magic | `0xBA11_5407` (4 bytes inside bincode) | `b"twasm-artifact01"` (16 bytes, outside) |
| Compression | zstd, configurable via `SnapshotWriter::with_level` | zstd, owned by `DiskArtifactStore` (level 3, not yet operator-tunable) |
| MAC | Optional HMAC-SHA256 trailer (v3) | Mandatory HMAC-SHA256 trailer |
| MAC key source | `SnapshotWriter::with_hmac_sha256_key` | `DiskArtifactStore::new(_, key)` |
| Content addressing | No (caller routes blobs externally) | Yes — `ContentHash = blake3(payload)` |
| Atomic write | Up to the caller | `tempfile::persist` (built in) |
| Key rotation | Operator-managed (rewrite-on-rotate) | Filename partitioned by `blake3(key)[..8]` (rotated-out keys appear as `NotFound`) |

The `SnapshotWriter::zstd_level` and `SnapshotWriter::hmac_key` fields
are **not** consulted by `capture_to_artifact_store` — the store owns
those settings. Mirror the same on the reader: `max_decompressed`,
`hmac_key`, and `require_signature` are not consulted by
`restore_from_artifact_store`.

### Validation order (reader)

1. `DiskArtifactStore::get` validates magic, version, HMAC trailer (in
   constant time), and the BLAKE3 content hash. A tampered, wrong-key,
   or foreign-format blob is rejected here, before any snapshot-crate
   code runs on the bytes.
2. The returned bincode payload is decoded under
   `bincode::config::legacy().with_limit::<MAX_TOTAL_PAYLOAD_BYTES>()`
   — the same static allocator ceiling the legacy path uses, so a
   tampered length prefix inside the bincode payload cannot drive a
   runaway allocation.
3. Inner magic, version (`2` or `3`), per-blob caps, CRC32, and
   `metadata.total_uncompressed_bytes` are validated exactly as on the
   legacy path. These are defence-in-depth on top of the artifact
   store's own integrity guarantees: a writer bug that produced a
   `Snapshot` with a stale CRC32 should still be rejected even though
   the artifact store happily authenticated the payload.

### v0.4 default-cutover — LANDED in T40

The v0.4 cutover landed as T40. The `artifact-backing` feature is now
a default cargo feature; `SnapshotWriter::capture` routes through the
unified envelope when an HMAC key is configured, and
`SnapshotReader::restore` auto-detects either shape via the leading
magic. The inline v2/v3 path is preserved as a legacy decoder
indefinitely — every existing on-disk snapshot continues to load — and
two opt-out hooks (`SnapshotWriter::with_legacy_envelope` and
`SnapshotWriter::capture_legacy`) let operators keep emitting the
legacy wire format when they need to. See the "v4
(artifact-store-backed, default in v0.4 — T40)" section above for
the detection ordering and opt-out semantics, and
`docs/ARTIFACT-STORE.md` § "Convergence plan — v0.4" for the full
rollout sequence.
