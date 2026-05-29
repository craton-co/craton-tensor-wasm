// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Unified content-addressed signed artifact store (roadmap feature #9).
//!
//! Folds the JIT L2 disk cache (`tensor-wasm-jit::cache::DiskCache`) and the
//! snapshot store (`tensor-wasm-snapshot::SnapshotWriter`/`SnapshotReader`)
//! into one primitive: a content-addressed, HMAC-signed, on-disk byte
//! blob keyed by BLAKE3 of the payload.
//!
//! ## v0.3.7 status: scaffold
//!
//! The trait + an in-memory impl + a disk impl land. Migration of the
//! JIT cache and snapshot crate to use this store is a v0.4 follow-up;
//! today they continue to use their own format. The new crate provides
//! the reference shape both implementations will converge on.
//!
//! ## Format
//!
//! ```text
//! magic(16) || version(4) || content_hash(32) || zstd(payload) || hmac_tag(32)
//! ```
//!
//! The HMAC covers magic..end-of-zstd. Verification is constant-time
//! via `subtle::ConstantTimeEq`. Key rotation: filenames include the
//! first 8 bytes of `blake3(hmac_key)` so distinct keys partition the
//! on-disk namespace cleanly.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use thiserror::Error;
use tracing::warn;
use zeroize::Zeroizing;

/// I/O buffer size used by both the streaming `put` writer and the
/// streaming `get` reader. 64 KiB matches the slab the zstd CLI prefers
/// and is large enough to keep syscall overhead negligible on big blobs
/// without wasting RAM on tiny ones.
const STREAM_BUF_LEN: usize = 64 * 1024;

/// Magic bytes identifying a TensorWasm unified-artifact blob.
///
/// Read by [`DiskArtifactStore::get`] before any HMAC work so foreign or
/// stale blobs short-circuit cheaply. The literal is exactly 16 ASCII
/// bytes — do not change without bumping [`ARTIFACT_VERSION`].
pub const ARTIFACT_MAGIC: [u8; 16] = *b"twasm-artifact01";

/// On-disk schema version for the unified artifact envelope.
pub const ARTIFACT_VERSION: u32 = 1;

/// Length of the trailing HMAC tag (HMAC-SHA256 output size).
pub const ARTIFACT_HMAC_LEN: usize = 32;

/// Length of the fixed header that precedes the zstd body:
/// `magic(16) || version(4) || content_hash(32)`.
pub const ARTIFACT_HEADER_LEN: usize = 16 + 4 + 32;

/// Default zstd compression level. Matches `tensor-wasm-snapshot`'s
/// `DEFAULT_ZSTD_LEVEL` so the two stores converge on the same setting.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Hard ceiling on the size of an individual payload accepted by
/// [`DiskArtifactStore::put`].
///
/// Without this cap an attacker who can drive `put` could request an
/// allocation proportional to an arbitrary input length, exhausting
/// process memory. 256 MiB matches the snapshot reader's
/// `MAX_DECOMPRESSED_BYTES` so the store's ingest ceiling lines up with
/// the largest legitimately-restorable snapshot today.
pub const MAX_PAYLOAD_LEN: usize = 256 * 1024 * 1024;

/// Hard ceiling on the size of the decompressed body emitted by
/// [`DiskArtifactStore::get`].
///
/// Defends against zstd "zip bomb" inputs that decompress at very high
/// ratios (a few MB of attacker-controlled bytes can otherwise expand
/// to gigabytes). The decoder is driven through
/// [`std::io::Read::take`] with a probe of `MAX_DECOMPRESSED_LEN + 1`
/// so we can distinguish "exactly cap" (allowed) from ">cap" (rejected)
/// without ever allocating past the cap. Sized larger than
/// [`MAX_PAYLOAD_LEN`] so a legitimate round-trip can still expand
/// slightly past the put-side cap due to compression accounting; the
/// snapshot reader uses a tighter 256 MiB cap for its own scenario.
pub const MAX_DECOMPRESSED_LEN: usize = 1024 * 1024 * 1024;

/// Errors returned by [`ArtifactStore`] implementations.
///
/// `Io` deliberately collapses the inner [`std::io::Error`] into a
/// unit-variant so the type stays `PartialEq`-able by callers that
/// match on it; the underlying error is logged at the call site. The
/// JIT L2 cache uses the same "log + swallow" convention for I/O
/// failures and treats them as a miss.
#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("artifact not found: {0}")]
    NotFound(String),
    #[error("magic mismatch")]
    BadMagic,
    #[error("unsupported version: {0}")]
    BadVersion(u32),
    #[error("HMAC verification failed")]
    BadHmac,
    #[error("content hash mismatch (expected {expected}, got {actual})")]
    HashMismatch { expected: String, actual: String },
    #[error("zstd decompression failed: {0}")]
    Decompression(String),
    /// Either the payload handed to `put` exceeded [`MAX_PAYLOAD_LEN`],
    /// or the body handed to `get` decompressed to more than
    /// [`MAX_DECOMPRESSED_LEN`] bytes. Carries the offending size and
    /// the cap that was tripped so operators can tell which side of the
    /// round-trip refused the request. Both fields are `usize` (i.e.
    /// `Eq`) so this variant stays compatible with any future
    /// `PartialEq` derive on [`ArtifactError`].
    #[error("artifact too large: {actual} bytes exceeds cap of {limit} bytes")]
    TooLarge { actual: usize, limit: usize },
    #[error("I/O error")]
    Io,
}

/// 32-byte BLAKE3 content hash. The `put` path computes this from the
/// uncompressed payload; the `get` path recomputes it from the decoded
/// body and rejects on mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl ContentHash {
    /// Compute the content hash of `payload` (BLAKE3 of the uncompressed bytes).
    pub fn of(payload: &[u8]) -> Self {
        ContentHash(blake3::hash(payload).into())
    }

    /// Wrap 32 raw bytes as a [`ContentHash`]. The bytes are assumed to
    /// be a BLAKE3 digest; this constructor does no hashing. Used by the
    /// disk store when reconstructing hashes from on-disk filenames and
    /// by callers that already hold a digest.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        ContentHash(bytes)
    }

    /// Borrow the raw 32-byte digest. Counterpart to [`Self::from_bytes`].
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Hex representation, equivalent to `format!("{self}")`. Useful at
    /// call sites that want a `String` without going through the
    /// formatter.
    pub fn to_hex(&self) -> String {
        format!("{self}")
    }
}

/// Artifact store trait. v0.3.7 ships an in-memory and a disk implementation.
///
/// Implementations MUST be `Send + Sync` so the store can be shared
/// across worker threads behind an `Arc`. They MUST also be safe under
/// concurrent `put` for the same payload — a put after a put for the
/// same content hash is idempotent.
pub trait ArtifactStore: Send + Sync {
    /// Insert `payload` into the store. The returned [`ContentHash`] is
    /// `blake3(payload)`; repeated calls with identical payloads return
    /// identical hashes and overwrite the existing entry in place.
    fn put(&self, payload: &[u8]) -> Result<ContentHash, ArtifactError>;
    /// Fetch the payload previously inserted under `hash`. Returns
    /// [`ArtifactError::NotFound`] on a genuine miss; integrity-failure
    /// variants ([`ArtifactError::BadMagic`], [`ArtifactError::BadVersion`],
    /// [`ArtifactError::BadHmac`], [`ArtifactError::HashMismatch`]) are
    /// returned when a record was present but the format checks rejected
    /// it.
    fn get(&self, hash: &ContentHash) -> Result<Vec<u8>, ArtifactError>;
    /// Enumerate the content hashes currently stored. Order is
    /// implementation-defined; callers that need a deterministic order
    /// must sort the result themselves.
    ///
    /// Returns [`ArtifactError::Io`] on an underlying enumeration
    /// failure (e.g. a `read_dir` permissions/I/O fault). This is
    /// deliberately distinct from an empty result so GC/audit callers
    /// can tell "store is empty" apart from "could not read the store".
    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError>;
}

// =====================================================================
// HMAC helpers and tee adapters — shared by `DiskArtifactStore` put and
// get paths so the streaming sides cannot drift on hash/key choice.
// =====================================================================

/// HMAC-SHA256 instance type used by both the streaming `put` and `get`
/// paths. Centralising the type alias keeps the two sides from drifting
/// on hash/key choice.
type ArtifactMac = hmac::Hmac<sha2::Sha256>;

/// Construct a fresh incremental HMAC instance over `key`. Both the
/// streaming put and streaming get build one of these and feed bytes in
/// chunk-by-chunk via `Mac::update`, then `finalize_into_tag`.
fn new_mac(key: &[u8; 32]) -> ArtifactMac {
    use hmac::Mac;
    // `new_from_slice` only errors on invalid key length; ours is a
    // fixed 32 bytes so the unwrap is sound (mirrors the same pattern
    // in `tensor-wasm-snapshot::SnapshotWriter::capture`).
    <ArtifactMac as Mac>::new_from_slice(&key[..]).expect("HMAC-SHA256 accepts any 32-byte key")
}

/// Finalise an incremental HMAC into the fixed-length tag.
fn finalize_into_tag(mac: ArtifactMac) -> [u8; ARTIFACT_HMAC_LEN] {
    use hmac::Mac;
    let out = mac.finalize().into_bytes();
    let mut tag = [0u8; ARTIFACT_HMAC_LEN];
    tag.copy_from_slice(out.as_slice());
    tag
}

/// 8-byte hex fingerprint of the HMAC key. Used to partition the on-disk
/// namespace when multiple stores share a directory under different
/// keys — without this, two writers with different keys would clobber
/// each other's files on a hash collision, and the second reader would
/// always fail the HMAC check.
fn key_fingerprint_hex(key: &[u8; 32]) -> String {
    let h = blake3::hash(&key[..]);
    h.as_bytes()[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// `Write` adapter that tees every byte into both an inner writer (the
/// `BufWriter<File>` backing the temp envelope) and an HMAC instance
/// (the streaming MAC over the same bytes).
///
/// This is what lets `put` avoid the intermediate framed `Vec`: as zstd
/// emits compressed bytes through its encoder, the encoder writes into a
/// `MacWriter` whose downstream is the on-disk file. The HMAC sees the
/// exact byte sequence that lands on disk (header + zstd body) without
/// any second pass over a materialised buffer.
struct MacWriter<'a, W: Write> {
    inner: W,
    mac: &'a mut ArtifactMac,
}

impl<'a, W: Write> MacWriter<'a, W> {
    fn new(inner: W, mac: &'a mut ArtifactMac) -> Self {
        Self { inner, mac }
    }
}

impl<W: Write> Write for MacWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Write to the underlying sink first. If the file-write fails we
        // do NOT update the MAC, so a torn write doesn't leave the MAC
        // covering bytes that didn't actually reach disk.
        let n = self.inner.write(buf)?;
        // Only feed the prefix that the writer actually accepted; this
        // matches the contract of `Write::write` and keeps the MAC in
        // sync with the file's byte stream.
        use hmac::Mac;
        self.mac.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// `Read` adapter that tees every byte read from an inner reader into
/// an HMAC instance.
///
/// `get` uses this so the body bytes feeding the zstd decoder also feed
/// the HMAC in a single pass — no second walk of the prefix to compute
/// the expected tag. The HMAC sees whatever bytes were actually
/// delivered to the consumer (the decoder), which is exactly the
/// invariant we need for the MAC to be byte-compatible with `put`.
struct MacReader<'a, R: Read> {
    inner: R,
    mac: &'a mut ArtifactMac,
}

impl<'a, R: Read> MacReader<'a, R> {
    fn new(inner: R, mac: &'a mut ArtifactMac) -> Self {
        Self { inner, mac }
    }
}

impl<R: Read> Read for MacReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        // Feed exactly the bytes the upstream observed. A short read is
        // fine; the MAC simply sees fewer bytes this round and picks up
        // the rest on the next call.
        use hmac::Mac;
        self.mac.update(&buf[..n]);
        Ok(n)
    }
}

// =====================================================================
// Disk store
// =====================================================================

/// On-disk content-addressed signed artifact store.
///
/// Format on disk for a single blob:
///
/// ```text
/// magic(16) || version(4) || content_hash(32) || zstd(payload) || hmac_tag(32)
/// ```
///
/// The HMAC covers everything except the trailing 32-byte tag itself.
/// Verification uses constant-time comparison.
pub struct DiskArtifactStore {
    dir: PathBuf,
    hmac_key: Zeroizing<[u8; 32]>,
    /// `blake3(hmac_key)[..8]` rendered as 16 ascii-hex chars, computed
    /// once in [`Self::new`]. Used as the per-key filename segment by
    /// `path_for`, `get`, `put`, and `list`; caching it here avoids
    /// re-hashing the key on every store operation.
    key_fp_hex: String,
}

impl DiskArtifactStore {
    /// Construct a disk store rooted at `dir`, signing with `hmac_key`.
    /// The directory is created lazily on the first `put`.
    pub fn new(dir: PathBuf, hmac_key: [u8; 32]) -> Self {
        let key_fp_hex = key_fingerprint_hex(&hmac_key);
        Self {
            dir,
            hmac_key: Zeroizing::new(hmac_key),
            key_fp_hex,
        }
    }

    /// Compute the on-disk path for `hash` under this store's key.
    ///
    /// Filename format: `{content_hash_hex}.{key_fp_hex}.bin`. The key
    /// fingerprint segment partitions the namespace per HMAC key so
    /// two stores in the same dir under different keys never collide.
    fn path_for(&self, hash: &ContentHash) -> PathBuf {
        let hash_hex = hash.to_string();
        // The rendered hash MUST be exactly 64 lowercase ascii-hex chars
        // before it becomes a path component — `ContentHash`'s `Display`
        // guarantees this (32 bytes, two hex digits each), but assert it
        // so a future change to the digest size or formatter can never
        // silently feed a traversal-shaped or wrong-length segment into
        // `Path::join`.
        debug_assert!(
            hash_hex.len() == 64
                && hash_hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "ContentHash rendered to an unexpected filename segment: {hash_hex:?}"
        );
        let key_hex = &self.key_fp_hex;
        self.dir.join(format!("{hash_hex}.{key_hex}.bin"))
    }
}

impl ArtifactStore for DiskArtifactStore {
    fn put(&self, payload: &[u8]) -> Result<ContentHash, ArtifactError> {
        // Reject oversized payloads BEFORE any allocation or I/O. An
        // attacker who can drive `put` could otherwise request an
        // allocation proportional to `payload.len()` and OOM the
        // process. The check is on the already-borrowed `payload`
        // slice — we do not materialise the caller's bytes ourselves —
        // so this is a pure refusal, not a second allocation.
        if payload.len() > MAX_PAYLOAD_LEN {
            warn!(
                target: "tensor_wasm_artifacts",
                actual = payload.len(),
                limit = MAX_PAYLOAD_LEN,
                "rejecting oversized payload"
            );
            return Err(ArtifactError::TooLarge {
                actual: payload.len(),
                limit: MAX_PAYLOAD_LEN,
            });
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| {
            warn!(
                target: "tensor_wasm_artifacts",
                dir = %self.dir.display(),
                error = %e,
                "create_dir_all failed"
            );
            ArtifactError::Io
        })?;
        let hash = ContentHash::of(payload);

        // T22 streaming write: header + zstd(body) + HMAC tag stream
        // directly through a buffered `MacWriter` into a `NamedTempFile`,
        // with no intermediate `Vec<u8>` for the framed envelope. The
        // HMAC sees the exact bytes that land on disk; the writer is
        // wrapped so every chunk also feeds `Mac::update` on its way
        // through. Result: peak heap during `put` is bounded by the
        // 64 KiB BufWriter slab plus zstd's internal window, regardless
        // of payload size.
        let final_path = self.path_for(&hash);
        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "tempfile create failed");
            ArtifactError::Io
        })?;
        let mut mac = new_mac(&self.hmac_key);

        // Streaming sink composition:
        //   file <- BufWriter <- MacWriter (tees to MAC) <- zstd encoder
        //
        // The encoder writes compressed bytes into `tee`, which forks
        // each byte into the 64 KiB BufWriter (then the on-disk file)
        // AND the running HMAC. No materialised framed buffer.
        {
            let buf_writer = BufWriter::with_capacity(STREAM_BUF_LEN, tmp.as_file_mut());
            let mut tee = MacWriter::new(buf_writer, &mut mac);

            // Header (magic || version || content_hash) is written
            // through the tee so the HMAC covers the same prefix the
            // old one-shot path did.
            tee.write_all(&ARTIFACT_MAGIC).map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "header magic write failed");
                ArtifactError::Io
            })?;
            tee.write_all(&ARTIFACT_VERSION.to_le_bytes()).map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "header version write failed");
                ArtifactError::Io
            })?;
            tee.write_all(&hash.0).map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "header hash write failed");
                ArtifactError::Io
            })?;

            // Compress the payload streaming-style through the encoder.
            // `zstd::stream::write::Encoder` consumes raw bytes and
            // emits compressed bytes downstream — those compressed bytes
            // pass through `tee`, so they're both written to disk AND
            // hashed into the MAC in one pass.
            let mut encoder = zstd::stream::write::Encoder::new(&mut tee, DEFAULT_ZSTD_LEVEL)
                .map_err(|e| {
                    warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed");
                    ArtifactError::Io
                })?;
            encoder.write_all(payload).map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "zstd write failed");
                ArtifactError::Io
            })?;
            // `finish()` flushes the zstd footer through the tee. After
            // this point the MAC has consumed exactly `magic || version
            // || content_hash || zstd_body` — byte-identical to what
            // the old buffered path used to hash.
            encoder.finish().map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "zstd finish failed");
                ArtifactError::Io
            })?;

            // Drop the BufWriter explicitly via the tee so its 64 KiB
            // slab is flushed BEFORE we append the HMAC tag below.
            // `BufWriter::drop` swallows flush errors, so call `flush`
            // explicitly first to surface any deferred write failure.
            tee.flush().map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "buf flush failed");
                ArtifactError::Io
            })?;
            // `tee` (and the BufWriter inside it) goes out of scope here,
            // releasing the `&mut File` borrow so we can write the tag
            // directly to `tmp.as_file_mut()` below.
        }

        // Finalise the MAC over `header || zstd_body` and append the
        // 32-byte tag. The tag is NOT fed back into the MAC (and indeed
        // bypasses the `MacWriter`); it's the trailer the `get` reader
        // strips before recomputing.
        let tag = finalize_into_tag(mac);
        tmp.as_file_mut().write_all(&tag).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "hmac tag write failed");
            ArtifactError::Io
        })?;

        // Atomic publish: temp-then-rename in the same directory,
        // mirroring the JIT L2 disk-cache pattern so a partial write
        // can never leave a half-formed entry that a concurrent reader
        // trips over.
        tmp.persist(&final_path).map_err(|e| {
            // `tempfile::PersistError` wraps the underlying `io::Error`
            // plus the temp handle; the `Display` impl forwards to the
            // io error so we don't need to reach for the field.
            warn!(target: "tensor_wasm_artifacts", error = %e, "tempfile persist failed");
            ArtifactError::Io
        })?;
        Ok(hash)
    }

    fn get(&self, hash: &ContentHash) -> Result<Vec<u8>, ArtifactError> {
        let path = self.path_for(hash);

        // T22 streaming read: open the file behind a 64 KiB BufReader
        // and stream the prefix (header + zstd body) through a
        // `MacReader` -> `zstd::Decoder` chain. The HMAC is built up as
        // bytes flow into the decoder, so we never materialise the
        // whole compressed body in RAM. Only the decoded payload is
        // buffered (capped at MAX_DECOMPRESSED_LEN), and even that is
        // only released to the caller AFTER the trailing tag verifies.
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ArtifactError::NotFound(hash.to_string()));
            }
            Err(e) => {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %path.display(),
                    error = %e,
                    "open failed"
                );
                return Err(ArtifactError::Io);
            }
        };
        let file_len = file
            .metadata()
            .map_err(|e| {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %path.display(),
                    error = %e,
                    "metadata failed"
                );
                ArtifactError::Io
            })?
            .len();

        // Minimum-length gate: header + at least one byte of zstd frame + HMAC.
        // Mirrors the old in-memory check; rejected here before we
        // start any keyed work or decoder setup.
        let min_len = (ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN) as u64;
        if file_len < min_len {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                len = file_len,
                "artifact too short for header+hmac"
            );
            return Err(ArtifactError::BadMagic);
        }

        // Compute the byte ranges in the file:
        //   [0 .. ARTIFACT_HEADER_LEN)                — header
        //   [ARTIFACT_HEADER_LEN .. prefix_end)        — zstd body
        //   [prefix_end .. file_len)                   — HMAC tag (32 B)
        //
        // `prefix_end` is what the MAC must cover.
        let prefix_end = file_len - ARTIFACT_HMAC_LEN as u64;
        let body_len = prefix_end - ARTIFACT_HEADER_LEN as u64;

        let mut reader = BufReader::with_capacity(STREAM_BUF_LEN, file);
        let mut mac = new_mac(&self.hmac_key);

        // ---- Read and validate the fixed header. ----
        let mut header = [0u8; ARTIFACT_HEADER_LEN];
        reader.read_exact(&mut header).map_err(|e| {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                error = %e,
                "header read failed"
            );
            ArtifactError::Io
        })?;
        if header[..16] != ARTIFACT_MAGIC {
            return Err(ArtifactError::BadMagic);
        }
        let version = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
        if version != ARTIFACT_VERSION {
            return Err(ArtifactError::BadVersion(version));
        }
        let mut hash_on_disk = [0u8; 32];
        hash_on_disk.copy_from_slice(&header[20..52]);
        // Feed the header into the MAC — same prefix the writer hashed.
        {
            use hmac::Mac;
            mac.update(&header);
        }

        // ---- Stream the zstd body through MacReader -> Decoder. ----
        //
        // `Read::take(body_len)` clips the source to exactly the zstd
        // body, so the decoder cannot accidentally read into the
        // trailing HMAC tag. The `MacReader` tees those same bytes into
        // the running HMAC, so by the time the decoder returns EOF we
        // have the MAC for the full prefix.
        //
        // The decoder output goes into another `Take(cap + 1)` so a
        // zstd-bomb cannot blow past `MAX_DECOMPRESSED_LEN`. Same
        // probe-by-one shape T10 uses on the snapshot reader.
        let cap = MAX_DECOMPRESSED_LEN;
        let probe_limit = u64::try_from(cap)
            .ok()
            .and_then(|c| c.checked_add(1))
            .unwrap_or(u64::MAX);
        // Scope the decoder/MacReader chain so it drops (releasing the
        // mutable borrows on `reader` and `mac`) before we drain any
        // residual body bytes and read the HMAC tag.
        //
        // Decoder failures are deferred: a tampered body byte will
        // typically make zstd return a frame-format error, but the
        // pre-existing contract is that an unauthenticated artifact
        // returns `BadHmac` — not `Decompression`. So we capture the
        // decode result here, drain the rest of the body through the
        // MAC so the running tag stays byte-aligned with the writer,
        // verify the HMAC, and only THEN surface the decode error.
        // That preserves the "BadHmac wins over Decompression on
        // tampered input" invariant the tamper-rejection tests assert.
        // Size the initial allocation from the compressed body length
        // (a 4:1 decompression estimate) clamped to the cap, rather than
        // a fixed 1 MiB regardless of payload size.
        let initial_capacity = usize::try_from(body_len)
            .unwrap_or(cap)
            .saturating_mul(4)
            .min(cap);
        let mut payload: Vec<u8> = Vec::with_capacity(initial_capacity);
        let mut decode_result: Result<(), ArtifactError> = Ok(());
        {
            let body_take = Read::take(&mut reader, body_len);
            let mac_reader = MacReader::new(body_take, &mut mac);
            match zstd::stream::read::Decoder::new(mac_reader) {
                Ok(decoder) => {
                    if let Err(e) = decoder.take(probe_limit).read_to_end(&mut payload) {
                        warn!(target: "tensor_wasm_artifacts", error = %e, "zstd decode failed");
                        decode_result = Err(ArtifactError::Decompression(e.to_string()));
                    }
                }
                Err(e) => {
                    warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed");
                    decode_result = Err(ArtifactError::Decompression(e.to_string()));
                }
            }
            // `decoder` (if it existed) was consumed by `.take(...)`,
            // which was a temporary consumed by `read_to_end`. The
            // MacReader/Take chain is gone now; the `&mut reader` and
            // `&mut mac` borrows are released at the end of this block.
        }
        // Whatever happened, the decoder/MacReader/Take chain is now
        // dropped — `mac` and `reader` are reborrowable below.

        // ---- Drain any residual body bytes the decoder skipped. ----
        //
        // For a well-formed zstd frame the decoder consumes every body
        // byte (frames are self-delimiting and end at the body's
        // boundary, which we enforce via the `body_take` adapter). But
        // when decoding aborts early (tampered frame, truncated input)
        // the BufReader may sit somewhere inside the body, leaving the
        // HMAC's running state short of what the writer hashed.
        //
        // Drain whatever's left in the body region through a fresh
        // MacReader so the MAC sees the FULL body bytes — the same
        // prefix the writer's MAC covered. This is what lets a
        // decompression-failure-on-tamper still surface as `BadHmac`
        // instead of `Decompression`, matching the old buffered code's
        // failure-mode ordering.
        use std::io::Seek;
        let consumed = reader.stream_position().map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "stream_position failed");
            ArtifactError::Io
        })?;
        if consumed < prefix_end {
            let gap = prefix_end - consumed;
            let drain_take = Read::take(&mut reader, gap);
            let mut drain_mac = MacReader::new(drain_take, &mut mac);
            // Discard the bytes — we only care about feeding the MAC.
            let mut scratch = [0u8; STREAM_BUF_LEN];
            loop {
                let n = drain_mac.read(&mut scratch).map_err(|e| {
                    warn!(target: "tensor_wasm_artifacts", error = %e, "tail drain failed");
                    ArtifactError::Io
                })?;
                if n == 0 {
                    break;
                }
            }
        } else if consumed > prefix_end {
            // Decoder over-read past the body's bound (shouldn't happen
            // because `body_take` caps it). Reposition so the next
            // read_exact pulls the tag; the MAC has over-counted and
            // will trip BadHmac below, which is the safe failure mode.
            reader
                .seek(std::io::SeekFrom::Start(prefix_end))
                .map_err(|e| {
                    warn!(target: "tensor_wasm_artifacts", error = %e, "seek to tag failed");
                    ArtifactError::Io
                })?;
        }

        let mut tag_bytes = [0u8; ARTIFACT_HMAC_LEN];
        reader.read_exact(&mut tag_bytes).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "tag read failed");
            ArtifactError::Io
        })?;

        let expected = finalize_into_tag(mac);
        use subtle::ConstantTimeEq;
        // `expected` is `[u8; 32]`; `ConstantTimeEq` is implemented on
        // `[u8]`. Slice-vs-slice keeps the comparison constant-time.
        let mac_ok = bool::from(expected.as_slice().ct_eq(&tag_bytes[..]));
        if !mac_ok {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                "HMAC mismatch (possible tampering or stale key)"
            );
            // CRITICAL: drop the decoded payload WITHOUT returning it —
            // a failed MAC means we have not authenticated the bytes
            // we just decompressed, and the existing invariant is
            // "HMAC verified BEFORE any decoded bytes are exposed to
            // callers". Returning here (with `payload` going out of
            // scope) preserves that invariant. BadHmac is also the
            // right answer when decode failed on a tampered body — the
            // old buffered path always returned BadHmac before reaching
            // the decompressor.
            return Err(ArtifactError::BadHmac);
        }

        // MAC verified. Now surface any deferred decode error — a
        // legitimate decoder failure on an UNTAMPERED body (e.g. a zstd
        // version mismatch in a future migration) still reports as
        // `Decompression`, same as the old path.
        decode_result?;

        // Decompressed-size cap check happens AFTER MAC verification so
        // a tampered payload can't make us return `TooLarge` before
        // `BadHmac`. The `Take(probe_limit)` adapter already prevented
        // any allocation past `cap + 1` bytes regardless.
        if payload.len() > cap {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                actual = payload.len(),
                limit = cap,
                "rejecting oversized decompressed payload (possible zstd bomb)"
            );
            return Err(ArtifactError::TooLarge {
                actual: payload.len(),
                limit: cap,
            });
        }

        // Defence-in-depth: recompute the content hash from the
        // decoded payload and compare to both the requested key AND the
        // header value. A header-vs-payload mismatch would mean a
        // valid-HMAC blob was somehow constructed under a wrong content
        // hash; that's impossible with a non-leaked key, but cheap to
        // check.
        let recomputed = ContentHash::of(&payload);
        if recomputed.0 != hash_on_disk {
            return Err(ArtifactError::HashMismatch {
                expected: hex_of(&hash_on_disk),
                actual: recomputed.to_string(),
            });
        }
        if recomputed != *hash {
            return Err(ArtifactError::HashMismatch {
                expected: hash.to_string(),
                actual: recomputed.to_string(),
            });
        }

        Ok(payload)
    }

    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError> {
        let suffix = format!(".{}.bin", self.key_fp_hex);
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            // A missing store directory legitimately means "empty" (the
            // dir is created lazily on the first `put`). Any other
            // failure — permissions, I/O — is propagated so GC/audit
            // callers don't mistake a read fault for an empty store.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                warn!(
                    target: "tensor_wasm_artifacts",
                    dir = %self.dir.display(),
                    error = %e,
                    "read_dir failed"
                );
                return Err(ArtifactError::Io);
            }
        };
        let mut out = Vec::new();
        for entry in entries {
            // A per-entry error (e.g. the directory was racing with a
            // concurrent unlink, or an underlying I/O fault) is a real
            // enumeration failure, not a skippable filename mismatch —
            // propagate it rather than silently shortening the listing.
            let entry = entry.map_err(|e| {
                warn!(
                    target: "tensor_wasm_artifacts",
                    dir = %self.dir.display(),
                    error = %e,
                    "read_dir entry failed"
                );
                ArtifactError::Io
            })?;
            let name = match entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            // Filenames look like `{64 hex chars}.{16 hex chars}.bin`.
            // Match the suffix so we ignore files written by stores
            // under a different key.
            if !name.ends_with(&suffix) {
                continue;
            }
            let hash_hex = &name[..name.len() - suffix.len()];
            if hash_hex.len() != 64 {
                continue;
            }
            let mut bytes = [0u8; 32];
            let mut ok = true;
            for (i, chunk) in hash_hex.as_bytes().chunks(2).enumerate() {
                let s = match std::str::from_utf8(chunk) {
                    Ok(s) => s,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                };
                match u8::from_str_radix(s, 16) {
                    Ok(b) => bytes[i] = b,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                out.push(ContentHash::from_bytes(bytes));
            }
        }
        Ok(out)
    }
}

fn hex_of(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// =====================================================================
// In-memory envelope encode/decode (T40 — snapshot v0.4 default flip)
// =====================================================================
//
// The streaming disk-store paths above own the canonical encode/decode
// loop, but the snapshot crate's default `SnapshotWriter::capture` /
// `SnapshotReader::restore` deal in `Vec<u8>` (not a `&DiskArtifactStore`),
// so they need a pure-bytes door into the same envelope. These two
// helpers expose exactly that: a `Vec<u8>`-in / `Vec<u8>`-out pair that
// produces and consumes bytes byte-identical to what `DiskArtifactStore`
// would write to disk.
//
// They are intentionally NOT `pub` on the `ArtifactStore` trait — the
// trait abstracts over storage *backends*; these helpers are a framing
// utility. Callers who want a persistent store should still go through
// `DiskArtifactStore::put` / `get`; callers who need the framed bytes
// in memory (e.g. to bundle inside another envelope, or to attach to
// an HTTP body) use these.

/// Encode `payload` into the unified artifact-store envelope (v0.4
/// snapshot default).
///
/// Returns `Vec<u8>` containing the byte sequence
/// `ARTIFACT_MAGIC || ARTIFACT_VERSION || blake3(payload) || zstd(payload) || hmac_sha256(prefix)` —
/// the same bytes [`DiskArtifactStore::put`] would write to its tempfile
/// before atomic-rename. Useful for callers that need the framed envelope
/// in memory (the snapshot crate's default `capture` path is the
/// motivating consumer).
///
/// The HMAC covers `magic || version || content_hash || zstd(payload)`;
/// the trailing 32-byte tag is appended after. Verification is the
/// counterpart [`decode_envelope_from_bytes`], which recomputes the MAC
/// in constant time and rejects on mismatch before any decoded bytes
/// are exposed to the caller.
///
/// Errors are reported via [`ArtifactError`] to keep the error surface
/// homogeneous with the disk store; the only failure modes are
/// `TooLarge` (when `payload.len() > MAX_PAYLOAD_LEN`) and `Io` (when
/// the in-memory zstd encoder reports an internal error).
pub fn encode_envelope_to_vec(
    payload: &[u8],
    hmac_key: &[u8; 32],
) -> Result<Vec<u8>, ArtifactError> {
    if payload.len() > MAX_PAYLOAD_LEN {
        warn!(
            target: "tensor_wasm_artifacts",
            actual = payload.len(),
            limit = MAX_PAYLOAD_LEN,
            "rejecting oversized payload (encode_envelope_to_vec)"
        );
        return Err(ArtifactError::TooLarge {
            actual: payload.len(),
            limit: MAX_PAYLOAD_LEN,
        });
    }

    let hash = ContentHash::of(payload);
    let mut mac = new_mac(hmac_key);

    // Pre-size the framing buffer conservatively: header + a quarter of
    // the payload (zstd typically compresses better than 4:1 on tensor
    // memory, so this overshoots harmlessly for small inputs and
    // undershoots only marginally on incompressible blobs) + the HMAC
    // trailer. The Vec grows on demand if the estimate is too small.
    let mut buf: Vec<u8> =
        Vec::with_capacity(ARTIFACT_HEADER_LEN + payload.len() / 4 + ARTIFACT_HMAC_LEN);

    // Header: 16-byte magic + 4-byte version + 32-byte content hash.
    buf.extend_from_slice(&ARTIFACT_MAGIC);
    buf.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    buf.extend_from_slice(&hash.0);

    // Stream zstd into the buffer through a MacWriter so the HMAC sees
    // exactly the bytes we write. The MAC has already consumed the
    // header by way of the explicit `mac.update`s above? No — the
    // header was extended into `buf` directly (no tee). Feed it to
    // the MAC explicitly here so the prefix the MAC covers matches the
    // disk-store layout byte-for-byte (`header || zstd_body`).
    {
        use hmac::Mac;
        mac.update(&buf[..ARTIFACT_HEADER_LEN]);
    }

    // Compress the payload into the buffer; tee through MacWriter so the
    // MAC also consumes the compressed bytes. The MacWriter takes the
    // buffer by mutable reference (Vec<u8> implements Write via
    // `extend_from_slice`-flavoured semantics), so the resulting bytes
    // continue to land in `buf` while the MAC observes the same
    // sequence the disk-store writer would see.
    {
        let mut tee = MacWriter::new(&mut buf, &mut mac);
        let mut encoder = zstd::stream::write::Encoder::new(&mut tee, DEFAULT_ZSTD_LEVEL)
            .map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed (encode_envelope_to_vec)");
                ArtifactError::Io
            })?;
        encoder.write_all(payload).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd write failed (encode_envelope_to_vec)");
            ArtifactError::Io
        })?;
        encoder.finish().map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd finish failed (encode_envelope_to_vec)");
            ArtifactError::Io
        })?;
    }

    // Append the HMAC tag. The tag is NOT fed back into the MAC.
    let tag = finalize_into_tag(mac);
    buf.extend_from_slice(&tag);
    Ok(buf)
}

/// Decode the unified artifact-store envelope from `bytes`, returning
/// the inner payload after verifying the HMAC in constant time and the
/// BLAKE3 content hash as defence-in-depth.
///
/// Counterpart to [`encode_envelope_to_vec`]. Used by the snapshot
/// crate's default `SnapshotReader::restore` to detect and consume the
/// v0.4 envelope before falling through to the legacy v3 / v2 readers.
/// Returns [`ArtifactError::BadMagic`] if the leading 16 bytes do not
/// match [`ARTIFACT_MAGIC`] — callers rely on that variant to know they
/// should try a different envelope shape, so the magic check is the
/// first thing this function does (cheap, allocation-free).
///
/// Validation order:
/// 1. Minimum length, then magic and version (cheap, before any keyed
///    work).
/// 2. HMAC verification in constant time over `magic || version ||
///    content_hash || zstd_body`. Failure returns [`ArtifactError::BadHmac`]
///    without touching the decoded payload.
/// 3. zstd decompression with a [`MAX_DECOMPRESSED_LEN`] cap, mirroring
///    the disk store's zip-bomb defence.
/// 4. Recompute the content hash and compare to the header value —
///    catches a writer bug that hashed the wrong bytes even if the
///    HMAC verified (impossible without key leak, but cheap to check).
pub fn decode_envelope_from_bytes(
    bytes: &[u8],
    hmac_key: &[u8; 32],
) -> Result<Vec<u8>, ArtifactError> {
    decode_envelope_from_bytes_with_cap(bytes, hmac_key, MAX_DECOMPRESSED_LEN)
}

/// Decode the unified artifact-store envelope from `bytes`, like
/// [`decode_envelope_from_bytes`], but with a caller-supplied
/// `max_decompressed` ceiling instead of the crate-wide
/// [`MAX_DECOMPRESSED_LEN`].
///
/// This exists so a consumer with its own decompressed-size budget — the
/// snapshot reader's `with_max_decompressed` knob is the motivating
/// case — can enforce a tighter (or looser) zip-bomb cap than the
/// default without round-tripping through the disk store. `decode_envelope_from_bytes`
/// is a thin wrapper that calls this with `max_decompressed =
/// MAX_DECOMPRESSED_LEN`.
///
/// Behaviour, validation order, and error surface are otherwise
/// identical to [`decode_envelope_from_bytes`]: the only difference is
/// which ceiling drives the `Take` probe and the post-decode
/// [`ArtifactError::TooLarge`] check.
pub fn decode_envelope_from_bytes_with_cap(
    bytes: &[u8],
    hmac_key: &[u8; 32],
    max_decompressed: usize,
) -> Result<Vec<u8>, ArtifactError> {
    // Minimum-length gate: header + at least one byte of zstd frame + HMAC tag.
    if bytes.len() < ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN {
        return Err(ArtifactError::BadMagic);
    }
    // Magic check first — cheap rejection for foreign envelopes (the
    // snapshot reader uses this branch to fall through to v3 / v2).
    if bytes[..16] != ARTIFACT_MAGIC {
        return Err(ArtifactError::BadMagic);
    }
    let version = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    if version != ARTIFACT_VERSION {
        return Err(ArtifactError::BadVersion(version));
    }
    let mut hash_on_disk = [0u8; 32];
    hash_on_disk.copy_from_slice(&bytes[20..52]);

    // HMAC verification: the tag is the trailing 32 bytes, the MAC
    // covers everything before that (header + zstd body).
    let hmac_start = bytes.len() - ARTIFACT_HMAC_LEN;
    let (prefix, tag_bytes) = bytes.split_at(hmac_start);
    let mut mac = new_mac(hmac_key);
    {
        use hmac::Mac;
        mac.update(prefix);
    }
    let expected = finalize_into_tag(mac);
    use subtle::ConstantTimeEq;
    let mac_ok = bool::from(expected.as_slice().ct_eq(tag_bytes));
    if !mac_ok {
        warn!(
            target: "tensor_wasm_artifacts",
            "HMAC mismatch (decode_envelope_from_bytes; possible tampering or stale key)"
        );
        return Err(ArtifactError::BadHmac);
    }

    // Decompress the body that sits between the header and the HMAC tag.
    // Use a `Read::take` probe one byte past the cap so a zstd-bomb is
    // rejected before the buffer grows past `MAX_DECOMPRESSED_LEN`,
    // matching the disk-store streaming reader.
    let body = &prefix[ARTIFACT_HEADER_LEN..];
    let cap = max_decompressed;
    let probe_limit = u64::try_from(cap)
        .ok()
        .and_then(|c| c.checked_add(1))
        .unwrap_or(u64::MAX);
    // Size the initial allocation from the compressed body length rather
    // than a fixed 1 MiB: a 4:1 decompression estimate covers typical
    // tensor-memory payloads, clamped to the cap so a tiny envelope
    // never reserves more than its ceiling allows.
    let initial_capacity = body.len().saturating_mul(4).min(cap);
    let mut payload: Vec<u8> = Vec::with_capacity(initial_capacity);
    let decoder = zstd::stream::read::Decoder::new(body).map_err(|e| {
        warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed (decode_envelope_from_bytes)");
        ArtifactError::Decompression(e.to_string())
    })?;
    decoder
        .take(probe_limit)
        .read_to_end(&mut payload)
        .map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd decode failed (decode_envelope_from_bytes)");
            ArtifactError::Decompression(e.to_string())
        })?;
    if payload.len() > cap {
        return Err(ArtifactError::TooLarge {
            actual: payload.len(),
            limit: cap,
        });
    }

    // Defence-in-depth: recompute and compare. A header-vs-payload
    // mismatch would mean a valid-HMAC blob was constructed under a
    // wrong content hash (impossible without key leak, but cheap).
    let recomputed = ContentHash::of(&payload);
    if recomputed.0 != hash_on_disk {
        return Err(ArtifactError::HashMismatch {
            expected: hex_of(&hash_on_disk),
            actual: recomputed.to_string(),
        });
    }

    Ok(payload)
}

// =====================================================================
// In-memory store
// =====================================================================

/// In-memory artifact store. Intended for tests, fuzzers, and ephemeral
/// caches. The HMAC key is still held (and zeroized on drop) so
/// behaviour matches the disk store, but no signing is exercised — the
/// in-memory map already guarantees integrity.
pub struct InMemoryArtifactStore {
    entries: parking_lot::Mutex<HashMap<ContentHash, Vec<u8>>>,
    // Held for parity with `DiskArtifactStore::hmac_key`; not read on
    // any hot path. Future migrations (snapshot replay-protection
    // matrix, signed-kernel-registry) may want to surface the key
    // fingerprint from an in-memory store too.
    #[allow(dead_code)]
    hmac_key: Zeroizing<[u8; 32]>,
}

impl InMemoryArtifactStore {
    /// Construct an empty in-memory store under `hmac_key`.
    pub fn new(hmac_key: [u8; 32]) -> Self {
        Self {
            entries: parking_lot::Mutex::new(HashMap::new()),
            hmac_key: Zeroizing::new(hmac_key),
        }
    }
}

impl ArtifactStore for InMemoryArtifactStore {
    fn put(&self, payload: &[u8]) -> Result<ContentHash, ArtifactError> {
        let hash = ContentHash::of(payload);
        self.entries.lock().insert(hash, payload.to_vec());
        Ok(hash)
    }
    fn get(&self, hash: &ContentHash) -> Result<Vec<u8>, ArtifactError> {
        self.entries
            .lock()
            .get(hash)
            .cloned()
            .ok_or_else(|| ArtifactError::NotFound(hash.to_string()))
    }
    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError> {
        // An in-memory map cannot fail to enumerate, so this is
        // infallible — but the signature matches the trait so callers
        // treat both stores uniformly.
        Ok(self.entries.lock().keys().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_blake3() {
        let h = ContentHash::of(b"hello");
        let expected: [u8; 32] = blake3::hash(b"hello").into();
        assert_eq!(h.0, expected);
        assert_eq!(h.to_hex().len(), 64);
    }

    #[test]
    fn in_memory_put_get_round_trip() {
        let store = InMemoryArtifactStore::new([7u8; 32]);
        let hash = store.put(b"payload").unwrap();
        assert_eq!(store.get(&hash).unwrap(), b"payload");
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn key_fingerprint_is_8_bytes_hex() {
        let fp = key_fingerprint_hex(&[0u8; 32]);
        assert_eq!(fp.len(), 16); // 8 bytes -> 16 hex chars
    }

    #[test]
    fn disk_format_constants() {
        assert_eq!(ARTIFACT_HEADER_LEN, 16 + 4 + 32);
        assert_eq!(ARTIFACT_HMAC_LEN, 32);
        assert_eq!(ARTIFACT_MAGIC.len(), 16);
    }
}
