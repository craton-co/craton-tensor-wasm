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
use std::sync::Arc;

use serde::{Deserialize, Serialize};
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
/// without ever allocating past the cap.
///
/// SECURITY (memory-amplification fix): this MUST stay tied to
/// [`MAX_PAYLOAD_LEN`]. Every blob the store can hold was put-capped at
/// `MAX_PAYLOAD_LEN` *uncompressed* bytes, so a legitimate decode never
/// needs to yield more than that. Sizing the read cap larger (the old
/// 1 GiB = 4x value) handed an attacker holding a valid/leaked key a
/// 4x memory-amplification primitive: a blob that put refused at
/// 256 MiB could still be hand-crafted to decompress to ~1 GiB on the
/// read path. We therefore pin the read cap to the put cap plus a small
/// fixed framing slack (a few KiB) so no legitimately-`put` blob — which
/// is at most `MAX_PAYLOAD_LEN` uncompressed — is ever rejected, while
/// the read path can no longer be coerced into a larger allocation than
/// the write path would have admitted.
pub const MAX_DECOMPRESSED_LEN: usize = MAX_PAYLOAD_LEN + 8 * 1024;

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
    /// (De)serialisation of an [`ArtifactMetadata`] sidecar failed. Carries
    /// the serde error rendered to a string so the variant stays
    /// `PartialEq`-compatible with the rest of [`ArtifactError`] (which
    /// deliberately avoids embedding non-`Eq` inner error types). Only
    /// reachable through the metadata-aware paths
    /// ([`DiskArtifactStore::put_with_metadata`] /
    /// [`DiskArtifactStore::metadata`]); the plain `put`/`get` round-trip
    /// never touches a sidecar.
    #[error("artifact metadata codec error: {0}")]
    Metadata(String),
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
    /// Stream the verified-then-decoded body of `hash` into `out`,
    /// returning the number of bytes written. This completes the
    /// streaming story [`Self::put`] already has on the write side: where
    /// [`Self::get`] returns an owned `Vec<u8>` (up to
    /// [`MAX_DECOMPRESSED_LEN`] resident),
    /// `get_to` lets a caller pipe a large artifact straight into a file,
    /// socket, or hashing sink without first materialising the whole
    /// decoded payload as a return value.
    ///
    /// ## Integrity ordering
    ///
    /// The HMAC covers the *entire* compressed blob, so it cannot be
    /// verified until the last body byte has been seen. To preserve the
    /// crate-wide invariant — **no unverified bytes are ever exposed to a
    /// caller** — implementations MUST authenticate the blob *before*
    /// any decoded byte reaches `out`. [`DiskArtifactStore`] does this
    /// with a two-pass scheme (pass 1: stream the compressed body through
    /// the HMAC to verify; pass 2: re-open and stream-decode directly to
    /// `out`), so peak heap stays bounded by the I/O buffers regardless
    /// of payload size and `out` only ever sees authenticated bytes. The
    /// error variants match [`Self::get`].
    ///
    /// The default implementation delegates to [`Self::get`] and copies
    /// the resulting buffer into `out`; it is correct (the `Vec` `get`
    /// returns is already verified) but not memory-bounded. Backends that
    /// can stream override it.
    fn get_to(&self, hash: &ContentHash, out: &mut dyn Write) -> Result<u64, ArtifactError> {
        let bytes = self.get(hash)?;
        out.write_all(&bytes).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "get_to default writer write failed");
            ArtifactError::Io
        })?;
        Ok(bytes.len() as u64)
    }
    /// Enumerate the content hashes currently stored. Order is
    /// implementation-defined; callers that need a deterministic order
    /// must sort the result themselves.
    ///
    /// Returns [`ArtifactError::Io`] on an underlying enumeration
    /// failure (e.g. a `read_dir` permissions/I/O fault). This is
    /// deliberately distinct from an empty result so GC/audit callers
    /// can tell "store is empty" apart from "could not read the store".
    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError>;
    /// Cheap existence probe for `hash`: returns `true` if an entry is
    /// present, `false` otherwise. This is deliberately a stat-only
    /// check — it does NOT decode, decompress, or HMAC-verify the
    /// record, so it is dramatically cheaper than [`Self::get`] and
    /// suitable for the GC/audit roadmap's "is this content already
    /// resident?" question.
    ///
    /// Because no integrity work happens, a `true` result means only
    /// that a file (disk) or map entry (in-memory) exists under the
    /// content-addressed key; a subsequent [`Self::get`] can still fail
    /// with an integrity variant if that record was tampered with.
    ///
    /// For [`DiskArtifactStore`] this probe is rotation-aware: it resolves
    /// against every accepted read key (active or retired), so a blob
    /// written under a now-retired key — and still visible to [`Self::get`]
    /// — also reports `true` here, rather than appearing absent and leaking
    /// past GC.
    ///
    /// Returns [`ArtifactError::Io`] only on an underlying probe fault
    /// that is neither "present" nor "absent" (e.g. a `metadata` call
    /// failing for a reason other than not-found).
    fn contains(&self, hash: &ContentHash) -> Result<bool, ArtifactError>;
    /// Remove the entry stored under `hash`. Returns `true` if a record
    /// was removed, `false` if nothing was stored under `hash` (a
    /// no-op delete is not an error — it mirrors `HashMap::remove`'s
    /// "was it there?" boolean and the POSIX `unlink`-of-missing
    /// convention GC callers expect).
    ///
    /// For [`DiskArtifactStore`] this is rotation-aware: it resolves which
    /// accepted key (active or retired) actually holds the blob — the same
    /// resolution [`Self::get`] / [`Self::list`] use — and unlinks the
    /// `{hash}.{key_fp}.bin` file (plus any sidecar) under THAT key, so a
    /// blob written under a now-retired key can still be GC'd. For
    /// [`InMemoryArtifactStore`] it drops the map entry. Returns
    /// [`ArtifactError::Io`] on an underlying delete fault other than
    /// not-found.
    fn remove(&self, hash: &ContentHash) -> Result<bool, ArtifactError>;
}

// =====================================================================
// Key provider — pluggable signing key + accepted-for-read key set
// =====================================================================

/// Supplies the signing material a [`DiskArtifactStore`] uses, abstracting
/// over key rotation.
///
/// A store always *writes* under a single active key (so a fresh `put`
/// lands under one fingerprint and round-trips cleanly), but during a
/// rotation it must still be able to *read* blobs that were written under
/// an older, now-retired key. A `KeyProvider` therefore exposes two
/// things:
///
/// * [`Self::active_key`] — the one key new `put`s sign with, and the key
///   `list` reports the store's "own" namespace under; and
/// * [`Self::read_keys`] — every key a `get` / `list` is allowed to try,
///   newest first. The active key MUST appear in this set.
///
/// On a `get`, the store tries each read key's namespaced file in turn
/// (keys partition the on-disk namespace by fingerprint, so at most one
/// can match a given content hash). On a `list`, the store unions the
/// hashes visible under every read key, so an audit sees blobs across the
/// rotation boundary.
///
/// Implementations MUST be `Send + Sync` (the store is shared behind an
/// `Arc`). Keys are returned by value (`[u8; 32]`) rather than borrowed so
/// a provider backing onto a rotating KMS can mint them on demand.
pub trait KeyProvider: Send + Sync {
    /// The key new `put`s sign with. MUST be one of [`Self::read_keys`].
    fn active_key(&self) -> [u8; 32];
    /// Every key a read (`get` / `list`) may try, conventionally newest
    /// first so the common "just-written under the active key" case hits
    /// on the first probe. The active key MUST be present.
    fn read_keys(&self) -> Vec<[u8; 32]>;
}

/// The default single-key provider: writes and reads under exactly one
/// key. This is what [`DiskArtifactStore::new`] wraps the caller's
/// `[u8; 32]` in, so the historical single-key constructor keeps its exact
/// behaviour (one active key, that same key the only accepted read key).
///
/// The key is held in a [`Zeroizing`] so it is scrubbed on drop, matching
/// the store's own `hmac_key` handling.
pub struct SingleKeyProvider {
    key: Zeroizing<[u8; 32]>,
}

impl SingleKeyProvider {
    /// Wrap a single key as a provider.
    pub fn new(key: [u8; 32]) -> Self {
        Self {
            key: Zeroizing::new(key),
        }
    }
}

impl KeyProvider for SingleKeyProvider {
    fn active_key(&self) -> [u8; 32] {
        *self.key
    }
    fn read_keys(&self) -> Vec<[u8; 32]> {
        vec![*self.key]
    }
}

/// A rotation-aware key provider: one active (write) key plus an ordered
/// list of additional keys accepted for reads only.
///
/// During a rotation an operator constructs this with the new key as
/// `active` and the previous key(s) in `also_accept`; new `put`s sign
/// under `active`, while `get` / `list` continue to see blobs written
/// under the retired keys until they are migrated or GC'd. The active key
/// is automatically included in [`KeyProvider::read_keys`] (first), so the
/// caller passes only the *extra* accepted keys.
pub struct RotatingKeyProvider {
    active: Zeroizing<[u8; 32]>,
    also_accept: Vec<Zeroizing<[u8; 32]>>,
}

impl RotatingKeyProvider {
    /// Build a provider whose write/active key is `active` and which also
    /// accepts every key in `also_accept` for reads. Duplicates of the
    /// active key in `also_accept` are harmless (the store dedups by
    /// fingerprint when listing).
    pub fn new(active: [u8; 32], also_accept: impl IntoIterator<Item = [u8; 32]>) -> Self {
        Self {
            active: Zeroizing::new(active),
            also_accept: also_accept.into_iter().map(Zeroizing::new).collect(),
        }
    }
}

impl KeyProvider for RotatingKeyProvider {
    fn active_key(&self) -> [u8; 32] {
        *self.active
    }
    fn read_keys(&self) -> Vec<[u8; 32]> {
        // Active key first (the hot path: reading something just written),
        // then the retired keys in the order supplied.
        let mut keys = Vec::with_capacity(1 + self.also_accept.len());
        keys.push(*self.active);
        keys.extend(self.also_accept.iter().map(|k| **k));
        keys
    }
}

// =====================================================================
// Artifact metadata sidecar
// =====================================================================

/// Optional per-artifact metadata, stored as a serde-encoded sidecar next
/// to the blob.
///
/// The blob itself is opaque content-addressed bytes; nothing in the
/// envelope records *when* or *from where* it was produced. For
/// `list`-based auditing and GC (roadmap feature #9) that provenance is
/// useful, so [`DiskArtifactStore::put_with_metadata`] writes one of these
/// alongside the blob and [`DiskArtifactStore::metadata`] reads it back.
///
/// Metadata is strictly optional: a plain [`ArtifactStore::put`] writes no
/// sidecar, and [`DiskArtifactStore::metadata`] returns
/// [`ArtifactError::NotFound`] for a blob that has none.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Wall-clock creation time in Unix milliseconds, as recorded by the
    /// writer at `put_with_metadata` time.
    pub created_unix_ms: u64,
    /// Length in bytes of the *uncompressed* payload. Lets an auditor size
    /// the store's logical footprint without decoding every blob.
    pub original_len: u64,
    /// Free-form origin tag (e.g. `"jit-l2"`, `"snapshot"`, `"import"`) so
    /// GC policy can treat artifacts from different producers differently.
    pub source_tier: String,
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
    /// Active signing key — the key new `put`s sign with. Cached out of the
    /// provider once at construction so the hot write path doesn't re-enter
    /// the provider (a rotating provider may mint keys on demand). Held in a
    /// `Zeroizing` so it's scrubbed on drop. Reads (`get` / `get_to`) and
    /// the rotation-aware probes (`contains` / `remove` / `metadata`)
    /// resolve against [`KeyProvider::read_keys`], not just this key.
    hmac_key: Zeroizing<[u8; 32]>,
    /// `blake3(active_key)[..8]` rendered as 16 ascii-hex chars, computed
    /// once at construction. Used as the active-key filename segment by
    /// `path_for` / `meta_path_for` (and thus `put` / `put_with_metadata`);
    /// caching it here avoids re-hashing the active key on every write. The
    /// rotation-aware read/probe paths derive per-key fingerprints on demand
    /// via `key_fingerprint_hex` instead.
    key_fp_hex: String,
    /// Pluggable source of signing keys. For a single-key store this is a
    /// [`SingleKeyProvider`]; for a rotating store it yields the active
    /// key plus the retired keys still accepted for reads. `get` and
    /// `list` consult [`KeyProvider::read_keys`] so they span a rotation.
    key_provider: Arc<dyn KeyProvider>,
}

impl DiskArtifactStore {
    /// Construct a disk store rooted at `dir`, signing with `hmac_key`.
    /// The directory is created lazily on the first `put`.
    ///
    /// This wraps `hmac_key` in a [`SingleKeyProvider`] internally, so the
    /// store writes and reads under exactly that one key — identical to
    /// the historical single-key behaviour. Use
    /// [`Self::with_key_provider`] for rotation support.
    pub fn new(dir: PathBuf, hmac_key: [u8; 32]) -> Self {
        Self::with_key_provider(dir, Arc::new(SingleKeyProvider::new(hmac_key)))
    }

    /// Construct a disk store rooted at `dir` whose signing keys come from
    /// `key_provider`.
    ///
    /// New `put`s sign under [`KeyProvider::active_key`]; `get` and `list`
    /// try every key in [`KeyProvider::read_keys`], so a store built with
    /// a [`RotatingKeyProvider`] can read blobs written under a retired
    /// key after rotation. The directory is created lazily on the first
    /// `put`.
    pub fn with_key_provider(dir: PathBuf, key_provider: Arc<dyn KeyProvider>) -> Self {
        let active = key_provider.active_key();
        let key_fp_hex = key_fingerprint_hex(&active);
        Self {
            dir,
            hmac_key: Zeroizing::new(active),
            key_fp_hex,
            key_provider,
        }
    }

    /// Compute the on-disk path for `hash` under this store's *active*
    /// key. Counterpart helper [`Self::path_for_key`] resolves under an
    /// arbitrary accepted read key.
    ///
    /// Filename format: `{content_hash_hex}.{key_fp_hex}.bin`. The key
    /// fingerprint segment partitions the namespace per HMAC key so
    /// two stores in the same dir under different keys never collide.
    fn path_for(&self, hash: &ContentHash) -> PathBuf {
        self.path_for_key(hash, &self.key_fp_hex)
    }

    /// Compute the on-disk path for `hash` under the key whose fingerprint
    /// is `key_fp_hex`. Used by the rotation-aware `get` to probe each
    /// accepted read key's namespace in turn.
    fn path_for_key(&self, hash: &ContentHash, key_fp_hex: &str) -> PathBuf {
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
        self.dir.join(format!("{hash_hex}.{key_fp_hex}.bin"))
    }

    /// On-disk path for the metadata sidecar of `hash` under the active
    /// key. Sits next to the blob with a `.meta.json` extension in place
    /// of `.bin`, so it shares the blob's content-addressed + key-
    /// partitioned naming and is filtered out of `list` (which only
    /// matches `.bin`). Counterpart helper [`Self::meta_path_for_key`]
    /// resolves the sidecar under an arbitrary accepted read key.
    fn meta_path_for(&self, hash: &ContentHash) -> PathBuf {
        self.meta_path_for_key(hash, &self.key_fp_hex)
    }

    /// On-disk path for the metadata sidecar of `hash` under the key whose
    /// fingerprint is `key_fp_hex`. Used by the rotation-aware `metadata`
    /// read so a sidecar written under a now-retired key is still found,
    /// and by `remove` so the sidecar is unlinked under the same key as the
    /// blob it describes.
    fn meta_path_for_key(&self, hash: &ContentHash, key_fp_hex: &str) -> PathBuf {
        let hash_hex = hash.to_string();
        self.dir.join(format!("{hash_hex}.{key_fp_hex}.meta.json"))
    }

    /// Insert `payload` (exactly like [`ArtifactStore::put`]) and write an
    /// [`ArtifactMetadata`] sidecar next to the blob under the active key.
    ///
    /// The blob and the sidecar are independent files: the blob is the
    /// content-addressed envelope, the sidecar is a `.meta.json` serde
    /// document. A later [`Self::metadata`] reads the sidecar back. Plain
    /// `put` continues to write no sidecar, so metadata stays optional.
    ///
    /// `metadata.original_len` is recorded as supplied by the caller (it
    /// is provenance, not a re-derived fact); pass `payload.len()` for the
    /// common "this is the blob I just stored" case.
    pub fn put_with_metadata(
        &self,
        payload: &[u8],
        metadata: &ArtifactMetadata,
    ) -> Result<ContentHash, ArtifactError> {
        // Store the blob first; if that fails we never write an orphan
        // sidecar. (A crash between the two writes can leave a blob with
        // no sidecar — that's the benign direction: `metadata` simply
        // reports `NotFound` and the blob still round-trips via `get`.)
        let hash = self.put(payload)?;

        let encoded = serde_json::to_vec(metadata).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata serialize failed");
            ArtifactError::Metadata(e.to_string())
        })?;

        // Atomic publish of the sidecar via temp-then-rename, mirroring
        // the blob's own publish so a torn write never leaves a partial
        // sidecar a concurrent reader trips over.
        let meta_path = self.meta_path_for(&hash);
        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata tempfile create failed");
            ArtifactError::Io
        })?;
        tmp.write_all(&encoded).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata write failed");
            ArtifactError::Io
        })?;
        // Durability (crash-consistency fix): flush the sidecar's data
        // before the atomic rename, mirroring the blob publish above.
        tmp.as_file().sync_all().map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata fsync (pre-persist) failed");
            ArtifactError::Io
        })?;
        tmp.persist(&meta_path).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata persist failed");
            ArtifactError::Io
        })?;
        // Durability: best-effort directory fsync to persist the rename
        // (tolerant of Windows, where directory sync is unsupported).
        self.sync_dir_best_effort();
        Ok(hash)
    }

    /// Read the [`ArtifactMetadata`] sidecar for `hash` under whichever
    /// accepted key (active or retired) the blob actually lives under.
    ///
    /// Rotation-aware: this resolves the blob's key with the same
    /// [`Self::resolve_read_key`] probe `get` / `list` use, then reads the
    /// sidecar partitioned under that key's fingerprint. A blob written
    /// under a now-retired key (and its sidecar) is therefore still
    /// readable here after rotation, matching what `get` exposes.
    ///
    /// Returns [`ArtifactError::NotFound`] if no blob exists under any
    /// accepted key, or if the blob exists but carries no sidecar (e.g. it
    /// was written via plain `put`). A malformed sidecar surfaces as
    /// [`ArtifactError::Metadata`].
    pub fn metadata(&self, hash: &ContentHash) -> Result<ArtifactMetadata, ArtifactError> {
        // Find which accepted key holds the blob, then read the sidecar
        // under that same key's fingerprint. If no blob is resolvable the
        // sidecar (if any) is orphaned/foreign — report NotFound.
        let (_blob_path, key) = match self.resolve_read_key(hash)? {
            Some(found) => found,
            None => return Err(ArtifactError::NotFound(hash.to_string())),
        };
        let fp = key_fingerprint_hex(&key);
        let meta_path = self.meta_path_for_key(hash, &fp);
        let bytes = match std::fs::read(&meta_path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ArtifactError::NotFound(hash.to_string()));
            }
            Err(e) => {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %meta_path.display(),
                    error = %e,
                    "metadata read failed"
                );
                return Err(ArtifactError::Io);
            }
        };
        serde_json::from_slice(&bytes).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "metadata deserialize failed");
            ArtifactError::Metadata(e.to_string())
        })
    }

    /// Pass 1 of the streaming `get_to`: verify the HMAC of the blob held
    /// open by `reader` under `key` *without* exposing any decoded bytes.
    /// Streams the compressed body through the running MAC and compares the
    /// trailing tag in constant time. On success returns the validated
    /// header's content hash and the byte range of the zstd body, so pass
    /// 2 can rewind the SAME handle and decode without re-reading the
    /// header.
    ///
    /// TOCTOU note: `get_to` opens the file exactly once and passes the
    /// resulting handle here for pass 1 and to [`Self::decode_to_writer`]
    /// for pass 2 (after a `seek` back to start). Because both passes read
    /// the one handle's bytes, the bytes decoded in pass 2 are *exactly*
    /// the bytes authenticated in pass 1 — a concurrent `remove`+`put` that
    /// swaps the path's contents between passes cannot smuggle unverified
    /// bytes to `out` (a swap unlinks the old inode but our open handle
    /// keeps reading it). `reader` is positioned at the start of the file
    /// on entry. The caller supplies `file_len` (already stat'd) so this
    /// does not re-`metadata` the handle.
    ///
    /// Returns the same integrity error surface as [`ArtifactStore::get`].
    fn verify_blob(
        &self,
        reader: &mut BufReader<File>,
        file_len: u64,
        key: &[u8; 32],
        path: &std::path::Path,
    ) -> Result<VerifiedBlob, ArtifactError> {
        let min_len = (ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN) as u64;
        if file_len < min_len {
            return Err(ArtifactError::BadMagic);
        }
        let prefix_end = file_len - ARTIFACT_HMAC_LEN as u64;
        let body_len = prefix_end - ARTIFACT_HEADER_LEN as u64;

        let mut mac = new_mac(key);

        let mut header = [0u8; ARTIFACT_HEADER_LEN];
        reader.read_exact(&mut header).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "header read failed");
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
        {
            use hmac::Mac;
            mac.update(&header);
        }

        // Stream the whole compressed body through the MAC, discarding the
        // bytes. We do NOT decode here: the goal of pass 1 is solely to
        // authenticate, so no decoded byte exists yet to leak.
        {
            let mut body_take = Read::take(&mut *reader, body_len);
            let mut scratch = [0u8; STREAM_BUF_LEN];
            loop {
                let n = body_take.read(&mut scratch).map_err(|e| {
                    warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "body read failed");
                    ArtifactError::Io
                })?;
                if n == 0 {
                    break;
                }
                use hmac::Mac;
                mac.update(&scratch[..n]);
            }
        }

        let mut tag_bytes = [0u8; ARTIFACT_HMAC_LEN];
        reader.read_exact(&mut tag_bytes).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "tag read failed");
            ArtifactError::Io
        })?;
        let expected = finalize_into_tag(mac);
        use subtle::ConstantTimeEq;
        if !bool::from(expected.as_slice().ct_eq(&tag_bytes[..])) {
            warn!(target: "tensor_wasm_artifacts", file = %path.display(), "HMAC mismatch (get_to verify pass)");
            return Err(ArtifactError::BadHmac);
        }
        Ok(VerifiedBlob {
            hash_on_disk,
            body_len,
        })
    }

    /// Pass 2 of the streaming `get_to`: stream-decode the already-verified
    /// blob directly into `out`, enforcing the decompressed-size cap and
    /// recomputing the content hash as defence-in-depth. Only called after
    /// [`Self::verify_blob`] has authenticated the bytes of the SAME open
    /// handle, so every byte written to `out` is authenticated.
    ///
    /// TOCTOU fix: this consumes the very handle pass 1 verified — the
    /// caller rewinds it (`seek(SeekFrom::Start(0))`) and hands it back
    /// here rather than re-opening `path`. Re-opening was the old race: a
    /// concurrent `remove`+`put` could swap the path's contents between the
    /// two opens, so pass 2 would decode bytes that pass 1 never
    /// HMAC-authenticated (the content-hash recheck below caught a swapped
    /// payload, but only *after* unverified bytes had already streamed to
    /// `out`). Reusing the handle pins the inode, so the decoded bytes are
    /// byte-for-byte the authenticated ones. `reader` is positioned at the
    /// start of the file on entry. `path` is used only for diagnostics.
    fn decode_to_writer(
        &self,
        reader: &mut BufReader<File>,
        verified: &VerifiedBlob,
        requested: &ContentHash,
        out: &mut dyn Write,
        path: &std::path::Path,
    ) -> Result<u64, ArtifactError> {
        // Skip the header — already validated in pass 1.
        let mut header = [0u8; ARTIFACT_HEADER_LEN];
        reader.read_exact(&mut header).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "header reread failed (decode pass)");
            ArtifactError::Io
        })?;

        let cap = MAX_DECOMPRESSED_LEN;
        let probe_limit = u64::try_from(cap)
            .ok()
            .and_then(|c| c.checked_add(1))
            .unwrap_or(u64::MAX);

        // Decode the body straight into a hashing+counting+capping sink
        // that forwards to `out`. Because the blob is already
        // authenticated, streaming decoded bytes to the caller honours
        // the "no unverified bytes exposed" invariant.
        let body_take = Read::take(&mut *reader, verified.body_len);
        let decoder = zstd::stream::read::Decoder::new(body_take).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed (decode pass)");
            ArtifactError::Decompression(e.to_string())
        })?;
        let mut sink = HashingWriter::new(out);
        let copied = std::io::copy(&mut Read::take(decoder, probe_limit), &mut sink);
        // A `copy` error is one of two things: the downstream writer
        // refused a byte (the `HashingWriter` records this in
        // `downstream_failed`, regardless of the io error kind), or the
        // zstd decoder faulted on the body. Check the downstream flag
        // first so a writer fault is reported as `Io`, not `Decompression`.
        let written = match copied {
            Ok(n) => n,
            Err(e) => {
                if sink.downstream_failed {
                    warn!(target: "tensor_wasm_artifacts", error = %e, "get_to writer write failed");
                    return Err(ArtifactError::Io);
                }
                warn!(target: "tensor_wasm_artifacts", error = %e, "zstd decode failed (decode pass)");
                return Err(ArtifactError::Decompression(e.to_string()));
            }
        };
        if written > cap as u64 {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                actual = written,
                limit = cap,
                "rejecting oversized decompressed payload (possible zstd bomb)"
            );
            return Err(ArtifactError::TooLarge {
                actual: written as usize,
                limit: cap,
            });
        }
        // Defence-in-depth: the streamed bytes must hash to both the
        // header value and the requested key.
        let recomputed: [u8; 32] = sink.hasher.finalize().into();
        if recomputed != verified.hash_on_disk {
            return Err(ArtifactError::HashMismatch {
                expected: hex_of(&verified.hash_on_disk),
                actual: hex_of(&recomputed),
            });
        }
        if recomputed != *requested.as_bytes() {
            return Err(ArtifactError::HashMismatch {
                expected: requested.to_string(),
                actual: hex_of(&recomputed),
            });
        }
        Ok(written)
    }

    /// Resolve which accepted read key holds `hash`, if any. Probes each
    /// key from [`KeyProvider::read_keys`] (active first) with a stat-only
    /// existence check and returns the `(path, key)` of the first match —
    /// keys partition the namespace by fingerprint, so there is at most
    /// one. Returns `Ok(None)` on a genuine miss across all keys, and
    /// [`ArtifactError::Io`] on a probe fault that is neither present nor
    /// absent.
    fn resolve_read_key(
        &self,
        hash: &ContentHash,
    ) -> Result<Option<(PathBuf, [u8; 32])>, ArtifactError> {
        for key in self.key_provider.read_keys() {
            let fp = key_fingerprint_hex(&key);
            let path = self.path_for_key(hash, &fp);
            match std::fs::metadata(&path) {
                Ok(_) => return Ok(Some((path, key))),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    warn!(
                        target: "tensor_wasm_artifacts",
                        file = %path.display(),
                        error = %e,
                        "resolve_read_key metadata probe failed"
                    );
                    return Err(ArtifactError::Io);
                }
            }
        }
        Ok(None)
    }

    /// Best-effort fsync of the store's containing directory after an
    /// atomic rename, so the rename (a directory-entry mutation) is
    /// persisted and not just the file data.
    ///
    /// Durability fix (LOW): a crash between `persist` and the OS
    /// flushing the directory metadata could otherwise resurrect the old
    /// directory state even though the file data is already on stable
    /// storage. fsyncing the directory closes that window on POSIX.
    ///
    /// This is deliberately best-effort: on Windows you cannot open a
    /// directory as a syncable `File` (the open or the `sync_all` errors
    /// with e.g. `PermissionDenied`/`InvalidInput`), so a failure here is
    /// logged at debug level and swallowed rather than failing the
    /// otherwise-successful `put`. The file's own `sync_all` (issued
    /// before the rename) is the load-bearing durability step on every
    /// platform; the directory sync is the extra POSIX guarantee.
    fn sync_dir_best_effort(&self) {
        match File::open(&self.dir).and_then(|d| d.sync_all()) {
            Ok(()) => {}
            Err(e) => {
                tracing::debug!(
                    target: "tensor_wasm_artifacts",
                    dir = %self.dir.display(),
                    error = %e,
                    "directory fsync skipped (best-effort; unsupported on this platform?)"
                );
            }
        }
    }
}

/// Result of [`DiskArtifactStore::verify_blob`]: the authenticated header
/// hash and the zstd body length, handed to the decode pass.
struct VerifiedBlob {
    hash_on_disk: [u8; 32],
    body_len: u64,
}

/// `Write` adapter that tees bytes into a BLAKE3 hasher (for the
/// defence-in-depth content-hash recheck), counts them, and forwards to a
/// downstream writer. Used by the streaming `get_to` decode pass so the
/// decoded bytes are hashed *as they stream to the caller* — no second
/// buffer.
///
/// A downstream write failure is recorded in `downstream_failed` and
/// surfaced as an `io::ErrorKind::Other` so the caller can map it to
/// [`ArtifactError::Io`] (and distinguish it from a zstd decode fault,
/// which `std::io::copy` reports with the same kind).
struct HashingWriter<'a> {
    inner: &'a mut dyn Write,
    hasher: blake3::Hasher,
    downstream_failed: bool,
}

impl<'a> HashingWriter<'a> {
    fn new(inner: &'a mut dyn Write) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            downstream_failed: false,
        }
    }
}

impl Write for HashingWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Forward first; only hash the bytes the downstream accepted so
        // the recomputed hash matches exactly what reached the caller.
        match self.inner.write(buf) {
            Ok(n) => {
                self.hasher.update(&buf[..n]);
                Ok(n)
            }
            Err(e) => {
                self.downstream_failed = true;
                Err(e)
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
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

        // Durability (crash-consistency fix): flush the temp file's data
        // to stable storage BEFORE the atomic rename. Without this
        // `sync_all`, a crash after `persist` could leave the renamed
        // file's directory entry pointing at data still sitting in the
        // page cache, so the blob the doc comments promise is durable
        // could come back truncated or zero-length on the next boot.
        tmp.as_file().sync_all().map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "blob fsync (pre-persist) failed");
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

        // Durability: fsync the containing directory so the rename itself
        // is persisted, not just the file data. On Windows a directory
        // `sync_all` typically fails (you cannot open a directory as a
        // syncable file handle); that is tolerated as best-effort — the
        // file data is already durable from the `sync_all` above, which
        // is the part that matters most for crash consistency.
        self.sync_dir_best_effort();
        Ok(hash)
    }

    fn get(&self, hash: &ContentHash) -> Result<Vec<u8>, ArtifactError> {
        // Rotation-aware key resolution: keys partition the on-disk
        // namespace by fingerprint, so at most one accepted read key has a
        // file for this content hash. Probe each (active first) and use
        // whichever one's file exists. If none exist it's a genuine miss.
        let (path, key) = match self.resolve_read_key(hash)? {
            Some(found) => found,
            None => return Err(ArtifactError::NotFound(hash.to_string())),
        };

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
        let mut mac = new_mac(&key);

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
        // clamped to the cap, then let the Vec grow on demand. PERF/
        // security: the old 4x multiplier reserved 256 MiB up front for a
        // 64 MiB compressed body — a large speculative allocation driven
        // by attacker-controlled body length. Use a conservative 2x
        // estimate (most tensor-memory payloads compress better than 2:1,
        // so this rarely under-reserves much) and rely on `Vec`'s
        // amortised growth for the incompressible tail.
        let initial_capacity = usize::try_from(body_len)
            .unwrap_or(cap)
            .saturating_mul(2)
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

    fn get_to(&self, hash: &ContentHash, out: &mut dyn Write) -> Result<u64, ArtifactError> {
        // Two-pass streaming over a SINGLE open handle: the HMAC covers the
        // whole compressed blob, so we must authenticate before exposing any
        // decoded byte. Pass 1 (`verify_blob`) streams the compressed body
        // through the MAC and checks the tag WITHOUT decoding — no decoded
        // byte exists to leak. Pass 2 (`decode_to_writer`) then rewinds the
        // SAME handle and stream-decodes straight into `out`.
        //
        // TOCTOU fix: the file is opened exactly once and both passes read
        // that handle. The previous code re-opened the path for pass 2, so a
        // concurrent `remove`+`put` between the two opens could swap the
        // path's contents and pass 2 would decode bytes pass 1 never
        // HMAC-verified — weakening the doc-promised invariant that `out`
        // only ever sees authenticated bytes (the content-hash recheck
        // caught a swapped payload, but only after unverified bytes had
        // streamed out). Holding one handle pins the inode: an unlink+swap
        // unlinks the directory entry but our handle keeps reading the
        // original, authenticated bytes. We rewind with `seek(Start(0))`
        // between passes rather than re-opening.
        //
        // Tradeoff: this reads the compressed body off disk twice (re-read
        // via `seek`, not buffered in RAM). That keeps peak heap bounded by
        // the I/O buffers regardless of payload size — the alternative
        // (buffer the whole decoded payload, verify, then write) would hold
        // up to MAX_DECOMPRESSED_LEN resident, which is exactly what `get`
        // already does and what the streaming path exists to avoid.
        use std::io::{Seek, SeekFrom};

        let (path, key) = match self.resolve_read_key(hash)? {
            Some(found) => found,
            None => return Err(ArtifactError::NotFound(hash.to_string())),
        };

        // Open ONCE; both passes consume this handle.
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Raced an unlink between resolve and open: a genuine miss.
                return Err(ArtifactError::NotFound(hash.to_string()));
            }
            Err(e) => {
                warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "open failed (get_to)");
                return Err(ArtifactError::Io);
            }
        };
        let file_len = file
            .metadata()
            .map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "metadata failed (get_to)");
                ArtifactError::Io
            })?
            .len();

        let mut reader = BufReader::with_capacity(STREAM_BUF_LEN, file);

        // Pass 1: authenticate from this handle.
        let verified = self.verify_blob(&mut reader, file_len, &key, &path)?;

        // Rewind the SAME handle so pass 2 decodes the exact bytes pass 1
        // authenticated — not whatever a racing writer may have swapped in.
        reader.seek(SeekFrom::Start(0)).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", file = %path.display(), error = %e, "rewind failed (get_to decode pass)");
            ArtifactError::Io
        })?;

        // Pass 2: decode from the same (now-rewound) handle.
        self.decode_to_writer(&mut reader, &verified, hash, out, &path)
    }

    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError> {
        // Rotation-aware enumeration: union the hashes visible under every
        // accepted read-key fingerprint so an audit spans the rotation
        // boundary. Dedup because a content hash may be present under more
        // than one key (e.g. re-put after rotation). Probe each key's
        // `.bin` suffix in turn.
        let mut seen: std::collections::HashSet<ContentHash> = std::collections::HashSet::new();
        let mut fps: Vec<String> = self
            .key_provider
            .read_keys()
            .iter()
            .map(key_fingerprint_hex)
            .collect();
        // The single-key common case has exactly one fingerprint; dedup the
        // fingerprint list so a provider that repeats the active key in
        // `read_keys` doesn't scan the directory twice.
        fps.sort();
        fps.dedup();
        for fp in fps {
            self.list_one_key(&fp, &mut seen)?;
        }
        Ok(seen.into_iter().collect())
    }

    fn contains(&self, hash: &ContentHash) -> Result<bool, ArtifactError> {
        // Rotation-aware stat-only existence probe: no open, no decode, no
        // HMAC. Resolves against every accepted read key (active first) with
        // the same [`Self::resolve_read_key`] probe `get` / `list` use, so a
        // blob written under a now-retired key reports `true` (the old code
        // checked only the active key, so such a blob was visible to `get`
        // yet reported absent here — a GC-leak hazard). `resolve_read_key`
        // returns `Ok(None)` on a genuine miss across all keys and
        // [`ArtifactError::Io`] on a probe fault that is neither present nor
        // absent.
        Ok(self.resolve_read_key(hash)?.is_some())
    }

    fn remove(&self, hash: &ContentHash) -> Result<bool, ArtifactError> {
        // Rotation-aware unlink: resolve which accepted key (active or
        // retired) actually holds the blob with the same
        // [`Self::resolve_read_key`] probe `get` / `list` use, then unlink
        // under THAT key. The old code unlinked only under the active key,
        // so a blob written under a now-retired key was visible to `get` but
        // could never be removed here — a GC leak that contradicted the
        // rotation story. A genuine miss across all keys returns `false`,
        // not an error — this matches `HashMap::remove`'s boolean and lets
        // idempotent GC retries stay quiet.
        let (path, key) = match self.resolve_read_key(hash)? {
            Some(found) => found,
            None => return Ok(false),
        };
        let removed = match std::fs::remove_file(&path) {
            Ok(()) => true,
            // Raced a concurrent remove between resolve and unlink: the
            // entry is already gone, which is the same "nothing removed"
            // outcome as a genuine miss.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %path.display(),
                    error = %e,
                    "remove unlink failed"
                );
                return Err(ArtifactError::Io);
            }
        };
        // Best-effort: drop the metadata sidecar too so it never outlives
        // the blob it describes. The sidecar is partitioned under the SAME
        // key fingerprint as the blob, so resolve its path under the key we
        // just unlinked. A missing sidecar is the common case (plain `put`
        // writes none); any other unlink fault is logged but does not flip
        // the blob-removal result, since the blob — the authoritative entry
        // — is already gone.
        let fp = key_fingerprint_hex(&key);
        let meta_path = self.meta_path_for_key(hash, &fp);
        if let Err(e) = std::fs::remove_file(&meta_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %meta_path.display(),
                    error = %e,
                    "remove sidecar unlink failed (blob already removed)"
                );
            }
        }
        Ok(removed)
    }
}

impl DiskArtifactStore {
    /// Enumerate the content hashes stored under the single key fingerprint
    /// `fp`, inserting each into `seen`. Shared by [`ArtifactStore::list`]'s
    /// per-key loop. A missing store directory is treated as empty (the dir
    /// is created lazily on the first `put`); any other `read_dir` fault is
    /// propagated as [`ArtifactError::Io`].
    fn list_one_key(
        &self,
        fp: &str,
        seen: &mut std::collections::HashSet<ContentHash>,
    ) -> Result<(), ArtifactError> {
        let suffix = format!(".{fp}.bin");
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
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
            // Match the suffix so we ignore files written under a
            // different key (and the `.meta.json` sidecars).
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
                seen.insert(ContentHash::from_bytes(bytes));
            }
        }
        Ok(())
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
    // Size the initial allocation from the compressed body length,
    // clamped to the cap, and let the Vec grow on demand. PERF/security:
    // the old 4x multiplier reserved up to 4x the compressed body up
    // front (256 MiB for a 64 MiB body) on an attacker-controlled
    // length. A conservative 2x estimate covers typical tensor-memory
    // payloads while `Vec`'s amortised growth handles the rest.
    let initial_capacity = body.len().saturating_mul(2).min(cap);
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
/// caches.
///
/// # SECURITY WARNING: NO integrity or signature verification
///
/// Unlike [`DiskArtifactStore`], this store performs **NO HMAC signing
/// and NO HMAC/content-hash verification** on `put` / `get`. It stores
/// the plaintext payload in a `HashMap` and hands it straight back. The
/// `hmac_key` below is accepted only so the constructor signature
/// matches the disk store's; it is never used to sign or verify anything
/// (hence `#[allow(dead_code)]`).
///
/// This is safe **only** because the in-memory map is the trust boundary:
/// the bytes returned are exactly the bytes a (trusted) caller inserted
/// in the same process, so there is no untrusted on-the-wire/on-disk
/// envelope to authenticate. Do **NOT** use this type to back any data
/// path that crosses a trust boundary (untrusted input, persistence,
/// IPC, network). For anything that must detect tampering or forged
/// blobs, use [`DiskArtifactStore`], which signs and constant-time
/// verifies every record. Treat this store as test/ephemeral-only.
pub struct InMemoryArtifactStore {
    entries: parking_lot::Mutex<HashMap<ContentHash, Vec<u8>>>,
    // Held for parity with `DiskArtifactStore::hmac_key` (and zeroized on
    // drop), but DELIBERATELY never read: this store does no signing or
    // verification (see the type-level SECURITY WARNING). Not read on any
    // path. Future migrations (snapshot replay-protection matrix,
    // signed-kernel-registry) may want to surface the key fingerprint
    // from an in-memory store too.
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
    fn get_to(&self, hash: &ContentHash, out: &mut dyn Write) -> Result<u64, ArtifactError> {
        // The in-memory map already holds the plaintext payload (integrity
        // is guaranteed by the map itself — no envelope to verify), so
        // streaming is a single write of the stored bytes while the lock
        // is held. We write under the lock to avoid cloning the payload
        // first; the bytes are authentic by construction.
        let guard = self.entries.lock();
        let bytes = guard
            .get(hash)
            .ok_or_else(|| ArtifactError::NotFound(hash.to_string()))?;
        out.write_all(bytes).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "in-memory get_to writer write failed");
            ArtifactError::Io
        })?;
        Ok(bytes.len() as u64)
    }
    fn list(&self) -> Result<Vec<ContentHash>, ArtifactError> {
        // An in-memory map cannot fail to enumerate, so this is
        // infallible — but the signature matches the trait so callers
        // treat both stores uniformly.
        Ok(self.entries.lock().keys().copied().collect())
    }
    fn contains(&self, hash: &ContentHash) -> Result<bool, ArtifactError> {
        // `contains_key` is the cheap probe — no clone of the payload,
        // matching the disk store's stat-only contract. Infallible for
        // an in-memory map; wrapped in `Ok` for trait uniformity.
        Ok(self.entries.lock().contains_key(hash))
    }
    fn remove(&self, hash: &ContentHash) -> Result<bool, ArtifactError> {
        // `HashMap::remove` returns the old value; map it to the
        // trait's "was something removed?" boolean.
        Ok(self.entries.lock().remove(hash).is_some())
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
    fn in_memory_contains_and_remove() {
        let store = InMemoryArtifactStore::new([9u8; 32]);
        let hash = store.put(b"x").unwrap();
        assert!(store.contains(&hash).unwrap());
        assert!(store.remove(&hash).unwrap(), "first remove deletes");
        assert!(!store.contains(&hash).unwrap());
        assert!(!store.remove(&hash).unwrap(), "second remove is a no-op");
    }

    #[test]
    fn in_memory_get_to_streams_payload() {
        let store = InMemoryArtifactStore::new([5u8; 32]);
        let hash = store.put(b"streamed").unwrap();
        let mut out: Vec<u8> = Vec::new();
        let n = store.get_to(&hash, &mut out).unwrap();
        assert_eq!(n, 8);
        assert_eq!(out, b"streamed");
    }

    #[test]
    fn single_key_provider_active_is_only_read_key() {
        let p = SingleKeyProvider::new([3u8; 32]);
        assert_eq!(p.active_key(), [3u8; 32]);
        assert_eq!(p.read_keys(), vec![[3u8; 32]]);
    }

    #[test]
    fn rotating_provider_active_first_then_accepted() {
        let p = RotatingKeyProvider::new([1u8; 32], [[2u8; 32], [3u8; 32]]);
        assert_eq!(p.active_key(), [1u8; 32]);
        // Active key leads, then the accepted-for-read keys in order.
        assert_eq!(p.read_keys(), vec![[1u8; 32], [2u8; 32], [3u8; 32]]);
    }

    #[test]
    fn metadata_serde_round_trips() {
        let m = ArtifactMetadata {
            created_unix_ms: 123,
            original_len: 456,
            source_tier: "test".to_string(),
        };
        let json = serde_json::to_vec(&m).unwrap();
        let back: ArtifactMetadata = serde_json::from_slice(&json).unwrap();
        assert_eq!(m, back);
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
