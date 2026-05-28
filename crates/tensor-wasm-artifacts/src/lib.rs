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
use std::io::Write;
use std::path::PathBuf;

use thiserror::Error;
use tracing::warn;
use zeroize::Zeroizing;

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
    #[error("I/O error")]
    Io,
}

/// 32-byte BLAKE3 content hash. The `put` path computes this from the
/// uncompressed payload; the `get` path recomputes it from the decoded
/// body and rejects on mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

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
    fn list(&self) -> Vec<ContentHash>;
}

// =====================================================================
// HMAC helper — shared by `DiskArtifactStore` put and get paths.
// =====================================================================

/// Compute the HMAC-SHA256 tag over `bytes` with `key`. Centralised so
/// the put and get paths cannot drift apart; both call this helper.
fn hmac_tag(key: &[u8; 32], bytes: &[u8]) -> [u8; ARTIFACT_HMAC_LEN] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    // `new_from_slice` only errors on invalid key length; ours is a
    // fixed 32 bytes so the unwrap is sound (mirrors the same pattern
    // in `tensor-wasm-snapshot::SnapshotWriter::capture`).
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key[..])
        .expect("HMAC-SHA256 accepts any 32-byte key");
    mac.update(bytes);
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
    h.as_bytes()[..8].iter().map(|b| format!("{:02x}", b)).collect()
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
}

impl DiskArtifactStore {
    /// Construct a disk store rooted at `dir`, signing with `hmac_key`.
    /// The directory is created lazily on the first `put`.
    pub fn new(dir: PathBuf, hmac_key: [u8; 32]) -> Self {
        Self {
            dir,
            hmac_key: Zeroizing::new(hmac_key),
        }
    }

    /// Compute the on-disk path for `hash` under this store's key.
    ///
    /// Filename format: `{content_hash_hex}.{key_fp_hex}.bin`. The key
    /// fingerprint segment partitions the namespace per HMAC key so
    /// two stores in the same dir under different keys never collide.
    fn path_for(&self, hash: &ContentHash) -> PathBuf {
        let key_hex = key_fingerprint_hex(&self.hmac_key);
        self.dir.join(format!("{}.{}.bin", hash, key_hex))
    }
}

impl ArtifactStore for DiskArtifactStore {
    fn put(&self, payload: &[u8]) -> Result<ContentHash, ArtifactError> {
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

        // Buffer the framed bytes in memory: header + zstd(body) + HMAC.
        // For very large payloads this still holds the compressed body
        // in RAM, matching the snapshot writer's behaviour. A streaming
        // variant can be added later without changing the on-disk
        // layout.
        let mut buf: Vec<u8> = Vec::with_capacity(ARTIFACT_HEADER_LEN + payload.len() / 2 + ARTIFACT_HMAC_LEN);
        buf.extend_from_slice(&ARTIFACT_MAGIC);
        buf.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
        buf.extend_from_slice(&hash.0);

        // Compress payload directly into the framed buffer. Identical
        // shape to `SnapshotWriter::capture` — encode into the encoder,
        // then `finish` to flush the zstd footer.
        let mut encoder = zstd::stream::write::Encoder::new(&mut buf, DEFAULT_ZSTD_LEVEL)
            .map_err(|e| {
                warn!(target: "tensor_wasm_artifacts", error = %e, "zstd init failed");
                ArtifactError::Io
            })?;
        encoder.write_all(payload).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd write failed");
            ArtifactError::Io
        })?;
        encoder.finish().map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd finish failed");
            ArtifactError::Io
        })?;

        // HMAC over everything written so far (header + zstd body).
        let tag = hmac_tag(&self.hmac_key, &buf);
        buf.extend_from_slice(&tag);

        // Atomic write: temp-then-rename in the same directory, mirroring
        // the JIT L2 disk-cache pattern so a partial write can never
        // leave a half-formed entry that a concurrent reader trips over.
        let final_path = self.path_for(&hash);
        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "tempfile create failed");
            ArtifactError::Io
        })?;
        tmp.as_file_mut().write_all(&buf).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "tempfile write failed");
            ArtifactError::Io
        })?;
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
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ArtifactError::NotFound(hash.to_string()));
            }
            Err(e) => {
                warn!(
                    target: "tensor_wasm_artifacts",
                    file = %path.display(),
                    error = %e,
                    "read failed"
                );
                return Err(ArtifactError::Io);
            }
        };

        // Minimum-length gate: header + at least one byte of zstd frame + HMAC.
        if bytes.len() < ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                len = bytes.len(),
                "artifact too short for header+hmac"
            );
            return Err(ArtifactError::BadMagic);
        }

        // Magic check before any keyed work — keeps the verifier cheap
        // for foreign blobs and matches the snapshot reader's order.
        // `PartialEq<[B; N]> for [A]` lets us compare the leading slice
        // against the magic array directly.
        if bytes[..16] != ARTIFACT_MAGIC {
            return Err(ArtifactError::BadMagic);
        }

        let version = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        if version != ARTIFACT_VERSION {
            return Err(ArtifactError::BadVersion(version));
        }

        let mut hash_on_disk = [0u8; 32];
        hash_on_disk.copy_from_slice(&bytes[20..52]);

        // HMAC verification: split off the trailing tag, recompute over
        // the prefix, compare in constant time.
        let hmac_start = bytes.len() - ARTIFACT_HMAC_LEN;
        let (prefix, tag_bytes) = bytes.split_at(hmac_start);
        let expected = hmac_tag(&self.hmac_key, prefix);
        use subtle::ConstantTimeEq;
        // Length is structural metadata (the file's size), not secret
        // content, so checking it before `ct_eq` is safe. `ct_eq`
        // itself requires equal-length inputs to be meaningful.
        let length_ok = tag_bytes.len() == ARTIFACT_HMAC_LEN;
        // `expected` is `[u8; 32]`; `ConstantTimeEq` is implemented on
        // `[u8]` (the slice). `as_slice()` lowers the array to that
        // slice without copying. Mirrors the call shape in
        // `tensor-wasm-snapshot::reader` exactly.
        let mac_ok = length_ok && bool::from(expected.as_slice().ct_eq(tag_bytes));
        if !mac_ok {
            warn!(
                target: "tensor_wasm_artifacts",
                file = %path.display(),
                "HMAC mismatch (possible tampering or stale key)"
            );
            return Err(ArtifactError::BadHmac);
        }

        // Decompress the body that sits between the header and the HMAC tag.
        let body = &prefix[ARTIFACT_HEADER_LEN..];
        let payload = zstd::decode_all(body).map_err(|e| {
            warn!(target: "tensor_wasm_artifacts", error = %e, "zstd decode failed");
            ArtifactError::Decompression(e.to_string())
        })?;

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

    fn list(&self) -> Vec<ContentHash> {
        let key_hex = key_fingerprint_hex(&self.hmac_key);
        let suffix = format!(".{}.bin", key_hex);
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
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
                    Err(_) => { ok = false; break; }
                };
                match u8::from_str_radix(s, 16) {
                    Ok(b) => bytes[i] = b,
                    Err(_) => { ok = false; break; }
                }
            }
            if ok {
                out.push(ContentHash(bytes));
            }
        }
        out
    }
}

fn hex_of(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
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
    fn list(&self) -> Vec<ContentHash> {
        self.entries.lock().keys().copied().collect()
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
        assert_eq!(store.list().len(), 1);
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
