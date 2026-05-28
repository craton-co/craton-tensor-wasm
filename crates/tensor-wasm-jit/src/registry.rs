// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Signed kernel registry (roadmap feature #3).
//!
//! Operators publish vetted PTX kernels (matmul, attention, conv2d) as
//! HMAC-SHA256-signed [`KernelManifest`] records. Guests reference kernels
//! by `name@version` (or content-addressed digest); the runtime resolves
//! the manifest, verifies the signature, and exposes the PTX text to
//! the JIT cache as a pre-populated entry.
//!
//! ## v0.3.7 status: scaffold
//!
//! Manifest types, signing helpers, and the registry trait surface land.
//! Actual on-disk store, signing CLI, and wire-format integration are
//! v0.4 deliverables — see `docs/KERNEL-REGISTRY.md`.
//!
//! ## Signing envelope
//!
//! The HMAC-SHA256 input is the byte concatenation
//!
//! ```text
//! name || 0x00 || version || 0x00 || sm_version_le_u32 || digest
//! ```
//!
//! where `digest` is the BLAKE3 hash of the UTF-8 PTX text. The
//! `0x00` separators prevent length-extension confusion between, e.g.,
//! `("matmul", "f32-1.0")` and `("matmul.f32", "-1.0")`. The
//! `published_unix_ms` and `publisher` fields are NOT covered by the
//! signature in v0.3.7 — they are advisory metadata only. The v0.4
//! wire format will extend the envelope to cover them once a stable
//! canonical encoding is settled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Signed kernel manifest. Cargo for vetted PTX.
///
/// Marked `#[non_exhaustive]` so future revisions can add fields
/// without breaking downstream pattern-matching consumers. Construct
/// instances via [`KernelManifest::new`] from outside this crate (the
/// `non_exhaustive` attribute disallows struct-literal construction
/// from foreign crates).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KernelManifest {
    /// Stable name (e.g. `"matmul.f32"`).
    pub name: String,
    /// SemVer-style version (`"1.0.0"`).
    pub version: String,
    /// Compute capability the PTX was built for (e.g. `80` for sm_80).
    pub sm_version: u32,
    /// BLAKE3 hash of the PTX text. Content-addresses the artifact.
    pub digest: [u8; 32],
    /// HMAC-SHA256 over (name || 0 || version || 0 || sm_version_le || digest).
    pub signature: [u8; 32],
    /// Wall-clock publish timestamp (Unix millis). Advisory metadata —
    /// NOT covered by the signature in v0.3.7.
    pub published_unix_ms: u64,
    /// Publisher identifier (typically a tenant id or signing-key id).
    /// Advisory metadata — NOT covered by the signature in v0.3.7.
    pub publisher: String,
}

impl KernelManifest {
    /// Construct a manifest from its component fields.
    ///
    /// `KernelManifest` is `#[non_exhaustive]`, so foreign crates cannot
    /// build one via a struct literal — they must go through this
    /// constructor. The `signature` field is typically left as
    /// `[0u8; 32]` here and filled in afterwards by [`sign_manifest`],
    /// because the publisher's HMAC key is what produces the
    /// signature value.
    pub fn new(
        name: String,
        version: String,
        sm_version: u32,
        digest: [u8; 32],
        signature: [u8; 32],
        published_unix_ms: u64,
        publisher: String,
    ) -> Self {
        Self {
            name,
            version,
            sm_version,
            digest,
            signature,
            published_unix_ms,
            publisher,
        }
    }

    /// First 8 bytes of `digest` interpreted as a little-endian `u64`.
    /// Used as the synthetic `fingerprint` field on the `CachedKernel`
    /// that the registry path promotes into L1.
    pub fn digest_as_u64(&self) -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.digest[..8]);
        u64::from_le_bytes(buf)
    }
}

/// Resolves a JIT cache key to a (name, version) tuple that the
/// registry can look up. The cache is keyed by (tenant, blueprint
/// fingerprint, sm_version, emit_config_hash); the registry is keyed
/// by (name, version). This trait is the bridge — embedders provide
/// the mapping policy (e.g. a YAML manifest baked into the deploy,
/// or a tenant-level metadata table).
pub trait BlueprintResolver: Send + Sync {
    /// Resolve a (blueprint fingerprint, sm_version) pair to a
    /// `(name, version)` tuple that [`KernelRegistry::get`] understands.
    /// Returns `None` if the embedder has no mapping for this
    /// blueprint — the caller then proceeds with fresh PTX emission.
    fn resolve(&self, blueprint_fp: u64, sm_version: u32) -> Option<(String, String)>;
}

/// In-memory [`BlueprintResolver`] backed by a `HashMap`. Test-only
/// convenience for the cache integration tests in v0.3.8; production
/// embedders supply their own implementation that consults a
/// deployment-baked manifest or a tenant-level metadata table.
pub struct InMemoryBlueprintResolver {
    map: HashMap<(u64, u32), (String, String)>,
}

impl InMemoryBlueprintResolver {
    /// Construct an empty resolver. Use [`Self::insert`] to populate
    /// the (blueprint, sm) → (name, version) mapping.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Build a resolver pre-populated from a map of
    /// `(blueprint_fp, sm_version)` → `(name, version)` entries.
    pub fn from_map(map: HashMap<(u64, u32), (String, String)>) -> Self {
        Self { map }
    }

    /// Insert a single `(blueprint_fp, sm_version)` → `(name, version)`
    /// mapping. Overwrites any prior entry for the same key.
    pub fn insert(
        &mut self,
        blueprint_fp: u64,
        sm_version: u32,
        name: String,
        version: String,
    ) {
        self.map.insert((blueprint_fp, sm_version), (name, version));
    }
}

impl Default for InMemoryBlueprintResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BlueprintResolver for InMemoryBlueprintResolver {
    fn resolve(&self, blueprint_fp: u64, sm_version: u32) -> Option<(String, String)> {
        self.map.get(&(blueprint_fp, sm_version)).cloned()
    }
}

/// Failure modes for registry operations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// No entry for the requested `name@version`.
    #[error("kernel not found: {0}")]
    NotFound(String),
    /// HMAC verification failed against the configured signing key.
    #[error("signature verification failed for {0}")]
    BadSignature(String),
    /// BLAKE3 of the PTX text does not match `manifest.digest`.
    #[error("digest mismatch for {0}")]
    DigestMismatch(String),
    /// A manifest with the same `name@version` already exists.
    #[error("name @ version already registered: {0}")]
    AlreadyRegistered(String),
}

/// Registry trait. v0.3.7 ships an in-memory impl; v0.4 lands disk +
/// remote backends. The pair returned by [`Self::get`] is
/// `(manifest, ptx_text)` so a caller can hand the PTX to the JIT cache
/// without a second registry round-trip.
pub trait KernelRegistry: Send + Sync {
    /// Resolve `(name, version)` to the verified manifest + PTX text.
    fn get(
        &self,
        name: &str,
        version: &str,
    ) -> Result<Arc<(KernelManifest, String)>, RegistryError>;
    /// Enumerate all registered manifests (PTX text omitted).
    fn list(&self) -> Vec<KernelManifest>;
}

/// In-memory implementation for v0.3.7. Holds manifests + PTX text in a
/// `Mutex<HashMap>` keyed by `"{name}@{version}"`.
///
/// The HMAC-SHA256 signing key is held in a [`zeroize::Zeroizing`] so the
/// scrub-on-drop guarantees from the snapshot signing path apply here too:
/// the secret material does not linger in the heap arena after the
/// registry is dropped.
pub struct InMemoryRegistry {
    entries: parking_lot::Mutex<HashMap<String, Arc<(KernelManifest, String)>>>,
    hmac_key: zeroize::Zeroizing<[u8; 32]>,
}

impl InMemoryRegistry {
    /// Construct an empty registry that accepts manifests signed under
    /// `hmac_key`.
    pub fn new(hmac_key: [u8; 32]) -> Self {
        Self {
            entries: parking_lot::Mutex::new(HashMap::new()),
            hmac_key: zeroize::Zeroizing::new(hmac_key),
        }
    }

    /// Verify, then insert, a signed manifest plus its PTX text.
    ///
    /// Returns [`RegistryError::DigestMismatch`] if BLAKE3 of `ptx_text`
    /// does not match `manifest.digest`, [`RegistryError::BadSignature`]
    /// if the HMAC does not verify under the configured key, and
    /// [`RegistryError::AlreadyRegistered`] if `name@version` is already
    /// present. The check order (digest → signature → uniqueness) is
    /// deliberate: a digest mismatch is the cheapest tell that the
    /// PTX/manifest pair was corrupted in transit, so we surface that
    /// before doing the constant-time HMAC compare.
    pub fn publish(
        &self,
        manifest: KernelManifest,
        ptx_text: String,
    ) -> Result<(), RegistryError> {
        // Verify digest matches PTX.
        let actual = blake3::hash(ptx_text.as_bytes());
        if actual.as_bytes() != &manifest.digest {
            return Err(RegistryError::DigestMismatch(manifest.name.clone()));
        }
        // Verify signature.
        self.verify_signature(&manifest)?;
        let key = format!("{}@{}", manifest.name, manifest.version);
        let mut entries = self.entries.lock();
        if entries.contains_key(&key) {
            return Err(RegistryError::AlreadyRegistered(key));
        }
        entries.insert(key, Arc::new((manifest, ptx_text)));
        Ok(())
    }

    /// Recompute and constant-time-compare the manifest's HMAC against
    /// the configured signing key. See the module-level docstring for
    /// the exact envelope layout.
    fn verify_signature(&self, manifest: &KernelManifest) -> Result<(), RegistryError> {
        use hmac::{Hmac, Mac};
        // `new_from_slice` only errors on invalid key length; ours is a
        // compile-time 32-byte array so the unwrap is sound.
        let mut mac = <Hmac<sha2::Sha256> as Mac>::new_from_slice(&self.hmac_key[..])
            .expect("32-byte key is always valid HMAC-SHA256 input");
        mac.update(manifest.name.as_bytes());
        mac.update(b"\0");
        mac.update(manifest.version.as_bytes());
        mac.update(b"\0");
        mac.update(&manifest.sm_version.to_le_bytes());
        mac.update(&manifest.digest);
        let expected = mac.finalize().into_bytes();
        let ok = subtle::ConstantTimeEq::ct_eq(&expected[..], &manifest.signature[..]);
        if bool::from(ok) {
            Ok(())
        } else {
            Err(RegistryError::BadSignature(manifest.name.clone()))
        }
    }
}

impl KernelRegistry for InMemoryRegistry {
    fn get(
        &self,
        name: &str,
        version: &str,
    ) -> Result<Arc<(KernelManifest, String)>, RegistryError> {
        let key = format!("{name}@{version}");
        self.entries
            .lock()
            .get(&key)
            .cloned()
            .ok_or(RegistryError::NotFound(key))
    }

    fn list(&self) -> Vec<KernelManifest> {
        self.entries
            .lock()
            .values()
            .map(|e| e.0.clone())
            .collect()
    }
}

/// Sign a manifest given a publisher's HMAC key. Helper for tests and
/// the v0.4 signing CLI.
///
/// Callers populate every field of `unsigned` (including
/// `signature: [0; 32]`); this helper computes the real signature over
/// the canonical envelope and returns it. The caller is then responsible
/// for writing it back into `unsigned.signature` before publishing.
pub fn sign_manifest(unsigned: &KernelManifest, hmac_key: &[u8; 32]) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    let mut mac = <Hmac<sha2::Sha256> as Mac>::new_from_slice(hmac_key)
        .expect("32-byte key is always valid HMAC-SHA256 input");
    mac.update(unsigned.name.as_bytes());
    mac.update(b"\0");
    mac.update(unsigned.version.as_bytes());
    mac.update(b"\0");
    mac.update(&unsigned.sm_version.to_le_bytes());
    mac.update(&unsigned.digest);
    mac.finalize().into_bytes().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a manifest for `ptx_text` signed under `key`.
    /// In-crate tests are allowed to use struct-literal construction
    /// (the `#[non_exhaustive]` attribute only restricts foreign crates),
    /// but the helper here mirrors the public `KernelManifest::new`
    /// flow so the unit and integration tests stay in lock-step.
    fn signed_manifest(name: &str, version: &str, ptx_text: &str, key: &[u8; 32]) -> KernelManifest {
        let digest = *blake3::hash(ptx_text.as_bytes()).as_bytes();
        let mut m = KernelManifest::new(
            name.to_string(),
            version.to_string(),
            80,
            digest,
            [0u8; 32],
            0,
            "test".to_string(),
        );
        m.signature = sign_manifest(&m, key);
        m
    }

    #[test]
    fn publish_and_get_roundtrip() {
        let key = [0x42u8; 32];
        let reg = InMemoryRegistry::new(key);
        let ptx = "// fake ptx\n".to_string();
        let m = signed_manifest("matmul.f32", "1.0.0", &ptx, &key);
        reg.publish(m.clone(), ptx.clone()).unwrap();
        let got = reg.get("matmul.f32", "1.0.0").unwrap();
        assert_eq!(got.0.name, "matmul.f32");
        assert_eq!(got.1, ptx);
        let listing = reg.list();
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].version, "1.0.0");
    }

    #[test]
    fn rejects_bad_signature() {
        let key = [0x42u8; 32];
        let reg = InMemoryRegistry::new(key);
        let ptx = "// fake ptx\n".to_string();
        let mut m = signed_manifest("matmul.f32", "1.0.0", &ptx, &key);
        // Flip a byte in the signature.
        m.signature[0] ^= 0xff;
        match reg.publish(m, ptx) {
            Err(RegistryError::BadSignature(name)) => assert_eq!(name, "matmul.f32"),
            other => panic!("expected BadSignature, got {other:?}"),
        }
    }

    #[test]
    fn rejects_digest_mismatch() {
        let key = [0x42u8; 32];
        let reg = InMemoryRegistry::new(key);
        let ptx = "// fake ptx\n".to_string();
        let m = signed_manifest("matmul.f32", "1.0.0", &ptx, &key);
        // Publish with different PTX than what was signed.
        match reg.publish(m, "// different ptx\n".to_string()) {
            Err(RegistryError::DigestMismatch(name)) => assert_eq!(name, "matmul.f32"),
            other => panic!("expected DigestMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_publish() {
        let key = [0x42u8; 32];
        let reg = InMemoryRegistry::new(key);
        let ptx = "// fake ptx\n".to_string();
        let m = signed_manifest("matmul.f32", "1.0.0", &ptx, &key);
        reg.publish(m.clone(), ptx.clone()).unwrap();
        match reg.publish(m, ptx) {
            Err(RegistryError::AlreadyRegistered(key)) => {
                assert_eq!(key, "matmul.f32@1.0.0")
            }
            other => panic!("expected AlreadyRegistered, got {other:?}"),
        }
    }

    #[test]
    fn get_returns_not_found_for_missing() {
        let key = [0u8; 32];
        let reg = InMemoryRegistry::new(key);
        match reg.get("nope", "0.0.0") {
            Err(RegistryError::NotFound(k)) => assert_eq!(k, "nope@0.0.0"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }
}
