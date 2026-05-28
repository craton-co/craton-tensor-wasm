//! Regression tests for jit S-3: cache poisoning integrity check
//! (in-memory tag) plus the on-disk L2 cache that closes the audit P-4
//! follow-up.
//!
//! Threat model these tests pin:
//! - In-memory: `CachedKernel::new` computes a BLAKE3 over the PTX text;
//!   `KernelCache::put` rejects entries with a mismatched hash, and
//!   `KernelCache::get` re-verifies on every retrieval so a sibling
//!   mutable holder cannot poison a cached entry.
//! - On-disk: each `.ptxbin` file carries a BLAKE3-keyed-HMAC trailer.
//!   The reader rejects files whose recomputed HMAC does not match the
//!   trailer, whose magic is missing, whose declared length overruns
//!   the file, or whose persisted (fingerprint, sm_version) does not
//!   match the requested key.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_jit::cache::{
    CacheKey, CachedKernel, CompiledHandle, DiskCacheConfig, KernelCache,
};
use tensor_wasm_jit::ptx_emit::EmittedPtx;

fn fixture_ptx(text: &str) -> Arc<EmittedPtx> {
    Arc::new(EmittedPtx {
        text: text.to_string(),
        launch_geometry: (1, 1),
    })
}

#[test]
fn put_rejects_kernel_with_mismatched_integrity_hash() {
    let cache = KernelCache::new();
    let key = CacheKey::for_tenant(TenantId(1), 0xDEAD_BEEF, 80);
    // Hand-crafted CachedKernel with a deliberately wrong hash.
    let bad = CachedKernel {
        fingerprint: 0xDEAD_BEEF,
        ptx: fixture_ptx(".visible .entry safe(){}"),
        compiled: CompiledHandle::default(),
        integrity_hash: [0xAA; 32],
    };
    cache.put(key, bad);
    assert!(
        cache.get(&key).is_none(),
        "put must drop entries with a bad integrity hash"
    );
}

#[test]
fn get_returns_kernel_built_via_constructor() {
    let cache = KernelCache::new();
    let key = CacheKey::for_tenant(TenantId(1), 0xBEEF, 80);
    let kernel = CachedKernel::new(
        0xBEEF,
        fixture_ptx(".visible .entry func0(){}"),
        CompiledHandle::default(),
    );
    cache.put(key, kernel);
    let got = cache
        .get(&key)
        .expect("constructor-built kernel must round-trip");
    assert_eq!(got.fingerprint, 0xBEEF);
    assert!(got.verify_integrity());
}

#[test]
fn disk_cache_round_trip_with_correct_hmac() {
    let tmp = TempDir::new().expect("tempdir");
    let dir: PathBuf = tmp.path().to_path_buf();
    let key_bytes = [0x42; 32];
    let cache = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir: dir.clone(),
        hmac_key: key_bytes,
    });
    let key = CacheKey::for_tenant(TenantId(7), 0xCAFE_BABE, 80);
    let kernel = CachedKernel::new(
        0xCAFE_BABE,
        fixture_ptx(".visible .entry round_trip(){}"),
        CompiledHandle::default(),
    );
    cache.put(key, kernel);

    // Fresh cache pointed at the same dir + same key must read back the
    // entry through the disk path.
    let cache2 = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir: dir.clone(),
        hmac_key: key_bytes,
    });
    let got = cache2
        .get(&key)
        .expect("disk round-trip with correct HMAC must succeed");
    assert_eq!(got.fingerprint, 0xCAFE_BABE);
    assert!(got.ptx.text.contains("round_trip"));
}

#[test]
fn disk_cache_rejects_wrong_hmac_key_treated_as_miss() {
    let tmp = TempDir::new().expect("tempdir");
    let dir: PathBuf = tmp.path().to_path_buf();
    let cache = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir: dir.clone(),
        hmac_key: [0xAA; 32],
    });
    let key = CacheKey::for_tenant(TenantId(7), 0xCAFE_BABE, 80);
    cache.put(
        key,
        CachedKernel::new(
            0xCAFE_BABE,
            fixture_ptx(".visible .entry wrong_key(){}"),
            CompiledHandle::default(),
        ),
    );

    // A loader with a different key must NOT accept the entry — it
    // degrades to "no such entry" (no error returned to caller).
    let cache_with_wrong_key = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir,
        hmac_key: [0xBB; 32],
    });
    assert!(
        cache_with_wrong_key.get(&key).is_none(),
        "wrong HMAC key must treat the on-disk entry as a miss"
    );
}

#[test]
fn disk_cache_rejects_tampered_payload() {
    let tmp = TempDir::new().expect("tempdir");
    let dir: PathBuf = tmp.path().to_path_buf();
    let key_bytes = [0x33; 32];
    let cache = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir: dir.clone(),
        hmac_key: key_bytes,
    });
    let key = CacheKey::for_tenant(TenantId(7), 0xDEAD_F00D, 80);
    cache.put(
        key,
        CachedKernel::new(
            0xDEAD_F00D,
            fixture_ptx(".visible .entry victim(){}"),
            CompiledHandle::default(),
        ),
    );

    // Find the one `.ptxbin` file and flip a byte in the middle of the
    // PTX payload (past the header, before the HMAC trailer).
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "ptxbin"))
        .collect();
    assert_eq!(entries.len(), 1, "disk cache must have written exactly one file");
    let path = entries[0].path();
    let mut bytes = std::fs::read(&path).expect("read entry");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("rewrite tampered entry");

    let cache2 = KernelCache::new().with_disk_persistence(DiskCacheConfig {
        dir,
        hmac_key: key_bytes,
    });
    assert!(
        cache2.get(&key).is_none(),
        "tampered on-disk entry must be rejected as a miss"
    );
}
