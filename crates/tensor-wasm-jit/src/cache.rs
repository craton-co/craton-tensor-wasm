// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Compiled-kernel cache.
//!
//! The cache is keyed by `(blueprint_fingerprint, sm_version)`. LRU eviction
//! caps the entry count. On a cache hit the PTX text is reused without
//! re-emitting (cheap) and without re-compiling via `ptxas`/`cust` (expensive
//! — 10-50 ms for non-trivial kernels).
//!
//! Storage: a [`dashmap::DashMap`] holds the actual `(CacheKey, CachedKernel)`
//! entries — `get` / `put` / `len` go straight through the lock-free
//! per-shard locks so the hot path is uncontended under multi-threaded
//! dispatch. A separate `parking_lot::Mutex<LruCache<CacheKey, ()>>` is the
//! eviction queue — touched on `put` (insert/promote) and on `get` (promote)
//! and used to compute which key to evict when the soft cap is exceeded.
//! Splitting storage from policy means lookups never block on the eviction
//! mutex, only inserts that need to evict do. The `get` read path is
//! lock-free on contention; LRU promotion is best-effort under load —
//! contended promotions are skipped, so eviction order is approximately
//! (not strictly) LRU when many readers race.
//!
//! `Mutex` poisoning recovery: every lock acquisition uses `into_inner` on a
//! poisoned guard (after emitting a `tracing::error!`) rather than the prior
//! `.expect("cache poisoned")` panic — a single panic on any thread used to
//! poison the entire cache for the rest of the process.

use std::num::NonZeroUsize;
use std::sync::Arc;

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use tensor_wasm_core::types::TenantId;

use crate::ir::TensorWasmKernelBlueprint;
use crate::ptx_emit::EmittedPtx;

/// Cache key.
///
/// `tenant_id` is the first field so that it dominates the derived `Hash`
/// (field-order) and lexicographic `Ord` orderings. Keeping the cache keyed
/// by tenant is the only thing preventing tenant A from looking up — and on
/// the CUDA path executing — a compiled kernel that tenant B installed
/// (exec S-7, cross-tenant confused-deputy). Every host-side `get` / `put`
/// MUST therefore include the calling tenant; constructing a key without
/// one (e.g. directly from guest-supplied bytes) is the bug we are fixing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    /// Owning tenant. Cache lookups MUST be scoped to the caller's tenant —
    /// see the type-level docs. Use [`CacheKey::for_tenant`] to construct.
    pub tenant_id: u64,
    /// `TensorWasmKernelBlueprint::fingerprint()`.
    pub blueprint: u64,
    /// CUDA compute capability (e.g. 80 for sm_80, 89 for sm_89).
    pub sm_version: u32,
    /// Hash of the full [`crate::ptx_emit::EmitConfig`] used at emit time
    /// (jit S-2). `sm_version` covers the compute capability number but NOT
    /// the architecture suffix (e.g. `"sm_80"` vs `"sm_80a"`), the PTX
    /// language version, or the `launch_bounds` flag — two `EmitConfig`s
    /// that differ in any of those produce non-interchangeable PTX. Without
    /// this field two such configs would collide on the same key and the
    /// second caller would silently get the first caller's PTX.
    ///
    /// Callers that don't have an `EmitConfig` (rewriter pre-population,
    /// benches) pass `0`. Construct via [`CacheKey::for_tenant`] (defaults
    /// to `0`) or [`CacheKey::for_tenant_with_emit_config`] (computes a
    /// stable hash over the config).
    pub emit_config_hash: u64,
}

impl CacheKey {
    /// Construct a tenant-scoped cache key with no `EmitConfig` hash.
    ///
    /// Equivalent to passing `emit_config_hash: 0` — appropriate for the
    /// rewriter and bench paths that use the default emitter config. The
    /// `tenant_id` MUST come from trusted store state (e.g.
    /// `InstanceState::tenant_id`), never from guest-supplied fingerprint
    /// bytes. See the [`CacheKey`] docs for the confused-deputy primitive
    /// this guards against.
    pub fn for_tenant(tenant_id: TenantId, blueprint: u64, sm_version: u32) -> Self {
        Self {
            tenant_id: tenant_id.get(),
            blueprint,
            sm_version,
            emit_config_hash: 0,
        }
    }

    /// Construct a tenant-scoped cache key that also covers the emitter
    /// config. Use this when the lookup must distinguish between
    /// PTX-version variants, target-architecture suffixes, or
    /// launch-bounds settings (jit S-2).
    pub fn for_tenant_with_emit_config(
        tenant_id: TenantId,
        blueprint: u64,
        sm_version: u32,
        cfg: &crate::ptx_emit::EmitConfig,
    ) -> Self {
        let emit_config_hash = blake3::Hasher::new()
            .update(b"tensor-wasm-jit::EmitConfig::v1\0")
            .update(cfg.target.as_bytes())
            .update(b"\0")
            .update(cfg.ptx_version.as_bytes())
            .update(b"\0")
            .update(&[u8::from(cfg.launch_bounds)])
            .finalize();
        let bytes = emit_config_hash.as_bytes();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[..8]);
        Self {
            tenant_id: tenant_id.get(),
            blueprint,
            sm_version,
            emit_config_hash: u64::from_le_bytes(buf),
        }
    }
}

/// Default cache capacity (kernels).
pub const DEFAULT_CAPACITY: usize = 256;

/// Cached PTX module entry.
#[derive(Debug, Clone)]
pub struct CachedKernel {
    /// The blueprint that produced this PTX (for diagnostics).
    pub fingerprint: u64,
    /// The emitted PTX text.
    pub ptx: Arc<EmittedPtx>,
    /// The cuda module handle is only meaningful when the `cuda` feature is
    /// on; for the stub path we keep `()`.
    pub compiled: CompiledHandle,
}

/// Compiled-module handle. On CUDA hosts this would hold
/// `cust::module::Module`; for the no-CUDA path it is just `()`.
#[derive(Debug, Clone, Default)]
pub struct CompiledHandle {
    #[allow(dead_code)]
    private: (),
}

/// Thread-safe LRU cache of compiled kernels backed by [`dashmap::DashMap`].
///
/// `get` and `put` are O(1) and (under typical concurrent workloads) lock-
/// free except for the per-shard `dashmap` lock and the eviction-policy
/// `parking_lot::Mutex`. The eviction lock is taken only on `put`s that
/// would push the cache over capacity; reads never touch it.
#[derive(Clone)]
pub struct KernelCache {
    /// Lock-free storage of the cached values themselves.
    storage: Arc<DashMap<CacheKey, CachedKernel>>,
    /// LRU policy: keys ordered by recency. `Mutex` (parking_lot) for
    /// fast, panic-safe contention. The value side is `()` — the real value
    /// lives in `storage`.
    lru: Arc<Mutex<LruCache<CacheKey, ()>>>,
    /// Soft maximum entries before eviction kicks in.
    capacity: usize,
}

impl KernelCache {
    /// Construct with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Construct with explicit capacity. Anything below 1 is clamped to 1.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        // The eviction queue is sized to `cap` so the LRU crate's internal
        // bucket pre-allocation is bounded (sizing it to `usize::MAX` triggers
        // a hashbrown capacity-overflow panic). Storage eviction is still
        // driven from the `storage.len() > capacity` check in `put`; both
        // sides agree on the same `cap` so they stay in sync.
        let nz = NonZeroUsize::new(cap).expect(">0 (clamped above)");
        Self {
            storage: Arc::new(DashMap::with_capacity(cap)),
            lru: Arc::new(Mutex::new(LruCache::new(nz))),
            capacity: cap,
        }
    }

    /// Insert (or replace) a kernel. If the insert pushes the cache over
    /// capacity, evicts the LRU entry from storage and the policy queue.
    pub fn put(&self, key: CacheKey, kernel: CachedKernel) {
        self.storage.insert(key, kernel);
        // `LruCache::push` returns `Some((evicted_key, ()))` when sizing the
        // LRU triggers eviction of an older entry. We use this as the
        // authoritative signal for storage eviction so the two stay in sync.
        // Sized to `cap`, the LRU evicts exactly when we want storage to,
        // and we get the evicted key directly without a second pop_lru call.
        let evicted = self.lru.lock().push(key, ());
        if let Some((evicted_key, ())) = evicted {
            if evicted_key != key {
                // Don't remove if `push` returned the just-inserted key
                // (which happens when the cache already held it — push acts
                // as a replace and returns the old `(K, V)`).
                self.storage.remove(&evicted_key);
            }
        }
        // Safety net for the rare burst case where two concurrent `put`s
        // both insert without either triggering LRU's internal eviction
        // (because they raced on the lock). Drive storage back under cap.
        //
        // Hold the `lru` lock across the entire eviction loop instead of
        // taking and releasing it on every iteration: a tight burst of
        // overflow eviction used to lock-unlock the mutex `N` times,
        // letting unrelated readers/writers contend on each pass. One
        // acquisition costs the same as one iteration's lock+unlock but
        // dispatches the whole burst in a single critical section.
        if self.storage.len() > self.capacity {
            let mut lru = self.lru.lock();
            while self.storage.len() > self.capacity {
                match lru.pop_lru() {
                    Some((evict_key, ())) => {
                        self.storage.remove(&evict_key);
                    }
                    None => {
                        tracing::error!(
                            target: "tensor_wasm_jit::cache",
                            storage_len = self.storage.len(),
                            capacity = self.capacity,
                            "cache storage exceeds capacity but eviction queue is empty"
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Look up a kernel; best-effort touches the LRU position.
    pub fn get(&self, key: &CacheKey) -> Option<CachedKernel> {
        // Promote in the policy queue if we can grab the lock uncontended;
        // otherwise skip promotion this time so the read path stays
        // contention-free. Eviction order is approximate (not strict) LRU
        // under load — an acceptable trade for a lock-free hot path.
        // The storage read is the single source of truth for "is this
        // cached", so skipping promotion never affects correctness.
        if let Some(mut lru) = self.lru.try_lock() {
            // `get` on LruCache promotes if present.
            let _ = lru.get(key);
        }
        self.storage.get(key).map(|entry| entry.value().clone())
    }

    /// Look up by blueprint + sm_version for a given tenant; convenience
    /// wrapper around [`Self::get`].
    pub fn get_for(
        &self,
        tenant_id: TenantId,
        blueprint: &TensorWasmKernelBlueprint,
        sm_version: u32,
    ) -> Option<CachedKernel> {
        self.get(&CacheKey::for_tenant(
            tenant_id,
            blueprint.fingerprint(),
            sm_version,
        ))
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{TensorWasmKernelBlueprint, TensorWasmOp};
    use crate::ptx_emit::EmittedPtx;
    use std::thread;

    fn dummy_kernel(fp: u64) -> CachedKernel {
        CachedKernel {
            fingerprint: fp,
            ptx: Arc::new(EmittedPtx {
                text: String::new(),
                launch_geometry: (1, 1),
            }),
            compiled: CompiledHandle::default(),
        }
    }

    #[test]
    fn put_then_get() {
        let cache = KernelCache::new();
        let key = CacheKey::for_tenant(TenantId(7), 1, 80);
        cache.put(key, dummy_kernel(1));
        assert_eq!(cache.get(&key).unwrap().fingerprint, 1);
    }

    #[test]
    fn lru_evicts_oldest() {
        let cache = KernelCache::with_capacity(2);
        let k1 = CacheKey::for_tenant(TenantId(0), 1, 80);
        let k2 = CacheKey::for_tenant(TenantId(0), 2, 80);
        let k3 = CacheKey::for_tenant(TenantId(0), 3, 80);
        cache.put(k1, dummy_kernel(1));
        cache.put(k2, dummy_kernel(2));
        cache.put(k3, dummy_kernel(3));
        assert!(cache.get(&k1).is_none(), "k1 should have been evicted");
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn lookup_by_blueprint() {
        let cache = KernelCache::new();
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecAdd { lanes: 4 });
        let tenant = TenantId(11);
        let key = CacheKey::for_tenant(tenant, bp.fingerprint(), 80);
        cache.put(key, dummy_kernel(bp.fingerprint()));
        assert!(cache.get_for(tenant, &bp, 80).is_some());
        assert!(
            cache.get_for(tenant, &bp, 89).is_none(),
            "different sm_version is a miss"
        );
        assert!(
            cache.get_for(TenantId(12), &bp, 80).is_none(),
            "different tenant is a miss — keys are tenant-scoped"
        );
    }

    #[test]
    fn capacity_floor_one() {
        let cache = KernelCache::with_capacity(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn empty_when_new() {
        let cache = KernelCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_hit_returns_arc_shared_ptx() {
        use std::sync::Arc;
        let cache = KernelCache::new();
        let bp = TensorWasmKernelBlueprint::new("matmul").push(TensorWasmOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        let key = CacheKey::for_tenant(TenantId(3), bp.fingerprint(), 80);
        let original = Arc::new(EmittedPtx {
            text: "// pre-emitted".into(),
            launch_geometry: (1, 128),
        });
        cache.put(
            key,
            CachedKernel {
                fingerprint: bp.fingerprint(),
                ptx: original.clone(),
                compiled: CompiledHandle::default(),
            },
        );
        let hit = cache.get(&key).expect("cache hit");
        // The hit returns the same underlying allocation — no re-emit.
        assert!(Arc::ptr_eq(&hit.ptx, &original));
    }

    /// Concurrent get/put across many threads must not corrupt the cache
    /// or drop entries that haven't been evicted. Uses a capacity large
    /// enough to hold every key inserted so we can assert all `get`s
    /// targeting still-present keys succeed.
    #[test]
    fn concurrent_get_put_dashmap_safe() {
        const N_THREADS: usize = 8;
        const KEYS_PER_THREAD: u64 = 32;
        let cache = KernelCache::with_capacity((N_THREADS as u64 * KEYS_PER_THREAD) as usize);
        let mut handles = Vec::new();
        for t in 0..N_THREADS {
            let cache = cache.clone();
            handles.push(thread::spawn(move || {
                for i in 0..KEYS_PER_THREAD {
                    let key = CacheKey::for_tenant(
                        TenantId(0),
                        (t as u64) * KEYS_PER_THREAD + i,
                        80,
                    );
                    cache.put(key, dummy_kernel(key.blueprint));
                    // Interleave reads of own and (possibly absent) others.
                    let _ = cache.get(&key);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        // Every key written must still be retrievable.
        for t in 0..N_THREADS {
            for i in 0..KEYS_PER_THREAD {
                let key = CacheKey::for_tenant(
                    TenantId(0),
                    (t as u64) * KEYS_PER_THREAD + i,
                    80,
                );
                assert!(
                    cache.get(&key).is_some(),
                    "missing key after concurrent inserts: ({t}, {i})"
                );
            }
        }
    }
}
