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
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use tensor_wasm_artifacts::{ArtifactError, ArtifactStore, ContentHash, DiskArtifactStore};
use tensor_wasm_core::types::TenantId;
use zeroize::Zeroizing;

use crate::ir::TensorWasmKernelBlueprint;
use crate::ptx_emit::EmittedPtx;
#[cfg(feature = "kernel-registry")]
use crate::registry::{BlueprintResolver, KernelRegistry};

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
    ///
    /// Cost note: each call builds a fresh `blake3::Hasher` and finalises
    /// over a handful of bytes. The amortised wall cost is ~µs on a
    /// modern x86-64 box — negligible compared to a kernel dispatch but
    /// non-zero on the hot path. Callers that resolve the same
    /// `EmitConfig` for every lookup (the typical pattern: emit-config is
    /// pinned at instance-spawn time) should hash it once at spawn and
    /// reuse the [`CacheKey`] rather than re-deriving it for every
    /// dispatch. The hasher itself is intentionally inline here (not
    /// memoised) so the function stays pure and `Send`-friendly for the
    /// rewriter's `rayon::par_iter` callers.
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
///
/// Memory-ceiling note: each entry holds an `Arc<EmittedPtx>` whose `text`
/// is the emitted PTX string. Typical kernels emit ~5-15 KB of PTX (a
/// vector-add lands around 2 KB; a small fused matmul around 12 KB), so at
/// 256 entries the steady-state L1 footprint is on the order of ~2.5 MB
/// (~10 KB PTX × 256) plus the per-entry `BLAKE3` hash (32 B) and the
/// LRU policy queue (one `CacheKey` per entry, 24 B). Hostile or
/// pathological blueprints could push individual entries into the
/// multi-MB range — a deliberately unrolled blueprint emitting 10 MB of
/// PTX would push the 256-slot cache to ~2.5 GB. Operators expecting
/// adversarial workloads should clamp this via
/// [`KernelCache::with_capacity`] (lower) and pair it with the on-disk
/// L2 cache so cold lookups still hit a persisted path. The cache does
/// NOT enforce a per-entry byte limit; the count cap is the only knob.
pub const DEFAULT_CAPACITY: usize = 256;

/// Construction-time configuration for [`KernelCache`].
///
/// Holds the small set of policy knobs the cache supports today: capacity
/// (count cap) and whether to recompute the per-entry BLAKE3 integrity
/// hash on every `get`. Future knobs (per-byte cap, eviction policy
/// choice) will land here without breaking the existing
/// [`KernelCache::with_capacity`] / [`KernelCache::with_disk_persistence`]
/// shorthands — those construct an equivalent `KernelCacheConfig` under
/// the hood.
///
/// Construct via [`KernelCacheConfig::default`] (which mirrors the
/// historical defaults — capacity [`DEFAULT_CAPACITY`], verify-on-get
/// `true`) and refine with the `with_*` builders, then hand the config
/// to [`KernelCache::with_config`].
#[derive(Clone)]
pub struct KernelCacheConfig {
    /// Soft maximum L1 entry count. Clamped to `>= 1` inside the cache.
    pub capacity: usize,
    /// When `true` (the default), [`KernelCache::get`] recomputes a
    /// BLAKE3 over the cached `ptx.text` on every L1 hit and compares
    /// the result against the entry's stored `integrity_hash` (jit S-3
    /// in-mem poisoning defence). When `false`, the recompute is skipped
    /// — the cache still refuses entries whose stored hash is all-zero
    /// (the construction signal for "built without [`CachedKernel::new`]")
    /// as defence-in-depth.
    ///
    /// The recompute costs ~10 µs over a typical multi-KB PTX blob;
    /// skipping it shaves that off every L1 hit at the cost of widening
    /// the in-memory poisoning window from "one `get` call" to "the
    /// lifetime of the entry in L1". Operators on a high-QPS path with
    /// multi-MB PTX where the recompute dominates can opt out, but the
    /// safe default is verify-on-get.
    pub verify_on_get: bool,

    #[cfg(feature = "kernel-registry")]
    /// Optional registry consulted on L1+L2 miss. v0.4 path: caller resolves
    /// (tenant, blueprint, sm_version) → (name, version) via an external
    /// lookup, then `KernelCache::get_with_registry_fallback` consults the
    /// registry by that pair. v0.3.8 ships a `resolve_by_blueprint_hint`
    /// trait method on the cache config for the resolver step; the
    /// in-memory test impl resolves blueprint fingerprint → name@version
    /// directly via a `HashMap`.
    pub registry: Option<Arc<dyn KernelRegistry>>,
}

impl Default for KernelCacheConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            verify_on_get: true,
            #[cfg(feature = "kernel-registry")]
            registry: None,
        }
    }
}

impl KernelCacheConfig {
    /// Override the L1 entry count cap. Clamped to `>= 1` at cache
    /// construction time; values below 1 are silently raised.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Toggle the per-`get` BLAKE3 recompute. Default `true`; setting
    /// `false` is the high-QPS opt-out. See the field-level docs on
    /// [`Self::verify_on_get`] for the threat-model trade-off.
    #[must_use]
    pub fn with_verify_on_get(mut self, on: bool) -> Self {
        self.verify_on_get = on;
        self
    }

    /// Attach a [`KernelRegistry`] as the L3 fallback consulted by
    /// [`KernelCache::get_with_registry_fallback`] on an L1+L2 miss.
    /// Default is `None` (registry path disabled). See the field-level
    /// docs on [`Self::registry`] for the resolution contract.
    #[cfg(feature = "kernel-registry")]
    #[must_use]
    pub fn with_registry(mut self, reg: Arc<dyn KernelRegistry>) -> Self {
        self.registry = Some(reg);
        self
    }
}

// Manual `Debug` impl: `Arc<dyn KernelRegistry>` does not implement
// `Debug` (the trait deliberately does not require it so embedder
// backends like `InMemoryRegistry` — whose interior `Mutex<HashMap>`
// is awkward to debug-format — stay easy to write). Render the
// registry field as a presence-only marker so `{:?}` on a cache
// config still surfaces whether the L3 path is wired without forcing
// every registry impl to derive `Debug`.
impl std::fmt::Debug for KernelCacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("KernelCacheConfig");
        d.field("capacity", &self.capacity);
        d.field("verify_on_get", &self.verify_on_get);
        #[cfg(feature = "kernel-registry")]
        d.field(
            "registry",
            &self.registry.as_ref().map(|_| "<dyn KernelRegistry>"),
        );
        d.finish()
    }
}

/// Cached PTX module entry.
///
/// `integrity_hash` is a BLAKE3 over `ptx.text` computed at construction
/// time and re-verified on every [`KernelCache::get`] (jit S-3). It defends
/// against in-memory poisoning by a sibling holder of a mutable reference
/// and forms the basis of the integrity tag stored in the on-disk
/// persistence layer (see [`DiskCacheConfig`]). The field is `pub` so
/// downstream tests and observability can inspect the recorded hash, but
/// construction via [`CachedKernel::new`] is the only path that produces a
/// correct-by-construction value — `KernelCache::put` calls `verify`
/// before accepting the entry, so a hand-crafted `CachedKernel` with a
/// mismatched hash will be rejected.
#[derive(Debug, Clone)]
pub struct CachedKernel {
    /// The blueprint that produced this PTX (for diagnostics).
    pub fingerprint: u64,
    /// The emitted PTX text.
    pub ptx: Arc<EmittedPtx>,
    /// The cuda module handle is only meaningful when the `cuda` feature is
    /// on; for the stub path we keep `()`.
    pub compiled: CompiledHandle,
    /// BLAKE3 over `ptx.text` (jit S-3). Recomputed and compared on every
    /// `get`. See the type-level doc for the threat model.
    ///
    /// The field is `pub` for backward compatibility with v0.1 struct-literal
    /// callers and so the forged-blob regression tests can hand-craft a
    /// `CachedKernel` with a deliberately wrong hash and assert
    /// [`KernelCache::put`] rejects it. It is hidden from generated docs and
    /// excluded from the stable surface; prefer [`Self::integrity_hash`] for
    /// read access and [`Self::new`] for construction.
    #[doc(hidden)]
    pub integrity_hash: [u8; 32],
}

impl CachedKernel {
    /// Borrow the stored BLAKE3 integrity hash over `ptx.text`. Prefer this
    /// over reaching for the (hidden) field directly — the field is
    /// retained as `pub` for source compatibility only and may be sealed
    /// to `pub(crate)` in a future minor release.
    #[must_use]
    pub fn integrity_hash(&self) -> &[u8; 32] {
        &self.integrity_hash
    }

    /// Construct a `CachedKernel`, computing `integrity_hash` from
    /// `ptx.text`. This is the only way to obtain a correct-by-
    /// construction value; the older struct-literal pattern still works
    /// (the field is `pub` for backward compatibility) but the literal
    /// must provide a matching hash or `KernelCache::put` will reject it.
    pub fn new(fingerprint: u64, ptx: Arc<EmittedPtx>, compiled: CompiledHandle) -> Self {
        let h = blake3::hash(ptx.text.as_bytes());
        Self {
            fingerprint,
            ptx,
            compiled,
            integrity_hash: *h.as_bytes(),
        }
    }

    /// Re-compute the integrity hash and compare it against the stored
    /// value. Returns `true` iff the PTX text has not been tampered with
    /// since [`Self::new`] ran.
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        let h = blake3::hash(self.ptx.text.as_bytes());
        h.as_bytes() == &self.integrity_hash
    }
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
    ///
    /// T20 perf: values are held as `Arc<CachedKernel>` so a cache hit only
    /// bumps the strong-count refcount instead of cloning the wrapper
    /// (32-byte integrity hash + fingerprint + an inner `Arc<EmittedPtx>`
    /// refcount bump). The inner PTX text was already `Arc`-shared; this
    /// extension closes the matching gap on the outer wrapper.
    storage: Arc<DashMap<CacheKey, Arc<CachedKernel>>>,
    /// LRU policy: keys ordered by recency. `Mutex` (parking_lot) for
    /// fast, panic-safe contention. The value side is `()` — the real value
    /// lives in `storage`.
    lru: Arc<Mutex<LruCache<CacheKey, ()>>>,
    /// Construction-time policy bag (capacity, verify-on-get). Held by
    /// value (cheap to clone, `Copy`-ish payload) so the `get` hot path
    /// can read `verify_on_get` without an extra `Arc` deref.
    config: KernelCacheConfig,
    /// Optional L2 on-disk cache. When present (configured via
    /// [`KernelCache::with_disk_persistence`]), `put` writes each entry to
    /// disk under an HMAC-keyed integrity tag, and `get` falls through to
    /// disk on an L1 miss — verifying the HMAC before deserialising. See
    /// [`DiskCacheConfig`] for the threat model that motivates the design
    /// (jit S-3).
    disk: Option<Arc<DiskCache>>,
    /// Cumulative count of `get` calls that skipped the BLAKE3 recompute
    /// because [`KernelCacheConfig::verify_on_get`] was `false`.
    /// Exposed via [`Self::verify_skipped_total`] so operators can wire
    /// it to the Prometheus counter
    /// `tensor_wasm_jit_cache_verify_skipped_total`. Wrapped in an `Arc`
    /// so clones of `KernelCache` share the same counter (the storage
    /// and LRU policy are likewise `Arc`-shared).
    verify_skipped_total: Arc<AtomicU64>,
    /// Cumulative count of `get` calls that returned an entry (L1 hit, L2
    /// disk hit, or L3 registry hit). Exposed via [`Self::cache_hits_total`]
    /// for the Prometheus counter `tensor_wasm_jit_cache_hits_total`.
    /// `Arc`-shared so cache clones agree on the count.
    cache_hits_total: Arc<AtomicU64>,
    /// Cumulative count of `get` calls that returned `None` (full miss after
    /// L1, L2, and any registry fallback). Exposed via
    /// [`Self::cache_misses_total`] for the Prometheus counter
    /// `tensor_wasm_jit_cache_misses_total`. `Arc`-shared so cache clones
    /// agree on the count.
    cache_misses_total: Arc<AtomicU64>,
}

impl KernelCache {
    /// Construct with default capacity.
    pub fn new() -> Self {
        Self::with_config(KernelCacheConfig::default())
    }

    /// Construct with explicit capacity. Anything below 1 is clamped to 1.
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_config(KernelCacheConfig::default().with_capacity(cap))
    }

    /// Construct from a full [`KernelCacheConfig`]. Anything below 1 in
    /// `config.capacity` is clamped to 1.
    pub fn with_config(mut config: KernelCacheConfig) -> Self {
        config.capacity = config.capacity.max(1);
        // The eviction queue is sized to `cap` so the LRU crate's internal
        // bucket pre-allocation is bounded (sizing it to `usize::MAX` triggers
        // a hashbrown capacity-overflow panic). Storage eviction is still
        // driven from the `storage.len() > capacity` check in `put`; both
        // sides agree on the same `cap` so they stay in sync.
        let nz = NonZeroUsize::new(config.capacity).expect(">0 (clamped above)");
        Self {
            storage: Arc::new(DashMap::with_capacity(config.capacity)),
            lru: Arc::new(Mutex::new(LruCache::new(nz))),
            config,
            disk: None,
            verify_skipped_total: Arc::new(AtomicU64::new(0)),
            cache_hits_total: Arc::new(AtomicU64::new(0)),
            cache_misses_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Enable the on-disk L2 cache, persisting entries to `cfg.dir` under
    /// an HMAC-keyed integrity tag (jit S-3).
    ///
    /// Construct via:
    /// ```
    /// # // Hidden doctest stub — real deployments must read the key from
    /// # // an out-of-process secret store (Vault, KMS, sealed file under
    /// # // mode 0400, etc.) and never embed it in source. This stub lets
    /// # // the doctest compile without shipping a fake key literal that
    /// # // operators might copy-paste verbatim.
    /// # fn load_hmac_key_from_secret_store() -> Result<[u8; 32], Box<dyn std::error::Error>> {
    /// #     Ok([0u8; 32])
    /// # }
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::PathBuf;
    /// use tensor_wasm_jit::cache::{KernelCache, DiskCacheConfig};
    /// let cache = KernelCache::new().with_disk_persistence(DiskCacheConfig {
    ///     dir: PathBuf::from("/var/cache/tensor-wasm/kernels"),
    ///     hmac_key: load_hmac_key_from_secret_store()?, /* loaded from secrets at startup */
    /// });
    /// # let _ = cache;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// jit S-3 (T13): the example deliberately routes the key through a
    /// `load_hmac_key_from_secret_store()` stub rather than an inline
    /// literal — operators have a habit of copy-pasting rustdoc examples
    /// verbatim, and an embedded `[0xAB; 32]` (or similar) would survive
    /// into production deployments as a fixed, attacker-known key.
    ///
    /// The directory is created lazily on the first `put`. The HMAC key
    /// MUST be stable across process restarts AND treated as a server-
    /// side secret — possession of the key lets an attacker forge cache
    /// entries that the loader will accept as authentic (and that would
    /// then be handed to `cust::module::Module::from_ptx` as trusted GPU
    /// code).
    #[must_use]
    pub fn with_disk_persistence(mut self, cfg: DiskCacheConfig) -> Self {
        self.disk = Some(Arc::new(DiskCache::new(cfg)));
        self
    }

    /// Insert (or replace) a kernel. If the insert pushes the cache over
    /// capacity, evicts the LRU entry from storage and the policy queue.
    ///
    /// jit S-3: the kernel's `integrity_hash` is recomputed and compared
    /// against the stored hash; a mismatch is treated as a programmer
    /// error (`CachedKernel` constructed via struct-literal with a wrong
    /// hash), logged at `error!`, and the entry is dropped rather than
    /// admitted. Use [`CachedKernel::new`] for the correct-by-
    /// construction path. The on-disk L2 also writes the entry when
    /// configured.
    pub fn put(&self, key: CacheKey, kernel: CachedKernel) {
        if !kernel.verify_integrity() {
            tracing::error!(
                target: "tensor_wasm_jit::cache",
                fingerprint = kernel.fingerprint,
                tenant = key.tenant_id,
                "refusing to cache kernel whose integrity hash does not match \
                 its PTX text -- likely a struct-literal construction with a \
                 stale hash; use CachedKernel::new"
            );
            return;
        }
        if let Some(disk) = &self.disk {
            if let Err(e) = disk.put(&key, &kernel) {
                tracing::warn!(
                    target: "tensor_wasm_jit::cache",
                    fingerprint = kernel.fingerprint,
                    error = %e,
                    "disk-cache put failed; entry remains in L1 only"
                );
            }
        }
        // T20 perf: storage holds `Arc<CachedKernel>` so cache hits return a
        // refcount bump rather than a wrapper-clone.
        self.storage.insert(key, Arc::new(kernel));
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
        if self.storage.len() > self.config.capacity {
            let mut lru = self.lru.lock();
            while self.storage.len() > self.config.capacity {
                match lru.pop_lru() {
                    Some((evict_key, ())) => {
                        self.storage.remove(&evict_key);
                    }
                    None => {
                        tracing::error!(
                            target: "tensor_wasm_jit::cache",
                            storage_len = self.storage.len(),
                            capacity = self.config.capacity,
                            "cache storage exceeds capacity but eviction queue is empty"
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Look up a kernel; best-effort touches the LRU position.
    ///
    /// T20 perf: returns `Option<Arc<CachedKernel>>` rather than the
    /// previous `Option<CachedKernel>` so a cache hit shares the wrapper
    /// allocation via refcount bump instead of cloning the 32-byte
    /// integrity hash + fingerprint + inner-`Arc` refcount bump. The
    /// inner PTX text was already shared; this closes the matching
    /// gap on the outer wrapper.
    pub fn get(&self, key: &CacheKey) -> Option<Arc<CachedKernel>> {
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
        if let Some(entry) = self.storage.get(key) {
            // T20 perf: clone the Arc (refcount bump) rather than the
            // inner `CachedKernel` value.
            let kernel: Arc<CachedKernel> = Arc::clone(entry.value());
            drop(entry); // release shard lock before the hash recompute
                         // jit S-3: verify the in-memory entry hasn't been tampered
                         // with since `put`. A mismatch should be impossible (the
                         // `CachedKernel` is owned by the cache and never handed out
                         // mutably) but failing closed costs ~µs over the public
                         // PTX bytes and definitively closes the in-mem poisoning
                         // path the audit flagged.
                         //
                         // The recompute can be opted out of via
                         // [`KernelCacheConfig::verify_on_get`] for high-QPS callers
                         // where the ~10 µs BLAKE3 cost over multi-MB PTX dominates.
                         // Even on the opt-out path we keep a cheap defence-in-depth
                         // check: refuse entries whose `integrity_hash` is all-zero,
                         // because that is the signature of a `CachedKernel`
                         // constructed via the `#[doc(hidden)]` struct-literal path
                         // without [`CachedKernel::new`] having computed a real hash.
                         // A real BLAKE3 over any PTX text is overwhelmingly unlikely
                         // to collide with the zero hash (probability `2^-256`), so
                         // the rejection is unambiguous.
            if self.config.verify_on_get {
                if !kernel.verify_integrity() {
                    tracing::error!(
                        target: "tensor_wasm_jit::cache",
                        fingerprint = kernel.fingerprint,
                        tenant = key.tenant_id,
                        "L1 cache entry failed integrity verification on get; \
                         evicting and refusing to return it"
                    );
                    self.storage.remove(key);
                    self.cache_misses_total.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            } else {
                // Defence-in-depth on the opt-out path: reject the
                // "constructed-without-`new`" signal (all-zero hash) so
                // hand-crafted `CachedKernel`s with a zeroed hash cannot
                // slip through. Counter increment records that the user
                // chose to skip the full recompute on this hit.
                self.verify_skipped_total.fetch_add(1, Ordering::Relaxed);
                if kernel.integrity_hash == [0u8; 32] {
                    tracing::error!(
                        target: "tensor_wasm_jit::cache",
                        fingerprint = kernel.fingerprint,
                        tenant = key.tenant_id,
                        "L1 cache entry has zero integrity_hash on verify-skip get; \
                         likely a struct-literal CachedKernel built without \
                         CachedKernel::new — evicting and refusing to return it"
                    );
                    self.storage.remove(key);
                    self.cache_misses_total.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
            self.cache_hits_total.fetch_add(1, Ordering::Relaxed);
            return Some(kernel);
        }
        // jit S-3 + audit P-4: L1 miss falls through to the optional L2
        // on-disk cache. The disk path HMAC-verifies the entry before
        // deserialising, so a tampered file on disk is rejected with the
        // same "no such entry" outcome as a real miss.
        if let Some(disk) = &self.disk {
            match disk.get(key) {
                Ok(Some(kernel)) => {
                    // Promote the disk hit into L1 so subsequent lookups
                    // stay on the fast path. Storage owns an `Arc<CachedKernel>`;
                    // wrap once here and clone the Arc for the return value
                    // so the caller and the cache share the allocation.
                    let arc = Arc::new(kernel);
                    self.storage.insert(*key, Arc::clone(&arc));
                    let _ = self.lru.lock().push(*key, ());
                    self.cache_hits_total.fetch_add(1, Ordering::Relaxed);
                    return Some(arc);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        target: "tensor_wasm_jit::cache",
                        tenant = key.tenant_id,
                        error = %e,
                        "disk-cache get failed; treating as miss"
                    );
                }
            }
        }
        self.cache_misses_total.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Look up by blueprint + sm_version for a given tenant; convenience
    /// wrapper around [`Self::get`].
    ///
    /// T20 perf: returns `Option<Arc<CachedKernel>>` to mirror
    /// [`Self::get`]; callers that only need to peek at fields go
    /// through auto-deref unchanged.
    pub fn get_for(
        &self,
        tenant_id: TenantId,
        blueprint: &TensorWasmKernelBlueprint,
        sm_version: u32,
    ) -> Option<Arc<CachedKernel>> {
        self.get(&CacheKey::for_tenant(
            tenant_id,
            blueprint.fingerprint(),
            sm_version,
        ))
    }

    /// L1 → L2 → L3(registry) resolution path. v0.3.8 scaffold: invokes the
    /// registered resolver + registry on every miss; v0.4 may add a
    /// resolver-level cache to amortise the (blueprint → name@version)
    /// translation.
    #[cfg(feature = "kernel-registry")]
    pub fn get_with_registry_fallback(
        &self,
        key: &CacheKey,
        resolver: &dyn BlueprintResolver,
    ) -> Option<Arc<CachedKernel>> {
        // L1 + L2 — `get` now already returns `Arc<CachedKernel>` (T20).
        if let Some(hit) = self.get(key) {
            return Some(hit);
        }
        // L3
        let registry = self.config.registry.as_ref()?;
        let (name, version) = resolver.resolve(key.blueprint, key.sm_version)?;
        let entry = registry.get(&name, &version).ok()?;
        // Promote into L1 via the standard put path (which re-checks
        // integrity) so subsequent calls hit fast.
        let (manifest, ptx_text) = (&entry.0, &entry.1);
        let emitted = crate::ptx_emit::EmittedPtx {
            text: ptx_text.clone(),
            launch_geometry: (0, 0), // v0.4: extend KernelManifest to carry geometry
        };
        let cached = CachedKernel::new(
            manifest.digest_as_u64(), // see Step 4 — add helper
            Arc::new(emitted),
            CompiledHandle::default(),
        );
        // Best-effort L1 promote; `put` is infallible-by-design (any
        // integrity-mismatch case logs + drops the entry rather than
        // returning an error), so there is nothing to propagate even if
        // the registry-derived `cached` were somehow rejected — the
        // caller still gets the verified `Arc<CachedKernel>` from this
        // call, the next call simply pays another L3 round-trip.
        //
        // T20 perf: `put` wraps the kernel in an `Arc` internally; we
        // hand it a clone of the value rather than threading the Arc
        // through to keep `put`'s public signature stable. The returned
        // `Arc<CachedKernel>` to the caller is a separate allocation
        // from the L1 copy — both share the inner `Arc<EmittedPtx>`,
        // so PTX text is not duplicated.
        self.put(*key, cached.clone());
        Some(Arc::new(cached))
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
        self.config.capacity
    }

    /// Borrow the construction-time [`KernelCacheConfig`]. Useful for
    /// tests and diagnostic endpoints that want to surface whether
    /// `verify_on_get` is on for this cache instance.
    pub fn config(&self) -> &KernelCacheConfig {
        &self.config
    }

    /// Cumulative count of L1 `get` hits that skipped the BLAKE3
    /// integrity recompute because [`KernelCacheConfig::verify_on_get`]
    /// is `false`. Surface this on the Prometheus counter
    /// `tensor_wasm_jit_cache_verify_skipped_total` so operators can see
    /// how often the cache is trusting an L1 entry without re-hashing
    /// the PTX. The counter is always present (returns `0` when
    /// `verify_on_get` is on) so dashboards can scrape it unconditionally.
    pub fn verify_skipped_total(&self) -> u64 {
        self.verify_skipped_total.load(Ordering::Relaxed)
    }

    /// Cumulative L1/L2/L3 cache hit count. Incremented once per `get`
    /// call that returns `Some`. Surface on the Prometheus counter
    /// `tensor_wasm_jit_cache_hits_total` so operators can compute the
    /// hit ratio against [`Self::cache_misses_total`]. (T20 perf.)
    pub fn cache_hits_total(&self) -> u64 {
        self.cache_hits_total.load(Ordering::Relaxed)
    }

    /// Cumulative cache miss count. Incremented once per `get` call that
    /// returns `None` — including the rare path where an L1 entry failed
    /// integrity verification and was evicted. Surface on the Prometheus
    /// counter `tensor_wasm_jit_cache_misses_total`. (T20 perf.)
    pub fn cache_misses_total(&self) -> u64 {
        self.cache_misses_total.load(Ordering::Relaxed)
    }

    /// Test-only insert that skips the `put`-side integrity check.
    ///
    /// `put` rejects any `CachedKernel` whose stored `integrity_hash`
    /// does not match a fresh BLAKE3 over its `ptx.text` (jit S-3).
    /// That is the correct production behaviour, but the
    /// `verify_on_get=false` regression test in
    /// `tests/cache_verify_opt_out.rs` needs to install a hand-crafted
    /// zero-hash entry to confirm the opt-out path still rejects it on
    /// `get`. This entry-point exists for that test only — it is
    /// `#[doc(hidden)]` (excluded from generated docs and rustdoc search)
    /// and named with a `__test_only_` prefix to broadcast "do NOT call
    /// this from production code". It is intentionally not behind a
    /// `#[cfg(test)]` or feature gate because integration tests under
    /// `crates/.../tests/` compile as external consumers of the library
    /// and therefore cannot see `cfg(test)` items.
    ///
    /// Production code MUST go through [`Self::put`].
    #[doc(hidden)]
    pub fn __test_only_insert_unchecked(&self, key: CacheKey, kernel: CachedKernel) {
        // T20 perf: storage holds `Arc<CachedKernel>`; wrap on insert.
        self.storage.insert(key, Arc::new(kernel));
        let _ = self.lru.lock().push(key, ());
    }
}

// ---------------------------------------------------------------------------
// Disk persistence (audit P-4 + jit S-3)
// ---------------------------------------------------------------------------

/// Configuration for the on-disk L2 cache.
///
/// The cache directory must be writable by the runtime user and SHOULD be
/// owned exclusively by that user (mode 0700 on Unix) — operators on a
/// hardened deployment can additionally `chattr +i` files after first
/// write so a parallel attacker process cannot tamper with cached
/// entries even with the same UID.
///
/// `hmac_key` is the secret that gates load: the writer HMACs each
/// persisted entry with the key, and the reader rejects any file whose
/// recomputed HMAC does not match. Without the key (or with a different
/// key) the loader treats every existing entry as a miss, so rotating
/// the key invalidates the disk cache without requiring an `rm -rf`.
///
/// The caller-supplied `hmac_key` field is plain `[u8; 32]` for ergonomic
/// construction; once handed to [`KernelCache::with_disk_persistence`] the
/// bytes are copied into a private `Zeroizing<[u8; 32]>` inside
/// [`DiskCache`] which wipes them on drop. The caller's own copy is the
/// caller's responsibility (e.g. construct via `Zeroizing::new` upstream
/// and let it drop after the call).
///
/// jit S-3 hardening (T13): `Debug` is implemented manually so the
/// `hmac_key` bytes are redacted in any `{:?}` formatting, panic message,
/// or `tracing` field expansion — the derived `Debug` would have dumped
/// the raw 32-byte array. `Drop` zeroizes `hmac_key` on drop so the
/// construction-time copy does not survive in freed memory.
#[derive(Clone)]
pub struct DiskCacheConfig {
    /// Directory where the L2 cache files live. Created lazily on first
    /// `put`.
    pub dir: PathBuf,
    /// 32-byte HMAC-keyed-BLAKE3 key. MUST be process-stable across the
    /// cache's lifetime and SHOULD be treated as a server-side secret.
    /// The long-lived copy held by the cache is zeroized on drop; this
    /// field is the construction-time hand-off only.
    ///
    /// jit S-3 (T13): redacted in the manual [`std::fmt::Debug`] impl
    /// below and zeroized in [`Drop`] so the construction-time copy
    /// does not linger in freed memory after the value is moved into
    /// the long-lived [`DiskCache`].
    pub hmac_key: [u8; 32],
}

// Manual `Debug` so `hmac_key` never appears verbatim in formatted output
// (jit S-3 T13). Any `{:?}` print, panic message, or `tracing::error!`
// field expansion that includes a `DiskCacheConfig` would otherwise dump
// the raw key bytes into the log stream — which is the exact server-side
// secret the disk cache is supposed to protect.
impl std::fmt::Debug for DiskCacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCacheConfig")
            .field("dir", &self.dir)
            .field("hmac_key", &"<redacted 32 bytes>")
            .finish()
    }
}

// Zeroize the in-place HMAC key bytes when the config is dropped so the
// construction-time copy does not survive in freed memory once
// [`KernelCache::with_disk_persistence`] has consumed the value (jit S-3
// T13). The long-lived copy in [`DiskCache`] already lives inside
// `Zeroizing<[u8; 32]>`; this `Drop` plugs the matching gap for the
// caller-side hand-off struct.
impl Drop for DiskCacheConfig {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.hmac_key.zeroize();
    }
}

/// Disk-backed L2 cache.
///
/// ## T30 layering (current)
///
/// Stores each kernel as two cooperating on-disk artefacts under
/// [`DiskCacheConfig::dir`]:
///
/// 1. A *sidecar* file at [`Self::path_for`] (`*.ptxbin`) containing a
///    fixed-size record that maps the [`CacheKey`] to the
///    [`ContentHash`] of the underlying blob (16-byte magic
///    [`SIDECAR_MAGIC_V1`] + 32-byte content hash).
/// 2. A *blob* file under the
///    [`tensor_wasm_artifacts::DiskArtifactStore`] layout
///    (`*.<key-fp>.bin`) holding the streaming-encoded HMAC-SHA256 +
///    zstd envelope around the V2 kernel-manifest framing.
///
/// The V2 kernel-manifest framing (16-byte `TWJIT-KRNL-v2\0\0\0` magic +
/// length-prefixed body: blueprint, sm_version, grid_x, block_x,
/// ptx_len, ptx bytes) is built ABOVE the artifact store: a `put` first
/// serialises the kernel into a V2 envelope `Vec<u8>`, then hands that
/// to [`DiskArtifactStore::put`] which streams it through an HMAC-tee +
/// zstd encoder onto disk. A `get` reverses the layering: look up the
/// sidecar to get the [`ContentHash`], call [`DiskArtifactStore::get`]
/// which streaming-verifies the outer HMAC + decompresses, then parse
/// the inner V2 envelope to recover the kernel.
///
/// ## Why two files
///
/// [`DiskArtifactStore`] is content-addressed: lookups go through a
/// [`ContentHash`] derived from the payload bytes, not through a caller
/// key. The kernel cache needs lookups by tenant-scoped [`CacheKey`],
/// so the sidecar maps `CacheKey -> ContentHash` while the blob owns
/// the bytes. This preserves the streaming HMAC + zstd properties of
/// the store (T22) without baking the cache-key shape into the store
/// itself. Filenames stay partitioned by HMAC-key fingerprint on both
/// sides — the sidecar via `path_for`'s `{key_prefix}-…` prefix, the
/// blob via [`DiskArtifactStore`]'s own `{hash}.{key_fp}.bin` shape —
/// so two stores in the same directory under different keys cannot
/// race on the same path.
///
/// ## V2 envelope (current writer, body inside the artifact store)
///
/// ```text
/// [0..16)   magic = "TWJIT-KRNL-v2\0\0"
/// [16..24)  blueprint fingerprint (u64 LE)
/// [24..28)  sm_version (u32 LE)
/// [28..32)  launch_geometry.grid_x (u32 LE)
/// [32..36)  launch_geometry.block_x (u32 LE)
/// [36..44)  ptx length (u64 LE)
/// [44..44+ptx_len)  PTX text (UTF-8, NOT null-terminated)
/// ```
///
/// The HMAC trailer that pre-T30 V2 sat at the end of the file is gone
/// from this layer — the artifact store provides streaming HMAC over
/// the entire envelope (T22), so a second per-envelope MAC would be
/// redundant.
///
/// ## V1 magic (read-only)
///
/// Pre-T30 V1/V2 files (magic `"TWJIT-KRNL-v1\0\0\0"` or
/// `"TWJIT-KRNL-v2\0\0\0"` at offset 0 of a `.ptxbin` file written by
/// the legacy raw-file writer) sit at a different path shape than the
/// new T30 sidecar and are silently invisible to the new reader: they
/// no longer occupy the post-T30 sidecar path, so a fresh `get` of
/// the corresponding key returns a clean miss and the next `put` rewrites
/// the entry as `(sidecar, blob)` under the unified store. This matches
/// the `cache.l2.miss.legacy_magic` behaviour of pre-T30 V1 detection
/// without any decoder support for the legacy raw layout.
struct DiskCache {
    /// Directory the cache writes to. Cloned out of the supplied
    /// [`DiskCacheConfig`] so we can drop the original config (and its
    /// plain `[u8; 32]` copy of the key) and keep only the zeroizing
    /// copy below as the long-lived owner.
    dir: PathBuf,
    /// 32-byte HMAC-keyed-BLAKE3 key, wiped on drop. `Zeroizing` Derefs
    /// to `[u8; 32]` so existing `&self.hmac_key[..]`-style call sites
    /// keep compiling. The artifact store holds an independent copy
    /// (also zeroising) — the duplication is intentional so the
    /// sidecar's path-prefix fingerprint and the store's own
    /// per-blob path-prefix fingerprint agree byte-for-byte without
    /// either side reaching into the other's private field.
    hmac_key: Zeroizing<[u8; 32]>,
    /// Underlying streaming content-addressed signed blob store.
    /// Holds the v2-envelope-wrapped kernel payloads. Wrapped in an
    /// `Arc` so the disk cache is cheaply clonable — `KernelCache`
    /// already holds the cache behind `Arc<DiskCache>`, this is the
    /// only field whose backend benefits from sharing.
    store: Arc<DiskArtifactStore>,
}

/// V2 magic for the inner kernel-manifest envelope wrapped inside each
/// artifact-store blob. Unchanged across the T30 migration so cross-
/// version compat is preserved: an older reader extracting this body
/// from any future archive can still parse it.
const DISK_CACHE_MAGIC_V2: &[u8; 16] = b"TWJIT-KRNL-v2\0\0\0";
/// V2 header: magic + fingerprint + sm_version + grid_x + block_x + ptx_len.
const DISK_CACHE_HEADER_LEN_V2: usize = 16 + 8 + 4 + 4 + 4 + 8;

/// 16-byte sidecar magic. Stamped at the head of every `*.ptxbin`
/// sidecar so the reader can tell a T30 sidecar apart from any legacy
/// pre-T30 raw-V2 file that happens to live under the same filename
/// scheme. Pre-T30 files start with `TWJIT-KRNL-v2\0\0\0`; T30 sidecars
/// start with this distinct magic, so a stale legacy file is treated
/// as a miss and the next `put` overwrites it.
const SIDECAR_MAGIC_V1: &[u8; 16] = b"TWJIT-IDX-v1\0\0\0\0";
/// Sidecar total length: 16-byte magic + 32-byte BLAKE3 content hash.
const SIDECAR_LEN_V1: usize = 16 + 32;

impl DiskCache {
    fn new(mut cfg: DiskCacheConfig) -> Self {
        // Move the key bytes into a `Zeroizing` newtype so the long-lived
        // copy is wiped on `DiskCache::drop`. We cannot partially move
        // fields out of `cfg` directly because `DiskCacheConfig`
        // implements `Drop` (jit S-3 T13) — the language forbids
        // partial moves out of a `Drop` type. Instead we `mem::take`
        // each field, leaving the source struct in a default-valued
        // state before its `Drop::drop` runs and zeroizes the (now
        // already-defaulted) `hmac_key` array a second time.
        let dir = std::mem::take(&mut cfg.dir);
        let key_bytes = std::mem::take(&mut cfg.hmac_key);
        // The artifact store gets its own copy of the same key so its
        // streaming HMAC matches the one the sidecar's path-prefix
        // fingerprint will agree with. Both copies are wrapped in
        // `Zeroizing` (the artifact store wraps internally) so neither
        // construction-time bytes linger after drop.
        let store = Arc::new(DiskArtifactStore::new(dir.clone(), key_bytes));
        Self {
            dir,
            hmac_key: Zeroizing::new(key_bytes),
            store,
        }
    }

    /// Build the on-disk path for a key. Hash the full key (including
    /// tenant_id and emit_config_hash) so two tenants cannot collide
    /// on the same blueprint+sm and so the file name itself does not
    /// leak the blueprint fingerprint to anyone with directory-list
    /// access.
    ///
    /// The filename is also prefixed with the first 8 bytes of
    /// `blake3::hash(hmac_key)` so two `KernelCache`s pointed at the
    /// same directory but configured with different HMAC keys produce
    /// *disjoint* paths. Without this prefix the two writers would
    /// race on the same final path (`tmp.persist` overwrites whichever
    /// landed first) and both readers would then fail the HMAC check
    /// on each other's writes — every put-then-get round-trip would
    /// look like a miss in steady state.
    ///
    /// The 8-byte HMAC-key fingerprint is NOT the key itself: it's
    /// `blake3::hash(key)` truncated, which is already publicly
    /// observable (anyone with directory-list access can read the
    /// filename). The actual MAC trailer still gates load — partitioning
    /// just avoids the inter-key collision, it is not a confidentiality
    /// boundary.
    ///
    /// Format: `{key_prefix:016x}-{cache_key_hex}.ptxbin`. The key
    /// prefix leads so `ls`-style directory listings group entries by
    /// the writing key — handy for operators rotating keys (each
    /// rotation lands under a new prefix and the old generation is
    /// trivially `rm`-able by prefix glob).
    fn path_for(&self, key: &CacheKey) -> PathBuf {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"tensor-wasm-jit::DiskCache::path::v1\0");
        hasher.update(&key.tenant_id.to_le_bytes());
        hasher.update(&key.blueprint.to_le_bytes());
        hasher.update(&key.sm_version.to_le_bytes());
        hasher.update(&key.emit_config_hash.to_le_bytes());
        let h = hasher.finalize();
        // First 16 bytes (32 hex chars) is plenty of entropy for filenames.
        //
        // T20 perf: render the digest into a fixed 32-byte stack buffer via
        // `hex::encode_to_slice` rather than 16× `format!("{b:02x}")` —
        // the per-byte `format!` path used to allocate 16 transient `String`s
        // (plus the `.collect()` target) on every disk-cache op. The slice
        // form writes ASCII hex directly into the buffer with no heap
        // traffic; `str::from_utf8` then borrows it as a `&str` for the
        // final `format!`-built filename.
        let digest = h.as_bytes();
        let mut cache_key_hex_buf = [0u8; 32];
        hex::encode_to_slice(&digest[..16], &mut cache_key_hex_buf)
            .expect("32 byte buf for 16 byte input");
        let cache_key_hex = std::str::from_utf8(&cache_key_hex_buf).expect("hex is utf8");
        // First 8 bytes of blake3(hmac_key), packed LE → u64 → 16 hex chars.
        // `&self.hmac_key[..]` Deref-borrows the underlying `[u8; 32]` and
        // takes the full slice; `Zeroizing` is `Deref<Target=[u8; 32]>`.
        //
        // T20 perf: same encode-to-slice treatment as the cache-key digest —
        // 8 bytes of HMAC-key fingerprint → 16 hex chars into a stack buf.
        let key_fp = blake3::hash(&self.hmac_key[..]);
        let mut key_prefix_buf = [0u8; 16];
        hex::encode_to_slice(&key_fp.as_bytes()[..8], &mut key_prefix_buf)
            .expect("16 byte buf for 8 byte input");
        let key_prefix_hex = std::str::from_utf8(&key_prefix_buf).expect("hex is utf8");
        self.dir
            .join(format!("{key_prefix_hex}-{cache_key_hex}.ptxbin"))
    }

    /// Encode a kernel into the inner V2 envelope (the same byte layout
    /// pre-T30 used at the head of every `.ptxbin` file, minus the
    /// per-envelope HMAC trailer — the streaming HMAC is now the
    /// artifact store's job). The result is what gets handed to
    /// [`DiskArtifactStore::put`].
    fn encode_v2_envelope(key: &CacheKey, kernel: &CachedKernel) -> Vec<u8> {
        let ptx_bytes = kernel.ptx.text.as_bytes();
        let (grid_x, block_x) = kernel.ptx.launch_geometry;
        let mut buf = Vec::with_capacity(DISK_CACHE_HEADER_LEN_V2 + ptx_bytes.len());
        buf.extend_from_slice(DISK_CACHE_MAGIC_V2);
        buf.extend_from_slice(&key.blueprint.to_le_bytes());
        buf.extend_from_slice(&key.sm_version.to_le_bytes());
        // jit S-3 follow-up: persist `launch_geometry` so L2 hits round-trip
        // the (grid_x, block_x) hint that `ptx_emit` populates. Prior to V2
        // the reconstructor defaulted this to (0, 0) on every disk hit.
        buf.extend_from_slice(&grid_x.to_le_bytes());
        buf.extend_from_slice(&block_x.to_le_bytes());
        buf.extend_from_slice(&(ptx_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(ptx_bytes);
        buf
    }

    /// Write a kernel via the layered `(sidecar -> blob)` representation:
    ///
    /// 1. Build the inner V2 envelope `Vec<u8>` (cross-version-compat
    ///    format, T12-aligned).
    /// 2. Hand the envelope bytes to [`DiskArtifactStore::put`] which
    ///    streams HMAC + zstd onto disk under a content-addressed path
    ///    and returns the [`ContentHash`] of the envelope.
    /// 3. Write a small sidecar at [`Self::path_for`] mapping the
    ///    [`CacheKey`] to that [`ContentHash`] via an atomic temp-then-
    ///    rename, so a partial write never strands a half-formed
    ///    sidecar that a concurrent reader might trip over.
    ///
    /// The artifact store inherits T22's streaming property: the
    /// envelope's bytes are not re-buffered into another `Vec` before
    /// HMAC + zstd — the store's `put` tees through a `MacWriter` /
    /// zstd encoder pipeline straight to the file. The only buffered
    /// allocation here is the v2 envelope itself, which is bounded by
    /// the kernel's PTX size and lives just long enough to be consumed
    /// by `store.put`.
    fn put(&self, key: &CacheKey, kernel: &CachedKernel) -> std::io::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(&self.dir)?;
        // Step 1: build the inner V2 envelope (no per-envelope MAC —
        // the artifact store's HMAC covers it transitively).
        let envelope = Self::encode_v2_envelope(key, kernel);
        // Step 2: stream the envelope through the artifact store. The
        // store handles atomic temp-then-rename of the blob itself.
        let hash = self.store.put(&envelope).map_err(|e| {
            // Wrap as `io::Error` so the existing `Result<(), io::Error>`
            // signature of `DiskCache::put` (and the warn-level log path
            // in `KernelCache::put`) stays unchanged.
            std::io::Error::other(e.to_string())
        })?;
        // Step 3: stamp the sidecar atomically. The sidecar is small
        // (48 bytes) and lives at a path derived from the cache key, so
        // the lookup side only needs one read to find the content
        // hash and one more (through the artifact store) to fetch the
        // verified envelope bytes.
        let mut sidecar = Vec::with_capacity(SIDECAR_LEN_V1);
        sidecar.extend_from_slice(SIDECAR_MAGIC_V1);
        sidecar.extend_from_slice(&hash.0);
        debug_assert_eq!(sidecar.len(), SIDECAR_LEN_V1);
        let sidecar_path = self.path_for(key);
        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir)?;
        tmp.as_file_mut().write_all(&sidecar)?;
        tmp.persist(&sidecar_path).map_err(std::io::Error::other)?;
        Ok(())
    }

    /// Read and verify a kernel from disk. Returns `Ok(None)` on a
    /// genuine miss (sidecar does not exist), `Err` on I/O failure,
    /// and `Ok(None)` (with a warn-level log) on any integrity failure
    /// — magic mismatch on the sidecar, blob lookup failure,
    /// artifact-store HMAC mismatch, or envelope header mismatch. The
    /// loader treats every integrity failure as "no such entry" so a
    /// poisoned file behaves identically to a fresh cache.
    fn get(&self, key: &CacheKey) -> std::io::Result<Option<CachedKernel>> {
        // ---- Sidecar lookup. ----
        let sidecar_path = self.path_for(key);
        let sidecar = match std::fs::read(&sidecar_path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        if sidecar.len() != SIDECAR_LEN_V1 {
            tracing::warn!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                len = sidecar.len(),
                "disk-cache sidecar wrong length; treating as miss"
            );
            return Ok(None);
        }
        if &sidecar[..16] != SIDECAR_MAGIC_V1 {
            // Either a legacy pre-T30 raw-V2 record sitting at the same
            // path (TWJIT-KRNL-v2 magic) or unrelated garbage. Either
            // way the new reader does not understand it — treat as a
            // miss so the next `put` rewrites it under the T30 layout.
            tracing::info!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                "cache.l2.miss.legacy_or_unknown_magic: sidecar will be rewritten on next put"
            );
            return Ok(None);
        }
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(&sidecar[16..48]);
        let content_hash = ContentHash(hash_bytes);

        // ---- Streaming-verified blob fetch via the artifact store. ----
        //
        // The store handles HMAC-SHA256 verification (constant-time),
        // zstd decompression with a [`MAX_DECOMPRESSED_LEN`] cap, and
        // content-hash defence-in-depth. Any failure (NotFound,
        // BadHmac, HashMismatch, …) collapses to a miss here, mirroring
        // the pre-T30 reader's "log + return Ok(None)" convention so
        // the call site's `cache.misses_total` counter still ticks
        // correctly on integrity rejection.
        let envelope = match self.store.get(&content_hash) {
            Ok(bytes) => bytes,
            Err(ArtifactError::NotFound(_)) => {
                tracing::warn!(
                    target: "tensor_wasm_jit::cache",
                    file = %sidecar_path.display(),
                    "disk-cache sidecar references missing artifact blob; treating as miss"
                );
                return Ok(None);
            }
            Err(e) => {
                tracing::warn!(
                    target: "tensor_wasm_jit::cache",
                    file = %sidecar_path.display(),
                    error = %e,
                    "disk-cache artifact-store read failed; treating as miss"
                );
                return Ok(None);
            }
        };

        // ---- Parse the inner V2 envelope. ----
        if envelope.len() < DISK_CACHE_HEADER_LEN_V2 {
            tracing::warn!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                len = envelope.len(),
                "disk-cache V2 envelope too short; treating as miss"
            );
            return Ok(None);
        }
        if &envelope[..16] != DISK_CACHE_MAGIC_V2 {
            tracing::warn!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                "disk-cache V2 envelope magic mismatch; treating as miss"
            );
            return Ok(None);
        }
        // Header integrity is implied by the artifact store's HMAC, but
        // cross-check fingerprint and sm_version against the requested
        // key as defence-in-depth — a sidecar that points at a foreign
        // blob (e.g. someone hand-edited the sidecar's content hash)
        // would still survive the store's MAC because each blob carries
        // its own self-consistent envelope; only this final check
        // refuses the mismatch.
        let mut bp_bytes = [0u8; 8];
        bp_bytes.copy_from_slice(&envelope[16..24]);
        let mut sm_bytes = [0u8; 4];
        sm_bytes.copy_from_slice(&envelope[24..28]);
        let mut grid_x_bytes = [0u8; 4];
        grid_x_bytes.copy_from_slice(&envelope[28..32]);
        let mut block_x_bytes = [0u8; 4];
        block_x_bytes.copy_from_slice(&envelope[32..36]);
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&envelope[36..44]);
        let fingerprint_on_disk = u64::from_le_bytes(bp_bytes);
        let sm_version_on_disk = u32::from_le_bytes(sm_bytes);
        let grid_x_on_disk = u32::from_le_bytes(grid_x_bytes);
        let block_x_on_disk = u32::from_le_bytes(block_x_bytes);
        let ptx_len_on_disk = u64::from_le_bytes(len_bytes) as usize;
        if fingerprint_on_disk != key.blueprint || sm_version_on_disk != key.sm_version {
            tracing::warn!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                "disk-cache V2 envelope header key mismatch; treating as miss"
            );
            return Ok(None);
        }
        let ptx_start = DISK_CACHE_HEADER_LEN_V2;
        let ptx_end = ptx_start.saturating_add(ptx_len_on_disk);
        if ptx_end > envelope.len() {
            tracing::warn!(
                target: "tensor_wasm_jit::cache",
                file = %sidecar_path.display(),
                "disk-cache declared ptx_len overruns envelope; treating as miss"
            );
            return Ok(None);
        }
        let ptx_text = match std::str::from_utf8(&envelope[ptx_start..ptx_end]) {
            Ok(s) => s.to_string(),
            Err(_) => {
                tracing::warn!(
                    target: "tensor_wasm_jit::cache",
                    file = %sidecar_path.display(),
                    "disk-cache PTX bytes are not valid UTF-8; treating as miss"
                );
                return Ok(None);
            }
        };
        // Reconstruct the kernel via the integrity-aware constructor so
        // the L1 cache accepts it without further verification work.
        // V2 persists the emit-time `launch_geometry` hint; reading it
        // back here closes the lost-geometry bug (previously this defaulted
        // to (0, 0) and the dispatch path silently fell back to guest-
        // declared launch params for every L2 hit).
        let ptx = Arc::new(EmittedPtx {
            text: ptx_text,
            launch_geometry: (grid_x_on_disk, block_x_on_disk),
        });
        Ok(Some(CachedKernel::new(
            fingerprint_on_disk,
            ptx,
            CompiledHandle::default(),
        )))
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
        // Route through `CachedKernel::new` so `integrity_hash` matches the
        // PTX text by construction; `KernelCache::put` rejects entries
        // whose hash disagrees with the text (jit S-3).
        CachedKernel::new(
            fp,
            Arc::new(EmittedPtx {
                text: String::new(),
                launch_geometry: (1, 1),
            }),
            CompiledHandle::default(),
        )
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
            CachedKernel::new(
                bp.fingerprint(),
                original.clone(),
                CompiledHandle::default(),
            ),
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
                    let key =
                        CacheKey::for_tenant(TenantId(0), (t as u64) * KEYS_PER_THREAD + i, 80);
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
                let key = CacheKey::for_tenant(TenantId(0), (t as u64) * KEYS_PER_THREAD + i, 80);
                assert!(
                    cache.get(&key).is_some(),
                    "missing key after concurrent inserts: ({t}, {i})"
                );
            }
        }
    }

    /// T20 perf regression: `KernelCache::get` returns a shared
    /// `Arc<CachedKernel>` rather than a cloned wrapper value. Two
    /// successive hits on the same key must yield pointer-equal Arcs
    /// (same allocation, just a refcount bump per hit) — proving the
    /// hot path no longer clones the 32-byte integrity hash, the
    /// fingerprint word, and the inner `Arc<EmittedPtx>` refcount on
    /// every dispatch.
    #[test]
    fn cache_get_returns_arc_no_clone() {
        let cache = KernelCache::new();
        let key = CacheKey::for_tenant(TenantId(1), 0xABCD, 80);
        let original_ptx = Arc::new(EmittedPtx {
            text: ".visible .entry t20_arc(){}".into(),
            launch_geometry: (1, 32),
        });
        cache.put(
            key,
            CachedKernel::new(0xABCD, original_ptx.clone(), CompiledHandle::default()),
        );
        let first = cache.get(&key).expect("first hit");
        let second = cache.get(&key).expect("second hit");
        // The two `Arc<CachedKernel>` handles must point at the same
        // allocation. If `get` had reverted to cloning the wrapper, the
        // two handles would point at distinct allocations (still sharing
        // the inner `Arc<EmittedPtx>`, but that is a different invariant).
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache.get must return refcount-bumped Arc handles, not cloned \
             wrappers — distinct allocations indicate the T20 perf fix \
             regressed (clone-on-hit reintroduced)"
        );
        // Bonus: the inner PTX is still shared with the original (this was
        // already pinned by `cache_hit_returns_arc_shared_ptx` above).
        assert!(Arc::ptr_eq(&first.ptx, &original_ptx));
    }

    /// T20 perf: `get` increments `cache_hits_total` on every Some-returning
    /// call and `cache_misses_total` on every None-returning call. A miss
    /// followed by a hit must produce one of each, with neither counter
    /// double-counting.
    #[test]
    fn cache_hits_misses_counter() {
        let cache = KernelCache::new();
        assert_eq!(cache.cache_hits_total(), 0);
        assert_eq!(cache.cache_misses_total(), 0);

        let key = CacheKey::for_tenant(TenantId(9), 0xC0FFEE, 80);

        // Miss: nothing has been inserted yet.
        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(
            cache.cache_misses_total(),
            1,
            "first miss must bump the miss counter exactly once"
        );
        assert_eq!(
            cache.cache_hits_total(),
            0,
            "a miss must not bump the hit counter"
        );

        // Hit: install then look up.
        cache.put(key, dummy_kernel(0xC0FFEE));
        assert!(cache.get(&key).is_some(), "post-put get must hit");
        assert_eq!(
            cache.cache_hits_total(),
            1,
            "first hit must bump the hit counter exactly once"
        );
        assert_eq!(
            cache.cache_misses_total(),
            1,
            "a hit must not bump the miss counter"
        );

        // Second hit increments hits again.
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.cache_hits_total(), 2);
        assert_eq!(cache.cache_misses_total(), 1);
    }

    /// jit S-3 T13 regression: `DiskCacheConfig`'s `Debug` impl MUST NOT
    /// dump the raw `hmac_key` bytes. A derived `Debug` would have
    /// formatted the array as `[222, 222, 222, …]` (or `[de, de, …]`
    /// under hex), leaking the server-side secret into any log line,
    /// panic message, or `tracing` field expansion that happens to
    /// include a `DiskCacheConfig`.
    ///
    /// We use the sentinel byte `0xDE` repeated 32 times because the
    /// derived `Debug` would have produced either `222` (decimal —
    /// stdlib default for `u8` arrays) or `de` (hex) at every position;
    /// asserting that neither pattern appears anywhere in the formatted
    /// output catches both cases. We also positively assert the
    /// "redacted" substring is present so the test fails informatively
    /// if someone replaces the manual impl with something else that
    /// happens to omit the bytes but also omits the redaction marker.
    #[test]
    fn disk_cache_config_debug_redacts_hmac_key() {
        let cfg = DiskCacheConfig {
            dir: PathBuf::from("/tmp/tensor-wasm-jit-debug-test"),
            hmac_key: [0xDEu8; 32],
        };
        let dbg = format!("{cfg:?}");
        let lowered = dbg.to_ascii_lowercase();
        // Negative: neither decimal nor hex representation of the key
        // bytes should appear in the formatted output. Derived `Debug`
        // for `[u8; 32]` would have produced `222` thirty-two times
        // (decimal) or `de` thirty-two times (hex/upper-hex
        // alternatives); a single occurrence of either is suspicious,
        // but to keep the assertion robust against incidental
        // substrings in `dir` we count occurrences instead.
        let decimal_hits = lowered.matches("222").count();
        let hex_hits = lowered.matches("de").count();
        assert!(
            decimal_hits < 32,
            "DiskCacheConfig Debug output appears to contain the raw key in \
             decimal form ({decimal_hits} occurrences of \"222\"): {dbg}"
        );
        assert!(
            hex_hits < 32,
            "DiskCacheConfig Debug output appears to contain the raw key in \
             hex form ({hex_hits} occurrences of \"de\"): {dbg}"
        );
        // Positive: the redaction marker is present.
        assert!(
            lowered.contains("redacted"),
            "DiskCacheConfig Debug output should mark hmac_key as redacted: {dbg}"
        );
    }
}
