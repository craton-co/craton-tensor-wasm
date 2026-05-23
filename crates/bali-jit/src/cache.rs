//! Compiled-kernel cache.
//!
//! The cache is keyed by `(blueprint_fingerprint, sm_version)`. LRU eviction
//! caps the entry count. On a cache hit the PTX text is reused without
//! re-emitting (cheap) and without re-compiling via `ptxas`/`cust` (expensive
//! — 10-50 ms for non-trivial kernels).

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;

use crate::ir::BaliKernelBlueprint;
use crate::ptx_emit::EmittedPtx;

/// Cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// `BaliKernelBlueprint::fingerprint()`.
    pub blueprint: u64,
    /// CUDA compute capability (e.g. 80 for sm_80, 89 for sm_89).
    pub sm_version: u32,
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

/// Thread-safe LRU cache of compiled kernels.
#[derive(Clone)]
pub struct KernelCache {
    inner: Arc<Mutex<LruCache<CacheKey, CachedKernel>>>,
    capacity: usize,
}

impl KernelCache {
    /// Construct with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Construct with explicit capacity.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        let inner = Arc::new(Mutex::new(LruCache::new(
            NonZeroUsize::new(cap).expect(">0"),
        )));
        Self {
            inner,
            capacity: cap,
        }
    }

    /// Insert (or replace) a kernel.
    pub fn put(&self, key: CacheKey, kernel: CachedKernel) {
        self.inner.lock().expect("cache poisoned").put(key, kernel);
    }

    /// Look up a kernel; touches the LRU position.
    pub fn get(&self, key: &CacheKey) -> Option<CachedKernel> {
        self.inner.lock().expect("cache poisoned").get(key).cloned()
    }

    /// Look up by blueprint + sm_version; convenience wrapper around [`Self::get`].
    pub fn get_for(
        &self,
        blueprint: &BaliKernelBlueprint,
        sm_version: u32,
    ) -> Option<CachedKernel> {
        self.get(&CacheKey {
            blueprint: blueprint.fingerprint(),
            sm_version,
        })
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("cache poisoned").len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    use crate::ir::{BaliKernelBlueprint, BaliOp};
    use crate::ptx_emit::EmittedPtx;

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
        let key = CacheKey {
            blueprint: 1,
            sm_version: 80,
        };
        cache.put(key, dummy_kernel(1));
        assert_eq!(cache.get(&key).unwrap().fingerprint, 1);
    }

    #[test]
    fn lru_evicts_oldest() {
        let cache = KernelCache::with_capacity(2);
        let k1 = CacheKey {
            blueprint: 1,
            sm_version: 80,
        };
        let k2 = CacheKey {
            blueprint: 2,
            sm_version: 80,
        };
        let k3 = CacheKey {
            blueprint: 3,
            sm_version: 80,
        };
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
        let bp = BaliKernelBlueprint::new("k").push(BaliOp::VecAdd { lanes: 4 });
        let key = CacheKey {
            blueprint: bp.fingerprint(),
            sm_version: 80,
        };
        cache.put(key, dummy_kernel(bp.fingerprint()));
        assert!(cache.get_for(&bp, 80).is_some());
        assert!(
            cache.get_for(&bp, 89).is_none(),
            "different sm_version is a miss"
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
        let bp = BaliKernelBlueprint::new("matmul").push(BaliOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        let key = CacheKey {
            blueprint: bp.fingerprint(),
            sm_version: 80,
        };
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
}
