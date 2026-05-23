//! Wasmtime [`MemoryCreator`] backed by [`UnifiedBuffer`].
//!
//! When a Wasm module declares a `(memory 1)` (one page), Wasmtime asks the
//! [`MemoryCreator`] to materialise that memory. We pre-allocate the requested
//! `maximum` (or a default cap), so growth is just a size check — no realloc
//! and no host-to-device copy. The `as_ptr` returned points directly into
//! Unified Memory on CUDA hosts, meaning CUDA kernels can read/write the
//! buffer the Wasm guest sees, with no explicit DMA.
//!
//! # `memory_protection_keys` — known gap vs the S5 plan
//!
//! The S5 plan calls for enabling Wasmtime's `memory_protection_keys` for
//! intra-process Wasm isolation "where available". In wasmtime 25 this knob
//! lives on `PoolingAllocationConfig::memory_protection_keys`, NOT on
//! `Config`, and it is mutually exclusive with `Config::with_host_memory`:
//! pooling MPK colours wasmtime-owned slabs, whereas this crate's whole
//! purpose is to hand Wasmtime host-owned `UnifiedBuffer` memories. Inter-
//! tenant isolation therefore relies on (a) Wasmtime's Cranelift-emitted
//! bounds checks on every Wasm load/store, (b) each tenant owning a
//! distinct `UnifiedBuffer` with no shared backing store, and (c) the
//! per-instance stream and per-tenant context machinery wired up in S7
//! (`bali-exec`) and S16 (`bali-tenant`) — not on CPU protection keys
//! and not on OS-level page guards (managed memory is migrated by the
//! CUDA driver, which is incompatible with host `mprotect`). See
//! `SECURITY.md` at the repo root for the full threat model.
//!
//! MPK is now available as an *alternate* engine mode via
//! `bali_exec::engine::MemoryBackend::PoolingMpk`, at the explicit cost of
//! dropping `BaliMemoryCreator` / `UnifiedBuffer` integration (and therefore
//! the GPU integration path). Operators can pick the mode that fits their
//! workload: `UnifiedBuffer` for the GPU integration path described above,
//! or `PoolingMpk` for CPU-only / batch-GPU workloads that want intra-process
//! Wasm isolation via CPU PKU. The architectural exclusivity between the
//! two modes is real and enforced by Wasmtime.

use std::ops::Range;
use std::sync::Arc;

use wasmtime::{LinearMemory, MemoryCreator, MemoryType};

use crate::pool::UnifiedMemoryPool;
use crate::unified::{DeviceId, UnifiedBuffer, UnifiedError};

/// Default maximum capacity, in bytes, when a Wasm module declares no upper
/// bound on its linear memory. 256 MiB matches the plan's working-set cap.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// A Wasm linear memory backed by [`UnifiedBuffer`].
///
/// The buffer is allocated at construction time with the requested *maximum*
/// (or [`DEFAULT_MAX_BYTES`]) so `grow_to` becomes a size check. This avoids
/// the cost of `cudaMemcpy` on every `memory.grow` and keeps the kernel-side
/// pointer stable across growth events.
#[derive(Debug)]
pub struct BaliLinearMemory {
    buffer: UnifiedBuffer,
    /// Currently-visible (logical) size. `<= buffer.len()`. Mutation goes
    /// through `&mut self` (see `LinearMemory::grow_to`), so no interior
    /// lock is required; `Send + Sync` are auto-derived from the fields.
    current_size: usize,
    /// Hard cap. Always `<= buffer.len()`.
    maximum_size: usize,
}

impl BaliLinearMemory {
    /// Create a new linear memory.
    ///
    /// - `minimum_bytes`: initial visible size (Wasm pages × 65 536).
    /// - `maximum_bytes`: cap on growth. If `None`, [`DEFAULT_MAX_BYTES`] is used.
    pub fn new(minimum_bytes: usize, maximum_bytes: Option<usize>) -> Result<Self, UnifiedError> {
        Self::new_on(minimum_bytes, maximum_bytes, DeviceId::default())
    }

    /// Same as [`new`](Self::new) but on a specific device.
    pub fn new_on(
        minimum_bytes: usize,
        maximum_bytes: Option<usize>,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        let max = maximum_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        if minimum_bytes > max {
            return Err(UnifiedError::Allocation(format!(
                "minimum {minimum_bytes} > maximum {max}"
            )));
        }
        // Allocate at least 1 byte so the underlying allocator never sees zero.
        let cap = max.max(1);
        let buffer = UnifiedBuffer::new_on(cap, device_id)?;
        Ok(Self {
            buffer,
            current_size: minimum_bytes,
            maximum_size: max,
        })
    }

    /// Current logical size in bytes.
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Pre-allocated capacity (the hard cap).
    pub fn capacity(&self) -> usize {
        self.maximum_size
    }

    /// Borrow the buffer as a shared byte slice over the *current* size.
    pub fn as_slice(&self) -> &[u8] {
        let size = self.current_size;
        // SAFETY: the underlying buffer covers `maximum_size >= size` bytes.
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), size) }
    }
}

unsafe impl LinearMemory for BaliLinearMemory {
    fn byte_size(&self) -> usize {
        self.current_size()
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        Some(self.maximum_size)
    }

    fn grow_to(&mut self, new_size: usize) -> anyhow::Result<()> {
        if new_size > self.maximum_size {
            return Err(anyhow::anyhow!(
                "memory.grow requested {new_size} > maximum {}",
                self.maximum_size
            ));
        }
        if new_size < self.current_size {
            return Err(anyhow::anyhow!(
                "memory.grow cannot shrink ({new_size} < current {})",
                self.current_size
            ));
        }
        self.current_size = new_size;
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        // SAFETY: returning the raw underlying pointer is the contract of
        // LinearMemory. Wasmtime guarantees synchronised access via its own
        // borrow tracking.
        self.buffer.as_ptr() as *mut u8
    }

    fn wasm_accessible(&self) -> Range<usize> {
        let base = self.buffer.as_ptr() as usize;
        base..(base + self.maximum_size)
    }
}

/// Wasm linear memory carved out of an [`UnifiedMemoryPool`] slab.
///
/// Holds an `Arc<UnifiedMemoryPool>` keepalive so the slab outlives the Wasm
/// instance, plus a raw pointer + length into the slab's bytes. The
/// [`crate::pool::PoolAllocation`] drop guard is intentionally leaked at
/// construction time (see `BaliMemoryCreator::new_memory`); the live-allocation
/// counter therefore stays elevated for the lifetime of this memory and serves
/// as a misuse signal if the caller tries to [`UnifiedMemoryPool::reset`] while
/// any pool-backed memory still exists.
struct PooledLinearMemory {
    /// Keeps the slab alive while this memory is in use. Never read
    /// directly — its Drop is the whole point.
    #[allow(dead_code)]
    pool_keepalive: Arc<UnifiedMemoryPool>,
    /// Pointer to the first byte of the carved region.
    base_ptr: *mut u8,
    /// Bytes carved (the hard cap).
    size: usize,
    /// Currently-visible (logical) size. `<= size`.
    current_size: usize,
    /// Hard cap exposed to Wasm. Equals `size`.
    max_size: usize,
}

impl std::fmt::Debug for PooledLinearMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledLinearMemory")
            .field("base_ptr", &self.base_ptr)
            .field("size", &self.size)
            .field("current_size", &self.current_size)
            .field("max_size", &self.max_size)
            .finish()
    }
}

// SAFETY: `base_ptr` points into the slab owned by `pool_keepalive`, which is
// an `Arc<UnifiedMemoryPool>`. `UnifiedMemoryPool` (via `UnifiedBuffer`) is
// `Send + Sync` and the carved region is disjoint from every other pool
// allocation by the bump allocator's invariant. Concurrent mutation through
// the raw pointer requires external synchronisation — same contract as
// `BaliLinearMemory` and `Vec<u8>` once a `&mut [u8]` exists.
unsafe impl Send for PooledLinearMemory {}
unsafe impl Sync for PooledLinearMemory {}

unsafe impl LinearMemory for PooledLinearMemory {
    fn byte_size(&self) -> usize {
        self.current_size
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        Some(self.max_size)
    }

    fn grow_to(&mut self, new_size: usize) -> anyhow::Result<()> {
        if new_size > self.max_size {
            return Err(anyhow::anyhow!(
                "memory.grow requested {new_size} > maximum {}",
                self.max_size
            ));
        }
        if new_size < self.current_size {
            return Err(anyhow::anyhow!(
                "memory.grow cannot shrink ({new_size} < current {})",
                self.current_size
            ));
        }
        self.current_size = new_size;
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        self.base_ptr
    }

    fn wasm_accessible(&self) -> Range<usize> {
        let base = self.base_ptr as usize;
        base..(base + self.max_size)
    }
}

/// A [`MemoryCreator`] that hands out [`BaliLinearMemory`] instances.
///
/// Wrap in [`Arc`] and pass to `wasmtime::Config::with_host_memory`.
#[derive(Clone)]
pub struct BaliMemoryCreator {
    inner: Arc<MemoryCreatorState>,
}

struct MemoryCreatorState {
    device_id: DeviceId,
    /// Optional pre-allocated slab. When set, `new_memory` carves Wasm linear
    /// memories from this pool via [`UnifiedMemoryPool::allocate`]; if the
    /// slab is exhausted it falls back to a fresh [`UnifiedBuffer`].
    pool: Option<Arc<UnifiedMemoryPool>>,
}

impl Default for BaliMemoryCreator {
    fn default() -> Self {
        Self::new(DeviceId::default())
    }
}

impl BaliMemoryCreator {
    /// Construct without an underlying pool. New memories allocate fresh
    /// [`UnifiedBuffer`]s on every `new_memory` call.
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: None,
            }),
        }
    }

    /// Construct with a pre-allocated pool. `new_memory` carves out of this
    /// slab; on exhaustion it falls back to a fresh [`UnifiedBuffer`] so a
    /// short slab does not turn into a fatal error.
    ///
    /// **Lifetime contract.** Pool-backed linear memories form an
    /// all-or-nothing batch: the caller must drop every Wasm instance handed
    /// out by this creator *before* dropping the last `Arc<UnifiedMemoryPool>`
    /// reference and calling [`UnifiedMemoryPool::reset`]. See `new_memory`
    /// for the `unsafe` rationale.
    pub fn with_pool(device_id: DeviceId, pool: Arc<UnifiedMemoryPool>) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: Some(pool),
            }),
        }
    }

    /// The device this creator targets.
    pub fn device_id(&self) -> DeviceId {
        self.inner.device_id
    }

    /// The underlying pool, if one was provided at construction.
    pub fn pool(&self) -> Option<&Arc<UnifiedMemoryPool>> {
        self.inner.pool.as_ref()
    }
}

unsafe impl MemoryCreator for BaliMemoryCreator {
    fn new_memory(
        &self,
        _ty: MemoryType,
        minimum: usize,
        maximum: Option<usize>,
        _reserved_size_in_bytes: Option<usize>,
        _guard_size_in_bytes: usize,
    ) -> Result<Box<dyn LinearMemory>, String> {
        let max = maximum.unwrap_or(DEFAULT_MAX_BYTES);
        if minimum > max {
            return Err(format!("minimum {minimum} > maximum {max}"));
        }

        if let Some(pool) = self.inner.pool.as_ref() {
            // Wasm linear memory is page-aligned by spec (64 KiB pages).
            const WASM_PAGE: usize = 65_536;
            let carve_size = max.max(1);
            match pool.allocate(carve_size, WASM_PAGE) {
                Ok(mut alloc) => {
                    let slice = alloc.as_mut_slice();
                    let base_ptr = slice.as_mut_ptr();
                    let size = slice.len();
                    // SAFETY: We leak the PoolAllocation drop guard intentionally.
                    // The pool uses "batch reclaim" semantics (reset() succeeds
                    // only when live count is zero) and pool-backed linear
                    // memories are torn down together when the parent
                    // BaliMemoryCreator's Arc<UnifiedMemoryPool> drops. The
                    // PoolAllocation's Drop would decrement the live counter;
                    // we keep that counter as a leak-detection signal for misuse.
                    //
                    // Trade-off: this prevents the pool from being reset while
                    // ANY pooled memory exists. Caller's responsibility to drop
                    // the creator (and thus the pool Arc) before reset.
                    //
                    // The raw `base_ptr` remains valid because `pool_keepalive`
                    // holds the slab alive for the lifetime of the returned
                    // `PooledLinearMemory`, and the bump allocator never hands
                    // the same byte range to another allocation.
                    std::mem::forget(alloc);
                    return Ok(Box::new(PooledLinearMemory {
                        pool_keepalive: Arc::clone(pool),
                        base_ptr,
                        size,
                        current_size: minimum,
                        max_size: size,
                    }));
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        requested = carve_size,
                        remaining = pool.remaining(),
                        "pool exhausted; falling back to fresh UnifiedBuffer"
                    );
                }
            }
        }

        let mem = BaliLinearMemory::new_on(minimum, maximum, self.inner.device_id)
            .map_err(|e| e.to_string())?;
        Ok(Box::new(mem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_query_size() {
        let mem = BaliLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        assert_eq!(mem.byte_size(), 64 * 1024);
        assert_eq!(mem.maximum_byte_size(), Some(1024 * 1024));
        assert_eq!(mem.capacity(), 1024 * 1024);
    }

    #[test]
    fn grow_increases_visible_size() {
        let mut mem = BaliLinearMemory::new(64 * 1024, Some(256 * 1024)).unwrap();
        mem.grow_to(128 * 1024).expect("grow");
        assert_eq!(mem.byte_size(), 128 * 1024);
    }

    #[test]
    fn grow_beyond_maximum_rejected() {
        let mut mem = BaliLinearMemory::new(64 * 1024, Some(128 * 1024)).unwrap();
        let err = mem.grow_to(256 * 1024).unwrap_err();
        assert!(err.to_string().contains("maximum"));
    }

    #[test]
    fn shrink_rejected() {
        let mut mem = BaliLinearMemory::new(128 * 1024, Some(1024 * 1024)).unwrap();
        let err = mem.grow_to(64 * 1024).unwrap_err();
        assert!(err.to_string().contains("shrink"));
    }

    #[test]
    fn ptr_stable_across_grow() {
        let mut mem = BaliLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        let p1 = mem.as_ptr();
        mem.grow_to(256 * 1024).unwrap();
        let p2 = mem.as_ptr();
        assert_eq!(p1, p2, "as_ptr must be stable across grow_to");
    }

    #[test]
    fn wasm_accessible_covers_capacity() {
        let mem = BaliLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        let r = mem.wasm_accessible();
        assert_eq!(r.end - r.start, 1024 * 1024);
    }

    #[test]
    fn minimum_exceeds_maximum_rejected() {
        let err = BaliLinearMemory::new(1024 * 1024, Some(512 * 1024)).unwrap_err();
        assert!(matches!(err, UnifiedError::Allocation(_)));
    }

    #[test]
    fn creator_default_device_zero() {
        let c = BaliMemoryCreator::default();
        assert_eq!(c.device_id(), DeviceId(0));
    }

    #[test]
    fn creator_default_pool_is_none() {
        let c = BaliMemoryCreator::default();
        assert!(c.pool().is_none());
    }

    #[test]
    fn creator_with_pool_round_trip() {
        let pool = std::sync::Arc::new(crate::pool::UnifiedMemoryPool::new(64 * 1024).unwrap());
        let creator = BaliMemoryCreator::with_pool(DeviceId(2), pool.clone());
        assert_eq!(creator.device_id(), DeviceId(2));
        assert!(creator.pool().is_some());
        assert!(std::sync::Arc::ptr_eq(creator.pool().unwrap(), &pool));
    }

    #[test]
    fn creator_with_pool_carves_from_slab() {
        use std::sync::Arc;
        let pool = Arc::new(crate::pool::UnifiedMemoryPool::new(8 * 1024 * 1024).unwrap());
        let creator = BaliMemoryCreator::with_pool(DeviceId::default(), pool.clone());
        let mt = wasmtime::MemoryType::new(1, Some(2));
        use wasmtime::MemoryCreator;
        let mem = creator
            .new_memory(mt, 64 * 1024, Some(128 * 1024), None, 0)
            .expect("new_memory");
        assert!(mem.byte_size() == 64 * 1024);
        assert!(mem.maximum_byte_size() == Some(128 * 1024));
        // Verify the pool's live count incremented (proving the carving path ran)
        assert_eq!(pool.live_allocations(), 1);
        // Note: pool.reset() will fail until the leaked PoolAllocation count is
        // cleared — by design (see SAFETY comment in new_memory).
    }
}
