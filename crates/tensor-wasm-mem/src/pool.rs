// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `UnifiedMemoryPool` — a bump allocator that amortises the cost of CUDA
//! Unified Memory allocations.
//!
//! `cudaMallocManaged` is expensive (tens to hundreds of microseconds per call
//! on busy systems). For Wasm linear memory we want sub-microsecond allocations
//! at instance-spawn time. The pool pre-allocates one large [`UnifiedBuffer`]
//! and hands out aligned sub-slices via a simple bump pointer.

use std::fmt;

use parking_lot::Mutex;

use crate::unified::{DeviceId, UnifiedBuffer, UnifiedError};

/// A bump-allocated pool carved from a single underlying [`UnifiedBuffer`].
///
/// Allocations succeed until the slab is exhausted; the pool does not reclaim
/// freed regions (callers maintain their own life cycle by dropping
/// [`PoolAllocation`]s, which mark the bump pointer as logically released only
/// in the *all-released-at-end* discipline used by ephemeral Wasm instances).
pub struct UnifiedMemoryPool {
    slab: UnifiedBuffer,
    state: Mutex<PoolState>,
}

struct PoolState {
    /// Next free byte offset within the slab.
    bump: usize,
    /// Outstanding allocations counter; the slab is "reset-eligible" when this hits zero.
    live: usize,
    /// Total bytes ever issued (sticky counter for metrics).
    issued_total: u64,
}

/// A region of memory carved from a pool. Drops decrement the pool's live count.
pub struct PoolAllocation<'p> {
    pool: &'p UnifiedMemoryPool,
    offset: usize,
    size: usize,
}

impl UnifiedMemoryPool {
    /// Create a pool that owns `capacity` bytes on the default device.
    pub fn new(capacity: usize) -> Result<Self, UnifiedError> {
        Self::new_on(capacity, DeviceId::default())
    }

    /// Create a pool that owns `capacity` bytes on the named device.
    pub fn new_on(capacity: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        let slab = UnifiedBuffer::new_on(capacity, device_id)?;
        Ok(Self {
            slab,
            state: Mutex::new(PoolState {
                bump: 0,
                live: 0,
                issued_total: 0,
            }),
        })
    }

    /// Slab capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.slab.len()
    }

    /// Device that the underlying slab is anchored to.
    pub fn device_id(&self) -> DeviceId {
        self.slab.device_id()
    }

    /// Bytes still available for new allocations.
    pub fn remaining(&self) -> usize {
        let st = self.state.lock();
        self.slab.len().saturating_sub(st.bump)
    }

    /// Outstanding allocation count.
    pub fn live_allocations(&self) -> usize {
        self.state.lock().live
    }

    /// Total bytes issued since the pool was created (or last reset).
    pub fn issued_total(&self) -> u64 {
        self.state.lock().issued_total
    }

    /// Allocate `size` bytes aligned to `align` (must be a power of two).
    ///
    /// Returns `Err(UnifiedError::Allocation(...))` if the slab is exhausted or
    /// `align` is zero/not a power of two.
    pub fn allocate(&self, size: usize, align: usize) -> Result<PoolAllocation<'_>, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        if align == 0 || !align.is_power_of_two() {
            return Err(UnifiedError::Allocation(format!(
                "alignment {align} is not a non-zero power of two"
            )));
        }
        // Cap alignment at 1 GiB. Larger values would dwarf any realistic slab
        // size and only existed in the API surface to satisfy `is_power_of_two`
        // (which accepts `1 << 63`). Rejecting them early prevents
        // `(bump + align - 1)` from silently wrapping into a tiny positive
        // value that then bypasses the exhaustion check below.
        const MAX_ALIGN: usize = 1 << 30;
        if align > MAX_ALIGN {
            return Err(UnifiedError::Allocation(format!(
                "alignment {align} exceeds maximum {MAX_ALIGN}"
            )));
        }

        let mut st = self.state.lock();
        // Compute `(bump + align - 1) & !(align - 1)` with overflow checks so
        // a pathological `bump` near `usize::MAX` cannot wrap.
        let aligned_bump = st
            .bump
            .checked_add(align - 1)
            .map(|v| v & !(align - 1))
            .ok_or_else(|| UnifiedError::Allocation("alignment overflow".into()))?;
        let end = aligned_bump
            .checked_add(size)
            .ok_or_else(|| UnifiedError::Allocation("offset overflow".into()))?;
        if end > self.slab.len() {
            return Err(UnifiedError::Allocation(format!(
                "pool exhausted: need {size} bytes (aligned offset {aligned_bump}), capacity {}",
                self.slab.len()
            )));
        }
        st.bump = end;
        st.live += 1;
        st.issued_total = st.issued_total.saturating_add(size as u64);
        // Drop the lock before the (potentially large) zero-fill so other
        // allocators aren't blocked on memset for tenants that aren't us. The
        // bump pointer has already moved past `[aligned_bump, end)` so no
        // concurrent allocate() can hand the same range to another caller.
        drop(st);

        // Cross-tenant data-leak mitigation (audit H1):
        // -------------------------------------------------------------------
        // The slab is recycled across tenants via `reset()`, which only resets
        // the bump pointer — recycled bytes still carry the previous tenant's
        // data. Zero the freshly-carved region before we hand the
        // `PoolAllocation` to the caller so a guest cannot observe a peer's
        // memory through an uninitialised read.
        //
        // We use `ptr::write_bytes` rather than `slice::fill(0)` for two
        // reasons: (1) it lowers to a single `memset` intrinsic on every
        // backend LLVM cares about, where `.fill(0)` historically optimises
        // less reliably; (2) it sidesteps the need to construct a `&mut [u8]`
        // alias over the slab, which is awkward to do soundly while another
        // thread may hold a different `&mut [u8]` slice into a disjoint
        // region of the same slab.
        //
        // Cost: O(size) per allocation. For a 256 MiB Wasm linear memory this
        // is on the order of tens of milliseconds — large enough that callers
        // doing many small allocations should batch where possible, but
        // unavoidable for correctness given the recycle discipline. Skipping
        // it would re-open the H1 cross-tenant disclosure window.
        //
        // SAFETY: `aligned_bump + size <= self.slab.len()` (checked above);
        // the slab's pointer is non-null and points to `len()` valid bytes
        // for the lifetime of `&self`; the bump allocator guarantees the
        // `[aligned_bump, aligned_bump + size)` byte range is disjoint from
        // every other live `PoolAllocation`, so this write cannot race
        // another thread's `as_mut_slice()`.
        unsafe {
            let base = self.slab.as_ptr().add(aligned_bump) as *mut u8;
            std::ptr::write_bytes(base, 0u8, size);
        }

        Ok(PoolAllocation {
            pool: self,
            offset: aligned_bump,
            size,
        })
    }

    /// Reset the bump pointer back to zero. Safe to call only when there are
    /// no outstanding [`PoolAllocation`]s; returns an error otherwise.
    pub fn reset(&self) -> Result<(), UnifiedError> {
        let mut st = self.state.lock();
        if st.live != 0 {
            return Err(UnifiedError::Allocation(format!(
                "cannot reset: {} live allocations outstanding",
                st.live
            )));
        }
        st.bump = 0;
        // `issued_total` is sticky on purpose — it reflects lifetime activity.
        Ok(())
    }

    fn release(&self, _offset: usize, _size: usize) {
        let mut st = self.state.lock();
        st.live = st.live.saturating_sub(1);
    }

    /// Raw pointer to the start of the slab (intended for tests / FFI).
    pub fn slab_ptr(&self) -> *const u8 {
        self.slab.as_ptr()
    }
}

impl<'p> PoolAllocation<'p> {
    /// Byte length of this region.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length (never for a successfully created allocation).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Offset within the underlying slab.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Borrow as a shared byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the pool's slab is alive (we borrow `&'p UnifiedMemoryPool`)
        // and we carved [offset, offset+size) out of it during allocation.
        unsafe {
            let base = self.pool.slab.as_ptr().add(self.offset);
            std::slice::from_raw_parts(base, self.size)
        }
    }

    /// Borrow as a mutable byte slice.
    ///
    /// `&mut self` and the disjoint-region invariant of bump allocation
    /// together prove no other live alias points at `[offset, offset+size)`.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: see above; `&mut self` proves uniqueness of this PoolAllocation
        // and the bump allocator guarantees disjoint regions across allocations.
        unsafe {
            let base = self.pool.slab.as_ptr().add(self.offset) as *mut u8;
            std::slice::from_raw_parts_mut(base, self.size)
        }
    }
}

impl fmt::Debug for PoolAllocation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolAllocation")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for PoolAllocation<'_> {
    fn drop(&mut self) {
        self.pool.release(self.offset, self.size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_within_capacity() {
        let pool = UnifiedMemoryPool::new(1024).unwrap();
        assert_eq!(pool.capacity(), 1024);
        let a = pool.allocate(100, 16).unwrap();
        assert_eq!(a.len(), 100);
        assert_eq!(a.offset() % 16, 0);
        assert_eq!(pool.live_allocations(), 1);
    }

    #[test]
    fn allocations_are_disjoint_and_aligned() {
        let pool = UnifiedMemoryPool::new(4096).unwrap();
        let a = pool.allocate(100, 64).unwrap();
        let b = pool.allocate(200, 64).unwrap();
        assert!(b.offset() >= a.offset() + a.len());
        assert_eq!(a.offset() % 64, 0);
        assert_eq!(b.offset() % 64, 0);
    }

    #[test]
    fn exhaustion_returns_error() {
        let pool = UnifiedMemoryPool::new(128).unwrap();
        let _a = pool.allocate(64, 16).unwrap();
        let err = pool.allocate(128, 16).expect_err("should exhaust");
        assert!(matches!(err, UnifiedError::Allocation(_)));
    }

    #[test]
    fn drop_decrements_live() {
        let pool = UnifiedMemoryPool::new(256).unwrap();
        {
            let _a = pool.allocate(32, 1).unwrap();
            assert_eq!(pool.live_allocations(), 1);
        }
        assert_eq!(pool.live_allocations(), 0);
    }

    #[test]
    fn reset_only_when_empty() {
        let pool = UnifiedMemoryPool::new(256).unwrap();
        let a = pool.allocate(64, 1).unwrap();
        assert!(
            pool.reset().is_err(),
            "reset must fail while allocations live"
        );
        drop(a);
        pool.reset().expect("reset must succeed when empty");
        assert_eq!(pool.remaining(), pool.capacity());
    }

    #[test]
    fn issued_total_is_sticky_across_reset() {
        let pool = UnifiedMemoryPool::new(1024).unwrap();
        {
            let _a = pool.allocate(64, 1).unwrap();
            let _b = pool.allocate(64, 1).unwrap();
        }
        assert_eq!(pool.issued_total(), 128);
        pool.reset().unwrap();
        assert_eq!(pool.issued_total(), 128); // sticky
    }

    #[test]
    fn invalid_alignment_rejected() {
        let pool = UnifiedMemoryPool::new(256).unwrap();
        assert!(pool.allocate(8, 0).is_err());
        assert!(pool.allocate(8, 3).is_err()); // not a power of two
        assert!(pool.allocate(0, 8).is_err());
    }

    #[test]
    fn excessive_alignment_rejected() {
        let pool = UnifiedMemoryPool::new(4096).unwrap();
        // Power-of-two alignment beyond the 1 GiB cap must be rejected
        // (would otherwise overflow when computing the aligned bump).
        let huge = (1usize << 31).max(1 << 30) + 1; // > 1 GiB
        // Use a clean power-of-two above the cap.
        let too_big = 1usize << 31;
        assert!(pool.allocate(8, too_big).is_err());
        let _ = huge; // silence unused
    }

    #[test]
    fn writes_visible_in_slice() {
        let pool = UnifiedMemoryPool::new(256).unwrap();
        let mut a = pool.allocate(16, 8).unwrap();
        for (i, byte) in a.as_mut_slice().iter_mut().enumerate() {
            *byte = i as u8;
        }
        for (i, &byte) in a.as_slice().iter().enumerate() {
            assert_eq!(byte, i as u8);
        }
    }

    #[test]
    fn recycled_allocation_reads_as_zero() {
        // Cross-tenant data-leak regression test (audit H1):
        // -------------------------------------------------------------------
        // Simulates the recycle path: tenant A writes 0xAB into a pool
        // allocation, releases it, the pool is reset, and tenant B requests
        // an allocation that lands on the same bytes. The recycled bytes
        // MUST read as zero — otherwise tenant B can observe tenant A's
        // private data. This pins the `ptr::write_bytes` zero-fill added in
        // `UnifiedMemoryPool::allocate`.
        const SIZE: usize = 4 * 1024; // 4 KiB — large enough to detect any
                                       // off-by-one in the memset bounds.
        let pool = UnifiedMemoryPool::new(64 * 1024).unwrap();

        // Tenant A: poison every byte with the sentinel.
        {
            let mut a = pool.allocate(SIZE, 64).unwrap();
            a.as_mut_slice().fill(0xAB);
            // sanity: poison actually went in
            assert!(a.as_slice().iter().all(|&b| b == 0xAB));
        }
        // Reset the bump pointer so the same byte range is reachable again.
        pool.reset().expect("reset after tenant A drops");

        // Tenant B: ask for the same shape and assert every byte reads zero.
        let b = pool.allocate(SIZE, 64).unwrap();
        let leaked: Vec<usize> = b
            .as_slice()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(i) } else { None })
            .collect();
        assert!(
            leaked.is_empty(),
            "recycled pool allocation leaked tenant-A data at {} byte offsets (first: {:?})",
            leaked.len(),
            leaked.first(),
        );
    }

    #[test]
    fn first_allocation_reads_as_zero() {
        // Companion to `recycled_allocation_reads_as_zero`: even the *first*
        // allocation out of a fresh pool must read as zero. On the heap
        // backing (`Box<[u8]>`) this is trivially true because `vec![0u8; n]`
        // zeroes. On the cust path, `cust::memory::UnifiedBuffer::new(&0u8,
        // size)` seeds with zero. On the cudarc path, `cuMemAllocManaged`
        // does NOT zero — the `unified.rs` change closes that hole. Either
        // way the contract surfaced by `UnifiedMemoryPool::allocate` must be
        // identical across all three backings, so we test it here.
        let pool = UnifiedMemoryPool::new(4 * 1024).unwrap();
        let a = pool.allocate(1024, 16).unwrap();
        assert!(
            a.as_slice().iter().all(|&b| b == 0),
            "first allocation out of fresh pool must read as zero"
        );
    }
}
