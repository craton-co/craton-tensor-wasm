// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! [`DeviceMemRegistry`] — instance-scoped store of explicit device-memory
//! allocations.
//!
//! The `wasi:cuda` host surface historically relied on CUDA Unified Memory:
//! a guest pointer argument was bounds-checked against linear memory and
//! handed to `cuLaunchKernel` verbatim, the UVM driver migrating pages on
//! demand. That limits portability to UVM-capable setups. The explicit
//! device-buffer surface (`alloc` / `free` / `memcpy-h2d` / `memcpy-d2h`)
//! lets a guest manage discrete device buffers that work on any CUDA host.
//!
//! This registry is the host-side bookkeeping for those buffers. It mirrors
//! [`crate::registry::KernelRegistry`] exactly:
//!
//!   * every allocation is tagged with its owning [`InstanceId`], and
//!     [`DeviceMemRegistry::lookup`] / [`DeviceMemRegistry::free`] refuse a
//!     handle that belongs to a different instance (`AbiError::InvalidHandle`)
//!     — a guest cannot forge another instance's handle;
//!   * an aggregate-bytes cap ([`MAX_TOTAL_DEVICE_BYTES`]) is enforced with a
//!     compare-and-swap loop so one instance cannot pin unbounded device
//!     memory before tripping the per-instance count cap
//!     ([`MAX_DEVICE_ALLOCS_PER_INSTANCE`]).
//!
//! ## Process-wide aggregate ceiling (M-3)
//!
//! The per-instance caps above are defence-in-depth but do not bound the
//! *aggregate* across instances: each [`WasiCudaContext`] historically owns
//! its own registry, so N instances could each admit up to
//! [`MAX_TOTAL_DEVICE_BYTES`] with no global ceiling. Every registry therefore
//! also charges a shared [`DeviceMemBudget`] — a process-wide counter of live
//! device bytes and allocations — and refuses an `insert` that would push the
//! shared total over [`MAX_PROCESS_DEVICE_BYTES`] (or the shared count over
//! [`MAX_PROCESS_DEVICE_ALLOCS`]). By default every [`DeviceMemRegistry::new`]
//! attaches to the same [`process_device_budget`] singleton, so the ceiling is
//! genuinely process-wide regardless of how many contexts are created. This
//! mirrors the intent of the shared [`crate::async_dispatch::BackPressure`]
//! cap (one process-wide ceiling shared by every instance), but does not
//! depend on the executor remembering to thread a clone: the singleton is the
//! default. Tests that need an isolated budget use
//! [`DeviceMemRegistry::with_budget`].
//!
//! [`WasiCudaContext`]: crate::host::WasiCudaContext
//! On no-CUDA builds the registry records the *requested* size and a synthetic
//! handle (no real `cuMemAlloc` runs); the host functions validate arguments
//! and then return [`AbiError::NotAvailable`]. On CUDA builds the registry
//! additionally carries the real device pointer (`CUdeviceptr`) so the
//! `memcpy` paths can drive `cuMemcpyHtoD` / `cuMemcpyDtoH`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use tensor_wasm_core::types::InstanceId;

use crate::abi::AbiError;

/// Maximum size of a single `alloc` request, in bytes (256 MiB).
///
/// A request above this cap is rejected with [`AbiError::QuotaExceeded`]
/// before any driver call. Sized to comfortably hold a large tensor tile
/// while bounding the damage a single hostile `alloc` can do.
pub const MAX_DEVICE_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum number of live device allocations a single instance may hold.
pub const MAX_DEVICE_ALLOCS_PER_INSTANCE: usize = 4096;

/// Soft cap on aggregate retained device bytes per instance (sum of live
/// allocation sizes). Set to 16x the per-call cap (4 GiB) so a single
/// instance cannot pin unbounded device memory below the per-call ceiling
/// before tripping [`MAX_DEVICE_ALLOCS_PER_INSTANCE`].
pub const MAX_TOTAL_DEVICE_BYTES: u64 = 16 * MAX_DEVICE_ALLOC_BYTES;

/// Process-wide ceiling on aggregate live device bytes across *every*
/// instance's registry (M-3).
///
/// The per-instance [`MAX_TOTAL_DEVICE_BYTES`] cap bounds one registry, but
/// without a shared ceiling N instances could each pin that much, so the true
/// process exposure scales with the instance count. This constant is the
/// shared upper bound that the [`DeviceMemBudget`] enforces in addition to the
/// per-instance caps.
///
/// Sized at 4x the per-instance aggregate cap (64 GiB): generous enough that a
/// handful of well-behaved instances are never throttled by it, while still
/// putting a finite ceiling on total pinned device memory so a fan-out of
/// instances cannot collectively exhaust the device. The multiple (rather than
/// equality with the per-instance cap) keeps the per-instance cap meaningful
/// as the *first* line of defence — a single runaway instance trips its own
/// cap long before it can monopolise the process budget. Hosts with smaller
/// GPUs should treat this as an upper bound and rely on the per-instance cap
/// plus the real `cuMemAlloc` failure for tighter physical limits.
pub const MAX_PROCESS_DEVICE_BYTES: u64 = 4 * MAX_TOTAL_DEVICE_BYTES;

/// Process-wide ceiling on the number of live device allocations across every
/// instance's registry (M-3).
///
/// Mirrors [`MAX_PROCESS_DEVICE_BYTES`] for the count axis: 4x the per-instance
/// count cap, so the per-instance count cap remains the first tripwire while a
/// process-wide ceiling still bounds total live handles (each of which costs a
/// driver allocation and registry slot).
pub const MAX_PROCESS_DEVICE_ALLOCS: usize = 4 * MAX_DEVICE_ALLOCS_PER_INSTANCE;

/// Process-wide aggregate device-memory budget shared by every
/// [`DeviceMemRegistry`].
///
/// Tracks the sum of live allocation sizes and the count of live allocations
/// across all instances. A registry charges this budget on `insert` (refusing
/// the allocation with [`AbiError::QuotaExceeded`] when it would push the
/// shared total over [`MAX_PROCESS_DEVICE_BYTES`] / [`MAX_PROCESS_DEVICE_ALLOCS`])
/// and credits it back on `free` / drop. All arithmetic is checked /
/// saturating so a bookkeeping slip can never wrap the counters.
///
/// This is the device-memory analogue of the shared
/// [`crate::async_dispatch::BackPressure`] cap: one process-wide ceiling shared
/// by every instance. Unlike `BackPressure`, the default registry constructor
/// attaches to the [`process_device_budget`] singleton automatically, so the
/// ceiling holds even if an embedder constructs contexts without explicitly
/// threading shared state.
#[derive(Debug, Default)]
pub struct DeviceMemBudget {
    /// Sum of live allocation sizes across all attached registries.
    total_bytes: AtomicU64,
    /// Count of live allocations across all attached registries.
    total_allocs: AtomicU64,
}

impl DeviceMemBudget {
    /// Construct an empty budget. Most callers want the shared
    /// [`process_device_budget`] singleton instead; this is for tests that
    /// need an isolated budget.
    pub fn new() -> Self {
        Self {
            total_bytes: AtomicU64::new(0),
            total_allocs: AtomicU64::new(0),
        }
    }

    /// Reserve `size` bytes (one allocation) against the process-wide budget.
    ///
    /// Uses a compare-and-swap loop on the byte counter so the check + add is
    /// atomic against concurrent allocations from other instances. On success
    /// the count is bumped and `Ok(())` returned; if the reservation would
    /// breach either process cap, nothing is mutated and
    /// [`AbiError::QuotaExceeded`] is returned.
    fn try_reserve(&self, size: u64) -> Result<(), AbiError> {
        // Count axis first: a single fetch_add we roll back if it overshoots,
        // mirroring the byte CAS below. Doing the count check up front keeps
        // the (rarer) byte CAS from spinning when the count cap is the gate.
        let prev_allocs = self.total_allocs.fetch_add(1, Ordering::AcqRel);
        if prev_allocs >= MAX_PROCESS_DEVICE_ALLOCS as u64 {
            // Roll back the speculative bump and refuse.
            self.total_allocs.fetch_sub(1, Ordering::AcqRel);
            return Err(AbiError::QuotaExceeded);
        }
        let mut current = self.total_bytes.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(size);
            if next > MAX_PROCESS_DEVICE_BYTES {
                // Byte cap tripped — release the count reservation we took.
                self.total_allocs.fetch_sub(1, Ordering::AcqRel);
                return Err(AbiError::QuotaExceeded);
            }
            match self.total_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    /// Credit `size` bytes (one allocation) back to the budget on free / drop.
    /// Saturating so a double-credit can never wrap below zero.
    fn release(&self, size: u64) {
        let _ = self
            .total_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(size))
            });
        let _ = self
            .total_allocs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(1))
            });
    }

    /// Aggregate live device bytes across every attached registry. Visible for
    /// metrics and tests.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Acquire)
    }

    /// Aggregate live allocation count across every attached registry. Visible
    /// for metrics and tests.
    pub fn total_allocs(&self) -> u64 {
        self.total_allocs.load(Ordering::Acquire)
    }
}

/// The process-wide [`DeviceMemBudget`] singleton.
///
/// Every [`DeviceMemRegistry::new`] attaches to this, so the
/// [`MAX_PROCESS_DEVICE_BYTES`] / [`MAX_PROCESS_DEVICE_ALLOCS`] ceiling is
/// enforced across all instances in the process without the embedder having to
/// thread shared state explicitly. Tests that need isolation construct their
/// own budget via [`DeviceMemRegistry::with_budget`].
pub fn process_device_budget() -> &'static Arc<DeviceMemBudget> {
    static BUDGET: OnceLock<Arc<DeviceMemBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| Arc::new(DeviceMemBudget::new()))
}

/// Metadata about a single device-memory allocation.
#[derive(Debug)]
pub struct DeviceMemEntry {
    /// Owning instance (used to authorise `free` / `memcpy` calls).
    pub owner: InstanceId,
    /// Requested allocation size in bytes.
    pub size: u64,
    /// CUDA device pointer; only meaningful when the `cuda` feature is
    /// enabled. On no-CUDA builds the field is absent so the registry can
    /// still be exercised by tests.
    #[cfg(feature = "cuda")]
    pub device_ptr: cust::sys::CUdeviceptr,
}

/// Cheap handle to a device-memory entry's stable fields.
///
/// Returned by [`DeviceMemRegistry::lookup`]. Carries the device pointer on
/// CUDA builds so the `memcpy` paths can drive the driver without holding the
/// `dashmap` entry borrow across the copy.
#[derive(Clone, Debug)]
pub struct DeviceMemHandle {
    /// Owning instance.
    pub owner: InstanceId,
    /// Allocation size in bytes.
    pub size: u64,
    /// Device pointer on CUDA builds.
    #[cfg(feature = "cuda")]
    pub device_ptr: cust::sys::CUdeviceptr,
}

/// Instance-scoped device-memory registry.
pub struct DeviceMemRegistry {
    next_handle: AtomicU64,
    entries: DashMap<u64, DeviceMemEntry>,
    /// Sum of live allocation sizes. Tracked separately so `insert` can
    /// reject above [`MAX_TOTAL_DEVICE_BYTES`] without scanning the map.
    total_device_bytes: AtomicU64,
    /// Process-wide aggregate budget this registry charges in addition to its
    /// own per-instance caps (M-3). Shared by every registry created via
    /// [`Self::new`] (the [`process_device_budget`] singleton); tests may
    /// supply an isolated budget via [`Self::with_budget`].
    budget: Arc<DeviceMemBudget>,
}

impl Default for DeviceMemRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceMemRegistry {
    /// Construct an empty registry.
    ///
    /// Handles start at 1 so callers may reserve 0 as a sentinel; they are
    /// otherwise sequential. Cross-instance forgery is prevented by the
    /// owner-`InstanceId` check in [`Self::lookup`] / [`Self::free`], so the
    /// handle space does not need the randomised-seed treatment the kernel
    /// registry uses for its ids.
    pub fn new() -> Self {
        Self::with_budget(Arc::clone(process_device_budget()))
    }

    /// Construct an empty registry charging the given process-wide
    /// [`DeviceMemBudget`] instead of the shared [`process_device_budget`]
    /// singleton.
    ///
    /// Intended for tests that need an isolated aggregate so the process-wide
    /// caps can be exercised without the global singleton bleeding state
    /// across test cases (which run in the same process). Production code
    /// should use [`Self::new`] so every instance shares the one ceiling.
    pub fn with_budget(budget: Arc<DeviceMemBudget>) -> Self {
        Self {
            next_handle: AtomicU64::new(1),
            entries: DashMap::new(),
            total_device_bytes: AtomicU64::new(0),
            budget,
        }
    }

    /// Reserve aggregate-bytes budget for a new allocation, returning the
    /// freshly-assigned handle on success.
    ///
    /// Enforces [`MAX_DEVICE_ALLOCS_PER_INSTANCE`] and
    /// [`MAX_TOTAL_DEVICE_BYTES`] (the per-instance caps) **and** the
    /// process-wide [`MAX_PROCESS_DEVICE_BYTES`] / [`MAX_PROCESS_DEVICE_ALLOCS`]
    /// caps via the shared [`DeviceMemBudget`] (all returning
    /// [`AbiError::QuotaExceeded`]) before inserting the entry. The per-call
    /// [`MAX_DEVICE_ALLOC_BYTES`] cap is the caller's responsibility (the host
    /// function checks it before any driver call) — this method only enforces
    /// the aggregate caps so the check + add stays atomic against concurrent
    /// allocations.
    ///
    /// The per-instance caps are checked first (defence in depth: a single
    /// runaway instance trips its own cap before it can charge the shared
    /// budget), then the process-wide budget. If the process budget refuses,
    /// the per-instance byte reservation is rolled back so the two counters
    /// never drift.
    pub fn insert(&self, entry: DeviceMemEntry) -> Result<u64, AbiError> {
        if self.entries.len() >= MAX_DEVICE_ALLOCS_PER_INSTANCE {
            return Err(AbiError::QuotaExceeded);
        }
        let add = entry.size;
        // Compare-and-swap loop so the check + add is atomic against
        // concurrent allocations — mirrors `KernelRegistry::register`.
        let mut current = self.total_device_bytes.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(add);
            if next > MAX_TOTAL_DEVICE_BYTES {
                return Err(AbiError::QuotaExceeded);
            }
            match self.total_device_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        // Charge the process-wide budget. On refusal, release the
        // per-instance reservation we just took so the counters stay in sync.
        if let Err(e) = self.budget.try_reserve(add) {
            let _ =
                self.total_device_bytes
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                        Some(cur.saturating_sub(add))
                    });
            return Err(e);
        }
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.entries.insert(handle, entry);
        Ok(handle)
    }

    /// Look up an allocation by handle, returning an independent handle copy.
    ///
    /// Returns `Err(AbiError::InvalidHandle)` if the handle is unknown or
    /// belongs to a different instance.
    pub fn lookup(&self, handle: u64, owner: InstanceId) -> Result<DeviceMemHandle, AbiError> {
        let r = self.entries.get(&handle).ok_or(AbiError::InvalidHandle)?;
        if r.owner != owner {
            return Err(AbiError::InvalidHandle);
        }
        Ok(DeviceMemHandle {
            owner: r.owner,
            size: r.size,
            #[cfg(feature = "cuda")]
            device_ptr: r.device_ptr,
        })
    }

    /// Remove an allocation owned by `owner`, returning its entry.
    ///
    /// Returns `Err(AbiError::InvalidHandle)` when the handle is unknown or
    /// belongs to another instance — a guest cannot free a buffer it does not
    /// own. On success both the per-instance aggregate-bytes counter and the
    /// shared process-wide [`DeviceMemBudget`] are credited back.
    pub fn free(&self, handle: u64, owner: InstanceId) -> Result<DeviceMemEntry, AbiError> {
        // Authorise the owner before removing so a cross-owner `free` cannot
        // even observe whether the handle exists (it always sees
        // `InvalidHandle`, matching the `lookup` discrimination).
        {
            let r = self.entries.get(&handle).ok_or(AbiError::InvalidHandle)?;
            if r.owner != owner {
                return Err(AbiError::InvalidHandle);
            }
        }
        let (_, entry) = self
            .entries
            .remove(&handle)
            .ok_or(AbiError::InvalidHandle)?;
        let _ = self
            .total_device_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(entry.size))
            });
        self.budget.release(entry.size);
        Ok(entry)
    }

    /// Number of currently-live allocations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if there are no live allocations.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Aggregate device bytes currently retained. Visible for metrics and
    /// tests.
    pub fn total_device_bytes(&self) -> u64 {
        self.total_device_bytes.load(Ordering::Acquire)
    }

    /// Borrow the process-wide [`DeviceMemBudget`] this registry charges.
    /// Visible for metrics and tests.
    pub fn budget(&self) -> &Arc<DeviceMemBudget> {
        &self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(owner: InstanceId, size: u64) -> DeviceMemEntry {
        DeviceMemEntry {
            owner,
            size,
            #[cfg(feature = "cuda")]
            device_ptr: 0,
        }
    }

    #[test]
    fn insert_then_lookup() {
        let reg = DeviceMemRegistry::new();
        let h = reg.insert(entry(InstanceId(1), 4096)).unwrap();
        let found = reg.lookup(h, InstanceId(1)).unwrap();
        assert_eq!(found.owner, InstanceId(1));
        assert_eq!(found.size, 4096);
    }

    #[test]
    fn lookup_wrong_owner_rejected() {
        let reg = DeviceMemRegistry::new();
        let h = reg.insert(entry(InstanceId(1), 4096)).unwrap();
        assert_eq!(
            reg.lookup(h, InstanceId(2)).unwrap_err(),
            AbiError::InvalidHandle
        );
    }

    #[test]
    fn free_wrong_owner_rejected() {
        let reg = DeviceMemRegistry::new();
        let h = reg.insert(entry(InstanceId(1), 4096)).unwrap();
        assert_eq!(
            reg.free(h, InstanceId(2)).unwrap_err(),
            AbiError::InvalidHandle
        );
        // The entry is still present and still owned by instance 1.
        assert!(reg.lookup(h, InstanceId(1)).is_ok());
    }

    #[test]
    fn free_unknown_rejected() {
        let reg = DeviceMemRegistry::new();
        assert_eq!(
            reg.free(999, InstanceId(1)).unwrap_err(),
            AbiError::InvalidHandle
        );
    }

    #[test]
    fn alloc_free_lifecycle_tracks_bytes() {
        let reg = DeviceMemRegistry::new();
        assert!(reg.is_empty());
        let h = reg.insert(entry(InstanceId(1), 8192)).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.total_device_bytes(), 8192);
        let freed = reg.free(h, InstanceId(1)).unwrap();
        assert_eq!(freed.size, 8192);
        assert!(reg.is_empty());
        assert_eq!(reg.total_device_bytes(), 0);
        // The handle is gone now; a second free fails.
        assert_eq!(
            reg.free(h, InstanceId(1)).unwrap_err(),
            AbiError::InvalidHandle
        );
    }

    /// A registry charging an isolated [`DeviceMemBudget`] so process-wide
    /// counters cannot bleed across tests (all unit tests share one process,
    /// hence one [`process_device_budget`] singleton).
    fn iso_reg() -> DeviceMemRegistry {
        DeviceMemRegistry::with_budget(Arc::new(DeviceMemBudget::new()))
    }

    #[test]
    fn aggregate_byte_cap_enforced() {
        let reg = iso_reg();
        let per = MAX_DEVICE_ALLOC_BYTES; // 256 MiB
        let cap_count = (MAX_TOTAL_DEVICE_BYTES / per) as usize; // 16
        for _ in 0..cap_count {
            reg.insert(entry(InstanceId(1), per)).expect("under cap");
        }
        // The next allocation at the per-call max trips the aggregate cap.
        assert_eq!(
            reg.insert(entry(InstanceId(1), per)).unwrap_err(),
            AbiError::QuotaExceeded
        );
    }

    #[test]
    fn per_instance_count_cap_enforced() {
        let reg = iso_reg();
        // Tiny allocations so the count cap (not the byte cap) gates.
        for _ in 0..MAX_DEVICE_ALLOCS_PER_INSTANCE {
            reg.insert(entry(InstanceId(1), 1))
                .expect("under count cap");
        }
        assert_eq!(
            reg.insert(entry(InstanceId(1), 1)).unwrap_err(),
            AbiError::QuotaExceeded
        );
    }

    /// Two registries sharing one [`DeviceMemBudget`] (the production setup,
    /// where every instance's registry charges the same process-wide budget)
    /// cannot collectively exceed [`MAX_PROCESS_DEVICE_BYTES`], even though
    /// neither has tripped its own per-instance [`MAX_TOTAL_DEVICE_BYTES`] cap.
    #[test]
    fn process_byte_cap_enforced_across_registries() {
        let budget = Arc::new(DeviceMemBudget::new());
        let per = MAX_DEVICE_ALLOC_BYTES; // 256 MiB
                                          // Per-instance aggregate cap holds this many per-call-max allocations.
        let per_reg = (MAX_TOTAL_DEVICE_BYTES / per) as usize; // 16
                                                               // Process cap is 4x the per-instance cap, so it takes 4 saturated
                                                               // registries to reach it.
        let regs: Vec<DeviceMemRegistry> = (0..4)
            .map(|_| DeviceMemRegistry::with_budget(Arc::clone(&budget)))
            .collect();
        for (i, reg) in regs.iter().enumerate() {
            for _ in 0..per_reg {
                reg.insert(entry(InstanceId(i as u128 + 1), per))
                    .expect("each registry stays under its own per-instance cap");
            }
        }
        // Every registry is at its per-instance cap and the shared budget is
        // at the process ceiling. A fresh registry — well under its own
        // per-instance cap — is refused on the process budget.
        assert_eq!(budget.total_bytes(), MAX_PROCESS_DEVICE_BYTES);
        let extra = DeviceMemRegistry::with_budget(Arc::clone(&budget));
        assert_eq!(
            extra.insert(entry(InstanceId(99), per)).unwrap_err(),
            AbiError::QuotaExceeded
        );
        // Freeing one allocation credits the shared budget back, re-admitting
        // exactly one more allocation.
        let freed_handle = 1u64; // first handle in regs[0]
        regs[0]
            .free(freed_handle, InstanceId(1))
            .expect("free a live allocation");
        assert_eq!(budget.total_bytes(), MAX_PROCESS_DEVICE_BYTES - per);
        let h = extra
            .insert(entry(InstanceId(99), per))
            .expect("re-admitted after a free freed shared headroom");
        assert!(extra.lookup(h, InstanceId(99)).is_ok());
    }

    /// The process-wide *count* cap ([`MAX_PROCESS_DEVICE_ALLOCS`]) is enforced
    /// across registries sharing one budget, independent of the byte cap.
    #[test]
    fn process_alloc_count_cap_enforced_across_registries() {
        let budget = Arc::new(DeviceMemBudget::new());
        // Tiny 1-byte allocations so the count cap (not the byte cap) gates.
        // Spread across enough registries that no single one trips its own
        // per-instance count cap first.
        let regs: Vec<DeviceMemRegistry> = (0..4)
            .map(|_| DeviceMemRegistry::with_budget(Arc::clone(&budget)))
            .collect();
        for reg in &regs {
            for _ in 0..MAX_DEVICE_ALLOCS_PER_INSTANCE {
                reg.insert(entry(InstanceId(1), 1))
                    .expect("under count cap");
            }
        }
        assert_eq!(budget.total_allocs(), MAX_PROCESS_DEVICE_ALLOCS as u64);
        let extra = DeviceMemRegistry::with_budget(Arc::clone(&budget));
        assert_eq!(
            extra.insert(entry(InstanceId(1), 1)).unwrap_err(),
            AbiError::QuotaExceeded
        );
    }

    /// A `free` credits the shared budget back so the process counters track
    /// *live* bytes / allocations, not cumulative.
    #[test]
    fn free_credits_shared_budget() {
        let budget = Arc::new(DeviceMemBudget::new());
        let reg = DeviceMemRegistry::with_budget(Arc::clone(&budget));
        let h = reg.insert(entry(InstanceId(1), 8192)).unwrap();
        assert_eq!(budget.total_bytes(), 8192);
        assert_eq!(budget.total_allocs(), 1);
        reg.free(h, InstanceId(1)).unwrap();
        assert_eq!(budget.total_bytes(), 0);
        assert_eq!(budget.total_allocs(), 0);
    }

    #[test]
    fn handles_are_unique_and_increasing() {
        let reg = DeviceMemRegistry::new();
        let a = reg.insert(entry(InstanceId(1), 1)).unwrap();
        let b = reg.insert(entry(InstanceId(1), 1)).unwrap();
        assert_ne!(a, b);
        assert_eq!(a + 1, b);
    }
}
