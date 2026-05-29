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
//! On no-CUDA builds the registry records the *requested* size and a synthetic
//! handle (no real `cuMemAlloc` runs); the host functions validate arguments
//! and then return [`AbiError::NotAvailable`]. On CUDA builds the registry
//! additionally carries the real device pointer (`CUdeviceptr`) so the
//! `memcpy` paths can drive `cuMemcpyHtoD` / `cuMemcpyDtoH`.

use std::sync::atomic::{AtomicU64, Ordering};

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
        Self {
            next_handle: AtomicU64::new(1),
            entries: DashMap::new(),
            total_device_bytes: AtomicU64::new(0),
        }
    }

    /// Reserve aggregate-bytes budget for a new allocation, returning the
    /// freshly-assigned handle on success.
    ///
    /// Enforces [`MAX_DEVICE_ALLOCS_PER_INSTANCE`] and
    /// [`MAX_TOTAL_DEVICE_BYTES`] (returning [`AbiError::QuotaExceeded`]) and
    /// inserts the entry. The per-call [`MAX_DEVICE_ALLOC_BYTES`] cap is the
    /// caller's responsibility (the host function checks it before any driver
    /// call) — this method only enforces the aggregate caps so the check + add
    /// stays atomic against concurrent allocations.
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
    /// own. On success the aggregate-bytes counter is decremented.
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
        let (_, entry) = self.entries.remove(&handle).ok_or(AbiError::InvalidHandle)?;
        let _ = self
            .total_device_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(entry.size))
            });
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

    #[test]
    fn aggregate_byte_cap_enforced() {
        let reg = DeviceMemRegistry::new();
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
        let reg = DeviceMemRegistry::new();
        // Tiny allocations so the count cap (not the byte cap) gates.
        for _ in 0..MAX_DEVICE_ALLOCS_PER_INSTANCE {
            reg.insert(entry(InstanceId(1), 1)).expect("under count cap");
        }
        assert_eq!(
            reg.insert(entry(InstanceId(1), 1)).unwrap_err(),
            AbiError::QuotaExceeded
        );
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
