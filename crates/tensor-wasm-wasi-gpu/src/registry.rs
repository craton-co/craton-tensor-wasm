// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! [`KernelRegistry`] — instance-scoped store of compiled PTX kernels.
//!
//! Every Wasm instance gets its own registry. Kernel IDs are scoped to the
//! owning [`InstanceId`]; the host functions
//! refuse to launch a kernel using an ID that belongs to a different
//! instance (`AbiError::InvalidKernel`).
//!
//! ## Memory safety: why `Arc<cust::module::Module>`
//!
//! The compiled module is held behind an `Arc` so that `lookup` can hand back
//! a strong reference whose lifetime is independent of the underlying
//! `dashmap` entry. A previous design returned a raw pointer materialised
//! from a transient `dashmap::Ref` — that ref was dropped before the launch
//! site dereferenced the pointer, producing a use-after-free under
//! concurrent `register` + `remove`. With `Arc`, `lookup` clones the
//! pointer-bump (atomic +1) and the launch path holds the strong reference
//! for the full duration of `cuLaunchKernel` and the subsequent
//! `stream.synchronize()`. Concurrent `remove` only decrements the strong
//! count; the module is dropped (and CUDA's `cuModuleUnload` runs) once the
//! last launch has finished.

use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use tensor_wasm_core::types::{InstanceId, KernelId};
use dashmap::DashMap;

use crate::abi::{AbiError, MAX_KERNELS_PER_INSTANCE, MAX_PTX_BYTES};

/// Soft cap on aggregate retained PTX bytes per instance (sum of
/// `ptx_bytes_len` across live entries). Set to 64x the per-call cap so a
/// single malicious instance cannot pin 2 GiB of host memory in the
/// registry before tripping `MAX_KERNELS_PER_INSTANCE`.
pub const MAX_TOTAL_PTX_BYTES: usize = 64 * MAX_PTX_BYTES;

/// Metadata about a single compiled kernel. The actual compiled module is
/// held opaquely behind a feature gate — on non-CUDA builds the field is
/// absent so the registry can still be exercised by tests.
#[derive(Debug)]
pub struct KernelEntry {
    /// Owning instance (used to authorise `launch` calls).
    pub owner: InstanceId,
    /// Entry-point symbol name inside the PTX module.
    pub entry: String,
    /// Size of the PTX source that produced this kernel (bytes).
    pub ptx_bytes_len: usize,
    /// CUDA-side handle; only meaningful when the `cuda` feature is enabled.
    ///
    /// Held in an `Arc` so the launch path can take a strong reference
    /// independent of the `dashmap` entry's lifetime — preventing a UAF
    /// under concurrent `register`/`remove`.
    #[cfg(feature = "cuda")]
    pub module: Option<Arc<cust::module::Module>>,
}

/// Instance-scoped kernel registry.
pub struct KernelRegistry {
    next_id: AtomicU64,
    entries: DashMap<KernelId, KernelEntry>,
    /// Sum of `ptx_bytes_len` across live entries. Tracked separately so
    /// `register` can reject above [`MAX_TOTAL_PTX_BYTES`] without scanning
    /// the whole map.
    total_ptx_bytes: AtomicU64,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Cheap, cloneable handle to a kernel entry's stable fields.
///
/// Returned by [`KernelRegistry::lookup`]. The handle owns an `Arc` to the
/// compiled module on CUDA builds, so callers may drop the originating
/// `dashmap` ref before launching.
#[derive(Clone, Debug)]
pub struct KernelHandle {
    /// Owning instance.
    pub owner: InstanceId,
    /// Entry-point symbol name.
    pub entry: String,
    /// PTX source size in bytes.
    pub ptx_bytes_len: usize,
    /// Strong reference to the compiled module on CUDA builds.
    #[cfg(feature = "cuda")]
    pub module: Option<Arc<cust::module::Module>>,
}

/// Pick a random seed for [`KernelRegistry::next_id`] so that kernel ids are
/// not trivially guessable (wasi-gpu 1.1).
///
/// Cross-tenant id forgery is already prevented by the owner-`InstanceId`
/// check in [`KernelRegistry::lookup`], so this seed is defence-in-depth: it
/// stops a guest from enumerating its own id space without bookkeeping (and,
/// by extension, fingerprinting which ids exist by observing the
/// `InvalidKernel`-vs-`InvalidDimensions` discrimination on `launch`).
///
/// The seed is sampled from the OS RNG. We reserve the high bit so callers
/// reading the id as a signed `i64` from the WIT interface stay in the
/// positive range, and we mask off the bottom 16 bits so the first few
/// thousand registrations don't land suspiciously close to 0.
fn random_kernel_id_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    // `RandomState` seeds itself from the OS RNG on construction. Using two
    // hashers gives us 64 bits of entropy without taking an explicit
    // dependency on `rand` (which the workspace does not currently pull
    // into this crate). The construction is `std`-only.
    let mut h = RandomState::new().build_hasher();
    h.write_u64(0xa5a5_a5a5_a5a5_a5a5);
    let raw = h.finish();
    // Clear the high bit (keep ids in the positive `i64` range) and the
    // low 16 bits (avoid an obviously-small starting value). The +1 floor
    // keeps the invariant `next_id != 0` so callers can reserve 0 as a
    // sentinel if they wish.
    (raw & 0x7fff_ffff_ffff_0000) | 1
}

impl KernelRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            // Seeded with `random_kernel_id_seed` so the id space is
            // unguessable (wasi-gpu 1.1).
            next_id: AtomicU64::new(random_kernel_id_seed()),
            entries: DashMap::new(),
            total_ptx_bytes: AtomicU64::new(0),
        }
    }

    /// Register a new kernel and return its assigned [`KernelId`].
    ///
    /// Returns [`AbiError::QuotaExceeded`] if either the per-instance kernel
    /// count cap ([`MAX_KERNELS_PER_INSTANCE`]) or the aggregate PTX-bytes
    /// cap ([`MAX_TOTAL_PTX_BYTES`]) would be exceeded.
    pub fn register(&self, entry: KernelEntry) -> Result<KernelId, AbiError> {
        if self.entries.len() >= MAX_KERNELS_PER_INSTANCE {
            return Err(AbiError::QuotaExceeded);
        }
        // Aggregate-bytes check: prevents an instance from pinning gigabytes
        // of registry memory with many large PTX modules below the per-call
        // cap.
        let add = entry.ptx_bytes_len as u64;
        // Use a compare-and-swap loop so the check + add is atomic against
        // concurrent registrations.
        let mut current = self.total_ptx_bytes.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(add);
            if next > MAX_TOTAL_PTX_BYTES as u64 {
                return Err(AbiError::QuotaExceeded);
            }
            match self.total_ptx_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        let id = KernelId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.entries.insert(id, entry);
        Ok(id)
    }

    /// Look up a kernel by id and return an independent handle.
    ///
    /// Returns `Err(AbiError::InvalidKernel)` if the id is unknown or belongs
    /// to a different instance. The returned [`KernelHandle`] holds an
    /// `Arc<Module>` on CUDA builds, so callers may drop any borrowed
    /// `dashmap` entry before performing the launch — eliminating the UAF
    /// that the previous `dashmap::Ref`-derived raw-pointer scheme had.
    pub fn lookup(&self, id: KernelId, owner: InstanceId) -> Result<KernelHandle, AbiError> {
        let r = self.entries.get(&id).ok_or(AbiError::InvalidKernel)?;
        if r.owner != owner {
            return Err(AbiError::InvalidKernel);
        }
        Ok(KernelHandle {
            owner: r.owner,
            entry: r.entry.clone(),
            ptx_bytes_len: r.ptx_bytes_len,
            #[cfg(feature = "cuda")]
            module: r.module.clone(),
        })
    }

    /// Remove a kernel from the registry (caller releases its handle).
    ///
    /// On CUDA builds the underlying `cust::module::Module` is held inside
    /// an `Arc`; dropping the registry entry only decrements the strong
    /// count. If a concurrent `launch` is still holding a clone of the
    /// `Arc`, the module is not actually unloaded until that launch returns
    /// — eliminating the UAF window that existed when the registry owned
    /// the module by value.
    pub fn remove(&self, id: KernelId) -> Option<KernelEntry> {
        let (_, entry) = self.entries.remove(&id)?;
        // Decrement aggregate-bytes counter. Saturating_sub guards against
        // underflow if ever the counter drifts; the entry's own
        // `ptx_bytes_len` is the source of truth.
        let _ = self.total_ptx_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |cur| Some(cur.saturating_sub(entry.ptx_bytes_len as u64)),
        );
        Some(entry)
    }

    /// Number of currently-registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if there are no registered kernels.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Aggregate PTX bytes currently retained by the registry. Visible for
    /// metrics and tests.
    pub fn total_ptx_bytes(&self) -> u64 {
        self.total_ptx_bytes.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(owner: InstanceId, entry: &str) -> KernelEntry {
        KernelEntry {
            owner,
            entry: entry.into(),
            ptx_bytes_len: 1024,
            #[cfg(feature = "cuda")]
            module: None,
        }
    }

    fn make_entry_sized(owner: InstanceId, entry: &str, bytes: usize) -> KernelEntry {
        KernelEntry {
            owner,
            entry: entry.into(),
            ptx_bytes_len: bytes,
            #[cfg(feature = "cuda")]
            module: None,
        }
    }

    #[test]
    fn register_then_lookup() {
        let reg = KernelRegistry::new();
        let id = reg
            .register(make_entry(InstanceId(1), "vector_add"))
            .unwrap();
        let entry = reg.lookup(id, InstanceId(1)).unwrap();
        assert_eq!(entry.owner, InstanceId(1));
        assert_eq!(entry.entry, "vector_add");
    }

    #[test]
    fn lookup_wrong_owner_rejected() {
        let reg = KernelRegistry::new();
        let id = reg
            .register(make_entry(InstanceId(1), "vector_add"))
            .unwrap();
        let err = reg.lookup(id, InstanceId(2)).unwrap_err();
        assert_eq!(err, AbiError::InvalidKernel);
    }

    #[test]
    fn lookup_unknown_rejected() {
        let reg = KernelRegistry::new();
        let err = reg.lookup(KernelId(42), InstanceId(1)).unwrap_err();
        assert_eq!(err, AbiError::InvalidKernel);
    }

    #[test]
    fn remove_drops_entry() {
        let reg = KernelRegistry::new();
        let id = reg
            .register(make_entry(InstanceId(1), "vector_add"))
            .unwrap();
        assert!(reg.remove(id).is_some());
        assert!(reg.lookup(id, InstanceId(1)).is_err());
    }

    #[test]
    fn ids_are_unique_and_increasing() {
        let reg = KernelRegistry::new();
        let a = reg.register(make_entry(InstanceId(1), "k1")).unwrap();
        let b = reg.register(make_entry(InstanceId(1), "k2")).unwrap();
        let c = reg.register(make_entry(InstanceId(1), "k3")).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(a.0 + 1, b.0);
        assert_eq!(b.0 + 1, c.0);
    }

    #[test]
    fn len_tracks_entries() {
        let reg = KernelRegistry::new();
        assert!(reg.is_empty());
        let id = reg.register(make_entry(InstanceId(1), "k")).unwrap();
        assert_eq!(reg.len(), 1);
        reg.remove(id);
        assert!(reg.is_empty());
    }

    #[test]
    fn aggregate_byte_cap_enforced() {
        let reg = KernelRegistry::new();
        // Try to register enough entries to exceed MAX_TOTAL_PTX_BYTES.
        // We pick a sub-MAX_PTX_BYTES per-entry size so the per-call cap is
        // not the gating constraint here.
        let per = MAX_PTX_BYTES; // 8 MiB
        let cap_count = MAX_TOTAL_PTX_BYTES / per; // 64
        for i in 0..cap_count {
            reg.register(make_entry_sized(InstanceId(1), &format!("k{i}"), per))
                .expect("under aggregate cap");
        }
        // The 65th registration at the per-entry max trips the aggregate cap.
        let err = reg
            .register(make_entry_sized(InstanceId(1), "k_over", per))
            .unwrap_err();
        assert_eq!(err, AbiError::QuotaExceeded);
    }

    #[test]
    fn aggregate_bytes_counter_decreases_on_remove() {
        let reg = KernelRegistry::new();
        let id = reg
            .register(make_entry_sized(InstanceId(1), "k", 4096))
            .unwrap();
        assert_eq!(reg.total_ptx_bytes(), 4096);
        reg.remove(id);
        assert_eq!(reg.total_ptx_bytes(), 0);
    }

    #[test]
    fn lookup_returns_independent_handle() {
        // The handle is cheap to clone and outlives the entry's dashmap ref:
        // construct it, drop nothing explicit, then immediately remove the
        // entry. Handle fields stay valid (no UAF) because they were copied.
        let reg = KernelRegistry::new();
        let id = reg.register(make_entry(InstanceId(1), "v")).unwrap();
        let handle = reg.lookup(id, InstanceId(1)).unwrap();
        assert!(reg.remove(id).is_some());
        assert_eq!(handle.entry, "v");
        assert_eq!(handle.owner, InstanceId(1));
    }
}
