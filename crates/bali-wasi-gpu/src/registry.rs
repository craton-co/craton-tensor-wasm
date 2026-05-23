//! [`KernelRegistry`] — instance-scoped store of compiled PTX kernels.
//!
//! Every Wasm instance gets its own registry. Kernel IDs are scoped to the
//! owning [`InstanceId`]; the host functions
//! refuse to launch a kernel using an ID that belongs to a different
//! instance (`AbiError::InvalidKernel`).

use std::sync::atomic::{AtomicU64, Ordering};

use bali_core::types::{InstanceId, KernelId};
use dashmap::DashMap;

use crate::abi::{AbiError, MAX_KERNELS_PER_INSTANCE};

/// Metadata about a single compiled kernel. The actual compiled module is
/// held opaquely behind a feature gate — on non-CUDA builds the field is
/// unit type so the registry can still be exercised by tests.
#[derive(Debug)]
pub struct KernelEntry {
    /// Owning instance (used to authorise `launch` calls).
    pub owner: InstanceId,
    /// Entry-point symbol name inside the PTX module.
    pub entry: String,
    /// Size of the PTX source that produced this kernel (bytes).
    pub ptx_bytes_len: usize,
    /// CUDA-side handle; only meaningful when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    pub module: Option<cust::module::Module>,
}

/// Instance-scoped kernel registry.
pub struct KernelRegistry {
    next_id: AtomicU64,
    entries: DashMap<KernelId, KernelEntry>,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            entries: DashMap::new(),
        }
    }

    /// Register a new kernel and return its assigned [`KernelId`].
    pub fn register(&self, entry: KernelEntry) -> Result<KernelId, AbiError> {
        if self.entries.len() >= MAX_KERNELS_PER_INSTANCE {
            return Err(AbiError::QuotaExceeded);
        }
        let id = KernelId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.entries.insert(id, entry);
        Ok(id)
    }

    /// Look up a kernel by id. Returns `Err(AbiError::InvalidKernel)` if the
    /// id is unknown or belongs to a different instance.
    pub fn lookup(
        &self,
        id: KernelId,
        owner: InstanceId,
    ) -> Result<dashmap::mapref::one::Ref<'_, KernelId, KernelEntry>, AbiError> {
        let r = self.entries.get(&id).ok_or(AbiError::InvalidKernel)?;
        if r.owner != owner {
            return Err(AbiError::InvalidKernel);
        }
        Ok(r)
    }

    /// Remove a kernel from the registry (caller releases its handle).
    pub fn remove(&self, id: KernelId) -> Option<KernelEntry> {
        self.entries.remove(&id).map(|(_, v)| v)
    }

    /// Number of currently-registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if there are no registered kernels.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
}
