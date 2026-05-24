// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `TenantRegistry`: `TenantId` → `TenantContext` mapping and lifecycle.
//!
//! The registry is a thin wrapper over [`dashmap::DashMap`] keyed by
//! [`TenantId`]. Every entry is wrapped in [`Arc`] so call sites (executors,
//! WASI host functions, the API layer) can hold onto a context across `await`
//! points without holding a `DashMap` shard lock. Double-registration of the
//! same tenant returns [`RegistryError::AlreadyRegistered`] rather than
//! silently overwriting — accidentally clobbering a live tenant's context
//! would be a stream/quota leak.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use dashmap::DashMap;
use thiserror::Error;

use crate::context::TenantContext;

/// Errors specific to [`TenantRegistry`] bookkeeping.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegistryError {
    /// `register` was called with a tenant id already present in the registry.
    #[error("tenant {0} already registered")]
    AlreadyRegistered(TenantId),
}

/// Decision returned by [`TenantRegistry::mps_or_fallback`].
///
/// On Linux hosts where the MPS control daemon's pipe directory exists, the
/// caller should use MPS-backed `ContextIsolated` tenants. Everywhere else
/// (no daemon, non-Linux, CI without `nvidia-cuda-mps-control`) the registry
/// reports [`MpsDecision::Fallback`] and the caller falls back to
/// `cuCtxCreate` per tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsDecision {
    /// MPS daemon detected; use shared-context-with-client mode.
    Mps,
    /// No MPS daemon; create per-tenant contexts directly.
    Fallback,
}

/// Default location of the MPS control pipe directory on Linux.
///
/// Used as a fallback when the `CUDA_MPS_PIPE_DIRECTORY` environment
/// variable is **not** set. The full lookup precedence applied by
/// [`TenantRegistry::mps_or_fallback`] is:
///
/// 1. `$CUDA_MPS_PIPE_DIRECTORY/control` — honours the operator-facing
///    NVIDIA env var that the MPS control daemon itself reads. This is
///    the path the documented `nvidia-cuda-mps-control -d` setup uses
///    when the variable is exported into the daemon's environment.
/// 2. `MPS_CONTROL_PATH/control` (i.e. `/tmp/nvidia-mps/control`) — the
///    historical default when `CUDA_MPS_PIPE_DIRECTORY` is unset.
///
/// See `docs/MPS-SETUP.md` for how to start the daemon. On Windows
/// neither path exists, so this always returns `false`, which is what
/// we want: MPS is not available on Windows, so the registry always
/// falls back to per-context isolation there.
pub const MPS_CONTROL_PATH: &str = "/tmp/nvidia-mps";

/// Environment variable consulted before [`MPS_CONTROL_PATH`].
///
/// If set, [`TenantRegistry::mps_or_fallback`] looks for a `control`
/// file inside the directory it names; only if that file is absent does
/// it fall back to the hard-coded default.
pub const MPS_PIPE_DIRECTORY_ENV: &str = "CUDA_MPS_PIPE_DIRECTORY";

/// Concurrent registry of live tenants.
///
/// Cloning is cheap: the inner `DashMap` is wrapped in `Arc` so callers can
/// share the registry across the API layer, the executor, and the JIT cache
/// without ferrying a `&'static` reference.
#[derive(Debug, Clone, Default)]
pub struct TenantRegistry {
    inner: Arc<DashMap<TenantId, Arc<TenantContext>>>,
}

impl TenantRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `ctx` into the registry.
    ///
    /// Returns the `Arc<TenantContext>` now stored, or
    /// [`RegistryError::AlreadyRegistered`] if a tenant with the same id is
    /// already present. The caller may clone the returned `Arc` freely.
    pub fn register(&self, ctx: TenantContext) -> Result<Arc<TenantContext>, RegistryError> {
        let id = ctx.id();
        let entry = self.inner.entry(id);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(_) => Err(RegistryError::AlreadyRegistered(id)),
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                let arc = Arc::new(ctx);
                slot.insert(Arc::clone(&arc));
                Ok(arc)
            }
        }
    }

    /// Look up a tenant by id. Returns `None` if no tenant is registered.
    pub fn get(&self, tenant_id: TenantId) -> Option<Arc<TenantContext>> {
        self.inner.get(&tenant_id).map(|r| Arc::clone(r.value()))
    }

    /// Remove a tenant. Returns the removed context, or `None` if absent.
    pub fn unregister(&self, tenant_id: TenantId) -> Option<Arc<TenantContext>> {
        self.inner.remove(&tenant_id).map(|(_, v)| v)
    }

    /// Number of tenants currently registered.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` when no tenants are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Snapshot the registered tenants into a `Vec`.
    ///
    /// We return a `Vec` rather than a live iterator because `DashMap`'s
    /// iterators hold shard locks; handing one out across an `await` boundary
    /// is a deadlock waiting to happen. The snapshot is cheap — each entry is
    /// an `Arc` clone.
    pub fn tenants(&self) -> Vec<Arc<TenantContext>> {
        self.inner.iter().map(|r| Arc::clone(r.value())).collect()
    }

    /// Decide whether to use MPS or per-context isolation by probing the
    /// MPS control pipe.
    ///
    /// Precedence (matches the behaviour of `nvidia-cuda-mps-control`):
    ///
    /// 1. If [`MPS_PIPE_DIRECTORY_ENV`] (`CUDA_MPS_PIPE_DIRECTORY`) is set
    ///    in the process environment, the check looks for a `control`
    ///    file inside that directory.
    /// 2. Otherwise it falls back to `MPS_CONTROL_PATH/control`
    ///    (i.e. `/tmp/nvidia-mps/control`).
    ///
    /// The check is intentionally a filesystem probe rather than a CUDA API
    /// call so it works on hosts without `cust`. On Windows neither path
    /// exists, so this always returns [`MpsDecision::Fallback`].
    pub fn mps_or_fallback() -> MpsDecision {
        let dir = std::env::var_os(MPS_PIPE_DIRECTORY_ENV)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(MPS_CONTROL_PATH));
        let pipe = dir.join("control");
        if pipe.exists() {
            MpsDecision::Mps
        } else {
            MpsDecision::Fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::IsolationKind;

    fn ctx(id: u64) -> TenantContext {
        TenantContext::builder(TenantId(id))
            .with_isolation(IsolationKind::StreamIsolated)
            .with_memory_quota_bytes(4096)
            .build()
    }

    #[test]
    fn register_lookup_unregister() {
        let reg = TenantRegistry::new();
        assert!(reg.is_empty());
        let arc = reg.register(ctx(1)).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(arc.id(), TenantId(1));
        let found = reg.get(TenantId(1)).unwrap();
        assert_eq!(found.id(), TenantId(1));
        let removed = reg.unregister(TenantId(1)).unwrap();
        assert_eq!(removed.id(), TenantId(1));
        assert!(reg.is_empty());
        assert!(reg.get(TenantId(1)).is_none());
    }

    #[test]
    fn double_register_is_rejected() {
        let reg = TenantRegistry::new();
        reg.register(ctx(2)).unwrap();
        let err = reg.register(ctx(2)).unwrap_err();
        assert_eq!(err, RegistryError::AlreadyRegistered(TenantId(2)));
        // First registration is still intact.
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn unregister_missing_returns_none() {
        let reg = TenantRegistry::new();
        assert!(reg.unregister(TenantId(404)).is_none());
    }

    #[test]
    fn tenants_snapshot_lists_all() {
        let reg = TenantRegistry::new();
        for i in 0..5u64 {
            reg.register(ctx(i)).unwrap();
        }
        let mut ids: Vec<u64> = reg.tenants().iter().map(|c| c.id().get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn registry_clone_shares_state() {
        let a = TenantRegistry::new();
        let b = a.clone();
        a.register(ctx(10)).unwrap();
        assert_eq!(b.len(), 1);
        assert!(b.get(TenantId(10)).is_some());
    }

    #[test]
    fn mps_decision_uses_filesystem_probe() {
        // On Windows this path never exists, so we always get Fallback.
        // On Linux without MPS configured the same holds. We assert the value
        // matches what an independent probe with the same precedence rules
        // says rather than hard-coding a platform expectation. The dedicated
        // env-var test lives in `tests/mps_pipe_env_var.rs`.
        let dir = std::env::var_os(MPS_PIPE_DIRECTORY_ENV)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(MPS_CONTROL_PATH));
        let expected = if dir.join("control").exists() {
            MpsDecision::Mps
        } else {
            MpsDecision::Fallback
        };
        assert_eq!(TenantRegistry::mps_or_fallback(), expected);
    }
}
