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
//!
//! # Admin capability
//!
//! Registry-wide operations that can enumerate or evict arbitrary tenants
//! ([`TenantRegistry::get`], [`TenantRegistry::unregister`],
//! [`TenantRegistry::tenants`], [`TenantRegistry::len`]) are gated behind
//! [`RegistryAdminCapability`]. Exactly one capability is minted by
//! [`TenantRegistry::new`] and returned alongside the registry; downstream
//! crates that only hold an `Arc<TenantRegistry>` cannot enumerate or evict
//! tenants because the capability's sole constructor is crate-private.
//! Per-tenant operations (`register`) remain open: a caller that already
//! knows a tenant's id and supplies a fresh [`TenantContext`] is not granted
//! any additional authority over tenants it did not create.

use std::sync::{Arc, Weak};

use tensor_wasm_core::types::TenantId;
use dashmap::DashMap;
use thiserror::Error;

use crate::context::{TenantCapability, TenantContext};

/// Errors specific to [`TenantRegistry`] bookkeeping.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegistryError {
    /// `register` was called with a tenant id already present in the registry.
    #[error("tenant {0} already registered")]
    AlreadyRegistered(TenantId),
    /// `register` was called with an id that was previously unregistered,
    /// but at least one `Arc<TenantContext>` from the prior registration is
    /// still alive (an "orphan"). Re-registering with the same id while the
    /// orphan exists would let in-flight `consume_bytes_with_capability`
    /// calls commit to the orphan's counter while the new registration's
    /// counter remained zero — effectively a per-tenant quota reset
    /// (tenant 1.6 #9). Wait for the orphan to drop and call
    /// [`TenantRegistry::collect_tombstones`] before retrying.
    #[error("tenant {0} cannot be re-registered while an orphan Arc<TenantContext> is still alive")]
    OrphanStillAlive(TenantId),
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

/// Unforgeable token that authorises registry-wide admin operations.
///
/// Holding a `&RegistryAdminCapability` is the only way to invoke
/// [`TenantRegistry::get`], [`TenantRegistry::unregister`],
/// [`TenantRegistry::tenants`], or [`TenantRegistry::len`]. The struct's
/// sole field is private and its only constructor ([`Self::mint`]) is
/// crate-private, so external crates cannot synthesise a capability — they
/// can only borrow the one minted at [`TenantRegistry::new`] time.
///
/// Capabilities are not cloneable on purpose: an operator that delegates
/// admin authority to a sub-system passes a `&RegistryAdminCapability`
/// reference rather than handing out independent copies.
#[derive(Debug)]
pub struct RegistryAdminCapability {
    _seal: (),
}

impl RegistryAdminCapability {
    /// Mint a fresh capability. Crate-private so external crates cannot
    /// forge admin authority over a `TenantRegistry` they did not
    /// construct.
    pub(crate) fn mint() -> Self {
        Self { _seal: () }
    }
}

/// Concurrent registry of live tenants.
///
/// Cloning is cheap: the inner `DashMap` is wrapped in `Arc` so callers can
/// share the registry across the API layer, the executor, and the JIT cache
/// without ferrying a `&'static` reference.
///
/// Construction returns a tuple `(TenantRegistry, RegistryAdminCapability)`;
/// see [`Self::new`]. `Default` is intentionally not derived: a registry
/// produced by `Default::default()` would have no associated
/// [`RegistryAdminCapability`], contradicting the documented contract
/// that every registry is constructed alongside exactly one cap.
#[derive(Debug, Clone)]
pub struct TenantRegistry {
    inner: Arc<DashMap<TenantId, Arc<TenantContext>>>,
    /// Weak refs to previously-unregistered contexts. On re-`register` of
    /// an id, the tombstone (if any) is consulted: if its strong count is
    /// still nonzero, an orphan is alive and re-registration is refused
    /// (tenant 1.6 #9). Dead tombstones are pruned by
    /// [`Self::collect_tombstones`] or implicitly on a successful
    /// re-register.
    tombstones: Arc<DashMap<TenantId, Weak<TenantContext>>>,
}

impl TenantRegistry {
    /// Construct an empty registry and mint its admin capability.
    ///
    /// The capability is the only key that opens [`Self::get`],
    /// [`Self::unregister`], [`Self::tenants`], and [`Self::len`]; the
    /// operator that owns the returned cap is the only one authorised to
    /// enumerate or evict tenants. Cloning the registry shares the inner
    /// `DashMap`, but does NOT clone the cap — admin authority stays with
    /// whoever the original constructor handed it to.
    pub fn new() -> (Self, RegistryAdminCapability) {
        let reg = Self {
            inner: Arc::new(DashMap::new()),
            tombstones: Arc::new(DashMap::new()),
        };
        (reg, RegistryAdminCapability::mint())
    }

    /// Insert `ctx` into the registry.
    ///
    /// Returns the `Arc<TenantContext>` now stored, or
    /// [`RegistryError::AlreadyRegistered`] if a tenant with the same id is
    /// already present. The caller may clone the returned `Arc` freely.
    ///
    /// This signature is preserved for backwards compatibility. Prefer
    /// [`Self::register_with_capability`], which returns the
    /// [`TenantCapability`] required by the quota-mutation methods
    /// (`*_with_capability`) — without it, only the unchecked
    /// `#[deprecated]` variants of `consume_bytes`/`release_bytes` work.
    pub fn register(&self, ctx: TenantContext) -> Result<Arc<TenantContext>, RegistryError> {
        self.register_with_capability(ctx).map(|(arc, _cap)| arc)
    }

    /// Insert `ctx` into the registry and return the `Arc<TenantContext>`
    /// alongside a [`TenantCapability`] bound to the same tenant.
    ///
    /// The capability is the *only* way to call the checked
    /// `consume_bytes_with_capability` / `release_bytes_with_capability`
    /// quota-mutation methods on the returned context. Because the
    /// `TenantCapability` type cannot be constructed outside this crate,
    /// holding an `Arc<TenantContext>` for tenant A grants no power to
    /// mutate tenant B's accounting — even if A guesses or fabricates B's
    /// numeric `TenantId`.
    ///
    /// On [`RegistryError::AlreadyRegistered`] the in-flight `ctx` is
    /// dropped and no capability is minted; the previously registered
    /// tenant's accounting is untouched.
    pub fn register_with_capability(
        &self,
        ctx: TenantContext,
    ) -> Result<(Arc<TenantContext>, TenantCapability), RegistryError> {
        let id = ctx.id();
        // tenant 1.6 #9: refuse re-registration while an orphan Arc from a
        // prior registration is still alive. Without this, an in-flight
        // `consume_bytes_with_capability` on the orphan could commit to a
        // dead counter while the new registration's counter stays at 0 —
        // a per-tenant quota reset.
        if let Some(tomb) = self.tombstones.get(&id) {
            if tomb.value().strong_count() > 0 {
                return Err(RegistryError::OrphanStillAlive(id));
            }
        }
        // Orphan (if any) is gone; clear the tombstone so the slot is free.
        self.tombstones.remove(&id);
        let entry = self.inner.entry(id);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(_) => Err(RegistryError::AlreadyRegistered(id)),
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                let arc = Arc::new(ctx);
                slot.insert(Arc::clone(&arc));
                let cap = TenantCapability::mint(id);
                Ok((arc, cap))
            }
        }
    }

    /// Look up a tenant by id. Returns `None` if no tenant is registered.
    ///
    /// Gated behind [`RegistryAdminCapability`] because an unrestricted
    /// `get` lets any holder of an `Arc<TenantRegistry>` enumerate other
    /// tenants' contexts by id and mutate their quota counters.
    pub fn get(
        &self,
        tenant_id: TenantId,
        _cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        self.inner.get(&tenant_id).map(|r| Arc::clone(r.value()))
    }

    /// Remove a tenant. Returns the removed context, or `None` if absent.
    ///
    /// Gated behind [`RegistryAdminCapability`]: without this, any holder
    /// of an `Arc<TenantRegistry>` could evict arbitrary tenants from the
    /// registry, breaking their kernel pipelines.
    pub fn unregister(
        &self,
        tenant_id: TenantId,
        _cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        let removed = self.inner.remove(&tenant_id).map(|(_, v)| v);
        if let Some(ref arc) = removed {
            // tenant 1.6 #9: track the orphan so a future re-register can
            // refuse until the last strong ref is dropped.
            self.tombstones.insert(tenant_id, Arc::downgrade(arc));
        }
        removed
    }

    /// Prune dead orphan tombstones. Returns the number pruned.
    ///
    /// Gated behind [`RegistryAdminCapability`] — same threat model as
    /// [`Self::tenants`]: orphan presence is a global property of the
    /// registry that should not leak across tenant boundaries.
    ///
    /// Operators can call this periodically (e.g. from a background task)
    /// to keep the tombstone map from growing with every churned tenant.
    /// Callers do not normally need to invoke it manually — a successful
    /// re-`register` of a now-clean id implicitly clears its tombstone.
    pub fn collect_tombstones(&self, _cap: &RegistryAdminCapability) -> usize {
        let mut pruned = 0;
        self.tombstones.retain(|_id, weak| {
            if weak.strong_count() == 0 {
                pruned += 1;
                false
            } else {
                true
            }
        });
        pruned
    }

    /// Number of tenants currently registered.
    ///
    /// Gated behind [`RegistryAdminCapability`] because the count is a
    /// global property of the registry that should not leak across the
    /// tenant boundary — a tenant counting its peers is itself a
    /// side-channel.
    pub fn len(&self, _cap: &RegistryAdminCapability) -> usize {
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
    ///
    /// Gated behind [`RegistryAdminCapability`] because the snapshot
    /// enumerates every registered tenant — a primitive that, in the wrong
    /// hands, defeats the whole point of multi-tenant isolation.
    pub fn tenants(&self, _cap: &RegistryAdminCapability) -> Vec<Arc<TenantContext>> {
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
        let (reg, cap) = TenantRegistry::new();
        assert!(reg.is_empty());
        let arc = reg.register(ctx(1)).unwrap();
        assert_eq!(reg.len(&cap), 1);
        assert_eq!(arc.id(), TenantId(1));
        let found = reg.get(TenantId(1), &cap).unwrap();
        assert_eq!(found.id(), TenantId(1));
        let removed = reg.unregister(TenantId(1), &cap).unwrap();
        assert_eq!(removed.id(), TenantId(1));
        assert!(reg.is_empty());
        assert!(reg.get(TenantId(1), &cap).is_none());
    }

    #[test]
    fn double_register_is_rejected() {
        let (reg, cap) = TenantRegistry::new();
        reg.register(ctx(2)).unwrap();
        let err = reg.register(ctx(2)).unwrap_err();
        assert_eq!(err, RegistryError::AlreadyRegistered(TenantId(2)));
        // First registration is still intact.
        assert_eq!(reg.len(&cap), 1);
    }

    #[test]
    fn unregister_missing_returns_none() {
        let (reg, cap) = TenantRegistry::new();
        assert!(reg.unregister(TenantId(404), &cap).is_none());
    }

    #[test]
    fn tenants_snapshot_lists_all() {
        let (reg, cap) = TenantRegistry::new();
        for i in 0..5u64 {
            reg.register(ctx(i)).unwrap();
        }
        let mut ids: Vec<u64> = reg.tenants(&cap).iter().map(|c| c.id().get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn registry_clone_shares_state() {
        let (a, cap) = TenantRegistry::new();
        let b = a.clone();
        a.register(ctx(10)).unwrap();
        assert_eq!(b.len(&cap), 1);
        assert!(b.get(TenantId(10), &cap).is_some());
    }

    #[test]
    fn capability_from_one_tenant_cannot_mutate_another() {
        // Threat model: a workload holding `Arc<TenantContext>` for tenant A
        // (perhaps by guessing B's numeric id and calling `registry.get`)
        // must not be able to drive B's quota counter using A's capability.
        let (reg, _admin_cap) = TenantRegistry::new();
        let (a_ctx, a_cap) = reg.register_with_capability(ctx(1001)).unwrap();
        let (b_ctx, b_cap) = reg.register_with_capability(ctx(1002)).unwrap();

        // Baseline: each context starts empty.
        assert_eq!(a_ctx.bytes_in_use(), 0);
        assert_eq!(b_ctx.bytes_in_use(), 0);

        // Legitimate path: A's cap on A's ctx succeeds.
        a_ctx.consume_bytes_with_capability(&a_cap, 128).unwrap();
        assert_eq!(a_ctx.bytes_in_use(), 128);

        // Cross-tenant attack: try to drive A's counter with B's cap.
        let err = a_ctx
            .consume_bytes_with_capability(&b_cap, 256)
            .expect_err("cross-tenant consume must be rejected");
        match err {
            tensor_wasm_core::error::TensorWasmError::TenantIsolationViolation {
                tenant_id,
                ..
            } => {
                assert_eq!(tenant_id, TenantId(1002));
            }
            other => panic!("expected TenantIsolationViolation, got {other:?}"),
        }
        // Counter unchanged by the rejected attempt.
        assert_eq!(a_ctx.bytes_in_use(), 128);

        // Same for release: B's cap cannot tamper with A's counter.
        a_ctx.release_bytes_with_capability(&b_cap, 128).expect_err(
            "cross-tenant release must be rejected",
        );
        assert_eq!(a_ctx.bytes_in_use(), 128);

        // And B's own counter is untouched throughout.
        assert_eq!(b_ctx.bytes_in_use(), 0);
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
