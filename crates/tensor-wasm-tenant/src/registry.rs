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
    /// An admin-cap-gated method was invoked with a
    /// [`RegistryAdminCapability`] that was minted by a *different*
    /// `TenantRegistry`. Only emitted when the `strict-cap-binding` feature
    /// is enabled; without it, caps from independent registries are
    /// interchangeable (the surface invariant has always been "you must
    /// hold *some* cap" rather than "your cap must match this exact
    /// registry"). Strict mode is the recommended posture for multi-tenant
    /// deployments where two independent `TenantRegistry` instances live
    /// in the same process — see the `## Cap binding` section in the crate
    /// README for the upgrade path.
    #[cfg(feature = "strict-cap-binding")]
    #[error("capability was minted by a different TenantRegistry; refusing cross-registry operation")]
    CapabilityFromForeignRegistry,
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
///
/// # Registry binding (`strict-cap-binding` feature)
///
/// Under the `strict-cap-binding` feature, every admin capability also
/// carries an `Arc<()>` token that points to its minting registry's
/// per-instance allocation. Comparison is by `Arc::ptr_eq`, so a cap
/// minted by registry A is rejected with
/// [`RegistryError::CapabilityFromForeignRegistry`] when presented
/// against registry B even though both caps statically have the same
/// type. Without this feature the cap is an opaque "you-hold-*some*-cap"
/// token; the foreign-cap test at
/// `tests/admin_cap_required.rs::independent_constructions_yield_independent_caps`
/// asserts the legacy behaviour.
#[derive(Debug)]
pub struct RegistryAdminCapability {
    _seal: (),
    /// Pointer-identity stamp of the registry that minted this capability.
    /// Only present under the `strict-cap-binding` feature; the field
    /// disappears entirely (zero memory cost, no API surface) when the
    /// feature is off, preserving 0.3 ABI for embedders that don't opt
    /// into the strict mode.
    #[cfg(feature = "strict-cap-binding")]
    pub(crate) registry_token: std::sync::Arc<()>,
}

impl RegistryAdminCapability {
    /// Mint a fresh capability. Crate-private so external crates cannot
    /// forge admin authority over a `TenantRegistry` they did not
    /// construct. The non-strict-binding signature is unchanged.
    #[cfg(not(feature = "strict-cap-binding"))]
    pub(crate) fn mint() -> Self {
        Self { _seal: () }
    }

    /// Mint a fresh capability bound to the minting registry's
    /// `registry_token` (an `Arc::clone` of the registry's per-instance
    /// allocation). Comparison at admin-method call time is by
    /// `Arc::ptr_eq`; two registries that happen to allocate
    /// `Arc::new(())` at the same address would still be distinct
    /// allocations and `ptr_eq` would return `false`.
    #[cfg(feature = "strict-cap-binding")]
    pub(crate) fn mint(registry_token: std::sync::Arc<()>) -> Self {
        Self {
            _seal: (),
            registry_token,
        }
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
    /// Per-instance identity token used to bind capabilities to this
    /// specific registry under the `strict-cap-binding` feature.
    ///
    /// Allocated fresh inside [`Self::new`] and cloned (cheap `Arc::clone`)
    /// into every cap minted by this registry. Cloning the registry
    /// itself shares the same token allocation, which is what we want:
    /// `Arc::clone(&reg)` is the documented way to hand the same registry
    /// to a subsystem, and caps minted against either handle must
    /// continue to work against the other. Two *independent*
    /// `TenantRegistry::new()` calls produce two distinct allocations,
    /// so `Arc::ptr_eq` on the tokens identifies registry provenance.
    #[cfg(feature = "strict-cap-binding")]
    registry_token: Arc<()>,
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
        #[cfg(feature = "strict-cap-binding")]
        let registry_token: Arc<()> = Arc::new(());
        let reg = Self {
            inner: Arc::new(DashMap::new()),
            tombstones: Arc::new(DashMap::new()),
            #[cfg(feature = "strict-cap-binding")]
            registry_token: Arc::clone(&registry_token),
        };
        #[cfg(feature = "strict-cap-binding")]
        let cap = RegistryAdminCapability::mint(registry_token);
        #[cfg(not(feature = "strict-cap-binding"))]
        let cap = RegistryAdminCapability::mint();
        (reg, cap)
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
        #[allow(unused_mut)] mut ctx: TenantContext,
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
        // Under `strict-cap-binding` stamp the context with this
        // registry's token *before* wrapping it in `Arc`, so the
        // `check_capability` path can compare token identity at every
        // `consume_bytes_with_capability` / `release_bytes_with_capability`
        // call.
        #[cfg(feature = "strict-cap-binding")]
        {
            ctx.registry_token = Some(Arc::clone(&self.registry_token));
        }
        let entry = self.inner.entry(id);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(_) => Err(RegistryError::AlreadyRegistered(id)),
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                let arc = Arc::new(ctx);
                slot.insert(Arc::clone(&arc));
                #[cfg(feature = "strict-cap-binding")]
                let cap = TenantCapability::mint(id, Arc::clone(&self.registry_token));
                #[cfg(not(feature = "strict-cap-binding"))]
                let cap = TenantCapability::mint(id);
                Ok((arc, cap))
            }
        }
    }

    /// Verify that `cap` was minted by *this* registry's [`Self::new`]
    /// call. No-op when `strict-cap-binding` is disabled (the 0.3
    /// behaviour: caps from independent registries are interchangeable);
    /// under the feature, returns
    /// [`RegistryError::CapabilityFromForeignRegistry`] on mismatch and
    /// every admin method calls this before doing any work.
    ///
    /// We compare with `Arc::ptr_eq` on the per-registry token allocation
    /// rather than hashing addresses: two registries that happen to
    /// recycle the same heap address sequentially would still be distinct
    /// allocations from `Arc::clone`'s point of view (each `Arc::new(())`
    /// is its own refcount block), so `ptr_eq` is the only correct
    /// comparison.
    #[cfg(feature = "strict-cap-binding")]
    fn check_admin_cap(&self, cap: &RegistryAdminCapability) -> Result<(), RegistryError> {
        if Arc::ptr_eq(&self.registry_token, &cap.registry_token) {
            Ok(())
        } else {
            Err(RegistryError::CapabilityFromForeignRegistry)
        }
    }

    /// Look up a tenant by id. Returns `None` if no tenant is registered.
    ///
    /// Gated behind [`RegistryAdminCapability`] because an unrestricted
    /// `get` lets any holder of an `Arc<TenantRegistry>` enumerate other
    /// tenants' contexts by id and mutate their quota counters.
    ///
    /// Under the `strict-cap-binding` feature an additional runtime check
    /// rejects caps minted by a different registry; mismatch is observed
    /// here as `None`. The strict-mode test
    /// (`tests/cap_binding_strict.rs`) calls the typed [`Self::get_strict`]
    /// variant for explicit error propagation; this method preserves the
    /// `Option`-returning signature for the 0.3 line.
    pub fn get(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        #[cfg(feature = "strict-cap-binding")]
        {
            self.check_admin_cap(cap).ok()?;
        }
        let _ = cap;
        self.inner.get(&tenant_id).map(|r| Arc::clone(r.value()))
    }

    /// Remove a tenant. Returns the removed context, or `None` if absent.
    ///
    /// Gated behind [`RegistryAdminCapability`]: without this, any holder
    /// of an `Arc<TenantRegistry>` could evict arbitrary tenants from the
    /// registry, breaking their kernel pipelines. Under the
    /// `strict-cap-binding` feature, a foreign cap is observed as `None`
    /// here (no eviction happens). Use [`Self::unregister_strict`] when
    /// the explicit error propagation is required.
    pub fn unregister(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        #[cfg(feature = "strict-cap-binding")]
        {
            self.check_admin_cap(cap).ok()?;
        }
        let _ = cap;
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
    /// Under the `strict-cap-binding` feature, a foreign cap silently
    /// returns `0` (no pruning happens).
    pub fn collect_tombstones(&self, cap: &RegistryAdminCapability) -> usize {
        #[cfg(feature = "strict-cap-binding")]
        {
            if self.check_admin_cap(cap).is_err() {
                return 0;
            }
        }
        let _ = cap;
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
    /// side-channel. Under the `strict-cap-binding` feature, a foreign
    /// cap is observed as `0` here (no enumeration). Use
    /// [`Self::len_strict`] for explicit error propagation.
    pub fn len(&self, cap: &RegistryAdminCapability) -> usize {
        #[cfg(feature = "strict-cap-binding")]
        {
            if self.check_admin_cap(cap).is_err() {
                return 0;
            }
        }
        let _ = cap;
        self.inner.len()
    }

    /// Strict-mode variant of [`Self::get`] that returns a typed error
    /// on foreign-cap mismatch instead of collapsing it into `None`.
    /// Only available under the `strict-cap-binding` feature; the
    /// non-strict line has no foreign-cap concept so the variant would be
    /// vacuous.
    #[cfg(feature = "strict-cap-binding")]
    pub fn get_strict(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Result<Option<Arc<TenantContext>>, RegistryError> {
        self.check_admin_cap(cap)?;
        Ok(self.inner.get(&tenant_id).map(|r| Arc::clone(r.value())))
    }

    /// Strict-mode variant of [`Self::unregister`].
    #[cfg(feature = "strict-cap-binding")]
    pub fn unregister_strict(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Result<Option<Arc<TenantContext>>, RegistryError> {
        self.check_admin_cap(cap)?;
        let removed = self.inner.remove(&tenant_id).map(|(_, v)| v);
        if let Some(ref arc) = removed {
            self.tombstones.insert(tenant_id, Arc::downgrade(arc));
        }
        Ok(removed)
    }

    /// Strict-mode variant of [`Self::len`].
    #[cfg(feature = "strict-cap-binding")]
    pub fn len_strict(&self, cap: &RegistryAdminCapability) -> Result<usize, RegistryError> {
        self.check_admin_cap(cap)?;
        Ok(self.inner.len())
    }

    /// Strict-mode variant of [`Self::tenants`].
    #[cfg(feature = "strict-cap-binding")]
    pub fn tenants_strict(
        &self,
        cap: &RegistryAdminCapability,
    ) -> Result<Vec<Arc<TenantContext>>, RegistryError> {
        self.check_admin_cap(cap)?;
        Ok(self.inner.iter().map(|r| Arc::clone(r.value())).collect())
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
    /// hands, defeats the whole point of multi-tenant isolation. Under
    /// the `strict-cap-binding` feature, a foreign cap returns an empty
    /// `Vec` (no enumeration). Use [`Self::tenants_strict`] for explicit
    /// error propagation.
    pub fn tenants(&self, cap: &RegistryAdminCapability) -> Vec<Arc<TenantContext>> {
        #[cfg(feature = "strict-cap-binding")]
        {
            if self.check_admin_cap(cap).is_err() {
                return Vec::new();
            }
        }
        let _ = cap;
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
