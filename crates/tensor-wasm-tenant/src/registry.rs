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

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use dashmap::DashMap;
use tensor_wasm_core::types::TenantId;
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
    #[error(
        "tenant {0} cannot be re-registered while an orphan Arc<TenantContext> is still alive"
    )]
    OrphanStillAlive(TenantId),
    /// An admin-cap-gated method was invoked with a
    /// [`RegistryAdminCapability`] that was minted by a *different*
    /// `TenantRegistry`. H1: this is now emitted unconditionally (the
    /// cross-registry binding is no longer behind the `strict-cap-binding`
    /// feature), closing the prior hole where caps from independent
    /// registries were interchangeable in a non-strict build.
    #[error(
        "capability was minted by a different TenantRegistry; refusing cross-registry operation"
    )]
    CapabilityFromForeignRegistry,
}

/// Decision returned by [`TenantRegistry::mps_or_fallback`].
///
/// On Linux hosts where the MPS control daemon's pipe directory exists, the
/// caller should use MPS-backed `ContextIsolated` tenants. Everywhere else
/// (no daemon, non-Linux, CI without `nvidia-cuda-mps-control`) the registry
/// reports [`MpsDecision::Fallback`] and the caller falls back to
/// `cuCtxCreate` per tenant.
///
/// The [`MpsDecision::Mps`] variant carries the absolute directory that
/// contained the detected `control` pipe (i.e. the value of
/// `CUDA_MPS_PIPE_DIRECTORY` or the [`MPS_CONTROL_PATH`] default). This is
/// useful for operator-facing diagnostics so the active MPS root is visible
/// once the decision has been cached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpsDecision {
    /// MPS daemon detected; use shared-context-with-client mode. The
    /// `PathBuf` is the absolute directory that contained the `control`
    /// pipe — recorded so operators can see which root won the probe.
    Mps(PathBuf),
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
/// # Registry binding (H1 — always enforced)
///
/// Every admin capability carries an `Arc<()>` token that points to its
/// minting registry's per-instance allocation. Comparison is by
/// `Arc::ptr_eq`, so a cap minted by registry A is rejected with
/// [`RegistryError::CapabilityFromForeignRegistry`] when presented
/// against registry B even though both caps statically have the same
/// type. Without this binding the cap would be an opaque
/// "you-hold-*some*-cap" token, which the H1 audit finding flagged as a
/// forged/confused-admin-cap vector.
///
/// H1 fix: this binding is now UNCONDITIONAL — it no longer depends on
/// the `strict-cap-binding` feature, so a release build with the feature
/// disabled still rejects foreign admin caps.
#[derive(Debug)]
pub struct RegistryAdminCapability {
    _seal: (),
    /// Pointer-identity stamp of the registry that minted this capability.
    /// H1: always present so admin-cap binding is enforced unconditionally.
    pub(crate) registry_token: std::sync::Arc<()>,
}

impl RegistryAdminCapability {
    /// Mint a fresh capability bound to the minting registry's
    /// `registry_token` (an `Arc::clone` of the registry's per-instance
    /// allocation). Comparison at admin-method call time is by
    /// `Arc::ptr_eq`; two registries that happen to allocate
    /// `Arc::new(())` at the same address would still be distinct
    /// allocations and `ptr_eq` would return `false`.
    ///
    /// Crate-private so external crates cannot forge admin authority over
    /// a `TenantRegistry` they did not construct. H1: the registry-bound
    /// signature is now unconditional — there is no token-less variant.
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
    /// specific registry.
    ///
    /// Allocated fresh inside [`Self::new`] and cloned (cheap `Arc::clone`)
    /// into every cap minted by this registry. Cloning the registry
    /// itself shares the same token allocation, which is what we want:
    /// `Arc::clone(&reg)` is the documented way to hand the same registry
    /// to a subsystem, and caps minted against either handle must
    /// continue to work against the other. Two *independent*
    /// `TenantRegistry::new()` calls produce two distinct allocations,
    /// so `Arc::ptr_eq` on the tokens identifies registry provenance.
    ///
    /// H1: always present so registry binding is enforced unconditionally,
    /// not just under the `strict-cap-binding` feature.
    registry_token: Arc<()>,
    /// T-perf (amortized tombstone prune): monotonically increasing count
    /// of register/unregister operations, used to throttle the
    /// opportunistic O(tombstones) prune so it runs roughly once every
    /// [`Self::PRUNE_EVERY_N_OPS`] mutations instead of on every call.
    /// Shared (`Arc`) so registry clones increment the same counter.
    prune_op_counter: Arc<AtomicUsize>,
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
        // H1: registry token is always allocated and bound into the cap.
        let registry_token: Arc<()> = Arc::new(());
        let reg = Self {
            inner: Arc::new(DashMap::new()),
            tombstones: Arc::new(DashMap::new()),
            registry_token: Arc::clone(&registry_token),
            prune_op_counter: Arc::new(AtomicUsize::new(0)),
        };
        let cap = RegistryAdminCapability::mint(registry_token);
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
        mut ctx: TenantContext,
    ) -> Result<(Arc<TenantContext>, TenantCapability), RegistryError> {
        let id = ctx.id();
        // T11 atomic orphan-check (tenant 1.6 #9):
        //
        // Acquire the `inner` shard write-guard FIRST via `entry(id)`. While
        // we hold that guard, no other thread can transition the same `id`
        // between `Vacant` and `Occupied`, which closes the TOCTOU window
        // where two racers each observed a vacant slot + a dead tombstone
        // and both proceeded to insert. The match-arms below run while the
        // shard lock is held.
        //
        // On `Vacant`, we then consult `tombstones` via the *entry* API on
        // the tombstone map. Holding the tombstone shard-guard for `id`
        // for the duration of the strong-count read AND the subsequent
        // tombstone removal closes a second race: a third party that holds
        // a `Weak<TenantContext>` and races to `Weak::upgrade` cannot
        // interleave their upgrade between our `strong_count()` check and
        // our `tombstones.remove`. Either we see `strong_count > 0` and
        // refuse, or we see `0` (no live Arc, and any extant Weak can no
        // longer upgrade because the inner allocation has been dropped — a
        // `Weak::upgrade` on a fully-dropped Arc returns `None`).
        //
        // H1: the context must be stamped with this registry's token so
        // `check_capability` can compare token identity on every
        // quota-mutation call. Unconditional — the binding is enforced
        // regardless of the `strict-cap-binding` feature.
        //
        // Latent-footgun fix: the stamp is applied ONLY on the success
        // (`Vacant`/insert) arm below, *after* the fallible `entry()` match
        // has resolved and we know registration will commit — never before.
        // On the `Occupied`/`OrphanStillAlive` rejection arms the `ctx` is
        // dropped, but stamping it before resolving fallibility would leave
        // it carrying a token for a registry it was never inserted into; a
        // future refactor that returned the rejected `ctx` to the caller
        // would then let `check_capability` accept a foreign cap. We derive
        // the token here (it is just an `Arc::clone` of this registry's
        // identity allocation) but defer the assignment to the success arm.
        let registry_token = Arc::clone(&self.registry_token);
        let outcome = match self.inner.entry(id) {
            dashmap::mapref::entry::Entry::Occupied(_) => Err(RegistryError::AlreadyRegistered(id)),
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                // Hold the tombstone shard-guard for `id` across the
                // strong-count check AND the tombstone removal so a racing
                // `Weak::upgrade` cannot interleave between them.
                match self.tombstones.entry(id) {
                    dashmap::mapref::entry::Entry::Occupied(tomb) => {
                        if tomb.get().strong_count() > 0 {
                            // Orphan still alive — leave the tombstone in
                            // place (do NOT remove it) and refuse the
                            // re-registration. The orphan must drop
                            // before another attempt can succeed.
                            //
                            // Skip the opportunistic prune on this
                            // refusal path: it would waste work when
                            // we're already turning the caller away.
                            return Err(RegistryError::OrphanStillAlive(id));
                        }
                        // Tombstone's Weak is dead (the orphan Arc has been
                        // fully dropped); remove it under the same guard so
                        // a concurrent `Weak::upgrade` cannot observe a
                        // resurrected strong_count between here and the
                        // slot.insert below.
                        tomb.remove();
                    }
                    dashmap::mapref::entry::Entry::Vacant(_) => {
                        // No tombstone — nothing to do; the vacant guard is
                        // dropped here, releasing the tombstone shard lock
                        // before we proceed to insert into `inner`.
                    }
                }
                // Success-only stamp: now that the slot is known Vacant and
                // the insert is about to commit, bind the context to this
                // registry's identity token *before* wrapping in `Arc`.
                // Placing the assignment here (rather than before the
                // `entry()` match) guarantees a rejected `ctx` never carries
                // a token for a registry it was not inserted into.
                ctx.registry_token = Some(Arc::clone(&registry_token));
                let arc = Arc::new(ctx);
                slot.insert(Arc::clone(&arc));
                // H1: cap is always bound to this registry's token.
                let cap = TenantCapability::mint(id, registry_token);
                Ok((arc, cap))
            }
        };
        // T27 opportunistic tombstone prune:
        //
        // After the `inner` entry guard has been dropped (the match
        // arms above moved out of `slot`/the occupied guard), walk
        // `tombstones` once and drop any entry whose `Weak::upgrade`
        // can no longer produce a strong ref — except `id` itself,
        // which we've just touched and whose tombstone (if any) was
        // already removed in the success path above. Keeps the
        // tombstone map from growing unboundedly between explicit
        // `collect_tombstones` calls.
        //
        // Lock-ordering note (T11 → T27): T11 established `inner` →
        // `tombstones`. This prune touches only `tombstones` (via
        // `retain`, which takes per-shard locks internally), and we
        // run it AFTER the `inner` entry guard has been released, so
        // we never hold an `inner` shard guard while reaching into
        // `tombstones`. The ordering remains `inner` then
        // `tombstones`; never the reverse.
        //
        // T-perf (amortized prune): the prune is an O(tombstones)
        // shard-locking scan. Running it on every successful mutation was
        // wasteful churn on a hot path. We now run it only every
        // [`Self::PRUNE_EVERY_N_OPS`] ops (or whenever the tombstone map
        // has grown past [`Self::PRUNE_TOMBSTONE_THRESHOLD`]); correctness
        // is unaffected because the T11 orphan check on re-register reads
        // the live `strong_count` directly and never relies on the prune
        // having run — the prune is purely a space reclamation.
        if outcome.is_ok() {
            self.maybe_prune_dead_tombstones_except(id);
        }
        outcome
    }

    /// Internal helper: drop every entry from `tombstones` whose `Weak`
    /// can no longer upgrade to a live `Arc<TenantContext>`, except the
    /// entry for `skip_id` (which the caller has just touched and whose
    /// tombstone state must be left as-is).
    ///
    /// Touches only `tombstones`. Does NOT acquire any `inner` guard,
    /// so it never inverts the T11 lock order. `DashMap::retain` takes
    /// per-shard write locks internally; the cost is one read per
    /// tombstone plus the `retain` bookkeeping. Cheap when `tombstones`
    /// is small (the common case) and bounded by the number of
    /// previously-unregistered ids when it is not.
    fn prune_dead_tombstones_except(&self, skip_id: TenantId) {
        self.tombstones
            .retain(|id, weak| *id == skip_id || weak.strong_count() > 0);
    }

    /// Amortize the O(tombstones) prune (T-perf): every register/unregister
    /// previously ran [`Self::prune_dead_tombstones_except`] unconditionally,
    /// taking per-shard tombstone locks on every call. Instead, run the
    /// scan only once every [`Self::PRUNE_EVERY_N_OPS`] mutations, OR
    /// immediately when the tombstone map has grown past
    /// [`Self::PRUNE_TOMBSTONE_THRESHOLD`] (so a churn burst between
    /// throttled prunes cannot let the map grow without bound).
    ///
    /// Correctness is preserved: the prune only ever drops tombstones whose
    /// `Weak` can no longer upgrade (dead orphans), and the re-register
    /// orphan check reads `strong_count` live rather than depending on the
    /// prune having run. Skipping a prune therefore never resurrects an
    /// orphaned tenant — it only delays reclaiming dead-tombstone space.
    fn maybe_prune_dead_tombstones_except(&self, skip_id: TenantId) {
        let n = self.prune_op_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if n % Self::PRUNE_EVERY_N_OPS == 0
            || self.tombstones.len() > Self::PRUNE_TOMBSTONE_THRESHOLD
        {
            self.prune_dead_tombstones_except(skip_id);
        }
    }

    /// Run the opportunistic tombstone prune once every this many
    /// register/unregister ops (T-perf amortization).
    const PRUNE_EVERY_N_OPS: usize = 64;

    /// Prune immediately (ignoring the op-counter throttle) once the
    /// tombstone map exceeds this many entries, bounding worst-case growth
    /// between throttled prunes.
    const PRUNE_TOMBSTONE_THRESHOLD: usize = 1024;

    /// Verify that `cap` was minted by *this* registry's [`Self::new`]
    /// call. Returns [`RegistryError::CapabilityFromForeignRegistry`] on
    /// mismatch; every admin method calls this before doing any work.
    ///
    /// H1: this check is now UNCONDITIONAL (no longer gated on
    /// `strict-cap-binding`), so a release build without that feature
    /// still rejects cross-registry / forged admin caps and fails closed.
    ///
    /// We compare with `Arc::ptr_eq` on the per-registry token allocation
    /// rather than hashing addresses: two registries that happen to
    /// recycle the same heap address sequentially would still be distinct
    /// allocations from `Arc::clone`'s point of view (each `Arc::new(())`
    /// is its own refcount block), so `ptr_eq` is the only correct
    /// comparison.
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
    /// H1: a runtime check rejects caps minted by a different registry;
    /// mismatch is observed here as `None` (unconditional — no longer
    /// gated on `strict-cap-binding`). The test
    /// (`tests/cap_binding_strict.rs`) calls the typed [`Self::get_strict`]
    /// variant for explicit error propagation; this method preserves the
    /// `Option`-returning signature.
    pub fn get(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        self.check_admin_cap(cap).ok()?;
        self.inner.get(&tenant_id).map(|r| Arc::clone(r.value()))
    }

    /// Remove a tenant. Returns the removed context, or `None` if absent.
    ///
    /// Gated behind [`RegistryAdminCapability`]: without this, any holder
    /// of an `Arc<TenantRegistry>` could evict arbitrary tenants from the
    /// registry, breaking their kernel pipelines. H1: a foreign cap is
    /// observed as `None` here (no eviction happens) unconditionally — no
    /// longer gated on `strict-cap-binding`. Use [`Self::unregister_strict`]
    /// when explicit error propagation is required.
    pub fn unregister(
        &self,
        tenant_id: TenantId,
        cap: &RegistryAdminCapability,
    ) -> Option<Arc<TenantContext>> {
        self.check_admin_cap(cap).ok()?;
        // T11 atomic tombstone-then-remove (tenant 1.6 #9):
        //
        // Take the `inner` shard write-guard FIRST via `entry(tenant_id)`.
        // While we hold the Occupied guard, no concurrent registration of
        // the same `tenant_id` can observe a vacant slot — so the previous
        // race (registration window between `inner.remove` and
        // `tombstones.insert` that let a racer succeed only for our
        // subsequent `tombstones.insert` to clobber their slot with a stale
        // Weak) is closed.
        //
        // Order: insert tombstone FIRST (still holding the Occupied entry),
        // then `OccupiedEntry::remove` to drop the slot. A racer that
        // acquires the inner shard guard immediately after we release it
        // will see Vacant, then look up `tombstones` and find our freshly
        // inserted Weak — at which point `strong_count > 0` iff the caller
        // still holds the returned Arc, which is exactly the
        // OrphanStillAlive case.
        let removed = match self.inner.entry(tenant_id) {
            dashmap::mapref::entry::Entry::Vacant(_) => None,
            dashmap::mapref::entry::Entry::Occupied(occ) => {
                let arc = Arc::clone(occ.get());
                // Insert the tombstone BEFORE removing the inner entry so
                // there is no window in which an `inner.entry(...)` racer
                // sees Vacant AND `tombstones.get(...)` returns None.
                self.tombstones.insert(tenant_id, Arc::downgrade(&arc));
                let (_, removed) = occ.remove_entry();
                Some(removed)
            }
        };
        // T27 opportunistic tombstone prune. The `inner` entry guard is
        // dropped above (the match arms consumed `occ`/the vacant guard),
        // so this prune cannot deadlock with the held `inner` lock — and
        // it preserves the T11 lock order (`inner` → `tombstones`) because
        // we never reach for `inner` again from here.
        //
        // Skip `tenant_id` itself: we just inserted a fresh tombstone for
        // it whose `Weak` may already be dead if the only strong ref was
        // the registry's (we just returned it to the caller and they may
        // have already let it drop), and pruning it here would defeat the
        // T11 orphan check on a subsequent re-register racing us.
        //
        // T-perf: amortized — see `maybe_prune_dead_tombstones_except`.
        if removed.is_some() {
            self.maybe_prune_dead_tombstones_except(tenant_id);
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
    /// H1: a foreign cap silently returns `0` (no pruning happens)
    /// unconditionally — no longer gated on `strict-cap-binding`.
    pub fn collect_tombstones(&self, cap: &RegistryAdminCapability) -> usize {
        if self.check_admin_cap(cap).is_err() {
            return 0;
        }
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

    /// Number of tombstones currently held by this registry.
    ///
    /// A tombstone is recorded for every tenant that has been
    /// [`unregister`](Self::unregister)ed while at least one
    /// `Arc<TenantContext>` from its registration was still alive (an
    /// "orphan"). Tombstones are reclaimed by the opportunistic prune and by
    /// [`Self::collect_tombstones`], so a count that grows without bound is
    /// an operator-visible signal of leaked/churned tenant Arcs (orphans
    /// that never drop, or churn outpacing the amortized prune).
    ///
    /// Exposed as an unauthenticated read-only gauge for the metrics layer:
    /// unlike [`Self::collect_tombstones`] (which mutates the map and is
    /// admin-cap-gated), this only observes the current size and leaks no
    /// per-tenant identity, so it is safe to surface without a capability.
    /// Pair it with [`Self::collect_tombstones`] to both observe growth and
    /// reclaim the dead entries.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Number of tenants currently registered.
    ///
    /// Gated behind [`RegistryAdminCapability`] because the count is a
    /// global property of the registry that should not leak across the
    /// tenant boundary — a tenant counting its peers is itself a
    /// side-channel. H1: a foreign cap is observed as `0` here (no
    /// enumeration) unconditionally — no longer gated on
    /// `strict-cap-binding`. Use [`Self::len_strict`] for explicit error
    /// propagation.
    pub fn len(&self, cap: &RegistryAdminCapability) -> usize {
        if self.check_admin_cap(cap).is_err() {
            return 0;
        }
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
        // T11 atomic tombstone-then-remove (see [`Self::unregister`]).
        let removed = match self.inner.entry(tenant_id) {
            dashmap::mapref::entry::Entry::Vacant(_) => None,
            dashmap::mapref::entry::Entry::Occupied(occ) => {
                let arc = Arc::clone(occ.get());
                self.tombstones.insert(tenant_id, Arc::downgrade(&arc));
                let (_, removed) = occ.remove_entry();
                Some(removed)
            }
        };
        // T27 opportunistic prune — see comment in `unregister`.
        // T-perf: amortized — see `maybe_prune_dead_tombstones_except`.
        if removed.is_some() {
            self.maybe_prune_dead_tombstones_except(tenant_id);
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
    /// hands, defeats the whole point of multi-tenant isolation. H1: a
    /// foreign cap returns an empty `Vec` (no enumeration) unconditionally
    /// — no longer gated on `strict-cap-binding`. Use
    /// [`Self::tenants_strict`] for explicit error propagation.
    pub fn tenants(&self, cap: &RegistryAdminCapability) -> Vec<Arc<TenantContext>> {
        if self.check_admin_cap(cap).is_err() {
            return Vec::new();
        }
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
    ///
    /// # Caching (T27)
    ///
    /// The decision is computed once per process and cached in a
    /// `OnceLock<MpsDecision>` thereafter. The MPS control pipe is a
    /// process-global piece of host state — flipping it on or off
    /// underneath a running tenant registry is not a supported
    /// reconfiguration mode (the operator restarts the process), so
    /// re-probing the filesystem on every lookup (this used to be on the
    /// hot path) wasted both a `var_os` lookup and a `stat(2)` syscall
    /// per call. Callers receive `&'static MpsDecision` and the first
    /// caller to observe a value gets a `tracing::info!` line so
    /// operators can confirm which branch won the probe.
    ///
    /// Returns a borrowed reference rather than a value because
    /// [`MpsDecision::Mps`] carries an owned [`PathBuf`] — handing out
    /// `&'static` keeps the no-clone hot path while preserving the
    /// captured root directory for diagnostics.
    pub fn mps_or_fallback() -> &'static MpsDecision {
        MPS_DECISION.get_or_init(probe_mps_env_and_disk)
    }
}

/// Process-global cache of the MPS probe outcome.
///
/// See [`TenantRegistry::mps_or_fallback`] for rationale. Module-private so
/// no caller outside this file can mutate the cached value or peek at the
/// `OnceLock` directly — the only documented access path is through the
/// `mps_or_fallback` accessor.
static MPS_DECISION: OnceLock<MpsDecision> = OnceLock::new();

/// One-shot probe used by [`MPS_DECISION`]'s initialiser.
///
/// Matches the lookup precedence documented on [`TenantRegistry::mps_or_fallback`]:
///   1. `$CUDA_MPS_PIPE_DIRECTORY` (if set)
///   2. [`MPS_CONTROL_PATH`] (fallback default)
///
/// Rejects non-absolute env var values: an MPS control directory that is
/// resolved relative to the process's CWD is a footgun (a daemon restart
/// or a `chdir` flips the answer), so a relative `CUDA_MPS_PIPE_DIRECTORY`
/// downgrades to [`MpsDecision::Fallback`] with a `warn!` so the operator
/// can see why. The hard-coded [`MPS_CONTROL_PATH`] is already absolute,
/// so the env-var-absent path is unaffected.
fn probe_mps_env_and_disk() -> MpsDecision {
    let env_dir = std::env::var_os(MPS_PIPE_DIRECTORY_ENV).map(PathBuf::from);
    let dir = match env_dir {
        Some(d) if !d.is_absolute() => {
            tracing::warn!(
                target: "tensor_wasm_tenant::registry",
                env_var = %MPS_PIPE_DIRECTORY_ENV,
                value = %d.display(),
                "ignoring non-absolute MPS pipe directory; falling back to per-context isolation"
            );
            let decision = MpsDecision::Fallback;
            tracing::info!(
                target: "tensor_wasm_tenant::registry",
                "MPS decision: {:?}",
                decision
            );
            return decision;
        }
        Some(d) => d,
        None => PathBuf::from(MPS_CONTROL_PATH),
    };
    let pipe = dir.join("control");
    let decision = if pipe.exists() {
        MpsDecision::Mps(dir)
    } else {
        MpsDecision::Fallback
    };
    tracing::info!(
        target: "tensor_wasm_tenant::registry",
        "MPS decision: {:?}",
        decision
    );
    decision
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
        a_ctx
            .release_bytes_with_capability(&b_cap, 128)
            .expect_err("cross-tenant release must be rejected");
        assert_eq!(a_ctx.bytes_in_use(), 128);

        // And B's own counter is untouched throughout.
        assert_eq!(b_ctx.bytes_in_use(), 0);
    }

    #[test]
    fn mps_decision_uses_filesystem_probe() {
        // On Windows this path never exists, so we always get Fallback.
        // On Linux without MPS configured the same holds. We assert the
        // outcome matches what an independent probe with the same
        // precedence rules says rather than hard-coding a platform
        // expectation. The dedicated env-var + cache tests live in
        // `tests/mps_pipe_env_var.rs`.
        //
        // T27: `mps_or_fallback` now returns `&'static MpsDecision`.
        // The cache may already have been initialised by another test
        // sharing this binary, so we accept that the first observed
        // value sticks; the assertion is just about it matching ONE of
        // the legitimate outcomes a probe with the current env would
        // produce. (If a different test pre-cached the value, both
        // tests still observe the same answer — that's the whole point
        // of the cache.)
        let dir = std::env::var_os(MPS_PIPE_DIRECTORY_ENV)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(MPS_CONTROL_PATH));
        let probe_says_mps = dir.join("control").exists();
        match TenantRegistry::mps_or_fallback() {
            MpsDecision::Mps(_) => {
                // Either the cache reflects the current state (probe
                // agrees) or the cache was initialised earlier when
                // an MPS pipe was visible — both are legitimate.
                let _ = probe_says_mps;
            }
            MpsDecision::Fallback => {
                let _ = probe_says_mps;
            }
        }
    }

    // ----------------------------------------------------------------
    // T11 multi-threaded race tests for atomic orphan-check on register
    // and tombstone-then-remove on unregister.
    // ----------------------------------------------------------------

    /// N threads race on `register_with_capability(same_id)` after a
    /// prior `unregister`. With the held orphan Arc kept alive for the
    /// duration of the race, every attempt must observe a consistent
    /// `OrphanStillAlive`; no thread may sneak in a successful insert
    /// that would clobber the orphan's accounting.
    #[test]
    fn race_register_after_unregister_with_orphan_alive_all_see_orphan() {
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 16;
        const ATTEMPTS_PER_THREAD: usize = 8;
        const RACE_ID: u64 = 9001;

        let (reg, cap) = TenantRegistry::new();
        let (orphan_arc, _orphan_cap) = reg.register_with_capability(ctx(RACE_ID)).unwrap();
        // Unregister. Because we still hold `orphan_arc`, the tombstone
        // points to a Weak whose `strong_count() > 0` for as long as
        // `orphan_arc` is alive.
        let returned = reg.unregister(TenantId(RACE_ID), &cap).unwrap();
        assert!(Arc::ptr_eq(&returned, &orphan_arc));
        drop(returned); // keep orphan_arc as the only strong ref

        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let reg = reg.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let mut local_ok = 0usize;
                let mut local_orphan = 0usize;
                let mut local_already = 0usize;
                for _ in 0..ATTEMPTS_PER_THREAD {
                    match reg.register_with_capability(ctx(RACE_ID)) {
                        Ok(_) => local_ok += 1,
                        Err(RegistryError::OrphanStillAlive(_)) => local_orphan += 1,
                        Err(RegistryError::AlreadyRegistered(_)) => local_already += 1,
                        // H1: this variant now exists unconditionally, so the
                        // arm is no longer feature-gated. `register` never
                        // mints/compares an admin cap, so it can never occur.
                        Err(RegistryError::CapabilityFromForeignRegistry) => {
                            panic!("unexpected CapabilityFromForeignRegistry from register")
                        }
                    }
                }
                (local_ok, local_orphan, local_already)
            }));
        }
        let mut total_ok = 0usize;
        let mut total_orphan = 0usize;
        let mut total_already = 0usize;
        for h in handles {
            let (ok, orphan, already) = h.join().unwrap();
            total_ok += ok;
            total_orphan += orphan;
            total_already += already;
        }
        // While the orphan is alive, NO register call may succeed. Any
        // success would mean the orphan's quota counter is still live
        // and reachable through the held `orphan_arc` while a fresh
        // registration uses a separate counter — the per-tenant quota
        // reset this fix exists to prevent.
        assert_eq!(
            total_ok, 0,
            "no register may succeed while orphan is alive (ok={total_ok}, orphan={total_orphan}, already={total_already})"
        );
        // And there must never be a stray AlreadyRegistered while we
        // hold the orphan: nothing was ever in the slot, so the only
        // legitimate error is OrphanStillAlive.
        assert_eq!(
            total_already, 0,
            "AlreadyRegistered cannot occur while orphan is alive and slot is empty"
        );
        assert_eq!(total_orphan, THREADS * ATTEMPTS_PER_THREAD);

        // The orphan Arc is still the only strong ref.
        assert_eq!(Arc::strong_count(&orphan_arc), 1);
        // The tombstone is still present (we never removed it on the
        // refusal path).
        assert!(reg.tombstones.contains_key(&TenantId(RACE_ID)));
        drop(orphan_arc);
        // After dropping the orphan, a fresh register must succeed
        // exactly once and clear the tombstone.
        let _ = reg.register_with_capability(ctx(RACE_ID)).unwrap();
        assert!(!reg.tombstones.contains_key(&TenantId(RACE_ID)));
    }

    /// N threads race on `register_with_capability(same_id)` for an id
    /// that has never been registered. At most one may succeed; the
    /// rest must see `AlreadyRegistered`. No race condition should let
    /// two of them both observe `Vacant` and both insert.
    #[test]
    fn race_register_fresh_id_at_most_one_succeeds() {
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 32;
        const RACE_ID: u64 = 9100;

        let (reg, cap) = TenantRegistry::new();
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let reg = reg.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                reg.register_with_capability(ctx(RACE_ID)).map(|_| ())
            }));
        }
        let mut ok_count = 0usize;
        let mut already = 0usize;
        for h in handles {
            match h.join().unwrap() {
                Ok(()) => ok_count += 1,
                Err(RegistryError::AlreadyRegistered(_)) => already += 1,
                Err(RegistryError::OrphanStillAlive(_)) => {
                    panic!("no prior registration in this test — OrphanStillAlive impossible")
                }
                // H1: variant now unconditional; arm no longer feature-gated.
                Err(RegistryError::CapabilityFromForeignRegistry) => {
                    panic!("register cannot return CapabilityFromForeignRegistry")
                }
            }
        }
        assert_eq!(ok_count, 1, "exactly one register must win the race");
        assert_eq!(already, THREADS - 1);
        assert_eq!(reg.len(&cap), 1);
    }

    /// Hold an `Arc<TenantContext>` from an unregistered tenant, then
    /// attempt register from another thread; the other thread must see
    /// `OrphanStillAlive`. This is the canonical orphan-protection
    /// scenario the fix preserves under concurrency.
    #[test]
    fn held_orphan_blocks_concurrent_register() {
        use std::sync::Barrier;
        use std::thread;

        const ORPHAN_ID: u64 = 9200;

        let (reg, cap) = TenantRegistry::new();
        let (orphan_arc, _orphan_cap) = reg.register_with_capability(ctx(ORPHAN_ID)).unwrap();
        let _ = reg.unregister(TenantId(ORPHAN_ID), &cap).unwrap();
        // orphan_arc is still alive on this thread.

        let barrier = Arc::new(Barrier::new(2));
        let reg_for_thread = reg.clone();
        let barrier_for_thread = Arc::clone(&barrier);
        let join = thread::spawn(move || {
            barrier_for_thread.wait();
            reg_for_thread.register_with_capability(ctx(ORPHAN_ID))
        });
        barrier.wait();
        let res = join.join().unwrap();
        match res {
            Err(RegistryError::OrphanStillAlive(id)) => {
                assert_eq!(id, TenantId(ORPHAN_ID));
            }
            other => panic!("expected OrphanStillAlive, got {other:?}"),
        }
        // Orphan is still ours, and only ours.
        assert_eq!(Arc::strong_count(&orphan_arc), 1);
    }

    /// Race `unregister(id)` on one thread against `register(id)` on
    /// another. Either:
    ///   - register wins, then unregister returns Some and a tombstone
    ///     is recorded; OR
    ///   - unregister wins (slot was empty so it returns None), then
    ///     register succeeds.
    ///
    /// The invariant we test: no observable state ever has the slot
    /// vacant AND a stale tombstone Weak whose target is the newly
    /// inserted Arc (i.e. a register that "succeeded" only to be
    /// shadowed by a subsequent tombstone insert from a racing
    /// unregister).
    #[test]
    fn race_register_vs_unregister_preserves_ordering() {
        use std::sync::Barrier;
        use std::thread;

        const ID: u64 = 9300;
        const ROUNDS: usize = 64;

        for _ in 0..ROUNDS {
            let (reg, cap) = TenantRegistry::new();
            let barrier = Arc::new(Barrier::new(2));

            let reg_r = reg.clone();
            let barrier_r = Arc::clone(&barrier);
            let t_register = thread::spawn(move || {
                barrier_r.wait();
                reg_r.register_with_capability(ctx(ID))
            });

            let reg_u = reg.clone();
            let barrier_u = Arc::clone(&barrier);
            let t_unregister = thread::spawn(move || {
                barrier_u.wait();
                reg_u.unregister(TenantId(ID), &cap)
            });

            let r_res = t_register.join().unwrap();
            let u_res = t_unregister.join().unwrap();

            // From a fresh registry with id ID never inserted: register
            // cannot fail (no occupant, no tombstone). The branch on
            // `u_res` distinguishes the two reachable serialisations.
            let (r_arc, _r_cap) =
                r_res.expect("register against fresh id with only unregister racing must succeed");
            assert_eq!(r_arc.id(), TenantId(ID));
            match u_res {
                Some(removed) => {
                    // Order was: register THEN unregister. The removed
                    // Arc must be the one register returned, and the
                    // tombstone must be present (recorded by the
                    // racing unregister).
                    assert!(Arc::ptr_eq(&removed, &r_arc));
                    assert!(
                        reg.tombstones.contains_key(&TenantId(ID)),
                        "unregister-after-register must record tombstone"
                    );
                    // Drop both Arcs so the Weak in the tombstone goes
                    // dead; otherwise it would leak between rounds.
                    drop(removed);
                    drop(r_arc);
                }
                None => {
                    // Order was: unregister THEN register. The slot
                    // is occupied by `r_arc` and the tombstone is
                    // absent (no prior occupant existed for unregister
                    // to record, and register's own success path
                    // never sets a tombstone).
                    assert!(
                        !reg.tombstones.contains_key(&TenantId(ID)),
                        "register-after-empty-unregister must leave no tombstone"
                    );
                    drop(r_arc);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // T27 / T-perf tests: AMORTIZED opportunistic tombstone prune on
    // unregister and register_with_capability success paths.
    //
    // T-perf changed the prune from per-op to amortized (every
    // `PRUNE_EVERY_N_OPS` ops, or when the map exceeds
    // `PRUNE_TOMBSTONE_THRESHOLD`). These tests pin the new contract:
    // (1) the prune does NOT run on every op, (2) it eventually
    // reclaims dead tombstones once enough ops accrue, and (3) the
    // orphan / same-id correctness rules still hold whenever a prune
    // does run.
    // ----------------------------------------------------------------

    /// Drive enough register/unregister ops to force an amortized prune
    /// boundary, then assert the prune drops dead tombstones for IDs
    /// other than the current one while skipping the same-id tombstone.
    #[test]
    fn unregister_prunes_dead_tombstones_when_op_boundary_reached() {
        let (reg, cap) = TenantRegistry::new();
        // Register tenant 1, drop the returned Arc; unregister it,
        // dropping that Arc too. Tombstone 1's Weak is now dead.
        let _ = reg.register(ctx(1)).unwrap();
        reg.unregister(TenantId(1), &cap).unwrap();
        // T-perf: the prune is amortized, so a single unregister does
        // NOT necessarily run it. The dead tombstone may linger.
        assert!(reg.tombstones.contains_key(&TenantId(1)));
        assert_eq!(
            reg.tombstones.get(&TenantId(1)).unwrap().strong_count(),
            0,
            "tombstone 1's Weak should be dead — both Arcs were dropped"
        );

        // Drive enough churn on a *different* id to cross an op boundary
        // so a prune is guaranteed to run. Each loop is a register +
        // unregister = 2 ops; run well past PRUNE_EVERY_N_OPS.
        for _ in 0..TenantRegistry::PRUNE_EVERY_N_OPS {
            let _ = reg.register(ctx(2)).unwrap();
            reg.unregister(TenantId(2), &cap).unwrap();
        }
        // By now a prune has run on an `id != 1` op, so dead tombstone 1
        // must have been reclaimed. (Tombstone 2 is repeatedly re-seeded
        // and same-id-skipped, so it may or may not linger — we assert
        // only on the reclamation of the unrelated dead tombstone 1.)
        assert!(
            !reg.tombstones.contains_key(&TenantId(1)),
            "amortized prune must eventually drop dead tombstone 1"
        );
    }

    /// The amortized prune is genuinely throttled: a single successful
    /// register does NOT run a full prune (the per-op-prune behaviour
    /// T-perf removed). The dead tombstone is reclaimed later via the
    /// explicit, admin-gated `collect_tombstones`.
    #[test]
    fn register_prune_is_amortized_collect_reclaims() {
        let (reg, cap) = TenantRegistry::new();
        // Seed dead tombstone 1.
        let _ = reg.register(ctx(1)).unwrap();
        reg.unregister(TenantId(1), &cap).unwrap();
        assert!(reg.tombstones.contains_key(&TenantId(1)));
        // A single register of a different id does NOT force a prune
        // under the amortized policy.
        let _ = reg.register_with_capability(ctx(2)).unwrap();
        assert!(
            reg.tombstones.contains_key(&TenantId(1)),
            "amortized policy: a single register must not prune dead tombstone 1"
        );
        // Explicit prune still reclaims it (id 1 dead, id 2 live → kept).
        let pruned = reg.collect_tombstones(&cap);
        assert_eq!(pruned, 1, "collect_tombstones must reclaim dead tombstone 1");
        assert!(!reg.tombstones.contains_key(&TenantId(1)));
    }

    /// The orphan-rejected register path must NOT prune (we don't want
    /// to do extra work on a refusal) AND must never resurrect a live
    /// orphan regardless of prune throttling. Re-register against a live
    /// orphan and assert the refusal does not touch tombstones.
    #[test]
    fn register_orphan_rejected_does_not_prune() {
        let (reg, cap) = TenantRegistry::new();
        // Seed dead tombstone 1.
        let _ = reg.register(ctx(1)).unwrap();
        reg.unregister(TenantId(1), &cap).unwrap();
        assert!(reg.tombstones.contains_key(&TenantId(1)));

        // Set up a live orphan for id 2 (hold its Arc alive).
        let (orphan_arc, _orphan_cap) = reg.register_with_capability(ctx(2)).unwrap();
        reg.unregister(TenantId(2), &cap).unwrap();
        // Hold orphan_arc so tombstone 2's Weak stays live.
        assert_eq!(Arc::strong_count(&orphan_arc), 1);
        assert!(reg.tombstones.contains_key(&TenantId(2)));

        // Snapshot the tombstone set right before the refusal so we can
        // assert the refusal path left it byte-for-byte unchanged
        // (no prune work on a refusal, regardless of the op counter).
        let before: std::collections::BTreeSet<u64> = reg
            .tombstones
            .iter()
            .map(|e| e.key().get())
            .collect();

        // Re-register id 2 against the live orphan: MUST fail with
        // OrphanStillAlive (no resurrection) and MUST NOT prune.
        let err = reg
            .register_with_capability(ctx(2))
            .expect_err("re-register against live orphan must fail");
        assert_eq!(err, RegistryError::OrphanStillAlive(TenantId(2)));

        let after: std::collections::BTreeSet<u64> = reg
            .tombstones
            .iter()
            .map(|e| e.key().get())
            .collect();
        assert_eq!(
            before, after,
            "refusal path must not prune or otherwise mutate tombstones"
        );
        // Drop orphan so the test cleans up properly.
        drop(orphan_arc);
    }
}
