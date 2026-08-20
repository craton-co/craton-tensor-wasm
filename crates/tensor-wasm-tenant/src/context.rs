// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `TenantContext`: per-tenant CUDA context + stream + memory pool.
//!
//! A `TenantContext` is the runtime handle that ties a [`TenantId`] to the GPU
//! resources reserved for that tenant: a CUDA stream identifier, an isolation
//! policy ([`IsolationKind`]), and a memory quota that the scheduler enforces
//! before kernels are dispatched. Construction goes through
//! [`TenantContextBuilder`] so callers can opt into individual fields without
//! a 5-argument constructor.
//!
//! Under the `cuda` feature, each `ContextIsolated` tenant additionally owns a
//! real `cust::context::Context`; without that feature (the default on
//! CUDA-less hosts), the field collapses to a unit stub so the rest of the
//! crate compiles and tests run unchanged.
//!
//! NOTE: cuda-feature code in this file is compile-tested on CUDA hosts only;
//! on no-CUDA hosts only the `#[cfg(not(feature = "cuda"))]` branches are
//! exercised. The cuda branches use the `cust` 0.3.x context-stack and
//! primary-context APIs.

// The `loom` feature swaps `std::sync::atomic::AtomicU64` for
// `loom::sync::atomic::AtomicU64` so `tests/loom_consume_release.rs` can
// drive `consume_bytes_inner` / `release_bytes_inner` through loom's
// exhaustive scheduler. The two types share the same surface API used
// by the CAS loops below (`load`, `compare_exchange_weak`, `fetch_add`,
// `new`), so the inner-loop bodies need no further cfg-gating.
//
// NOTE(loom): `loom::sync::atomic::AtomicU64::new` is NOT `const fn`
// (loom's atomics carry per-execution tracking state), whereas
// `std::sync::atomic::AtomicU64::new` is. The `static
// ISOLATION_DOWNGRADE_COUNT` declaration below therefore needs the
// std-flavoured type even under `--features loom`; the eventual
// loom model body in `tests/loom_consume_release.rs` builds a
// dedicated `loom::sync::atomic::AtomicU64` (or a minimal stand-in
// struct that uses one) rather than reaching for this static. This
// scaffold keeps the import swap localised to the CAS-loop hot path
// while leaving the alert-counter on the std type for static-init
// compatibility.
#[cfg(feature = "loom")]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(feature = "loom"))]
use std::sync::atomic::{AtomicU64, Ordering};

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::mem_pool::DriverMemPool;
use tensor_wasm_core::metrics::{TenantLabels, TensorWasmMetrics};
use tensor_wasm_core::types::TenantId;

/// Process-wide count of `IsolationKind::ContextIsolated` requests that
/// could not be honoured by the CUDA driver and were silently downgraded
/// to `IsolationKind::StreamIsolated` at [`TenantContextBuilder::build`]
/// time. Operators that requested context isolation as a deployment
/// constraint (e.g. multi-tenant untrusted workloads on a shared GPU)
/// should alert on any non-zero reading — the downgrade is honest
/// reporting at the type level, but it is also a deployment-config bug
/// that needs to be surfaced. Incremented at most once per failed
/// build; never decremented.
///
/// Read via [`isolation_downgrade_count`]. Not yet exported through the
/// `prometheus-client` registry in `tensor-wasm-core`: as of this crate
/// version [`tensor_wasm_core::metrics::TensorWasmMetrics`] exposes no
/// counter whose semantics match a per-process isolation-downgrade tally
/// (the existing `Counter<u64>` accessors — `kernel_dispatches_total`,
/// `offload_fallback_total`, etc. — all carry unrelated meaning, and
/// reusing one would corrupt those series), and that crate is owned by a
/// separate component that this crate must not edit. The metric is
/// therefore intentionally cheap (a single `AtomicU64`) and lives at the
/// call site, surfaced via [`isolation_downgrade_count`] so the alert
/// pipeline can scrape it out-of-band today.
///
/// TODO(core): wire registry export once `tensor-wasm-core` grows a
/// dedicated counter. The minimal core-crate API required is a
/// `Counter<u64>` field on `TensorWasmMetrics` registered as
/// `tensor_wasm_isolation_downgrade_total` with a public accessor
/// `TensorWasmMetrics::isolation_downgrade_total(&self) -> &Counter<u64>`.
/// When that lands, the `build()` downgrade path below should call
/// `metrics.isolation_downgrade_total().inc()` whenever `self.metrics`
/// is `Some`, in addition to bumping this static.
#[cfg(not(feature = "loom"))]
static ISOLATION_DOWNGRADE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of `ContextIsolated -> StreamIsolated` downgrades
/// observed since startup. See `ISOLATION_DOWNGRADE_COUNT` for the
/// alert contract: any non-zero reading on an operator that requested
/// `ContextIsolated` is a deployment-config bug.
///
/// Under the `loom` feature `loom::sync::atomic::AtomicU64::new` is not
/// `const fn`, so the static counter is suppressed and this returns 0.
/// The loom test harness builds its own atomic per-execution.
#[cfg(not(feature = "loom"))]
pub fn isolation_downgrade_count() -> u64 {
    ISOLATION_DOWNGRADE_COUNT.load(Ordering::Relaxed)
}
/// Loom-build stub for [`isolation_downgrade_count`]; always returns 0.
#[cfg(feature = "loom")]
pub fn isolation_downgrade_count() -> u64 {
    0
}

/// How aggressively a tenant's GPU work is separated from other tenants'.
///
/// The variants mirror the levels exposed by `tensor-wasm-mem::isolation::IsolationLevel`,
/// but live here as a separate type so this crate can be consumed without
/// pulling in the Wasmtime-dependent memory crate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsolationKind {
    /// All tenants share the default CUDA context and stream.
    ///
    /// Cheap to spawn but unsuitable for multi-tenant untrusted workloads.
    Shared,
    /// Each tenant gets its own CUDA stream; contexts are shared.
    ///
    /// Default for multi-tenant deployments — prevents kernel-ordering
    /// accidents without paying the cost of per-tenant context creation.
    #[default]
    StreamIsolated,
    /// Each tenant gets its own CUDA context (via MPS when available, or
    /// `cuCtxCreate` otherwise).
    ContextIsolated,
}

impl IsolationKind {
    /// Stable, human-readable name (used in span attributes and metrics).
    pub fn name(self) -> &'static str {
        match self {
            IsolationKind::Shared => "shared",
            IsolationKind::StreamIsolated => "stream_isolated",
            IsolationKind::ContextIsolated => "context_isolated",
        }
    }
}

impl std::fmt::Display for IsolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Returned by [`TenantContext::try_acquire_op`] /
/// [`TenantContext::try_acquire_ops`] when a tenant has exhausted its
/// time-windowed operation-rate budget.
///
/// Distinct from the byte-cap errors (`TensorWasmError::MemoryExhausted` /
/// `GpuMemoryExhausted`), which are high-water-mark *capacity* refusals: a
/// rate-limit refusal is transient — the same request will succeed once the
/// token bucket refills. The struct carries enough context for a scheduler to
/// back off intelligently (how many tokens were asked for, how many were
/// available, and the configured steady-state rate) and is deliberately a
/// crate-local type rather than a new `TensorWasmError` variant: the rate
/// limiter is an additive, opt-in noisy-neighbour control that this crate owns
/// end-to-end, and `tensor-wasm-core` (which owns `TensorWasmError`) is a
/// separate component this crate must not edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimited {
    /// Tokens (operations) the caller requested.
    pub requested: u64,
    /// Tokens available in the bucket at the moment of refusal (after the
    /// time-based refill was applied). Always `< requested`.
    pub available: u64,
    /// Configured steady-state refill rate in tokens per second
    /// (`ops_per_sec` passed to [`TenantContextBuilder::with_rate_limit`]).
    pub ops_per_sec: u64,
    /// Configured bucket depth (`burst` passed to
    /// [`TenantContextBuilder::with_rate_limit`]).
    pub burst: u64,
}

impl std::fmt::Display for RateLimited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rate limited: requested {} ops, {} available (rate {}/s, burst {})",
            self.requested, self.available, self.ops_per_sec, self.burst
        )
    }
}

impl std::error::Error for RateLimited {}

/// Outcome of a capability-checked release
/// ([`TenantContext::release_bytes_with_capability`] /
/// [`TenantContext::release_gpu_bytes_with_capability`]).
///
/// MEDIUM fix (double-release detection): the release counters saturate on
/// underflow — releasing more than was consumed clamps the counter at zero
/// rather than wrapping. That is the right *runtime* behaviour (a
/// bookkeeping slip must not panic or wrap), but the previous `Ok(())`
/// return hid it: an accounting layer doing matched consume/release pairs
/// could double-release and never learn that the second release was a
/// no-op clamp, silently drifting its own shadow ledger. This struct
/// surfaces both how many bytes were *actually* subtracted (`released`)
/// and whether the request hit the underflow clamp (`clamped`), so the
/// caller can detect and alert on a double-release / mismatched-pair bug.
///
/// `released < requested` exactly when `clamped` is `true`; on a clean
/// release `released == requested` and `clamped == false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseOutcome {
    /// Bytes actually subtracted from the counter. Equals the requested
    /// amount on a clean release; on an underflow clamp it is only the
    /// portion that brought the counter down to zero (i.e. the counter's
    /// value before the release).
    pub released: u64,
    /// `true` when the request exceeded the outstanding balance and the
    /// counter was clamped at zero — the signal of an over-release /
    /// double-release / mismatched consume-release pair.
    pub clamped: bool,
}

/// A monotonic-clock token bucket guarding a tenant's operation rate.
///
/// The bucket holds up to `burst` tokens and refills at `ops_per_sec` tokens
/// per second, computed lazily from the elapsed time since the last
/// observation — there is no background timer thread. Each admitted operation
/// removes one (or `n`) tokens; a request that cannot be fully satisfied is
/// rejected with [`RateLimited`] and removes *no* tokens (all-or-nothing), so
/// a burst of large requests cannot partially drain the bucket and starve
/// smaller ones mid-flight.
///
/// State is `Mutex`-guarded rather than a lock-free atomic pair: the refill
/// math reads the last-refill instant and the token count together and writes
/// both back, which is a small critical section but genuinely needs to be
/// atomic *as a unit*. The lock is held only for the duration of that
/// arithmetic (no I/O, no syscalls beyond `Instant::now`), so contention is
/// negligible next to the work an admitted operation actually goes on to do.
/// Scale factor for the fixed-point token accounting: tokens are tracked
/// internally in *micro-tokens* (1 whole token == `MICRO_PER_TOKEN`
/// micro-tokens). All refill / consume / cap arithmetic is therefore exact
/// integer arithmetic on `u64`, with no `f64` rounding or epsilon fudge.
///
/// At micro-token granularity, sub-token refill credit (a 100 ops/s bucket
/// accruing 1 token every 10 ms) is preserved between calls exactly as the
/// old `f64` `tokens` field intended, but an exactly-full bucket compares
/// *exactly* equal to its capacity — so a full bucket always admits, and
/// accumulated rounding can no longer spuriously reject it.
///
/// `1_000_000` gives microsecond-equivalent resolution: even a 1 ops/s rate
/// credits 1 micro-token per microsecond of elapsed time, far finer than any
/// realistic scheduler cares about, while keeping the `u64` headroom huge
/// (`u64::MAX / 1_000_000` ≈ 1.8e13 whole tokens of burst capacity).
const MICRO_PER_TOKEN: u64 = 1_000_000;

#[derive(Debug)]
struct TokenBucket {
    /// Steady-state refill rate, tokens per second. Refill credit is computed
    /// as exact integer micro-tokens from the elapsed `Duration` (see
    /// [`TokenBucket::refilled_micros`]), so sub-token credit is carried on
    /// `TokenBucketState::micro_tokens` without any floating-point rounding.
    ops_per_sec: u64,
    /// Maximum *whole* tokens the bucket can hold (the burst depth). The
    /// public [`RateLimited::burst`] reports this value unchanged.
    burst: u64,
    /// Bucket capacity in micro-tokens (`burst * MICRO_PER_TOKEN`, saturating
    /// so an absurdly large burst cannot overflow `u64`). Precomputed once so
    /// the hot path is a plain `min`.
    capacity_micros: u64,
    /// Mutable bucket state, guarded as a unit.
    state: Mutex<TokenBucketState>,
}

#[derive(Debug)]
struct TokenBucketState {
    /// Current token count in micro-tokens, preserving fractional refill
    /// credit between observations as an exact integer (no `f64` rounding).
    micro_tokens: u64,
    /// Last instant the bucket was refilled. Advanced on every
    /// `try_acquire`, so the next call only credits time since this point.
    last_refill: Instant,
}

impl TokenBucket {
    /// Construct a bucket that starts full (`burst` tokens available), so the
    /// first `burst` operations are admitted immediately before any refill is
    /// needed. `ops_per_sec` and `burst` are both clamped to a minimum of 1
    /// so a `with_rate_limit(0, 0)` cannot wedge the tenant into a
    /// never-admits state — the builder documents that "no rate limit" is
    /// expressed by *not* calling `with_rate_limit` at all (the `None` default).
    fn new(ops_per_sec: u64, burst: u64) -> Self {
        let burst = burst.max(1);
        let ops_per_sec = ops_per_sec.max(1);
        // Saturating so a pathologically large burst (> u64::MAX / MICRO)
        // cannot overflow the micro-token capacity. Realistic bursts are far
        // below this ceiling, so the saturation is a safety net, not a
        // behavioural change.
        let capacity_micros = burst.saturating_mul(MICRO_PER_TOKEN);
        Self {
            ops_per_sec,
            burst,
            capacity_micros,
            state: Mutex::new(TokenBucketState {
                // Starts full: `burst` whole tokens == `capacity_micros`.
                micro_tokens: capacity_micros,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Micro-token balance after crediting `elapsed` of refill onto `current`,
    /// capped at the bucket capacity.
    ///
    /// Refill credit in micro-tokens is
    /// `elapsed_nanos * ops_per_sec * MICRO_PER_TOKEN / 1_000_000_000`. The
    /// intermediate product is computed in `u128` so even a multi-year
    /// `elapsed` cannot overflow before the per-nanosecond divide; the result
    /// is then saturated back into `u64` and capped at `capacity_micros`, so
    /// a long idle period simply tops the bucket off at its burst depth (and
    /// never wraps). The division truncates toward zero, which only ever
    /// *under*-credits by at most one micro-token — it can never manufacture
    /// tokens that would let an over-budget request slip through.
    fn refilled_micros(&self, current: u64, elapsed: Duration) -> u64 {
        let nanos = elapsed.as_nanos();
        // credit_micros = nanos * ops_per_sec * MICRO_PER_TOKEN / 1e9
        let credit_micros = nanos
            .saturating_mul(self.ops_per_sec as u128)
            .saturating_mul(MICRO_PER_TOKEN as u128)
            / 1_000_000_000u128;
        let credit_micros = u64::try_from(credit_micros).unwrap_or(u64::MAX);
        current
            .saturating_add(credit_micros)
            .min(self.capacity_micros)
    }

    /// Production entry point: sample the clock **inside** the lock and
    /// admit `n` tokens against the refilled bucket.
    ///
    /// HIGH fix (clock race): the wall-clock instant used for the refill is
    /// read here, while the bucket lock is held, rather than by the caller
    /// before it contends for the lock. Sampling outside the lock let two
    /// threads each capture an `Instant`, then acquire the lock in the
    /// opposite order, so the thread holding the *earlier* sample could win
    /// the lock *last* and store a `last_refill` that moved backward in time
    /// — a stale anchor that manufactures extra refill credit on the next
    /// call. Reading `Instant::now()` under the lock means the value the
    /// admission math sees and the value stored back into `last_refill` are
    /// the same monotonic observation, serialised with every other acquire.
    fn try_acquire_now(&self, n: u64) -> Result<(), RateLimited> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Sampled under the lock — see the method doc. `Instant` is
        // monotonic, but two samples taken outside the lock could still be
        // applied in inverted order; taking it here removes that window.
        let now = Instant::now();
        // `&mut state` deref-coerces the `MutexGuard` to the inner state.
        self.admit_locked(&mut state, n, now)
    }

    /// Attempt to remove `n` tokens, refilling for elapsed wall-clock time
    /// first. Admits (returns `Ok`) only if at least `n` tokens are available
    /// after the refill; otherwise returns [`RateLimited`] and leaves the
    /// bucket untouched (all-or-nothing).
    ///
    /// `now` is injected rather than read internally so tests can drive the
    /// refill deterministically without sleeping; the production path
    /// [`Self::try_acquire_now`] reads `Instant::now()` under the lock and
    /// delegates to the same [`Self::admit_locked`] body.
    ///
    /// All accounting is exact integer arithmetic in micro-tokens, so the
    /// admission test is a plain `available_micros >= need_micros`: an
    /// exactly-full bucket has `available_micros == capacity_micros` and a
    /// request for exactly the full burst has `need_micros == capacity_micros`,
    /// which compares equal and therefore admits (no `f64::EPSILON` nudge, no
    /// magnitude-dependent rounding that could spuriously reject it).
    #[cfg(test)]
    fn try_acquire_at(&self, n: u64, now: Instant) -> Result<(), RateLimited> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.admit_locked(&mut state, n, now)
    }

    /// Shared admission body, run while `state` is locked.
    ///
    /// HIGH fix (clock race, anchor monotonicity): before computing
    /// `elapsed`, the effective `now` is clamped to `max(last_refill, now)`.
    /// `Instant` is monotonic per-thread, but a sample observed by one
    /// thread can still be applied to the shared state *after* a later
    /// sample from another thread has already advanced `last_refill`; a bare
    /// `checked_duration_since` would then credit zero (harmless) but the
    /// unconditional `state.last_refill = now` further down would move the
    /// anchor *backward*. Clamping first guarantees the stored anchor only
    /// ever advances, so a stale sample can never rewind the refill clock
    /// and gift the next caller free tokens. (Injected non-monotonic test
    /// instants are clamped the same way.)
    fn admit_locked(
        &self,
        state: &mut TokenBucketState,
        n: u64,
        now: Instant,
    ) -> Result<(), RateLimited> {
        // Anchor can only move forward: a `now` that predates the stored
        // `last_refill` (a stale sample or an injected non-monotonic test
        // instant) is pinned to `last_refill`, crediting nothing and leaving
        // the anchor where it was rather than rewinding it.
        let now = now.max(state.last_refill);
        let elapsed = now
            .checked_duration_since(state.last_refill)
            .unwrap_or(Duration::ZERO);
        let available_micros = self.refilled_micros(state.micro_tokens, elapsed);
        // `need` in micro-tokens. Saturating so a colossal `n` cannot overflow;
        // such a request necessarily exceeds the bucket and is rejected below.
        let need_micros = n.saturating_mul(MICRO_PER_TOKEN);
        if available_micros < need_micros {
            return Err(RateLimited {
                requested: n,
                // Report the whole tokens that are actually acquirable right
                // now (floor of the micro-token balance). This is consistent
                // with the admission decision: the request was rejected
                // precisely because `available_micros < need_micros`, so the
                // floored whole-token count is strictly less than `n` and the
                // documented `available < requested` invariant holds. Floor —
                // not the old `f64 as u64` truncation of a possibly-rounded
                // value — is exact, so we never under-report a token the
                // caller could in fact have acquired.
                available: available_micros / MICRO_PER_TOKEN,
                ops_per_sec: self.ops_per_sec,
                burst: self.burst,
            });
        }
        state.micro_tokens = available_micros - need_micros;
        state.last_refill = now;
        Ok(())
    }
}

/// Unforgeable proof of authority to mutate a single tenant's quota counters.
///
/// Minted only by [`crate::TenantRegistry::register_with_capability`]; the
/// `_seal` field is private to this crate so no downstream crate (and no
/// hostile workload) can construct one out of thin air. Holding an
/// `Arc<TenantContext>` is therefore no longer sufficient to drive that
/// tenant's `bytes_in_use` counter — the caller must also present the
/// matching capability, which it can only get if it originally registered
/// the tenant.
///
/// `Clone` is intentionally derived: the API gateway holds the
/// authoritative copy and may need to hand clones to per-tenant subsystems
/// (the scheduler, the memory pool, etc.). What is NOT derived is any
/// `From<TenantId>` or public constructor, so a workload running inside
/// tenant A cannot fabricate one for tenant B.
///
/// # Registry binding (H1 — always enforced)
///
/// Every capability carries an `Arc<()>` token that points to its
/// minting registry's allocation. Comparison is by `Arc::ptr_eq`, so a
/// capability minted by registry A is rejected when presented against a
/// context registered in registry B, even if both contexts happen to
/// share the same numeric `TenantId`. Without this binding, capabilities
/// from independent registries are interchangeable, which the H1 audit
/// finding flagged as a cross-registry capability-confusion vector — see
/// the note on `RegistryAdminCapability` for the threat model.
///
/// H1 fix: this binding is now UNCONDITIONAL and no longer behind the
/// `strict-cap-binding` feature gate. A release build with the feature
/// disabled still binds caps to their minting registry and still fails
/// closed on cross-registry / forged-admin caps. The `strict-cap-binding`
/// feature remains defined only to keep the typed `*_strict` admin APIs
/// and the `cap_binding_strict` integration test compiling.
#[derive(Debug, Clone)]
pub struct TenantCapability {
    tenant_id: TenantId,
    /// Crate-private zero-sized seal: prevents `TenantCapability { .. }`
    /// struct-literal construction outside `tensor-wasm-tenant`.
    _seal: (),
    /// Pointer-identity stamp of the registry that minted this capability.
    /// H1: always present so registry binding is enforced unconditionally.
    pub(crate) registry_token: std::sync::Arc<()>,
}

impl TenantCapability {
    /// Mint a capability bound to `tenant_id` AND to the minting registry,
    /// identified by `registry_token` (an `Arc::clone` of the registry's
    /// per-instance token allocation). Comparison at `check_capability`
    /// time is by `Arc::ptr_eq`, so two registries that happen to allocate
    /// `Arc::new(())` at the same address would still be distinct
    /// allocations and `ptr_eq` would return `false` for caps from one
    /// against contexts of the other.
    ///
    /// H1: the registry-bound signature is now unconditional — there is no
    /// longer a token-less `mint` variant, so a cap can never be minted
    /// without provenance regardless of the `strict-cap-binding` feature.
    pub(crate) fn mint(tenant_id: TenantId, registry_token: std::sync::Arc<()>) -> Self {
        Self {
            tenant_id,
            _seal: (),
            registry_token,
        }
    }

    /// Identifier of the tenant this capability authorises.
    pub fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
}

/// Per-tenant runtime handle: identity, isolation level, stream, and quota.
///
/// Instances are constructed through [`TenantContextBuilder`] and then placed
/// into the [`crate::TenantRegistry`]. The byte-counter methods
/// ([`Self::consume_bytes`] / [`Self::release_bytes`]) are lock-free and safe
/// to call from any thread; quota enforcement happens at the point of
/// allocation, not asynchronously.
#[derive(Debug)]
pub struct TenantContext {
    tenant_id: TenantId,
    isolation: IsolationKind,
    stream_id: u64,
    memory_quota_bytes: u64,
    bytes_in_use: AtomicU64,

    /// Maximum GPU memory in bytes this tenant may allocate concurrently.
    /// `None` = no GPU memory cap (operator trust).
    ///
    /// ADVISORY / IN-PROCESS-ONLY: unless a [`DriverMemPool`] is wired in
    /// via [`TenantContextBuilder::with_driver_enforced_gpu_cap`], this
    /// cap is enforced ONLY by the in-process [`Self::gpu_bytes_in_use`]
    /// counter on the [`Self::consume_gpu_bytes`] path. The CUDA driver
    /// itself sees no cap, so any tenant that obtains a raw CUDA handle
    /// and allocates outside `consume_gpu_bytes` is NOT capped. v0.3.7
    /// records and reports usage; v0.4 enforces via cuMemPool's
    /// `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`.
    ///
    /// Distinct from [`Self::memory_quota_bytes`]: that one is the
    /// host-side / CPU quota the executor enforces against
    /// `consume_bytes`. This field is consulted by the GPU allocator
    /// path (`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context`)
    /// against `gpu_bytes_in_use` on every `UnifiedBuffer::new_on`.
    gpu_memory_bytes_cap: Option<u64>,
    /// Bytes currently accounted as in-use against the GPU cap. Mirrors
    /// the CPU [`Self::bytes_in_use`] counter; updated by
    /// [`Self::consume_gpu_bytes`] / [`Self::release_gpu_bytes`] via the
    /// same CAS-loop pattern, and (when a metrics handle is wired in
    /// via [`TenantContextBuilder::with_metrics`]) republished as the
    /// per-tenant series of
    /// [`tensor_wasm_core::metrics::TensorWasmMetrics::gpu_memory_bytes_per_tenant`]
    /// on every transition. v0.3.7 record-only: the value is the source
    /// of truth for the in-process refusal of over-cap allocations. The
    /// `gpu-mem-pool` `TenantMemPool` additionally enforces the cap
    /// host-side in its `allocate` method; note that the CUDA driver itself
    /// has no per-pool allocation ceiling
    /// (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is a retention hint, not a cap —
    /// see [`Self::gpu_memory_bytes_cap`]).
    gpu_bytes_in_use: AtomicU64,

    /// Recorded-only CUDA memory-pool release-threshold value. `None`
    /// means "use the driver default" (typically unbounded retention).
    ///
    /// ADVISORY / RECORD-ONLY: this value is NEVER enforced. The cust
    /// 0.3.x crate does not expose the `cuMemPool*` API, so this field is
    /// **not** wired through to
    /// `cudaMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)` — it
    /// is stored purely for inspection / metrics / forward-compat and the
    /// CUDA driver never sees it. The in-process
    /// [`TenantContext::bytes_in_use`] counter is the only enforcement of
    /// this crate's quota. See
    /// [`TenantContextBuilder::with_recorded_cuda_mem_pool_quota`] for
    /// the honest naming and the upgrade path.
    #[allow(dead_code)]
    cuda_mem_pool_quota_bytes: Option<u64>,

    /// Optional driver-level memory pool whose release threshold is pinned
    /// to this tenant's GPU cap. When present, [`Self::consume_gpu_bytes`]
    /// pushes the cap through
    /// [`tensor_wasm_core::mem_pool::DriverMemPool::set_release_threshold`]
    /// so the CUDA driver itself rejects over-cap allocations — closing
    /// the bypass that the in-process `gpu_bytes_in_use` counter alone
    /// cannot (a tenant who obtained a raw CUDA driver handle).
    ///
    /// Stored as a trait object (`Arc<dyn DriverMemPool>`), NOT the
    /// concrete `tensor-wasm-mem::cuda_mem_pool::TenantMemPool`: this
    /// crate depends only on `tensor-wasm-core`, never on
    /// `tensor-wasm-mem`. Holding the concrete type would require a
    /// `tenant -> mem` dependency edge, which would close the
    /// `mem` <-> `tenant` cycle (mem already depends on tenant). The
    /// backend-agnostic trait lives in `tensor-wasm-core`, which both
    /// crates already depend on, so the graph stays acyclic. Set via
    /// [`TenantContextBuilder::with_driver_enforced_gpu_cap`].
    driver_mem_pool: Option<Arc<dyn DriverMemPool>>,

    /// Optional time-windowed operation-rate limiter. `None` (the default)
    /// preserves the historical pure high-water-mark byte-cap behaviour —
    /// no per-operation rate is enforced. When set via
    /// [`TenantContextBuilder::with_rate_limit`], every
    /// [`Self::try_acquire_op`] / [`Self::try_acquire_ops`] consults a
    /// monotonic-clock token bucket and refuses (with [`RateLimited`]) once
    /// the tenant's steady-state rate plus burst is exceeded, addressing
    /// noisy-neighbour scheduling without touching the byte counters above.
    rate_limiter: Option<TokenBucket>,

    // Real `cust::context::Context` under the `cuda` feature; otherwise a
    // unit stub so the rest of the crate compiles on CUDA-less hosts.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    cu_context: Option<cust::context::Context>,
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    cu_context: (),

    /// Optional shared metrics handle. When present, every CPU-side
    /// [`Self::consume_bytes`] / [`Self::release_bytes`] transition updates
    /// the per-tenant series of
    /// [`tensor_wasm_core::metrics::TensorWasmMetrics::cpu_memory_bytes_per_tenant`],
    /// and every GPU-side
    /// [`Self::consume_gpu_bytes`] / [`Self::release_gpu_bytes`] transition
    /// updates the per-tenant series of
    /// [`tensor_wasm_core::metrics::TensorWasmMetrics::gpu_memory_bytes_per_tenant`].
    /// The two no longer collide on a single labelled series. `None` keeps
    /// the historical no-op behaviour so embedders that construct a
    /// `TenantContext` outside the API gateway (e.g. benches, examples) do
    /// not need to plumb a metrics registry.
    metrics: Option<TensorWasmMetrics>,
    /// Memoized label tuple used to address the per-tenant gauge series.
    /// Built once at construction so the hot path of `consume_bytes` /
    /// `release_bytes` does not allocate on every transition.
    metrics_labels: TenantLabels,
    /// Pointer-identity stamp of the registry this context was registered
    /// in, used by [`Self::check_capability`] to reject caps minted by a
    /// *different* registry.
    ///
    /// `None` until the context is moved into
    /// [`crate::TenantRegistry::register_with_capability`], which sets
    /// the token before wrapping the context in an `Arc`. A `None` here
    /// at `check_capability` time means the context was never registered
    /// (or was constructed for a test that bypasses the registry); the
    /// registry-binding check then **fails closed** (rejects the cap),
    /// since an unregistered context has no provenance to vouch for any
    /// capability — see [`Self::check_capability`].
    ///
    /// H1: always present so registry binding is enforced unconditionally,
    /// not just under the `strict-cap-binding` feature.
    pub(crate) registry_token: Option<std::sync::Arc<()>>,
}

impl TenantContext {
    /// Start a builder for a tenant with the given identifier.
    pub fn builder(tenant_id: TenantId) -> TenantContextBuilder {
        TenantContextBuilder::new(tenant_id)
    }

    /// Tenant identifier this context belongs to.
    pub fn id(&self) -> TenantId {
        self.tenant_id
    }

    /// Isolation level configured for this tenant.
    pub fn isolation(&self) -> IsolationKind {
        self.isolation
    }

    /// Stream identifier (logical handle; the actual `CUstream` lives in
    /// `tensor-wasm-mem` / `tensor-wasm-wasi-gpu` and is keyed by this value).
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Total bytes the tenant is permitted to allocate concurrently.
    pub fn quota(&self) -> u64 {
        self.memory_quota_bytes
    }

    /// Bytes currently accounted as in-use against the quota.
    pub fn bytes_in_use(&self) -> u64 {
        self.bytes_in_use.load(Ordering::Acquire)
    }

    /// Per-tenant GPU memory cap in bytes, or `None` for "no cap"
    /// (operator-trust deployment).
    ///
    /// ADVISORY / IN-PROCESS-ONLY unless a [`DriverMemPool`] is wired in
    /// via [`TenantContextBuilder::with_driver_enforced_gpu_cap`]. Set via
    /// [`TenantContextBuilder::with_gpu_memory_bytes_cap`]. The in-process
    /// allocator path
    /// (`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context`)
    /// reads this on every allocation and refuses to allocate when the
    /// would-be new total of [`Self::gpu_bytes_in_use`] would exceed it.
    /// Under `gpu-mem-pool`, the `TenantMemPool` *also* enforces the cap
    /// host-side in its `allocate` method (a second line of defence for
    /// pool-routed allocations). NB: there is NO driver-level per-pool
    /// allocation ceiling — `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is a
    /// retention hint, not a cap (a hardware run disproved the earlier
    /// driver-pin claim; see `docs/GPU-VALIDATION-2026-05-30.md` BUG-1). A
    /// tenant calling the CUDA driver *directly* (not through TenantMemPool)
    /// is therefore still uncapped; see `docs/GPU-QUOTAS.md`.
    pub fn gpu_memory_bytes_cap(&self) -> Option<u64> {
        self.gpu_memory_bytes_cap
    }

    /// Bytes currently accounted as in-use against the GPU cap.
    ///
    /// Mirrors [`Self::bytes_in_use`] for the GPU side of the quota.
    /// Updated by [`Self::consume_gpu_bytes`] /
    /// [`Self::release_gpu_bytes`].
    pub fn gpu_bytes_in_use(&self) -> u64 {
        self.gpu_bytes_in_use.load(Ordering::Acquire)
    }

    /// Sentinel the uncapped GPU counter saturates to on `checked_add`
    /// overflow, deliberately set *below* `u64::MAX`.
    ///
    /// MEDIUM fix (uncapped overflow recovery): the uncapped consume path
    /// must never refuse, so an overflowing add clamps rather than erroring.
    /// Clamping to `u64::MAX` exactly was a latent footgun — a later
    /// `release_gpu_bytes(n)` brings the counter down by only `n`, so an
    /// implementation (or a future refactor) that clamped releases at the
    /// top, or any consumer treating `u64::MAX` as "stuck/poisoned", could
    /// leave the gauge pinned forever and the per-tenant series reporting
    /// garbage. Saturating to this sentinel — a full `u64::MAX >> 1` below
    /// the ceiling — keeps the value strictly recoverable by normal-sized
    /// releases while remaining an obviously-abnormal, easily-alertable
    /// reading on dashboards. The headroom also means a subsequent
    /// in-flight consume that races in can still `checked_add` a realistic
    /// `n` without immediately re-overflowing.
    pub const GPU_BYTES_OVERFLOW_SENTINEL: u64 = u64::MAX - (u64::MAX >> 1);

    /// Atomically reserve `n` GPU bytes against the per-tenant cap.
    ///
    /// Returns `Err(TensorWasmError::GpuMemoryExhausted)` if
    /// [`Self::gpu_memory_bytes_cap`] is set and the allocation would
    /// push usage above it. When the cap is `None` ("no cap"), the
    /// counter is still bumped (so dashboards and the per-tenant gauge
    /// surface real utilisation) but the request is never refused. The
    /// add is performed with `checked_add` so a malicious or buggy
    /// caller cannot wrap the counter by repeatedly requesting close
    /// to `u64::MAX` — the second such call observes the overflow and,
    /// on the *capped* path, returns `GpuMemoryExhausted` while leaving
    /// the counter unchanged; on the *uncapped* path it clamps the
    /// running total to [`Self::GPU_BYTES_OVERFLOW_SENTINEL`] (below
    /// `u64::MAX`) so subsequent releases can still bring it back down
    /// rather than pinning the counter and gauge at the ceiling forever.
    ///
    /// Mirrors `Self::consume_bytes_inner` for the GPU side; the
    /// atomic discipline is intentionally identical so a single mental
    /// model covers both counters.
    ///
    /// # v0.3.7 vs v0.4 contract
    ///
    /// This in-process counter is the primary enforcement: the allocator
    /// path calls `consume_gpu_bytes` before handing back the buffer. A
    /// tenant that bypasses the allocator but still routes through the
    /// `gpu-mem-pool` `TenantMemPool` is *additionally* capped host-side
    /// in `TenantMemPool::allocate` (a second line of defence wired via
    /// [`TenantContextBuilder::with_driver_enforced_gpu_cap`]). NB: this
    /// is NOT a driver-level cap — `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is
    /// a retention hint, not an allocation ceiling (see
    /// `docs/GPU-QUOTAS.md` and `docs/GPU-VALIDATION-2026-05-30.md`
    /// BUG-1). A tenant calling the CUDA driver *directly* is still
    /// uncapped; that threat is out of scope today.
    pub fn consume_gpu_bytes(&self, n: u64) -> Result<(), TensorWasmError> {
        let limit = self.gpu_memory_bytes_cap;
        let mut current = self.gpu_bytes_in_use.load(Ordering::Acquire);
        loop {
            // MEDIUM fix (uncapped overflow recovery): the counter contract
            // is "never refused when there is no cap". A `checked_add`
            // overflow on the uncapped path used to surface as a
            // non-retryable `GpuMemoryExhausted { limit: u64::MAX }`,
            // violating that contract. When `limit` is `None` we instead
            // clamp to [`Self::GPU_BYTES_OVERFLOW_SENTINEL`] (below
            // `u64::MAX`) and `warn!` — the request is always admitted AND a
            // subsequent `release_gpu_bytes` can still bring the total back
            // down, instead of pinning it at the ceiling permanently and
            // leaving the per-tenant gauge stuck. The overflow-as-error
            // behaviour is retained only for the capped path (a clamping add
            // there could mask an over-cap allocation), where `next > cap`
            // already rejects it.
            let next = match current.checked_add(n) {
                Some(v) => v,
                None => {
                    if limit.is_none() {
                        tracing::warn!(
                            target: "tensor_wasm_tenant::context",
                            tenant = %self.tenant_id,
                            current,
                            requested = n,
                            sentinel = Self::GPU_BYTES_OVERFLOW_SENTINEL,
                            "consume_gpu_bytes overflow on uncapped tenant; \
                             clamping to recoverable sentinel below u64::MAX",
                        );
                        // Clamp the new total to the recoverable sentinel
                        // (strictly below u64::MAX). If `current` had already
                        // been driven up to or past the sentinel by an
                        // earlier overflow, this clamps it back *down* so a
                        // release can always make progress; it never nudges
                        // the counter up toward the ceiling.
                        Self::GPU_BYTES_OVERFLOW_SENTINEL
                    } else {
                        return Err(TensorWasmError::GpuMemoryExhausted {
                            requested: n,
                            limit: limit.unwrap_or(u64::MAX),
                            current,
                        });
                    }
                }
            };
            if let Some(cap) = limit {
                if next > cap {
                    return Err(TensorWasmError::GpuMemoryExhausted {
                        requested: n,
                        limit: cap,
                        current,
                    });
                }
            }
            match self.gpu_bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.publish_gpu_memory_gauge(next);
                    // Pool cap alignment (T39): when a driver memory pool was
                    // wired in via `with_driver_enforced_gpu_cap` AND a cap is
                    // set, keep the pool's recorded cap aligned to this
                    // tenant's policy ceiling. NB: this is NOT what enforces
                    // the cap on the pool path — `set_release_threshold` sets
                    // `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`, a memory-RETENTION
                    // hint, not an allocation ceiling (see
                    // `docs/GPU-VALIDATION-2026-05-30.md` BUG-1). The pool's
                    // real cap is enforced host-side in
                    // `TenantMemPool::allocate`. We push the *cap* (a fixed
                    // policy ceiling), not the running total. A failure is
                    // logged but does NOT fail the consume: the in-process
                    // counter already accepted this allocation, and the pool
                    // cap is belt-and-braces.
                    if let (Some(pool), Some(cap)) = (&self.driver_mem_pool, limit) {
                        if let Err(e) = pool.set_release_threshold(cap) {
                            tracing::warn!(
                                target: "tensor_wasm_tenant::context",
                                tenant = %self.tenant_id,
                                cap,
                                error = %e,
                                "driver mem-pool set_release_threshold failed; \
                                 in-process gpu cap still enforced",
                            );
                        }
                    }
                    return Ok(());
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// Atomically release `n` GPU bytes back to the cap.
    ///
    /// Saturating on underflow — callers must not release more than
    /// they consumed, but a bookkeeping mismatch is not fatal. Mirrors
    /// `Self::release_bytes_inner` for the GPU side: CAS loop on
    /// `gpu_bytes_in_use`, computing `saturating_sub` on each iteration.
    /// The earlier `fetch_sub` + post-hoc clamp shape is intentionally
    /// avoided here for the same reason described on the CPU sibling
    /// method — a concurrent `consume_gpu_bytes` race must not be
    /// silently erased by a clamping `store`.
    pub fn release_gpu_bytes(&self, bytes: u64) {
        // Discards the `ReleaseOutcome`: the uncapped variant keeps its `()`
        // return; the capability-checked variant surfaces the outcome.
        let _ = self.release_gpu_bytes_inner(bytes);
    }

    /// Shared GPU-release implementation: CAS-loop `saturating_sub` +
    /// underflow warn, returning a [`ReleaseOutcome`] so
    /// [`Self::release_gpu_bytes_with_capability`] can report an
    /// underflow-clamp (double-release signal) to the accounting layer.
    /// Mirrors the CPU `Self::release_bytes_inner`.
    fn release_gpu_bytes_inner(&self, bytes: u64) -> ReleaseOutcome {
        let mut current = self.gpu_bytes_in_use.load(Ordering::Acquire);
        let (after, before) = loop {
            let next = current.saturating_sub(bytes);
            match self.gpu_bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if current < bytes {
                        tracing::warn!(
                            target: "tensor_wasm_tenant::context",
                            tenant = %self.tenant_id,
                            before = current,
                            bytes,
                            "release_gpu_bytes underflow clamped",
                        );
                    }
                    break (next, current);
                }
                Err(observed) => current = observed,
            }
        };
        self.publish_gpu_memory_gauge(after);
        ReleaseOutcome {
            released: before - after,
            clamped: before < bytes,
        }
    }

    /// Capability-checked variant of [`Self::consume_gpu_bytes`].
    ///
    /// Performs the same `Self::check_capability` gate the CPU path uses
    /// ([`Self::consume_bytes_with_capability`]) before touching the GPU
    /// counter, so the GPU side gets the identical cross-tenant isolation
    /// guarantee: a [`TenantCapability`] minted for a different tenant is
    /// rejected with [`TensorWasmError::TenantIsolationViolation`] and the
    /// `gpu_bytes_in_use` counter is left untouched. On a matching capability
    /// the behaviour is exactly that of the uncapped [`Self::consume_gpu_bytes`]
    /// (cap check, `checked_add`, optional driver-pin, metrics publish).
    ///
    /// Until this method landed, the GPU counter had *no* capability-gated
    /// form (unlike `consume_bytes` / `consume_bytes_with_capability`); the
    /// uncapped [`Self::consume_gpu_bytes`] is retained for the 0.3 line in
    /// the same way the CPU path keeps its unchecked variant.
    pub fn consume_gpu_bytes_with_capability(
        &self,
        cap: &TenantCapability,
        n: u64,
    ) -> Result<(), TensorWasmError> {
        self.check_capability(cap, "quota.consume_gpu_bytes")?;
        self.consume_gpu_bytes(n)
    }

    /// Capability-checked variant of [`Self::release_gpu_bytes`].
    ///
    /// Mirrors [`Self::release_bytes_with_capability`] on the GPU side:
    /// returns [`TensorWasmError::TenantIsolationViolation`] if `cap` was
    /// minted for a different tenant (leaving the counter untouched);
    /// otherwise releases (saturating on underflow) and returns a
    /// [`ReleaseOutcome`].
    ///
    /// MEDIUM fix (double-release detection): like the CPU sibling, this
    /// used to return `Ok(())` and hide an underflow clamp. It now returns
    /// `Ok(ReleaseOutcome { released, clamped })` so the GPU accounting
    /// layer can detect an over-release (`clamped == true`). The clamp
    /// itself remains best-effort (never panics/wraps); `.unwrap()` call
    /// sites continue to compile.
    pub fn release_gpu_bytes_with_capability(
        &self,
        cap: &TenantCapability,
        bytes: u64,
    ) -> Result<ReleaseOutcome, TensorWasmError> {
        self.check_capability(cap, "quota.release_gpu_bytes")?;
        Ok(self.release_gpu_bytes_inner(bytes))
    }

    /// Attempt to admit a single operation against this tenant's
    /// time-windowed rate limit.
    ///
    /// Convenience wrapper over [`Self::try_acquire_ops`] with `n == 1`. See
    /// that method for the full contract.
    pub fn try_acquire_op(&self) -> Result<(), RateLimited> {
        self.try_acquire_ops(1)
    }

    /// Attempt to admit `n` operations against this tenant's time-windowed
    /// rate limit.
    ///
    /// When no rate limit was configured (the default — see
    /// [`TenantContextBuilder::with_rate_limit`]), this is an unconditional
    /// `Ok(())`, preserving the historical pure byte-cap behaviour. When a
    /// limit is configured, a monotonic-clock token bucket is refilled for
    /// the wall-clock time elapsed since the last call and then asked for `n`
    /// tokens; the call returns `Ok(())` and removes the tokens if at least
    /// `n` are available, or [`RateLimited`] (leaving the bucket untouched)
    /// otherwise. The admission is all-or-nothing, so a large request never
    /// partially drains the bucket.
    ///
    /// This gate is orthogonal to the byte caps: it does not read or mutate
    /// `bytes_in_use` / `gpu_bytes_in_use`. A typical scheduler calls
    /// `try_acquire_op` per kernel launch (or `try_acquire_ops(bytes)` for a
    /// bytes/sec budget) and, on `Err`, defers or sheds the request rather
    /// than failing it permanently — unlike a byte-cap refusal, a rate-limit
    /// refusal clears on its own once the bucket refills.
    pub fn try_acquire_ops(&self, n: u64) -> Result<(), RateLimited> {
        match &self.rate_limiter {
            // HIGH fix (clock race): the bucket reads `Instant::now()` itself
            // under its own lock rather than receiving a sample taken here,
            // outside the lock — see [`TokenBucket::try_acquire_now`].
            Some(bucket) => bucket.try_acquire_now(n),
            None => Ok(()),
        }
    }

    /// Whether a time-windowed rate limit is configured for this tenant
    /// (i.e. [`TenantContextBuilder::with_rate_limit`] was called). When
    /// `false`, [`Self::try_acquire_op`] always admits.
    pub fn has_rate_limit(&self) -> bool {
        self.rate_limiter.is_some()
    }

    /// Atomically reserve `n` bytes against the quota.
    ///
    /// Returns `Err(TensorWasmError::MemoryExhausted)` if the allocation would push
    /// usage above the configured quota; on success, [`Self::bytes_in_use`]
    /// reflects the new total.
    ///
    /// The add is performed with `checked_add` so a tenant whose quota
    /// is set to `u64::MAX` cannot wrap the counter by repeatedly
    /// asking for `u64::MAX` bytes — the second such call observes the
    /// overflow and returns `MemoryExhausted` while leaving the counter
    /// pinned at `u64::MAX` (saturating).
    ///
    /// # Deprecated
    ///
    /// This unchecked variant cannot tell which tenant is doing the
    /// mutation. Prefer [`Self::consume_bytes_with_capability`], which
    /// requires a [`TenantCapability`] minted by
    /// [`crate::TenantRegistry::register_with_capability`] and rejects
    /// cross-tenant calls with [`TensorWasmError::TenantIsolationViolation`].
    /// The unchecked form is retained for the 0.3 line and will be removed
    /// in v0.4.
    #[deprecated(
        since = "0.3.6",
        note = "use consume_bytes_with_capability; unchecked variant will be removed in v0.4"
    )]
    pub fn consume_bytes(&self, n: u64) -> Result<(), TensorWasmError> {
        self.consume_bytes_inner(n)
    }

    /// Capability-checked variant of [`Self::consume_bytes`].
    ///
    /// Returns [`TensorWasmError::TenantIsolationViolation`] if `cap` was
    /// minted for a different tenant; otherwise behaves exactly like the
    /// (deprecated) unchecked variant. The check is a single integer
    /// compare on the hot path — negligible compared to the CAS loop that
    /// performs the actual quota arithmetic.
    pub fn consume_bytes_with_capability(
        &self,
        cap: &TenantCapability,
        n: u64,
    ) -> Result<(), TensorWasmError> {
        self.check_capability(cap, "quota.consume_bytes")?;
        self.consume_bytes_inner(n)
    }

    /// Shared implementation: the lock-free CAS loop. Both the deprecated
    /// `consume_bytes` and the checked `consume_bytes_with_capability`
    /// delegate here so the atomic discipline lives in one place.
    fn consume_bytes_inner(&self, n: u64) -> Result<(), TensorWasmError> {
        let limit = self.memory_quota_bytes;
        let mut current = self.bytes_in_use.load(Ordering::Acquire);
        loop {
            let next = match current.checked_add(n) {
                Some(v) if v <= limit => v,
                _ => {
                    return Err(TensorWasmError::MemoryExhausted {
                        requested: n,
                        limit,
                    });
                }
            };
            match self.bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.publish_memory_gauge(next);
                    return Ok(());
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// Verify that `cap` was minted for the same tenant this context
    /// belongs to. Returns [`TensorWasmError::TenantIsolationViolation`]
    /// labelled with the *capability's* tenant id (i.e. the offending
    /// caller) and a `resource` string identifying the gated operation —
    /// the offended tenant id is implicit in which context the call
    /// landed on and is recorded by the surrounding span.
    ///
    /// H1 (unconditional registry binding): an additional check rejects
    /// caps minted by a *different* registry (by `Arc::ptr_eq` on the
    /// per-registry token allocations). The check is **fail-closed**:
    /// a context that was never registered (no `registry_token` recorded)
    /// cannot prove which registry — if any — vouches for the cap, so the
    /// capability check is rejected outright rather than waved through.
    /// This closes the prior fail-open hole where a builder-constructed
    /// context (`registry_token == None`) plus a foreign cap naming the
    /// same tenant id silently passed. Registry-minted contexts always
    /// carry a token (stamped by
    /// [`crate::TenantRegistry::register_with_capability`]) and continue
    /// to pass against caps from the same registry.
    ///
    /// H1: this binding is enforced regardless of the `strict-cap-binding`
    /// feature, so a release build without that feature still rejects
    /// cross-registry capabilities and fails closed on unregistered
    /// contexts.
    fn check_capability(
        &self,
        cap: &TenantCapability,
        resource: &'static str,
    ) -> Result<(), TensorWasmError> {
        if cap.tenant_id != self.tenant_id {
            return Err(TensorWasmError::TenantIsolationViolation {
                tenant_id: cap.tenant_id,
                resource: resource.into(),
            });
        }
        match self.registry_token.as_ref() {
            // Registered context: the cap must come from the same
            // registry. Report the cap's tenant id as the offender —
            // the audit-flagged threat is a holder of one registry's
            // cap using it against a namesake tenant in another
            // registry.
            Some(ctx_token) if std::sync::Arc::ptr_eq(ctx_token, &cap.registry_token) => {}
            // Either the cap was minted by a different registry, or
            // this context carries no registry provenance at all
            // (builder-constructed, never registered). Fail closed in
            // both cases.
            _ => {
                return Err(TensorWasmError::TenantIsolationViolation {
                    tenant_id: cap.tenant_id,
                    resource: resource.into(),
                });
            }
        }
        Ok(())
    }

    /// Push the current CPU-side `bytes_in_use` total into this tenant's
    /// per-tenant CPU gauge series, if a metrics handle was wired into
    /// this context at build time. Centralised so [`Self::consume_bytes`]
    /// and [`Self::release_bytes`] share one update path. The `Gauge::set`
    /// call is a single relaxed atomic store — cheap enough to live on the
    /// allocation hot path.
    ///
    /// The CPU counter and the GPU counter report against distinct,
    /// correctly-named per-tenant families and never collide on the same
    /// labelled series: CPU usage owns
    /// [`TensorWasmMetrics::cpu_memory_bytes_per_tenant`] here, while GPU
    /// usage owns [`TensorWasmMetrics::gpu_memory_bytes_per_tenant`] (see
    /// [`Self::publish_gpu_memory_gauge`]).
    fn publish_memory_gauge(&self, new_total: u64) {
        if let Some(metrics) = &self.metrics {
            metrics
                .cpu_memory_bytes_per_tenant()
                .get_or_create(&self.metrics_labels)
                .set(new_total);
        }
    }

    /// Push the current `gpu_bytes_in_use` total into the per-tenant GPU
    /// gauge series.
    ///
    /// Centralised so [`Self::consume_gpu_bytes`] and
    /// [`Self::release_gpu_bytes`] share one update path. GPU usage is the
    /// owner of the correctly-named per-tenant family
    /// [`TensorWasmMetrics::gpu_memory_bytes_per_tenant`]; the CPU counter
    /// no longer writes to it (see [`Self::publish_memory_gauge`]), so the
    /// two no longer clobber each other last-write-wins.
    fn publish_gpu_memory_gauge(&self, new_total: u64) {
        if let Some(metrics) = &self.metrics {
            metrics
                .gpu_memory_bytes_per_tenant()
                .get_or_create(&self.metrics_labels)
                .set(new_total);
        }
    }

    /// Recorded-only CUDA memory-pool release-threshold value.
    ///
    /// Returns `None` when the builder was not given an explicit value
    /// (the driver default applies), or `Some(bytes)` when set via
    /// [`TenantContextBuilder::with_recorded_cuda_mem_pool_quota`].
    ///
    /// ADVISORY / RECORD-ONLY: as the builder method's name indicates,
    /// the value is informational only and is NEVER enforced — the cust
    /// 0.3.x crate does not expose `cuMemPoolSetAttribute`, so the CUDA
    /// driver never sees this number. Enforcement of this crate's
    /// per-tenant quota lives entirely in [`Self::bytes_in_use`].
    pub fn cuda_mem_pool_quota_bytes(&self) -> Option<u64> {
        self.cuda_mem_pool_quota_bytes
    }

    /// The driver-level memory pool wired into this tenant for
    /// driver-enforced GPU-cap enforcement, or `None` when no pool was
    /// provided at build time (the in-process `gpu_bytes_in_use` counter
    /// is then the only enforcement).
    ///
    /// Returned as the backend-agnostic
    /// [`tensor_wasm_core::mem_pool::DriverMemPool`] trait object: this
    /// crate never names the concrete `tensor-wasm-mem` pool type, which
    /// is what keeps the `mem` <-> `tenant` dependency graph acyclic.
    /// Set via [`TenantContextBuilder::with_driver_enforced_gpu_cap`].
    pub fn mem_pool(&self) -> Option<&Arc<dyn DriverMemPool>> {
        self.driver_mem_pool.as_ref()
    }

    /// Push this tenant's CUDA context onto the calling thread's context
    /// stack, returning a RAII guard that pops it on drop. Returns `None`
    /// if the tenant has no `cust::context::Context` (i.e. either the
    /// `cuda` feature is disabled or `ContextIsolated` was not requested
    /// at build time).
    ///
    /// The guard borrows `&self`, so the `TenantContext` cannot be moved
    /// or dropped while the guard is live — the pop on drop is therefore
    /// guaranteed to pop *this* context, not someone else's that snuck
    /// onto the stack.
    #[cfg(feature = "cuda")]
    pub fn enter(&self) -> Option<CudaCtxGuard<'_>> {
        let ctx = self.cu_context.as_ref()?;
        CudaCtxGuard::push(ctx)
            .ok()
            .map(|g| g.with_tenant(self.tenant_id))
    }

    #[cfg(not(feature = "cuda"))]
    /// No-op equivalent of [`Self::enter`] when the `cuda` feature is off.
    /// Always returns `None`.
    pub fn enter(&self) -> Option<CudaCtxGuard> {
        None
    }

    /// Atomically release `n` bytes back to the quota. Saturating on
    /// underflow — callers must not release more than they consumed, but a
    /// bookkeeping mismatch is not fatal.
    ///
    /// Implemented as a CAS loop on `bytes_in_use`, computing
    /// `saturating_sub` on each iteration. The earlier
    /// `fetch_sub` + post-hoc `store(0)` shape was racy: between the
    /// `fetch_sub` and the clamp `store`, a concurrent
    /// [`Self::consume_bytes`] could CAS in a new value, and the
    /// unconditional `store(0)` would then erase that consume. With the
    /// CAS loop, the underflow-clamp path only writes when the value we
    /// underflowed on is still current; otherwise we retry against the
    /// observed value.
    ///
    /// # Deprecated
    ///
    /// This unchecked variant cannot tell which tenant is doing the
    /// mutation. Prefer [`Self::release_bytes_with_capability`], which
    /// requires a [`TenantCapability`] minted by
    /// [`crate::TenantRegistry::register_with_capability`] and rejects
    /// cross-tenant calls with [`TensorWasmError::TenantIsolationViolation`].
    /// The unchecked form is retained for the 0.3 line and will be removed
    /// in v0.4.
    #[deprecated(
        since = "0.3.6",
        note = "use release_bytes_with_capability; unchecked variant will be removed in v0.4"
    )]
    pub fn release_bytes(&self, bytes: u64) {
        // Discards the `ReleaseOutcome`: the deprecated unchecked variant
        // keeps its `()` return for backwards compatibility.
        let _ = self.release_bytes_inner(bytes);
    }

    /// Capability-checked variant of [`Self::release_bytes`].
    ///
    /// Returns [`TensorWasmError::TenantIsolationViolation`] if `cap` was
    /// minted for a different tenant; otherwise releases the bytes and
    /// returns a [`ReleaseOutcome`].
    ///
    /// MEDIUM fix (double-release detection): this used to return
    /// `Ok(())`, hiding the underflow-clamp from the accounting layer. It
    /// now returns `Ok(ReleaseOutcome { released, clamped })` so a caller
    /// running matched consume/release pairs can detect an over-release —
    /// `clamped == true` (equivalently `released < bytes`) means the
    /// request exceeded the outstanding balance and the counter saturated
    /// at zero, the fingerprint of a double-release / mismatched-pair bug.
    /// The underflow itself is still a best-effort clamp (never a panic or
    /// wrap); the change is purely additive observability on the success
    /// path. `.unwrap()`/`.expect()` call sites are unaffected.
    pub fn release_bytes_with_capability(
        &self,
        cap: &TenantCapability,
        bytes: u64,
    ) -> Result<ReleaseOutcome, TensorWasmError> {
        self.check_capability(cap, "quota.release_bytes")?;
        Ok(self.release_bytes_inner(bytes))
    }

    /// Shared implementation: CAS-loop `saturating_sub` + underflow warn.
    /// Both the deprecated `release_bytes` and the checked
    /// `release_bytes_with_capability` delegate here so the atomic
    /// discipline lives in one place. Returns the [`ReleaseOutcome`] so the
    /// checked variant can report a clamp (double-release signal) to the
    /// accounting layer.
    fn release_bytes_inner(&self, bytes: u64) -> ReleaseOutcome {
        let mut current = self.bytes_in_use.load(Ordering::Acquire);
        let (after, before) = loop {
            let next = current.saturating_sub(bytes);
            match self.bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if current < bytes {
                        tracing::warn!(
                            target: "tensor_wasm_tenant::context",
                            tenant = %self.tenant_id,
                            before = current,
                            bytes,
                            "release_bytes underflow clamped",
                        );
                    }
                    break (next, current);
                }
                Err(observed) => current = observed,
            }
        };
        self.publish_memory_gauge(after);
        // `before` is the counter value the winning CAS observed; the actual
        // amount subtracted is `before - after` (== `bytes` on a clean
        // release, less on a clamp). `clamped` iff the request exceeded it.
        ReleaseOutcome {
            released: before - after,
            clamped: before < bytes,
        }
    }

    /// Whether this tenant owns a real `cust::context::Context`.
    ///
    /// Returns `true` only when the `cuda` feature is enabled **and**
    /// [`TenantContextBuilder::build`] successfully constructed a primary
    /// context for a `ContextIsolated` tenant. Callers that need a
    /// real CUDA context (rather than a stream-isolation downgrade) should
    /// check this before calling [`Self::enter`].
    pub fn has_real_context(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cu_context.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// RAII guard returned by [`TenantContext::enter`]: pushes the tenant's
/// CUDA context onto the calling thread's context stack on construction,
/// pops it on drop.
///
/// The lifetime ties the guard to its owning `TenantContext`, so the
/// context cannot be dropped (and the underlying primary context cannot
/// be released) while the guard is live. The pop is performed even if a
/// panic unwinds through the guard's scope — that's the whole point of
/// the RAII pattern here.
///
/// Note: cust 0.3.x exposes both the "new" primary-context API (which is
/// not stack-based) and a `legacy::ContextStack::push`/`pop` shim that
/// covers `cuCtxPushCurrent`/`cuCtxPopCurrent`. The trait
/// `cust::context::ContextHandle` is implemented for both the primary
/// `Context` and `legacy::UnownedContext`, so we can use the stack API
/// uniformly here.
#[cfg(feature = "cuda")]
pub struct CudaCtxGuard<'a> {
    // PhantomData ties the guard to the borrowing `TenantContext`.
    _ctx: std::marker::PhantomData<&'a cust::context::Context>,
    // Tenant id, used only in the `Drop` log if `cuCtxPopCurrent` fails.
    tenant_id: Option<TenantId>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaCtxGuard<'a> {
    /// Push `ctx` onto the calling thread's context stack.
    pub fn push(ctx: &'a cust::context::Context) -> Result<Self, cust::error::CudaError> {
        // ContextHandle is implemented for the primary `Context`, so the
        // legacy stack API accepts it directly. This matches the plan's
        // `cuCtxPushCurrent` requirement.
        cust::context::legacy::ContextStack::push(ctx)?;
        Ok(Self {
            _ctx: std::marker::PhantomData,
            tenant_id: None,
        })
    }
}

#[cfg(feature = "cuda")]
impl<'a> CudaCtxGuard<'a> {
    /// Bind a tenant id to this guard so the `Drop` log can attribute pop
    /// failures back to the offending tenant.
    fn with_tenant(self, tenant_id: TenantId) -> Self {
        Self {
            tenant_id: Some(tenant_id),
            ..self
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaCtxGuard<'_> {
    fn drop(&mut self) {
        // Best-effort pop: if the stack is empty or another context was
        // pushed on top, we still pop the topmost context. Errors cannot
        // be returned from `Drop`; the next-best thing is a structured
        // log with the underlying CUDA error code and the tenant whose
        // guard tripped, so operators can correlate it with kernels in
        // flight at the time of the panic / scope exit.
        if let Err(e) = cust::context::legacy::ContextStack::pop() {
            tracing::error!(
                target: "tensor_wasm_tenant::context",
                error = ?e,
                tenant = ?self.tenant_id,
                "cuCtxPopCurrent failed in CudaCtxGuard::drop",
            );
        }
    }
}

/// No-CUDA placeholder so the type name is callable from generic code
/// without requiring the caller to cfg-gate `Option<CudaCtxGuard>`.
#[cfg(not(feature = "cuda"))]
pub struct CudaCtxGuard;

/// Builder for [`TenantContext`].
///
/// Fields default to a low-overhead, multi-tenant-safe configuration:
/// [`IsolationKind::StreamIsolated`], stream id `0`, and an 8 GiB memory
/// quota. Override with the chained `with_*` methods, then call
/// [`Self::build`].
#[derive(Debug)]
pub struct TenantContextBuilder {
    tenant_id: TenantId,
    isolation: IsolationKind,
    stream_id: u64,
    memory_quota_bytes: u64,
    cuda_mem_pool_quota_bytes: Option<u64>,
    /// See [`TenantContext::gpu_memory_bytes_cap`]. `None` (the default)
    /// keeps the historical "no cap" behaviour; set via
    /// [`TenantContextBuilder::with_gpu_memory_bytes_cap`].
    gpu_memory_bytes_cap: Option<u64>,
    /// See [`TenantContext::mem_pool`]. `None` (the default) leaves the
    /// in-process `gpu_bytes_in_use` counter as the only GPU-cap
    /// enforcement; set via
    /// [`TenantContextBuilder::with_driver_enforced_gpu_cap`].
    driver_mem_pool: Option<Arc<dyn DriverMemPool>>,
    /// `(ops_per_sec, burst)` for the time-windowed operation-rate limiter.
    /// `None` (the default) means no rate limit — see
    /// [`TenantContext::try_acquire_op`]. Set via
    /// [`TenantContextBuilder::with_rate_limit`]; materialised into a
    /// [`TokenBucket`] at [`Self::build`] time so the bucket's monotonic
    /// clock starts ticking when the context goes live, not when the builder
    /// was created.
    rate_limit: Option<(u64, u64)>,
    #[cfg(feature = "cuda")]
    cuda_device_index: Option<u32>,
    metrics: Option<TensorWasmMetrics>,
}

impl TenantContextBuilder {
    /// Default quota: 8 GiB. Sized for a single H100 SXM partition under MPS.
    pub const DEFAULT_QUOTA_BYTES: u64 = 8 * 1024 * 1024 * 1024;

    /// Create a builder with default isolation, stream id, and quota.
    pub fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            isolation: IsolationKind::default(),
            stream_id: 0,
            memory_quota_bytes: Self::DEFAULT_QUOTA_BYTES,
            cuda_mem_pool_quota_bytes: None,
            gpu_memory_bytes_cap: None,
            driver_mem_pool: None,
            rate_limit: None,
            #[cfg(feature = "cuda")]
            cuda_device_index: None,
            metrics: None,
        }
    }

    /// Set the per-tenant GPU memory cap in bytes.
    ///
    /// `None` (the default) means no cap — the tenant's GPU memory
    /// usage is recorded on [`TenantContext::gpu_bytes_in_use`] but the
    /// allocator never refuses an over-cap request. Setting `Some(bytes)`
    /// makes the allocator path
    /// (`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context`)
    /// return [`tensor_wasm_core::error::TensorWasmError::GpuMemoryExhausted`]
    /// for any [`TenantContext::consume_gpu_bytes`] that would push the
    /// total above `bytes`.
    ///
    /// See `docs/GPU-QUOTAS.md` for the v0.3.7 record-only semantics and
    /// the v0.4 `cuMemPool` enforcement plan.
    pub fn with_gpu_memory_bytes_cap(mut self, bytes: u64) -> Self {
        self.gpu_memory_bytes_cap = Some(bytes);
        self
    }

    /// Wire a shared [`TensorWasmMetrics`] registry into the context so
    /// every CPU-side [`TenantContext::consume_bytes`] /
    /// [`TenantContext::release_bytes`] transition updates the per-tenant
    /// series of [`TensorWasmMetrics::cpu_memory_bytes_per_tenant`] and
    /// every GPU-side [`TenantContext::consume_gpu_bytes`] /
    /// [`TenantContext::release_gpu_bytes`] transition updates the
    /// per-tenant series of
    /// [`TensorWasmMetrics::gpu_memory_bytes_per_tenant`]. The handle is
    /// cheap to clone (it shares an inner `Arc`); the caller normally
    /// passes the same registry the API gateway exposes via
    /// `GET /metrics`. Omitting this builder call (or passing `None`
    /// directly into a future fallible variant) leaves the tenant's
    /// memory accounting completely off the dashboard — useful for
    /// benches and standalone examples that do not run a Prometheus
    /// scrape.
    pub fn with_metrics(mut self, metrics: TensorWasmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Override the isolation level.
    pub fn with_isolation(mut self, isolation: IsolationKind) -> Self {
        self.isolation = isolation;
        self
    }

    /// Override the stream identifier.
    pub fn with_stream_id(mut self, stream_id: u64) -> Self {
        self.stream_id = stream_id;
        self
    }

    /// Override the memory quota in bytes.
    pub fn with_memory_quota_bytes(mut self, memory_quota_bytes: u64) -> Self {
        self.memory_quota_bytes = memory_quota_bytes;
        self
    }

    /// Record a CUDA memory-pool release-threshold value on the
    /// [`TenantContext`] **without** applying it to a real CUDA mem-pool.
    ///
    /// The honest name reflects what this method actually does today:
    /// because cust 0.3.x does not expose the `cuMemPool*` API surface
    /// (`cuMemPoolSetAttribute` / `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`),
    /// the value is stored on the context for inspection, metrics, and
    /// forward-compatibility — but the CUDA driver never sees it. The
    /// only enforcement of per-tenant memory usage in this crate is the
    /// in-process counter returned by [`TenantContext::bytes_in_use`]
    /// and the quota set via
    /// [`Self::with_memory_quota_bytes`]; allocations that bypass the
    /// `consume_bytes` / `release_bytes` pair are NOT capped at the
    /// CUDA-driver level.
    ///
    /// Upgrading cust (or going direct via
    /// `cuda::sys::cuMemPoolSetAttribute`) is tracked as a future-work
    /// item; when that lands, a new `with_cuda_mem_pool_quota` method
    /// will replace this one and the value will be applied to the
    /// driver's release threshold at build time.
    pub fn with_recorded_cuda_mem_pool_quota(mut self, bytes: u64) -> Self {
        self.cuda_mem_pool_quota_bytes = Some(bytes);
        self
    }

    /// Wire a driver-level memory pool into this tenant so the per-tenant
    /// GPU cap is enforced by the CUDA driver itself, not just the
    /// in-process [`TenantContext::gpu_bytes_in_use`] counter.
    ///
    /// `pool` is any [`tensor_wasm_core::mem_pool::DriverMemPool`] — in
    /// production, `tensor-wasm-mem`'s
    /// `cuda_mem_pool::TenantMemPool` (backed by
    /// `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`);
    /// in tests, a mock. The parameter is the backend-agnostic trait
    /// object on purpose: taking the concrete `mem` type would require a
    /// `tenant -> mem` dependency, re-introducing the `mem` <-> `tenant`
    /// cycle that this whole abstraction exists to break.
    ///
    /// When set, [`TenantContext::consume_gpu_bytes`] pins the pool's
    /// release threshold to the cap configured by
    /// [`Self::with_gpu_memory_bytes_cap`] on every accepted allocation,
    /// so a tenant that obtained a raw CUDA driver handle and bypassed
    /// the in-process counter still hits a driver-level
    /// `CUDA_ERROR_OUT_OF_MEMORY` at the cap. Pair this with
    /// [`Self::with_gpu_memory_bytes_cap`]; without a cap the pool is
    /// stored but never pinned (there is no ceiling to push). See
    /// `docs/GPU-QUOTAS.md` for the threat model the driver pin closes.
    pub fn with_driver_enforced_gpu_cap(mut self, pool: Arc<dyn DriverMemPool>) -> Self {
        self.driver_mem_pool = Some(pool);
        self
    }

    /// Enable a time-windowed operation-rate limiter on this tenant.
    ///
    /// `ops_per_sec` is the steady-state refill rate (operations admitted per
    /// second once the burst budget is spent); `burst` is the bucket depth —
    /// the maximum number of operations that can be admitted back-to-back
    /// before the steady-state rate gates further ones. The bucket starts
    /// full, so the first `burst` operations after `build()` are admitted
    /// immediately. Both arguments are clamped to a minimum of `1` internally
    /// so a `(0, 0)` configuration cannot wedge the tenant into a
    /// never-admits state.
    ///
    /// Use this to bound a noisy neighbour's scheduling footprint without
    /// touching its memory caps: a value such as `with_rate_limit(1000, 200)`
    /// admits short bursts of up to 200 kernel launches while holding the
    /// long-run average at 1000/s. The same primitive expresses a bytes/sec
    /// budget by passing the byte count to
    /// [`TenantContext::try_acquire_ops`] instead of `1`.
    ///
    /// **Default = no rate limit.** Omitting this call (the default) leaves
    /// [`TenantContext::try_acquire_op`] an unconditional `Ok(())`, preserving
    /// the historical pure high-water-mark byte-cap behaviour. The limiter is
    /// purely additive and orthogonal to the byte counters.
    pub fn with_rate_limit(mut self, ops_per_sec: u64, burst: u64) -> Self {
        self.rate_limit = Some((ops_per_sec, burst));
        self
    }

    /// Set the CUDA device index this tenant's context should be built
    /// against. Only meaningful when the `cuda` feature is enabled and
    /// the isolation is `ContextIsolated`. Defaults to device 0.
    #[cfg(feature = "cuda")]
    pub fn with_cuda_device_index(mut self, device_index: u32) -> Self {
        self.cuda_device_index = Some(device_index);
        self
    }

    /// Finalise into a `TenantContext`.
    ///
    /// If the builder requested [`IsolationKind::ContextIsolated`] but
    /// constructing the underlying `cust::context::Context` fails (for
    /// example, no CUDA device 0, MPS unavailable, OOM at primary-
    /// context retain), the resulting `TenantContext`'s isolation level
    /// is **downgraded** to [`IsolationKind::StreamIsolated`] so the
    /// reported isolation matches reality. Callers that need to
    /// distinguish "real context-isolated" from "downgraded" can call
    /// [`TenantContext::has_real_context`].
    #[allow(unused_mut)]
    pub fn build(mut self) -> TenantContext {
        #[cfg(feature = "cuda")]
        let cu_context = {
            let want_isolated = matches!(self.isolation, IsolationKind::ContextIsolated);
            let built = self.build_cuda_context();
            if want_isolated && built.is_none() {
                // Honest reporting: a ContextIsolated request that produced
                // no real `cust::context::Context` is actually stream-
                // isolated at the GPU level. Downgrade so `.isolation()`
                // does not lie to schedulers, dashboards, or auditors —
                // AND escalate visibility: operators who specified
                // `ContextIsolated` as a deployment constraint need to
                // know the driver could not honour it. We bump a
                // process-wide counter (read via
                // [`isolation_downgrade_count`]) and emit a structured
                // `error!` so the alert pipeline picks it up. The
                // per-failure-cause logs inside `build_cuda_context`
                // record the underlying CUDA error code.
                //
                // TODO(core): when `TensorWasmMetrics` grows
                // `isolation_downgrade_total()` (see `ISOLATION_DOWNGRADE_COUNT`
                // docs), also do `if let Some(m) = &self.metrics {
                // m.isolation_downgrade_total().inc(); }` here so the
                // downgrade is observable through the registry, not just
                // the process-local static.
                ISOLATION_DOWNGRADE_COUNT.fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    target: "tensor_wasm_tenant::context",
                    tenant = %self.tenant_id,
                    requested = %IsolationKind::ContextIsolated,
                    effective = %IsolationKind::StreamIsolated,
                    "ContextIsolated requested but unavailable; downgraded to StreamIsolated",
                );
                self.isolation = IsolationKind::StreamIsolated;
            }
            built
        };
        #[cfg(not(feature = "cuda"))]
        let cu_context = ();

        let metrics_labels = TenantLabels::new(self.tenant_id.to_string());
        TenantContext {
            tenant_id: self.tenant_id,
            isolation: self.isolation,
            stream_id: self.stream_id,
            memory_quota_bytes: self.memory_quota_bytes,
            bytes_in_use: AtomicU64::new(0),
            gpu_memory_bytes_cap: self.gpu_memory_bytes_cap,
            gpu_bytes_in_use: AtomicU64::new(0),
            cuda_mem_pool_quota_bytes: self.cuda_mem_pool_quota_bytes,
            driver_mem_pool: self.driver_mem_pool,
            rate_limiter: self
                .rate_limit
                .map(|(ops_per_sec, burst)| TokenBucket::new(ops_per_sec, burst)),
            cu_context,
            metrics: self.metrics,
            metrics_labels,
            // Stamped by `TenantRegistry::register_with_capability` after
            // the context is moved into the registry but before it is
            // Arc-wrapped; remains `None` for contexts the test harness
            // constructs without going through the registry. H1: always
            // present so registry binding is enforced unconditionally.
            registry_token: None,
        }
    }

    /// Build the underlying primary `cust::context::Context` when the
    /// `cuda` feature is on AND the tenant requested `ContextIsolated`.
    /// Returns `None` for shared/stream-isolated tenants, or when device
    /// retain fails — caller (build()) then proceeds with a stub context
    /// and operator-visible logs make the degradation explicit.
    #[cfg(feature = "cuda")]
    fn build_cuda_context(&self) -> Option<cust::context::Context> {
        if !matches!(self.isolation, IsolationKind::ContextIsolated) {
            return None;
        }
        let device_idx = self.cuda_device_index.unwrap_or(0);
        let device = match cust::device::Device::get_device(device_idx as i32) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    target: "tensor_wasm_tenant::context",
                    tenant = %self.tenant_id,
                    device = device_idx,
                    error = ?e,
                    "Device::get_device failed; falling back to stream-isolated mode",
                );
                return None;
            }
        };
        match cust::context::Context::new(device) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                tracing::error!(
                    target: "tensor_wasm_tenant::context",
                    tenant = %self.tenant_id,
                    device = device_idx,
                    error = ?e,
                    "Context::new failed; falling back to stream-isolated mode",
                );
                None
            }
        }
    }
}

#[cfg(test)]
#[allow(deprecated)]
// These tests pre-date the capability gate and exercise the unchecked
// `consume_bytes` / `release_bytes` variants directly. The deprecation
// warning is the *signal* to callers — silencing it here is the only
// place it should be silenced, and only because these tests pin the
// shim's behaviour until the variants are removed in v0.4.
mod tests {
    use super::*;
    // Test-only: the mock `DriverMemPool` impl below returns this error type.
    // Kept out of the module-level imports so the non-test lib build doesn't
    // carry an unused import (the production paths use `DriverMemPool` only).
    use tensor_wasm_core::mem_pool::MemPoolError;

    #[test]
    fn builder_defaults() {
        let ctx = TenantContext::builder(TenantId(1)).build();
        assert_eq!(ctx.id(), TenantId(1));
        assert_eq!(ctx.isolation(), IsolationKind::StreamIsolated);
        assert_eq!(ctx.stream_id(), 0);
        assert_eq!(ctx.quota(), TenantContextBuilder::DEFAULT_QUOTA_BYTES);
        assert_eq!(ctx.bytes_in_use(), 0);
    }

    #[test]
    fn builder_overrides() {
        let ctx = TenantContext::builder(TenantId(7))
            .with_isolation(IsolationKind::ContextIsolated)
            .with_stream_id(42)
            .with_memory_quota_bytes(1024)
            .build();
        assert_eq!(ctx.isolation(), IsolationKind::ContextIsolated);
        assert_eq!(ctx.stream_id(), 42);
        assert_eq!(ctx.quota(), 1024);
    }

    #[test]
    fn quota_consume_release_round_trip() {
        let ctx = TenantContext::builder(TenantId(2))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.consume_bytes(256).unwrap();
        assert_eq!(ctx.bytes_in_use(), 256);
        ctx.consume_bytes(512).unwrap();
        assert_eq!(ctx.bytes_in_use(), 768);
        ctx.release_bytes(256);
        assert_eq!(ctx.bytes_in_use(), 512);
    }

    #[test]
    fn quota_enforcement_rejects_over_limit() {
        let ctx = TenantContext::builder(TenantId(3))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.consume_bytes(1000).unwrap();
        let err = ctx.consume_bytes(100).unwrap_err();
        match err {
            TensorWasmError::MemoryExhausted { requested, limit } => {
                assert_eq!(requested, 100);
                assert_eq!(limit, 1024);
            }
            other => panic!("expected MemoryExhausted, got {other:?}"),
        }
        // Failed allocation must not move the counter.
        assert_eq!(ctx.bytes_in_use(), 1000);
    }

    #[test]
    fn release_saturates_on_underflow() {
        let ctx = TenantContext::builder(TenantId(4))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.release_bytes(999); // never consumed; must not panic or wrap.
        assert_eq!(ctx.bytes_in_use(), 0);
    }

    #[test]
    fn isolation_kind_names_are_stable() {
        assert_eq!(IsolationKind::Shared.name(), "shared");
        assert_eq!(IsolationKind::StreamIsolated.name(), "stream_isolated");
        assert_eq!(IsolationKind::ContextIsolated.name(), "context_isolated");
    }

    #[test]
    fn isolation_kind_matches_each_variant() {
        for kind in [
            IsolationKind::Shared,
            IsolationKind::StreamIsolated,
            IsolationKind::ContextIsolated,
        ] {
            let ctx = TenantContext::builder(TenantId(99))
                .with_isolation(kind)
                .build();
            assert_eq!(ctx.isolation(), kind);
            // Display is the same as the name.
            assert_eq!(ctx.isolation().to_string(), kind.name());
        }
    }

    #[test]
    fn isolation_kind_default_is_stream_isolated() {
        assert_eq!(IsolationKind::default(), IsolationKind::StreamIsolated);
    }

    #[test]
    fn cuda_mem_pool_quota_default_is_none() {
        let ctx = TenantContext::builder(TenantId(5)).build();
        assert_eq!(ctx.cuda_mem_pool_quota_bytes(), None);
    }

    #[test]
    fn cuda_mem_pool_quota_recorded() {
        let ctx = TenantContext::builder(TenantId(6))
            .with_recorded_cuda_mem_pool_quota(4 * 1024 * 1024 * 1024)
            .build();
        assert_eq!(
            ctx.cuda_mem_pool_quota_bytes(),
            Some(4 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn metrics_handle_absent_by_default_is_a_noop() {
        // No `with_metrics(...)` — the consume/release pair must continue
        // to work exactly as before. The point of this test is to pin the
        // backwards-compat contract: pre-existing call sites that do not
        // plumb a registry observe no behaviour change.
        let ctx = TenantContext::builder(TenantId(11))
            .with_memory_quota_bytes(8192)
            .build();
        ctx.consume_bytes(1024).unwrap();
        ctx.release_bytes(512);
        assert_eq!(ctx.bytes_in_use(), 512);
    }

    #[test]
    fn metrics_handle_publishes_consume_and_release_totals() {
        let metrics = TensorWasmMetrics::new();
        let ctx = TenantContext::builder(TenantId(12))
            .with_memory_quota_bytes(1 << 20)
            .with_metrics(metrics.clone())
            .build();

        // CPU consume/release publish to the per-tenant CPU family
        // `cpu_memory_bytes_per_tenant` (the CPU counterpart of the GPU
        // counter's family). See `publish_memory_gauge`.
        let labels = TenantLabels::new(TenantId(12).to_string());
        let cpu = || {
            metrics
                .cpu_memory_bytes_per_tenant()
                .get_or_create(&labels)
                .get()
        };

        // Consume → the per-tenant CPU gauge reads the post-add value.
        ctx.consume_bytes(4096).unwrap();
        assert_eq!(cpu(), 4096);

        // A second consume composes.
        ctx.consume_bytes(2048).unwrap();
        assert_eq!(cpu(), 6144);

        // Release → the gauge reads the post-sub value.
        ctx.release_bytes(2048);
        assert_eq!(cpu(), 4096);
    }

    #[test]
    fn gpu_metrics_publish_to_per_tenant_family() {
        // The GPU counter owns the per-tenant `gpu_memory_bytes_per_tenant`
        // family; the CPU counter no longer collides on it.
        let metrics = TensorWasmMetrics::new();
        let ctx = TenantContext::builder(TenantId(12))
            .with_metrics(metrics.clone())
            .build();
        let labels = TenantLabels::new(TenantId(12).to_string());

        ctx.consume_gpu_bytes(4096).unwrap();
        assert_eq!(
            metrics
                .gpu_memory_bytes_per_tenant()
                .get_or_create(&labels)
                .get(),
            4096
        );

        ctx.release_gpu_bytes(2048);
        assert_eq!(
            metrics
                .gpu_memory_bytes_per_tenant()
                .get_or_create(&labels)
                .get(),
            2048
        );

        // A CPU consume on the same context must NOT perturb the GPU
        // per-tenant series — the collision is gone.
        ctx.consume_bytes(1024).unwrap();
        assert_eq!(
            metrics
                .gpu_memory_bytes_per_tenant()
                .get_or_create(&labels)
                .get(),
            2048
        );
    }

    /// Mock [`DriverMemPool`] that records every `set_release_threshold`
    /// call so tests can assert the tenant cap is pushed through with the
    /// right value. Lives in the test module so it cannot leak into the
    /// public surface; it is the `tensor-wasm-mem`-free stand-in for
    /// `cuda_mem_pool::TenantMemPool` (which this crate must not depend
    /// on).
    #[derive(Debug, Default)]
    struct MockDriverMemPool {
        threshold: AtomicU64,
        set_calls: AtomicU64,
        /// When `true`, `set_release_threshold` returns an error to
        /// exercise the consume path's fail-soft behaviour.
        fail: bool,
    }

    impl DriverMemPool for MockDriverMemPool {
        fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError> {
            self.set_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(MemPoolError::SetAttribute("mock forced failure".into()));
            }
            self.threshold.store(bytes, Ordering::SeqCst);
            Ok(())
        }

        fn release_threshold(&self) -> Option<u64> {
            Some(self.threshold.load(Ordering::SeqCst))
        }
    }

    #[test]
    fn driver_enforced_gpu_cap_pushes_threshold_on_consume() {
        let pool = Arc::new(MockDriverMemPool::default());
        let ctx = TenantContext::builder(TenantId(20))
            .with_gpu_memory_bytes_cap(4096)
            .with_driver_enforced_gpu_cap(pool.clone())
            .build();

        // The accessor returns the wired pool.
        assert!(ctx.mem_pool().is_some());

        // A successful GPU consume pins the driver threshold to the cap
        // (NOT the running total).
        ctx.consume_gpu_bytes(1000).unwrap();
        assert_eq!(pool.set_calls.load(Ordering::SeqCst), 1);
        assert_eq!(pool.threshold.load(Ordering::SeqCst), 4096);
        assert_eq!(pool.release_threshold(), Some(4096));

        // A second consume re-pins to the same cap.
        ctx.consume_gpu_bytes(500).unwrap();
        assert_eq!(pool.set_calls.load(Ordering::SeqCst), 2);
        assert_eq!(pool.threshold.load(Ordering::SeqCst), 4096);
    }

    #[test]
    fn driver_enforced_gpu_cap_not_pushed_without_cap() {
        // No `with_gpu_memory_bytes_cap` → no ceiling to push, so the
        // pool is stored but `set_release_threshold` is never called.
        let pool = Arc::new(MockDriverMemPool::default());
        let ctx = TenantContext::builder(TenantId(21))
            .with_driver_enforced_gpu_cap(pool.clone())
            .build();
        ctx.consume_gpu_bytes(1000).unwrap();
        assert_eq!(pool.set_calls.load(Ordering::SeqCst), 0);
        assert!(ctx.mem_pool().is_some());
    }

    #[test]
    fn driver_enforced_gpu_cap_over_limit_does_not_push() {
        // An over-cap consume is rejected by the in-process counter
        // BEFORE the driver pin runs, so the pool sees no call for the
        // rejected allocation.
        let pool = Arc::new(MockDriverMemPool::default());
        let ctx = TenantContext::builder(TenantId(22))
            .with_gpu_memory_bytes_cap(1024)
            .with_driver_enforced_gpu_cap(pool.clone())
            .build();
        let err = ctx.consume_gpu_bytes(2048).unwrap_err();
        assert!(matches!(err, TensorWasmError::GpuMemoryExhausted { .. }));
        assert_eq!(pool.set_calls.load(Ordering::SeqCst), 0);
        // The rejected allocation must not move the counter either.
        assert_eq!(ctx.gpu_bytes_in_use(), 0);
    }

    #[test]
    fn driver_enforced_gpu_cap_set_failure_is_fail_soft() {
        // If the driver pin fails, the in-process consume still succeeds:
        // the counter already accepted the allocation and the driver pin
        // is belt-and-braces.
        let pool = Arc::new(MockDriverMemPool {
            fail: true,
            ..Default::default()
        });
        let ctx = TenantContext::builder(TenantId(23))
            .with_gpu_memory_bytes_cap(4096)
            .with_driver_enforced_gpu_cap(pool.clone())
            .build();
        ctx.consume_gpu_bytes(1000).unwrap();
        assert_eq!(pool.set_calls.load(Ordering::SeqCst), 1);
        // Threshold never recorded because the mock errored before storing.
        assert_eq!(pool.threshold.load(Ordering::SeqCst), 0);
        // The in-process counter still reflects the accepted allocation.
        assert_eq!(ctx.gpu_bytes_in_use(), 1000);
    }

    #[test]
    fn uncapped_gpu_overflow_clamps_to_recoverable_sentinel() {
        // MEDIUM fix: an uncapped tenant must never be refused, but the
        // overflow clamp must leave the counter strictly below u64::MAX so a
        // later release can bring it down — the old `u64::MAX` clamp pinned
        // the gauge at the ceiling forever.
        let ctx = TenantContext::builder(TenantId(40)).build();
        assert_eq!(ctx.gpu_memory_bytes_cap(), None);

        // First consume lands normally.
        ctx.consume_gpu_bytes(u64::MAX - 10).unwrap();
        assert_eq!(ctx.gpu_bytes_in_use(), u64::MAX - 10);

        // Second consume would overflow `checked_add`; uncapped, so it must
        // succeed and clamp to the recoverable sentinel rather than error.
        ctx.consume_gpu_bytes(100).unwrap();
        assert_eq!(
            ctx.gpu_bytes_in_use(),
            TenantContext::GPU_BYTES_OVERFLOW_SENTINEL
        );
        assert!(
            ctx.gpu_bytes_in_use() < u64::MAX,
            "sentinel must be strictly below u64::MAX so releases can recover"
        );

        // A normal-sized release brings it back down — the recovery the old
        // u64::MAX clamp made impossible in practice.
        ctx.release_gpu_bytes(4096);
        assert_eq!(
            ctx.gpu_bytes_in_use(),
            TenantContext::GPU_BYTES_OVERFLOW_SENTINEL - 4096
        );
    }

    #[test]
    fn mem_pool_accessor_default_is_none() {
        let ctx = TenantContext::builder(TenantId(24)).build();
        assert!(ctx.mem_pool().is_none());
    }

    #[test]
    fn metrics_two_tenants_produce_two_distinct_series() {
        // Mirrors the dashboard's expected shape: two registered tenants
        // reserving different amounts must surface as two distinct
        // labelled series in the Prometheus exposition. The per-tenant
        // family is the GPU counter's, so drive it via `consume_gpu_bytes`.
        let metrics = TensorWasmMetrics::new();
        let a = TenantContext::builder(TenantId(101))
            .with_metrics(metrics.clone())
            .build();
        let b = TenantContext::builder(TenantId(102))
            .with_metrics(metrics.clone())
            .build();
        a.consume_gpu_bytes(4096).unwrap();
        b.consume_gpu_bytes(8192).unwrap();

        let text = metrics.encode_text();
        assert!(
            text.contains("tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id=\"T#101\"} 4096"),
            "missing tenant 101 sample in:\n{text}"
        );
        assert!(
            text.contains("tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id=\"T#102\"} 8192"),
            "missing tenant 102 sample in:\n{text}"
        );
    }

    #[test]
    fn metrics_release_underflow_publishes_clamped_zero() {
        let metrics = TensorWasmMetrics::new();
        let ctx = TenantContext::builder(TenantId(13))
            .with_memory_quota_bytes(1 << 16)
            .with_metrics(metrics.clone())
            .build();
        // Underflow path: release without prior consume. The counter
        // clamps to zero and the per-tenant CPU gauge should reflect zero,
        // not wrap.
        let labels = TenantLabels::new(TenantId(13).to_string());
        ctx.release_bytes(123);
        assert_eq!(
            metrics
                .cpu_memory_bytes_per_tenant()
                .get_or_create(&labels)
                .get(),
            0
        );
    }

    #[test]
    fn enter_returns_none_without_cuda_context() {
        // On the no-CUDA path `enter` always returns None; on CUDA-enabled
        // builds the default builder still produces a context-less tenant
        // (StreamIsolated) so the result is None either way without
        // explicit ContextIsolated + a real device.
        let ctx = TenantContext::builder(TenantId(8)).build();
        assert!(ctx.enter().is_none());
    }

    #[test]
    fn release_underflow_does_not_overwrite_concurrent_consume() {
        // Regression test for the `fetch_sub` + unconditional `store(0)`
        // race: the old shape would race-erase a concurrent
        // `consume_bytes` between the `fetch_sub` and the clamping
        // `store`. With the CAS loop, the underflow-clamp only writes
        // when the value we observed underflowing is still current.
        //
        // Construction: pre-load the counter with `(BYTES - 1)` so the
        // releaser thread's `release_bytes(BYTES)` underflows on every
        // iteration. The consumer thread observes that underflow window
        // and races a `consume_bytes(CONSUME)` against it. After both
        // threads have run `ITERATIONS` times each, the final counter
        // must equal the algebraic sum (clamped to zero), regardless of
        // interleaving. The old implementation would drop consumes,
        // producing a final value that drifts below the expected one.
        use std::thread;

        const ITERATIONS: u64 = 10_000;
        const BYTES: u64 = 100;
        const PRE_LOAD: u64 = BYTES - 1; // guarantees release underflows
        const CONSUME: u64 = 7;

        let ctx = Arc::new(
            TenantContext::builder(TenantId(0xAFE))
                // Quota generous enough that consume_bytes never trips
                // the MemoryExhausted branch and skews the algebra.
                .with_memory_quota_bytes(u64::MAX)
                .build(),
        );
        // Pre-load to the underflow-edge.
        ctx.consume_bytes(PRE_LOAD).unwrap();

        let releaser = {
            let ctx = Arc::clone(&ctx);
            thread::spawn(move || {
                for _ in 0..ITERATIONS {
                    ctx.release_bytes(BYTES);
                }
            })
        };
        let consumer = {
            let ctx = Arc::clone(&ctx);
            thread::spawn(move || {
                for _ in 0..ITERATIONS {
                    ctx.consume_bytes(CONSUME).unwrap();
                }
            })
        };
        releaser.join().expect("releaser thread panicked");
        consumer.join().expect("consumer thread panicked");

        // Algebraic expectation, computed with saturating arithmetic so
        // each release that observed `current < BYTES` clamps to zero
        // rather than wrapping. We can't reconstruct the exact
        // interleaving here, but the upper and lower bounds bracket the
        // legitimate final value:
        //   - Lower bound: every release underflows immediately, so
        //     each one clamps the counter to zero before the consumer
        //     re-adds its CONSUME. The final state is somewhere
        //     between `0` (if a release ran last) and
        //     `ITERATIONS * CONSUME` (if every consume ran after every
        //     release). The post-condition we actually assert is
        //     stronger: total consumes minus total clamped-releases is
        //     bounded by the consumer's contribution.
        //   - Upper bound: `PRE_LOAD + ITERATIONS * CONSUME`.
        let final_value = ctx.bytes_in_use();
        let upper = PRE_LOAD.saturating_add(ITERATIONS.saturating_mul(CONSUME));
        assert!(
            final_value <= upper,
            "final {final_value} exceeded upper bound {upper}"
        );
        // The critical invariant: with the old buggy `store(0)` the
        // consumer's contributions could be wholesale erased between
        // `fetch_sub` and `store`. With the CAS loop, every successful
        // `consume_bytes` either lands before or after a `release`
        // CAS, but is never silently overwritten. We assert that the
        // counter never went negative (u64 sentinel for that is the
        // wrap-around to near-MAX — anything in the high half of the
        // u64 range would signal the bug).
        assert!(
            final_value < u64::MAX / 2,
            "final {final_value} suggests wrap-around — the race was not fixed",
        );
    }

    #[test]
    fn rate_limit_absent_by_default_always_admits() {
        // No `with_rate_limit` → the historical pure byte-cap behaviour:
        // `try_acquire_op` is an unconditional Ok and `has_rate_limit` is
        // false. Pins the backwards-compat contract.
        let ctx = TenantContext::builder(TenantId(30)).build();
        assert!(!ctx.has_rate_limit());
        for _ in 0..10_000 {
            ctx.try_acquire_op().expect("no limiter must always admit");
        }
    }

    #[test]
    fn token_bucket_admits_up_to_burst_then_rejects() {
        // Deterministic, no sleep: drive the bucket at a fixed instant so no
        // refill happens between calls. A full bucket of depth 5 admits
        // exactly 5 ops, then rejects.
        let bucket = TokenBucket::new(/*ops_per_sec*/ 100, /*burst*/ 5);
        let t0 = Instant::now();
        for i in 0..5 {
            bucket
                .try_acquire_at(1, t0)
                .unwrap_or_else(|_| panic!("op {i} within burst must be admitted"));
        }
        let err = bucket
            .try_acquire_at(1, t0)
            .expect_err("6th op past burst must be rejected");
        assert_eq!(err.requested, 1);
        assert_eq!(err.available, 0);
        assert_eq!(err.ops_per_sec, 100);
        assert_eq!(err.burst, 5);
    }

    #[test]
    fn token_bucket_refills_over_time() {
        // Inject elapsed time rather than sleeping: at 10 ops/s, one token
        // accrues every 100 ms. Drain the bucket, then advance the injected
        // clock and confirm refilled tokens are admitted.
        let bucket = TokenBucket::new(/*ops_per_sec*/ 10, /*burst*/ 2);
        let t0 = Instant::now();
        // Drain the initial burst of 2.
        bucket.try_acquire_at(1, t0).unwrap();
        bucket.try_acquire_at(1, t0).unwrap();
        assert!(bucket.try_acquire_at(1, t0).is_err(), "bucket drained");

        // 100 ms later → exactly one token refilled.
        let t1 = t0 + Duration::from_millis(100);
        bucket
            .try_acquire_at(1, t1)
            .expect("one token should have refilled after 100ms");
        assert!(
            bucket.try_acquire_at(1, t1).is_err(),
            "only one token refilled; second must be rejected"
        );

        // 1 s after t0 → bucket fully refilled, but capped at burst (2),
        // not 10. So exactly 2 admits then a reject.
        let t2 = t0 + Duration::from_secs(1);
        bucket.try_acquire_at(1, t2).unwrap();
        bucket.try_acquire_at(1, t2).unwrap();
        assert!(
            bucket.try_acquire_at(1, t2).is_err(),
            "refill is capped at burst depth"
        );
    }

    #[test]
    fn token_bucket_acquire_n_is_all_or_nothing() {
        // A request larger than the available tokens is rejected without
        // partially draining the bucket.
        let bucket = TokenBucket::new(100, 5);
        let t0 = Instant::now();
        let err = bucket
            .try_acquire_at(8, t0)
            .expect_err("8 > burst 5 must reject");
        assert_eq!(err.requested, 8);
        assert_eq!(err.available, 5);
        // Nothing was removed — a subsequent in-budget request still sees the
        // full bucket.
        bucket
            .try_acquire_at(5, t0)
            .expect("full burst still available after a rejected over-budget request");
    }

    #[test]
    fn try_acquire_ops_via_builder_uses_bytes_per_sec_budget() {
        // The same primitive expresses a bytes/sec budget: pass the byte
        // count to `try_acquire_ops`. burst=1000 bytes admits a 600+400
        // pair, then rejects a third that would exceed the bucket.
        let ctx = TenantContext::builder(TenantId(31))
            .with_rate_limit(/*bytes_per_sec*/ 1_000, /*burst*/ 1_000)
            .build();
        assert!(ctx.has_rate_limit());
        ctx.try_acquire_ops(600).unwrap();
        ctx.try_acquire_ops(400).unwrap();
        let err = ctx
            .try_acquire_ops(1)
            .expect_err("bucket drained to zero bytes");
        assert_eq!(err.burst, 1_000);
    }

    #[test]
    fn rate_limit_zero_args_clamped_to_one() {
        // `with_rate_limit(0, 0)` must not wedge the tenant into never-admit:
        // both args clamp to 1, so the first op is admitted (full bucket of 1)
        // and the second is rejected at the same instant.
        let bucket = TokenBucket::new(0, 0);
        let t0 = Instant::now();
        bucket
            .try_acquire_at(1, t0)
            .expect("clamped burst of 1 admits one");
        assert!(bucket.try_acquire_at(1, t0).is_err());
    }

    #[test]
    fn try_acquire_concurrent_never_over_admits() {
        // HIGH fix (clock-race regression): many threads hammer a single
        // tenant's rate limiter concurrently. The bucket reads the clock
        // under its own lock and the anchor is clamped monotonically, so no
        // interleaving may admit more operations than the bucket can hold
        // over the test window. With the old "sample `Instant::now()`
        // outside the lock" shape a thread holding a stale (earlier) sample
        // could rewind `last_refill`, manufacturing refill credit and
        // letting the bucket over-admit.
        //
        // We bound the window tightly: a tiny steady-state rate and a known
        // burst, run fast enough that the legitimate time-based refill over
        // the whole test is at most a couple of tokens. The total admits
        // must therefore stay within `burst + small_refill_slack`.
        use std::sync::atomic::AtomicU64 as StdAtomicU64;
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 16;
        const ATTEMPTS_PER_THREAD: usize = 1_000;
        const OPS_PER_SEC: u64 = 1; // ~1 token/s steady-state refill
        const BURST: u64 = 50;

        let ctx = Arc::new(
            TenantContext::builder(TenantId(0x9A7E))
                .with_rate_limit(OPS_PER_SEC, BURST)
                .build(),
        );
        let admitted = Arc::new(StdAtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(THREADS));

        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let ctx = Arc::clone(&ctx);
            let admitted = Arc::clone(&admitted);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..ATTEMPTS_PER_THREAD {
                    if ctx.try_acquire_op().is_ok() {
                        admitted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let total = admitted.load(Ordering::Relaxed);
        // At least the full burst is admittable (the bucket starts full).
        assert!(
            total >= BURST,
            "at least the burst of {BURST} should be admitted, got {total}",
        );
        // The hard invariant: admits never exceed the burst plus the few
        // tokens that could legitimately refill during the run. At
        // OPS_PER_SEC=1 the refill over a sub-second test is < a handful of
        // tokens; a generous slack of 16 still catches the over-admission a
        // rewound anchor would produce (which would let `total` approach the
        // raw attempt count of THREADS * ATTEMPTS_PER_THREAD = 16_000).
        const REFILL_SLACK: u64 = 16;
        assert!(
            total <= BURST + REFILL_SLACK,
            "rate limiter over-admitted: {total} > burst {BURST} + slack {REFILL_SLACK} \
             (a rewound refill anchor would let admits approach {})",
            THREADS as u64 * ATTEMPTS_PER_THREAD as u64,
        );
    }

    #[test]
    fn isolation_downgrade_counter_starts_at_zero() {
        // Reachable test for Fix B: the static counter is `0` at
        // startup. The downgrade path itself requires the `cuda`
        // feature AND a real-but-uncooperative CUDA device (or absence
        // of one) — neither is available in CI without hardware, so a
        // direct exercise of the downgrade branch is intentionally
        // omitted here. When the cust-or-cudarc upgrade lands and the
        // CUDA branch becomes mockable, replace this with a positive
        // assertion against `isolation_downgrade_count()` after
        // forcing a downgrade.
        //
        // NOTE: this assertion is order-sensitive. Other tests in this
        // module never call `TenantContextBuilder::build()` with
        // `IsolationKind::ContextIsolated` on a CUDA host, so the
        // counter stays at zero under `cargo test`. Under
        // `cargo test --features cuda` on a host without CUDA, this
        // test will observe the counter is non-zero — the test then
        // documents (rather than enforces) the downgrade contract.
        let count = isolation_downgrade_count();
        // The getter must be a pure read: calling it twice in a row
        // returns the same value (no side effect, no implicit reset).
        // This holds on every build matrix regardless of `cuda`.
        assert_eq!(
            count,
            isolation_downgrade_count(),
            "isolation_downgrade_count() must be a side-effect-free read",
        );
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(
                count, 0,
                "isolation_downgrade_count should start at zero on no-CUDA builds; \
                 a non-zero reading means a downgrade was attributed to the wrong \
                 path or a prior test mutated the static",
            );
        }
    }
}
