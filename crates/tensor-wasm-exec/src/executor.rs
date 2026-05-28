// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! [`TensorWasmExecutor`] — async executor for TensorWasm Wasm instances.
//!
//! Owns a shared [`TensorWasmEngine`] and a registry of live [`TensorWasmInstance`]s
//! keyed by [`InstanceId`]. Exposes the trio of operations
//! [`TensorWasmExecutor::spawn_instance`], [`TensorWasmExecutor::call_export`], and
//! [`TensorWasmExecutor::terminate`] — all async, all driven from the calling
//! Tokio runtime.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::{InstanceId, TenantId};
use dashmap::{mapref::entry::Entry, DashMap};
use lru::LruCache;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};
use wasmtime::{ExternType, Module, ResourceLimiter, Store, Val};

use crate::engine::TensorWasmEngine;
use crate::instance::{TensorWasmInstance, InstanceState};

/// Convert a wall-clock [`Duration`] into a number of epoch ticks suitable
/// for [`wasmtime::Store::set_epoch_deadline`].
///
/// Rounds up so a sub-tick remainder still terminates, with a floor of 1 so
/// `Duration::from_nanos(1)` does not silently become "never trip". Clamps
/// at [`u64::MAX`] if a caller supplies a pathologically long deadline. The
/// `tick` parameter is the engine's `epoch_tick` cadence; a zero-or-less tick
/// is treated as 1 ms to avoid division-by-zero on a malformed config.
fn duration_to_epoch_ticks(d: Duration, tick: Duration) -> u64 {
    let d_ms = d.as_millis();
    let t_ms = tick.as_millis().max(1);
    let ticks_u128 = d_ms.div_ceil(t_ms).max(1);
    u64::try_from(ticks_u128).unwrap_or(u64::MAX)
}

/// Hard upper bound on how long a Wasm module's `start` function (and any
/// other code that runs inside [`wasmtime::Instance::new_async`]) is allowed
/// to execute before the epoch interrupt trips.
///
/// Without this cap a `SpawnConfig { deadline: None, .. }` would set the
/// per-store epoch deadline to `u64::MAX`, which means an infinite-loop
/// start function would burn forever inside `Instance::new_async`. Because
/// the instance is not registered with the executor until that call
/// returns, [`TensorWasmExecutor::terminate`] cannot reach it — the only
/// thing that can interrupt the loop is the epoch deadline. 30 seconds is
/// generous for legitimate start functions (which typically just call out
/// to a few initialisers) while still bounding the worst case.
pub const MAX_START_FN_DURATION: Duration = Duration::from_secs(30);

/// Errors raised by the executor.
#[derive(Debug, Error)]
pub enum ExecError {
    /// Wasmtime returned an error during compile / instantiate / call.
    ///
    /// The full wasmtime error chain (including any inner trap, backtrace,
    /// or compile-error context) is preserved via `#[from]` (which thiserror
    /// also wires as `#[source]`) so callers converting to
    /// [`tensor_wasm_core::error::TensorWasmError`] do not lose detail.
    #[error("wasmtime error")]
    Wasmtime(#[from] wasmtime::Error),
    /// Looked up an instance that does not exist (or has terminated).
    #[error("no such instance: {0}")]
    NotFound(InstanceId),
    /// Looked up an export that the instance does not provide.
    #[error("instance has no export `{0}`")]
    MissingExport(String),
    /// The instance ran past its deadline before the call completed.
    ///
    /// Carries the offending [`InstanceId`] plus the real `elapsed_ms` /
    /// `deadline_ms` figures captured at the time the epoch interrupt
    /// fired. Surfaced as [`tensor_wasm_core::error::TensorWasmError::KernelTimeout`]
    /// with the same numbers on the conversion boundary.
    ///
    /// Tuple-shaped (rather than a struct variant) so existing match arms
    /// like `ExecError::Timeout(_)` keep compiling.
    #[error("{0}")]
    Timeout(TimeoutContext),
    /// The module declares — via an exported or imported
    /// [`wasmtime::ExternType::Memory`] — an initial or maximum linear
    /// memory size that exceeds `EngineConfig::max_memory_bytes`.
    ///
    /// Surfaced *before* `Instance::new_async` because Wasmtime's
    /// [`wasmtime::ResourceLimiter::memory_growing`] only fires on
    /// `memory.grow`, not on the initial allocation. A guest declaring
    /// `(memory 1 65536)` would otherwise force a 4 GiB allocation at
    /// instantiation. Maps to
    /// [`tensor_wasm_core::error::TensorWasmError::MemoryExhausted`].
    #[error("module-declared linear memory {requested_bytes} bytes exceeds engine cap {limit_bytes} bytes")]
    ModuleMemoryTooLarge {
        /// Bytes the module asked for (initial or declared maximum,
        /// whichever tripped the check first).
        requested_bytes: u64,
        /// Configured engine-wide per-instance cap in bytes.
        limit_bytes: u64,
    },
    /// The executor refused to admit a new instance because the
    /// engine-wide live-instance cap
    /// ([`crate::engine::EngineConfig::max_instances`]) is already
    /// saturated.
    ///
    /// Surfaced from [`TensorWasmExecutor::spawn_instance`] *before* any
    /// compile / instantiate work; the failed spawn never consumes a
    /// slot in the registry. Mapped to
    /// [`tensor_wasm_core::error::TensorWasmError::MemoryExhausted`] on
    /// the conversion boundary (the API layer surfaces it as 503).
    #[error("instance capacity exhausted: {active} active, limit {limit}")]
    CapacityExhausted {
        /// Live-instance count observed at the rejection point
        /// (post-increment, so `active > limit`).
        active: usize,
        /// Configured engine-wide ceiling.
        limit: usize,
    },
}

/// Payload for [`ExecError::Timeout`]. Carries the real elapsed and deadline
/// figures captured when the epoch interrupt fired so the error mapping
/// layer can surface them through [`tensor_wasm_core::error::TensorWasmError::KernelTimeout`].
#[derive(Debug, Clone, Copy)]
pub struct TimeoutContext {
    /// Instance that exceeded its deadline.
    pub id: InstanceId,
    /// Wall-clock milliseconds the call took before being interrupted.
    pub elapsed_ms: u64,
    /// Configured per-call deadline in milliseconds.
    pub deadline_ms: u64,
}

impl std::fmt::Display for TimeoutContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "instance {} exceeded deadline (elapsed {} ms, deadline {} ms)",
            self.id, self.elapsed_ms, self.deadline_ms,
        )
    }
}

impl From<ExecError> for tensor_wasm_core::error::TensorWasmError {
    fn from(e: ExecError) -> Self {
        use tensor_wasm_core::error::TensorWasmError;
        match e {
            ExecError::Wasmtime(err) => {
                // Distinguish runtime traps from compile/instantiate errors.
                // wasmtime wraps runtime traps as `wasmtime::Trap` inside the
                // anyhow error; compile/parse failures do NOT. We classify
                // accordingly so the unified `TensorWasmError` carries the right
                // variant (and `is_retryable` / `kind` reflect it).
                //
                // SECURITY (exec S-9): the full wasmtime error chain
                // (`format!("{err:#}")`) walks every `#[source]` link and
                // routinely surfaces host pointer addresses, host file paths,
                // and internal stack-frame names — none of which are safe to
                // hand back to an untrusted caller. We therefore log the full
                // chain server-side and return a stable opaque string in the
                // payload so external observers cannot fingerprint the host.
                let is_trap = err.downcast_ref::<wasmtime::Trap>().is_some();
                tracing::error!(
                    target: "tensor_wasm_exec::executor",
                    error = ?err,
                    error_chain = %format!("{err:#}"),
                    is_trap,
                    "wasmtime trap",
                );
                if is_trap {
                    TensorWasmError::WasmTrap("wasm trap".into())
                } else {
                    TensorWasmError::WasmCompile("wasm compile failed".into())
                }
            }
            ExecError::NotFound(id) => {
                TensorWasmError::Serialization(format!("instance not found: {id}").into())
            }
            ExecError::MissingExport(name) => {
                TensorWasmError::Serialization(format!("instance missing export: {name}").into())
            }
            ExecError::Timeout(ctx) => TensorWasmError::KernelTimeout {
                elapsed_ms: ctx.elapsed_ms,
                deadline_ms: ctx.deadline_ms,
            },
            ExecError::ModuleMemoryTooLarge {
                requested_bytes,
                limit_bytes,
            } => TensorWasmError::MemoryExhausted {
                requested: requested_bytes,
                limit: limit_bytes,
            },
            ExecError::CapacityExhausted { active, limit } => TensorWasmError::MemoryExhausted {
                requested: active as u64,
                limit: limit as u64,
            },
        }
    }
}

/// Configuration passed to [`TensorWasmExecutor::spawn_instance`].
#[derive(Debug, Clone)]
pub struct SpawnConfig {
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Optional per-call deadline.
    pub deadline: Option<Duration>,
    /// Arguments forwarded to the first [`TensorWasmExecutor::call_export_with_args`]
    /// invocation against this instance.
    ///
    /// Callers that drive a single spawn-then-call flow (CLI `run`, API
    /// `/invoke`) populate this field so the caller's argument list survives
    /// the trip across crate boundaries without a parallel `CallConfig`.
    /// Multi-call flows should ignore this field and pass arguments directly
    /// to each `call_export_with_args` invocation.
    pub args: Vec<WasmArg>,
}

impl SpawnConfig {
    /// Construct with just a tenant and no deadline.
    pub fn for_tenant(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            deadline: None,
            args: Vec::new(),
        }
    }

    /// Add a deadline relative to "now at spawn time".
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Attach an argument list for the upcoming call. See [`SpawnConfig::args`].
    pub fn with_args(mut self, args: Vec<WasmArg>) -> Self {
        self.args = args;
        self
    }
}

/// A typed Wasm value supplied to [`TensorWasmExecutor::call_export_with_args`].
///
/// Mirrors the four core wasm value types (`i32`, `i64`, `f32`, `f64`). Held
/// `Copy` so callers can clone an argument list cheaply when retrying. Marked
/// `#[non_exhaustive]` so additional value types (e.g. `v128`, reference
/// types) can be added in a future minor release without breaking the
/// match-arm count on downstream code.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum WasmArg {
    /// A 32-bit signed integer argument.
    I32(i32),
    /// A 64-bit signed integer argument.
    I64(i64),
    /// A 32-bit IEEE-754 float argument.
    F32(f32),
    /// A 64-bit IEEE-754 float argument.
    F64(f64),
}

impl WasmArg {
    /// Convert a [`serde_json::Value`] into the closest-fitting [`WasmArg`]
    /// variant.
    ///
    /// Integer literals that fit in `i32` become [`WasmArg::I32`]; larger
    /// integers become [`WasmArg::I64`]; non-integer numerics become
    /// [`WasmArg::F64`]. Any non-numeric value is rejected with an error
    /// string suitable for forwarding into a user-facing CLI / HTTP error.
    /// `f32` cannot be selected from JSON unambiguously — callers needing a
    /// 32-bit float should construct [`WasmArg::F32`] directly.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, &'static str> {
        match v {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    if let Ok(i32v) = i32::try_from(i) {
                        Ok(WasmArg::I32(i32v))
                    } else {
                        Ok(WasmArg::I64(i))
                    }
                } else if let Some(f) = n.as_f64() {
                    Ok(WasmArg::F64(f))
                } else {
                    Err("unsupported number")
                }
            }
            _ => Err("unsupported arg type — only numbers"),
        }
    }

    /// Convert a [`WasmArg`] into the wasmtime [`Val`] expected by
    /// `Func::call_async`. `f32`/`f64` are stored as bit patterns per the
    /// wasmtime ABI.
    pub fn into_val(self) -> wasmtime::Val {
        match self {
            WasmArg::I32(v) => wasmtime::Val::I32(v),
            WasmArg::I64(v) => wasmtime::Val::I64(v),
            WasmArg::F32(v) => wasmtime::Val::F32(v.to_bits()),
            WasmArg::F64(v) => wasmtime::Val::F64(v.to_bits()),
        }
    }
}

/// Render a wasmtime [`Val`] as the closest-fitting [`serde_json::Value`].
///
/// `i32`/`i64` become JSON numbers (integer); `f32`/`f64` become JSON
/// numbers (floating-point); other value types — `v128`, references —
/// degrade to a JSON `null` so callers see a stable shape rather than a
/// runtime error. Used by [`TensorWasmExecutor::call_export_with_args`]
/// to project the wasmtime result slice into a JSON array.
fn val_to_json(v: &Val) -> serde_json::Value {
    match v {
        Val::I32(n) => serde_json::json!(*n),
        Val::I64(n) => serde_json::json!(*n),
        Val::F32(bits) => serde_json::json!(f32::from_bits(*bits)),
        Val::F64(bits) => serde_json::json!(f64::from_bits(*bits)),
        // Unsupported value types fall through as JSON null rather than
        // erroring — keeps the response shape predictable for callers that
        // only ever return numeric scalars (the common case for B5.6).
        _ => serde_json::Value::Null,
    }
}

/// Per-store [`ResourceLimiter`] that caps linear-memory growth at the
/// engine-configured `max_memory_bytes`.
///
/// One instance is attached to each [`Store`] via [`Store::limiter`].
/// Constructing it is cheap (a single `usize` plus the cached limit). The
/// `engine_max` field is duplicated from [`crate::engine::EngineConfig::max_memory_bytes`]
/// so the limiter does not need to re-borrow the engine during the hot
/// `memory.grow` path.
#[derive(Debug)]
pub struct TensorWasmResourceLimiter {
    /// Per-instance hard cap on linear memory. Mirrored from the engine config.
    engine_max: usize,
}

impl TensorWasmResourceLimiter {
    /// Construct a limiter that denies any growth past `engine_max` bytes.
    pub fn new(engine_max: usize) -> Self {
        Self { engine_max }
    }
}

impl ResourceLimiter for TensorWasmResourceLimiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        // Reject if the requested size exceeds either the engine-wide cap
        // or the module's own declared maximum. Returning `Ok(false)` causes
        // `memory.grow` to return -1 in guest land, mirroring wasmtime's
        // `StoreLimits` convention. Hard traps would not surface a stable
        // error to the host; instead the executor maps subsequent OOM
        // behaviour via `ExecError::Wasmtime`.
        if desired > self.engine_max {
            return Ok(false);
        }
        if let Some(m) = maximum {
            if desired > m {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: u32,
        desired: u32,
        maximum: Option<u32>,
    ) -> wasmtime::Result<bool> {
        // Cap table growth proportionally to the per-instance memory budget.
        // Each table entry costs ~16 bytes of host memory on wasmtime (a
        // tagged pointer plus type-index slot). Without this cap a guest
        // could `table.grow` up to u32::MAX entries (~64 GiB of host RAM at
        // 16 B/entry), bypassing the `memory_growing` cap entirely.
        //
        // Using `engine_max` (the linear-memory byte cap) as the table-byte
        // budget keeps the policy a single dial: a tenant gets at most
        // `engine_max` bytes of *either* linear memory *or* table backing
        // store. That's loose (allows ~engine_max bytes for each) but it
        // bounds the worst case from u32::MAX entries down to engine_max/16
        // entries — the qualitative DoS vector closes.
        const TABLE_ENTRY_BYTES: u64 = 16;
        let bytes_needed = u64::from(desired).saturating_mul(TABLE_ENTRY_BYTES);
        if bytes_needed > self.engine_max as u64 {
            return Ok(false);
        }
        if let Some(m) = maximum {
            if desired > m {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The async executor.
#[derive(Clone)]
pub struct TensorWasmExecutor {
    engine: Arc<TensorWasmEngine>,
    instances: Arc<DashMap<InstanceId, Arc<Mutex<TensorWasmInstance>>>>,
    next_instance_id: Arc<AtomicU64>,
    /// Per-engine compiled-module cache keyed by the full 256-bit BLAKE3
    /// digest of the wasm bytes. Avoids re-running Cranelift on every
    /// `spawn_instance` for repeat tenants. Hash is computed with `blake3`
    /// (SIMD-accelerated; ~5x faster than SipHash on multi-MiB wasm
    /// modules). The full 32-byte digest is used as the key — truncating
    /// to 8 bytes would expose a ~2⁻³² birthday-collision window across
    /// tenants (cross-tenant module-cache poisoning) that an attacker
    /// crafting modules with colliding prefixes could exploit at scale.
    ///
    /// Bounded with LRU eviction (cap from
    /// [`crate::engine::EngineConfig::max_module_cache_entries`], default
    /// 1024) — closes exec S-5 where an unbounded `DashMap` let a
    /// misbehaving tenant pin arbitrarily many compiled modules. The
    /// guard is a `parking_lot::Mutex` rather than a `DashMap` because
    /// `lru::LruCache` is not concurrency-safe (every `get` mutates the
    /// recency list).
    module_cache: Arc<parking_lot::Mutex<LruCache<[u8; 32], Module>>>,
    /// Live-instance counter, used to enforce
    /// [`crate::engine::EngineConfig::max_instances`] (exec S-10).
    /// Atomically bumped *before* compile/instantiate in `spawn_instance`
    /// (with rollback on failure) and decremented in `terminate`. We keep
    /// this separate from `instances.len()` so the admission decision
    /// commits in a single CAS rather than racing against in-flight
    /// spawns that have already passed the check but not yet inserted.
    instance_count: Arc<AtomicUsize>,
    /// Optional metrics handle. When `Some`, spawn/terminate operations
    /// increment the corresponding Prometheus counters / gauges.
    metrics: Option<TensorWasmMetrics>,
    /// One-shot guard for the "epoch ticker not running" operator warning.
    ///
    /// Initialised lazily on first observation of a missing ticker; the
    /// inner [`AtomicBool`] flips to `true` once the warning has fired so
    /// subsequent spawns on the same executor stay quiet (the warning is
    /// load-bearing for operators, but at 1 line per spawn it would flood
    /// the log). Scoped to the executor (and therefore the engine) so
    /// distinct engines in the same process each get their own warning.
    ticker_warned: Arc<OnceLock<AtomicBool>>,
}

/// Resolve a non-zero LRU cache capacity from a possibly-zero config
/// value. We coerce 0 to 1 because `LruCache::new(NonZeroUsize)` requires
/// a non-zero capacity, and operators who set the knob to 0 most plausibly
/// meant "as small as possible" rather than "panic on construction".
fn lru_cap(requested: usize) -> NonZeroUsize {
    NonZeroUsize::new(requested).unwrap_or_else(|| NonZeroUsize::new(1).expect("1 is non-zero"))
}

/// RAII guard that rolls back a successful `instance_count.fetch_add`
/// if the spawn path drops it without committing. `commit()` defuses
/// the rollback once the instance has been inserted into the registry;
/// every other exit path (`?`, panic during `Instance::new_async`,
/// store-construction failure) leaves the guard alive and triggers a
/// decrement on drop.
///
/// Without this guard, exec S-10 admission control would leak a slot
/// for every failed spawn — a misbehaving tenant could trip an
/// always-failing instantiation in a loop and exhaust the cap with
/// zero live instances.
struct InstanceSlotGuard {
    counter: Arc<AtomicUsize>,
    committed: bool,
}

impl InstanceSlotGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        Self { counter, committed: false }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for InstanceSlotGuard {
    fn drop(&mut self) {
        if !self.committed {
            // Relaxed is fine here: the matching `fetch_add` used AcqRel
            // for admission ordering; the rollback only undoes a count
            // that no other thread depends on observing.
            self.counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Walk every exported and imported [`ExternType::Memory`] in `module` and
/// reject the spawn if either the initial (`minimum`) or the declared
/// `maximum` size, expressed in bytes via the memory type's own
/// `page_size()`, exceeds `cap_bytes`.
///
/// Returns [`ExecError::ModuleMemoryTooLarge`] on the first offending
/// memory found. The check runs against the compiled [`Module`] before
/// `Instance::new_async`, so a rejected module is never instantiated and
/// no host allocation is attempted on its behalf.
fn check_module_memory_within_cap(module: &Module, cap_bytes: usize) -> Result<(), ExecError> {
    let cap_u64 = cap_bytes as u64;
    let mut check = |mt: &wasmtime::MemoryType| -> Result<(), ExecError> {
        let page_size = mt.page_size();
        // `minimum()` is in pages; multiply with overflow-safe saturating
        // arithmetic so a pathological declaration cannot wrap on cast.
        let min_pages = mt.minimum();
        let min_bytes = min_pages.saturating_mul(page_size);
        if min_bytes > cap_u64 {
            return Err(ExecError::ModuleMemoryTooLarge {
                requested_bytes: min_bytes,
                limit_bytes: cap_u64,
            });
        }
        if let Some(max_pages) = mt.maximum() {
            let max_bytes = max_pages.saturating_mul(page_size);
            if max_bytes > cap_u64 {
                return Err(ExecError::ModuleMemoryTooLarge {
                    requested_bytes: max_bytes,
                    limit_bytes: cap_u64,
                });
            }
        }
        Ok(())
    };
    for ex in module.exports() {
        if let ExternType::Memory(mt) = ex.ty() {
            check(&mt)?;
        }
    }
    for im in module.imports() {
        if let ExternType::Memory(mt) = im.ty() {
            check(&mt)?;
        }
    }
    Ok(())
}

impl TensorWasmExecutor {
    /// Construct an executor over the given shared engine.
    pub fn new(engine: Arc<TensorWasmEngine>) -> Self {
        let cap = lru_cap(engine.config().max_module_cache_entries);
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(parking_lot::Mutex::new(LruCache::new(cap))),
            instance_count: Arc::new(AtomicUsize::new(0)),
            metrics: None,
            ticker_warned: Arc::new(OnceLock::new()),
        }
    }

    /// Construct an executor that publishes spawn/terminate events to the
    /// supplied [`TensorWasmMetrics`] registry. Metric handles are cheaply cloneable;
    /// pass a clone of the process-wide registry.
    pub fn with_metrics(engine: Arc<TensorWasmEngine>, metrics: TensorWasmMetrics) -> Self {
        let cap = lru_cap(engine.config().max_module_cache_entries);
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(parking_lot::Mutex::new(LruCache::new(cap))),
            instance_count: Arc::new(AtomicUsize::new(0)),
            metrics: Some(metrics),
            ticker_warned: Arc::new(OnceLock::new()),
        }
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &TensorWasmEngine {
        &self.engine
    }

    /// Number of currently live instances.
    pub fn live_count(&self) -> usize {
        self.instances.len()
    }

    /// Number of compiled modules retained in the per-executor cache.
    /// Exposed for tests and operators that want to confirm cache reuse.
    pub fn cached_module_count(&self) -> usize {
        self.module_cache.lock().len()
    }

    /// Current number of entries held by the bounded LRU module cache.
    /// Alias for [`Self::cached_module_count`] under the name used by the
    /// exec S-5 admission-control bound work; both delegate to the same
    /// underlying length so callers can pick whichever reads better at
    /// the call site.
    pub fn module_cache_len(&self) -> usize {
        self.module_cache.lock().len()
    }

    /// Current number of live instances, sampled atomically. Mirrors the
    /// counter the admission check in `spawn_instance` consults to decide
    /// whether a new instance fits under
    /// [`crate::engine::EngineConfig::max_instances`].
    pub fn instances_len(&self) -> usize {
        self.instance_count.load(Ordering::Acquire)
    }

    /// Generate a fresh, vacant [`InstanceId`].
    ///
    /// `next_instance_id` is an `AtomicU64` widened to a `u128` on insert.
    /// At 1 instance per nanosecond it would take ~584 years to wrap, but
    /// we still defend against collisions: if the freshly-allocated id is
    /// already occupied (post-wrap or external reservation), bump and
    /// retry. A `warn!` event fires on every collision so operators see
    /// it long before the registry corrupts.
    fn allocate_instance_id(&self) -> InstanceId {
        loop {
            let raw = self.next_instance_id.fetch_add(1, Ordering::Relaxed);
            let id = InstanceId(u128::from(raw));
            if !self.instances.contains_key(&id) {
                return id;
            }
            warn!(
                target: "tensor_wasm_exec::executor",
                raw,
                "instance id collision detected; retrying with next sequence value",
            );
        }
    }

    /// Compile `wasm` via wasmtime, caching the result so repeat calls with
    /// the same bytes return without re-running Cranelift. Cache key is the
    /// full 32-byte BLAKE3 digest of the wasm bytes — stable across runs
    /// and platforms, and ~5x faster than SipHash on the multi-MiB modules
    /// we actually compile. Using the full digest (rather than truncating
    /// to 8 bytes) closes a cross-tenant cache-poisoning vector: at 8
    /// bytes, a 65k-module corpus has a ~2⁻³² collision chance per pair,
    /// which an attacker crafting prefix-colliding modules can amplify.
    fn compile_module_cached(&self, wasm: &[u8]) -> Result<Module, ExecError> {
        let digest = blake3::hash(wasm);
        // BLAKE3 outputs a fixed 32-byte digest; use it whole as the cache key.
        let key: [u8; 32] = *digest.as_bytes();
        // Scoped lock for the get: releasing the mutex before the
        // potentially-expensive `Module::from_binary` call below is what
        // lets concurrent spawns of *different* modules compile in
        // parallel. The cost is that two spawns of the *same* fresh
        // module may both compile it — but the second one's `put` simply
        // overwrites the first, no correctness hazard.
        if let Some(m) = self.module_cache.lock().get(&key).cloned() {
            return Ok(m);
        }
        let module = Module::from_binary(self.engine.inner(), wasm)
            .map_err(ExecError::Wasmtime)?;
        self.module_cache.lock().put(key, module.clone());
        Ok(module)
    }

    /// Compile + instantiate a Wasm module. Returns the assigned [`InstanceId`].
    #[instrument(skip(self, wasm), fields(tenant = %cfg.tenant_id, instance_id = tracing::field::Empty))]
    pub async fn spawn_instance(
        &self,
        cfg: SpawnConfig,
        wasm: &[u8],
    ) -> Result<InstanceId, ExecError> {
        // Admission control (exec S-10). Bump the live-instance counter
        // BEFORE doing any compile / instantiate work; if the new total
        // exceeds the engine cap, roll back immediately and surface a
        // typed `CapacityExhausted` so the API layer can map it to 503.
        //
        // The fetch_add must precede the limit check so concurrent spawns
        // see a consistent atomic view: if two threads both observe `N-1`
        // active when the cap is `N`, both pre-incrementing produces
        // `N` and `N+1` respectively — the second one fails the check
        // and rolls back. Doing the read+check+inc separately would let
        // both threads pass and overshoot the cap.
        if let Some(max) = self.engine.config().max_instances {
            let new_count = self.instance_count.fetch_add(1, Ordering::AcqRel) + 1;
            if new_count > max {
                self.instance_count.fetch_sub(1, Ordering::Relaxed);
                return Err(ExecError::CapacityExhausted {
                    active: new_count,
                    limit: max,
                });
            }
        } else {
            // No cap configured — still bump so `instances_len` stays
            // accurate. The drop guard below covers rollback on any
            // subsequent error path.
            self.instance_count.fetch_add(1, Ordering::AcqRel);
        }
        // Rollback guard for every failure path between here and the
        // registry insert. `commit()` is called only after the instance
        // is published into `self.instances`.
        let slot_guard = InstanceSlotGuard::new(self.instance_count.clone());

        let id = self.allocate_instance_id();
        tracing::Span::current().record("instance_id", tracing::field::display(id));
        let max_memory_bytes = self.engine.config().max_memory_bytes;
        let mut state =
            InstanceState::new(cfg.tenant_id, id).with_memory_limit(max_memory_bytes);
        if let Some(d) = cfg.deadline {
            // Seed the absolute deadline at spawn time so the first call has
            // a meaningful window even if it fires before `call_export` gets
            // a chance to re-arm (which it always does in practice, but the
            // invariant is "deadline is set whenever deadline_duration is").
            // The per-call re-arm in `call_export` keeps subsequent calls
            // honest — without it a second call inherits the elapsed window
            // from the first and the timeout report degenerates to 0/0.
            state = state
                .with_deadline(Instant::now() + d)
                .with_deadline_duration(d);
        }
        // Translate the SpawnConfig deadline (a wall-clock Duration) into the
        // number of epoch ticks after which Wasmtime should interrupt execution.
        //
        // Wasmtime's `Store::set_epoch_deadline(ticks_beyond_current)` is
        // *relative* to the engine's current epoch — i.e. the deadline trips
        // once `Engine::increment_epoch` has fired that many more times since
        // this call. That's exactly the semantics we want: each Store starts
        // fresh, and the ticker drives the only progression of the counter.
        let tick = self.engine.config().epoch_tick;
        let epoch_deadline_ticks = match cfg.deadline {
            Some(d) => duration_to_epoch_ticks(d, tick),
            // No deadline → effectively unbounded. u64::MAX is fine: the
            // engine's epoch counter would need ~5.8 billion years at 10 ms/
            // tick to reach it, so the deadline will never trip in practice.
            None => u64::MAX,
        };
        // Separate epoch budget for the *start* function (and anything else
        // that runs inside `Instance::new_async`). Without this cap, a
        // `SpawnConfig { deadline: None, .. }` would set the epoch deadline
        // to `u64::MAX`, so an infinite-loop start function would burn
        // forever inside `new_async`. The instance is not registered with
        // the executor until that call returns, so `terminate` cannot
        // reach it — the only thing that can interrupt the loop is the
        // epoch deadline. Cap at the MIN of the caller's per-call deadline
        // (if any) and `MAX_START_FN_DURATION`.
        let max_start_ticks = {
            let d_ms = MAX_START_FN_DURATION.as_millis();
            let t_ms = tick.as_millis().max(1);
            let ticks_u128 = d_ms.div_ceil(t_ms).max(1);
            u64::try_from(ticks_u128).unwrap_or(u64::MAX)
        };
        let start_deadline_ticks = epoch_deadline_ticks.min(max_start_ticks);
        // Warn (once per engine/executor) if the ticker isn't running —
        // without it, neither the start-function cap above nor any
        // call-time deadline will actually fire, and a runaway guest
        // will wedge the worker thread until it returns of its own accord.
        if !self.engine.is_epoch_ticker_running() {
            let flag = self
                .ticker_warned
                .get_or_init(|| AtomicBool::new(false));
            if !flag.swap(true, Ordering::AcqRel) {
                tracing::error!(
                    target: "tensor_wasm_exec::executor",
                    "epoch ticker not running — deadlines will not fire; call `engine.spawn_epoch_ticker(Handle::current())` before serving traffic",
                );
            }
        }
        let module = self.compile_module_cached(wasm)?;

        // Pre-instantiation memory cap (closes mem-H5 / exec-S-2 / exec-S-10).
        // Wasmtime's `ResourceLimiter::memory_growing` fires only on
        // `memory.grow`, not on the initial allocation a module declares
        // with `(memory N M)`. A guest could therefore force a multi-GiB
        // allocation at instantiation time without ever calling
        // `memory.grow` — and the per-store `TensorWasmResourceLimiter`
        // would never see it. Walk every exported AND imported memory
        // type and reject the spawn if its initial OR maximum size
        // exceeds the engine's configured cap. We use the memory type's
        // own `page_size()` so this stays correct for both the wasm32
        // default 64 KiB pages and any future custom-page-size proposal
        // memory types Wasmtime accepts.
        check_module_memory_within_cap(&module, max_memory_bytes)?;

        let mut store = Store::new(self.engine.inner(), state);
        // Cap linear-memory growth at the engine-configured maximum. The
        // limiter lives inside the store payload (`InstanceState::limiter`)
        // so wasmtime can borrow it without any extra heap allocation per
        // `memory.grow` call. The explicit return type pins the trait
        // object coercion so type inference doesn't choose the concrete
        // `&mut TensorWasmResourceLimiter`.
        store.limiter(|state| &mut state.limiter as &mut dyn ResourceLimiter);
        // Use the *start*-function deadline for the instantiation phase.
        store.set_epoch_deadline(start_deadline_ticks);
        let instance = wasmtime::Instance::new_async(&mut store, &module, &[]).await?;
        // Restore the per-call deadline budget so subsequent
        // `call_export` invocations get the full configured deadline
        // (or unbounded `u64::MAX` when the caller did not supply one).
        store.set_epoch_deadline(epoch_deadline_ticks);
        let bi = TensorWasmInstance::new(store, instance);
        // Final occupancy check via `Entry::Vacant` — `allocate_instance_id`
        // already guards against active collisions, but a concurrent
        // `spawn_instance` racing on the same retry sequence is still
        // theoretically possible. `Vacant` insertion is atomic.
        match self.instances.entry(id) {
            Entry::Vacant(v) => {
                v.insert(Arc::new(Mutex::new(bi)));
            }
            Entry::Occupied(_) => {
                warn!(
                    target: "tensor_wasm_exec::executor",
                    %id,
                    "instance id race after allocation; this is a serious bug — please file an issue",
                );
                // slot_guard drops here → rollback counter.
                return Err(ExecError::Wasmtime(wasmtime::Error::msg(
                    "instance id collision after allocation",
                )));
            }
        }
        // Instance is now live in the registry — defuse the rollback
        // guard so the admission slot stays charged until `terminate`.
        slot_guard.commit();
        if let Some(m) = &self.metrics {
            m.instance_spawns_total().inc();
            m.active_instances().inc();
        }
        info!(target: "tensor_wasm_exec::executor", tenant = %cfg.tenant_id, instance = %id, "instance spawned");
        Ok(id)
    }

    /// Invoke `export` with no arguments and no return value.
    ///
    /// This is the minimal signature needed for the 100-instance integration
    /// test. Richer signatures arrive in S17 (HTTP API) and S18 (CLI).
    ///
    /// # Concurrency note
    ///
    /// The per-instance mutex is held across the inner `call_async` await
    /// point. Concurrent calls into the **same instance** therefore serialise
    /// — this matches wasmtime's `Store`-is-not-`Sync` contract. Concurrent
    /// calls into **different instances** run in parallel. If you need
    /// pipelined invocation on a single instance, spawn additional instances
    /// over the same module bytes (the executor's module cache makes that
    /// nearly free).
    ///
    /// If the executor's engine has not had `spawn_epoch_ticker` called and
    /// the instance was spawned with a deadline, this call will run until
    /// the wasm returns of its own accord — the deadline cannot fire without
    /// the ticker. A warning is logged the first time this combination is
    /// observed per call.
    #[instrument(skip(self), fields(instance = %id, export = %export))]
    pub async fn call_export(&self, id: InstanceId, export: &str) -> Result<(), ExecError> {
        // Back-compat wrapper: most callers (the bench loop, the executor's
        // own tests, the orphan-cleanup integration test) only need the
        // `() -> ()` signature and explicitly assert success via `.unwrap()`
        // or `?`. Threading the new `Result<serde_json::Value, _>` shape
        // through every call site would be churn for no behavioural gain;
        // instead we keep the unit-typed surface here and delegate to the
        // shared implementation. The result value is discarded — callers
        // wanting it should use [`Self::call_export_with_args`] directly.
        self.call_export_with_args(id, export, &[]).await.map(|_| ())
    }

    /// Invoke `export` with the supplied `args` (which may be empty) and
    /// return the export's result list, serialised as a JSON array.
    ///
    /// This is the general entry point for guest-export invocation; the
    /// `()`-shaped [`Self::call_export`] is a thin wrapper that discards
    /// the result. The choice of `serde_json::Value` for the return type
    /// keeps the executor's public API free of any tensor-wasm-api types
    /// while still giving the HTTP transport a structured payload to
    /// forward verbatim.
    ///
    /// When `args` is empty the implementation uses the typed
    /// `func.typed::<(), ()>()` fast path, matching the historical
    /// behaviour and keeping every existing `() -> ()` export call
    /// branch-for-branch identical. With a non-empty `args` slice the
    /// dynamic `func.call_async(&[Val], &mut [Val])` path runs instead;
    /// the result slice is sized from the export's declared result arity
    /// at runtime, so an export returning `(i32, i32)` produces a JSON
    /// array with two numbers.
    #[instrument(skip(self, args), fields(instance = %id, export = %export, args_len = args.len()))]
    pub async fn call_export_with_args(
        &self,
        id: InstanceId,
        export: &str,
        args: &[WasmArg],
    ) -> Result<serde_json::Value, ExecError> {
        let handle = self
            .instances
            .get(&id)
            .ok_or(ExecError::NotFound(id))?
            .value()
            .clone();
        let mut guard = handle.lock().await;
        // Re-arm the deadline at the start of each call.
        //
        // The fields on `InstanceState` work in concert: `deadline_duration`
        // is the configured per-call budget (immutable for the life of the
        // instance), and `deadline` is the absolute `Instant` we expect to
        // not cross. At spawn time we seeded `deadline = now + d`, but if a
        // caller invokes `call_export` twice with delay in between, the
        // second call would inherit an already-elapsed deadline — and the
        // wasmtime epoch counter set at spawn would already be consumed.
        // That used to surface as `Timeout { elapsed_ms: 0, deadline_ms: 0 }`
        // because the wasmtime trap fired before any real work happened and
        // the legacy `deadline_at.saturating_duration_since(started_at)`
        // returned zero. Re-arming here gives every call its own honest
        // window (and honest numbers if it does time out).
        let call_start = Instant::now();
        let configured_deadline = guard.store.data().deadline_duration;
        if let Some(d) = configured_deadline {
            let new_deadline = call_start + d;
            guard.store.data_mut().deadline = Some(new_deadline);
            let tick = self.engine.config().epoch_tick;
            guard
                .store
                .set_epoch_deadline(duration_to_epoch_ticks(d, tick));
        }
        let deadline_at = guard.store.data().deadline;
        let configured_deadline_ms = configured_deadline
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let wasmtime_instance = *guard.wasmtime_instance();
        let func = wasmtime_instance
            .get_func(&mut guard.store, export)
            .ok_or_else(|| ExecError::MissingExport(export.to_string()))?;

        // Branch on argument arity: the empty-args case uses the typed
        // fast path so we don't disturb the behaviour every existing
        // `() -> ()` test relies on (and so the dynamic-call overhead
        // stays off the bench path). The non-empty case takes the
        // dynamic `Func::call_async` path with `&[Val]` IO buffers; the
        // result vec is pre-sized to the export's declared result arity
        // so wasmtime can write straight into it.
        let call_outcome = if args.is_empty() {
            match func.typed::<(), ()>(&guard.store) {
                Ok(typed) => typed
                    .call_async(&mut guard.store, ())
                    .await
                    .map(|()| serde_json::Value::Array(Vec::new())),
                Err(e) => Err(e),
            }
        } else {
            let params: Vec<Val> = args.iter().copied().map(WasmArg::into_val).collect();
            let func_ty = func.ty(&guard.store);
            // `Val::I32(0)` is just a placeholder — wasmtime overwrites
            // every slot before returning. The element count must match
            // the export's declared result arity exactly or wasmtime
            // returns an error.
            let mut results: Vec<Val> = vec![Val::I32(0); func_ty.results().len()];
            match func.call_async(&mut guard.store, &params, &mut results).await {
                Ok(()) => {
                    let json: Vec<serde_json::Value> = results.iter().map(val_to_json).collect();
                    Ok(serde_json::Value::Array(json))
                }
                Err(e) => Err(e),
            }
        };

        match call_outcome {
            Ok(value) => Ok(value),
            Err(err) => {
                // If we had a deadline AND the wall clock has tripped past
                // it, classify the failure as Timeout with real numbers.
                // Otherwise propagate the raw wasmtime error.
                let elapsed = call_start.elapsed();
                let past_deadline = deadline_at
                    .map(|d| Instant::now() >= d)
                    .unwrap_or(false);
                if past_deadline {
                    Err(ExecError::Timeout(TimeoutContext {
                        id,
                        elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                        deadline_ms: configured_deadline_ms,
                    }))
                } else {
                    Err(ExecError::Wasmtime(err))
                }
            }
        }
    }

    /// Drop the instance, releasing its resources.
    #[instrument(skip(self), fields(instance = %id))]
    pub async fn terminate(&self, id: InstanceId) -> Result<(), ExecError> {
        match self.instances.remove(&id) {
            Some(_) => {
                // Release the admission slot reserved at spawn time
                // (exec S-10). The decrement only runs on successful
                // removal — a `NotFound` terminate must not free a
                // slot it never charged, or a tenant could double-
                // terminate to inflate their effective cap.
                self.instance_count.fetch_sub(1, Ordering::AcqRel);
                if let Some(m) = &self.metrics {
                    m.instance_terminations_total().inc();
                    m.active_instances().dec();
                }
                debug!(target: "tensor_wasm_exec::executor", instance = %id, "instance terminated");
                Ok(())
            }
            None => Err(ExecError::NotFound(id)),
        }
    }

    /// Call an export, then unconditionally terminate the instance —
    /// even if the returned future is dropped mid-await (api S-20 +
    /// orphan-instance cleanup).
    ///
    /// The previous flow (`call_export` + explicit `terminate` from the
    /// caller) leaks the instance into `instances` when the caller's
    /// future is dropped by an outer cancellation (e.g. tower's
    /// `TimeoutLayer` firing). The leaked entry holds the wasmtime
    /// `Store` and counts against `max_instances`, but is unreachable
    /// by id (the caller lost the handle). This wrapper installs an
    /// `AutoTerminateGuard`: on the normal exit paths it disarms the
    /// guard and calls `terminate` via the async API; on a Future-drop
    /// the guard's `Drop` synchronously removes the registry entry
    /// (and frees the admission slot) so the leak window closes at
    /// the await boundary.
    ///
    /// Known limitation: this wrapper cannot stop CPU work that is
    /// already running inside Wasmtime's blocking compile path
    /// (`Instance::new_async` invokes Cranelift, which does not expose
    /// a cancellation hook). For now, the cap on
    /// [`crate::engine::EngineConfig::max_module_cache_entries`]
    /// limits the worst case to one compile per unique module.
    /// Per-store epoch cancellation interrupts wasm execution at the
    /// next epoch tick, which is what closes the actual run-time
    /// window.
    pub async fn call_export_then_terminate(
        &self,
        id: InstanceId,
        export: &str,
    ) -> Result<(), ExecError> {
        // Unit-typed back-compat surface, mirrors [`Self::call_export`].
        self.call_export_with_args_then_terminate(id, export, &[])
            .await
            .map(|_| ())
    }

    /// Argument-aware sibling of [`Self::call_export_then_terminate`].
    ///
    /// Identical lifecycle / drop-guard semantics — auto-terminates on
    /// success, failure, and Future-drop — but routes through
    /// [`Self::call_export_with_args`] so callers receive the export's
    /// result list as a JSON array.
    pub async fn call_export_with_args_then_terminate(
        &self,
        id: InstanceId,
        export: &str,
        args: &[WasmArg],
    ) -> Result<serde_json::Value, ExecError> {
        let guard = AutoTerminateGuard {
            instances: Arc::clone(&self.instances),
            instance_count: Arc::clone(&self.instance_count),
            metrics: self.metrics.clone(),
            id,
            // Re-arm on construction; only the success/error path below
            // is allowed to disarm.
            armed: true,
        };
        let result = self.call_export_with_args(id, export, args).await;
        // Disarm BEFORE the async terminate so a panic in `terminate`
        // does not double-fire. Both paths still remove the instance
        // exactly once: the guard via the sync DashMap::remove if it
        // is armed, the explicit terminate via the same DashMap::remove
        // when the guard is disarmed.
        let mut guard = guard;
        guard.armed = false;
        let _ = self.terminate(id).await; // ignore NotFound on success-after-cancel races
        result
    }
}

/// RAII drop-guard that synchronously removes an instance from the
/// registry if it is still armed when dropped. See
/// [`TensorWasmExecutor::call_export_then_terminate`] for the threat
/// model that motivates the design.
///
/// Holds `Arc` clones of the registry and the admission counter so
/// the guard can run without borrowing the executor — which matters
/// because the original `&self` reference is consumed by the
/// `call_export` await this guard wraps.
struct AutoTerminateGuard {
    instances: Arc<DashMap<InstanceId, Arc<Mutex<TensorWasmInstance>>>>,
    instance_count: Arc<AtomicUsize>,
    metrics: Option<TensorWasmMetrics>,
    id: InstanceId,
    armed: bool,
}

impl Drop for AutoTerminateGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Sync remove: we cannot await in Drop. The async `terminate`
        // method does exactly the same work plus a debug! log, so this
        // is a faithful sync mirror.
        if self.instances.remove(&self.id).is_some() {
            self.instance_count.fetch_sub(1, Ordering::AcqRel);
            if let Some(m) = &self.metrics {
                m.instance_terminations_total().inc();
                m.active_instances().dec();
            }
            tracing::warn!(
                target: "tensor_wasm_exec::executor",
                instance = %self.id,
                "instance auto-terminated by drop-guard (handler future cancelled \
                 mid-call_export; see api S-20)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_wasm() -> Vec<u8> {
        wat::parse_str(r#"(module (func (export "noop")))"#).unwrap()
    }

    #[tokio::test]
    async fn spawn_then_terminate() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        assert_eq!(exec.live_count(), 1);
        exec.call_export(id, "noop").await.unwrap();
        exec.terminate(id).await.unwrap();
        assert_eq!(exec.live_count(), 0);
    }

    #[tokio::test]
    async fn missing_export() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        let err = exec.call_export(id, "does_not_exist").await.unwrap_err();
        assert!(matches!(err, ExecError::MissingExport(_)));
    }

    #[tokio::test]
    async fn terminate_unknown() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let err = exec.terminate(InstanceId(999)).await.unwrap_err();
        assert!(matches!(err, ExecError::NotFound(_)));
    }

    #[tokio::test]
    async fn metrics_increment_on_spawn_and_terminate() {
        use tensor_wasm_core::metrics::TensorWasmMetrics;
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let metrics = TensorWasmMetrics::new();
        let exec = TensorWasmExecutor::with_metrics(engine, metrics.clone());
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        let text = metrics.encode_text();
        assert!(
            text.contains("tensor_wasm_instance_spawns_total 1"),
            "got:\n{text}"
        );
        assert!(text.contains("tensor_wasm_active_instances 1"), "got:\n{text}");
        exec.terminate(id).await.unwrap();
        let text = metrics.encode_text();
        assert!(
            text.contains("tensor_wasm_instance_terminations_total 1"),
            "got:\n{text}"
        );
        assert!(text.contains("tensor_wasm_active_instances 0"), "got:\n{text}");
    }

    #[test]
    fn exec_error_converts_to_tensor_wasm_error() {
        use tensor_wasm_core::error::TensorWasmError;
        let e = ExecError::NotFound(InstanceId(99));
        let b: TensorWasmError = e.into();
        assert!(matches!(b, TensorWasmError::Serialization(_)));
        assert!(b.to_string().contains("instance not found"));

        let e = ExecError::Timeout(TimeoutContext {
            id: InstanceId(1),
            elapsed_ms: 150,
            deadline_ms: 100,
        });
        let b: TensorWasmError = e.into();
        match b {
            TensorWasmError::KernelTimeout {
                elapsed_ms,
                deadline_ms,
            } => {
                assert_eq!(elapsed_ms, 150);
                assert_eq!(deadline_ms, 100);
            }
            other => panic!("expected KernelTimeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn module_cache_reuses_compilation() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let wasm = trivial_wasm();
        let _id1 = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
            .await
            .unwrap();
        let _id2 = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(2)), &wasm)
            .await
            .unwrap();
        // Both spawns hit the same wasm bytes — the cache should hold one entry.
        assert_eq!(exec.cached_module_count(), 1);
    }

    #[test]
    fn resource_limiter_allows_under_cap() {
        let mut lim = TensorWasmResourceLimiter::new(2 * 1024 * 1024);
        assert!(lim.memory_growing(0, 1024 * 1024, None).unwrap());
    }

    #[test]
    fn resource_limiter_rejects_over_cap() {
        let mut lim = TensorWasmResourceLimiter::new(1024 * 1024);
        assert!(!lim.memory_growing(0, 2 * 1024 * 1024, None).unwrap());
    }

    #[test]
    fn resource_limiter_respects_module_maximum() {
        let mut lim = TensorWasmResourceLimiter::new(usize::MAX);
        // Even if the engine cap is unbounded, the module's declared max wins.
        assert!(!lim
            .memory_growing(0, 4096, Some(2048))
            .unwrap());
    }

    #[test]
    fn resource_limiter_rejects_huge_table_growth() {
        // 1 MiB engine cap → at 16 B/entry that's ~65k table entries max.
        // u32::MAX (~4.3 billion entries × 16 B = ~64 GiB) must be denied.
        let mut lim = TensorWasmResourceLimiter::new(1024 * 1024);
        assert!(!lim.table_growing(0, u32::MAX, None).unwrap());
    }

    #[test]
    fn resource_limiter_allows_modest_table_growth() {
        // 1 MiB engine cap → ~65k entries should still fit.
        let mut lim = TensorWasmResourceLimiter::new(1024 * 1024);
        assert!(lim.table_growing(0, 1024, None).unwrap());
    }

    #[test]
    fn resource_limiter_table_respects_module_maximum() {
        // Even with an unbounded engine cap, the module's declared table max wins.
        let mut lim = TensorWasmResourceLimiter::new(usize::MAX);
        assert!(!lim.table_growing(0, 4096, Some(2048)).unwrap());
    }
}
