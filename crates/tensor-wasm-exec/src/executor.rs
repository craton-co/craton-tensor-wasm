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

use dashmap::{mapref::entry::Entry, DashMap};
use lru::LruCache;
use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_jit::cache::KernelCache;
use tensor_wasm_jit::rewrite::{rewrite_wasm, RewriteOptions};
use thiserror::Error;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, instrument, warn};
use wasmtime::{ExternType, Module, ResourceLimiter, Store, UpdateDeadline, Val};

use crate::engine::TensorWasmEngine;
use crate::instance::{InstanceState, TensorWasmInstance};
use crate::instance_pool::{InstancePool, ModuleHash};
use crate::jit_dispatch::add_jit_dispatch_to_linker;
use tensor_wasm_wasi_gpu::scheduler::add_scheduler_to_linker;
use tensor_wasm_wasi_gpu::streaming::{
    add_input_to_linker, add_streaming_to_linker, InputContext, StreamingContext,
};

/// Convert a wall-clock [`Duration`] into a number of epoch ticks suitable
/// for [`wasmtime::Store::set_epoch_deadline`].
///
/// Rounds up so a sub-tick remainder still terminates, with a floor of 1 so
/// `Duration::from_nanos(1)` does not silently become "never trip". Clamps
/// at [`u64::MAX`] if a caller supplies a pathologically long deadline. The
/// `tick` parameter is the engine's `epoch_tick` cadence; a zero-or-less tick
/// is treated as 1 ms to avoid division-by-zero on a malformed config.
///
/// # Granularity
///
/// Both the deadline and the tick are measured in **whole milliseconds**
/// (`Duration::as_millis`). Sub-millisecond resolution is therefore clamped:
/// a deadline of `Duration::from_micros(500)` and one of
/// `Duration::from_micros(1)` both round up to a single tick. This is
/// intentional — the epoch interrupt itself only fires on the background
/// ticker's cadence (10 ms by default), so sub-ms precision in the deadline
/// would be illusory anyway. Callers needing finer-grained interruption must
/// shorten [`crate::engine::EngineConfig::epoch_tick`], not the deadline.
///
/// The result is clamped to [`MAX_EPOCH_DEADLINE_TICKS`] (not `u64::MAX`)
/// because `set_epoch_deadline` is *relative*: wasmtime computes
/// `current_epoch + ticks`, so a `u64::MAX` result overflows once the
/// background ticker has advanced the epoch.
fn duration_to_epoch_ticks(d: Duration, tick: Duration) -> u64 {
    let d_ms = d.as_millis();
    let t_ms = tick.as_millis().max(1);
    let ticks_u128 = d_ms.div_ceil(t_ms).max(1);
    u64::try_from(ticks_u128)
        .unwrap_or(MAX_EPOCH_DEADLINE_TICKS)
        .min(MAX_EPOCH_DEADLINE_TICKS)
}

/// Install the COOPERATIVE epoch-deadline scheme on `store`, arming the first
/// deadline `first_ticks` ticks out.
///
/// This replaces the historical trap-only configuration (`set_epoch_deadline`
/// alone, which leaves the store in wasmtime's default
/// [`Store::epoch_deadline_trap`](wasmtime::Store::epoch_deadline_trap) mode).
/// Under trap mode a compute-bound guest never returns `Pending`: the
/// `call_async` poll blocks the worker until the epoch traps, so an outer
/// `timeout` / future-drop cannot cancel it cooperatively and a single-thread
/// runtime can be wedged by a spinning guest. Cooperative yielding fixes both.
///
/// The scheme is two-level:
///
///   * **Cooperative yield (between yields the guest is cancellable).** The
///     store is configured via
///     [`Store::epoch_deadline_callback`](wasmtime::Store::epoch_deadline_callback).
///     Each time the epoch deadline trips, the callback returns
///     [`UpdateDeadline::Yield`] with [`COOPERATIVE_YIELD_TICKS`], so the guest
///     yields `Pending` to the async executor and the deadline re-arms for
///     another window. While parked at that yield, the surrounding
///     `call_async` future can be dropped — which terminates the guest fiber
///     promptly (wasmtime turns a cancel-while-yielded into a trap on the
///     unwinding fiber). This is exactly the property the
///     `orphan_cleanup_on_drop` test needs.
///
///   * **Hard deadline (still TRAPs a runaway guest).** Before yielding, the
///     callback consults the per-store
///     [`InstanceState::hard_deadline`](crate::instance::InstanceState::hard_deadline).
///     Once that absolute wall-clock instant has elapsed the callback returns
///     `Err(..)` instead of yielding, which wasmtime converts into a trap —
///     bit-for-bit the same termination the old `set_epoch_deadline` trap
///     produced. The executor sets `hard_deadline` to the per-call deadline
///     (and, during instantiation, to a value additionally clamped by
///     [`MAX_START_FN_DURATION`]), so the public timeout contract is preserved:
///     `call_export_with_args` still classifies the resulting error as
///     [`ExecError::Timeout`] because the wall clock has crossed `deadline`.
///
/// The error message intentionally matches the spirit of a wasmtime epoch
/// trap ("epoch deadline reached"); the executor never surfaces this string
/// to callers — it is reclassified into the typed [`ExecError::Timeout`] (or,
/// for the no-deadline start-fn cap, propagated as `ExecError::Wasmtime`,
/// exactly as a trap did before).
fn arm_cooperative_epoch(store: &mut Store<InstanceState>, first_ticks: u64) {
    // Arm the FIRST deadline relative to the current epoch. Subsequent
    // re-arms are driven by the value the callback returns.
    store.set_epoch_deadline(first_ticks);
    store.epoch_deadline_callback(|ctx| {
        if ctx.data().is_cancelled() {
            // HIGH finding 1: `terminate` flipped the cooperative-cancellation
            // flag (and set `hard_deadline = now`). Trap the guest at this
            // epoch tick so an in-flight, possibly deadline-less call is
            // interrupted instead of running until it returns of its own
            // accord. The error string mirrors the hard-deadline trap; the
            // surrounding `call_export_with_args` propagates it as
            // `ExecError::Wasmtime` (no deadline crossed, so it is not
            // reclassified as a timeout).
            Err(wasmtime::Error::msg("instance terminated"))
        } else if ctx.data().hard_deadline_elapsed() {
            // HARD deadline crossed: trap, terminating the guest. Mirrors the
            // historical `epoch_deadline_trap` behaviour. The executor's
            // timeout classification keys off the wall clock, not this string.
            Err(wasmtime::Error::msg("epoch deadline reached"))
        } else {
            // Within the hard deadline: yield `Pending` so the guest is
            // cancellable, then re-arm for another cooperative window.
            Ok(UpdateDeadline::Yield(COOPERATIVE_YIELD_TICKS))
        }
    });
}

/// Overflow-safe sentinel for "effectively no deadline", in epoch ticks.
///
/// [`wasmtime::Store::set_epoch_deadline`] takes a deadline **relative** to
/// the current epoch — internally `current_epoch + ticks`. A `u64::MAX`
/// sentinel therefore overflows the moment the background epoch ticker has
/// advanced the counter past 0: in a debug build that is an `attempt to add
/// with overflow` panic inside wasmtime; in release (overflow checks off) it
/// wraps to a deadline *in the past*, so the guest is interrupted
/// immediately. `u64::MAX / 2` is effectively infinite (≈9.2e18 ticks —
/// millions of years at any realistic cadence) while guaranteeing
/// `current_epoch + MAX_EPOCH_DEADLINE_TICKS` cannot overflow for the life of
/// the process.
const MAX_EPOCH_DEADLINE_TICKS: u64 = u64::MAX / 2;

/// Number of epoch ticks between cooperative yields under the async-yield
/// epoch scheme.
///
/// The executor configures every store with
/// [`wasmtime::Store::epoch_deadline_callback`] (see
/// [`TensorWasmExecutor::arm_cooperative_epoch`]) so that — instead of trapping
/// the instant the epoch deadline trips — a compute-bound guest *yields*
/// `Pending` back to the async runtime and the deadline is re-armed for
/// another `COOPERATIVE_YIELD_TICKS` window. Yielding is what makes a
/// runaway guest *cancellable*: while it is parked at a yield point the
/// surrounding `call_async` future can be dropped (by an outer `timeout`
/// or task cancellation), which terminates the guest promptly instead of
/// blocking the worker until a trap fires. The HARD deadline is still
/// enforced — on each yield the callback first checks the per-store
/// [`InstanceState::hard_deadline`](crate::instance::InstanceState::hard_deadline)
/// and traps once it has elapsed (see `arm_cooperative_epoch`).
///
/// One tick (the engine's `epoch_tick`, 10 ms by default) keeps yields
/// frequent enough that future-drop cancellation is observed within a
/// single tick of cadence, while staying coarse enough that a well-behaved
/// guest doing real work is not penalised with excessive yield churn. The
/// background epoch ticker ([`crate::engine::TensorWasmEngine::spawn_epoch_ticker`])
/// must be running for the deadline to advance at all — the same
/// pre-existing requirement as the trap scheme, enforced by the
/// [`ExecError::EpochTickerNotRunning`] admission check.
const COOPERATIVE_YIELD_TICKS: u64 = 1;

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

/// Hard upper bound on the byte length of a Wasm module the executor will
/// accept for compilation. Modules above this size are rejected with
/// [`ExecError::ModuleTooLarge`] *before* `Module::from_binary` runs —
/// pathological code-section blow-ups can otherwise force Cranelift to
/// burn arbitrary CPU on adversarial input. 64 MiB is comfortably above
/// any legitimate ML kernel module we've seen (single-digit MiB is
/// typical), while keeping the Cranelift worst case bounded.
///
/// This constant is the floor: embedders can tighten via
/// [`EngineConfig::max_module_bytes`](crate::engine::EngineConfig::max_module_bytes).
pub const MAX_MODULE_BYTES: usize = 64 * 1024 * 1024;

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
    /// The executor refused to admit a new instance because an
    /// admission-control ceiling is already saturated. Surfaced for two
    /// rejections that the API layer treats identically (503, retryable):
    ///
    ///   * the engine-wide live-instance ceiling
    ///     ([`crate::engine::EngineConfig::max_instances`], exec S-10); or
    ///   * the spawning tenant's per-tenant fairness cap
    ///     ([`crate::engine::EngineConfig::max_instances_per_tenant`]), even
    ///     though the engine-wide ceiling may still have headroom. One tenant
    ///     hitting its per-tenant cap is refused here WITHOUT affecting any
    ///     other tenant's ability to spawn.
    ///
    /// Surfaced from [`TensorWasmExecutor::spawn_instance`] *before* any
    /// compile / instantiate work; the failed spawn never consumes a slot
    /// (engine-wide or per-tenant). Mapped to
    /// [`tensor_wasm_core::error::TensorWasmError::MemoryExhausted`] on the
    /// conversion boundary (the API layer surfaces it as 503).
    ///
    /// This is the *engine-wide* ceiling: the shared `max_instances` budget is
    /// saturated and the spawn would succeed once aggregate load drops. The
    /// remedy is retry-with-backoff and it is no single tenant's fault, so the
    /// API layer maps it to 503. The per-tenant fairness cap is a distinct
    /// condition — see [`ExecError::TenantCapacityExhausted`].
    #[error("instance capacity exhausted: {active} active, limit {limit}")]
    CapacityExhausted {
        /// Engine-wide live-instance count observed at the rejection point
        /// (post-increment, so `active > limit`).
        active: usize,
        /// Configured engine-wide `max_instances` ceiling that was exceeded.
        limit: usize,
    },
    /// A single tenant exceeded its per-tenant fairness cap
    /// ([`crate::engine::EngineConfig::max_instances_per_tenant`]) while the
    /// shared engine-wide budget still had room. This is semantically
    /// distinct from [`ExecError::CapacityExhausted`]: the offending tenant is
    /// over *its own* quota, so the corrective action is for that tenant to
    /// reduce concurrency — not to wait for global load to drop. The API layer
    /// maps it to 429 (`tenant_capacity_exhausted`) rather than 503, giving
    /// callers a quota-specific retry signal. Other tenants are unaffected.
    #[error("tenant {tenant} instance capacity exhausted: {active} active, limit {limit}")]
    TenantCapacityExhausted {
        /// The tenant whose per-tenant cap was hit (the spawn's owner).
        tenant: TenantId,
        /// That tenant's live-instance count at the rejection point
        /// (post-increment, so `active > limit`).
        active: usize,
        /// The configured per-tenant `max_instances_per_tenant` ceiling.
        limit: usize,
    },
    /// The submitted Wasm module is larger than the configured
    /// pre-compile size cap
    /// ([`crate::engine::EngineConfig::max_module_bytes`], floored at
    /// [`MAX_MODULE_BYTES`]).
    ///
    /// Rejected before `Module::from_binary` runs so a pathological
    /// code section cannot force Cranelift to burn CPU on adversarial
    /// input. Mapped to
    /// [`tensor_wasm_core::error::TensorWasmError::MemoryExhausted`] on
    /// the conversion boundary (the API layer surfaces it as 503).
    #[error("wasm module byte length {len} exceeds cap {max}")]
    ModuleTooLarge {
        /// Length of the rejected wasm blob, in bytes.
        len: usize,
        /// Configured per-executor cap, in bytes.
        max: usize,
    },
    /// `spawn_instance` was called with a deadline configured but the
    /// engine's epoch ticker is not running. Without the ticker, the
    /// per-store epoch counter never advances, so neither the per-call
    /// deadline nor [`MAX_START_FN_DURATION`] can fire — a runaway
    /// guest would wedge the worker thread until it returned of its
    /// own accord. Refuse the spawn instead of silently dropping the
    /// deadline contract.
    #[error("epoch ticker not running — refusing spawn with deadline; call `engine.spawn_epoch_ticker()` first")]
    EpochTickerNotRunning,
    /// A guest export returned a value type the executor cannot represent
    /// in its JSON result projection (LOW finding).
    ///
    /// `call_export_with_args` projects each returned [`wasmtime::Val`] into a
    /// [`serde_json::Value`]: the four numeric core types (`i32`/`i64`/
    /// `f32`/`f64`) map to JSON numbers, but `v128` and the reference types
    /// (`funcref`/`externref`/`anyref`) have no faithful JSON encoding.
    /// Previously they were silently collapsed to JSON `null`, masking a
    /// genuine signature mismatch (a guest returning a `v128` looked like a
    /// guest returning nothing). We now fail with this typed error instead so
    /// the caller learns the export's return type is unsupported rather than
    /// receiving a misleading `null`. Carries the offending value-type name
    /// (e.g. `"v128"`) for diagnostics.
    #[error("export returned an unsupported value type: {0}")]
    UnsupportedReturnType(&'static str),
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
            // The per-tenant fairness cap collapses to the same resource-
            // exhaustion shape on the native `TensorWasmError` boundary (the
            // tenant distinction is preserved on the richer API-layer mapping,
            // not here). Tenant id is dropped — `TensorWasmError` has no field
            // for it and it is logged at the rejection site.
            ExecError::TenantCapacityExhausted { active, limit, .. } => {
                TensorWasmError::MemoryExhausted {
                    requested: active as u64,
                    limit: limit as u64,
                }
            }
            ExecError::ModuleTooLarge { len, max } => TensorWasmError::MemoryExhausted {
                requested: len as u64,
                limit: max as u64,
            },
            ExecError::EpochTickerNotRunning => {
                // Surface as a compile-class failure: the spawn never
                // got off the ground, no guest code executed, and the
                // remedy is operational (start the ticker) rather than
                // anything the caller can retry.
                TensorWasmError::WasmCompile("epoch ticker not running".into())
            }
            ExecError::UnsupportedReturnType(kind) => {
                // The guest ran fine but returned a value the host cannot
                // project into JSON. It is a contract mismatch between the
                // export's signature and this transport, not a trap or a
                // resource failure — surface it as a serialization-class
                // error carrying the offending type name.
                TensorWasmError::Serialization(
                    format!("unsupported export return type: {kind}").into(),
                )
            }
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
    /// Optional streaming context (roadmap feature #2). When `Some`,
    /// [`TensorWasmExecutor::spawn_instance`] builds a wasmtime
    /// [`wasmtime::Linker`] that wires the `wasi:tensor/host` host
    /// functions (`emit-chunk`, `flush`) against this context — guest
    /// emits land on the matching `mpsc::Receiver<Vec<u8>>` the
    /// gateway is draining into the SSE / chunked HTTP response.
    ///
    /// `None` means streaming is disabled: the spawn uses the
    /// historical empty-imports `Instance::new_async` path, and a
    /// guest that imports `wasi:tensor/host` will fail to link.
    /// `/invoke` (the synchronous route) takes the `None` path; only
    /// `/invoke-stream` opts in.
    pub streaming: Option<StreamingContext>,
    /// Bytes staged for the guest to pull via the `wasi:tensor/host`
    /// pull-model input channel (`input-len` / `read-input`).
    ///
    /// Empty by default (the historical behaviour: the guest has no
    /// input channel). When non-empty,
    /// [`TensorWasmExecutor::spawn_instance`] installs an
    /// [`InputContext`] on
    /// the per-instance state and registers the `input-len` /
    /// `read-input` host functions on the spawn linker — so a guest can
    /// copy these bytes into its own linear memory at the start of the
    /// invocation. The OpenAI completions shim sets this to the assembled
    /// prompt bytes.
    pub input: Vec<u8>,
}

impl SpawnConfig {
    /// Construct with just a tenant and no deadline.
    pub fn for_tenant(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            deadline: None,
            args: Vec::new(),
            streaming: None,
            input: Vec::new(),
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

    /// Attach a streaming context (roadmap feature #2). See
    /// [`SpawnConfig::streaming`].
    ///
    /// Builder method; pairs with [`Self::for_tenant`] /
    /// [`Self::with_deadline`]. The API gateway's `/invoke-stream`
    /// route constructs an `mpsc::channel`, wraps the sender in a
    /// `StreamingContext` via [`StreamingContext::with_channel`], and
    /// passes it here so the guest's `wasi:tensor/host.emit-chunk`
    /// calls land on the matching receiver — which the gateway
    /// concurrently drains into the SSE / chunked response body.
    pub fn with_streaming(mut self, ctx: StreamingContext) -> Self {
        self.streaming = Some(ctx);
        self
    }

    /// Stage `input` bytes for the guest to pull via the
    /// `wasi:tensor/host` input channel (`input-len` / `read-input`). See
    /// [`SpawnConfig::input`].
    ///
    /// Builder method; pairs with [`Self::for_tenant`] /
    /// [`Self::with_deadline`] / [`Self::with_streaming`]. An empty slice
    /// is a no-op (the default) — the guest then observes `input-len() ==
    /// 0`. The OpenAI completions shim passes the assembled prompt bytes
    /// here so the guest receives the prompt.
    pub fn with_input(mut self, input: Vec<u8>) -> Self {
        self.input = input;
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
/// numbers (floating-point). Other value types — `v128` and the reference
/// types — have no faithful JSON encoding, so rather than silently collapse
/// them to `null` (LOW finding: that masked a genuine return-type mismatch —
/// a `v128` return looked identical to a void return) we fail with a typed
/// [`ExecError::UnsupportedReturnType`] carrying the offending type name.
/// Used by [`TensorWasmExecutor::call_export_with_args`] to project the
/// wasmtime result slice into a JSON array.
fn val_to_json(v: &Val) -> Result<serde_json::Value, ExecError> {
    match v {
        Val::I32(n) => Ok(serde_json::json!(*n)),
        Val::I64(n) => Ok(serde_json::json!(*n)),
        Val::F32(bits) => Ok(serde_json::json!(f32::from_bits(*bits))),
        Val::F64(bits) => Ok(serde_json::json!(f64::from_bits(*bits))),
        // No faithful JSON encoding for these. `Val::ty()` needs a store
        // handle we don't thread here, so describe the variant directly —
        // enough for a caller to tell a `v128` apart from a reference type.
        other => {
            let kind = match other {
                Val::V128(_) => "v128",
                Val::FuncRef(_) => "funcref",
                Val::ExternRef(_) => "externref",
                Val::AnyRef(_) => "anyref",
                _ => "unknown",
            };
            debug!(
                target: "tensor_wasm_exec::executor",
                val_kind = kind,
                "export returned a non-numeric value type with no JSON encoding; \
                 surfacing UnsupportedReturnType instead of a misleading null",
            );
            Err(ExecError::UnsupportedReturnType(kind))
        }
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
        _current: usize,
        desired: usize,
        maximum: Option<usize>,
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
        // Shared with the pooling allocator's `table_elements` derivation in
        // `engine.rs` (MED finding) so both backends budget tables against
        // the same per-instance byte ceiling.
        let bytes_needed = (desired as u64).saturating_mul(crate::engine::TABLE_ENTRY_BYTES);
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

/// Registry value held against each live [`InstanceId`].
///
/// Bundles the per-instance `Arc<Mutex<TensorWasmInstance>>` with three
/// pieces of metadata cached at register time so the hot lifecycle paths
/// (`terminate`, `detach_pooled_instance`, the per-call result projection)
/// never have to take the per-instance mutex just to read them:
///
///   * `tenant_id` (PERF finding 6): `terminate` / `detach` decrement the
///     per-tenant fairness counter from a plain field instead of locking the
///     mutex — which an in-flight call may be holding across `call_async`.
///   * `cancelled` (HIGH finding 1): the shared cooperative-cancellation
///     flag the epoch callback consults. `terminate` flips it WITHOUT the
///     mutex so a running, deadline-less call is trapped at the next epoch
///     tick rather than merely de-registered.
///   * `export_arity` (PERF finding 6): per-export declared result counts,
///     cached at register time so `call_export_with_args` skips the
///     per-call `Func::ty` allocation on repeat invocations.
#[derive(Clone)]
struct RegistryEntry {
    /// The live instance, behind the per-instance serialising mutex.
    handle: Arc<Mutex<TensorWasmInstance>>,
    /// Owning tenant, cached so `terminate` / `detach` need no lock to
    /// decrement the per-tenant count (PERF finding 6).
    tenant_id: TenantId,
    /// Cooperative-cancellation handle shared with the per-store
    /// [`InstanceState::cancelled`]. `terminate` flips it lock-free
    /// (HIGH finding 1).
    cancelled: Arc<AtomicBool>,
    /// Per-export declared result arity, populated lazily on first call and
    /// reused thereafter so the typed/dynamic branch in
    /// `call_export_with_args` avoids the per-call `Func::ty` allocation
    /// (PERF finding 6).
    export_arity: Arc<dashmap::DashMap<String, usize>>,
}

/// The async executor.
#[derive(Clone)]
pub struct TensorWasmExecutor {
    engine: Arc<TensorWasmEngine>,
    instances: Arc<DashMap<InstanceId, RegistryEntry>>,
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
    /// Per-tenant live-instance counts, used to enforce
    /// [`crate::engine::EngineConfig::max_instances_per_tenant`] (fairness
    /// bound). Keyed by the spawning [`TenantId`]; the value is the number
    /// of slots that tenant currently holds. Bumped (with rollback on a
    /// failed spawn) alongside the engine-wide `instance_count` in
    /// [`Self::charge_instance_slot`] and decremented in [`Self::terminate`]
    /// / [`Self::release_instance_slot`]. Empty entries are pruned on
    /// decrement so a tenant that churns instances does not leak map keys.
    /// Independent of `instance_count` — the engine-wide cap and the
    /// per-tenant cap are checked in the same admission step but tracked
    /// separately so one tenant hitting its cap never perturbs another's
    /// accounting.
    tenant_counts: Arc<DashMap<TenantId, usize>>,
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
    /// Optional pre-instantiated instance pool (roadmap feature #5).
    ///
    /// When set, [`Self::invoke`] routes through
    /// [`crate::instance_pool::InstancePool::acquire`] /
    /// [`InstancePool::release`]: `acquire` draws a warm instance from the
    /// per-`(tenant, module-hash)` channel (wait-free `try_recv`) and only
    /// falls through to [`Self::spawn_instance`] on a cold/empty channel or a
    /// streaming/input carve-out; `release` re-instantiates from the cached
    /// [`wasmtime::Module`] and returns a fresh instance to the channel. The
    /// warm path is live — the historical "scaffold that falls through on
    /// every call" doc was stale (DOCS finding). Executors constructed without
    /// [`Self::with_instance_pool`] see the pool-free path (every `invoke`
    /// spawns + calls + terminates).
    pool: Option<Arc<InstancePool>>,
    /// Optional JIT kernel cache. When set, `instantiate_detached` registers
    /// the `tensor-wasm:jit/host` `dispatch`/`alloc`/`free` imports
    /// (`jit_dispatch::add_jit_dispatch_to_linker`) on every spawn's linker
    /// so auto-offloaded guests can reach the cached kernels. `None` (the
    /// default) leaves the JIT surface unlinked — a guest importing it then
    /// fails to link, matching the historical behaviour for embedders that
    /// have not opted in. The cache is intentionally per-executor and
    /// cross-tenant (lookups are tenant-scoped inside the dispatch closure
    /// via `CacheKey::for_tenant`); see `jit_dispatch.rs`.
    jit_cache: Option<Arc<KernelCache>>,
    /// Bounds the number of concurrent `Module::from_binary` compiles on the
    /// Tokio blocking pool (MEDIUM finding). A permit is acquired around the
    /// `spawn_blocking` compile in [`Self::compile_module_cached`]; the
    /// permit is released as soon as the compile resolves (cache hits never
    /// touch the semaphore). Capacity comes from
    /// [`crate::engine::EngineConfig::max_concurrent_compiles`], defaulting
    /// to [`std::thread::available_parallelism`] (floored at 1). Independent
    /// of `max_instances` — that caps live instances, this caps in-flight
    /// Cranelift work.
    compile_semaphore: Arc<Semaphore>,
    /// Per-digest single-flight guard for [`Self::compile_module_cached`]
    /// (PERF finding 6).
    ///
    /// Before this, two concurrent first-spawns of the SAME fresh module both
    /// missed the `module_cache` read and each ran a full Cranelift compile —
    /// the loser's `put` simply overwrote the winner's, so the duplicate
    /// compile was wasted CPU (and, under the compile semaphore, a wasted
    /// permit that could starve a distinct module's compile). A per-digest
    /// `OnceCell` collapses those into one compile: the winner runs it; every
    /// concurrent caller for the same digest parks on `get_or_try_init` and
    /// clones the winner's [`Module`]. Cache hits never touch this map; the
    /// cell is removed once resolved so the map does not grow unbounded.
    /// Mirrors the pool's per-`PoolKey` single-flight discipline in
    /// `instance_pool.rs`.
    inflight_compiles: Arc<DashMap<[u8; 32], Arc<tokio::sync::OnceCell<Module>>>>,
}

/// Resolve the concurrent-compile permit count from the engine config,
/// falling back to the host parallelism (floored at 1) when the operator
/// left [`crate::engine::EngineConfig::max_concurrent_compiles`] unset.
/// A configured value of 0 is coerced to 1 (a 0-permit semaphore would
/// deadlock every compile).
fn resolve_compile_permits(requested: Option<usize>) -> usize {
    match requested {
        Some(0) | None => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1),
        Some(n) => n,
    }
    .max(1)
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
    /// Per-tenant rollback handle. `Some` carries the per-tenant count map
    /// plus the tenant whose count was bumped alongside the engine-wide
    /// `counter`; on a non-committed drop both the engine-wide and the
    /// per-tenant count are rolled back in lockstep. `None` means no
    /// per-tenant charge was made (per-tenant cap disabled, or this guard
    /// only re-protects the engine-wide count — e.g. the post-charge
    /// register guard in `spawn_instance`).
    tenant_rollback: Option<(Arc<DashMap<TenantId, usize>>, TenantId)>,
    committed: bool,
}

impl InstanceSlotGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        Self {
            counter,
            tenant_rollback: None,
            committed: false,
        }
    }

    /// Construct a guard that, on a non-committed drop, rolls back BOTH the
    /// engine-wide count and the `tenant`'s per-tenant count. Used by
    /// [`TensorWasmExecutor::charge_instance_slot`] when a per-tenant cap is
    /// configured so a failed spawn never leaks either counter.
    fn with_tenant(
        counter: Arc<AtomicUsize>,
        tenant_counts: Arc<DashMap<TenantId, usize>>,
        tenant: TenantId,
    ) -> Self {
        Self {
            counter,
            tenant_rollback: Some((tenant_counts, tenant)),
            committed: false,
        }
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
            if let Some((tenant_counts, tenant)) = &self.tenant_rollback {
                decrement_tenant_count(tenant_counts, *tenant);
            }
        }
    }
}

/// Decrement (and prune-on-zero) a tenant's per-tenant live-instance count.
/// Shared by [`InstanceSlotGuard`]'s rollback, [`TensorWasmExecutor::terminate`],
/// and [`TensorWasmExecutor::release_instance_slot`] so the prune-empty-entry
/// policy lives in exactly one place. A `None`/zero entry is a no-op (a
/// double-decrement cannot drive the count negative).
fn decrement_tenant_count(tenant_counts: &DashMap<TenantId, usize>, tenant: TenantId) {
    if let Entry::Occupied(mut e) = tenant_counts.entry(tenant) {
        let v = e.get_mut();
        *v = v.saturating_sub(1);
        if *v == 0 {
            e.remove();
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
    let check = |mt: &wasmtime::MemoryType| -> Result<(), ExecError> {
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

/// Pre-compile sibling of [`check_module_memory_within_cap`] that walks the
/// module's declared linear-memory types directly from the raw Wasm bytes via
/// [`wasmparser`], *before* [`Module::new`] runs.
///
/// `check_module_memory_within_cap` needs a compiled [`Module`], but under the
/// pooling allocator a module declaring an oversized *initial* memory is
/// rejected by wasmtime at compile time — so that post-compile walk never runs
/// for the initial-size case and the caller sees an opaque
/// [`ExecError::Wasmtime`] instead of the structured
/// [`ExecError::ModuleMemoryTooLarge`]. Running the identical cap arithmetic on
/// the raw bytes up-front makes the rejection backend-independent and preserves
/// the precise requested-byte figure regardless of the configured allocator.
///
/// Parse/validation errors are deliberately *not* surfaced here: a malformed
/// module is left for [`Module::new`] to reject with its richer diagnostics —
/// this pass only ever refines the memory-cap case, never swallows other
/// failures.
fn check_raw_module_memory_within_cap(wasm: &[u8], cap_bytes: usize) -> Result<(), ExecError> {
    use wasmparser::{Parser, Payload, TypeRef};
    let cap_u64 = cap_bytes as u64;
    let check = |mt: &wasmparser::MemoryType| -> Result<(), ExecError> {
        // Wasm linear-memory sizes are page counts; the default page size is
        // 64 KiB, overridable per-memory by the custom-page-sizes proposal
        // (`page_size_log2`). Saturating arithmetic so a pathological
        // declaration cannot wrap on the multiply.
        let page_size: u64 = 1u64 << u64::from(mt.page_size_log2.unwrap_or(16));
        let min_bytes = mt.initial.saturating_mul(page_size);
        if min_bytes > cap_u64 {
            return Err(ExecError::ModuleMemoryTooLarge {
                requested_bytes: min_bytes,
                limit_bytes: cap_u64,
            });
        }
        if let Some(max_pages) = mt.maximum {
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
    for payload in Parser::new(0).parse_all(wasm) {
        // Bail to `Module::new` on any parse hiccup — we only refine here.
        let Ok(payload) = payload else { return Ok(()) };
        match payload {
            Payload::MemorySection(reader) => {
                for mem in reader {
                    let Ok(mt) = mem else { return Ok(()) };
                    check(&mt)?;
                }
            }
            Payload::ImportSection(reader) => {
                for im in reader {
                    let Ok(im) = im else { return Ok(()) };
                    if let TypeRef::Memory(mt) = im.ty {
                        check(&mt)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Substring marker that, together with [`MEMORY_LIMIT_PHRASINGS`], identifies
/// a wasmtime pooling-allocator memory-slot sizing refusal in
/// [`classify_instantiation_error`].
///
/// Pulled out as a named constant (rather than an inline literal) so the
/// version-coupling is a single source of truth the
/// `tests/classify_error_phrasing_pinned.rs` pinned-phrasing test references
/// directly — if a wasmtime bump rewords these, the test (which builds a real
/// oversized-module error and asserts the refinement still fires) fails loudly
/// instead of the refinement silently degrading (LOW finding).
///
/// `pub` (not `pub(crate)`) only so that out-of-crate integration test — which
/// compiles as an external consumer and cannot see `pub(crate)` items — can pin
/// the exact same phrasings the classifier uses.
pub const MEMORY_FAILURE_MARKER: &str = "memory";

/// The limit/exceeds phrasings wasmtime's pooling `memory_pool` uses for a
/// memory-slot sizing refusal:
///   * "memory index N has a minimum byte size of M which exceeds the limit
///      of L bytes" (per-slot size refusal)
///   * "maximum memory size of 0x… bytes exceeds the configured maximum size"
///      (pool construction / sizing refusal)
///
/// See [`MEMORY_FAILURE_MARKER`] for why these live as named constants (and why
/// they are `pub`).
pub const MEMORY_LIMIT_PHRASINGS: &[&str] = &["exceeds the limit", "exceeds the configured"];

/// Best-effort refinement of a pooling-allocator instantiation error into a
/// typed [`ExecError::ModuleMemoryTooLarge`] (MED finding).
///
/// The pooling allocator surfaces a memory-slot sizing failure as an opaque
/// [`wasmtime::Error`]; its message chain contains the allocator's
/// memory-size signature (wasmtime phrases these as "memory ... exceeds the
/// limit" / "memory minimum size of N pages exceeds ..."). When we recognise
/// that signature we re-tag the error as `ModuleMemoryTooLarge` so callers
/// see a stable `MemoryExhausted` on the conversion boundary instead of a
/// generic compile error. Anything we do not recognise is returned verbatim
/// as [`ExecError::Wasmtime`] — we only ever *refine*, never swallow.
fn classify_instantiation_error(err: wasmtime::Error, cap_bytes: usize) -> ExecError {
    let chain = format!("{err:#}").to_ascii_lowercase();
    // LOW finding (fragile error classification): this is *substring* matching
    // on English error text because wasmtime does not expose a structured
    // error type for a pooling-allocator memory-slot sizing refusal. It is
    // therefore WASMTIME-VERSION-COUPLED — a wasmtime upgrade can silently
    // reword these phrasings and quietly stop the refinement. That degradation
    // is SAFE BY CONSTRUCTION: this function only ever *refines* a recognised
    // error into the typed `ModuleMemoryTooLarge`; every unrecognised error
    // falls through to the `else` branch below and is returned verbatim as
    // `ExecError::Wasmtime(err)` — the original error is never swallowed or
    // dropped, only (best-effort) re-tagged.
    //
    // The phrasings now live in the named `MEMORY_FAILURE_MARKER` /
    // `MEMORY_LIMIT_PHRASINGS` constants above (a single source of truth) so
    // the pinned-phrasing test (`tests/classify_error_phrasing_pinned.rs`)
    // references them directly: that test builds a REAL oversized-module
    // wasmtime error and asserts this refinement still fires, so a wasmtime
    // bump that rewords the messages fails CI loudly rather than letting the
    // refinement silently degrade. When bumping wasmtime, update the constants
    // (and the test will confirm the new phrasings match).
    //
    // Conservative match: require both a "memory" mention AND one of the
    // exceeds/limit phrasings so unrelated traps/link errors are not
    // misclassified. The pooling allocator's memory-size refusals all contain
    // one of these limit phrasings.
    let is_memory_size_failure = chain.contains(MEMORY_FAILURE_MARKER)
        && MEMORY_LIMIT_PHRASINGS.iter().any(|p| chain.contains(p));
    if is_memory_size_failure {
        ExecError::ModuleMemoryTooLarge {
            // We do not know the exact requested figure here (the allocator
            // does not expose it structurally), so report the cap as both
            // bounds — the typed variant's value is the classification, and
            // the full opaque chain is already logged server-side on the
            // conversion boundary.
            requested_bytes: cap_bytes as u64,
            limit_bytes: cap_bytes as u64,
        }
    } else {
        ExecError::Wasmtime(err)
    }
}

impl TensorWasmExecutor {
    /// Construct an executor over the given shared engine.
    pub fn new(engine: Arc<TensorWasmEngine>) -> Self {
        let cap = lru_cap(engine.config().max_module_cache_entries);
        let permits = resolve_compile_permits(engine.config().max_concurrent_compiles);
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(parking_lot::Mutex::new(LruCache::new(cap))),
            instance_count: Arc::new(AtomicUsize::new(0)),
            tenant_counts: Arc::new(DashMap::new()),
            metrics: None,
            ticker_warned: Arc::new(OnceLock::new()),
            pool: None,
            jit_cache: None,
            compile_semaphore: Arc::new(Semaphore::new(permits)),
            inflight_compiles: Arc::new(DashMap::new()),
        }
    }

    /// Construct an executor that publishes spawn/terminate events to the
    /// supplied [`TensorWasmMetrics`] registry. Metric handles are cheaply cloneable;
    /// pass a clone of the process-wide registry.
    pub fn with_metrics(engine: Arc<TensorWasmEngine>, metrics: TensorWasmMetrics) -> Self {
        let cap = lru_cap(engine.config().max_module_cache_entries);
        let permits = resolve_compile_permits(engine.config().max_concurrent_compiles);
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(parking_lot::Mutex::new(LruCache::new(cap))),
            instance_count: Arc::new(AtomicUsize::new(0)),
            tenant_counts: Arc::new(DashMap::new()),
            metrics: Some(metrics),
            ticker_warned: Arc::new(OnceLock::new()),
            pool: None,
            jit_cache: None,
            compile_semaphore: Arc::new(Semaphore::new(permits)),
            inflight_compiles: Arc::new(DashMap::new()),
        }
    }

    /// Attach an [`InstancePool`] to this executor and return the
    /// modified executor.
    ///
    /// Builder method; pairs with [`Self::new`] / [`Self::with_metrics`].
    /// The pool itself is opt-in — embedders that do not call this method get
    /// the pool-free path (every [`Self::invoke`] does a fresh
    /// spawn/compile/instantiate + call + terminate). With a pool attached,
    /// `invoke` draws warm instances from the per-`(tenant, module-hash)`
    /// channel and re-instantiates spent ones on release (DOCS finding: the
    /// warm path is live, not the historical "v0.3.6 scaffold" placeholder).
    pub fn with_instance_pool(mut self, pool: Arc<InstancePool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Attach a JIT [`KernelCache`] to this executor and return the modified
    /// executor.
    ///
    /// Builder method; pairs with [`Self::new`] / [`Self::with_metrics`].
    /// When set, every instantiation registers the `tensor-wasm:jit/host`
    /// `dispatch`/`alloc`/`free` imports (see
    /// [`crate::jit_dispatch::add_jit_dispatch_to_linker`]) against this
    /// cache, so auto-offloaded guests link successfully. Embedders that do
    /// not call this leave the JIT surface unlinked — a guest importing it
    /// then fails to link, matching the pre-wiring behaviour.
    pub fn with_jit_cache(mut self, cache: Arc<KernelCache>) -> Self {
        self.jit_cache = Some(cache);
        self
    }

    /// Borrow the attached JIT [`KernelCache`], if any. Returns `None` for
    /// executors constructed without [`Self::with_jit_cache`].
    pub fn jit_cache(&self) -> Option<&Arc<KernelCache>> {
        self.jit_cache.as_ref()
    }

    /// Borrow the attached [`InstancePool`], if any. Returns `None` for
    /// executors constructed without [`Self::with_instance_pool`].
    pub fn instance_pool(&self) -> Option<&Arc<InstancePool>> {
        self.pool.as_ref()
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &TensorWasmEngine {
        &self.engine
    }

    /// Number of instances currently present in the **registry** — i.e.
    /// those that have been spawned and `register_pooled_instance`'d but
    /// not yet `terminate`'d. This is the size of the `instances` `DashMap`.
    ///
    /// Contrast with [`Self::instances_len`], which reports the
    /// **admission** count (the atomic counter that
    /// [`crate::engine::EngineConfig::max_instances`] is enforced against).
    /// The two can briefly diverge: `instances_len` is bumped *before*
    /// compile/instantiate in `spawn_instance` (with rollback on failure)
    /// and a pool can hold an admitted-but-unregistered instance in a warm
    /// channel, so `instances_len() >= live_count()` always holds, with
    /// equality once all in-flight spawns have either registered or rolled
    /// back. Use `live_count()` when you mean "how many handles can
    /// `call_export` resolve right now"; use `instances_len()` when you mean
    /// "how close are we to the admission cap".
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

    /// Current per-tenant **admission** count for `tenant`, or `0` if the
    /// tenant holds no live slots. This is the counter the per-tenant
    /// fairness cap
    /// ([`crate::engine::EngineConfig::max_instances_per_tenant`]) is
    /// enforced against in `spawn_instance`. Exposed for tests and operators
    /// that want to confirm a tenant's footprint; the count only moves when
    /// the per-tenant cap is configured (it is otherwise left at 0 and the
    /// map stays empty).
    pub fn tenant_instance_count(&self, tenant: TenantId) -> usize {
        self.tenant_counts
            .get(&tenant)
            .map(|e| *e.value())
            .unwrap_or(0)
    }

    /// Current **admission** count, sampled atomically. This is the counter
    /// the admission check in `spawn_instance` consults to decide whether a
    /// new instance fits under
    /// [`crate::engine::EngineConfig::max_instances`].
    ///
    /// This is NOT the registry size — see [`Self::live_count`] for the
    /// distinction. This counter includes instances that have been admitted
    /// but are not yet (or are no longer) in the registry: an in-flight
    /// `spawn_instance` between the admission bump and the registry insert,
    /// and pool-held warm instances that were detached from the registry but
    /// still occupy a slot. `instances_len() >= live_count()` always holds.
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

    /// The effective per-call wall-clock budget for a call whose per-call
    /// deadline is `per_call`, reconciled with the engine-wide
    /// [`EngineConfig::max_call_duration`](crate::engine::EngineConfig::max_call_duration)
    /// ceiling (MED finding).
    ///
    /// Returns the *minimum* of the two when both are set (a per-call deadline
    /// can only tighten, never loosen, the engine ceiling); whichever is set
    /// when only one is; and `None` only when BOTH are `None` — i.e. a
    /// deadline-less spawn on an engine that has opted out of the
    /// `max_call_duration` cap, the sole "run until it returns" case. This is
    /// the single source of truth the spawn path and the per-call re-arm both
    /// consult so they agree on when a call traps.
    fn effective_call_budget(&self, per_call: Option<Duration>) -> Option<Duration> {
        match (per_call, self.engine.config().max_call_duration) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
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
    ///
    /// The actual `Module::from_binary` call runs inside
    /// [`tokio::task::spawn_blocking`]: Cranelift compile is CPU-bound and
    /// can exceed 100 ms on multi-MiB modules — running it on a Tokio
    /// worker thread blocks every other I/O task multiplexed onto that
    /// worker. Offloading to the blocking pool keeps the reactor responsive.
    /// The byte-length cap above runs synchronously before the offload so
    /// an oversized blob fails fast without entering the blocking pool.
    ///
    /// Returns the compiled [`Module`] alongside the 32-byte BLAKE3 digest
    /// it was keyed under, so callers (e.g. [`Self::build_pooled_instance`])
    /// that also need the digest for the pool key do not re-hash the bytes
    /// (PERF: the hash was previously computed here AND in
    /// `build_pooled_instance`).
    ///
    /// PERF finding 6: concurrent first-spawns of the same digest are now
    /// single-flighted (see [`Self::inflight_compiles`]), so a duplicate
    /// Cranelift run no longer wastes CPU / a compile permit.
    async fn compile_module_cached(&self, wasm: &[u8]) -> Result<(Module, ModuleHash), ExecError> {
        // BLAKE3 outputs a fixed 32-byte digest; use it whole as the cache key.
        let key: [u8; 32] = *blake3::hash(wasm).as_bytes();
        self.compile_module_cached_with_hash(wasm, key).await
    }

    /// Digest-precomputed sibling of [`Self::compile_module_cached`].
    ///
    /// PERF finding 6: the pool already hashes the wasm bytes to build its
    /// per-tuple `PoolKey` (`instance_pool.rs`), and the spawn path hashed
    /// them AGAIN here. Threading the precomputed `key` through lets the pool's
    /// miss / pre-warm paths skip the second BLAKE3 over multi-MiB modules.
    /// The bare [`Self::compile_module_cached`] computes the digest once and
    /// delegates here, so the single-hash invariant holds for both entry
    /// points. `key` MUST be `blake3::hash(wasm)` — passing a mismatched
    /// digest would key the compiled module under the wrong cache slot.
    pub(crate) async fn compile_module_cached_with_hash(
        &self,
        wasm: &[u8],
        key: [u8; 32],
    ) -> Result<(Module, ModuleHash), ExecError> {
        // Pre-compile size cap (exec hardening). Reject pathologically
        // large blobs *before* handing them to Cranelift — a wasm with a
        // malicious code section can otherwise force arbitrary compile-time
        // CPU. The configured cap is floored at `MAX_MODULE_BYTES` upstream in
        // `EngineConfig`, but we still observe the configured value here so a
        // stricter operator policy wins.
        let cap = self.engine.config().max_module_bytes;
        if wasm.len() > cap {
            return Err(ExecError::ModuleTooLarge {
                len: wasm.len(),
                max: cap,
            });
        }
        // Fast path: already compiled. The scoped lock releases before any
        // compile so concurrent spawns of DIFFERENT modules compile in
        // parallel.
        if let Some(m) = self.module_cache.lock().get(&key).cloned() {
            return Ok((m, key));
        }
        // Single-flight (PERF finding 6): collapse concurrent first-spawns of
        // the SAME digest onto one Cranelift run. The winner runs the compile
        // inside `get_or_try_init`; concurrent callers for the same digest park
        // there and clone the winner's `Module`. The shard write-guard on
        // `inflight_compiles` is dropped immediately after we clone the
        // `Arc<OnceCell>` (the `entry`/`or_insert_with` statement borrows the
        // shard only for that statement) so it is never held across the await.
        let cell = self
            .inflight_compiles
            .entry(key)
            .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
            .value()
            .clone();
        let result = cell
            .get_or_try_init(|| self.compile_uncached(wasm, key))
            .await
            .map(|m| m.clone());
        // Drop the single-flight cell so the map does not grow unbounded; a
        // straggler that already cloned this `Arc` is unaffected, and a fresh
        // caller hits the `module_cache` fast path (the compile, on success,
        // populated it). On error the cell holds nothing, so a later spawn
        // retries the compile cleanly.
        self.inflight_compiles.remove(&key);
        result.map(|module| (module, key))
    }

    /// Run the actual Cranelift compile for `wasm` (no cache / single-flight
    /// bookkeeping — that lives in [`Self::compile_module_cached_with_hash`]).
    /// Offloads to the blocking pool under a compile permit, then populates
    /// the module cache on success. Invoked at most once per digest via the
    /// `inflight_compiles` `OnceCell`.
    async fn compile_uncached(&self, wasm: &[u8], key: [u8; 32]) -> Result<Module, ExecError> {
        // Cranelift compile is CPU-bound — offload to the blocking pool. We
        // clone the wasmtime `Engine` (cheap `Arc`-shaped internally) and the
        // wasm bytes so the closure is fully owning. `spawn_blocking` returns a
        // `JoinError` which we surface as a wasmtime error: a panic inside
        // Cranelift is not something a caller can usefully distinguish from a
        // parse failure, and either way the spawn must be aborted.
        let engine = self.engine.inner().clone();
        let bytes = wasm.to_vec();
        // Bound concurrent Cranelift compiles (MEDIUM finding). The permit is
        // held only for the duration of the `spawn_blocking` compile and
        // dropped immediately after. `acquire_owned` cannot fail unless the
        // semaphore is closed, which we never do; surface the (unreachable)
        // closed case as a wasmtime error rather than panicking.
        let _permit = self
            .compile_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ExecError::Wasmtime(wasmtime::Error::msg("compile semaphore closed")))?;
        let module = tokio::task::spawn_blocking(move || Module::from_binary(&engine, &bytes))
            .await
            .map_err(|join_err| {
                ExecError::Wasmtime(wasmtime::Error::msg(format!(
                    "wasm compile task failed: {join_err}"
                )))
            })?
            .map_err(ExecError::Wasmtime)?;
        drop(_permit);
        self.module_cache.lock().put(key, module.clone());
        Ok(module)
    }

    /// Internal: charge a live-instance slot, enforcing both the engine-wide
    /// [`max_instances`](crate::engine::EngineConfig::max_instances) ceiling
    /// and the optional per-tenant
    /// [`max_instances_per_tenant`](crate::engine::EngineConfig::max_instances_per_tenant)
    /// fairness cap keyed by `tenant`. Returns an [`InstanceSlotGuard`] that
    /// rolls BOTH counts back unless `commit()` is called. Used by
    /// [`Self::build_pooled_instance`] / [`Self::rebuild_pooled_from_module`]
    /// (and transitively [`Self::spawn_instance`]) so the pool's pre-spawn /
    /// reset paths share the same admission accounting as the bare spawn
    /// path.
    ///
    /// Charge order is engine-wide first, then per-tenant. If the per-tenant
    /// cap rejects, the engine-wide charge is rolled back before returning so
    /// a tenant hitting its cap never erodes the shared ceiling.
    fn charge_instance_slot(&self, tenant: TenantId) -> Result<InstanceSlotGuard, ExecError> {
        // Engine-wide ceiling (exec S-10).
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
            self.instance_count.fetch_add(1, Ordering::AcqRel);
        }

        // Per-tenant fairness cap. When unset, the engine-wide charge above
        // is the whole story and the guard carries no per-tenant rollback.
        let Some(per_tenant_max) = self.engine.config().max_instances_per_tenant else {
            return Ok(InstanceSlotGuard::new(self.instance_count.clone()));
        };

        // Charge the per-tenant count under the DashMap entry lock so the
        // read-modify-write is atomic against concurrent spawns of the SAME
        // tenant (different tenants take different shards / entries and never
        // serialise against each other). On overflow, roll the engine-wide
        // charge back so the rejected spawn consumes neither counter.
        let mut entry = self.tenant_counts.entry(tenant).or_insert(0);
        let new_tenant_count = *entry + 1;
        if new_tenant_count > per_tenant_max {
            // Drop the entry guard before mutating other state; the count was
            // never incremented so there is nothing to roll back on the
            // per-tenant side.
            drop(entry);
            self.instance_count.fetch_sub(1, Ordering::Relaxed);
            // Distinct from the engine-wide `CapacityExhausted` above: this is
            // a per-tenant fairness rejection (the offending tenant is over its
            // own quota while the shared budget still has room). It carries the
            // tenant id so the API layer can surface a quota-specific 429
            // (`tenant_capacity_exhausted`) instead of a generic 503. Still log
            // it server-side for operator visibility.
            warn!(
                target: "tensor_wasm_exec::executor",
                tenant = %tenant,
                active = new_tenant_count,
                limit = per_tenant_max,
                "per-tenant instance cap exhausted; refusing spawn (other tenants unaffected)",
            );
            return Err(ExecError::TenantCapacityExhausted {
                tenant,
                active: new_tenant_count,
                limit: per_tenant_max,
            });
        }
        *entry = new_tenant_count;
        drop(entry);
        Ok(InstanceSlotGuard::with_tenant(
            self.instance_count.clone(),
            self.tenant_counts.clone(),
            tenant,
        ))
    }

    /// Internal: explicitly release a live-instance slot charged for
    /// `tenant`. Used by [`InstancePool`] when an instance held in a warm
    /// channel is dropped (channel full on release, reset failed, pool
    /// shutdown). Mirrors the slot release that [`Self::terminate`] performs
    /// for the registered case — both the engine-wide and the per-tenant
    /// count are decremented — but does not touch the registry, since
    /// pooled-but-not-handed-out instances were never registered. The
    /// `tenant` is the same one the matching
    /// [`Self::charge_instance_slot`] was keyed under (the pool always knows
    /// it via the channel's `(tenant, module_hash)` key).
    pub(crate) fn release_instance_slot(&self, tenant: TenantId) {
        self.instance_count.fetch_sub(1, Ordering::AcqRel);
        decrement_tenant_count(&self.tenant_counts, tenant);
    }

    /// Internal: compile + instantiate a Wasm module without registering
    /// the result in the executor registry. The admission slot is charged
    /// (and never rolled back on the success path) so the caller —
    /// [`InstancePool`] — can hold the instance in a warm channel and
    /// account it against the per-engine live-instance cap.
    ///
    /// Returns the detached [`TensorWasmInstance`] plus the compiled
    /// [`Module`] (cached by [`Self::compile_module_cached`], so it is
    /// nearly free to keep around) plus the wasm BLAKE3 digest. The pool
    /// uses the digest as half of its `(tenant_id, module_hash)` channel
    /// key, and the [`Module`] for the cheap reset-on-release path
    /// (re-instantiate from the cached compile, no Cranelift work).
    ///
    /// This is the shared implementation under both [`Self::spawn_instance`]
    /// (which registers immediately) and [`InstancePool::acquire`] /
    /// [`InstancePool::release`] (which hold the instance detached in a
    /// channel). Every deadline / ticker / module-cap check from
    /// [`Self::spawn_instance`] is preserved verbatim — the only behavioural
    /// difference is the missing `instances.insert` at the end.
    pub(crate) async fn build_pooled_instance(
        &self,
        cfg: &SpawnConfig,
        wasm: &[u8],
    ) -> Result<(TensorWasmInstance, Module, ModuleHash), ExecError> {
        // Compute the digest once and delegate; the hash-threaded variant is
        // the single implementation (PERF finding 6 — one BLAKE3 per build).
        let module_hash: ModuleHash = *blake3::hash(wasm).as_bytes();
        self.build_pooled_instance_with_hash(cfg, wasm, module_hash)
            .await
    }

    /// Digest-precomputed sibling of [`Self::build_pooled_instance`].
    ///
    /// PERF finding 6: the pool already hashes the wasm bytes to build its
    /// per-tuple `PoolKey`; passing that digest in lets the pool's pre-warm /
    /// rebuild paths skip the second BLAKE3 over multi-MiB modules that the
    /// `build_pooled_instance` → `compile_module_cached` path used to incur.
    /// `module_hash` MUST be `blake3::hash(wasm)`.
    pub(crate) async fn build_pooled_instance_with_hash(
        &self,
        cfg: &SpawnConfig,
        wasm: &[u8],
        module_hash: ModuleHash,
    ) -> Result<(TensorWasmInstance, Module, ModuleHash), ExecError> {
        // Pre-compile memory-cap walk (exec-S-2 / mem-H5): reject a module
        // whose *declared* linear memory exceeds the engine cap with a
        // structured `ModuleMemoryTooLarge` BEFORE `Module::new` (and before
        // any instance slot is charged). Under the pooling allocator an
        // oversized *initial* memory is otherwise refused by wasmtime at
        // compile time with an opaque "memory ... exceeds the limit" error
        // that masks the typed variant the post-compile
        // `check_module_memory_within_cap` already produces for the
        // declared-maximum and imported cases. Checking the raw bytes here
        // makes the rejection — and its precise requested-byte figure —
        // backend-independent (on-demand vs. pooling).
        check_raw_module_memory_within_cap(wasm, self.engine.config().effective_memory_cap())?;
        let slot_guard = self.charge_instance_slot(cfg.tenant_id)?;
        // Compile (and cache) the module first; the cap check lives inside
        // `compile_module_cached_with_hash` so an oversized blob still fails
        // fast. The precomputed `module_hash` is reused as the cache key so the
        // wasm bytes are hashed exactly once across the whole build (PERF).
        let (module, module_hash) = self
            .compile_module_cached_with_hash(wasm, module_hash)
            .await?;
        let inst = self.instantiate_detached(cfg, &module).await?;
        // A wasmtime instance was successfully created — count it
        // against the monotonic spawn counter exactly once per genuine
        // instantiation. The `active_instances` gauge moves only at
        // registry insert / detach time (see `register_pooled_instance`
        // and `detach_pooled_instance`).
        if let Some(m) = &self.metrics {
            m.instance_spawns_total().inc();
        }
        // Slot stays charged — defuse rollback so the pool's caller can
        // either register the instance (commit it for real) or release
        // the slot explicitly via [`Self::release_instance_slot`].
        slot_guard.commit();
        Ok((inst, module, module_hash))
    }

    /// Internal: build a detached instance from an already-cached [`Module`].
    /// Used by the pool's reset path: drop the spent instance, re-instantiate
    /// from the same compiled module (skipping the Cranelift step entirely),
    /// and stash the fresh instance back in the warm channel.
    ///
    /// The slot is charged on success and the caller decides whether to
    /// release it ([`Self::release_instance_slot`]) or register it
    /// ([`Self::register_pooled_instance`]).
    pub(crate) async fn rebuild_pooled_from_module(
        &self,
        cfg: &SpawnConfig,
        module: &Module,
    ) -> Result<TensorWasmInstance, ExecError> {
        let slot_guard = self.charge_instance_slot(cfg.tenant_id)?;
        let inst = self.instantiate_detached(cfg, module).await?;
        if let Some(m) = &self.metrics {
            m.instance_spawns_total().inc();
        }
        slot_guard.commit();
        Ok(inst)
    }

    /// Internal: shared instantiation logic. Builds the [`Store`], wires the
    /// limiter, arms the start-function epoch deadline, builds a
    /// [`wasmtime::Linker`] with every available host surface registered
    /// (scheduler always; the `wasi:tensor/host` input channel — `input-len`
    /// / `read-input` — always; streaming when [`SpawnConfig::streaming`] is
    /// set; JIT dispatch when a [`KernelCache`] is configured via
    /// [`Self::with_jit_cache`]), and instantiates against it. Does NOT touch
    /// the registry or the admission counter — callers must pair this with
    /// [`Self::charge_instance_slot`] / [`Self::register_pooled_instance`].
    async fn instantiate_detached(
        &self,
        cfg: &SpawnConfig,
        module: &Module,
    ) -> Result<TensorWasmInstance, ExecError> {
        // Reconcile the engine-wide cap with the pooling allocator's slot
        // size (MED finding): on the pooling backend a module larger than
        // the physical slot would fail to instantiate with an opaque
        // allocator error, so we cap the limiter AND the pre-instantiation
        // module check at `min(max_memory_bytes, pooling memory_bytes)`.
        // `effective_memory_cap()` is `max_memory_bytes` on the
        // UnifiedBuffer path, preserving prior behaviour there.
        let max_memory_bytes = self.engine.config().effective_memory_cap();
        let mut state =
            InstanceState::new(cfg.tenant_id, InstanceId(0)).with_memory_limit(max_memory_bytes);
        if let Some(ref s) = cfg.streaming {
            state = state.with_streaming(s.clone());
        }
        if !cfg.input.is_empty() {
            state = state.with_input(InputContext::new(cfg.input.clone()));
        }
        if let Some(d) = cfg.deadline {
            state = state
                .with_deadline(Instant::now() + d)
                .with_deadline_duration(d);
        }
        // HARD-deadline instant the cooperative epoch callback traps at while
        // `start` (and anything else running inside `instantiate_async`)
        // executes. Bounded by BOTH the per-call deadline (if any) and the
        // implicit [`MAX_START_FN_DURATION`] cap, mirroring the old
        // `start_deadline_ticks = min(epoch_deadline_ticks, max_start_ticks)`
        // trap point — except now expressed as a wall-clock instant the
        // callback consults on each yield rather than a one-shot trap count.
        // Until the callback observes this instant has passed it yields
        // cooperatively, so even a runaway `start` function is interruptible
        // by the epoch ticker AND traps no later than this instant.
        let start_phase_budget = match cfg.deadline {
            Some(d) => d.min(MAX_START_FN_DURATION),
            None => MAX_START_FN_DURATION,
        };
        state = state.with_hard_deadline(Instant::now() + start_phase_budget);
        let tick = self.engine.config().epoch_tick;
        let epoch_deadline_ticks = match cfg.deadline {
            Some(d) => duration_to_epoch_ticks(d, tick),
            // Overflow-safe "no deadline": `set_epoch_deadline` is relative
            // (`current_epoch + ticks`), so `u64::MAX` would overflow once the
            // ticker advances. See [`MAX_EPOCH_DEADLINE_TICKS`].
            None => MAX_EPOCH_DEADLINE_TICKS,
        };
        let max_start_ticks = {
            let d_ms = MAX_START_FN_DURATION.as_millis();
            let t_ms = tick.as_millis().max(1);
            let ticks_u128 = d_ms.div_ceil(t_ms).max(1);
            u64::try_from(ticks_u128).unwrap_or(u64::MAX)
        };
        let start_deadline_ticks = epoch_deadline_ticks.min(max_start_ticks);
        let deadline_class_applies =
            cfg.deadline.is_some() || MAX_START_FN_DURATION > Duration::ZERO;
        if deadline_class_applies && !self.engine.is_epoch_ticker_running() {
            let flag = self.ticker_warned.get_or_init(|| AtomicBool::new(false));
            if !flag.swap(true, Ordering::AcqRel) {
                tracing::error!(
                    target: "tensor_wasm_exec::executor",
                    "epoch ticker not running — refusing spawn; call `engine.spawn_epoch_ticker()` before serving traffic",
                );
            }
            return Err(ExecError::EpochTickerNotRunning);
        }
        check_module_memory_within_cap(module, max_memory_bytes)?;
        let mut store = Store::new(self.engine.inner(), state);
        store.limiter(|state| &mut state.limiter as &mut dyn ResourceLimiter);
        // Arm the COOPERATIVE epoch scheme for the start phase. The first
        // deadline is armed at the cooperative cadence (never beyond
        // `start_deadline_ticks`, which is the latest the hard deadline could
        // possibly fall) so the callback gets a chance to run — and yield —
        // long before the start-phase budget elapses. At each epoch trip the
        // guest yields `Pending` (so an in-flight `instantiate_async` is
        // cancellable on future-drop) and the callback traps only once the
        // per-store `hard_deadline` (set just above to the start-phase budget)
        // has elapsed — preserving the `MAX_START_FN_DURATION` guarantee that
        // a runaway `start` cannot burn forever. The relative tick counts now
        // govern only the YIELD cadence; the trap point is the wall-clock
        // `hard_deadline` instant the callback consults.
        arm_cooperative_epoch(
            &mut store,
            start_deadline_ticks.min(COOPERATIVE_YIELD_TICKS),
        );
        // HIGH finding fix: build a single `Linker<InstanceState>` and
        // register every host surface whose backing machinery is actually
        // present, then instantiate against it. Previously only the
        // streaming surface was wired (and only on the streaming path),
        // leaving the scheduler (`instance.rs` `SchedulerContext`) and
        // JIT-dispatch (`jit_dispatch.rs`) imports unlinkable on the real
        // spawn path — a guest importing `wasi:scheduler/host` or
        // `tensor-wasm:jit/host` failed to link and that machinery was dead.
        //
        // All three surfaces now coexist on the same linker:
        //   - scheduler: always registered. The per-store `SchedulerContext`
        //     is constructed unconditionally in `InstanceState::new`
        //     (`unbounded()` default, real budget when a deadline is set), so
        //     the surface is always safe to expose — a guest that doesn't
        //     import it pays nothing.
        //   - streaming: registered when `cfg.streaming.is_some()`
        //     (preserves prior behaviour; the per-store context is
        //     `disabled()` otherwise and the surface is only meaningful for
        //     the `/invoke-stream` route).
        //   - jit: registered when a `KernelCache` is configured on the
        //     executor (`with_jit_cache`); the cache is the cross-tenant
        //     backing store the dispatch closure consults (lookups are
        //     tenant-scoped internally).
        //
        // The scheduler surface is always available (its per-store context is
        // constructed unconditionally), so we always go through the linker
        // path now. A guest with zero imports still instantiates fine — a
        // linker with extra registered imports does not force the guest to
        // import them. (Wasmtime only errors on *missing* imports, never on
        // *unused* registered ones.) This replaces the old
        // `Instance::new_async(.., &[])` empty-imports branch.
        let mut linker: wasmtime::Linker<InstanceState> =
            wasmtime::Linker::new(self.engine.inner());
        // Scheduler host functions (`wasi:scheduler/host@0.1.0`). The getter
        // borrows the per-store `SchedulerContext`, so two instances sharing
        // a linker never cross-talk.
        add_scheduler_to_linker(&mut linker, |state: &InstanceState| state.scheduler())
            .map_err(ExecError::Wasmtime)?;
        // Guest-input pull channel (`wasi:tensor/host` `input-len` /
        // `read-input`): always registered. The per-store `InputContext`
        // is constructed unconditionally in `InstanceState::new`
        // (`empty()` default, populated when `cfg.input` is non-empty),
        // so the surface is always safe to expose — a guest that doesn't
        // import it pays nothing, and one that does sees `input-len() ==
        // 0` when no prompt was staged. Mirrors the always-on scheduler
        // surface above. Registered before streaming so both
        // `wasi:tensor/host` host-fn families coexist on one linker.
        add_input_to_linker(&mut linker).map_err(ExecError::Wasmtime)?;
        if cfg.streaming.is_some() {
            add_streaming_to_linker(&mut linker).map_err(ExecError::Wasmtime)?;
        }
        if let Some(cache) = &self.jit_cache {
            // `add_jit_dispatch_to_linker` is generic over the store payload
            // via `JitArenaProvider + TenantContext`, both of which
            // `InstanceState` implements (see `instance.rs` /
            // `jit_dispatch.rs`). The arena lives per-store; the cache is the
            // shared cross-tenant backing handle.
            add_jit_dispatch_to_linker(&mut linker, cache.clone()).map_err(ExecError::Wasmtime)?;
        }
        let instance = match linker.instantiate_async(&mut store, module).await {
            Ok(inst) => inst,
            Err(err) => {
                // MED finding: the pre-instantiation `check_module_memory_within_cap`
                // already rejects modules whose *declared* memory exceeds the
                // reconciled cap with a typed `ModuleMemoryTooLarge`. The
                // pooling allocator can still refuse instantiation for a
                // memory-sizing reason the static check cannot see (e.g. the
                // slot byte-size ceiling), surfacing it as an opaque
                // wasmtime error. Where the error chain carries the
                // allocator's memory-size signature, re-classify it as the
                // typed `ModuleMemoryTooLarge` so callers get a stable
                // `MemoryExhausted` rather than an opaque compile error.
                // This is best-effort string inspection — it only *refines*
                // the error; anything unrecognised still flows through as
                // `ExecError::Wasmtime` unchanged.
                return Err(classify_instantiation_error(err, max_memory_bytes));
            }
        };
        // Start phase complete: re-arm the cooperative epoch deadline for the
        // post-instantiation window and swap the hard-deadline instant from
        // the start-phase budget over to the per-call deadline.
        //
        //   * Deadline spawn: trap at the configured per-call `deadline`
        //     instant (`now + d`, seeded above). A later `call_export`
        //     re-arms both this instant and the ticks for that call's window.
        //   * Deadline-less spawn: clear the hard deadline so the callback
        //     yields cooperatively FOREVER (matching the old
        //     `MAX_EPOCH_DEADLINE_TICKS` "effectively no deadline" semantics) —
        //     but now the guest is CANCELLABLE on future-drop between yields,
        //     which the old trap-at-`u64::MAX/2` scheme could not offer.
        //
        // The relative tick count armed here is only the YIELD cadence (never
        // beyond `epoch_deadline_ticks`, the latest a configured deadline
        // could fall); the trap decision is driven entirely by the wall-clock
        // `hard_deadline` the callback consults. Arming at the cooperative
        // cadence — rather than the old `epoch_deadline_ticks` (which for a
        // deadline-less spawn was the near-infinite `MAX_EPOCH_DEADLINE_TICKS`
        // sentinel) — is what makes even a deadline-less compute-bound guest
        // yield, and therefore stay cancellable, instead of running an entire
        // epoch sentinel's worth of ticks before its first yield point.
        store.set_epoch_deadline(epoch_deadline_ticks.min(COOPERATIVE_YIELD_TICKS));
        // MED finding: the post-start hard deadline is the per-call deadline
        // when one is configured, otherwise the engine-wide
        // `max_call_duration` ceiling — so a `deadline: None` spawn is bounded
        // rather than yielding forever. `effective_call_budget` returns the
        // minimum of the two (either being `None` drops out), and `None` only
        // when BOTH are unset (the opt-out: genuinely unbounded). This is the
        // budget the FIRST call inherits; `call_export_with_args` re-arms it
        // per call so back-to-back invocations each get a fresh window.
        let post_start_hard_deadline = self
            .effective_call_budget(cfg.deadline)
            .map(|d| Instant::now() + d);
        store.data_mut().hard_deadline = post_start_hard_deadline;
        Ok(TensorWasmInstance::new(store, instance))
    }

    /// Internal: register a previously-detached [`TensorWasmInstance`] in
    /// the executor registry, allocating a fresh [`InstanceId`]. The slot
    /// is assumed to already be charged (via
    /// [`Self::charge_instance_slot`] or its [`InstanceSlotGuard::commit`]);
    /// this method does NOT bump `instance_count`.
    ///
    /// Used by [`InstancePool::acquire`] to surface a warm-channel
    /// instance through the standard [`InstanceId`] handle that
    /// [`Self::call_export_with_args`] consumes.
    pub(crate) fn register_pooled_instance(
        &self,
        mut inst: TensorWasmInstance,
    ) -> Result<InstanceId, ExecError> {
        let id = self.allocate_instance_id();
        // Overwrite the placeholder InstanceId(0) baked in at
        // instantiation time with the freshly-allocated registry id so
        // host imports observing `caller.data().instance_id` see the
        // same value the caller holds.
        inst.store.data_mut().instance_id = id;
        // Capture the metadata the registry value caches BEFORE the instance
        // is moved into the mutex: the owning tenant (PERF finding 6 — read
        // lock-free by `terminate` / `detach`) and the cooperative-cancellation
        // handle (HIGH finding 1 — flipped lock-free by `terminate`). Reading
        // them here is free; we already hold the only reference to `inst`.
        let tenant_id = inst.store.data().tenant_id;
        let cancelled = inst.store.data().cancel_handle();
        match self.instances.entry(id) {
            Entry::Vacant(v) => {
                v.insert(RegistryEntry {
                    handle: Arc::new(Mutex::new(inst)),
                    tenant_id,
                    cancelled,
                    export_arity: Arc::new(dashmap::DashMap::new()),
                });
            }
            Entry::Occupied(_) => {
                warn!(
                    target: "tensor_wasm_exec::executor",
                    %id,
                    "instance id race after allocation (pool register); this is a serious bug",
                );
                return Err(ExecError::Wasmtime(wasmtime::Error::msg(
                    "instance id collision after allocation",
                )));
            }
        }
        if let Some(m) = &self.metrics {
            // Only the gauge moves here — `instance_spawns_total` is
            // incremented at instantiate time inside
            // [`Self::build_pooled_instance`] / [`Self::rebuild_pooled_from_module`]
            // so the monotonic counter measures genuine wasmtime
            // instantiations, not registry insertions (the pool can
            // re-register the same underlying instance after a reset).
            m.active_instances().inc();
        }
        Ok(id)
    }

    /// Internal: remove an instance from the registry, returning the
    /// underlying [`TensorWasmInstance`] WITHOUT decrementing
    /// `instance_count`. The slot remains charged so the caller —
    /// [`InstancePool::release`] — can keep the instance alive in its
    /// warm channel without churning the admission counter.
    ///
    /// Returns `None` if the id is unknown (already terminated / never
    /// registered).
    pub(crate) async fn detach_pooled_instance(
        &self,
        id: InstanceId,
    ) -> Option<TensorWasmInstance> {
        let (_, entry) = self.instances.remove(&id)?;
        // Decrement the "active_instances" gauge: this is the moment the
        // instance leaves the externally-visible registry. The
        // `instance_spawns_total` counter is intentionally not paired
        // with a decrement (it is monotonic), and `instance_terminations_total`
        // is reserved for genuine terminate calls, not pool detach.
        if let Some(m) = &self.metrics {
            m.active_instances().dec();
        }
        // PERF finding 6: the owning tenant is cached on the registry value at
        // register time, so we read it from a plain field instead of locking
        // the per-instance mutex (which a racing in-flight call holds across
        // `call_async`). Only needed when a per-tenant cap is configured —
        // otherwise `tenant_counts` is never populated, mirroring `terminate`.
        let owning_tenant = if self.engine.config().max_instances_per_tenant.is_some() {
            Some(entry.tenant_id)
        } else {
            None
        };
        let handle = entry.handle;
        // Unwrap the Arc<Mutex<_>>: we need the sole strong reference to move
        // the instance out into the warm channel.
        //
        // LOW finding (detach try_unwrap race): the previous comment claimed
        // "the registry never hands out Arc clones" — that is NOT true.
        // `call_export_with_args` clones the value handle
        // (`.value().clone()`) and holds it across its `call_async` await, so
        // a `detach` racing an in-flight call on the SAME id observes an
        // outstanding strong reference and `try_unwrap` fails. The
        // `DashMap::remove` above prevents NEW clones (the entry is gone), but
        // an already-cloned handle from a call that started before the remove
        // can still be live. This is rare (the pool detaches idle instances)
        // but reachable, so the failure branch below must be correct, not
        // merely "should not happen".
        match Arc::try_unwrap(handle) {
            Ok(mutex) => Some(mutex.into_inner()),
            Err(_arc) => {
                // A concurrent `call_export_with_args` still holds a clone of
                // this handle. We drop our reference and skip the pool path —
                // returning `None` is the safe behaviour (no use-after-detach;
                // the racing call finishes and drops the last Arc, freeing the
                // instance). But we already removed the entry from the
                // registry, so the engine-wide admission slot it occupied
                // would otherwise leak permanently (no `terminate` /
                // `release_instance_slot` will ever run for this id). The
                // racing in-flight call is the pool path's *non-terminating*
                // `call_export_with_args` (see `invoke`): when it finishes it
                // merely drops its Arc clone — it never calls `terminate` /
                // `release_instance_slot`, and the registry entry is already
                // gone, so NO other code path will ever decrement this slot.
                // We therefore must release BOTH counters here, exactly
                // mirroring the `release_instance_slot` the `Some` path's
                // caller (`InstancePool::release`) would have run. This is the
                // sole decrement for this slot, so it cannot double-count:
                // - engine-wide slot: released here (was previously the only
                //   release, hence correct);
                // - per-tenant slot: now released here too (using the tenant
                //   id captured up-front above), closing the unbounded
                //   per-tenant leak that the old "lesser, bounded evil"
                //   comment described. The decrement is skipped when no
                //   per-tenant cap is configured (`owning_tenant` is `None`),
                //   matching `terminate` / `release_instance_slot`.
                // The happy path above intentionally leaves the slot charged
                // (the pool owns it and releases via `release_instance_slot`).
                self.instance_count.fetch_sub(1, Ordering::AcqRel);
                if let Some(tenant) = owning_tenant {
                    decrement_tenant_count(&self.tenant_counts, tenant);
                }
                warn!(
                    target: "tensor_wasm_exec::executor",
                    %id,
                    "detach_pooled_instance: outstanding Arc reference (racing in-flight call); \
                     instance not pooled, both engine-wide and per-tenant slots released",
                );
                None
            }
        }
    }

    /// Internal: when [`EngineConfig::auto_offload`](crate::engine::EngineConfig::auto_offload)
    /// is enabled, consult the analyser and rewrite offload-candidate
    /// function bodies into JIT-dispatch trampolines, returning the
    /// rewritten bytes. On every fallback condition — flag disabled, no JIT
    /// cache attached (the trampoline's `tensor-wasm:jit/host` imports would
    /// be unlinkable), analysis error, rewrite error, or a rewrite that
    /// swapped nothing — the original `wasm` slice is borrowed unchanged.
    /// This is the activation point for the swap the `auto_offload` module
    /// documents as consultation-only: with the flag off it stays exactly
    /// that.
    ///
    /// Never returns an error: a failure to analyse or rewrite must not fail
    /// a spawn, so every error path logs and falls back to the original
    /// module.
    fn maybe_rewrite_for_offload<'w>(
        &self,
        cfg: &SpawnConfig,
        wasm: &'w [u8],
    ) -> std::borrow::Cow<'w, [u8]> {
        use std::borrow::Cow;

        if !self.engine.config().auto_offload {
            return Cow::Borrowed(wasm);
        }
        // The rewritten module imports `tensor-wasm:jit/host` (dispatch /
        // alloc / free). Those imports only link when a `KernelCache` is
        // attached (`with_jit_cache`) — without one the rewritten module
        // would fail to instantiate, which is strictly worse than running
        // the original on the CPU. Skip the rewrite and fall back.
        let Some(cache) = self.jit_cache.as_ref() else {
            debug!(
                target: "tensor_wasm_exec::executor",
                tenant = %cfg.tenant_id,
                "auto_offload enabled but no JIT cache attached; falling back to original module",
            );
            return Cow::Borrowed(wasm);
        };

        // Resolve the detector thresholds once and use the SAME config for
        // the consultation pass and the rewrite so they agree on candidates.
        let detector = self
            .engine
            .config()
            .auto_offload_detector
            .unwrap_or_default();

        // Consultation pass: emit the per-function verdicts (the historical
        // consultation-only behaviour) and decide whether any function is
        // worth offloading before paying for the rewrite. If analysis
        // errors, fall back.
        let verdicts = match crate::auto_offload::analyse_with_config(wasm, &detector) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    target: "tensor_wasm_exec::executor",
                    tenant = %cfg.tenant_id,
                    error = %e,
                    "auto_offload analysis failed; falling back to original module",
                );
                return Cow::Borrowed(wasm);
            }
        };
        let any_offload = verdicts.iter().any(|v| {
            matches!(
                v.verdict,
                tensor_wasm_jit::detector::DetectorVerdict::Offload
            )
        });
        if !any_offload {
            // No candidate — nothing to rewrite. Borrow the original.
            return Cow::Borrowed(wasm);
        }

        // Rewrite. Thread the spawning tenant so the cache pre-population is
        // keyed under the tenant the runtime dispatch looks up against
        // (cache keys are tenant-scoped), and the resolved detector so the
        // rewrite swaps exactly the functions the consultation flagged.
        let opts = RewriteOptions {
            tenant_id: cfg.tenant_id,
            detector,
            ..RewriteOptions::default()
        };
        match rewrite_wasm(wasm, &opts, cache) {
            Ok(outcome) if !outcome.offloaded_functions.is_empty() => {
                info!(
                    target: "tensor_wasm_exec::executor",
                    tenant = %cfg.tenant_id,
                    offloaded = outcome.offloaded_functions.len(),
                    total_defined = outcome.total_defined_functions,
                    "auto_offload rewrite applied; instantiating trampoline-augmented module",
                );
                Cow::Owned(outcome.rewritten_wasm)
            }
            Ok(_) => {
                // The detector flagged a candidate but the rewriter declined
                // every swap (e.g. unsupported signature / lowering refusal).
                // Nothing changed — borrow the original to skip a needless
                // recompile of identical-but-reencoded bytes.
                debug!(
                    target: "tensor_wasm_exec::executor",
                    tenant = %cfg.tenant_id,
                    "auto_offload rewrite swapped no functions; using original module",
                );
                Cow::Borrowed(wasm)
            }
            Err(e) => {
                warn!(
                    target: "tensor_wasm_exec::executor",
                    tenant = %cfg.tenant_id,
                    error = %e,
                    "auto_offload rewrite failed; falling back to original module",
                );
                Cow::Borrowed(wasm)
            }
        }
    }

    /// Compile + instantiate a Wasm module. Returns the assigned [`InstanceId`].
    ///
    /// # Deadline / ticker contract
    ///
    /// If a [`SpawnConfig::deadline`] is set — or if the implicit
    /// [`MAX_START_FN_DURATION`] cap would otherwise apply (which it
    /// always does, since every spawn runs `Instance::new_async`) —
    /// the engine's epoch ticker MUST be running. Without it the
    /// per-store epoch counter never advances, so neither the per-call
    /// deadline nor the start-function cap can fire, and a runaway
    /// guest would wedge the worker thread until it returned of its
    /// own accord. We refuse the spawn with
    /// [`ExecError::EpochTickerNotRunning`] instead of silently
    /// dropping the deadline contract; operators must call
    /// `engine.spawn_epoch_ticker()` (typically inside a Tokio runtime
    /// at startup) before serving traffic. The engine constructor
    /// auto-spawns the ticker when invoked from inside a runtime, so
    /// this only trips for sync-startup setups that forget the
    /// explicit call.
    #[instrument(skip(self, wasm), fields(tenant = %cfg.tenant_id, instance_id = tracing::field::Empty))]
    pub async fn spawn_instance(
        &self,
        cfg: SpawnConfig,
        wasm: &[u8],
    ) -> Result<InstanceId, ExecError> {
        // Auto-offload (opt-in via `EngineConfig::auto_offload`). When
        // enabled, consult the analyser and — if any function is flagged —
        // rewrite the module's offload-candidate bodies into JIT-dispatch
        // trampolines, then instantiate the rewritten module instead of the
        // original. Any failure (analysis error, rewrite error, no JIT cache
        // to link the trampoline imports, or a rewrite that swaps nothing)
        // falls back to the original bytes — enabling the flag can never
        // fail a spawn that would otherwise have succeeded. Returns a `Cow`
        // so the disabled / fallback paths borrow the caller's slice with
        // zero copies.
        let effective_wasm = self.maybe_rewrite_for_offload(&cfg, wasm);
        // Refactored to share the detached compile+instantiate path
        // (`build_pooled_instance`) with [`InstancePool`]. The semantics
        // are byte-for-byte preserved: admission control runs first
        // (with rollback on failure), then compile / instantiate /
        // register. The split lets the pool reuse the heavy work
        // without the registry insert when holding warm instances in a
        // channel.
        let (inst, _module, _module_hash) = self
            .build_pooled_instance(&cfg, effective_wasm.as_ref())
            .await?;
        // `build_pooled_instance` returns with the slot committed (charged),
        // so a failure between here and the registry insert must release
        // the slot explicitly. Wrap it in a `defer`-style guard so any
        // `?` from `register_pooled_instance` does not leak the count.
        // `build_pooled_instance` committed BOTH the engine-wide and (when a
        // per-tenant cap is configured) the per-tenant count, so this
        // re-protect guard must roll back both on a failed register. Use the
        // tenant-aware constructor only when the per-tenant cap is active so
        // we don't decrement a per-tenant entry that was never charged.
        let slot_guard = if self.engine.config().max_instances_per_tenant.is_some() {
            InstanceSlotGuard::with_tenant(
                self.instance_count.clone(),
                self.tenant_counts.clone(),
                cfg.tenant_id,
            )
        } else {
            InstanceSlotGuard::new(self.instance_count.clone())
        };
        let id = self.register_pooled_instance(inst)?;
        // Successful register: defuse the rollback so the slot stays
        // charged until `terminate`. (Without this defuse the guard's
        // Drop would decrement the count we just committed.)
        slot_guard.commit();
        tracing::Span::current().record("instance_id", tracing::field::display(id));
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
    #[deprecated(
        since = "0.3.7",
        note = "use `call_export_with_args` with an empty `&[]` for the same semantics; \
                v0.4 removes this shim. See `docs/MIGRATING-FROM-WASMTIME-WASMER.md` § \"Typed exports\"."
    )]
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
        self.call_export_with_args(id, export, &[])
            .await
            .map(|_| ())
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
        // Clone the registry value (all `Arc` clones — cheap) so we hold the
        // per-instance handle and the cached export-arity map across the await
        // without keeping the `DashMap` shard locked.
        let entry = self
            .instances
            .get(&id)
            .ok_or(ExecError::NotFound(id))?
            .value()
            .clone();
        let handle = entry.handle;
        let export_arity = entry.export_arity;
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
        // MED finding: the effective wall-clock budget for THIS call is the
        // per-call deadline reconciled with the engine-wide
        // `max_call_duration` ceiling. For a deadline-less spawn this is the
        // engine ceiling — so the call is bounded rather than able to run
        // forever — and `None` only when BOTH are unset (the explicit opt-out).
        let effective_budget = self.effective_call_budget(configured_deadline);
        if let Some(d) = effective_budget {
            let new_deadline = call_start + d;
            guard.store.data_mut().deadline = Some(new_deadline);
            // Re-arm the HARD-deadline instant the cooperative epoch callback
            // traps at, in lockstep with `deadline`, so THIS call's window
            // (not a previous call's, nor the spawn-time start-phase budget)
            // governs when a runaway guest traps. The callback installed at
            // spawn (`arm_cooperative_epoch`) persists across calls; only the
            // instant it consults and the relative tick count are re-armed.
            // For a deadline-less spawn `d` is the engine ceiling, so even a
            // `deadline: None` call now traps once it elapses.
            guard.store.data_mut().hard_deadline = Some(new_deadline);
            let tick = self.engine.config().epoch_tick;
            // Re-arm at the cooperative cadence (never beyond this call's
            // budget window). The cooperative epoch callback installed at
            // spawn yields `Pending` at each trip and traps only once
            // `hard_deadline` (re-armed to `new_deadline` above) elapses — so
            // the relative ticks here govern only the yield cadence, while the
            // wall-clock `hard_deadline` governs the trap. Arming at the
            // cooperative cadence keeps a compute-bound guest yielding (hence
            // cancellable on future-drop) throughout the call.
            let call_ticks = duration_to_epoch_ticks(d, tick).min(COOPERATIVE_YIELD_TICKS);
            guard.store.set_epoch_deadline(call_ticks);
        } else {
            // No per-call deadline AND no engine ceiling: genuinely unbounded.
            // Clear any stale absolute deadline so the timeout classification
            // below does not misfire, and leave the cooperative callback
            // yielding forever (the historical opt-out behaviour). The guest
            // stays cancellable on future-drop / `terminate`.
            guard.store.data_mut().deadline = None;
            guard.store.data_mut().hard_deadline = None;
        }
        // Re-arm the cooperative-scheduler context's wall-clock origin
        // in lockstep with the deadline above. Without this, a guest
        // that calls `wasi:scheduler/host.yield()` on a back-to-back
        // call would see the elapsed time from the *previous* call
        // charged against its budget, and the first yield would
        // immediately return STOP. The re-arm is a no-op for spawns
        // without a configured deadline (the unbounded context
        // ignores `started_at`).
        guard.store.data_mut().rearm_scheduler();
        let deadline_at = guard.store.data().deadline;
        // Report the EFFECTIVE budget (per-call deadline reconciled with the
        // engine ceiling) in any resulting `TimeoutContext`, so a trap on a
        // deadline-less call surfaces the engine ceiling that actually bounded
        // it rather than `0` (MED finding). `0` only when the call is
        // genuinely unbounded (no per-call deadline and no engine ceiling),
        // in which case `deadline_at` is `None` and no timeout is classified.
        let configured_deadline_ms = effective_budget
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let wasmtime_instance = *guard.wasmtime_instance();
        let func = wasmtime_instance
            .get_func(&mut guard.store, export)
            .ok_or_else(|| ExecError::MissingExport(export.to_string()))?;

        // Branch on the export's full signature. The typed fast path is
        // reserved for genuine `() -> ()` exports — no args AND no results
        // — so we don't disturb the behaviour every existing void-export
        // test relies on (and so the dynamic-call overhead stays off the
        // bench path). A no-arg export that nonetheless RETURNS a value
        // (e.g. `tick: () -> i32`) must NOT take this branch: a
        // `typed::<(), ()>` view of it is rejected by wasmtime with
        // "type mismatch with results: expected 0 types, found N". Every
        // other shape — args present, results present, or both — takes the
        // dynamic `Func::call_async` path with `&[Val]` IO buffers (params
        // may legitimately be empty); the result vec is pre-sized to the
        // export's declared result arity so wasmtime can write into it.
        //
        // PERF finding 6: the declared result arity is the only thing this
        // branch needs from `Func::ty`, and a module's export signatures are
        // immutable for the instance's life. `Func::ty` allocates a fresh
        // `FuncType` (a `Vec` of param + result value types) on every call;
        // we cache the result count per export on the registry value so
        // repeat invocations skip that allocation. The first call for a given
        // export pays the `Func::ty` once and memoises; the value is stable.
        let results_len = match export_arity.get(export).map(|v| *v.value()) {
            Some(n) => n,
            None => {
                let n = func.ty(&guard.store).results().len();
                export_arity.insert(export.to_string(), n);
                n
            }
        };
        let call_outcome = if args.is_empty() && results_len == 0 {
            match func.typed::<(), ()>(&guard.store) {
                Ok(typed) => typed
                    .call_async(&mut guard.store, ())
                    .await
                    .map(|()| serde_json::Value::Array(Vec::new())),
                Err(e) => Err(e),
            }
        } else {
            let params: Vec<Val> = args.iter().copied().map(WasmArg::into_val).collect();
            // `Val::I32(0)` is just a placeholder — wasmtime overwrites
            // every slot before returning. The element count must match
            // the export's declared result arity exactly or wasmtime
            // returns an error.
            let mut results: Vec<Val> = vec![Val::I32(0); results_len];
            match func
                .call_async(&mut guard.store, &params, &mut results)
                .await
            {
                Ok(()) => {
                    // Project each returned Val into JSON. A non-numeric return
                    // type (`v128` / reference) yields `UnsupportedReturnType`
                    // rather than a silent `null` (LOW finding). Collect into a
                    // `Result` so the first unrepresentable value short-circuits;
                    // bubble it up through the shared `call_outcome` match below
                    // by mapping it back into a wasmtime-shaped error is NOT
                    // appropriate (it is a typed exec error), so we return it
                    // directly here.
                    match results
                        .iter()
                        .map(val_to_json)
                        .collect::<Result<Vec<serde_json::Value>, ExecError>>()
                    {
                        Ok(json) => return Ok(serde_json::Value::Array(json)),
                        Err(e) => return Err(e),
                    }
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
                let past_deadline = deadline_at.map(|d| Instant::now() >= d).unwrap_or(false);
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
            Some((_, entry)) => {
                // HIGH finding 1: flip the cooperative-cancellation flag so an
                // in-flight, possibly deadline-less call is TRAPPED at the next
                // epoch tick instead of merely being de-registered. The flag is
                // an `Arc<AtomicBool>` shared with the per-store
                // `InstanceState::cancelled`; we set it lock-free, WITHOUT
                // touching the per-instance mutex — which the running call holds
                // across its `call_async` await. The epoch callback checks
                // `is_cancelled()` first on every trip, so once the background
                // ticker advances the epoch the guest unwinds with an
                // "instance terminated" trap. Removing the registry entry (above)
                // and freeing the admission slot (below) closes the leak; the
                // flag closes the run-time interruption gap the previous
                // implementation could not. `Release` so the trap-side
                // `Acquire` load observes the flip. Idempotent: a double
                // terminate would already be `NotFound` at the `remove`.
                entry.cancelled.store(true, Ordering::Release);
                // Release the admission slot reserved at spawn time
                // (exec S-10). The decrement only runs on successful
                // removal — a `NotFound` terminate must not free a
                // slot it never charged, or a tenant could double-
                // terminate to inflate their effective cap.
                self.instance_count.fetch_sub(1, Ordering::AcqRel);
                // Mirror the engine-wide release on the per-tenant count
                // (fairness cap). Only taken when a per-tenant cap is
                // configured — otherwise `tenant_counts` is never populated
                // and we skip the decrement entirely. PERF finding 6: the
                // owning tenant is cached on the registry value at register
                // time, so we read it from a plain field — no per-instance
                // lock (which a racing in-flight call would be holding).
                if self.engine.config().max_instances_per_tenant.is_some() {
                    decrement_tenant_count(&self.tenant_counts, entry.tenant_id);
                }
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
    #[deprecated(
        since = "0.3.7",
        note = "use `call_export_with_args_then_terminate` with an empty `&[]` for the same semantics; \
                v0.4 removes this shim. See `docs/MIGRATING-FROM-WASMTIME-WASMER.md` § \"Typed exports\"."
    )]
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

    /// High-level "invoke" entry point: spawn (or draw from the warm pool),
    /// call the export, return the result, and clean up the instance —
    /// all in a single async call.
    ///
    /// When an [`InstancePool`] is attached via [`Self::with_instance_pool`],
    /// this routes through [`InstancePool::acquire`] / [`InstancePool::release`]
    /// so the per-(tenant, module-hash) warm channel absorbs the
    /// compile+instantiate cost on the hot path (T37). Without a pool
    /// attached, behaviour is identical to
    /// [`Self::spawn_instance`] + [`Self::call_export_with_args_then_terminate`]
    /// — every existing caller of that pair sees no semantic change.
    ///
    /// Mirrors the routes layer's invoke / invoke-stream / invoke-async
    /// pattern: a single `(wasm, cfg, export, args)` shot, no caller-side
    /// id juggling. T34's streaming and T36's back-pressure /
    /// deadline-near semantics propagate unchanged — the
    /// [`SpawnConfig::streaming`] context flows through `acquire`, and
    /// per-call deadline re-arming happens inside `call_export_with_args`
    /// regardless of which path produced the instance.
    ///
    /// Pool-side invariant: streaming spawns are NEVER recycled, even
    /// when a pool is attached — the streaming context is one-shot
    /// (the gateway's channel receiver is drained for the duration of
    /// the SSE response and then dropped). [`InstancePool::release`]
    /// drops the instance instead of returning it to the warm channel
    /// when `SpawnConfig::streaming.is_some()`.
    pub async fn invoke(
        &self,
        cfg: SpawnConfig,
        wasm: &[u8],
        export: &str,
        args: &[WasmArg],
    ) -> Result<serde_json::Value, ExecError> {
        if let Some(pool) = self.pool.clone() {
            // Pool path (T37): acquire a warm (or freshly-spawned)
            // instance, call, then release. Release routes through the
            // pool's reset path — on success a fresh replacement
            // instance returns to the warm channel; on streaming opt-in
            // or reset failure the slot is released without
            // replenishment.
            let pooled = pool.acquire(self, wasm, cfg.clone()).await?;
            // Capture the id by value so the subsequent `release`
            // (which consumes `pooled`) and the call below see the
            // same handle.
            let id = pooled.id();
            let result = self.call_export_with_args(id, export, args).await;
            // Release regardless of call outcome — a guest trap should
            // not poison the warm pool (the pool's reset re-instantiates
            // from the cached module, so post-trap state is irrelevant).
            // Streaming spawns are dropped (never recycled) inside
            // `release`.
            pool.release(self, pooled, &cfg).await;
            result
        } else {
            // No pool attached — preserve the historical spawn + call
            // + terminate flow verbatim, including the auto-terminate
            // drop guard (api S-20).
            let id = self.spawn_instance(cfg, wasm).await?;
            self.call_export_with_args_then_terminate(id, export, args)
                .await
        }
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
        // exec H2: capture the per-tenant rollback handle BEFORE the wrapped
        // call so the guard's `Drop` (which cannot await) can mirror the
        // per-tenant decrement that the async `terminate` path performs by
        // reading the tenant off the instance. Only populated when a
        // per-tenant cap is configured — otherwise `tenant_counts` is never
        // charged and we leave it `None` so the guard skips the per-tenant
        // decrement entirely (matching `terminate`, which also skips the
        // per-instance lock in that case). We read the owning tenant off the
        // already-registered instance here (we *can* await at construction
        // time); the cancellation race the guard defends against only opens
        // once we enter the `call_export_with_args` await below.
        let tenant_rollback = if self.engine.config().max_instances_per_tenant.is_some() {
            // PERF finding 6: the owning tenant is cached on the registry value
            // at register time, so we read it from a plain field — no
            // per-instance lock. If the id is unknown the call below returns
            // `NotFound` and there is nothing to roll back, so a `None` here is
            // harmless.
            self.instances
                .get(&id)
                .map(|e| (Arc::clone(&self.tenant_counts), e.value().tenant_id))
        } else {
            None
        };
        let guard = AutoTerminateGuard {
            instances: Arc::clone(&self.instances),
            instance_count: Arc::clone(&self.instance_count),
            metrics: self.metrics.clone(),
            id,
            tenant_rollback,
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
    instances: Arc<DashMap<InstanceId, RegistryEntry>>,
    instance_count: Arc<AtomicUsize>,
    metrics: Option<TensorWasmMetrics>,
    id: InstanceId,
    /// Per-tenant rollback handle (exec H2). `Some((tenant_counts, tenant))`
    /// only when a per-tenant fairness cap is configured — in that case the
    /// async `terminate` path decrements the per-tenant count too, so a
    /// cancelled-future drop MUST mirror it or the tenant leaks a slot
    /// permanently (lockout once the cap is reached). The owning `TenantId`
    /// is captured up front (before the wrapped call) because `Drop` cannot
    /// await to read it off the instance the way `terminate` does. `None`
    /// when the cap is disabled — `tenant_counts` is never populated then and
    /// the guard skips the per-tenant decrement entirely.
    tenant_rollback: Option<(Arc<DashMap<TenantId, usize>>, TenantId)>,
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
        if let Some((_, entry)) = self.instances.remove(&self.id) {
            // HIGH finding 1 (mirror of `terminate`): flip the cooperative-
            // cancellation flag so a guest still executing inside the cancelled
            // `call_export_with_args` future is trapped at the next epoch tick,
            // not merely de-registered. (Future-drop alone already unwinds the
            // fiber at its next yield; this is the lock-free belt-and-suspenders
            // that also covers a guest parked between yields.)
            entry.cancelled.store(true, Ordering::Release);
            self.instance_count.fetch_sub(1, Ordering::AcqRel);
            // exec H2: mirror the per-tenant decrement the async `terminate`
            // path performs. Without this a cancelled/dropped invoke future
            // released the engine-wide slot but leaked the per-tenant slot,
            // so a tenant that accumulated cancelled invokes would hit its
            // `max_instances_per_tenant` cap with zero live instances and be
            // locked out permanently. Same saturating / prune-on-zero
            // semantics as `terminate` (both go through
            // `decrement_tenant_count`).
            if let Some((tenant_counts, tenant)) = &self.tenant_rollback {
                decrement_tenant_count(tenant_counts, *tenant);
            }
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
        exec.call_export_with_args(id, "noop", &[]).await.unwrap();
        exec.terminate(id).await.unwrap();
        assert_eq!(exec.live_count(), 0);
    }

    #[tokio::test]
    async fn unsupported_return_type_is_typed_error() {
        // LOW finding: an export returning a `v128` has no faithful JSON
        // encoding. It must surface a typed `UnsupportedReturnType` rather
        // than silently projecting to JSON `null` (which masked a genuine
        // signature mismatch — a `v128` return looked like a void return).
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let wasm = wat::parse_str(
            r#"(module (func (export "vec") (result v128)
                  (v128.const i32x4 1 2 3 4)))"#,
        )
        .unwrap();
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
            .await
            .unwrap();
        let err = exec
            .call_export_with_args(id, "vec", &[])
            .await
            .unwrap_err();
        match err {
            ExecError::UnsupportedReturnType(kind) => assert_eq!(kind, "v128"),
            other => panic!("expected UnsupportedReturnType(v128), got {other:?}"),
        }
        exec.terminate(id).await.unwrap();
    }

    #[tokio::test]
    async fn missing_export() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        let err = exec
            .call_export_with_args(id, "does_not_exist", &[])
            .await
            .unwrap_err();
        assert!(matches!(err, ExecError::MissingExport(_)));
    }

    /// Back-compat smoke test: the `#[deprecated]` `call_export` shim must
    /// still drive a guest to completion until v0.4 removes it. Tagged
    /// `#[allow(deprecated)]` because exercising the deprecated path is
    /// the *point* of this test — without it the test would emit the
    /// migration warning we ship to external callers.
    ///
    /// The `let _f = TensorWasmExecutor::call_export;` line at the bottom
    /// is a static proof-of-presence: if the `#[deprecated]` attribute
    /// were ever dropped again (as it was in merge `66af7db`), the
    /// `#[allow(deprecated)]` would become an "unnecessary attribute"
    /// warning under `-D unused`, signalling regression. The same logic
    /// applies to `call_export_then_terminate`.
    #[tokio::test]
    #[allow(deprecated)]
    async fn legacy_call_export_still_works() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        exec.call_export(id, "noop").await.unwrap();
        exec.terminate(id).await.unwrap();

        // Static proof-of-presence: both deprecated shims must be
        // resolvable as `fn` items. If a future refactor removes the
        // `#[deprecated]` attribute, the `#[allow(deprecated)]` above
        // is flagged as unused and CI breaks — exactly the regression
        // guard we want.
        let _f = TensorWasmExecutor::call_export;
        let _g = TensorWasmExecutor::call_export_then_terminate;
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
        assert!(
            text.contains("tensor_wasm_active_instances 1"),
            "got:\n{text}"
        );
        exec.terminate(id).await.unwrap();
        let text = metrics.encode_text();
        assert!(
            text.contains("tensor_wasm_instance_terminations_total 1"),
            "got:\n{text}"
        );
        assert!(
            text.contains("tensor_wasm_active_instances 0"),
            "got:\n{text}"
        );
    }

    #[test]
    fn exec_error_converts_to_tensor_wasm_error() {
        use tensor_wasm_core::error::TensorWasmError;
        let e = ExecError::NotFound(InstanceId(99));
        let b: TensorWasmError = e.into();
        assert!(matches!(b, TensorWasmError::Serialization(_)));
        // `TensorWasmError`'s Display is deliberately sanitised and does NOT
        // echo the inner string (it can carry host paths / pointers from
        // third-party crates). The raw context is reachable via `inner()`.
        assert!(
            b.inner().unwrap_or("").contains("instance not found"),
            "inner: {:?}",
            b.inner()
        );

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
    fn classify_instantiation_error_refines_memory_size_failure() {
        // A wasmtime error whose chain carries the pooling allocator's
        // memory-size signature must be re-tagged as ModuleMemoryTooLarge
        // (MED finding) so callers see a typed MemoryExhausted.
        let err = wasmtime::Error::msg("memory index 0 has a minimum that exceeds the limit");
        let classified = classify_instantiation_error(err, 64 * 1024 * 1024);
        match classified {
            ExecError::ModuleMemoryTooLarge {
                requested_bytes,
                limit_bytes,
            } => {
                assert_eq!(limit_bytes, 64 * 1024 * 1024);
                assert_eq!(requested_bytes, 64 * 1024 * 1024);
            }
            other => panic!("expected ModuleMemoryTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn classify_instantiation_error_passes_through_unrelated() {
        // An unrelated error (e.g. an import-link failure) must NOT be
        // misclassified — it flows through verbatim as ExecError::Wasmtime.
        let err = wasmtime::Error::msg("unknown import: `env::foo` has not been defined");
        let classified = classify_instantiation_error(err, 64 * 1024 * 1024);
        assert!(
            matches!(classified, ExecError::Wasmtime(_)),
            "unrelated errors must not be re-tagged, got {classified:?}",
        );
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
        assert!(!lim.memory_growing(0, 4096, Some(2048)).unwrap());
    }

    #[test]
    fn resource_limiter_rejects_huge_table_growth() {
        // 1 MiB engine cap → at 16 B/entry that's ~65k table entries max.
        // u32::MAX (~4.3 billion entries × 16 B = ~64 GiB) must be denied.
        let mut lim = TensorWasmResourceLimiter::new(1024 * 1024);
        assert!(!lim.table_growing(0, usize::MAX, None).unwrap());
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

    /// exec H2 regression: a cancelled/dropped invoke future must release the
    /// per-tenant slot, not just the engine-wide one. Before the fix,
    /// `AutoTerminateGuard::drop` decremented `instance_count` but left
    /// `tenant_counts` untouched, so a tenant that accumulated cancelled
    /// invokes would hit `max_instances_per_tenant` with zero live instances
    /// and be locked out permanently.
    ///
    /// We exercise the guard directly (the struct + its `Drop` are the unit
    /// the fix lives in) rather than racing a real future-cancellation, which
    /// is non-deterministic: we charge a per-tenant slot exactly the way
    /// `spawn_instance` does, build an *armed* guard with the per-tenant
    /// rollback populated the way `call_export_with_args_then_terminate` now
    /// does, drop it, and assert BOTH counters returned to zero.
    #[tokio::test]
    async fn dropped_invoke_releases_per_tenant_slot() {
        use crate::engine::EngineConfig;
        let cfg = EngineConfig {
            max_instances_per_tenant: Some(1),
            ..EngineConfig::default()
        };
        let engine = Arc::new(TensorWasmEngine::with_config(cfg).unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let tenant = TenantId(7);

        // Spawn one instance: charges engine-wide + per-tenant to 1, and (with
        // the cap == 1) the tenant is now at its ceiling.
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(tenant), &trivial_wasm())
            .await
            .unwrap();
        assert_eq!(exec.tenant_instance_count(tenant), 1);
        assert_eq!(exec.instances_len(), 1);

        // Build the guard exactly as `call_export_with_args_then_terminate`
        // does: armed, with the per-tenant rollback captured up front. Then
        // drop it to simulate the cancelled-future path (the guard stays
        // armed because the normal disarm-then-terminate sequence never ran).
        {
            let _guard = AutoTerminateGuard {
                instances: Arc::clone(&exec.instances),
                instance_count: Arc::clone(&exec.instance_count),
                metrics: exec.metrics.clone(),
                id,
                tenant_rollback: Some((Arc::clone(&exec.tenant_counts), tenant)),
                armed: true,
            };
            // dropped here
        }

        // Both counters must be back to zero — the tenant is no longer locked
        // out and the engine-wide slot was freed.
        assert_eq!(
            exec.tenant_instance_count(tenant),
            0,
            "per-tenant slot leaked on dropped invoke (exec H2)"
        );
        assert_eq!(exec.instances_len(), 0, "engine-wide slot leaked");
        assert_eq!(exec.live_count(), 0, "registry entry leaked");

        // The tenant must be able to spawn again (proves no lockout).
        let id2 = exec
            .spawn_instance(SpawnConfig::for_tenant(tenant), &trivial_wasm())
            .await
            .expect("tenant should not be locked out after a cancelled invoke");
        exec.terminate(id2).await.unwrap();
        assert_eq!(exec.tenant_instance_count(tenant), 0);
    }

    /// exec H2 companion: with NO per-tenant cap configured, the guard's
    /// `tenant_rollback` is `None` and the drop path must not touch
    /// `tenant_counts` (it is never populated) — only the engine-wide slot is
    /// released, matching the async `terminate` path which also skips the
    /// per-tenant decrement when the cap is unset.
    #[tokio::test]
    async fn dropped_invoke_no_tenant_cap_releases_engine_slot_only() {
        let engine = Arc::new(TensorWasmEngine::new().unwrap());
        let exec = TensorWasmExecutor::new(engine);
        let tenant = TenantId(3);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(tenant), &trivial_wasm())
            .await
            .unwrap();
        assert_eq!(exec.instances_len(), 1);
        // No per-tenant cap => tenant_counts never charged.
        assert_eq!(exec.tenant_instance_count(tenant), 0);
        {
            let _guard = AutoTerminateGuard {
                instances: Arc::clone(&exec.instances),
                instance_count: Arc::clone(&exec.instance_count),
                metrics: exec.metrics.clone(),
                id,
                tenant_rollback: None,
                armed: true,
            };
        }
        assert_eq!(exec.instances_len(), 0, "engine-wide slot leaked");
        assert_eq!(exec.tenant_instance_count(tenant), 0);
    }
}
