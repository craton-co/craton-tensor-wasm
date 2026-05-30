// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! [`TensorWasmEngine`] — a [`wasmtime::Engine`] wrapper preconfigured for TensorWasm.
//!
//! - Async execution (cooperative fuel via epoch-based interruption).
//! - Custom [`MemoryCreator`](wasmtime::MemoryCreator) so linear memory is
//!   carved from [`tensor_wasm_mem::wasm_memory::TensorWasmMemoryCreator`] (CUDA Unified
//!   Memory on supported hosts; plain Box on others).
//! - Epoch ticker: a background Tokio task increments the engine's epoch
//!   counter every [`TensorWasmEngine::EPOCH_TICK`] so calls past their deadline are
//!   interrupted promptly.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_mem::wasm_memory::TensorWasmMemoryCreator;
use tokio::task::JoinHandle;
use wasmtime::{
    Config, Engine, InstanceAllocationStrategy, MpkEnabled, PoolingAllocationConfig, Strategy,
};

/// Default epoch tick. Matches the plan's 10 ms cadence.
const DEFAULT_EPOCH_TICK: Duration = Duration::from_millis(10);

/// Explicit ceiling on the wasm operand/value stack a single call may use,
/// in bytes (LOW/hardening finding). Wasmtime applies a default stack ceiling
/// today, but pinning it here makes the resource contract independent of any
/// future wasmtime default change. 1 MiB is generous for the deeply-nested
/// recursion / large-frame guests we expect while still bounding host stack
/// consumption per fiber. Kept STRICTLY BELOW the async fiber stack reserved
/// by wasmtime: under `async_support` wasmtime requires
/// `max_wasm_stack <= async_stack_size`, and the async stack also has to hold
/// the host frames of any async host import. We do not override
/// `async_stack_size`, so it stays at wasmtime's default (2 MiB), leaving
/// ample headroom (2 MiB >= 1 MiB) for host frames on top of the guest stack.
const MAX_WASM_STACK_BYTES: usize = 1_048_576;

/// Approximate host-memory cost of a single wasm table entry on wasmtime
/// (a tagged pointer plus a type-index slot). Kept in lockstep with the
/// same constant in
/// [`TensorWasmResourceLimiter::table_growing`](crate::executor::TensorWasmResourceLimiter)
/// so the pooling allocator's `table_elements` ceiling is derived from the
/// same per-instance budget the limiter enforces on the `UnifiedBuffer`
/// path (MED finding — the two backends must agree on the table budget).
pub(crate) const TABLE_ENTRY_BYTES: u64 = 16;

/// Selects the linear-memory backing strategy for the engine.
///
/// The two modes are mutually exclusive at the Wasmtime level:
/// `with_host_memory` (used by [`MemoryBackend::UnifiedBuffer`]) cannot coexist
/// with the pooling allocator (required by [`MemoryBackend::PoolingMpk`]).
/// Operators pick the mode that fits their workload.
#[derive(Debug, Clone, Default)]
pub enum MemoryBackend {
    /// Host-provided UnifiedBuffer-backed linear memory via `with_host_memory`.
    /// Required for the GPU integration path (kernels read/write the same
    /// allocation the Wasm guest sees). DOES NOT support MPK — Wasmtime's
    /// pooling+MPK machinery is mutually exclusive with custom MemoryCreator.
    #[default]
    UnifiedBuffer,
    /// Wasmtime's pooling allocator with MPK (memory protection keys).
    /// Trades the GPU integration path for intra-process Wasm isolation via
    /// CPU PKU. Suitable for CPU-only or batch-GPU workloads where kernel
    /// launches don't share memory with Wasm at byte level.
    PoolingMpk {
        /// Maximum total memories tracked by the pooling allocator.
        max_memories: u32,
        /// Bytes per memory slot.
        memory_bytes: usize,
    },
}

/// Configuration knobs for the engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum allocated linear memory per instance (bytes).
    pub max_memory_bytes: usize,
    /// Period between background `increment_epoch` ticks.
    pub epoch_tick: Duration,
    /// Compilation strategy. `Strategy::Cranelift` for production.
    pub strategy: Strategy,
    /// Enable Wasm component model.
    pub component_model: bool,
    /// Linear-memory backing strategy. See [`MemoryBackend`] for the
    /// UnifiedBuffer vs PoolingMpk trade-off.
    pub backend: MemoryBackend,
    /// Maximum number of compiled-module cache entries retained per
    /// executor before LRU eviction kicks in. Closes exec S-5: an
    /// unbounded `DashMap<digest, Module>` lets a misbehaving tenant
    /// pin arbitrarily many compiled modules (each multi-MiB of host
    /// RAM) by submitting unique wasm bytes in a loop. 1024 is enough
    /// to hold the working set of a typical multi-tenant deployment
    /// while bounding the worst case at ~a few GiB of compiled-code
    /// pages.
    pub max_module_cache_entries: usize,
    /// Hard upper bound on the number of concurrently-live instances
    /// the executor will admit. Closes exec S-10: an unbounded
    /// `DashMap<InstanceId, ...>` lets a tenant spawn instances in a
    /// loop until the host OOMs. `None` disables the cap (useful for
    /// tests / single-tenant deployments); production callers should
    /// keep the default ceiling. When the limit is hit `spawn_instance`
    /// returns [`crate::executor::ExecError::CapacityExhausted`].
    pub max_instances: Option<usize>,
    /// Pre-compile cap on the byte length of a submitted Wasm module.
    /// Bytes above this are rejected with
    /// [`crate::executor::ExecError::ModuleTooLarge`] *before*
    /// `Module::from_binary` runs, preventing pathological code
    /// sections from forcing Cranelift to burn arbitrary CPU on
    /// adversarial input. Default is
    /// [`crate::executor::MAX_MODULE_BYTES`] (64 MiB); embedders may
    /// tighten further but the constant is the documented floor.
    pub max_module_bytes: usize,
    /// Upper bound on the number of `Module::from_binary` (Cranelift)
    /// compiles allowed to run concurrently on the Tokio blocking pool.
    ///
    /// Each compile is offloaded via [`tokio::task::spawn_blocking`]; the
    /// blocking pool is a shared process resource (default 512 threads).
    /// Without a bound, an adversary submitting a stream of *unique* large
    /// modules — each a cache miss, each a multi-hundred-millisecond
    /// Cranelift run — can saturate the pool and starve every other
    /// blocking operation in the process. This cap is independent of
    /// [`Self::max_instances`] (which bounds live instances, not in-flight
    /// compiles) and is enforced by a per-executor
    /// [`tokio::sync::Semaphore`]. `None` selects a default derived from
    /// [`std::thread::available_parallelism`] (floored at 1) at executor
    /// construction time.
    pub max_concurrent_compiles: Option<usize>,
    /// Opt-in: activate the auto-offload Wasm rewrite on the spawn path.
    ///
    /// When `false` (the default) the
    /// [`auto_offload::analyse`](crate::auto_offload::analyse) pass remains
    /// consultation-only — it emits `tracing` verdicts but Wasmtime's
    /// Cranelift output is never replaced, exactly matching the historical
    /// behaviour. When `true`,
    /// [`TensorWasmExecutor::spawn_instance`](crate::executor::TensorWasmExecutor::spawn_instance)
    /// runs the analyser and, if any function is flagged for offload, feeds
    /// the module through
    /// [`tensor_wasm_jit::rewrite::rewrite_wasm`] to produce a
    /// trampoline-augmented module which is instantiated *instead of* the
    /// original. The original bytes are always retained as a fallback: any
    /// analysis or rewrite failure (or a rewrite that swaps nothing) is
    /// logged and the spawn proceeds with the unmodified module, so enabling
    /// this flag can never fail a spawn that would otherwise have succeeded.
    ///
    /// Requires a [`KernelCache`](tensor_wasm_jit::cache::KernelCache)
    /// attached to the executor via
    /// [`TensorWasmExecutor::with_jit_cache`](crate::executor::TensorWasmExecutor::with_jit_cache)
    /// for the rewritten guest's `tensor-wasm:jit/host` imports to link;
    /// without one, the rewrite is skipped (the trampoline imports would be
    /// unlinkable) and the original module is used.
    pub auto_offload: bool,
    /// Detector thresholds the auto-offload activation path uses to decide
    /// which function bodies are offload candidates.
    ///
    /// `None` (the default) selects
    /// [`tensor_wasm_jit::detector::DetectorConfig::default`] — the
    /// production-conservative thresholds. Embedders (and tests) that want a
    /// more aggressive offload policy can supply tuned thresholds here; the
    /// same config is threaded into BOTH the consultation pass
    /// ([`auto_offload::analyse_with_config`](crate::auto_offload::analyse_with_config))
    /// and the
    /// [`tensor_wasm_jit::rewrite::RewriteOptions::detector`] used for the
    /// rewrite, so the consultation verdict and the rewrite always agree on
    /// which functions to swap. Ignored entirely when
    /// [`Self::auto_offload`] is `false`.
    pub auto_offload_detector: Option<tensor_wasm_jit::detector::DetectorConfig>,
    /// Optional per-tenant cap on the number of concurrently-live instances
    /// a single [`TenantId`](tensor_wasm_core::types::TenantId) may hold.
    ///
    /// Complements the engine-wide [`Self::max_instances`] ceiling with a
    /// fairness bound: one tenant spawning in a loop cannot starve every
    /// other tenant of the shared instance budget. Enforced in the same
    /// admission path as `max_instances` (keyed by the spawning
    /// [`SpawnConfig::tenant_id`](crate::executor::SpawnConfig::tenant_id)),
    /// with the same charge-before-compile / roll-back-on-failure
    /// accounting. When a tenant exceeds its cap the spawn is refused with
    /// [`ExecError::CapacityExhausted`](crate::executor::ExecError::CapacityExhausted)
    /// (reused for the per-tenant case to keep the cross-crate error mapping
    /// non-breaking; the offending tenant is logged server-side) carrying the
    /// tenant's own count and per-tenant `limit`, without affecting any other
    /// tenant. `None` (the default) disables the per-tenant cap — only the
    /// engine-wide `max_instances` applies.
    pub max_instances_per_tenant: Option<usize>,
}

impl EngineConfig {
    /// The effective per-instance linear-memory cap, in bytes, reconciling
    /// the engine-wide [`Self::max_memory_bytes`] ceiling with the pooling
    /// allocator's own per-slot byte size when
    /// [`MemoryBackend::PoolingMpk`] is selected (MED finding).
    ///
    /// On the [`MemoryBackend::UnifiedBuffer`] path this is simply
    /// `max_memory_bytes`. On the pooling path the physical slot size
    /// (`memory_bytes`) is an independent hard ceiling the allocator
    /// enforces at instantiation; a module larger than that slot would fail
    /// to instantiate regardless of `max_memory_bytes`. Taking the minimum
    /// of the two means the executor's pre-instantiation module check
    /// ([`check_module_memory_within_cap`](crate::executor)) rejects an
    /// oversized module with the typed
    /// [`ExecError::ModuleMemoryTooLarge`](crate::executor::ExecError::ModuleMemoryTooLarge)
    /// *before* the pooling allocator can surface an opaque
    /// `ExecError::Wasmtime`, and the per-store
    /// [`TensorWasmResourceLimiter`](crate::executor::TensorWasmResourceLimiter)
    /// caps `memory.grow` against the same reconciled value.
    pub fn effective_memory_cap(&self) -> usize {
        match self.backend {
            MemoryBackend::UnifiedBuffer => self.max_memory_bytes,
            MemoryBackend::PoolingMpk { memory_bytes, .. } => {
                self.max_memory_bytes.min(memory_bytes)
            }
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024,
            epoch_tick: DEFAULT_EPOCH_TICK,
            strategy: Strategy::Cranelift,
            component_model: true,
            backend: MemoryBackend::default(),
            max_module_cache_entries: 1024,
            max_instances: Some(10_000),
            max_module_bytes: crate::executor::MAX_MODULE_BYTES,
            // None => derive from available_parallelism at executor
            // construction. Keeps the default tied to the host's core
            // count without pulling in a `num_cpus` dependency.
            max_concurrent_compiles: None,
            // Consultation-only by default: the analyser still runs and
            // emits verdicts, but Wasmtime's Cranelift output is not
            // replaced unless an embedder opts in.
            auto_offload: false,
            // None => default detector thresholds when the swap is enabled.
            auto_offload_detector: None,
            // No per-tenant fairness cap by default — only the engine-wide
            // `max_instances` ceiling applies.
            max_instances_per_tenant: None,
        }
    }
}

/// A configured [`wasmtime::Engine`] plus the background epoch ticker that
/// drives interruption.
pub struct TensorWasmEngine {
    engine: Engine,
    ticker_handle: Option<JoinHandle<()>>,
    config: EngineConfig,
}

impl TensorWasmEngine {
    /// Default epoch tick interval.
    pub const EPOCH_TICK: Duration = DEFAULT_EPOCH_TICK;

    /// Construct an engine with default configuration.
    pub fn new() -> Result<Self, wasmtime::Error> {
        Self::with_config(EngineConfig::default())
    }

    /// Construct an engine with explicit configuration.
    pub fn with_config(cfg: EngineConfig) -> Result<Self, wasmtime::Error> {
        let mut wt_cfg = Config::new();
        // wasmtime 45: async support is enabled by the `async` cargo feature
        // and the `*_async` call sites; `Config::async_support` is deprecated
        // and a no-op, so it is no longer set here.
        wt_cfg.epoch_interruption(true);
        wt_cfg.consume_fuel(false);
        wt_cfg.wasm_component_model(cfg.component_model);
        wt_cfg.strategy(cfg.strategy);

        // ─── Wasm stack ceiling (LOW/hardening pin) ─────────────────────────
        //
        // Pin the per-call wasm stack limit EXPLICITLY rather than relying on
        // wasmtime's built-in default, so a future wasmtime bump that widens
        // (or narrows) that default cannot silently change our resource
        // contract — the same reasoning that motivates the proposal pins
        // below. 1 MiB bounds host stack growth per fiber while comfortably
        // accommodating deeply-nested guests. Under `async_support` wasmtime
        // requires this to be <= `async_stack_size` (default 2 MiB, which we
        // do not override), with the difference reserved for host frames on
        // the fiber; 1 MiB leaves 1 MiB of headroom, so the invariant holds.
        wt_cfg.max_wasm_stack(MAX_WASM_STACK_BYTES);

        // ─── Wasm proposal deny-list (security pin) ─────────────────────────
        //
        // These flags are pinned because:
        //   (a) the hardened multi-tenant trust model of this crate depends
        //       on them — enabling a proposal we have not audited (threads,
        //       memory64, multi-memory, relaxed-simd, tail-call, GC, typed
        //       function references) would widen the sandbox attack surface
        //       and may invalidate isolation assumptions (e.g. `wasm_threads`
        //       interacts with our pooling/MPK backend in non-obvious ways);
        //   (b) a future `wasmtime` minor/patch bump must not silently change
        //       behaviour. If wasmtime flips a default upstream, we want the
        //       guest contract to remain identical until we explicitly opt in.
        //
        // The positive flags below are the proposals this codebase *does*
        // consume; pinning them to `true` defends against the symmetric
        // failure mode (a future bump silently disabling something we rely
        // on).
        // Proposal flags exposed by the workspace's wasmtime feature set
        // (`async`, `cranelift`, `component-model`, `runtime`). The list is
        // intentionally narrow — additional flags (`wasm_threads`,
        // `wasm_gc`, `wasm_function_references`, `wasm_reference_types`) are
        // gated behind feature flags we do NOT enable in this workspace, so
        // the corresponding proposals are already compiled out of the engine
        // and cannot be activated by config alone. If those wasmtime
        // features ever get pulled in, mirror them here with `_(false)`.
        wt_cfg.wasm_memory64(false);
        wt_cfg.wasm_multi_memory(false);
        wt_cfg.wasm_relaxed_simd(false);
        wt_cfg.wasm_tail_call(false);
        // Explicitly KEEP the proposals we depend on, so a wasmtime bump
        // cannot silently flip them:
        wt_cfg.wasm_simd(true);
        wt_cfg.wasm_bulk_memory(true);
        wt_cfg.wasm_multi_value(true);

        match cfg.backend {
            MemoryBackend::UnifiedBuffer => {
                let memory_creator = Arc::new(TensorWasmMemoryCreator::default());
                wt_cfg.with_host_memory(memory_creator);
                wt_cfg.guard_before_linear_memory(false);
                // wasmtime 45 consolidated the static/dynamic memory tuning
                // knobs: `static_memory_maximum_size` -> `memory_reservation`
                // and `dynamic_memory_guard_size` -> `memory_guard_size`.
                // Both are 0 here because the host `MemoryCreator` above owns
                // the allocation (no wasmtime-side reservation or guard page).
                wt_cfg.memory_reservation(0);
                wt_cfg.memory_guard_size(0);
            }
            MemoryBackend::PoolingMpk {
                max_memories,
                memory_bytes,
            } => {
                // Pooling owns the memory backing — do NOT install a host
                // memory creator, and leave Wasmtime's default guard sizes
                // in place (the pooling allocator depends on them).
                //
                // Reconcile the pooling per-slot byte size with the engine's
                // per-instance cap (MED finding). Two independent dials —
                // `max_memory_bytes` (the `ResourceLimiter` ceiling and the
                // module pre-instantiation check) and the pooling
                // `memory_bytes` (the allocator's physical slot size) — were
                // not reconciled: a module that passed the
                // `max_memory_bytes` check could still exceed the pooling
                // slot and fail instantiation with an opaque allocator error.
                // The effective per-instance memory cap is the *minimum* of
                // the two, and the pooling slot is sized to that value so the
                // allocator never has to refuse a module the cap check
                // already admitted. `effective_memory_cap()` computes the
                // same minimum that `check_module_memory_within_cap` validates
                // against in the executor.
                let effective = cfg.max_memory_bytes.min(memory_bytes);
                let mut pooling = PoolingAllocationConfig::default();
                pooling.total_memories(max_memories);
                pooling.max_memory_size(effective);
                // Derive the table limits from the same reconciled
                // per-instance budget the executor's `ResourceLimiter` uses
                // (`effective / TABLE_ENTRY_BYTES`, matching
                // `TensorWasmResourceLimiter::table_growing`), so a module
                // admitted by the limiter is not then refused by the pooling
                // allocator's own table ceiling. One table slot per memory
                // slot. The pooling allocator caps `table_elements` at a
                // `u32`, so saturate on the cast.
                pooling.total_tables(max_memories);
                let table_elems = (effective as u64 / TABLE_ENTRY_BYTES)
                    .try_into()
                    .unwrap_or(u32::MAX);
                pooling.table_elements(table_elems);
                pooling.memory_protection_keys(MpkEnabled::Auto);
                wt_cfg.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling));
            }
        }

        let engine = Engine::new(&wt_cfg)?;
        let mut this = Self {
            engine,
            ticker_handle: None,
            config: cfg,
        };
        // exec S-4: auto-spawn the epoch ticker if we're already inside a
        // Tokio runtime. Without the ticker, any `SpawnConfig::with_deadline`
        // becomes silently inert — deadlines cannot fire. Operators who
        // construct the engine OUTSIDE a runtime (synchronous startup) still
        // have to call `spawn_epoch_ticker()` after `Runtime::block_on` /
        // `Runtime::new()`; the `spawn_instance` path also emits a loud
        // `tracing::error!` (see executor.rs) the first time a deadline is
        // requested with no ticker running, so the silent-failure mode is
        // closed defence-in-depth.
        if tokio::runtime::Handle::try_current().is_ok() {
            this.spawn_epoch_ticker();
        }
        Ok(this)
    }

    /// Borrow the underlying wasmtime Engine. Cheap (it's `Arc`-shaped internally).
    pub fn inner(&self) -> &Engine {
        &self.engine
    }

    /// Borrow the engine config used at construction.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Spawn a background Tokio task that periodically increments the engine
    /// epoch counter. Must be called from inside a Tokio runtime.
    ///
    /// Idempotent: if a ticker is already running this is a no-op.
    pub fn spawn_epoch_ticker(&mut self) {
        if self.ticker_handle.is_some() {
            return;
        }
        let engine = self.engine.clone();
        let tick = self.config.epoch_tick;
        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(tick).await;
                engine.increment_epoch();
                // No per-tick trace event: at the default 10 ms cadence this
                // floods structured-logging backends. Operators wanting to
                // verify the ticker is alive should look at engine span
                // counts or use `TensorWasmEngine::tick` from a probe.
            }
        });
        self.ticker_handle = Some(handle);
    }

    /// Stop the epoch ticker if one is running.
    pub fn stop_epoch_ticker(&mut self) {
        if let Some(h) = self.ticker_handle.take() {
            h.abort();
        }
    }

    /// True if [`spawn_epoch_ticker`](Self::spawn_epoch_ticker) has been called
    /// on this engine and the ticker has not been
    /// [`stop_epoch_ticker`](Self::stop_epoch_ticker)'d.
    ///
    /// Used by [`TensorWasmExecutor`](crate::executor::TensorWasmExecutor) to
    /// emit a one-shot operator warning the first time an instance is spawned
    /// on an engine whose ticker is not running (deadlines are otherwise
    /// silently inert).
    pub fn is_epoch_ticker_running(&self) -> bool {
        self.ticker_handle
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }

    /// Increment the epoch once manually. Useful for tests that do not want
    /// to wait for the background ticker.
    pub fn tick(&self) {
        self.engine.increment_epoch();
    }
}

impl Drop for TensorWasmEngine {
    fn drop(&mut self) {
        self.stop_epoch_ticker();
    }
}

impl Default for TensorWasmEngine {
    fn default() -> Self {
        Self::new().expect("default TensorWasmEngine construction")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn engine_constructs() {
        let _engine = TensorWasmEngine::new().expect("construct");
    }

    #[tokio::test]
    async fn ticker_is_idempotent() {
        let mut engine = TensorWasmEngine::new().unwrap();
        engine.spawn_epoch_ticker();
        engine.spawn_epoch_ticker();
        engine.stop_epoch_ticker();
        engine.stop_epoch_ticker();
    }

    #[tokio::test]
    async fn manual_tick() {
        let engine = TensorWasmEngine::new().unwrap();
        engine.tick();
        engine.tick();
        // No assertions on the engine's internal epoch counter — wasmtime does
        // not expose it. We only verify the call doesn't panic.
    }

    #[test]
    fn default_config_values() {
        let c = EngineConfig::default();
        assert_eq!(c.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(c.epoch_tick, Duration::from_millis(10));
        assert!(c.component_model);
        assert!(matches!(c.backend, MemoryBackend::UnifiedBuffer));
    }

    #[test]
    fn effective_memory_cap_unified_is_max_memory_bytes() {
        let cfg = EngineConfig {
            max_memory_bytes: 128 * 1024 * 1024,
            backend: MemoryBackend::UnifiedBuffer,
            ..EngineConfig::default()
        };
        assert_eq!(cfg.effective_memory_cap(), 128 * 1024 * 1024);
    }

    #[test]
    fn effective_memory_cap_pooling_takes_minimum() {
        // Pooling slot smaller than the engine cap → slot wins.
        let cfg = EngineConfig {
            max_memory_bytes: 256 * 1024 * 1024,
            backend: MemoryBackend::PoolingMpk {
                max_memories: 8,
                memory_bytes: 64 * 1024 * 1024,
            },
            ..EngineConfig::default()
        };
        assert_eq!(cfg.effective_memory_cap(), 64 * 1024 * 1024);

        // Engine cap smaller than the pooling slot → engine cap wins.
        let cfg = EngineConfig {
            max_memory_bytes: 16 * 1024 * 1024,
            backend: MemoryBackend::PoolingMpk {
                max_memories: 8,
                memory_bytes: 64 * 1024 * 1024,
            },
            ..EngineConfig::default()
        };
        assert_eq!(cfg.effective_memory_cap(), 16 * 1024 * 1024);
    }

    #[tokio::test]
    async fn engine_constructs_with_unified_backend() {
        let cfg = EngineConfig {
            backend: MemoryBackend::UnifiedBuffer,
            ..EngineConfig::default()
        };
        let engine = TensorWasmEngine::with_config(cfg);
        assert!(
            engine.is_ok(),
            "engine should construct: {:?}",
            engine.err()
        );
    }

    #[tokio::test]
    async fn engine_constructs_with_pooling_mpk_backend() {
        let cfg = EngineConfig {
            backend: MemoryBackend::PoolingMpk {
                max_memories: 32,
                memory_bytes: 64 * 1024,
            },
            ..EngineConfig::default()
        };
        let engine = TensorWasmEngine::with_config(cfg);
        assert!(
            engine.is_ok(),
            "engine should construct: {:?}",
            engine.err()
        );
    }
}
