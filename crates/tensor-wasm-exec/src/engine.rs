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
        wt_cfg.async_support(true);
        wt_cfg.epoch_interruption(true);
        wt_cfg.consume_fuel(false);
        wt_cfg.wasm_component_model(cfg.component_model);
        wt_cfg.strategy(cfg.strategy);

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
                wt_cfg.static_memory_maximum_size(0);
                wt_cfg.dynamic_memory_guard_size(0);
            }
            MemoryBackend::PoolingMpk {
                max_memories,
                memory_bytes,
            } => {
                // Pooling owns the memory backing — do NOT install a host
                // memory creator, and leave Wasmtime's default guard sizes
                // in place (the pooling allocator depends on them).
                let mut pooling = PoolingAllocationConfig::default();
                pooling.total_memories(max_memories);
                pooling.max_memory_size(memory_bytes);
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
