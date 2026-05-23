//! [`BaliEngine`] — a [`wasmtime::Engine`] wrapper preconfigured for Bali.
//!
//! - Async execution (cooperative fuel via epoch-based interruption).
//! - Custom [`MemoryCreator`](wasmtime::MemoryCreator) so linear memory is
//!   carved from [`bali_mem::wasm_memory::BaliMemoryCreator`] (CUDA Unified
//!   Memory on supported hosts; plain Box on others).
//! - Epoch ticker: a background Tokio task increments the engine's epoch
//!   counter every [`BaliEngine::EPOCH_TICK`] so calls past their deadline are
//!   interrupted promptly.

use std::sync::Arc;
use std::time::Duration;

use bali_mem::wasm_memory::BaliMemoryCreator;
use tokio::task::JoinHandle;
use tracing::trace;
use wasmtime::{Config, Engine, Strategy};

/// Default epoch tick. Matches the plan's 10 ms cadence.
const DEFAULT_EPOCH_TICK: Duration = Duration::from_millis(10);

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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024,
            epoch_tick: DEFAULT_EPOCH_TICK,
            strategy: Strategy::Cranelift,
            component_model: true,
        }
    }
}

/// A configured [`wasmtime::Engine`] plus the background epoch ticker that
/// drives interruption.
pub struct BaliEngine {
    engine: Engine,
    ticker_handle: Option<JoinHandle<()>>,
    config: EngineConfig,
}

impl BaliEngine {
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

        let memory_creator = Arc::new(BaliMemoryCreator::default());
        wt_cfg.with_host_memory(memory_creator);
        wt_cfg.guard_before_linear_memory(false);
        wt_cfg.static_memory_maximum_size(0);
        wt_cfg.dynamic_memory_guard_size(0);

        let engine = Engine::new(&wt_cfg)?;
        Ok(Self {
            engine,
            ticker_handle: None,
            config: cfg,
        })
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
                trace!(target: "bali_exec::engine", "epoch incremented");
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

    /// Increment the epoch once manually. Useful for tests that do not want
    /// to wait for the background ticker.
    pub fn tick(&self) {
        self.engine.increment_epoch();
    }
}

impl Drop for BaliEngine {
    fn drop(&mut self) {
        self.stop_epoch_ticker();
    }
}

impl Default for BaliEngine {
    fn default() -> Self {
        Self::new().expect("default BaliEngine construction")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn engine_constructs() {
        let _engine = BaliEngine::new().expect("construct");
    }

    #[tokio::test]
    async fn ticker_is_idempotent() {
        let mut engine = BaliEngine::new().unwrap();
        engine.spawn_epoch_ticker();
        engine.spawn_epoch_ticker();
        engine.stop_epoch_ticker();
        engine.stop_epoch_ticker();
    }

    #[tokio::test]
    async fn manual_tick() {
        let engine = BaliEngine::new().unwrap();
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
    }
}
