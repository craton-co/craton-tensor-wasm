//! [`BaliExecutor`] — async executor for Bali Wasm instances.
//!
//! Owns a shared [`BaliEngine`] and a registry of live [`BaliInstance`]s
//! keyed by [`InstanceId`]. Exposes the trio of operations
//! [`BaliExecutor::spawn_instance`], [`BaliExecutor::call_export`], and
//! [`BaliExecutor::terminate`] — all async, all driven from the calling
//! Tokio runtime.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bali_core::metrics::BaliMetrics;
use bali_core::types::{InstanceId, TenantId};
use dashmap::DashMap;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument};
use wasmtime::{Module, Store};

use crate::engine::BaliEngine;
use crate::instance::{BaliInstance, InstanceState};

/// Errors raised by the executor.
#[derive(Debug, Error)]
pub enum ExecError {
    /// Wasmtime returned an error during compile / instantiate / call.
    #[error("wasmtime: {0}")]
    Wasmtime(#[from] wasmtime::Error),
    /// Looked up an instance that does not exist (or has terminated).
    #[error("no such instance: {0}")]
    NotFound(InstanceId),
    /// Looked up an export that the instance does not provide.
    #[error("instance has no export `{0}`")]
    MissingExport(String),
    /// The instance ran past its deadline before the call completed.
    #[error("instance {0} exceeded deadline")]
    Timeout(InstanceId),
}

impl From<ExecError> for bali_core::error::BaliError {
    fn from(e: ExecError) -> Self {
        use bali_core::error::BaliError;
        match e {
            // Wasmtime errors most commonly arise from instance traps and
            // compile failures. Without inspecting the inner error we can't
            // distinguish; pick `WasmTrap` as the catch-all per the existing
            // bali-core variant names.
            ExecError::Wasmtime(err) => BaliError::WasmTrap(format!("{err}")),
            ExecError::NotFound(id) => {
                BaliError::Serialization(format!("instance not found: {id}"))
            }
            ExecError::MissingExport(name) => {
                BaliError::Serialization(format!("instance missing export: {name}"))
            }
            // We have no real elapsed/deadline figures to plug in at this layer
            // — surface zeros so callers can still match on `KernelTimeout` and
            // the retryability classification works.
            ExecError::Timeout(_id) => BaliError::KernelTimeout {
                elapsed_ms: 0,
                deadline_ms: 0,
            },
        }
    }
}

/// Configuration passed to [`BaliExecutor::spawn_instance`].
#[derive(Debug, Clone)]
pub struct SpawnConfig {
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Optional per-call deadline.
    pub deadline: Option<Duration>,
}

impl SpawnConfig {
    /// Construct with just a tenant and no deadline.
    pub fn for_tenant(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            deadline: None,
        }
    }

    /// Add a deadline relative to "now at spawn time".
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// The async executor.
#[derive(Clone)]
pub struct BaliExecutor {
    engine: Arc<BaliEngine>,
    instances: Arc<DashMap<InstanceId, Arc<Mutex<BaliInstance>>>>,
    next_instance_id: Arc<AtomicU64>,
    /// Optional metrics handle. When `Some`, spawn/terminate operations
    /// increment the corresponding Prometheus counters / gauges.
    metrics: Option<BaliMetrics>,
}

impl BaliExecutor {
    /// Construct an executor over the given shared engine.
    pub fn new(engine: Arc<BaliEngine>) -> Self {
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            metrics: None,
        }
    }

    /// Construct an executor that publishes spawn/terminate events to the
    /// supplied [`BaliMetrics`] registry. Metric handles are cheaply cloneable;
    /// pass a clone of the process-wide registry.
    pub fn with_metrics(engine: Arc<BaliEngine>, metrics: BaliMetrics) -> Self {
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            metrics: Some(metrics),
        }
    }

    /// Borrow the underlying engine.
    pub fn engine(&self) -> &BaliEngine {
        &self.engine
    }

    /// Number of currently live instances.
    pub fn live_count(&self) -> usize {
        self.instances.len()
    }

    /// Compile + instantiate a Wasm module. Returns the assigned [`InstanceId`].
    #[instrument(skip(self, wasm), fields(tenant = %cfg.tenant_id, instance_id = tracing::field::Empty))]
    pub async fn spawn_instance(
        &self,
        cfg: SpawnConfig,
        wasm: &[u8],
    ) -> Result<InstanceId, ExecError> {
        let id = InstanceId(self.next_instance_id.fetch_add(1, Ordering::Relaxed) as u128);
        tracing::Span::current().record("instance_id", tracing::field::display(id));
        let mut state = InstanceState::new(cfg.tenant_id, id);
        if let Some(d) = cfg.deadline {
            state = state.with_deadline(Instant::now() + d);
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
            Some(d) => {
                // Convert duration → number of epoch ticks, rounding up,
                // with a floor of 1 so a sub-tick deadline still terminates.
                let d_ms = d.as_millis();
                let t_ms = tick.as_millis().max(1);
                d_ms.div_ceil(t_ms).max(1) as u64
            }
            // No deadline → effectively unbounded. u64::MAX is fine: the
            // engine's epoch counter would need ~5.8 billion years at 10 ms/
            // tick to reach it, so the deadline will never trip in practice.
            None => u64::MAX,
        };
        let module = Module::from_binary(self.engine.inner(), wasm)?;
        let mut store = Store::new(self.engine.inner(), state);
        store.set_epoch_deadline(epoch_deadline_ticks);
        let instance = wasmtime::Instance::new_async(&mut store, &module, &[]).await?;
        let bi = BaliInstance::new(store, instance);
        self.instances.insert(id, Arc::new(Mutex::new(bi)));
        if let Some(m) = &self.metrics {
            m.instance_spawns_total().inc();
            m.active_instances().inc();
        }
        info!(target: "bali_exec::executor", tenant = %cfg.tenant_id, instance = %id, "instance spawned");
        Ok(id)
    }

    /// Invoke `export` with no arguments and no return value.
    ///
    /// This is the minimal signature needed for the 100-instance integration
    /// test. Richer signatures arrive in S17 (HTTP API) and S18 (CLI).
    #[instrument(skip(self), fields(instance = %id, export = %export))]
    pub async fn call_export(&self, id: InstanceId, export: &str) -> Result<(), ExecError> {
        let handle = self
            .instances
            .get(&id)
            .ok_or(ExecError::NotFound(id))?
            .value()
            .clone();
        let mut guard = handle.lock().await;
        let wasmtime_instance = *guard.wasmtime_instance();
        let func = wasmtime_instance
            .get_func(&mut guard.store, export)
            .ok_or_else(|| ExecError::MissingExport(export.to_string()))?;
        let typed = func
            .typed::<(), ()>(&guard.store)
            .map_err(ExecError::Wasmtime)?;
        typed.call_async(&mut guard.store, ()).await?;
        Ok(())
    }

    /// Drop the instance, releasing its resources.
    #[instrument(skip(self), fields(instance = %id))]
    pub async fn terminate(&self, id: InstanceId) -> Result<(), ExecError> {
        match self.instances.remove(&id) {
            Some(_) => {
                if let Some(m) = &self.metrics {
                    m.instance_terminations_total().inc();
                    m.active_instances().dec();
                }
                debug!(target: "bali_exec::executor", instance = %id, "instance terminated");
                Ok(())
            }
            None => Err(ExecError::NotFound(id)),
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
        let engine = Arc::new(BaliEngine::new().unwrap());
        let exec = BaliExecutor::new(engine);
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
        let engine = Arc::new(BaliEngine::new().unwrap());
        let exec = BaliExecutor::new(engine);
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        let err = exec.call_export(id, "does_not_exist").await.unwrap_err();
        assert!(matches!(err, ExecError::MissingExport(_)));
    }

    #[tokio::test]
    async fn terminate_unknown() {
        let engine = Arc::new(BaliEngine::new().unwrap());
        let exec = BaliExecutor::new(engine);
        let err = exec.terminate(InstanceId(999)).await.unwrap_err();
        assert!(matches!(err, ExecError::NotFound(_)));
    }

    #[tokio::test]
    async fn metrics_increment_on_spawn_and_terminate() {
        use bali_core::metrics::BaliMetrics;
        let engine = Arc::new(BaliEngine::new().unwrap());
        let metrics = BaliMetrics::new();
        let exec = BaliExecutor::with_metrics(engine, metrics.clone());
        let id = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
            .await
            .unwrap();
        let text = metrics.encode_text();
        assert!(
            text.contains("bali_instance_spawns_total 1"),
            "got:\n{text}"
        );
        assert!(text.contains("bali_active_instances 1"), "got:\n{text}");
        exec.terminate(id).await.unwrap();
        let text = metrics.encode_text();
        assert!(
            text.contains("bali_instance_terminations_total 1"),
            "got:\n{text}"
        );
        assert!(text.contains("bali_active_instances 0"), "got:\n{text}");
    }

    #[test]
    fn exec_error_converts_to_bali_error() {
        use bali_core::error::BaliError;
        let e = ExecError::NotFound(InstanceId(99));
        let b: BaliError = e.into();
        assert!(matches!(b, BaliError::Serialization(_)));
        assert!(b.to_string().contains("instance not found"));

        let e = ExecError::Timeout(InstanceId(1));
        let b: BaliError = e.into();
        assert!(matches!(b, BaliError::KernelTimeout { .. }));
    }
}
