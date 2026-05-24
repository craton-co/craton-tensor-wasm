// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! [`TensorWasmExecutor`] — async executor for TensorWasm Wasm instances.
//!
//! Owns a shared [`TensorWasmEngine`] and a registry of live [`TensorWasmInstance`]s
//! keyed by [`InstanceId`]. Exposes the trio of operations
//! [`TensorWasmExecutor::spawn_instance`], [`TensorWasmExecutor::call_export`], and
//! [`TensorWasmExecutor::terminate`] — all async, all driven from the calling
//! Tokio runtime.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::{InstanceId, TenantId};
use dashmap::{mapref::entry::Entry, DashMap};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};
use wasmtime::{Module, ResourceLimiter, Store};

use crate::engine::TensorWasmEngine;
use crate::instance::{TensorWasmInstance, InstanceState};

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
                // variant (and `is_retryable` / `kind` reflect it). The full
                // error chain is preserved via `format!("{err:#}")` which
                // walks the `#[source]` chain wasmtime/anyhow expose.
                if err.downcast_ref::<wasmtime::Trap>().is_some() {
                    TensorWasmError::WasmTrap(format!("{err:#}").into())
                } else {
                    TensorWasmError::WasmCompile(format!("{err:#}").into())
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
        _desired: u32,
        _maximum: Option<u32>,
    ) -> wasmtime::Result<bool> {
        // Tables are unbounded in this limiter — only linear memory is
        // policed at the byte level for v0.1.0.
        Ok(true)
    }
}

/// The async executor.
#[derive(Clone)]
pub struct TensorWasmExecutor {
    engine: Arc<TensorWasmEngine>,
    instances: Arc<DashMap<InstanceId, Arc<Mutex<TensorWasmInstance>>>>,
    next_instance_id: Arc<AtomicU64>,
    /// Per-engine compiled-module cache keyed by a 64-bit digest of the wasm
    /// bytes. Avoids re-running Cranelift on every `spawn_instance` for
    /// repeat tenants. Hash is computed with `blake3` (SIMD-accelerated;
    /// ~5x faster than SipHash on multi-MiB wasm modules) and the first 8
    /// bytes of the digest are interpreted as a little-endian `u64` for the
    /// cache key — stable across runs and platforms.
    module_cache: Arc<DashMap<u64, Module>>,
    /// Optional metrics handle. When `Some`, spawn/terminate operations
    /// increment the corresponding Prometheus counters / gauges.
    metrics: Option<TensorWasmMetrics>,
}

impl TensorWasmExecutor {
    /// Construct an executor over the given shared engine.
    pub fn new(engine: Arc<TensorWasmEngine>) -> Self {
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(DashMap::new()),
            metrics: None,
        }
    }

    /// Construct an executor that publishes spawn/terminate events to the
    /// supplied [`TensorWasmMetrics`] registry. Metric handles are cheaply cloneable;
    /// pass a clone of the process-wide registry.
    pub fn with_metrics(engine: Arc<TensorWasmEngine>, metrics: TensorWasmMetrics) -> Self {
        Self {
            engine,
            instances: Arc::new(DashMap::new()),
            next_instance_id: Arc::new(AtomicU64::new(1)),
            module_cache: Arc::new(DashMap::new()),
            metrics: Some(metrics),
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
        self.module_cache.len()
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
    /// first 8 bytes of a BLAKE3 digest of the wasm bytes interpreted as a
    /// little-endian `u64` — stable across runs and platforms, and ~5x
    /// faster than SipHash on the multi-MiB modules we actually compile.
    fn compile_module_cached(&self, wasm: &[u8]) -> Result<Module, ExecError> {
        let digest = blake3::hash(wasm);
        let bytes = digest.as_bytes();
        let key = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if let Some(m) = self.module_cache.get(&key) {
            return Ok(m.clone());
        }
        let module = Module::from_binary(self.engine.inner(), wasm)
            .map_err(ExecError::Wasmtime)?;
        // `DashMap::entry` is racy-safe: if another task compiled the same
        // bytes between our `get` and now, we keep the existing entry.
        match self.module_cache.entry(key) {
            Entry::Occupied(occupied) => Ok(occupied.get().clone()),
            Entry::Vacant(vacant) => {
                vacant.insert(module.clone());
                Ok(module)
            }
        }
    }

    /// Compile + instantiate a Wasm module. Returns the assigned [`InstanceId`].
    #[instrument(skip(self, wasm), fields(tenant = %cfg.tenant_id, instance_id = tracing::field::Empty))]
    pub async fn spawn_instance(
        &self,
        cfg: SpawnConfig,
        wasm: &[u8],
    ) -> Result<InstanceId, ExecError> {
        let id = self.allocate_instance_id();
        tracing::Span::current().record("instance_id", tracing::field::display(id));
        let max_memory_bytes = self.engine.config().max_memory_bytes;
        let mut state =
            InstanceState::new(cfg.tenant_id, id).with_memory_limit(max_memory_bytes);
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
                // Use `u64::try_from` to clamp at `u64::MAX` if a caller
                // supplies a pathologically long deadline.
                let d_ms = d.as_millis();
                let t_ms = tick.as_millis().max(1);
                let ticks_u128 = d_ms.div_ceil(t_ms).max(1);
                u64::try_from(ticks_u128).unwrap_or(u64::MAX)
            }
            // No deadline → effectively unbounded. u64::MAX is fine: the
            // engine's epoch counter would need ~5.8 billion years at 10 ms/
            // tick to reach it, so the deadline will never trip in practice.
            None => u64::MAX,
        };
        let module = self.compile_module_cached(wasm)?;
        let mut store = Store::new(self.engine.inner(), state);
        // Cap linear-memory growth at the engine-configured maximum. The
        // limiter lives inside the store payload (`InstanceState::limiter`)
        // so wasmtime can borrow it without any extra heap allocation per
        // `memory.grow` call. The explicit return type pins the trait
        // object coercion so type inference doesn't choose the concrete
        // `&mut TensorWasmResourceLimiter`.
        store.limiter(|state| &mut state.limiter as &mut dyn ResourceLimiter);
        store.set_epoch_deadline(epoch_deadline_ticks);
        let instance = wasmtime::Instance::new_async(&mut store, &module, &[]).await?;
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
                return Err(ExecError::Wasmtime(wasmtime::Error::msg(
                    "instance id collision after allocation",
                )));
            }
        }
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
        let handle = self
            .instances
            .get(&id)
            .ok_or(ExecError::NotFound(id))?
            .value()
            .clone();
        let mut guard = handle.lock().await;
        // Snapshot the deadline (if any) before taking the call path so
        // we can synthesise a `Timeout` variant with real elapsed/deadline
        // figures when the epoch interrupt fires.
        let deadline_at = guard.store.data().deadline;
        let started_at = Instant::now();
        let wasmtime_instance = *guard.wasmtime_instance();
        let func = wasmtime_instance
            .get_func(&mut guard.store, export)
            .ok_or_else(|| ExecError::MissingExport(export.to_string()))?;
        let typed = func
            .typed::<(), ()>(&guard.store)
            .map_err(ExecError::Wasmtime)?;
        match typed.call_async(&mut guard.store, ()).await {
            Ok(()) => Ok(()),
            Err(err) => {
                // If we had a deadline AND the wall clock has tripped past
                // it, classify the failure as Timeout with real numbers.
                // Otherwise propagate the raw wasmtime error.
                let elapsed = started_at.elapsed();
                let past_deadline = deadline_at
                    .map(|d| Instant::now() >= d)
                    .unwrap_or(false);
                if past_deadline {
                    let deadline_ms = deadline_at
                        .map(|d| d.saturating_duration_since(started_at).as_millis())
                        .unwrap_or(0);
                    Err(ExecError::Timeout(TimeoutContext {
                        id,
                        elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                        deadline_ms: u64::try_from(deadline_ms).unwrap_or(u64::MAX),
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
}
