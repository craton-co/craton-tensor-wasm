// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pre-instantiated instance pool (roadmap feature #5).
//!
//! Pre-spawns N instances per (tenant, module-hash) tuple under MPS and
//! draws from a `crossbeam-channel` on `invoke`. Closes the
//! `Instance::new_async` cost on the warm path. The pooling allocator +
//! MPK backend already exists (see `EngineConfig::PoolingMpk`); this
//! module just adds the reuse loop on top.
//!
//! ## v0.3.6 status: scaffold only
//!
//! The public API is wired so embedders can opt in via the config; the
//! drawn-instance path currently falls through to `spawn_instance` for
//! every call (i.e. behaviour is equivalent to no pool). v0.4 lands the
//! actual pre-spawn loop, the channel-driven draw, and the reset-on-return
//! semantics. Until then, the pool is a no-op for correctness purposes
//! and a stable API surface for callers.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tensor_wasm_core::types::{InstanceId, TenantId};

use crate::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// Pool configuration. One pool per executor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InstancePoolConfig {
    /// Pre-spawn N instances per (tenant, module-hash) tuple. Default 0
    /// = pool disabled.
    pub warm_instances_per_tuple: usize,

    /// Maximum number of pre-spawned instances across all tuples. Pools
    /// honour this cap before spawning a new tuple. Default 0 = unlimited
    /// (within the executor's `max_instances`).
    pub max_total_warm: usize,
}

impl Default for InstancePoolConfig {
    fn default() -> Self {
        Self {
            warm_instances_per_tuple: 0,
            max_total_warm: 0,
        }
    }
}

/// Pre-instantiated instance pool. Backed by a per-tuple channel of
/// pre-spawned instances. See module docs for v0.3.6 status (scaffold).
///
/// The per-tuple channels are not yet allocated — `state` is a placeholder
/// for the v0.4 wiring. Embedders can construct and pass the pool through
/// the [`TensorWasmExecutor::with_instance_pool`] builder today; behaviour
/// is identical to "no pool" until v0.4 lands.
pub struct InstancePool {
    cfg: InstancePoolConfig,
    // v0.4: this is where the per-tuple channels live. For v0.3.6 we
    // hold the config and no instances.
    #[allow(dead_code)]
    state: Arc<Mutex<HashMap<PoolKey, ()>>>,
}

/// Key for the per-tuple warm-pool map. v0.4 replaces the `()` value
/// type with a `crossbeam_channel::Receiver<TensorWasmInstance>` (one
/// receiver per tuple) so the draw path is wait-free in the common
/// case.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
struct PoolKey {
    tenant_id: TenantId,
    module_hash: [u8; 32],
}

impl InstancePool {
    /// Construct a new pool with the given config.
    pub fn new(cfg: InstancePoolConfig) -> Self {
        Self {
            cfg,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Acquire a pre-spawned instance, or fall through to a fresh spawn.
    ///
    /// v0.3.6: ALWAYS falls through to [`TensorWasmExecutor::spawn_instance`].
    /// v0.4 will draw from the warm-pool channel for the (tenant,
    /// module-hash) tuple, only spawning fresh when the pool is empty.
    ///
    /// # v0.4 semantics (intended)
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use tensor_wasm_core::types::TenantId;
    /// use tensor_wasm_exec::engine::TensorWasmEngine;
    /// use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};
    /// use tensor_wasm_exec::instance_pool::{InstancePool, InstancePoolConfig};
    ///
    /// # async fn example(wasm: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    /// let engine = Arc::new(TensorWasmEngine::new()?);
    /// let exec = TensorWasmExecutor::new(engine);
    /// let pool = Arc::new(InstancePool::new(InstancePoolConfig {
    ///     warm_instances_per_tuple: 4,
    ///     max_total_warm: 256,
    /// }));
    /// let cfg = SpawnConfig::for_tenant(TenantId(1));
    /// // v0.4: draws from the warm channel if available; otherwise spawns.
    /// let pooled = pool.acquire(&exec, wasm, cfg).await?;
    /// // Drop returns the instance to the warm channel after a reset.
    /// drop(pooled);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn acquire(
        &self,
        executor: &TensorWasmExecutor,
        wasm: &[u8],
        cfg: SpawnConfig,
    ) -> Result<PooledInstance, ExecError> {
        let id = executor.spawn_instance(cfg, wasm).await?;
        Ok(PooledInstance {
            inner: Some(id),
            // v0.4: stash the pool reference here so `Drop` returns it
            // to the warm channel.
        })
    }

    /// Number of currently pre-spawned instances across all tuples.
    /// v0.3.6: always 0.
    pub fn warm_count(&self) -> usize {
        0
    }

    /// Borrow the pool's configuration.
    pub fn config(&self) -> &InstancePoolConfig {
        &self.cfg
    }
}

/// RAII wrapper around a drawn-from-pool instance.
///
/// On drop, v0.4 returns the instance to its warm channel (after a
/// reset). v0.3.6: just drops, which leaves the underlying instance
/// registered with the executor — callers must still call
/// [`TensorWasmExecutor::terminate`] (or use
/// [`TensorWasmExecutor::call_export_then_terminate`]) until v0.4
/// lands the reset-on-return path.
///
/// Until v0.4 the `inner` value is the assigned [`InstanceId`]; v0.4
/// will widen this to a richer handle that owns the warm channel
/// sender so `Drop` can return the instance without a registry lookup.
pub struct PooledInstance {
    inner: Option<InstanceId>,
}

impl PooledInstance {
    /// Borrow the underlying instance id. Available only as long as the
    /// [`PooledInstance`] is alive.
    ///
    /// v0.3.6 returns the bare [`InstanceId`] — the v0.4 widening to a
    /// richer borrow type is non-breaking for callers that only need
    /// the id (snapshot, observability, terminate).
    pub fn id(&self) -> InstanceId {
        self.inner
            .expect("PooledInstance was already returned")
    }

    /// Take ownership of the underlying instance id (caller becomes
    /// responsible for explicit termination via
    /// [`TensorWasmExecutor::terminate`]).
    pub fn into_inner(mut self) -> InstanceId {
        self.inner
            .take()
            .expect("PooledInstance was already returned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_disabled() {
        let cfg = InstancePoolConfig::default();
        assert_eq!(cfg.warm_instances_per_tuple, 0);
        assert_eq!(cfg.max_total_warm, 0);
    }

    #[test]
    fn warm_count_starts_at_zero() {
        let pool = InstancePool::new(InstancePoolConfig::default());
        assert_eq!(pool.warm_count(), 0);
    }

    #[test]
    fn config_round_trips() {
        let pool = InstancePool::new(InstancePoolConfig {
            warm_instances_per_tuple: 4,
            max_total_warm: 32,
        });
        assert_eq!(pool.config().warm_instances_per_tuple, 4);
        assert_eq!(pool.config().max_total_warm, 32);
    }
}
