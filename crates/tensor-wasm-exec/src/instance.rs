// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Per-instance state used by [`TensorWasmExecutor`](crate::executor::TensorWasmExecutor).
//!
//! Each running Wasm instance has a [`TensorWasmInstance`] which owns:
//! - the wasmtime `Store<InstanceState>` driving execution,
//! - the wasmtime `Instance` itself,
//! - identity (`TenantId`, `InstanceId`),
//! - per-instance deadlines and metadata used by metrics and tracing,
//! - the per-instance [`TensorWasmResourceLimiter`](crate::executor::TensorWasmResourceLimiter)
//!   that caps linear-memory growth.
//!
//! Instances are typically held inside an `Arc<Mutex<TensorWasmInstance>>` so the
//! executor can drive their lifecycle from a Tokio task while the API layer
//! invokes exported functions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tensor_wasm_core::types::{InstanceId, TenantId};

use crate::executor::TensorWasmResourceLimiter;

/// Side-channel data attached to each wasmtime `Store`.
///
/// Wasmtime gives every Store a `T` payload accessible from host functions —
/// we store identity, deadlines, and resource counters here so WASI-CUDA
/// host functions (S8) can access them via `caller.data()`.
#[derive(Debug)]
pub struct InstanceState {
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Unique instance identifier.
    pub instance_id: InstanceId,
    /// Walltime the instance was created.
    pub created_at: Instant,
    /// Soft deadline (epoch-driven) for the **current** call.
    ///
    /// `None` means "no host-imposed deadline". This is re-armed at the start
    /// of each [`TensorWasmExecutor::call_export`](crate::executor::TensorWasmExecutor::call_export)
    /// from [`InstanceState::deadline_duration`], so back-to-back calls each
    /// get a fresh wall-clock window instead of inheriting the elapsed time
    /// from a previous call.
    pub deadline: Option<Instant>,
    /// Per-call deadline as a [`Duration`], retained so the executor can
    /// re-arm the absolute [`InstanceState::deadline`] (and the wasmtime
    /// epoch deadline) at the start of each call. `None` for instances
    /// spawned without a deadline.
    pub deadline_duration: Option<Duration>,
    /// Total kernel dispatches issued by this instance (cumulative).
    pub kernel_dispatches: AtomicU64,
    /// Total bytes of GPU memory this instance has allocated.
    pub gpu_bytes_allocated: AtomicU64,
    /// Per-instance linear-memory limiter. Mirrors the engine's
    /// `max_memory_bytes`; wasmtime invokes it from `memory.grow`.
    pub limiter: TensorWasmResourceLimiter,
}

impl InstanceState {
    /// Construct a fresh state with `created_at = Instant::now()`.
    ///
    /// The limiter is initialised with `usize::MAX` (no enforcement); callers
    /// that want enforcement should overwrite the `limiter` field or use
    /// [`InstanceState::with_memory_limit`].
    pub fn new(tenant_id: TenantId, instance_id: InstanceId) -> Self {
        Self {
            tenant_id,
            instance_id,
            created_at: Instant::now(),
            deadline: None,
            deadline_duration: None,
            kernel_dispatches: AtomicU64::new(0),
            gpu_bytes_allocated: AtomicU64::new(0),
            limiter: TensorWasmResourceLimiter::new(usize::MAX),
        }
    }

    /// Set the per-instance linear-memory cap (in bytes). Returns `self`
    /// for builder-style use.
    pub fn with_memory_limit(mut self, max_memory_bytes: usize) -> Self {
        self.limiter = TensorWasmResourceLimiter::new(max_memory_bytes);
        self
    }

    /// Set a deadline; returns `self` for builder-style use.
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Record the per-call deadline duration so subsequent calls can re-arm
    /// the wall-clock deadline (and matching wasmtime epoch ticks) instead
    /// of inheriting the elapsed window from spawn time.
    pub fn with_deadline_duration(mut self, d: Duration) -> Self {
        self.deadline_duration = Some(d);
        self
    }

    /// Increment the kernel dispatch counter and return the new value.
    pub fn record_kernel_dispatch(&self) -> u64 {
        self.kernel_dispatches.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Add `bytes` to the GPU allocation total.
    pub fn record_gpu_alloc(&self, bytes: u64) {
        self.gpu_bytes_allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    /// True if the deadline has elapsed.
    pub fn is_past_deadline(&self) -> bool {
        match self.deadline {
            Some(d) => Instant::now() >= d,
            None => false,
        }
    }
}

/// A running Wasm instance.
pub struct TensorWasmInstance {
    /// Wasmtime store driving execution.
    pub(crate) store: wasmtime::Store<InstanceState>,
    /// Wasmtime instance after linking exports.
    pub(crate) instance: wasmtime::Instance,
}

impl TensorWasmInstance {
    /// Construct a `TensorWasmInstance` from a fully-instantiated wasmtime
    /// `(store, instance)` pair. Typically called by
    /// [`TensorWasmExecutor::spawn_instance`](crate::executor::TensorWasmExecutor::spawn_instance).
    pub fn new(store: wasmtime::Store<InstanceState>, instance: wasmtime::Instance) -> Self {
        Self { store, instance }
    }

    /// Tenant owning this instance.
    pub fn tenant_id(&self) -> TenantId {
        self.store.data().tenant_id
    }

    /// Unique instance identifier.
    pub fn instance_id(&self) -> InstanceId {
        self.store.data().instance_id
    }

    /// Borrow the wasmtime [`wasmtime::Store`].
    pub fn store(&mut self) -> &mut wasmtime::Store<InstanceState> {
        &mut self.store
    }

    /// Borrow the wasmtime [`wasmtime::Instance`].
    pub fn wasmtime_instance(&self) -> &wasmtime::Instance {
        &self.instance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn state_records_identity() {
        let s = InstanceState::new(TenantId(1), InstanceId(2));
        assert_eq!(s.tenant_id, TenantId(1));
        assert_eq!(s.instance_id, InstanceId(2));
        assert_eq!(s.kernel_dispatches.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn record_kernel_dispatch_increments() {
        let s = InstanceState::new(TenantId(0), InstanceId(0));
        assert_eq!(s.record_kernel_dispatch(), 1);
        assert_eq!(s.record_kernel_dispatch(), 2);
        assert_eq!(s.record_kernel_dispatch(), 3);
    }

    #[test]
    fn record_gpu_alloc_sums() {
        let s = InstanceState::new(TenantId(0), InstanceId(0));
        s.record_gpu_alloc(1024);
        s.record_gpu_alloc(2048);
        assert_eq!(s.gpu_bytes_allocated.load(Ordering::Relaxed), 3072);
    }

    #[test]
    fn deadline_check() {
        let past = Instant::now() - Duration::from_secs(1);
        let s = InstanceState::new(TenantId(0), InstanceId(0)).with_deadline(past);
        assert!(s.is_past_deadline());

        let future = Instant::now() + Duration::from_secs(60);
        let s = InstanceState::new(TenantId(0), InstanceId(0)).with_deadline(future);
        assert!(!s.is_past_deadline());

        let s = InstanceState::new(TenantId(0), InstanceId(0));
        assert!(!s.is_past_deadline());
    }
}
