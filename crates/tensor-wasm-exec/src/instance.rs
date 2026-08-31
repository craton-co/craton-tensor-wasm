// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Per-instance state used by [`TensorWasmExecutor`](crate::executor::TensorWasmExecutor).
//!
//! Each running Wasm instance has a [`TensorWasmInstance`] which owns:
//! - the wasmtime `Store<InstanceState>` driving execution,
//! - the wasmtime `Instance` itself,
//! - identity (`TenantId`, `InstanceId`),
//! - per-instance deadlines and metadata used by metrics and tracing,
//! - the per-instance [`TensorWasmResourceLimiter`]
//!   that caps linear-memory growth.
//!
//! Instances are typically held inside an `Arc<Mutex<TensorWasmInstance>>` so the
//! executor can drive their lifecycle from a Tokio task while the API layer
//! invokes exported functions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_wasi_gpu::scheduler::SchedulerContext;
use tensor_wasm_wasi_gpu::streaming::{HasInput, HasStreaming, InputContext, StreamingContext};

use crate::executor::TensorWasmResourceLimiter;
use crate::jit_dispatch::{ArenaState, JitArenaProvider};

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
    /// Absolute wall-clock instant past which the cooperative epoch
    /// callback must **trap** (terminate the guest) rather than yield.
    ///
    /// This is the source of truth for the HARD deadline guarantee under
    /// the cooperative-yield scheme (see
    /// [`crate::executor::TensorWasmExecutor`]'s epoch callback). The
    /// store's epoch deadline is configured with
    /// [`wasmtime::Store::epoch_deadline_callback`] in async-yield mode:
    /// every time the epoch deadline trips, the callback compares
    /// `Instant::now()` against this instant. If the instant has NOT
    /// passed, the callback returns [`wasmtime::UpdateDeadline::Yield`],
    /// so the guest yields `Pending` to the async runtime (making a
    /// compute-bound guest cancellable on future-drop) and the epoch
    /// re-arms for another cooperative window. Once this instant HAS
    /// passed, the callback returns an error, which wasmtime turns into
    /// a trap — preserving the exact trap-on-deadline behaviour the
    /// executor relied on under the old `set_epoch_deadline`-trap scheme.
    ///
    /// Distinct from [`Self::deadline`]: `deadline` is the per-call soft
    /// budget consulted by the scheduler/back-pressure surface and by the
    /// timeout *classification* in `call_export_with_args`, whereas
    /// `hard_deadline` is what the epoch callback enforces. The executor
    /// keeps them aligned — for a call with a configured deadline both are
    /// `now + d`; during instantiation `hard_deadline` is additionally
    /// clamped by [`crate::executor::MAX_START_FN_DURATION`] so a runaway
    /// `start` function still traps even on a deadline-less spawn.
    /// `None` means "no hard cap" — the callback yields cooperatively
    /// forever (used only if no cap of any kind applies).
    pub hard_deadline: Option<Instant>,
    /// Cooperative cancellation flag consulted by the epoch-deadline
    /// callback alongside [`Self::hard_deadline`].
    ///
    /// [`TensorWasmExecutor::terminate`](crate::executor::TensorWasmExecutor::terminate)
    /// flips this to `true` (and sets `hard_deadline = Some(now)`) so the
    /// next epoch tick traps an in-flight, possibly deadline-less call —
    /// closing the gap where `terminate` could only remove the registry
    /// entry and free the admission slot, but could not interrupt a guest
    /// already executing inside `call_async` (the per-instance mutex is held
    /// across that await, so the registry removal alone never reached the
    /// running fiber).
    ///
    /// Held as an `Arc<AtomicBool>` so a clone can live in the registry
    /// value beside the `Arc<Mutex<…>>`: `terminate` flips the flag WITHOUT
    /// taking the per-instance mutex (which the in-flight call holds), and
    /// the epoch callback reads it lock-free from the per-store payload. A
    /// freshly-spawned instance starts un-cancelled; the flag is one-shot
    /// (terminate is terminal — an instance is never resurrected) so it is
    /// never reset.
    pub(crate) cancelled: Arc<std::sync::atomic::AtomicBool>,
    /// Total kernel dispatches issued by this instance (cumulative).
    pub kernel_dispatches: AtomicU64,
    /// Total bytes of GPU memory this instance has allocated.
    pub gpu_bytes_allocated: AtomicU64,
    /// Per-instance linear-memory limiter. Mirrors the engine's
    /// `max_memory_bytes`; wasmtime invokes it from `memory.grow`.
    ///
    /// Crate-private: external code must not mutate the per-instance
    /// limiter mid-flight (doing so would let a host import widen its
    /// own memory cap inside a host call, defeating the engine cap).
    /// Construction routes through [`Self::with_memory_limit`] only.
    pub(crate) limiter: TensorWasmResourceLimiter,
    /// Per-instance bump arena backing the JIT `alloc`/`free`/`dispatch`
    /// imports. Living in the per-store payload (rather than captured in
    /// the linker closures) is what keeps two tenants instantiated from
    /// the same [`wasmtime::Linker`] from polluting each other's bump
    /// cursor and LIFO stack — see [`JitArenaProvider`].
    pub(crate) jit_arena: ArenaState,
    /// Per-instance cooperative-scheduler context backing the
    /// `wasi:scheduler/host@0.1.0` host functions (roadmap feature #4).
    ///
    /// Re-armed at the start of each [`crate::executor::TensorWasmExecutor::call_export`]
    /// alongside [`Self::deadline`] so back-to-back calls each get a
    /// fresh `started_at` instant — without this the second call would
    /// observe a deadline-elapsed reading immediately because the
    /// elapsed wall-clock from the first call would still be charged.
    ///
    /// On spawns without a configured deadline, this is constructed
    /// via [`SchedulerContext::unbounded`] and every `yield()` returns
    /// `YIELD_CODE_CONTINUE`.
    pub(crate) scheduler: SchedulerContext,
    /// Per-instance streaming context backing the
    /// `wasi:tensor/host@0.1.0` `emit-chunk` / `flush` host functions
    /// (roadmap feature #2). Constructed via
    /// [`StreamingContext::disabled`] for spawns the gateway did not
    /// opt into the streaming response path — every `emit-chunk` then
    /// returns the documented `-1` code immediately.
    ///
    /// Set by [`crate::executor::SpawnConfig::with_streaming`] at
    /// spawn time. v0.4 wires this through to the
    /// `POST /functions/{id}/invoke-stream` route — see
    /// `docs/STREAMING.md`.
    pub(crate) streaming: StreamingContext,
    /// Per-instance guest-input context backing the
    /// `wasi:tensor/host@0.1.0` `input-len` / `read-input` host functions
    /// (the pull-model input channel). Constructed via
    /// [`InputContext::empty`] by default — `input-len` then returns `0`
    /// and `read-input` copies nothing.
    ///
    /// Set by [`crate::executor::SpawnConfig::with_input`] at spawn time.
    /// The OpenAI completions shim stages the assembled prompt bytes here
    /// so the guest can pull them via `wasi:tensor/host.read-input`.
    pub(crate) input: InputContext,
}

impl InstanceState {
    /// Construct a fresh state with `created_at = Instant::now()`.
    ///
    /// The limiter is initialised with `usize::MAX` (no enforcement);
    /// callers that want enforcement should chain
    /// [`InstanceState::with_memory_limit`] (the `limiter` field is
    /// `pub(crate)` so external code cannot widen it post-construction).
    pub fn new(tenant_id: TenantId, instance_id: InstanceId) -> Self {
        Self {
            tenant_id,
            instance_id,
            created_at: Instant::now(),
            deadline: None,
            deadline_duration: None,
            hard_deadline: None,
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            kernel_dispatches: AtomicU64::new(0),
            gpu_bytes_allocated: AtomicU64::new(0),
            limiter: TensorWasmResourceLimiter::new(usize::MAX),
            jit_arena: ArenaState::default(),
            // No deadline configured by default; spawns with a
            // SpawnConfig::deadline overwrite this via
            // `with_deadline_duration` so the SchedulerContext budget
            // matches the wasmtime epoch deadline. Constructing the
            // unbounded shape here means a guest that imports
            // `wasi:scheduler/host` always gets a working surface
            // — `yield()` is a no-op CONTINUE and
            // `deadline-remaining-ms` returns u32::MAX.
            scheduler: SchedulerContext::unbounded(),
            // Streaming disabled by default; spawns through
            // `/invoke-stream` install a real channel-backed context
            // via `SpawnConfig::with_streaming`. Guests that import
            // `wasi:tensor/host` and run under the non-streaming
            // `/invoke` route see every `emit-chunk` return `-1`.
            streaming: StreamingContext::disabled(),
            // No input staged by default; spawns that stage a prompt
            // install a populated context via `SpawnConfig::with_input`.
            // Guests calling `wasi:tensor/host.input-len` then see `0`.
            input: InputContext::empty(),
        }
    }

    /// Install a [`StreamingContext`] on this state; returns `self`
    /// for builder-style chaining.
    ///
    /// Called from [`crate::executor::TensorWasmExecutor::spawn_instance`]
    /// when [`crate::executor::SpawnConfig::streaming`] is set so the
    /// `wasi:tensor/host.emit-chunk` host function reaches a live
    /// downstream receiver. Spawns without a streaming context retain
    /// the default [`StreamingContext::disabled`] shape.
    pub fn with_streaming(mut self, streaming: StreamingContext) -> Self {
        self.streaming = streaming;
        self
    }

    /// Borrow the per-instance streaming context. Used by the
    /// `wasi:tensor/host` linker registration to plumb the
    /// per-instance state into the `emit-chunk` and `flush` host
    /// functions.
    pub fn streaming(&self) -> &StreamingContext {
        &self.streaming
    }

    /// Install an [`InputContext`] on this state; returns `self` for
    /// builder-style chaining.
    ///
    /// Called from [`crate::executor::TensorWasmExecutor`]'s spawn path
    /// when [`crate::executor::SpawnConfig::input`] is non-empty so the
    /// `wasi:tensor/host.read-input` host function reaches the staged
    /// bytes. Spawns without input retain the default
    /// [`InputContext::empty`] shape.
    pub fn with_input(mut self, input: InputContext) -> Self {
        self.input = input;
        self
    }

    /// Borrow the per-instance input context. Used by the
    /// `wasi:tensor/host` linker registration to plumb the per-instance
    /// staged bytes into the `input-len` and `read-input` host functions.
    pub fn input(&self) -> &InputContext {
        &self.input
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

    /// Set the absolute instant at which the cooperative epoch callback
    /// must trap (the HARD deadline). Returns `self` for builder-style use.
    /// See [`Self::hard_deadline`].
    pub(crate) fn with_hard_deadline(mut self, at: Instant) -> Self {
        self.hard_deadline = Some(at);
        self
    }

    /// True iff a hard deadline is configured and `Instant::now()` has
    /// reached or passed it. Consulted by the executor's epoch-deadline
    /// callback to decide between a cooperative yield and a hard trap.
    pub(crate) fn hard_deadline_elapsed(&self) -> bool {
        match self.hard_deadline {
            Some(at) => Instant::now() >= at,
            None => false,
        }
    }

    /// True iff [`TensorWasmExecutor::terminate`](crate::executor::TensorWasmExecutor::terminate)
    /// has flipped the cooperative-cancellation flag. Consulted by the epoch
    /// callback alongside [`Self::hard_deadline_elapsed`] so a `terminate`
    /// traps an in-flight (and possibly deadline-less) call at the next
    /// epoch tick instead of merely de-registering it.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Clone the shared cancellation handle so the executor can stash it in
    /// the registry value (beside the `Arc<Mutex<…>>`) and flip it from
    /// `terminate` WITHOUT acquiring the per-instance mutex the in-flight
    /// call holds across `call_async`.
    pub(crate) fn cancel_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    /// Record the per-call deadline duration so subsequent calls can re-arm
    /// the wall-clock deadline (and matching wasmtime epoch ticks) instead
    /// of inheriting the elapsed window from spawn time.
    ///
    /// Also seeds the cooperative-scheduler context with the same
    /// budget (clamped to `u32::MAX` ms ≈ 49 days, which is well above
    /// any plausible production deadline) so guests calling
    /// `wasi:scheduler/host.yield()` observe a non-zero return code
    /// when the deadline is approaching.
    ///
    /// T36: also seeds the scheduler's absolute `Instant` deadline
    /// (`bp_deadline_instant`) to `[Self::deadline]`. With both fields
    /// installed, the guest's `yield()` verdict is derived from the
    /// same `Instant` that the back-pressure semaphore consults — so
    /// the cooperative path and the resource-admission path agree
    /// when the deadline trips.
    pub fn with_deadline_duration(mut self, d: Duration) -> Self {
        self.deadline_duration = Some(d);
        // Clamp to u32::MAX; a deadline larger than that is in
        // practice "unbounded" — the epoch interrupt would never fire
        // within u32::MAX ms either, so the cooperative path can
        // honestly report u32::MAX too.
        let deadline_ms = u32::try_from(d.as_millis()).unwrap_or(u32::MAX);
        let mut sched = SchedulerContext::new(Some(deadline_ms));
        // The absolute deadline is whatever `with_deadline` previously
        // installed (if anything); we copy it onto the scheduler so
        // `yield_now()` resolves from the same `Instant` that
        // `BackPressure::with_deadline_hint` consumes. If
        // `with_deadline` has not yet been called the field is `None`
        // and the scheduler falls back to its legacy ms-based path
        // until the executor re-arms us at the start of a call.
        sched.set_bp_deadline_instant(self.deadline);
        self.scheduler = sched;
        self
    }

    /// Borrow the cooperative-scheduler context. Used by the
    /// `wasi:scheduler/host` linker registration to plumb the
    /// per-instance state into the `yield` and `deadline-remaining-ms`
    /// host functions.
    pub fn scheduler(&self) -> &SchedulerContext {
        &self.scheduler
    }

    /// Re-arm the scheduler context's wall-clock origin. Called from
    /// [`crate::executor::TensorWasmExecutor::call_export`] at the
    /// start of each call so back-to-back invocations each see a
    /// fresh elapsed window (mirroring the per-call re-arm of
    /// [`Self::deadline`] and the wasmtime epoch deadline).
    ///
    /// T36: also re-installs the scheduler's absolute `Instant`
    /// deadline so the guest's cooperative-yield verdicts agree with
    /// the back-pressure semaphore's acquire decisions for THIS
    /// call's window — not the previous one. Pulls the current
    /// `Self::deadline` value so a fresh `call_export` (which
    /// re-seeds `deadline = now + d` before calling this) gets the
    /// up-to-date `Instant` propagated through.
    pub(crate) fn rearm_scheduler(&mut self) {
        // Only propagate the absolute `Instant` deadline to the scheduler when
        // the instance was explicitly configured with a per-call deadline
        // (`deadline_duration.is_some()`). Without this guard a deadline-less
        // spawn that has the engine-wide `max_call_duration` ceiling written
        // into `self.deadline` would arm `bp_deadline_instant` on the
        // `SchedulerContext`, causing `deadline_remaining_ms()` to return the
        // engine-ceiling remaining milliseconds instead of the `u32::MAX`
        // unbounded sentinel. The bp-aligned Instant path is an opt-in for
        // guests that explicitly operate under a deadline; deadline-less spawns
        // stay on the legacy `deadline_ms = None` → `u32::MAX` path.
        let instant = if self.deadline_duration.is_some() {
            self.deadline
        } else {
            None
        };
        self.scheduler.rearm_with_instant(instant);
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

    /// Borrow the per-instance JIT scratch arena mutably.
    ///
    /// Used by the host imports registered via
    /// [`crate::jit_dispatch::add_jit_dispatch_to_linker`] to keep one
    /// bump cursor and LIFO `live` stack per store, even when multiple
    /// instances share a single [`wasmtime::Linker`].
    pub fn jit_arena_mut(&mut self) -> &mut ArenaState {
        &mut self.jit_arena
    }

    /// Borrow the per-instance JIT scratch arena.
    ///
    /// Read-only counterpart to [`InstanceState::jit_arena_mut`]; useful
    /// for tests asserting per-store isolation of the bump cursor.
    pub fn jit_arena(&self) -> &ArenaState {
        &self.jit_arena
    }
}

impl JitArenaProvider for InstanceState {
    fn jit_arena_mut(&mut self) -> &mut ArenaState {
        InstanceState::jit_arena_mut(self)
    }
}

impl HasStreaming for InstanceState {
    fn streaming(&self) -> &StreamingContext {
        InstanceState::streaming(self)
    }
}

impl HasInput for InstanceState {
    fn input(&self) -> &InputContext {
        InstanceState::input(self)
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

    #[test]
    fn input_empty_by_default() {
        // A freshly-constructed InstanceState stages no guest input, so
        // `wasi:tensor/host.input-len` returns 0.
        let s = InstanceState::new(TenantId(0), InstanceId(0));
        assert!(s.input().is_empty());
        assert_eq!(s.input().len_u32(), 0);
    }

    #[test]
    fn with_input_stages_bytes() {
        let s = InstanceState::new(TenantId(0), InstanceId(0))
            .with_input(InputContext::new(b"prompt".to_vec()));
        assert!(!s.input().is_empty());
        assert_eq!(s.input().bytes(), b"prompt");
    }

    #[test]
    fn scheduler_unbounded_by_default() {
        // A freshly-constructed InstanceState carries an unbounded
        // SchedulerContext so guests that import `wasi:scheduler/host`
        // see a working surface even without a configured deadline.
        let s = InstanceState::new(TenantId(0), InstanceId(0));
        assert_eq!(s.scheduler().deadline_ms(), None);
    }

    #[test]
    fn with_deadline_duration_seeds_scheduler() {
        // The scheduler context's budget mirrors the configured
        // wall-clock deadline so cooperative yields and the epoch
        // interrupt agree on when the deadline trips.
        let s = InstanceState::new(TenantId(0), InstanceId(0))
            .with_deadline_duration(Duration::from_millis(750));
        assert_eq!(s.scheduler().deadline_ms(), Some(750));
    }

    #[test]
    fn with_deadline_duration_saturates_at_u32_max() {
        // A pathologically long deadline (decades) saturates the u32
        // ms field rather than wrapping. The epoch interrupt would
        // never fire within u32::MAX ms either, so reporting u32::MAX
        // is honest.
        let s = InstanceState::new(TenantId(0), InstanceId(0))
            .with_deadline_duration(Duration::from_secs(60 * 60 * 24 * 365 * 100));
        assert_eq!(s.scheduler().deadline_ms(), Some(u32::MAX));
    }

    #[test]
    fn with_deadline_duration_propagates_instant_when_present() {
        // T36: when both `with_deadline` (sets the absolute Instant)
        // and `with_deadline_duration` are chained, the scheduler
        // context picks up the same Instant for its BP-aligned query
        // path so guests and BackPressure agree on the trip point.
        let at = Instant::now() + Duration::from_millis(500);
        let s = InstanceState::new(TenantId(0), InstanceId(0))
            .with_deadline(at)
            .with_deadline_duration(Duration::from_millis(500));
        assert_eq!(s.scheduler().bp_deadline_instant(), Some(at));
    }

    #[test]
    fn rearm_scheduler_updates_bp_deadline_instant() {
        // T36: re-arming the scheduler at the start of a call
        // installs the FRESH absolute Instant (so back-to-back calls
        // observe distinct windows). Simulate the executor's per-call
        // re-arm by overwriting `deadline` and confirming the
        // scheduler tracks it.
        let mut s = InstanceState::new(TenantId(0), InstanceId(0))
            .with_deadline(Instant::now() + Duration::from_millis(100))
            .with_deadline_duration(Duration::from_millis(100));
        let next = Instant::now() + Duration::from_secs(5);
        s.deadline = Some(next);
        s.rearm_scheduler();
        assert_eq!(s.scheduler().bp_deadline_instant(), Some(next));
    }
}
