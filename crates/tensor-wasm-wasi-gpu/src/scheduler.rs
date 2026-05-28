// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `wasi:scheduler/host` cooperative-deadline interface (roadmap feature #4).
//!
//! Guests call `yield()` at safe boundaries (tile loop iterations,
//! between kernel dispatches, etc.) to offer a suspend point. The host
//! checks the per-instance deadline and returns non-zero to ask the
//! guest to stop. This is a cooperative protocol — adversarial guests
//! that never call `yield()` are still preempted by the existing
//! epoch-interruption path; this just lowers P99 for well-behaved
//! guests under MPS contention.
//!
//! See `wit/wasi-scheduler.wit` for the Component-Model interface
//! description and `docs/COOPERATIVE-YIELD.md` for the developer-facing
//! protocol guide.
//!
//! # Wiring
//!
//! [`SchedulerContext`] is the per-instance host state. The executor
//! constructs one at spawn time from [`SpawnConfig::deadline`] and
//! stashes it on its store-data payload (typically `InstanceState` in
//! `tensor-wasm-exec`). The host functions registered by
//! [`add_scheduler_to_linker`] reach the context through a closure-
//! provided getter, mirroring the pattern used by
//! `add_jit_dispatch_to_linker`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use wasmtime::{Caller, Linker};

/// WIT package/module name guests import from: `wasi:scheduler/host@0.1.0`.
///
/// Kept in lockstep with the `package wasi:scheduler@0.1.0;` declaration
/// in `wit/wasi-scheduler.wit`. Bumping one without the other will
/// cause guests generated from the WIT to fail to link against this
/// host.
pub const SCHEDULER_MODULE: &str = "wasi:scheduler/host@0.1.0";

/// Function name guests import for the cooperative yield call.
pub const FN_YIELD: &str = "yield";

/// Function name guests import to probe the remaining deadline budget.
pub const FN_DEADLINE_REMAINING_MS: &str = "deadline-remaining-ms";

/// `yield()` return value: the guest may continue executing.
pub const YIELD_CODE_CONTINUE: u32 = 0;

/// `yield()` return value: the deadline is approaching (remaining
/// budget below [`SUGGESTED_YIELD_THRESHOLD_MS`]). The guest SHOULD
/// wind down — finish the current tile, flush partial results — and
/// return promptly.
pub const YIELD_CODE_DEADLINE_APPROACHING: u32 = 1;

/// `yield()` return value: the deadline has elapsed. The guest MUST
/// stop — the host will trip the epoch interrupt at the next tick if
/// it does not.
pub const YIELD_CODE_STOP: u32 = 2;

/// Milliseconds-of-remaining-budget threshold below which `yield()`
/// returns [`YIELD_CODE_DEADLINE_APPROACHING`].
///
/// Chosen to be ~one epoch tick (10 ms in the default
/// [`EngineConfig::epoch_tick`](crate::registry) configuration). A
/// guest that respects the suggestion shaves up to one full tick off
/// the worst-case observed latency: without the cooperative path, the
/// guest only stops when the next epoch tick fires; with it, the guest
/// stops within one yield-loop iteration of the deadline approaching.
pub const SUGGESTED_YIELD_THRESHOLD_MS: u64 = 10;

/// Per-instance cooperative-scheduler context.
///
/// Cheap to clone — the yield count lives behind an `Arc<AtomicU32>`
/// so multiple closures observing the same context (the `yield` and
/// `deadline-remaining-ms` host functions) share counter state. The
/// `started_at` and `deadline_ms` fields are immutable for the life of
/// a call: the executor constructs a fresh context per call-export to
/// match how the epoch deadline is re-armed in
/// [`tensor_wasm_exec::executor::TensorWasmExecutor::call_export`].
#[derive(Debug, Clone)]
pub struct SchedulerContext {
    /// Wall-clock instant the invocation was started.
    started_at: Instant,
    /// Total deadline budget in milliseconds (None = unbounded).
    deadline_ms: Option<u32>,
    /// Cumulative yield-call count for telemetry. Behind `Arc` so the
    /// counter survives the per-host-function closure captures
    /// — each linker registration clones the context, and we want all
    /// clones to share the same counter.
    yield_count: Arc<AtomicU32>,
}

impl SchedulerContext {
    /// Construct a fresh context with the given deadline budget.
    ///
    /// `deadline_ms = None` produces an unbounded context: every
    /// `yield()` returns [`YIELD_CODE_CONTINUE`] and
    /// `deadline_remaining_ms()` returns `u32::MAX`. This is the
    /// shape spawns without a [`SpawnConfig::deadline`] get.
    pub fn new(deadline_ms: Option<u32>) -> Self {
        Self {
            started_at: Instant::now(),
            deadline_ms,
            yield_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Construct a context with no deadline. Equivalent to
    /// `SchedulerContext::new(None)`; spelled out for readability at
    /// call sites that explicitly want the unbounded shape.
    pub fn unbounded() -> Self {
        Self::new(None)
    }

    /// Re-arm the start instant. The executor calls this at the top
    /// of each `call_export` so back-to-back calls each get a fresh
    /// wall-clock window — mirroring how
    /// [`tensor_wasm_exec::instance::InstanceState::deadline`] is
    /// re-seeded in
    /// [`tensor_wasm_exec::executor::TensorWasmExecutor::call_export`].
    pub fn rearm(&mut self) {
        self.started_at = Instant::now();
        // Yield count is cumulative across calls on the same context;
        // re-arming only resets the time origin, not the counter.
        // Tests that need a fresh count construct a fresh context.
    }

    /// Update the deadline budget. Used when the embedder swaps the
    /// configured deadline between calls (rare; the typical pattern
    /// is one deadline per instance for its lifetime).
    pub fn set_deadline_ms(&mut self, deadline_ms: Option<u32>) {
        self.deadline_ms = deadline_ms;
    }

    /// Returns one of the `YIELD_CODE_*` constants.
    ///
    /// Pure-read except for the counter bump — safe to call from any
    /// host-function closure without further synchronisation. The
    /// counter uses `Ordering::Relaxed` because it is a telemetry
    /// surface only, not a happens-before signal.
    pub fn yield_now(&self) -> u32 {
        self.yield_count.fetch_add(1, Ordering::Relaxed);
        match self.deadline_ms {
            None => YIELD_CODE_CONTINUE,
            Some(total) => {
                let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
                let total_u64 = total as u64;
                let remaining = total_u64.saturating_sub(elapsed_ms);
                if remaining == 0 {
                    YIELD_CODE_STOP
                } else if remaining < SUGGESTED_YIELD_THRESHOLD_MS {
                    YIELD_CODE_DEADLINE_APPROACHING
                } else {
                    YIELD_CODE_CONTINUE
                }
            }
        }
    }

    /// Milliseconds of deadline budget remaining, or `u32::MAX` if
    /// the context is unbounded.
    ///
    /// Clamped at `u32::MAX` so a pathological caller cannot
    /// distinguish "real billion-ms remaining" from "unbounded" — the
    /// guest already gets the sentinel value for the unbounded case,
    /// and saturating to the same sentinel keeps the wire shape
    /// stable.
    pub fn deadline_remaining_ms(&self) -> u32 {
        match self.deadline_ms {
            None => u32::MAX,
            Some(total) => {
                let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
                let remaining = (total as u64).saturating_sub(elapsed_ms);
                // Cap at u32::MAX - 1 so the unbounded sentinel
                // (u32::MAX) remains distinguishable from a finite
                // (but very large) remaining budget. In practice
                // `total` is itself a u32 so this branch can only fire
                // on a future API that widens the field.
                u32::try_from(remaining).unwrap_or(u32::MAX.saturating_sub(1))
            }
        }
    }

    /// Cumulative count of `yield()` calls observed by this context.
    /// Useful for telemetry — operators can confirm whether a guest
    /// is actually exercising the cooperative path.
    pub fn yield_count(&self) -> u32 {
        self.yield_count.load(Ordering::Relaxed)
    }

    /// Configured deadline budget, if any. Returned as the original
    /// `Option<u32>` so callers can distinguish "unbounded" from
    /// "deadline exactly 0 ms" (the latter being a degenerate but
    /// expressible config).
    pub fn deadline_ms(&self) -> Option<u32> {
        self.deadline_ms
    }
}

impl Default for SchedulerContext {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Register the two `wasi:scheduler/host@0.1.0` host functions on a
/// wasmtime [`Linker`].
///
/// `T` is the store-data type and `get_ctx` is a getter that borrows
/// the per-instance [`SchedulerContext`] out of it. This shape mirrors
/// `add_jit_dispatch_to_linker` (which takes an `Arc<KernelCache>`
/// directly) — but the scheduler context is per-store, not shared, so
/// the getter form is what fits.
///
/// The getter is invoked from inside both host functions on every
/// call, so it must be cheap (typically just `&store_data.scheduler`).
/// `Send + Sync + 'static` is required by wasmtime's linker bounds.
pub fn add_scheduler_to_linker<T>(
    linker: &mut Linker<T>,
    get_ctx: impl Fn(&T) -> &SchedulerContext + Send + Sync + Copy + 'static,
) -> wasmtime::Result<()>
where
    T: 'static,
{
    linker.func_wrap(
        SCHEDULER_MODULE,
        FN_YIELD,
        move |caller: Caller<'_, T>| -> u32 {
            // No capability gate here — the scheduler surface is
            // cooperative and idempotent: a guest that doesn't have
            // a deadline configured sees a context constructed via
            // `SchedulerContext::unbounded()` (the default), which
            // always returns CONTINUE. Gating would just force every
            // embedder to wire a capability flag through for no
            // protection benefit (no resources are consumed by the
            // call beyond an atomic increment).
            get_ctx(caller.data()).yield_now()
        },
    )?;

    linker.func_wrap(
        SCHEDULER_MODULE,
        FN_DEADLINE_REMAINING_MS,
        move |caller: Caller<'_, T>| -> u32 {
            get_ctx(caller.data()).deadline_remaining_ms()
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn unbounded_yield_always_continue() {
        let ctx = SchedulerContext::unbounded();
        for _ in 0..16 {
            assert_eq!(ctx.yield_now(), YIELD_CODE_CONTINUE);
        }
    }

    #[test]
    fn unbounded_remaining_is_max() {
        let ctx = SchedulerContext::unbounded();
        assert_eq!(ctx.deadline_remaining_ms(), u32::MAX);
    }

    #[test]
    fn yield_count_increments() {
        let ctx = SchedulerContext::new(Some(1_000_000));
        assert_eq!(ctx.yield_count(), 0);
        for i in 1..=5 {
            ctx.yield_now();
            assert_eq!(ctx.yield_count(), i);
        }
    }

    #[test]
    fn yield_count_shared_across_clones() {
        // Because the counter is an `Arc<AtomicU32>`, cloning the
        // context must NOT fork the counter — both host-function
        // closures register against the same instance and must
        // observe the same yield total.
        let a = SchedulerContext::new(Some(1_000_000));
        let b = a.clone();
        a.yield_now();
        b.yield_now();
        a.yield_now();
        assert_eq!(a.yield_count(), 3);
        assert_eq!(b.yield_count(), 3);
    }

    #[test]
    fn expired_deadline_returns_stop() {
        // Construct a context whose 1 ms budget is guaranteed to be
        // burned by the time `yield_now` runs.
        let ctx = SchedulerContext::new(Some(1));
        thread::sleep(Duration::from_millis(15));
        assert_eq!(ctx.yield_now(), YIELD_CODE_STOP);
    }

    #[test]
    fn near_deadline_returns_approaching() {
        // 5 ms total budget — well below the 10 ms approaching
        // threshold, so the very first yield returns APPROACHING
        // regardless of how fast the test machine is. (A STOP would
        // also be acceptable on a very slow runner that burned the
        // full 5 ms between construction and the yield call; accept
        // either non-CONTINUE code.)
        let ctx = SchedulerContext::new(Some(5));
        let code = ctx.yield_now();
        assert!(
            code == YIELD_CODE_DEADLINE_APPROACHING || code == YIELD_CODE_STOP,
            "expected APPROACHING or STOP, got {code}"
        );
    }

    #[test]
    fn fresh_long_deadline_returns_continue() {
        // 60s deadline — far above the threshold — must return
        // CONTINUE on the first yield.
        let ctx = SchedulerContext::new(Some(60_000));
        assert_eq!(ctx.yield_now(), YIELD_CODE_CONTINUE);
    }

    #[test]
    fn deadline_remaining_ms_decreases() {
        let ctx = SchedulerContext::new(Some(200));
        let before = ctx.deadline_remaining_ms();
        thread::sleep(Duration::from_millis(50));
        let after = ctx.deadline_remaining_ms();
        assert!(
            after < before,
            "remaining should strictly decrease over a 50 ms sleep: before={before}, after={after}"
        );
        // Sanity bound: 200 ms budget minus a 50 ms sleep should
        // leave well under 200 ms.
        assert!(after < 200);
    }

    #[test]
    fn deadline_remaining_ms_saturates_to_zero() {
        let ctx = SchedulerContext::new(Some(1));
        thread::sleep(Duration::from_millis(15));
        // Past the deadline — saturating subtraction must report 0,
        // not wrap.
        assert_eq!(ctx.deadline_remaining_ms(), 0);
    }

    #[test]
    fn rearm_resets_elapsed_window() {
        let mut ctx = SchedulerContext::new(Some(50));
        thread::sleep(Duration::from_millis(30));
        ctx.rearm();
        // After re-arm the remaining budget should be back near the
        // full 50 ms — definitely more than the pre-rearm reading
        // would have been (which was approaching 20 ms).
        let remaining = ctx.deadline_remaining_ms();
        assert!(
            remaining > 30,
            "after rearm remaining should be near full budget, got {remaining}"
        );
    }

    #[test]
    fn set_deadline_ms_swaps_budget() {
        let mut ctx = SchedulerContext::new(None);
        assert_eq!(ctx.deadline_remaining_ms(), u32::MAX);
        ctx.set_deadline_ms(Some(100));
        assert!(ctx.deadline_remaining_ms() <= 100);
        ctx.set_deadline_ms(None);
        assert_eq!(ctx.deadline_remaining_ms(), u32::MAX);
    }

    #[test]
    fn default_is_unbounded() {
        let ctx = SchedulerContext::default();
        assert_eq!(ctx.deadline_ms(), None);
        assert_eq!(ctx.yield_now(), YIELD_CODE_CONTINUE);
    }
}
