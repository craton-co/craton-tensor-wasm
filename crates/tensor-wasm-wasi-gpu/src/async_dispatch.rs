// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Async GPU dispatch & back-pressure.
//!
//! After the `launch` host function records a kernel launch (via a CUDA
//! Event on real hardware), Wasmtime suspends the calling Wasm fiber by
//! awaiting a [`DispatchFuture`]. The runtime is free to schedule other
//! Wasm instances in the meantime. The future resolves when the CUDA Event
//! synchronises, signaling kernel completion.
//!
//! On no-CUDA hosts the future resolves immediately (it represents work
//! that "ran" only nominally), but the back-pressure machinery still
//! applies — useful for unit-testing the rate-limit logic.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, SemaphorePermit};

use crate::abi::AbiError;

/// Default maximum number of concurrent GPU operations across the process.
/// Mirrors the plan's choice of "a few times the number of SMs" — tuned
/// at startup to match the deployed hardware in S17.
pub const DEFAULT_MAX_CONCURRENT_GPU_OPS: usize = 256;

/// Window before the configured deadline during which the back-pressure
/// path tightens: new acquires are rejected with
/// [`BackPressureError::DeadlineNear`] so the in-flight cohort can
/// drain without being further saturated by fresh launches. Picked at
/// 50 ms — five default epoch ticks — which empirically lets a
/// per-tile loop wind down (typical tile completion ≤ a few ms)
/// without surrendering useful budget on the common path.
///
/// Kept distinct from the scheduler's own
/// [`crate::scheduler::SUGGESTED_YIELD_THRESHOLD_MS`] (10 ms): that
/// threshold biases the cooperative yield code; this one biases
/// resource admission. The wider window for back-pressure gives the
/// scheduler an additional 40 ms safety margin to land a STOP code
/// before any new in-flight work is permitted.
pub const DEADLINE_NEAR_WINDOW: Duration = Duration::from_millis(50);

/// Errors returned by the deadline-aware back-pressure acquire path.
///
/// Distinct from [`AbiError`] so callers that want to discriminate
/// "saturated forever" from "saturated because the per-instance
/// deadline is approaching" can do so without having to inspect the
/// surrounding context. Conversions to [`AbiError`] (for back-compat
/// with the existing host-function return path) collapse both
/// deadline variants onto [`AbiError::QuotaExceeded`] — the guest sees
/// the same "no permits available" signal it already handles, and the
/// richer variant survives in logs / metrics that bind directly to
/// `BackPressureError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackPressureError {
    /// The semaphore is saturated and (for the cap-0 sentinel) will
    /// never release a permit. Equivalent to the historical
    /// [`AbiError::QuotaExceeded`] surface — kept under that name so
    /// the conversion path is unambiguous.
    Saturated,
    /// The configured per-invocation deadline is within
    /// [`DEADLINE_NEAR_WINDOW`] of elapsing. In-flight permits are
    /// allowed to complete; new acquires are refused so the cohort
    /// drains under bounded budget.
    DeadlineNear,
    /// The configured per-invocation deadline has already elapsed.
    /// All future acquires (including pending awaiters that have not
    /// yet observed a permit) are rejected.
    DeadlineElapsed,
}

impl BackPressureError {
    /// Stable, human-readable name (used for log fields).
    pub fn name(self) -> &'static str {
        match self {
            BackPressureError::Saturated => "saturated",
            BackPressureError::DeadlineNear => "deadline_near",
            BackPressureError::DeadlineElapsed => "deadline_elapsed",
        }
    }
}

impl From<BackPressureError> for AbiError {
    fn from(e: BackPressureError) -> Self {
        // Collapse onto QuotaExceeded — the wire shape every existing
        // guest already knows. The richer variant survives in
        // structured logs / metrics that bind directly to
        // `BackPressureError`.
        match e {
            BackPressureError::Saturated
            | BackPressureError::DeadlineNear
            | BackPressureError::DeadlineElapsed => AbiError::QuotaExceeded,
        }
    }
}

/// A back-pressure semaphore plus a live-counter for observability.
///
/// The semaphore itself lives behind an `Arc<BackPressureInner>` so
/// multiple `BackPressure` clones share the same permit pool — this is
/// what makes the cap process-wide rather than per-instance. The
/// optional per-call deadline lives **on the clone** (not inside the
/// `Arc`) so two instances pulling from the same shared pool can each
/// carry their own deadline without racing on a shared `Mutex`.
#[derive(Clone)]
pub struct BackPressure {
    inner: Arc<BackPressureInner>,
    /// Per-clone deadline used by the deadline-aware acquire path.
    /// `None` means "no deadline configured" — the acquire path
    /// behaves exactly as before. Installed via
    /// [`BackPressure::with_deadline_hint`].
    deadline: Option<Instant>,
}

struct BackPressureInner {
    semaphore: Arc<Semaphore>,
    active: AtomicUsize,
    max_concurrent: usize,
}

impl BackPressure {
    /// Construct with the default concurrency cap.
    pub fn new() -> Self {
        Self::with_cap(DEFAULT_MAX_CONCURRENT_GPU_OPS)
    }

    /// Construct with an explicit concurrency cap.
    pub fn with_cap(max_concurrent: usize) -> Self {
        Self {
            inner: Arc::new(BackPressureInner {
                semaphore: Arc::new(Semaphore::new(max_concurrent)),
                active: AtomicUsize::new(0),
                max_concurrent,
            }),
            deadline: None,
        }
    }

    /// Attach a per-invocation deadline to this `BackPressure` clone.
    ///
    /// The deadline is consulted on every acquire (both `acquire` and
    /// `acquire_borrowed`):
    ///
    /// - If `Instant::now() >= deadline`, the acquire returns
    ///   [`BackPressureError::DeadlineElapsed`] without awaiting.
    /// - If `Instant::now() >= deadline - DEADLINE_NEAR_WINDOW`, the
    ///   acquire returns [`BackPressureError::DeadlineNear`] — the
    ///   in-flight cohort is permitted to complete but no new permits
    ///   are issued.
    /// - Otherwise the acquire behaves exactly as before (`None`
    ///   deadline = unchanged behaviour).
    ///
    /// Builder method: consumes `self` and returns a new clone with
    /// the deadline installed. The underlying semaphore is shared via
    /// `Arc` so two clones that pull from the same pool can each
    /// carry their own deadline. Passing `None` is the documented
    /// "no deadline" knob — equivalent to a fresh `BackPressure`
    /// constructed via [`BackPressure::with_cap`].
    ///
    /// See [`crate::scheduler::SchedulerContext`] for the matching
    /// cooperative-yield query path: the executor builds both from
    /// the same `Instant` so the guest's `yield()` verdicts and the
    /// host's acquire decisions agree on when the deadline trips.
    pub fn with_deadline_hint(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Borrow the per-clone deadline, if any. Mirrors
    /// [`BackPressure::with_deadline_hint`] — useful for tests and
    /// observability paths that want to confirm the executor wired the
    /// deadline through.
    pub fn deadline_hint(&self) -> Option<Instant> {
        self.deadline
    }

    /// Inspect the per-clone deadline and classify the current state.
    /// Returns `Ok(())` when the acquire path should proceed,
    /// `Err(DeadlineNear)` when in-flight may complete but new
    /// acquires are refused, and `Err(DeadlineElapsed)` when all
    /// acquires must be refused.
    fn check_deadline(&self) -> Result<(), BackPressureError> {
        let Some(d) = self.deadline else {
            return Ok(());
        };
        let now = Instant::now();
        if now >= d {
            return Err(BackPressureError::DeadlineElapsed);
        }
        // `d.checked_duration_since(now)` is positive here (now < d);
        // the remaining-window comparison reads more naturally that
        // way than juggling Instant arithmetic.
        let remaining = d.saturating_duration_since(now);
        if remaining <= DEADLINE_NEAR_WINDOW {
            return Err(BackPressureError::DeadlineNear);
        }
        Ok(())
    }

    /// Acquire one permit, awaiting back-pressure if necessary.
    ///
    /// Returns an *owned* permit (`'static`) suitable for callers that move
    /// the permit across `tokio::spawn` boundaries (tests, benches, any
    /// future fan-out path). Each call clones the inner `Arc<Semaphore>`.
    /// On the non-spawning production hot path, prefer
    /// [`BackPressure::acquire_borrowed`] which avoids that clone.
    ///
    /// Deadline-hint-unaware: this method returns an unconditional
    /// permit and never refuses on deadline grounds. Callers that
    /// need the deadline-aware rejection path must use
    /// [`BackPressure::try_acquire_with_deadline`] or
    /// [`BackPressure::acquire_borrowed`]; this method is kept on the
    /// pre-deadline surface for tests / benches that pre-date the
    /// deadline plumbing and would otherwise need a typed-error
    /// rewrite. (The production `launch_impl_async` path uses
    /// `acquire_borrowed`, which DOES honour the deadline.)
    pub async fn acquire(&self) -> DispatchPermit {
        let permit = self
            .inner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed unexpectedly");
        self.inner.active.fetch_add(1, Ordering::Relaxed);
        DispatchPermit {
            permit: Some(permit),
            counter: self.inner.clone(),
        }
    }

    /// Deadline-aware counterpart to [`BackPressure::acquire`] that
    /// returns the typed [`BackPressureError`] on rejection.
    ///
    /// On the no-deadline path this behaves exactly like
    /// [`BackPressure::acquire`] but with the typed error surface.
    /// When a deadline is configured (via
    /// [`BackPressure::with_deadline_hint`]):
    ///
    /// - Past the deadline: returns `Err(DeadlineElapsed)`.
    /// - Within `DEADLINE_NEAR_WINDOW` of the deadline: returns
    ///   `Err(DeadlineNear)` — in-flight acquires already issued are
    ///   not affected (they hold their permits to completion); only
    ///   *new* acquires are refused, allowing the cohort to drain.
    /// - More than `DEADLINE_NEAR_WINDOW` away: behaves as
    ///   [`BackPressure::acquire`] (awaits a permit).
    pub async fn acquire_with_deadline(&self) -> Result<DispatchPermit, BackPressureError> {
        // Pre-await deadline check: if we already know the acquire
        // must be refused, do so without entering the semaphore
        // queue. The post-await re-check below handles the case
        // where the deadline trips WHILE waiting.
        self.check_deadline()?;
        let permit_fut = self.inner.semaphore.clone().acquire_owned();
        // Wait for either the permit or the deadline — whichever
        // fires first. Without a deadline configured we just await
        // the permit directly.
        let permit = if let Some(d) = self.deadline {
            tokio::select! {
                p = permit_fut => p.expect("semaphore closed unexpectedly"),
                _ = tokio::time::sleep_until(d.into()) => {
                    return Err(BackPressureError::DeadlineElapsed);
                }
            }
        } else {
            permit_fut.await.expect("semaphore closed unexpectedly")
        };
        // Re-check after the await: the deadline may have entered
        // the NEAR window or even elapsed while we were queued for
        // a permit. Drop the just-acquired permit on rejection so
        // a queued cohort behind us still sees a fair release.
        if let Err(e) = self.check_deadline() {
            drop(permit);
            return Err(e);
        }
        self.inner.active.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchPermit {
            permit: Some(permit),
            counter: self.inner.clone(),
        })
    }

    /// Non-blocking variant of [`BackPressure::acquire_with_deadline`].
    /// Returns `Err(Saturated)` when no permit is immediately
    /// available *and* the deadline does not pre-empt the saturation
    /// path with a more specific code.
    pub fn try_acquire_with_deadline(&self) -> Result<DispatchPermit, BackPressureError> {
        self.check_deadline()?;
        let permit = self
            .inner
            .semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| BackPressureError::Saturated)?;
        self.inner.active.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchPermit {
            permit: Some(permit),
            counter: self.inner.clone(),
        })
    }

    /// Acquire one permit, awaiting back-pressure if necessary.
    ///
    /// Returns a *borrowed* permit whose lifetime is tied to `&self`. This
    /// avoids the `Arc<Semaphore>` clone that [`BackPressure::acquire`]
    /// performs and is the right choice for callers that hold the permit
    /// only within the current async scope (no `tokio::spawn`). The
    /// production `launch_impl_async` host-function path runs inside a
    /// wasmtime async fiber that never spawns, so it uses this variant.
    ///
    /// # Cap-0 / saturated semantics
    ///
    /// A cap of `0` is a "no permits, ever" sentinel: the semaphore was
    /// constructed with zero permits and none will ever be released
    /// because nothing can acquire to release. Awaiting `Semaphore::acquire`
    /// in that state parks the caller forever — a footgun, because guests
    /// observing back-pressure should see `QuotaExceeded` rather than
    /// hang. We therefore probe with `try_acquire` first: if it fails
    /// AND the configured cap is `0`, we return [`AbiError::QuotaExceeded`]
    /// rather than awaiting. For caps > 0 we fall through to the standard
    /// async acquire (permits will eventually return as in-flight
    /// dispatches drop their permits).
    pub async fn acquire_borrowed(&self) -> Result<BorrowedDispatchPermit<'_>, AbiError> {
        // Pre-await deadline check (T36 — cooperative deadlines).
        // If the per-invocation deadline says the acquire must be
        // refused, bail out *before* even probing the semaphore: a
        // guest hammering `launch` past its deadline must not be
        // able to drain in-flight permits by racing the rejection
        // check.
        if let Err(e) = self.check_deadline() {
            return Err(e.into());
        }
        // Fast path / saturated-cap-0 guard: try a synchronous acquire
        // first. On success we skip the async machinery entirely; on
        // failure we check whether the cap is the cap-0 sentinel and, if
        // so, surface `QuotaExceeded` instead of parking forever.
        if let Ok(permit) = self.inner.semaphore.try_acquire() {
            self.inner.active.fetch_add(1, Ordering::Relaxed);
            return Ok(BorrowedDispatchPermit {
                permit: Some(permit),
                counter: &self.inner,
            });
        }
        if self.inner.max_concurrent == 0 {
            // Cap-0 semaphores never release a permit; awaiting would
            // park the wasm fiber indefinitely. Return the saturated-
            // back-pressure signal instead.
            return Err(AbiError::QuotaExceeded);
        }
        // Race the semaphore acquire against the deadline (when
        // configured). The first arm to resolve wins; the others are
        // dropped. Without a deadline we just await the permit.
        let permit = if let Some(d) = self.deadline {
            tokio::select! {
                p = self.inner.semaphore.acquire() => p.expect("semaphore closed unexpectedly"),
                _ = tokio::time::sleep_until(d.into()) => {
                    return Err(BackPressureError::DeadlineElapsed.into());
                }
            }
        } else {
            self.inner
                .semaphore
                .acquire()
                .await
                .expect("semaphore closed unexpectedly")
        };
        // Post-await re-check: the deadline may have crossed into
        // the NEAR window (or elapsed) while we waited in the
        // semaphore queue. Drop the freshly-acquired permit on
        // rejection so a queued cohort behind us still sees fair
        // release of the slot.
        if let Err(e) = self.check_deadline() {
            drop(permit);
            return Err(e.into());
        }
        self.inner.active.fetch_add(1, Ordering::Relaxed);
        Ok(BorrowedDispatchPermit {
            permit: Some(permit),
            counter: &self.inner,
        })
    }

    /// Try to acquire a permit without awaiting. Returns `None` under load.
    pub fn try_acquire(&self) -> Option<DispatchPermit> {
        let permit = self.inner.semaphore.clone().try_acquire_owned().ok()?;
        self.inner.active.fetch_add(1, Ordering::Relaxed);
        Some(DispatchPermit {
            permit: Some(permit),
            counter: self.inner.clone(),
        })
    }

    /// Current number of in-flight dispatches.
    pub fn active(&self) -> usize {
        self.inner.active.load(Ordering::Relaxed)
    }

    /// Maximum concurrent dispatches.
    pub fn max_concurrent(&self) -> usize {
        self.inner.max_concurrent
    }
}

impl Default for BackPressure {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII permit returned by [`BackPressure::acquire`]. Dropping it releases
/// the underlying semaphore slot and decrements the live counter.
pub struct DispatchPermit {
    permit: Option<OwnedSemaphorePermit>,
    counter: Arc<BackPressureInner>,
}

impl std::fmt::Debug for DispatchPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchPermit")
            .field("permit_held", &self.permit.is_some())
            .finish()
    }
}

impl Drop for DispatchPermit {
    fn drop(&mut self) {
        // SAFETY: the permit's own Drop releases the slot.
        self.permit = None;
        self.counter.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Borrowed counterpart to [`DispatchPermit`] returned by
/// [`BackPressure::acquire_borrowed`]. Its lifetime is bound to the
/// `&BackPressure` it was acquired from, so it cannot cross a
/// `tokio::spawn` boundary — use [`DispatchPermit`] for that. In return,
/// acquisition skips the `Arc<Semaphore>` clone the owned variant pays.
///
/// Drop semantics are identical to [`DispatchPermit`]: releasing the
/// semaphore slot and decrementing the live-counter.
pub struct BorrowedDispatchPermit<'a> {
    permit: Option<SemaphorePermit<'a>>,
    counter: &'a BackPressureInner,
}

impl std::fmt::Debug for BorrowedDispatchPermit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorrowedDispatchPermit")
            .field("issued", &self.permit.is_some())
            .finish()
    }
}

impl Drop for BorrowedDispatchPermit<'_> {
    fn drop(&mut self) {
        // SAFETY: the permit's own Drop releases the slot.
        self.permit = None;
        self.counter.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Shared completion state driven by a `cuStreamAddCallback` host callback
/// (CUDA-only).
///
/// The driver invokes [`stream_completion_trampoline`] on one of its own
/// internal threads once all work enqueued on the stream *before* the
/// callback has finished. The trampoline flips [`done`](Self::done) and
/// wakes the [`waker`](Self::waker) the future last parked, so the
/// wasmtime fiber resumes exactly once — no busy-poll, no fixed-interval
/// sleep. The future, in turn, stores the latest waker on every `Pending`
/// poll so a runtime task-migration cannot strand a stale waker.
#[cfg(feature = "cuda")]
struct CallbackState {
    /// Set `true` by the driver callback once the stream work completes.
    done: std::sync::atomic::AtomicBool,
    /// The most recent waker parked by the future. The callback takes it
    /// and wakes it; the future replaces it on every `Pending` poll.
    waker: std::sync::Mutex<Option<std::task::Waker>>,
}

/// `extern "C"` trampoline handed to `cuStreamAddCallback`.
///
/// # Safety
///
/// `user_data` MUST be the `Arc::into_raw(Arc<CallbackState>)` pointer the
/// future passed when it registered the callback. The driver guarantees it
/// invokes this exactly once per registration, so reconstructing the `Arc`
/// with `Arc::from_raw` here consumes precisely the strong reference the
/// registration leaked — balancing the refcount with no leak and no
/// double-free.
#[cfg(feature = "cuda")]
unsafe extern "C" fn stream_completion_trampoline(
    _stream: cust::sys::CUstream,
    _status: cust::sys::CUresult,
    user_data: *mut std::ffi::c_void,
) {
    if user_data.is_null() {
        return;
    }
    // Reclaim the strong ref the registration leaked via `Arc::into_raw`.
    let state: Arc<CallbackState> = Arc::from_raw(user_data as *const CallbackState);
    // Mark complete BEFORE waking so the woken poll observes `done == true`.
    // `_status` may report an error; we still resolve the future — the
    // launch-side error handling (cuLaunchKernel return code, the
    // synchronize path) owns error reporting, and the dispatch future's
    // only job is to signal "the stream drained."
    state.done.store(true, Ordering::Release);
    if let Some(w) = state.waker.lock().unwrap_or_else(|e| e.into_inner()).take() {
        w.wake();
    }
    // `state` drops here, releasing the registration's strong ref.
}

/// A future representing an in-flight GPU dispatch.
///
/// On the no-CUDA stub path this resolves immediately. On CUDA hosts the
/// future is driven by a real `cuStreamAddCallback` waker: the driver wakes
/// the task once the stream's queued work has drained, so the future
/// returns `Pending` exactly once and resumes on a genuine completion
/// signal — no busy-spin and no fixed-interval sleep poll.
///
/// The future carries a `tracing::Span` captured at construction time
/// (the active span belonging to whichever host function created it,
/// typically `wasi_cuda.launch`). Every `poll` call enters that span,
/// so any work is attributed to the originating dispatch in the trace tree
/// even though `poll` is invoked from the Tokio runtime's reactor, not from
/// inside the host function. Without this the dispatch's poll events
/// would be parented to the runtime worker's empty context and would
/// disappear from the distributed trace.
pub struct DispatchFuture {
    _permit: DispatchPermit,
    /// Span entered on every poll so the future stays attached to the
    /// dispatch trace context across runtime task switches. Holding a
    /// `Span` (rather than an `EnteredSpan`) is cheap (it's a shallow
    /// `Arc`-clone) and lets us re-enter on each poll without leaking
    /// the guard.
    dispatch_span: tracing::Span,
    /// On CUDA builds: a recorded event kept alive for the future's
    /// lifetime so the stream's completion bookkeeping outlives the
    /// callback. `None` on the [`DispatchFuture::ready`] path (resolves
    /// immediately). On no-CUDA builds the field is absent.
    ///
    /// Held purely as a lifetime keepalive — the callback waker drives
    /// completion, so the field is never read, only dropped.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    event: Option<cust::event::Event>,
    /// On CUDA builds: the stream the callback is (or will be) registered
    /// on. `None` on the immediate-resolution path.
    #[cfg(feature = "cuda")]
    stream: Option<cust::stream::Stream>,
    /// On CUDA builds: the shared completion state the driver callback
    /// drives. Lazily created and the callback lazily registered on the
    /// first `Pending` poll so the registration captures a live waker.
    #[cfg(feature = "cuda")]
    callback_state: Option<Arc<CallbackState>>,
}

impl DispatchFuture {
    /// Build a future bound to the given back-pressure permit. The future
    /// resolves the next time it is polled (no-CUDA path, or the CUDA path
    /// with no event/stream attached).
    pub fn ready(permit: DispatchPermit) -> Self {
        Self {
            _permit: permit,
            dispatch_span: tracing::info_span!("wasi_cuda.dispatch"),
            #[cfg(feature = "cuda")]
            event: None,
            #[cfg(feature = "cuda")]
            stream: None,
            #[cfg(feature = "cuda")]
            callback_state: None,
        }
    }

    /// Attach a recorded CUDA event to this future (CUDA-only).
    ///
    /// Kept for source-compat with callers that only hold an event. Without
    /// a stream the future cannot register a `cuStreamAddCallback` waker, so
    /// it resolves on the next poll — the event is held only as a lifetime
    /// keepalive. Prefer [`DispatchFuture::with_stream_event`] for the
    /// real callback-driven path.
    #[cfg(feature = "cuda")]
    pub fn with_event(permit: DispatchPermit, event: cust::event::Event) -> Self {
        Self {
            _permit: permit,
            dispatch_span: tracing::info_span!("wasi_cuda.dispatch"),
            event: Some(event),
            stream: None,
            callback_state: None,
        }
    }

    /// Attach the launch stream + recorded event (CUDA-only).
    ///
    /// This is the real callback-driven path: on the first `Pending` poll
    /// the future registers [`stream_completion_trampoline`] on `stream`
    /// via `cuStreamAddCallback`, and the driver wakes the task once the
    /// stream's queued work has drained. The `event` is held purely as a
    /// lifetime keepalive so its completion bookkeeping outlives the
    /// callback.
    #[cfg(feature = "cuda")]
    pub fn with_stream_event(
        permit: DispatchPermit,
        stream: cust::stream::Stream,
        event: cust::event::Event,
    ) -> Self {
        Self {
            _permit: permit,
            dispatch_span: tracing::info_span!("wasi_cuda.dispatch"),
            event: Some(event),
            stream: Some(stream),
            callback_state: None,
        }
    }
}

impl std::future::Future for DispatchFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        // `self` is `Pin<&mut Self>`; none of our fields are structurally
        // pinned (no self-referential state), so projecting to an unpinned
        // `&mut Self` is sound. We take the `&mut` BEFORE entering the span so
        // the span guard (which borrows `this.dispatch_span`) and the
        // mutable field access below do not contend for `self`.
        let this = unsafe { self.get_unchecked_mut() };

        // Enter the dispatch span for the duration of this poll so the
        // trace stays attached to the originating `wasi_cuda.launch`
        // span tree regardless of which Tokio worker is running us. The
        // returned guard is dropped at the end of this function.
        let _entered = this.dispatch_span.enter();

        // On the no-CUDA path (and the `ready` / event-only constructors on
        // CUDA hosts) we resolve immediately. The permit is held until the
        // future is dropped, which provides the back-pressure semantics
        // needed by the host bridge even without real CUDA.
        #[cfg(feature = "cuda")]
        {
            let Some(stream) = this.stream.as_ref() else {
                // No stream attached → nothing to wait on. The event (if any)
                // is just a keepalive; resolve now.
                return std::task::Poll::Ready(());
            };

            // First Pending poll: create the shared state, park the waker,
            // and register the driver callback. The callback consumes one
            // leaked strong ref (`Arc::into_raw`); the trampoline reclaims it.
            if this.callback_state.is_none() {
                let state = Arc::new(CallbackState {
                    done: std::sync::atomic::AtomicBool::new(false),
                    waker: std::sync::Mutex::new(Some(cx.waker().clone())),
                });
                let raw = Arc::into_raw(Arc::clone(&state)) as *mut std::ffi::c_void;
                // SAFETY: `stream.as_inner()` is a live CUstream; `raw` is a
                // valid `Arc<CallbackState>` pointer the trampoline reclaims
                // exactly once. The driver guarantees a single invocation.
                let status = unsafe {
                    cust::sys::cuStreamAddCallback(
                        stream.as_inner(),
                        Some(stream_completion_trampoline),
                        raw,
                        0,
                    )
                };
                if status != cust::sys::CUresult::CUDA_SUCCESS {
                    // Registration failed: reclaim the leaked ref so we don't
                    // leak, and fall back to resolving immediately rather than
                    // hanging the fiber. A failed callback registration is
                    // surfaced by the launch-side synchronize path, not here.
                    // SAFETY: `raw` came from `Arc::into_raw` just above and
                    // the driver did not take ownership (registration failed).
                    drop(unsafe { Arc::from_raw(raw as *const CallbackState) });
                    return std::task::Poll::Ready(());
                }
                this.callback_state = Some(state);
                return std::task::Poll::Pending;
            }

            // Subsequent polls: the callback may already have fired. Check
            // the completion flag; if not done, refresh the parked waker so a
            // task migration between workers cannot strand a stale waker.
            let state = this.callback_state.as_ref().expect("state present");
            if state.done.load(Ordering::Acquire) {
                return std::task::Poll::Ready(());
            }
            *state.waker.lock().unwrap_or_else(|e| e.into_inner()) = Some(cx.waker().clone());
            // Re-check after re-parking to close the race where the callback
            // fired between the load above and storing the new waker.
            if state.done.load(Ordering::Acquire) {
                return std::task::Poll::Ready(());
            }
            return std::task::Poll::Pending;
        }
        #[cfg(not(feature = "cuda"))]
        std::task::Poll::Ready(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release() {
        let bp = BackPressure::with_cap(2);
        assert_eq!(bp.active(), 0);
        let a = bp.acquire().await;
        assert_eq!(bp.active(), 1);
        let b = bp.acquire().await;
        assert_eq!(bp.active(), 2);
        drop(a);
        drop(b);
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn try_acquire_under_pressure() {
        let bp = BackPressure::with_cap(1);
        let a = bp.acquire().await;
        assert!(
            bp.try_acquire().is_none(),
            "second permit should be unavailable"
        );
        drop(a);
        assert!(
            bp.try_acquire().is_some(),
            "permit should be available again"
        );
    }

    #[tokio::test]
    async fn dispatch_future_resolves_immediately() {
        let bp = BackPressure::with_cap(4);
        let permit = bp.acquire().await;
        let fut = DispatchFuture::ready(permit);
        fut.await;
        // Permit released — counter should be back to zero.
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn concurrent_acquire_progresses() {
        // 1000 awaits with a cap of 64 should all complete.
        let bp = BackPressure::with_cap(64);
        let mut handles = Vec::new();
        for _ in 0..1000 {
            let bp = bp.clone();
            handles.push(tokio::spawn(async move {
                let permit = bp.acquire().await;
                DispatchFuture::ready(permit).await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn acquire_borrowed_and_release() {
        // Mirrors `acquire_and_release` but for the borrowed variant: the
        // live-counter must rise on acquire and fall on drop just as it
        // does for the owned `DispatchPermit`.
        let bp = BackPressure::with_cap(2);
        assert_eq!(bp.active(), 0);
        let a = bp.acquire_borrowed().await.expect("permit");
        assert_eq!(bp.active(), 1);
        let b = bp.acquire_borrowed().await.expect("permit");
        assert_eq!(bp.active(), 2);
        drop(a);
        drop(b);
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn acquire_borrowed_cap_zero_returns_quota_exceeded() {
        // Regression: a cap-0 BackPressure used to hang on
        // `Semaphore::acquire` because no permit will ever be released.
        // The fix is to detect the saturated-cap-0 case and return
        // `QuotaExceeded` synchronously instead of parking forever.
        let bp = BackPressure::with_cap(0);
        let err = bp.acquire_borrowed().await.expect_err("cap=0 must error");
        assert_eq!(err, AbiError::QuotaExceeded);
        // Repeated calls keep returning the same error rather than
        // parking — the contract is "saturated forever".
        let err2 = bp.acquire_borrowed().await.expect_err("cap=0 must error");
        assert_eq!(err2, AbiError::QuotaExceeded);
        // No permit was ever issued; the live-counter stayed at zero.
        assert_eq!(bp.active(), 0);
    }

    #[test]
    fn defaults_are_consistent() {
        let bp = BackPressure::new();
        assert_eq!(bp.max_concurrent(), DEFAULT_MAX_CONCURRENT_GPU_OPS);
        assert_eq!(bp.active(), 0);
    }

    // ----------------------------------------------------------------
    // Deadline-aware tests (T36).
    // ----------------------------------------------------------------

    #[test]
    fn deadline_hint_round_trips() {
        // The builder consumes self and returns a clone with the
        // deadline installed; the underlying semaphore Arc is shared,
        // so two clones can carry distinct deadlines while still
        // contending on the same pool.
        let bp = BackPressure::with_cap(4);
        assert!(bp.deadline_hint().is_none());
        let d = Instant::now() + Duration::from_secs(1);
        let bp = bp.with_deadline_hint(Some(d));
        assert_eq!(bp.deadline_hint(), Some(d));
    }

    #[tokio::test]
    async fn acquire_borrowed_rejects_new_under_deadline_near() {
        // Deadline 30 ms out, inside the 50 ms NEAR window from
        // construction. The first acquire must be rejected with the
        // DeadlineNear-mapped QuotaExceeded code.
        let bp = BackPressure::with_cap(4)
            .with_deadline_hint(Some(Instant::now() + Duration::from_millis(30)));
        let err = bp
            .acquire_borrowed()
            .await
            .expect_err("near-deadline acquire must be refused");
        assert_eq!(err, AbiError::QuotaExceeded);
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn acquire_borrowed_rejects_past_deadline() {
        // Deadline in the past — every acquire is refused.
        let bp = BackPressure::with_cap(4)
            .with_deadline_hint(Some(Instant::now() - Duration::from_millis(5)));
        let err = bp
            .acquire_borrowed()
            .await
            .expect_err("elapsed deadline must refuse");
        assert_eq!(err, AbiError::QuotaExceeded);
    }

    #[tokio::test]
    async fn acquire_borrowed_passes_outside_near_window() {
        // Deadline well outside the 50 ms NEAR window — the acquire
        // proceeds as before.
        let bp = BackPressure::with_cap(4)
            .with_deadline_hint(Some(Instant::now() + Duration::from_secs(10)));
        let permit = bp.acquire_borrowed().await.expect("permit");
        assert_eq!(bp.active(), 1);
        drop(permit);
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn in_flight_completes_under_near_deadline() {
        // Acquire BEFORE the deadline enters the NEAR window; then
        // advance into the window. The in-flight permit is
        // unaffected (its Drop releases as normal), but a NEW
        // acquire is refused.
        let bp = BackPressure::with_cap(4)
            .with_deadline_hint(Some(Instant::now() + Duration::from_millis(80)));
        let in_flight = bp.acquire_borrowed().await.expect("first permit");
        assert_eq!(bp.active(), 1);
        // Sleep into the NEAR window (80 ms - 50 ms = 30 ms, so
        // 40 ms gets us safely inside).
        tokio::time::sleep(Duration::from_millis(40)).await;
        let err = bp
            .acquire_borrowed()
            .await
            .expect_err("second acquire must be refused");
        assert_eq!(err, AbiError::QuotaExceeded);
        // The in-flight permit is still live.
        assert_eq!(bp.active(), 1);
        drop(in_flight);
        assert_eq!(bp.active(), 0);
    }

    #[tokio::test]
    async fn typed_acquire_with_deadline_surfaces_variants() {
        // `acquire_with_deadline` keeps the typed `BackPressureError`
        // surface (rather than collapsing to AbiError) so callers
        // that want to distinguish the variants can.
        let bp = BackPressure::with_cap(2)
            .with_deadline_hint(Some(Instant::now() + Duration::from_millis(20)));
        let err = bp
            .acquire_with_deadline()
            .await
            .expect_err("near-deadline must refuse");
        assert_eq!(err, BackPressureError::DeadlineNear);

        let bp = BackPressure::with_cap(2)
            .with_deadline_hint(Some(Instant::now() - Duration::from_millis(5)));
        let err = bp
            .acquire_with_deadline()
            .await
            .expect_err("elapsed-deadline must refuse");
        assert_eq!(err, BackPressureError::DeadlineElapsed);
    }

    #[test]
    fn backpressure_error_converts_to_abi_error() {
        // All variants collapse to QuotaExceeded for back-compat
        // with the existing host-function return path.
        assert_eq!(
            AbiError::from(BackPressureError::Saturated),
            AbiError::QuotaExceeded
        );
        assert_eq!(
            AbiError::from(BackPressureError::DeadlineNear),
            AbiError::QuotaExceeded
        );
        assert_eq!(
            AbiError::from(BackPressureError::DeadlineElapsed),
            AbiError::QuotaExceeded
        );
    }
}
