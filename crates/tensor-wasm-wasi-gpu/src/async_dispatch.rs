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

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Default maximum number of concurrent GPU operations across the process.
/// Mirrors the plan's choice of "a few times the number of SMs" — tuned
/// at startup to match the deployed hardware in S17.
pub const DEFAULT_MAX_CONCURRENT_GPU_OPS: usize = 256;

/// A back-pressure semaphore plus a live-counter for observability.
#[derive(Clone)]
pub struct BackPressure {
    inner: Arc<BackPressureInner>,
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
        }
    }

    /// Acquire one permit, awaiting back-pressure if necessary.
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

impl Drop for DispatchPermit {
    fn drop(&mut self) {
        // SAFETY: the permit's own Drop releases the slot.
        self.permit = None;
        self.counter.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A future representing an in-flight GPU dispatch.
///
/// On the no-CUDA stub path this resolves immediately. On CUDA hosts the
/// future polls a [`cust::event::Event`] recorded on the launch stream:
/// `event.query()` returns `Ok(())` once the GPU has finished the
/// associated work. Until then we re-schedule via the waker so the
/// wasmtime fiber can continue to be suspended.
///
/// The future carries a `tracing::Span` captured at construction time
/// (the active span belonging to whichever host function created it,
/// typically `wasi_cuda.launch`). Every `poll` call enters that span,
/// so any work — including the `cust::event::Event::query` poll path —
/// is attributed to the originating dispatch in the trace tree even
/// though `poll` is invoked from the Tokio runtime's reactor, not from
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
    /// On CUDA builds: a recorded event whose completion signals kernel
    /// done. We poll `event.query()` from the future. On no-CUDA builds
    /// this field is absent and the future resolves on first poll.
    #[cfg(feature = "cuda")]
    event: Option<cust::event::Event>,
}

impl DispatchFuture {
    /// Build a future bound to the given back-pressure permit. The future
    /// resolves the next time it is polled (no-CUDA path) or once the
    /// CUDA event has fired (CUDA path with `bind_event`).
    pub fn ready(permit: DispatchPermit) -> Self {
        Self {
            _permit: permit,
            dispatch_span: tracing::info_span!("wasi_cuda.dispatch"),
            #[cfg(feature = "cuda")]
            event: None,
        }
    }

    /// Attach a recorded CUDA event to this future (CUDA-only).
    ///
    /// After this call, `poll` will return `Pending` until `event.query()`
    /// reports the work has completed. Without this call (the
    /// [`DispatchFuture::ready`] path) the future still resolves
    /// immediately, matching the no-CUDA semantics.
    #[cfg(feature = "cuda")]
    pub fn with_event(permit: DispatchPermit, event: cust::event::Event) -> Self {
        Self {
            _permit: permit,
            dispatch_span: tracing::info_span!("wasi_cuda.dispatch"),
            event: Some(event),
        }
    }
}

impl std::future::Future for DispatchFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        // Enter the dispatch span for the duration of this poll so the
        // trace stays attached to the originating `wasi_cuda.launch`
        // span tree regardless of which Tokio worker is running us. The
        // returned guard is dropped at the end of this function.
        let _entered = self.dispatch_span.enter();

        // On the no-CUDA path (and the `ready` constructor on CUDA hosts)
        // we resolve immediately. The permit is held until the future is
        // dropped, which provides the back-pressure semantics needed by
        // the host bridge even without real CUDA.
        #[cfg(feature = "cuda")]
        {
            if let Some(ev) = self.event.as_ref() {
                // cust 0.3.2's Event::query returns CudaResult<EventStatus>;
                // the enum is { Ready, NotReady }. EventStatus::Ready means the
                // GPU work has completed (equivalent to Event::synchronize for
                // Unified Memory per the cust docs).
                //
                // On NotReady / Err we DO NOT call `waker.wake_by_ref()`
                // immediately — that creates a busy-spin loop that pins
                // the Tokio worker at 100% CPU until the event finishes
                // (every poll re-wakes synchronously). Instead, we clone
                // the waker, spawn a 50 µs sleep, and have THAT task wake
                // us up. The worker is free to schedule other tasks in
                // the meantime. 50 µs is a reasonable poll interval for
                // GPU events: most kernels take longer than that, so we
                // miss at most one quantum of post-completion latency,
                // and the worker spends ≈ 50 µs / quantum doing other
                // work instead of spinning. v0.4 cuda-async work (see
                // RFC 0001 + F3 bench scaffold) replaces this with a
                // proper `cuStreamAddCallback`-driven waker.
                return match ev.query() {
                    Ok(cust::event::EventStatus::Ready) => std::task::Poll::Ready(()),
                    Ok(cust::event::EventStatus::NotReady) | Err(_) => {
                        let waker = cx.waker().clone();
                        tokio::spawn(async move {
                            tokio::time::sleep(std::time::Duration::from_micros(50)).await;
                            waker.wake();
                        });
                        std::task::Poll::Pending
                    }
                };
            }
        }
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

    #[test]
    fn defaults_are_consistent() {
        let bp = BackPressure::new();
        assert_eq!(bp.max_concurrent(), DEFAULT_MAX_CONCURRENT_GPU_OPS);
        assert_eq!(bp.active(), 0);
    }
}
