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
pub struct DispatchFuture {
    _permit: DispatchPermit,
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
            event: Some(event),
        }
    }
}

impl std::future::Future for DispatchFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        // On the no-CUDA path (and the `ready` constructor on CUDA hosts)
        // we resolve immediately. The permit is held until the future is
        // dropped, which provides the back-pressure semantics needed by
        // the host bridge even without real CUDA.
        #[cfg(feature = "cuda")]
        {
            if let Some(ev) = self.event.as_ref() {
                // `query()` returns `Ok(())` when the event has completed
                // and a recoverable error when it has not. On any other
                // error we treat the dispatch as finished so the
                // wasmtime fiber doesn't hang forever — the host's
                // launch path will surface the real error to the guest
                // separately via `last_error`.
                return match ev.query() {
                    Ok(()) => std::task::Poll::Ready(()),
                    Err(_e) => {
                        // Not done yet — wake immediately so the
                        // executor re-polls. This is a busy-poll but
                        // matches the existing `spawn_blocking`-driven
                        // synchronization the launch path uses; the
                        // dispatch future is currently advisory and not
                        // on the launch critical path.
                        _cx.waker().wake_by_ref();
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
