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
/// future will await a CUDA Event via `cust::stream::Stream::synchronize`
/// run inside `tokio::task::spawn_blocking`.
pub struct DispatchFuture {
    _permit: DispatchPermit,
    #[cfg(feature = "cuda")]
    event: Option<()>, // real cust::event::Event lands here in S16+
}

impl DispatchFuture {
    /// Build a future bound to the given back-pressure permit. The future
    /// resolves the next time it is polled (no-CUDA path).
    pub fn ready(permit: DispatchPermit) -> Self {
        Self {
            _permit: permit,
            #[cfg(feature = "cuda")]
            event: None,
        }
    }
}

impl std::future::Future for DispatchFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        // Stub: resolve immediately. The permit is held until the future is
        // dropped, which provides the back-pressure semantics needed by the
        // host bridge even without real CUDA.
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
