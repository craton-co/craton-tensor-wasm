// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! F3 bench: busy-poll `DispatchFuture` vs `cuda-async`-backed alternative.
//!
//! Harness for [RFC 0001](../../../rfcs/0001-cuda-oxide-integration.md)
//! Unresolved question #3 ("Does cuda-async's Tokio integration
//! outperform our hand-rolled `DispatchFuture` busy-poll?"). Runs two
//! `DispatchFutureBackend` impls through the same no-op dispatch loop
//! (acquire permit → await future → drop) and reports P50/P95/P99/P99.9
//! + max so a v0.4 reviewer reads the speedup (or regression) in one diff.
//!
//! # What runs today (finding 5: both backends wired against the GPU)
//!
//! Both backends now record a real `cust` event on a fresh stream per
//! dispatch (no kernel launched — the event is recorded on an empty stream,
//! so each number is the sync-machinery floor, not a kernel runtime). The
//! difference between them is the completion-wait strategy, which is exactly
//! RFC 0001 Unresolved question #3:
//!
//! - `BusyPollBackend` — drives
//!   `tensor_wasm_wasi_gpu::async_dispatch::DispatchFuture::with_event`, which
//!   polls `event.query()` and yields the Tokio worker for ~50 µs between
//!   polls (the W4.1-instrumented + post-CUDA-debug path). This is the
//!   hand-rolled busy-poll the RFC asks about.
//! - `CudaAsyncBackend` — awaits a
//!   `tokio::task::spawn_blocking(move || stream.synchronize())`, the
//!   blocking-sync-on-a-pool-thread strategy the production launch path uses.
//!   This is the stream-synchronize alternative the RFC contrasts against the
//!   busy-poll.
//!
//! Both require `--features cuda`. Without it the file ships a stub `main`
//! that announces two skips and exits cleanly (build smoke test).
//!
//! # v0.4/v0.5 note
//!
//! The genuine `cuda-async` / cuda-oxide async runtime is still the v0.5 cust
//! successor (it needs the nightly override the workspace defers). Until it
//! lands, `CudaAsyncBackend`'s mechanism is `cust`'s `stream.synchronize`,
//! the closest faithful stand-in. The `name()` stays `"cuda-async"` per the
//! stable-label contract, so when the real runtime ports in, only the
//! `dispatch_once` body changes.
//!
//! # Honest disclosure
//!
//! - Both numbers are the sync-machinery floor over a no-op (empty-stream)
//!   dispatch, NOT launch-to-completion of any particular kernel. The A/B is
//!   "busy-poll vs stream-synchronize wait strategy", which is the question.
//! - `CudaAsyncBackend` uses `cust`'s `stream.synchronize`, not the (not-yet-
//!   ported) cuda-async runtime; read the BusyPoll-vs-CudaAsync delta as
//!   "event-query poll vs blocking stream sync", not "old crate vs new crate".
//! - A skip line now means CUDA was unavailable at runtime, not that the
//!   backend is unimplemented.
//! - `harness = false` Criterion-shaped only so `cargo bench` discovers
//!   it; the real loop is hand-rolled, mirroring W4.6 `tail_latency.rs`
//!   (see its module docs for the 10 000-sample nearest-rank rationale).
//!
//! # Output
//!
//! One `DISPATCH_BACKEND {json}` line per backend to stdout (CI greps
//! for the prefix), plus a full document at
//! `bench-results/dispatch-future-backends.json`. Not consumed by the
//! regression gate — diagnostic only.

// `--features cuda` off: ship a minimal stub `main` that announces both
// skips and exits cleanly so `cargo bench --bench dispatch_future_backends`
// still works as a build smoke test on non-CUDA hosts.
#[cfg(not(feature = "cuda"))]
fn main() {
    println!(
        "DISPATCH_BACKEND {{\"backend\":\"busy-poll\",\"status\":\"skipped\",\"reason\":\"tensor-wasm-bench compiled without --features cuda; rebuild with `--features cuda` to exercise the DispatchFuture path\"}}"
    );
    println!(
        "DISPATCH_BACKEND {{\"backend\":\"cuda-async\",\"status\":\"skipped\",\"reason\":\"tensor-wasm-bench compiled without --features cuda; rebuild with `--features cuda` to exercise the stream-synchronize path\"}}"
    );
}

#[cfg(feature = "cuda")]
mod bench {

    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use criterion::black_box;
    use cust::event::{Event, EventFlags};
    use cust::stream::{Stream, StreamFlags};
    use tensor_wasm_bench::{percentile_nearest_rank, TailStats};
    use tensor_wasm_wasi_gpu::async_dispatch::{BackPressure, DispatchFuture};

    /// Raw observations per backend. Matches the W4.6 floor in
    /// `tail_latency.rs` so the P99.9 sample lands at rank 9 990.
    const SAMPLES: usize = 10_000;

    /// Warm-up iterations before the timed loop, mirroring W4.6.
    const WARMUP: usize = 1_000;

    /// Fallback skip reason for the `cuda-async` backend. Both real backends
    /// now drive `cust` against the GPU (finding 5), so a skip only happens
    /// when CUDA cannot init/bind at runtime (no device, despite the binary
    /// being built with `--features cuda`).
    const CUDA_ASYNC_RUNTIME_UNAVAILABLE: &str =
        "cuda-async: CUDA unavailable at runtime (no device / context bind failed)";

    /// Backend-agnostic shape for one no-op dispatch + permit RAII window.
    /// `Err(&'static str)` signals "skip this backend": today that only fires
    /// when CUDA cannot init at runtime. The bench loop converts it into a
    /// documented skip line rather than counting it as latency.
    trait DispatchFutureBackend {
        #[allow(async_fn_in_trait)] // bench-only trait, no external impls
        async fn dispatch_once(&self) -> Result<(), &'static str>;
        /// Stable JSON label. Must NOT change across releases — the v0.4
        /// wiring contract documents that `cuda-async` stays `cuda-async`.
        fn name(&self) -> &'static str;
    }

    /// Current behaviour: cap=1 back-pressure, drop. The `BackPressure` is
    /// hoisted out of `dispatch_once` so we measure permit acquire/release +
    /// future poll only, not semaphore construction.
    ///
    /// Finding 1: this used to call `DispatchFuture::ready`, which resolves on
    /// the first poll even under `--features cuda` — so the bench measured the
    /// scheduling floor, never GPU sync. We now record a real `cust` event on
    /// a fresh stream per dispatch and drive `DispatchFuture::with_event`, so
    /// this backend exercises the real `event.query()` busy-poll path — the
    /// exact path cuda-async is meant to replace. No kernel is launched (the
    /// event is recorded on an empty stream), so the number is the real-sync
    /// machinery floor, comparable across backends, not a kernel runtime.
    struct BusyPollBackend {
        back_pressure: Arc<BackPressure>,
    }

    impl BusyPollBackend {
        fn new() -> Self {
            Self {
                back_pressure: Arc::new(BackPressure::with_cap(1)),
            }
        }
    }

    impl DispatchFutureBackend for BusyPollBackend {
        async fn dispatch_once(&self) -> Result<(), &'static str> {
            // The polling thread (a tokio worker) must have the primary
            // context current before any `cust` driver call.
            tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context()
                .map_err(|_| "busy-poll: CUDA context bind failed")?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
                .map_err(|_| "busy-poll: Stream::new failed")?;
            let event = Event::new(EventFlags::DEFAULT).map_err(|_| "busy-poll: Event::new failed")?;
            event
                .record(&stream)
                .map_err(|_| "busy-poll: Event::record failed")?;
            let permit = self.back_pressure.acquire().await;
            // Real GPU-sync path: resolves once `event.query()` reports
            // `EventStatus::Ready` via the `poll`/sleep wake loop in
            // `DispatchFuture::poll`.
            DispatchFuture::with_event(permit, event).await;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "busy-poll"
        }
    }

    /// Real GPU-sync alternative to `BusyPollBackend` (finding 5, wired with
    /// the GPU present).
    ///
    /// RFC 0001 Unresolved question #3 asks whether a stream-synchronize
    /// strategy beats the hand-rolled `event.query()` busy-poll. With a device
    /// present we can answer it with real numbers instead of a perpetual skip:
    /// this backend records a `cust` event on a stream and then awaits a
    /// `tokio::task::spawn_blocking(move || stream.synchronize())` — the same
    /// blocking-sync-on-a-pool-thread shape the production launch path in
    /// `tensor_wasm_wasi_gpu::host` uses. The contrast with `BusyPollBackend`
    /// (async `event.query()` poll + 50 µs sleep-yield) is the A/B the RFC
    /// wanted: two real GPU-sync strategies over the same no-op dispatch.
    ///
    /// The `cuda-async`/`cuda-oxide` crate itself is still a v0.4/v0.5 port
    /// (it would need the nightly override the workspace defers), so the
    /// `name()` stays `"cuda-async"` per the stable-label contract and the
    /// underlying mechanism here is `cust`'s `stream.synchronize`, which is
    /// the closest faithful stand-in available in v0.3.x. When the real
    /// cuda-oxide async runtime lands, only this body changes.
    ///
    /// As with `BusyPollBackend`, no kernel is launched (the event is recorded
    /// on an empty stream), so the number is the synchronize-machinery floor,
    /// directly comparable to the busy-poll floor.
    struct CudaAsyncBackend;

    impl DispatchFutureBackend for CudaAsyncBackend {
        async fn dispatch_once(&self) -> Result<(), &'static str> {
            tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context()
                .map_err(|_| "cuda-async: CUDA context bind failed")?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
                .map_err(|_| "cuda-async: Stream::new failed")?;
            let event =
                Event::new(EventFlags::DEFAULT).map_err(|_| "cuda-async: Event::new failed")?;
            event
                .record(&stream)
                .map_err(|_| "cuda-async: Event::record failed")?;
            // Blocking synchronize moved onto the blocking pool so the runtime
            // worker is free meanwhile — the strategy RFC 0001 contrasts with
            // the busy-poll path. `_event` is moved in to keep it alive until
            // the stream drains.
            tokio::task::spawn_blocking(move || {
                let _event = event;
                tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context()
                    .map_err(|_| "cuda-async: context bind on blocking pool failed")?;
                stream
                    .synchronize()
                    .map_err(|_| "cuda-async: stream synchronize failed")
            })
            .await
            .map_err(|_| "cuda-async: spawn_blocking join failed")?
        }

        fn name(&self) -> &'static str {
            "cuda-async"
        }
    }

    // `TailStats` and `percentile_nearest_rank` now live in the crate's
    // `src/lib.rs` (imported above) so the percentile math and JSON schema are
    // shared with `tail_latency.rs` via a single source of truth, with real
    // `cargo test`-visible unit tests. `skip_json` and `render_file` stay local
    // because they are specific to this bench's output document.

    /// Skip line for a backend whose `dispatch_once` returned `Err` (today:
    /// CUDA unavailable at runtime).
    fn skip_json(backend: &str, reason: &str) -> String {
        format!(
            "{{\"backend\":\"{}\",\"status\":\"skipped\",\"reason\":\"{}\"}}",
            tensor_wasm_bench::json_escape(backend),
            tensor_wasm_bench::json_escape(reason),
        )
    }

    /// Run `iters` samples through `backend`. Returns `None` when the probe
    /// `dispatch_once` returns `Err` (CUDA unavailable at runtime); caller
    /// emits a skip line.
    // `async fn` in a trait isn't dyn-compatible (no vtable for the
    // returned `impl Future`), so the runner is generic over a concrete
    // `B: DispatchFutureBackend` instead of `&dyn`. Each backend gets its
    // own monomorphisation — fine because there are only two backends.
    fn bench_one<B: DispatchFutureBackend>(
        rt: &tokio::runtime::Runtime,
        backend: &B,
        iters: usize,
    ) -> Option<TailStats> {
        // Probe before warm-up so a runtime-unavailable backend short-circuits
        // without burning 11 000 iterations. Same shape as the timed loop.
        if rt.block_on(backend.dispatch_once()).is_err() {
            return None;
        }
        for _ in 0..WARMUP {
            let _ = rt.block_on(backend.dispatch_once());
        }
        let mut samples: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let outcome = rt.block_on(backend.dispatch_once());
            let elapsed = t0.elapsed();
            // `black_box` defends against the optimiser hoisting the
            // timing call around the dispatch.
            black_box(outcome.ok());
            black_box(elapsed);
            samples.push(elapsed);
        }
        samples.sort_unstable();
        let max_ns = samples.last().copied().unwrap_or_default().as_nanos();
        Some(TailStats {
            backend: backend.name().to_string(),
            samples: iters,
            p50_ns: percentile_nearest_rank(&samples, 0.50),
            p95_ns: percentile_nearest_rank(&samples, 0.95),
            p99_ns: percentile_nearest_rank(&samples, 0.99),
            p99_9_ns: percentile_nearest_rank(&samples, 0.999),
            max_ns,
        })
    }

    /// Locate `bench-results/`. Mirrors W4.6 `workspace_bench_results_dir`
    /// so both benches behave identically under `cargo bench`.
    fn workspace_bench_results_dir() -> Option<PathBuf> {
        if let Ok(cwd) = std::env::current_dir() {
            let candidate = cwd.join("bench-results");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidate = manifest.join("..").join("..").join("bench-results");
        let canonical = fs::canonicalize(&candidate).ok()?;
        canonical.is_dir().then_some(canonical)
    }

    fn render_file(lines: &[String]) -> String {
        let body = lines
            .iter()
            .map(|l| format!("    {}", l))
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
        concat!(
            "{{\n",
            "  \"// generated\": \"BusyPoll vs cuda-async DispatchFuture comparison (see crates/tensor-wasm-bench/benches/dispatch_future_backends.rs).\",\n",
            "  \"// rfc\": \"rfcs/0001-cuda-oxide-integration.md -- Unresolved question #3\",\n",
            "  \"// algorithm\": \"nearest-rank percentile per ISO/IEC 25062 over {} sorted samples per backend\",\n",
            "  \"backends\": [\n{}\n  ]\n",
            "}}\n",
        ),
        SAMPLES, body,
    )
    }

    /// Run `backend` and append either its measured line or a skip line.
    fn run_backend<B: DispatchFutureBackend>(
        rt: &tokio::runtime::Runtime,
        backend: &B,
        json_lines: &mut Vec<String>,
        skip_reason: &str,
    ) {
        let line = match bench_one(rt, backend, SAMPLES) {
            Some(stats) => stats.to_json(),
            None => skip_json(backend.name(), skip_reason),
        };
        println!("DISPATCH_BACKEND {}", line);
        json_lines.push(line);
    }

    pub fn run() {
        // 2-worker multi-thread runtime — matches `e2e_inference.rs` and the
        // `e2e_rt` in `tail_latency.rs` so scheduling noise lines up.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build multi-thread tokio runtime for dispatch_future_backends");

        let mut json_lines: Vec<String> = Vec::with_capacity(2);
        // BusyPoll now drives a real `cust` event; its skip path fires only
        // when CUDA cannot init/bind at runtime (e.g. no device present even
        // though the binary was built with `--features cuda`).
        run_backend(
            &rt,
            &BusyPollBackend::new(),
            &mut json_lines,
            "busy-poll: CUDA unavailable at runtime (no device / context bind failed)",
        );
        run_backend(
            &rt,
            &CudaAsyncBackend,
            &mut json_lines,
            CUDA_ASYNC_RUNTIME_UNAVAILABLE,
        );

        // File sink: best-effort. Missing dir is not a bench failure — the
        // stdout `DISPATCH_BACKEND` lines are what CI captures regardless.
        if let Some(dir) = workspace_bench_results_dir() {
            let path = dir.join("dispatch-future-backends.json");
            if let Err(e) = fs::write(&path, render_file(&json_lines)) {
                eprintln!(
                    "DISPATCH_BACKEND warn: could not write {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }

    // NOTE: percentile-math and JSON-serialisation tests now live in
    // `crates/tensor-wasm-bench/src/lib.rs`'s `#[cfg(test)] mod tests`, which
    // `cargo test` actually compiles and runs. A `#[cfg(test)] mod tests` here
    // would be dead code: this is a `harness = false` bench target built in the
    // bench profile with `cfg(test)` UNSET, and the module additionally sits
    // behind `#[cfg(feature = "cuda")]`, so it never compiled or ran.
} // end of `mod bench` (gated on `#[cfg(feature = "cuda")]`)

#[cfg(feature = "cuda")]
fn main() {
    bench::run();
}
