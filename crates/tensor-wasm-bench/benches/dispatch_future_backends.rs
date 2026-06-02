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
//! # What runs today
//!
//! - `BusyPollBackend` — the existing
//!   `tensor_wasm_wasi_gpu::async_dispatch::DispatchFuture::ready` path
//!   (W4.1-instrumented + post-CUDA-debug `EventStatus::NotReady ->
//!   wake_by_ref` busy-poll). On a CUDA host with `--features cuda` this
//!   measures real GPU sync; on a no-CUDA host it degenerates to the
//!   scheduling-overhead floor (same as `kernel_dispatch::bench_serial`).
//! - `CudaAsyncBackend` — a documented **stub**. Returns
//!   `Err("cuda-async-backend: not yet wired -- see RFC 0001 v0.4 port")`;
//!   bench loop emits a `"status":"skipped"` JSON line. Mirrors the
//!   `tensor_wasm_mem::cuda_oxide_backend` scaffold pattern so the v0.4
//!   port is a body-only diff.
//!
//! # v0.4 wiring contract
//!
//! When the v0.4 cuda-oxide port lands, `CudaAsyncBackend` must (1)
//! replace `dispatch_once` with a `cuda_async::Stream::synchronize`-
//! equivalent await holding a `BackPressure` permit for the same RAII
//! window; (2) drop the early-return `Err`; (3) keep `name() ->
//! "cuda-async"` unchanged so downstream consumers don't need a rename;
//! (4) delete the `TODO(v0.4)` markers.
//!
//! # Honest disclosure
//!
//! - CudaAsync emits no numbers today. Anyone quoting a "BusyPoll vs
//!   CudaAsync" speedup before v0.4 lands is quoting nothing.
//! - On a no-CUDA host BusyPoll numbers are the scheduling floor, not
//!   launch-to-completion latency. Same caveat as `kernel_dispatch.rs`.
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
        "DISPATCH_BACKEND {{\"backend\":\"cuda-async\",\"status\":\"skipped\",\"reason\":\"cuda-async-backend: not yet wired -- see RFC 0001 v0.4 port\"}}"
    );
}

#[cfg(feature = "cuda")]
mod bench {

    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use criterion::black_box;
    use tensor_wasm_bench::{percentile_nearest_rank, TailStats};
    use tensor_wasm_wasi_gpu::async_dispatch::{BackPressure, DispatchFuture};

    /// Raw observations per backend. Matches the W4.6 floor in
    /// `tail_latency.rs` so the P99.9 sample lands at rank 9 990.
    const SAMPLES: usize = 10_000;

    /// Warm-up iterations before the timed loop, mirroring W4.6.
    const WARMUP: usize = 1_000;

    /// Sentinel returned by every stub `dispatch_once`. Mirrors the
    /// `tensor_wasm_mem::cuda_oxide_backend::NOT_YET_WIRED` string shape so
    /// a grep for "not yet wired" surfaces both scaffold sites.
    const CUDA_ASYNC_NOT_YET_WIRED: &str =
        "cuda-async-backend: not yet wired -- see RFC 0001 v0.4 port";

    /// Backend-agnostic shape for one no-op dispatch + permit RAII window.
    /// `Err(&'static str)` signals "skip this backend, the impl is a stub";
    /// the bench loop converts it into a documented skip line rather than
    /// counting it as latency.
    trait DispatchFutureBackend {
        #[allow(async_fn_in_trait)] // bench-only trait, no external impls
        async fn dispatch_once(&self) -> Result<(), &'static str>;
        /// Stable JSON label. Must NOT change across releases — the v0.4
        /// wiring contract documents that `cuda-async` stays `cuda-async`.
        fn name(&self) -> &'static str;
    }

    /// Current behaviour: mirrors `kernel_dispatch::bench_serial` — cap=1
    /// back-pressure, `DispatchFuture::ready(permit).await`, drop. The
    /// `BackPressure` is hoisted out of `dispatch_once` so we measure permit
    /// acquire/release + future poll only, not semaphore construction.
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
            let permit = self.back_pressure.acquire().await;
            // Same constructor as `kernel_dispatch.rs` + `tail_latency.rs`,
            // so numbers line up against those baselines. On a CUDA host
            // with `Event` plumbed, this resolves via the
            // `EventStatus::NotReady -> wake_by_ref` busy-poll — the path
            // cuda-async is meant to replace.
            DispatchFuture::ready(permit).await;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "busy-poll"
        }
    }

    /// **Scaffold stub.** Always returns the sentinel `Err`. Carries no
    /// `cuda-async` dep import so v0.3.x builds don't need the
    /// nightly-2026-04-03 override; v0.4 reviewers flipping both
    /// `--features cuda` and `--features cuda-oxide-backend` get a clean
    /// body-only diff. TODO(v0.4): wire per module docs' four-point contract.
    struct CudaAsyncBackend;

    impl DispatchFutureBackend for CudaAsyncBackend {
        async fn dispatch_once(&self) -> Result<(), &'static str> {
            // TODO(v0.4): replace with `cuda_async::Stream::synchronize`-equivalent.
            Err(CUDA_ASYNC_NOT_YET_WIRED)
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

    /// Skip line for stub backends.
    fn skip_json(backend: &str, reason: &str) -> String {
        format!(
            "{{\"backend\":\"{}\",\"status\":\"skipped\",\"reason\":\"{}\"}}",
            tensor_wasm_bench::json_escape(backend),
            tensor_wasm_bench::json_escape(reason),
        )
    }

    /// Run `iters` samples through `backend`. Returns `None` for stub
    /// backends (first call returns sentinel `Err`); caller emits skip line.
    // `async fn` in a trait isn't dyn-compatible (no vtable for the
    // returned `impl Future`), so the runner is generic over a concrete
    // `B: DispatchFutureBackend` instead of `&dyn`. Each backend gets its
    // own monomorphisation — fine because there are only two backends.
    fn bench_one<B: DispatchFutureBackend>(
        rt: &tokio::runtime::Runtime,
        backend: &B,
        iters: usize,
    ) -> Option<TailStats> {
        // Probe before warm-up so a stub short-circuits without burning
        // 11 000 iterations. Same shape as the timed loop.
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
        // BusyPoll has no sentinel path; its "unexpected stub" reason only
        // fires if the trait contract drifted. CudaAsync is the documented
        // stub today; v0.4 wiring flips it to a measured line.
        run_backend(
            &rt,
            &BusyPollBackend::new(),
            &mut json_lines,
            "unexpected stub return -- check trait contract",
        );
        run_backend(
            &rt,
            &CudaAsyncBackend,
            &mut json_lines,
            CUDA_ASYNC_NOT_YET_WIRED,
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
