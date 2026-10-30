// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! W4.6 tail-latency bench: P99.9 long-tail capture for `dispatch/*` and `e2e/*`.
//!
//! Criterion's default reporter publishes mean, std-dev and median per metric
//! but does **not** expose P99 / P99.9, and its default sample count (~100)
//! is too small to resolve a meaningful P99.9 (a single sample at the tail
//! moves the estimate by 1 ppt of the population). This bench file deliberately
//! sidesteps Criterion's statistical pipeline and runs a hand-rolled sampling
//! loop instead: collect a fixed number of raw `Duration` observations, sort
//! them once, then read the percentile slots back via the **nearest-rank**
//! method (ISO/IEC 25062, also the method `hdrhistogram` documents).
//!
//! ## Sample count: 10 000
//!
//! P99.9 means "99.9 % of samples were faster than this". With 10 000 samples
//! the rank used for P99.9 under the nearest-rank rule is index 9 990 (0-based).
//! That is the 10th-worst observation, which is the smallest sample size that
//! still places the estimate inside the population's true tail rather than at
//! the global maximum. We could go to 100 000 for tighter CIs but the wall
//! time on the dispatch group (~6 µs / sample on a quiet host) would push the
//! bench past 1 s per metric which is still cheap; the 10 000 floor is the
//! anti-cheating-checklist line from `docs/BENCHMARKING.md` ("≥ 30 samples"
//! plus a 333× safety margin for tail resolution).
//!
//! ## Percentile algorithm: nearest-rank
//!
//! Two reasonable choices: nearest-rank (`samples[ceil(p * n) - 1]`) and linear
//! interpolation between the two surrounding ranks (NIST/Excel default).
//! We pick nearest-rank because (a) it always returns an actually-observed
//! value — no interpolation artefacts that don't exist in the real population,
//! (b) it matches what `hdrhistogram` and the Tigerbeetle / Datadog tail-tracking
//! literature use, and (c) on 10 000 samples the two methods differ by at most
//! the gap between two adjacent sorted samples, which on a tight micro-bench
//! is well inside the per-sample noise floor anyway.
//!
//! ## Tracing-overhead concern (W4.1 spans)
//!
//! The W4.1 wave wired OpenTelemetry spans through the axum middleware stack,
//! and those spans run during `e2e/healthz/get` and `e2e/invoke_not_found/post`.
//! A `tracing::Span::current()` lookup + drop is roughly 50-150 ns on a quiet
//! host even with no subscriber attached, which lifts the floor of the e2e
//! metrics but does **not** distort the tail measurement: the span cost is
//! constant per request and shows up identically in P50 and P99.9, so the
//! `p99_9 - p50` *gap* (the actual tail signal) is unchanged. We therefore
//! report the raw measured numbers without subtracting any tracing baseline;
//! the published numbers are end-to-end "what an operator sees" latencies.
//! If a future regression suggests the tail is being driven by tracing
//! sampling rather than real router work, set `TENSOR_WASM_TRACING=off` and
//! re-run this bench — the delta is the tracing tax.
//!
//! ## Backend axis (W4.4 — RFC 0001 Unresolved questions)
//!
//! The bench carries a second dimension beyond workload: the **backend** that
//! the surrounding tensor-wasm-mem layer was compiled against. The label is
//! resolved at compile time from the active feature set and falls into one
//! of three slots:
//!
//! | feature flag (`cargo bench --features ...`) | `BACKEND_LABEL` |
//! |---|---|
//! | (none, default)                             | `"unified-memory"` |
//! | `cudarc-backend`                            | `"cudarc"` |
//! | `cuda-oxide-backend`                        | `"cuda-oxide"` |
//!
//! The label flows into three places: (a) the Criterion benchmark group name
//! (`tail_latency_<backend>`), so split-by-backend report aggregators see a
//! distinct group per run; (b) every emitted `TAIL_LATENCY` JSON line as a
//! `"backend":` field, so CI log scrapers can demultiplex three runs by
//! prefix-grep alone; (c) the `bench-results/tail-latency.json` schema as a
//! top-level `backend` field plus a `backend` field on every metric entry,
//! so the three result files diff cleanly. Picking exactly one of the three
//! feature flags is the operator's responsibility — enabling multiple at
//! once is technically valid for the underlying mem crate but the bench
//! follows a priority order (cuda-oxide > cudarc > unified-memory) and the
//! report file calls out which slot won so the choice is auditable.
//!
//! The hard work of running the bench three times and producing three
//! result files (one per backend) is per-invocation operator work, not
//! something the bench file does itself. The CI matrix wiring is wave-4
//! ops work; see `docs/BENCHMARKING.md#tail-latency` for the manual
//! recipe in the meantime.
//!
//! ## Output
//!
//! For each metric this bench prints **one JSON line** to stdout prefixed by
//! `TAIL_LATENCY` so CI logs can grep for it, then (if the cwd is the
//! workspace root) appends/overwrites `bench-results/tail-latency.json` with
//! the full set of metrics. The file is a separate artefact from
//! `bench-results/baseline.json` and is **not** consumed by the regression
//! gate — it documents the long-tail floor for the v0.3 observability
//! milestone in `docs/PATH-TO-V1.md`.

use std::fs;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use tensor_wasm_api::middleware::{AuthConfig, TenantConfig};
use tensor_wasm_api::rate_limit::{RateLimitConfig, RateLimiter};
use tensor_wasm_api::{build_router_with_full_config, AppState};
use tensor_wasm_bench::{percentile_nearest_rank, TailResult};
use tensor_wasm_wasi_gpu::async_dispatch::{BackPressure, DispatchFuture};
use tower::ServiceExt;

/// Number of raw observations per metric. 10 000 puts the P99.9 sample at
/// rank 9 990 (the tenth-worst), well inside the population tail. See the
/// module docs for the rationale and the alternative (100 000) we
/// deliberately did not pick.
const SAMPLES: usize = 10_000;

/// Warm-up iterations executed before the timed loop. Mirrors Criterion's
/// own warm-up so the first-touch instruction-cache and allocator costs
/// do not pollute the first few samples (which would land in the tail).
const WARMUP: usize = 1_000;

/// Compile-time backend discriminator (W4.4 — RFC 0001 Unresolved questions
/// extension). Resolved by `#[cfg(feature = ...)]` over the bench crate's
/// `cudarc-backend` and `cuda-oxide-backend` feature flags; defaults to
/// `"unified-memory"` (the historical cust-backed path, see the workspace
/// nightly-2026-04-03 alignment story in RFC 0001). The label is used as
/// (1) a Criterion benchmark-group suffix, (2) a `"backend":` field on every
/// stdout JSON line, and (3) a top-level + per-metric field in the rendered
/// `bench-results/tail-latency.json` schema.
///
/// Priority order when multiple backend features are accidentally enabled
/// simultaneously: `cuda-oxide` > `cudarc` > `unified-memory`. cuda-oxide is
/// the v0.5 cust successor (RFC 0001), so an operator who flipped both
/// `cudarc-backend` AND `cuda-oxide-backend` almost certainly meant the
/// latter and the bench reports under that name. The mem crate permits all
/// three features active at once — the priority here is a reporting choice
/// only, not a build-time exclusion. The chosen slot is also logged to
/// stderr at bench startup so the run is unambiguous in CI logs.
///
/// The historical `unified-memory` label is the cust-backed path. Per RFC
/// 0001 Unresolved questions, that label is on a v0.4-deprecation-warning /
/// v0.5-removal track; the constant survives until then for backwards
/// compatibility with the v0.3.x baseline files.
// Single source of truth for the cfg priority (`cuda-oxide` > `cudarc` >
// `unified-memory`): the lib's `resolve_backend_label`, which is unit-tested
// across all combinations. `cfg!(...)` evaluates to a compile-time bool, so
// this stays a `const` selected by the active feature set.
const BACKEND_LABEL: &str = tensor_wasm_bench::backend_label();

// `TailResult` and `percentile_nearest_rank` now live in the crate's
// `src/lib.rs` (imported above) so the percentile math and the JSON schema
// have a single source of truth, plus real `cargo test`-visible unit tests.
// `harness = false` bench targets are compiled with `cfg(test)` UNSET, so any
// `#[cfg(test)] mod tests` here would never compile or run.

/// Collect `SAMPLES` raw durations from `f`, sort once, and compute P50/P95/
/// P99/P99.9/max via `percentile_nearest_rank`. `metric` is the label that
/// goes into the JSON line; it should match a `<group>/<id>` key from
/// `bench-results/baseline.json` so cross-references in the v0.3 docs line up.
fn measure<F>(metric: &str, mut f: F) -> TailResult
where
    F: FnMut(),
{
    // Warm-up — un-timed; un-counted. Mirrors Criterion's behaviour so the
    // first few samples don't dominate the tail with cold-cache costs.
    for _ in 0..WARMUP {
        f();
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed();
        // `black_box` on the elapsed value defends against the compiler
        // hoisting the timing call out of the loop (it can't, today, because
        // `Instant::now` is opaque — but the explicit barrier documents
        // intent and survives future inlining changes).
        black_box(elapsed);
        samples.push(elapsed);
    }

    samples.sort_unstable();
    let max_ns = samples.last().copied().unwrap_or_default().as_nanos();
    TailResult {
        metric: metric.to_string(),
        backend: BACKEND_LABEL,
        samples: SAMPLES,
        p50_ns: percentile_nearest_rank(&samples, 0.50),
        p95_ns: percentile_nearest_rank(&samples, 0.95),
        p99_ns: percentile_nearest_rank(&samples, 0.99),
        p99_9_ns: percentile_nearest_rank(&samples, 0.999),
        max_ns,
    }
}

/// Re-uses the back-pressure semaphore + dispatch-future path that
/// `benches/kernel_dispatch.rs::bench_serial` exercises so the tail numbers
/// can be compared directly against that bench's P50.
fn measure_dispatch_serial(rt: &tokio::runtime::Runtime) -> TailResult {
    let bp = BackPressure::with_cap(1);
    measure("dispatch/serial/100", || {
        rt.block_on(async {
            for _ in 0..100u32 {
                let permit = bp.acquire().await;
                DispatchFuture::ready(permit).await;
            }
        });
    })
}

/// Concurrent-cap-64 mirror of `benches/kernel_dispatch.rs::bench_concurrent`.
/// The runtime is a 4-worker multi-thread runtime to match that bench's
/// configuration — otherwise scheduling-related tail noise would diverge.
///
/// `cap=64` with `n=100` is *lightly* contended (the first 64 tasks acquire
/// immediately, the remaining 36 queue briefly); the dedicated
/// `dispatch/contended_cap1/1000` metric below is the heavily-contended tail
/// reference where the permit-queue cost dominates.
fn measure_dispatch_concurrent(rt: &tokio::runtime::Runtime) -> TailResult {
    let bp = BackPressure::with_cap(64);
    measure("dispatch/concurrent_cap64/100", || {
        rt.block_on(async {
            let mut handles = Vec::with_capacity(100);
            for _ in 0..100u32 {
                let bp = bp.clone();
                handles.push(tokio::spawn(async move {
                    let permit = bp.acquire().await;
                    DispatchFuture::ready(permit).await;
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    })
}

/// Heavily-contended dispatch tail: `cap=1`, `n=1000` so every task but the
/// first must wait on the back-pressure permit. The `p99_9 - p50` gap here is
/// the long-tail cost of permit queueing, which the uncontended cap-64 metric
/// (every task gets a permit immediately) cannot surface.
fn measure_dispatch_contended(rt: &tokio::runtime::Runtime) -> TailResult {
    let bp = BackPressure::with_cap(1);
    measure("dispatch/contended_cap1/1000", || {
        rt.block_on(async {
            let mut handles = Vec::with_capacity(1000);
            for _ in 0..1000u32 {
                let bp = bp.clone();
                handles.push(tokio::spawn(async move {
                    let permit = bp.acquire().await;
                    DispatchFuture::ready(permit).await;
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    })
}

/// Isolates the per-dispatch `tracing::info_span!` construction cost baked
/// into every `DispatchFuture`. A single `acquire` + `ready` + `await` per
/// sample, so the measured floor is span construction + permit RAII + one
/// poll — the span tax that `DispatchFuture::ready` adds to the serial and
/// concurrent numbers above. On a no-CUDA host this is also the entire
/// `ready` cost (the future resolves on first poll); on a CUDA host the
/// `dispatch/real_sync/1` metric below adds the event-query path on top.
fn measure_dispatch_span_only(rt: &tokio::runtime::Runtime) -> TailResult {
    let bp = BackPressure::with_cap(1);
    measure("dispatch/span_only/1", || {
        rt.block_on(async {
            let permit = bp.acquire().await;
            let fut = DispatchFuture::ready(permit);
            black_box(&fut);
            fut.await;
        });
    })
}

/// Real GPU-sync dispatch tail (CUDA only). Records a `cust` event on a fresh
/// stream per sample and drives [`DispatchFuture::with_event`], so the tail
/// captures the actual `event.query()` completion-poll path — the work
/// `DispatchFuture::ready` short-circuits. No kernel is launched; the event
/// is recorded on an empty stream, so this is the real-sync machinery floor,
/// not launch-to-completion of a particular kernel.
#[cfg(feature = "cuda")]
fn measure_dispatch_real_sync(rt: &tokio::runtime::Runtime) -> Option<TailResult> {
    use cust::event::{Event, EventFlags};
    use cust::stream::{Stream, StreamFlags};

    if tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context().is_err() {
        eprintln!("TAIL_LATENCY dispatch/real_sync: skipped (no usable CUDA context at runtime)");
        return None;
    }
    let bp = BackPressure::with_cap(1);
    Some(measure("dispatch/real_sync/1", || {
        rt.block_on(async {
            tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context()
                .expect("bind CUDA context on worker thread");
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("create CUDA stream");
            let event = Event::new(EventFlags::DEFAULT).expect("create CUDA event");
            event.record(&stream).expect("record CUDA event");
            let permit = bp.acquire().await;
            DispatchFuture::with_event(permit, event).await;
        });
    }))
}

fn make_router() -> axum::Router {
    let state = AppState::new();
    let auth = AuthConfig::default();
    let tenant = TenantConfig::default();
    let limiter = RateLimiter::new(RateLimitConfig::disabled());
    build_router_with_full_config(state, auth, tenant, limiter)
}

/// Healthz GET — the cheapest router path; isolates the axum middleware
/// + serde floor from any state-mutating work.
fn measure_healthz(rt: &tokio::runtime::Runtime) -> TailResult {
    let router = make_router();
    measure("e2e/healthz/get", || {
        rt.block_on(async {
            // `oneshot` consumes the service, so the per-sample `router.clone()`
            // is unavoidable. It is a shallow `Arc`-bump of the route table —
            // a constant offset that appears in P50 and the tail alike, so the
            // `p99_9 - p50` tail signal is unaffected (finding 4; same
            // rationale as the tracing-overhead note in the module docs).
            let resp = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/healthz")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            black_box(body);
        });
    })
}

/// Invoke-not-found POST — exercises the lookup-miss branch in the function
/// router. Picked over `create_function/post` because the latter mutates
/// state and would inflate the tail with allocator/registry-growth noise
/// rather than measuring a pure router round-trip.
fn measure_invoke_not_found(rt: &tokio::runtime::Runtime) -> TailResult {
    let router = make_router();
    measure("e2e/invoke_not_found/post", || {
        rt.block_on(async {
            let req = Request::builder()
                .method("POST")
                .uri("/functions/00000000-0000-0000-0000-000000000000/invoke")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap();
            // Constant Arc-bump bias — see `measure_healthz` (finding 4).
            let resp = router.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        });
    })
}

// `workspace_bench_results_dir` and the document renderer now live in
// `tensor_wasm_bench` (src/lib.rs) so both benches share one implementation
// and the path logic + document-level JSON validity are reachable from
// `cargo test`. `render_file` is a thin wrapper that pins this bench's
// `SAMPLES` / `BACKEND_LABEL` into the shared renderer.
fn render_file(results: &[TailResult]) -> String {
    tensor_wasm_bench::render_tail_results_document(results, BACKEND_LABEL, SAMPLES)
}

/// Entry point invoked under Criterion. We register a single dummy
/// benchmark so `cargo bench --bench tail_latency -- --bench` still picks
/// this up; the real work happens up-front, before Criterion's own loop,
/// so the JSON output lands even when Criterion is asked to filter every
/// metric out.
fn tail_latency_bench(c: &mut Criterion) {
    // Two runtimes: the serial bench uses current-thread to match
    // `kernel_dispatch::bench_serial`, the concurrent bench uses a 4-worker
    // multi-thread to match `kernel_dispatch::bench_concurrent`, and the e2e
    // benches share a 2-worker multi-thread to match `e2e_inference::rt`.
    let serial_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build current-thread tokio runtime");
    let concurrent_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build multi-thread tokio runtime for concurrent dispatch");
    let e2e_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build multi-thread tokio runtime for e2e");

    // W4.4: announce the active backend label up front so CI logs that
    // capture stderr line up with the JSON `backend` field. This is
    // strictly informational — the bench behaviour is identical across
    // labels today (the underlying mem-crate selection is wave-4 ops
    // work); the label exists so operators running the bench three times
    // (once per `--features` flag) can demultiplex the three result files
    // without parsing each one.
    eprintln!(
        "TAIL_LATENCY backend={} (compile-time; flip via --features cudarc-backend / cuda-oxide-backend)",
        BACKEND_LABEL,
    );

    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut results = vec![
        measure_dispatch_serial(&serial_rt),
        measure_dispatch_span_only(&serial_rt),
        measure_dispatch_concurrent(&concurrent_rt),
        measure_dispatch_contended(&concurrent_rt),
        measure_healthz(&e2e_rt),
        measure_invoke_not_found(&e2e_rt),
    ];
    // CUDA-only: append the real GPU-sync tail when a device is present.
    // On a no-CUDA host this is compiled out entirely; on a CUDA host that
    // cannot init a context at runtime it logs a skip and appends nothing.
    #[cfg(feature = "cuda")]
    if let Some(real_sync) = measure_dispatch_real_sync(&concurrent_rt) {
        results.push(real_sync);
    }

    // Stdout: one JSON line per metric, grep-prefixed. CI captures these.
    for r in &results {
        println!("TAIL_LATENCY {}", r.to_json());
    }

    // File sink: best-effort. A missing directory is not a bench failure.
    if let Some(dir) = tensor_wasm_bench::workspace_bench_results_dir() {
        let path = dir.join("tail-latency.json");
        let body = render_file(&results);
        if let Err(e) = fs::write(&path, body) {
            eprintln!(
                "TAIL_LATENCY warn: could not write {}: {}",
                path.display(),
                e
            );
        }
    }

    // Register a single no-op group so Criterion has *something* to run and
    // doesn't emit "no benches matched". `measurement_time` is dropped to
    // the minimum since the real work is already done above. The group name
    // carries the W4.4 backend label as a suffix so split-by-group report
    // aggregators (`target/criterion/tail_latency_<backend>/`) see a
    // distinct directory per backend — three sequential `cargo bench` runs
    // (default / `--features cudarc-backend` / `--features cuda-oxide-backend`)
    // produce three sibling criterion directories that diff cleanly.
    let group_name = format!("tail_latency_{}", BACKEND_LABEL);
    let mut group = c.benchmark_group(group_name);
    // The real measurement already ran above; this marker exists only so
    // Criterion has a benchmark to "run" and doesn't print "no benches
    // matched". Drive its cost to the floor: 10 is Criterion's minimum
    // `sample_size`, and the warm-up + measurement windows are pushed to the
    // smallest values Criterion accepts so the marker adds well under a
    // second of wall-clock instead of the previous ~0.5 s + warm-up
    // (finding 6).
    group.measurement_time(Duration::from_millis(50));
    group.warm_up_time(Duration::from_millis(50));
    group.sample_size(10);
    group.bench_function("noop_marker", |b| {
        b.iter(|| black_box(0u64));
    });
    group.finish();
}

// NOTE: percentile-math, JSON-serialisation, and backend-label tests now live
// in `crates/tensor-wasm-bench/src/lib.rs`'s `#[cfg(test)] mod tests`, which
// `cargo test` actually compiles and runs. A `#[cfg(test)] mod tests` here
// would be dead code: `harness = false` bench targets are built in the bench
// profile with `cfg(test)` UNSET.

criterion_group!(benches, tail_latency_bench);
criterion_main!(benches);
