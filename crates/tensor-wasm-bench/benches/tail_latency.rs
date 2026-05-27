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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use base64::Engine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use tensor_wasm_api::{build_router, AppState};
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
#[cfg(feature = "cuda-oxide-backend")]
const BACKEND_LABEL: &str = "cuda-oxide";
#[cfg(all(feature = "cudarc-backend", not(feature = "cuda-oxide-backend")))]
const BACKEND_LABEL: &str = "cudarc";
#[cfg(all(not(feature = "cudarc-backend"), not(feature = "cuda-oxide-backend")))]
const BACKEND_LABEL: &str = "unified-memory";

/// One line of the structured JSON output. Field names match the format
/// documented in `bench-results/README.md` so downstream parsers (Grafana
/// import scripts, the v0.3 observability dashboards) can consume it
/// without translation. The `backend` field was added in W4.4 (RFC 0001
/// Unresolved questions extension) and is the compile-time
/// [`BACKEND_LABEL`] of the run; it is always present today even though the
/// schema is still flagged "schemaless" upstream (wave-4 ops work
/// formalises the schema). Downstream parsers that pre-date W4.4 should
/// treat `backend` as optional and fall back to `"unified-memory"` when it
/// is missing — that matches the W4.6 historical default.
#[derive(Debug, Clone)]
struct TailResult {
    metric: String,
    backend: &'static str,
    samples: usize,
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    p99_9_ns: u128,
    max_ns: u128,
}

impl TailResult {
    fn to_json(&self) -> String {
        format!(
            "{{\"metric\":\"{}\",\"backend\":\"{}\",\"samples\":{},\"p50_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"p99_9_ns\":{},\"max_ns\":{}}}",
            self.metric,
            self.backend,
            self.samples,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.p99_9_ns,
            self.max_ns,
        )
    }
}

/// Nearest-rank percentile (`samples[ceil(p * n) - 1]`). `samples` must be
/// pre-sorted ascending; the function panics on an empty slice to surface
/// bench misconfiguration loudly rather than silently emitting 0 ns.
fn percentile_nearest_rank(sorted: &[Duration], p: f64) -> u128 {
    assert!(!sorted.is_empty(), "percentile of empty sample set");
    assert!((0.0..=1.0).contains(&p), "percentile p out of range");
    let n = sorted.len();
    let rank = ((p * n as f64).ceil() as usize).max(1);
    let idx = rank.min(n) - 1;
    sorted[idx].as_nanos()
}

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

fn make_router() -> axum::Router {
    build_router(Arc::new(AppState::default()))
}

/// Healthz GET — the cheapest router path; isolates the axum middleware
/// + serde floor from any state-mutating work.
fn measure_healthz(rt: &tokio::runtime::Runtime) -> TailResult {
    let router = make_router();
    measure("e2e/healthz/get", || {
        rt.block_on(async {
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
            let resp = router.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        });
    })
}

/// Locate `bench-results/` relative to either the current working directory
/// or the bench crate's `CARGO_MANIFEST_DIR`. Returns `None` if neither
/// candidate resolves — in that case we skip the file write and rely on the
/// stdout JSON lines (which CI captures regardless).
fn workspace_bench_results_dir() -> Option<PathBuf> {
    // First try cwd / bench-results — true when invoked from the workspace
    // root (`cargo bench -p tensor-wasm-bench --bench tail_latency`).
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("bench-results");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    // Fall back to manifest-relative — true when `cargo bench` is launched
    // from inside the crate directory and changes cwd accordingly.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.join("..").join("..").join("bench-results");
    let canonical = fs::canonicalize(&candidate).ok()?;
    if canonical.is_dir() {
        Some(canonical)
    } else {
        None
    }
}

/// Pretty-print the full result set as a stable JSON document so a reviewer
/// can `diff` two runs by eye. The schema is documented in
/// `bench-results/README.md`.
///
/// The W4.4 `backend` field is rendered both as a top-level discriminator
/// (so a reader sees which compile-time slot the run targeted without
/// scanning every metric) and as a per-metric field on each entry (so a
/// downstream parser that joins many files by `<metric, backend>` doesn't
/// need to reach back into the document root). The redundant per-metric
/// `backend` field is intentional — wave-4 ops work formalises the schema
/// and may keep both fields, or may demote the per-metric one once the
/// regression-gate Python parser is updated.
fn render_file(results: &[TailResult]) -> String {
    let metrics = results
        .iter()
        .map(|r| format!("    {}", r.to_json()))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        concat!(
            "{{\n",
            "  \"// generated\": \"P99.9 tail-latency snapshot (see crates/tensor-wasm-bench/benches/tail_latency.rs).\",\n",
            "  \"// algorithm\": \"nearest-rank percentile per ISO/IEC 25062 over {} sorted samples per metric\",\n",
            "  \"// backend-axis\": \"W4.4 RFC 0001 extension -- run the bench once per backend (default / --features cudarc-backend / --features cuda-oxide-backend) and diff the three files; see docs/BENCHMARKING.md for the operator recipe\",\n",
            "  \"backend\": \"{}\",\n",
            "  \"metrics\": [\n{}\n  ]\n",
            "}}\n",
        ),
        SAMPLES, BACKEND_LABEL, metrics,
    )
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

    let results = vec![
        measure_dispatch_serial(&serial_rt),
        measure_dispatch_concurrent(&concurrent_rt),
        measure_healthz(&e2e_rt),
        measure_invoke_not_found(&e2e_rt),
    ];

    // Stdout: one JSON line per metric, grep-prefixed. CI captures these.
    for r in &results {
        println!("TAIL_LATENCY {}", r.to_json());
    }

    // File sink: best-effort. A missing directory is not a bench failure.
    if let Some(dir) = workspace_bench_results_dir() {
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
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(10);
    group.bench_function("noop_marker", |b| {
        b.iter(|| black_box(0u64));
    });
    group.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_known_distribution() {
        // 1..=100 sorted: P50 = 50, P95 = 95, P99 = 99, P99.9 = 100, max = 100.
        // Nearest-rank: rank(0.50, 100) = ceil(50) = 50 → samples[49] = 50.
        let samples: Vec<Duration> = (1u128..=100).map(|n| Duration::from_nanos(n as u64)).collect();
        assert_eq!(percentile_nearest_rank(&samples, 0.50), 50);
        assert_eq!(percentile_nearest_rank(&samples, 0.95), 95);
        assert_eq!(percentile_nearest_rank(&samples, 0.99), 99);
        // P99.9 of 100 samples: ceil(99.9) = 100 → samples[99] = 100.
        assert_eq!(percentile_nearest_rank(&samples, 0.999), 100);
    }

    #[test]
    fn nearest_rank_p0_returns_min() {
        let samples = vec![Duration::from_nanos(7), Duration::from_nanos(13)];
        // p=0.0 clamps the rank to 1 → samples[0].
        assert_eq!(percentile_nearest_rank(&samples, 0.0), 7);
    }

    /// W4.4 sanity: the compile-time backend label must be one of the three
    /// slots RFC 0001 enumerates. Catches typos in any future feature-flag
    /// rename (e.g. if `cuda-oxide-backend` is shortened to `cuoxide`).
    #[test]
    fn backend_label_is_one_of_three_known_values() {
        assert!(
            matches!(BACKEND_LABEL, "unified-memory" | "cudarc" | "cuda-oxide"),
            "unexpected BACKEND_LABEL={BACKEND_LABEL:?}; expected one of unified-memory/cudarc/cuda-oxide per RFC 0001",
        );
    }

    /// W4.4 JSON-serialisation contract: `to_json` must emit the backend
    /// field so downstream parsers can demultiplex by backend. Guards
    /// against a future refactor accidentally dropping the field.
    #[test]
    fn tail_result_json_carries_backend_field() {
        let r = TailResult {
            metric: "test/metric".to_string(),
            backend: "unified-memory",
            samples: 10,
            p50_ns: 1, p95_ns: 2, p99_ns: 3, p99_9_ns: 4, max_ns: 5,
        };
        let s = r.to_json();
        assert!(s.contains("\"backend\":\"unified-memory\""), "missing backend field: {s}");
        assert!(s.contains("\"metric\":\"test/metric\""), "missing metric field: {s}");
    }
}

criterion_group!(benches, tail_latency_bench);
criterion_main!(benches);
