// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Concurrent-load regression test for W4.1 trace-id propagation through
//! the async-invoke spawn boundary.
//!
//! ## Why this test exists (audit Problem #11)
//!
//! The v0.3.2 audit flagged a latent risk in the W4.1 wiring around
//! [`tensor_wasm_api::routes::invoke_function_async`]: the handler opens
//! an `http.invoke_function_async` span via `#[tracing::instrument]`,
//! then manually constructs an `async_invoke.job` span and wraps the
//! `tokio::spawn` future with `tracing::Instrument::instrument(...)` so
//! the trace id survives the spawn boundary. Two interactions were
//! un-load-tested:
//!
//! 1. Could the `Instrument` wrap *plus* the per-request span the
//!    `trace_layer_with_propagation` opens for every inbound request
//!    end up double-counting the inner `invoke.run` span under
//!    concurrent contention (e.g. a span emitted both from the handler
//!    stack frame and again from the resumed spawn future)?
//! 2. Could a span silently *drop* under load — `#[instrument]` is a
//!    macro expansion; if the active subscriber is mid-mutation when a
//!    new span is constructed, would the callback fire?
//!
//! Both failure modes are subtle: integration tests that exercise the
//! happy path with one or two requests will never reproduce them. We
//! need real concurrent polling on a multi-thread runtime, a custom
//! `Layer` that *counts* span creation per name, and an assertion that
//! the totals line up exactly with the number of requests we issued.
//!
//! ## Expected span count derivation
//!
//! Each `POST /functions/{id}/invoke-async` request, when traced through
//! the production router, produces exactly four named spans we care
//! about:
//!
//! | # | Span name                       | Source                                              |
//! |---|---------------------------------|-----------------------------------------------------|
//! | 1 | `http.request`                  | `middleware::trace_layer_with_propagation` make_span |
//! | 2 | `http.invoke_function_async`    | `#[instrument]` on `routes::invoke_function_async`   |
//! | 3 | `async_invoke.job`              | `info_span!` opened around the spawn future          |
//! | 4 | `invoke.run`                    | `#[instrument]` on `routes::run_invoke`              |
//!
//! So for `N` concurrent async-invoke requests, the [`CountingLayer`]
//! must observe exactly `4 * N` `on_new_span` callbacks across those
//! four names — no more, no less.
//!
//! ## What a passing test guarantees
//!
//! * The `Instrument`-wrapped `tokio::spawn` future does **not**
//!   double-instrument: the `async_invoke.job` and `invoke.run` spans
//!   each fire exactly once per request, not twice.
//! * No `#[instrument]` macro silently drops under contention.
//! * Every request's `x-trace-id` response header is unique (no
//!   collisions across N parallel root contexts).
//! * Every observed span has a parent that is either (a) absent (root
//!   spans) or (b) one of the spans we recorded — no orphans with a
//!   dangling parent id.
//!
//! ## What a failing test would tell you
//!
//! * `count > 4 * N` for some name: the async-invoke spawn picked up an
//!   extra `Instrument` wrap (likely from a future merge that added a
//!   second `.instrument(span)` call) and is double-counting.
//! * `count < 4 * N` for some name: a `#[instrument]` macro is being
//!   skipped under contention — likely a layer ordering or
//!   `try_init`-race bug. Check that the subscriber is installed *once*
//!   before any router is constructed.
//! * Duplicate trace ids: the `install_w3c_propagator` global is being
//!   re-seeded between requests, or the per-request span is sharing a
//!   parent context across tasks. Either way, distributed tracing is
//!   broken.
//! * Orphan parent id: a span recorded a parent that was never opened
//!   in this test run, which means the `Instrument::instrument` wrap
//!   on the spawn future is parenting the inner span at a different
//!   context than the one the handler observed — i.e. trace stitching
//!   is broken across the spawn boundary.

#![allow(clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures::future::join_all;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, HEADER_TRACE_ID,
};
use tower::ServiceExt;
use tracing::span;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// Number of concurrent async-invoke requests fired by the test. Sized
/// so each of the 4 `worker_threads` polls ~16 in-flight requests
/// simultaneously — high enough to hit the concurrent-resume paths that
/// the audit was worried about, low enough to keep the test fast and
/// deterministic on CI (single-digit seconds wall clock).
const N_REQUESTS: usize = 64;

/// Exact number of named spans we expect per async-invoke request.
/// See the module-level table for the derivation.
const SPANS_PER_REQUEST: usize = 4;

/// Span names we count. Other spans (tower-http internals, `_start`
/// vs `main` fallback events, executor / WASI-GPU children) are not
/// asserted — this test exists to lock down the four W4.1 hops
/// specifically, not to freeze the entire span schema (the schema test
/// lives elsewhere in `docs/OBSERVABILITY.md`'s reference list).
const COUNTED_SPAN_NAMES: &[&str] = &[
    "http.request",
    "http.invoke_function_async",
    "async_invoke.job",
    "invoke.run",
];

/// Captured snapshot of a single span observation: its name, its own
/// `Id` (as `u64`), and its parent's id if any. Stored in the shared
/// `Mutex<Vec<Observed>>` by the [`CountingLayer`] on each
/// `on_new_span` callback.
#[derive(Debug, Clone)]
struct Observed {
    name: &'static str,
    id: u64,
    parent_id: Option<u64>,
}

/// A `tracing_subscriber::Layer` that records every `on_new_span`
/// callback into a shared `Vec`. We deliberately do **not** filter on
/// the name inside the layer — counting everything and filtering in
/// the assertion keeps the layer simple and makes "saw a span I didn't
/// expect" a thing we can surface in a debug print rather than
/// silently swallow.
#[derive(Clone)]
struct CountingLayer {
    observed: Arc<Mutex<Vec<Observed>>>,
}

impl<S> Layer<S> for CountingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        // Translate the span name into a `&'static str` from our known
        // list — that lets us store it without an allocation per
        // observation, which would otherwise add noise under load.
        let name = attrs.metadata().name();
        let static_name = COUNTED_SPAN_NAMES
            .iter()
            .copied()
            .find(|n| *n == name)
            .unwrap_or("__other__");

        // Parent id resolution: use `lookup_current` to grab whichever
        // span was active when this one was created. This matches the
        // pattern in `tensor-wasm-exec/tests/spans_emitted.rs` and is
        // the same lookup `tracing-opentelemetry` does under the hood,
        // so our orphan check below matches what an OTel exporter
        // would see. `Attributes::parent()` would only catch explicit
        // `parent:` directives, which `#[instrument]` does not use.
        let parent_id = ctx.lookup_current().map(|s| s.id().into_u64());

        let obs = Observed {
            name: static_name,
            id: id.into_u64(),
            parent_id,
        };
        if let Ok(mut guard) = self.observed.lock() {
            guard.push(obs);
        }
    }
}

/// Install the counting subscriber + a minimal `tracing-opentelemetry`
/// layer globally, exactly once. Multiple `#[tokio::test]`s in the same
/// process would otherwise race on `set_global_default`; we guard with
/// `Once` so the second caller becomes a no-op rather than panicking,
/// matching the pattern in
/// `trace_propagation::install_w3c_propagator`.
///
/// The OTel layer is required for the `x-trace-id` response header to
/// be populated — `inject_trace_id_header` reads the active span's
/// OTel `SpanContext::trace_id()`, which is only non-zero when an
/// `OpenTelemetrySpanExt`-bridged tracer is attached. We install an
/// `SdkTracerProvider` with no batch exporter so the OTel-layer side
/// effects are a pure in-memory id-assignment dance with no network I/O.
fn install_counting_subscriber(observed: Arc<Mutex<Vec<Observed>>>) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Minimal in-memory tracer: assigns trace ids to spans but
        // does not export anywhere. Sufficient to make
        // `current_trace_id()` return populated hex strings.
        use opentelemetry::trace::TracerProvider as _;
        let provider = opentelemetry_sdk::trace::TracerProvider::builder().build();
        let tracer = provider.tracer("tensor-wasm-api-trace-concurrent-load-test");
        opentelemetry::global::set_tracer_provider(provider);

        let counting_layer = CountingLayer { observed };
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::registry()
            .with(otel_layer)
            .with(counting_layer);
        // `try_init` rather than `init` so a sibling test that already
        // initialised a subscriber (none today, but cheap insurance)
        // doesn't panic this one.
        let _ = subscriber.try_init();
    });
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router_with_state() -> (axum::Router, Arc<AppState>) {
    let state = Arc::new(AppState::default());
    let router = build_router_with_config(
        Arc::clone(&state),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    (router, state)
}

async fn deploy_noop_module(router: &axum::Router) -> String {
    let wat = r#"(module (func (export "_start")))"#;
    let wasm_bytes = wat::parse_str(wat).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "trace-load", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

/// Fire one async-invoke request and return its response so we can
/// snapshot the `x-trace-id` header. Caller is responsible for
/// awaiting all N futures in parallel via `join_all`.
async fn fire_invoke_async(router: axum::Router, function_id: String) -> axum::response::Response {
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{function_id}/invoke-async"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    router.oneshot(req).await.expect("invoke-async")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_invoke_async_emits_exact_span_count_no_orphans_no_duplicate_trace_ids() {
    let observed: Arc<Mutex<Vec<Observed>>> = Arc::new(Mutex::new(Vec::new()));
    install_counting_subscriber(Arc::clone(&observed));

    // Snapshot the "spans seen before we started the load" baseline so
    // any concurrent tests in the same process (or the deploy hop
    // below) don't pollute the count we assert on. We diff against
    // this baseline at the end.
    let baseline_len = observed.lock().expect("observed lock").len();

    let (router, state) = router_with_state();
    let function_id = deploy_noop_module(&router).await;

    // After deploy, take a fresh snapshot — everything observed from
    // here on is the load we are testing.
    let pre_load_len = observed.lock().expect("observed lock").len();

    // Fire N concurrent async-invoke requests. We clone the router
    // per task (cheap — internally an `Arc`-routed tower service).
    let futures = (0..N_REQUESTS)
        .map(|_| fire_invoke_async(router.clone(), function_id.clone()))
        .collect::<Vec<_>>();
    let responses = join_all(futures).await;

    // All requests must have returned 202 Accepted; otherwise the
    // counts below are meaningless.
    assert_eq!(
        responses.len(),
        N_REQUESTS,
        "expected {N_REQUESTS} responses, got {}",
        responses.len()
    );
    for (i, resp) in responses.iter().enumerate() {
        assert_eq!(
            resp.status(),
            StatusCode::ACCEPTED,
            "request #{i} did not return 202 Accepted (got {})",
            resp.status()
        );
    }

    // Snapshot the trace-id headers BEFORE awaiting the spawned tasks
    // — the response header is set by `inject_trace_id_header` on the
    // way out, so it's already present on every response we just
    // collected. If no `tracing_opentelemetry` layer is installed the
    // header is omitted entirely (documented behaviour); we tolerate
    // that and only assert *uniqueness* among the headers that *do*
    // appear, so this test stays independent of whether a sibling
    // test installed an OTel layer.
    let mut trace_ids: Vec<String> = Vec::new();
    for resp in &responses {
        if let Some(v) = resp.headers().get(HEADER_TRACE_ID) {
            let s = v.to_str().expect("trace-id is ASCII").to_owned();
            // Guard against the all-zero sentinel — the
            // inject_trace_id_header middleware should suppress it,
            // but failing loud here surfaces any regression there.
            assert!(
                !s.chars().all(|c| c == '0'),
                "x-trace-id leaked the all-zero sentinel under load"
            );
            trace_ids.push(s);
        }
    }
    let unique_trace_ids: HashSet<&String> = trace_ids.iter().collect();
    assert_eq!(
        unique_trace_ids.len(),
        trace_ids.len(),
        "x-trace-id collision under concurrent load: {} responses, {} unique ids",
        trace_ids.len(),
        unique_trace_ids.len(),
    );

    // Now wait for the spawned `async_invoke.job` tasks to drain.
    // We use the same convergence check the C3 jobs_active test
    // uses: poll the gauge until it returns to zero. That's a
    // strong signal that every `.dec()` (which runs at the *end*
    // of every spawned task, after `invoke.run` has closed) has
    // fired — and therefore every span on the spawned side has
    // been observed by our layer.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if state.metrics.jobs_active().get() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("jobs_active returns to zero within 10s of firing N invokes");

    // Give the runtime one extra yield so any layer callbacks queued
    // after `dec` (in particular `on_close` for the just-closed
    // `invoke.run` span) flush into our Vec. `on_new_span` fires
    // *before* the span enters scope so it should already be in the
    // Vec, but the yield is cheap insurance.
    tokio::task::yield_now().await;

    // ---------------------------------------------------------------
    // Assertions
    // ---------------------------------------------------------------
    let snapshot: Vec<Observed> = observed.lock().expect("observed lock").clone();
    let load_obs: Vec<&Observed> = snapshot.iter().skip(pre_load_len).collect();

    // Per-name counts. We assert EXACTLY N for each of the four W4.1
    // span names. A failure here is exactly the audit's worry — see
    // the module doc for what each direction (too many / too few)
    // tells you.
    let mut per_name: HashMap<&'static str, usize> = HashMap::new();
    for obs in &load_obs {
        *per_name.entry(obs.name).or_default() += 1;
    }
    for &name in COUNTED_SPAN_NAMES {
        let got = per_name.get(name).copied().unwrap_or(0);
        assert_eq!(
            got, N_REQUESTS,
            "span `{name}` fired {got} times for {N_REQUESTS} concurrent requests \
             (expected exactly {N_REQUESTS}); \
             count > N => double-instrument (likely a duplicate `.instrument()` wrap), \
             count < N => `#[instrument]` dropped under contention",
        );
    }

    // Total counted spans across the four named hops = 4 * N. We
    // also assert this aggregate so a regression that shifts spans
    // between names (e.g. renames `invoke.run` and re-adds an old
    // `invoke.run` macro elsewhere) still trips the test.
    let counted_total: usize = COUNTED_SPAN_NAMES
        .iter()
        .map(|n| per_name.get(n).copied().unwrap_or(0))
        .sum();
    assert_eq!(
        counted_total,
        SPANS_PER_REQUEST * N_REQUESTS,
        "total W4.1 spans = {counted_total}, expected {SPANS_PER_REQUEST}*N = {}",
        SPANS_PER_REQUEST * N_REQUESTS,
    );

    // Orphan-parent check: every parent id recorded on a counted
    // span must correspond to a span this subscriber *also*
    // observed — either earlier in the load window, or in the
    // baseline (the per-request `http.request` may be parented to
    // a tower-http span we already saw before the load started).
    // Spans with no parent (roots) are fine.
    let all_observed_ids: HashSet<u64> = snapshot.iter().map(|o| o.id).collect();
    let mut orphans: Vec<&Observed> = Vec::new();
    for obs in &load_obs {
        if let Some(pid) = obs.parent_id {
            if !all_observed_ids.contains(&pid) {
                orphans.push(obs);
            }
        }
    }
    assert!(
        orphans.is_empty(),
        "{} span(s) recorded a parent id we never observed — \
         the `Instrument` wrap on `tokio::spawn` is parenting at a \
         different context than the handler observed; W4.1 trace \
         stitching is broken across the spawn boundary. First few: {:?}",
        orphans.len(),
        orphans.iter().take(5).collect::<Vec<_>>(),
    );

    // Sanity: the load window actually produced *some* observations
    // (defends against a future refactor that wires the subscriber
    // up wrong and silently observes nothing — which would make
    // every per-name count come out to zero AND would equal
    // `N_REQUESTS * 0`, so the per-name asserts above would still
    // fire, but a defensive bound never hurts).
    assert!(
        snapshot.len() > baseline_len,
        "counting subscriber observed nothing across the load window \
         (baseline={baseline_len}, after={})",
        snapshot.len(),
    );
}
