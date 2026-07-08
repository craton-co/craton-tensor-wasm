// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for T19 perf: `host_validate` must reject a forbidden
//! `Host` request before the trace layer allocates a span for it.
//!
//! ## Why this matters
//!
//! `host_validate` is a cheap string-compare against the
//! `TENSOR_WASM_API_TRUSTED_HOSTS` allowlist. The trace layer, by
//! contrast, opens an `http.request` span on every inbound request,
//! parses the W3C `traceparent` header through the propagator, and (in
//! a production deployment with a real OTel subscriber) drives a
//! downstream collector hop. Under a hostile Host-header probe storm,
//! the original layer order ran the trace layer FIRST and rejected
//! the request only afterwards — burning trace budget on probes the
//! gateway was about to throw out.
//!
//! The T19 fix re-orders the `common_layers` stack so the
//! `TrustedHosts` extension and `host_validate` middleware sit
//! OUTERMOST, ahead of `trace_layer_with_propagation`. The runtime
//! middleware logic itself is unchanged; only the layer order moved.
//!
//! ## What this test asserts
//!
//! 1. **Hostile-Host shape.** A `GET /healthz` carrying a `Host`
//!    header NOT in the configured allowlist returns `400` (the
//!    documented `host_validate` rejection envelope).
//! 2. **No span allocated.** With a `tracing_subscriber` `Layer`
//!    counting `on_new_span` events for spans named `http.request`,
//!    no such span is recorded for the rejected request. This is the
//!    load-bearing proof that the layer order moved as documented:
//!    if `host_validate` ran INSIDE the trace layer (the pre-T19
//!    order), the span would still be opened before the rejection
//!    short-circuited the chain.
//! 3. **Positive control.** A subsequent allowlisted request to the
//!    same router DOES produce an `http.request` span. This rules
//!    out the trivially-passing case where the capturing layer is
//!    simply broken or another test in the binary has claimed the
//!    global subscriber.
//!
//! ## Why a custom layer instead of `tracing_test`
//!
//! Same rationale as `host_validate_startup_warn.rs`: `tracing-test`
//! is not in the workspace dev-deps, and `tracing-subscriber` is
//! already a dev-dep for two other tests in this crate. Re-using it
//! keeps the dep graph lean and matches the established pattern for
//! "capture events into a Vec for assertion" tests.

#![allow(clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, ENV_TRUSTED_HOSTS,
};
use tower::ServiceExt;
use tracing::span;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Span name the production trace layer opens for every inbound
/// request — see `crate::middleware::trace_layer_with_propagation`.
/// Pinned as a constant so a future refactor that renames the span
/// surfaces here as a compile-time-stable failure rather than a
/// silent "saw zero spans" pass.
const HTTP_REQUEST_SPAN: &str = "http.request";

/// `tracing_subscriber::Layer` that counts how many `on_new_span`
/// callbacks fire for the documented `http.request` span. The counter is
/// per-test (owned by the thread-local subscriber installed below), so two
/// tests running in parallel never share a count.
struct CountingLayer {
    http_request_spans: Arc<AtomicUsize>,
}

impl<S> Layer<S> for CountingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &span::Id, _ctx: Context<'_, S>) {
        if attrs.metadata().name() == HTTP_REQUEST_SPAN {
            self.http_request_spans.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Install the counting subscriber as the **thread-local** default for the
/// duration of the returned guard, and hand back the span counter.
///
/// We use [`tracing::subscriber::set_default`] (thread-local) rather than
/// the process-global `set_global_default` / `try_init`. The global path is
/// set-once-per-process: under the shared test harness only the first test
/// to install wins, and every other test's counter stays at zero — which
/// used to trigger a vacuous "soft skip" (the `after_allow == positive_baseline`
/// branch) that let the load-bearing assertion never run. A thread-local
/// default is private to this test's thread, so the counter always reflects
/// THIS test's requests and the skip path is gone.
///
/// `#[tokio::test]` defaults to the current-thread runtime, so the whole
/// test future runs on the thread holding the guard; the caller MUST hold
/// the guard for the entire request flow.
#[must_use]
fn install_counting_subscriber() -> (tracing::subscriber::DefaultGuard, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let layer = CountingLayer {
        http_request_spans: Arc::clone(&counter),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let guard = tracing::subscriber::set_default(subscriber);
    (guard, counter)
}

#[tokio::test]
async fn host_validate_rejects_before_trace_layer_allocates_span() {
    // Configure a single-entry Host allowlist via the process env so
    // `build_router_with_config` reaches `TrustedHosts::from_env()`
    // with a non-empty value. `temp_env::with_vars_async` serialises
    // env mutations across the test binary so a parallel sibling test
    // cannot race on the variable.
    // Hold the thread-local subscriber guard for the whole test; dropping
    // it at the end of the function restores the previous default.
    let (_guard, counter) = install_counting_subscriber();
    temp_env::async_with_vars([(ENV_TRUSTED_HOSTS, Some("api.example.com"))], async {
        let router = build_router_with_config(
            Arc::new(AppState::default()),
            AuthConfig::default(),
            TenantConfig::default(),
        );

        // Snapshot the counter AFTER router construction. The
        // builder itself does not open `http.request` spans; this
        // pins the assertion to the request flow only.
        let baseline = counter.load(Ordering::Relaxed);

        // (1) Hostile-Host shape: not in the allowlist -> 400.
        let req = Request::builder()
            .method(Method::GET)
            .uri("/healthz")
            .header("host", "evil.example.com")
            .body(Body::empty())
            .expect("well-formed request");
        let resp = router.clone().oneshot(req).await.expect("router oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "hostile Host must be rejected with 400 (host_validate envelope)",
        );
        let after_reject = counter.load(Ordering::Relaxed);

        // (2) No span allocated for the rejected request. If
        // host_validate ran INSIDE the trace layer (pre-T19
        // order), the trace layer would have opened a span
        // before the rejection short-circuited the chain and
        // `after_reject - baseline` would be 1. With the T19
        // re-order the rejection short-circuits the chain at
        // the OUTER edge of common_layers, before the trace
        // layer is ever entered, so the delta must be 0.
        let allowed = Request::builder()
            .method(Method::GET)
            .uri("/healthz")
            .header("host", "api.example.com")
            .body(Body::empty())
            .expect("well-formed request");
        let resp = router.oneshot(allowed).await.expect("router oneshot");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "allowlisted Host must reach the handler",
        );
        let after_allow = counter.load(Ordering::Relaxed);

        // Positive control as a HARD assertion (no soft skip): our
        // thread-local subscriber is always active, so the allowed
        // request MUST have opened exactly one `http.request` span.
        // A zero here means the subscriber install or the trace layer
        // regressed — fail loudly rather than skipping.
        assert_eq!(
            after_allow - after_reject,
            1,
            "allowed request must open exactly one http.request span \
                 (saw delta={}) — our thread-local subscriber should always \
                 be active",
            after_allow - after_reject,
        );

        // Load-bearing assertion: the rejected request must have
        // opened ZERO spans.
        assert_eq!(
            after_reject - baseline,
            0,
            "host_validate rejection opened an http.request span \
                 (delta={}) — the T19 layer re-order regressed: \
                 host_validate must sit OUTSIDE trace_layer_with_propagation \
                 so hostile probes never burn trace budget",
            after_reject - baseline,
        );
    })
    .await;
}
