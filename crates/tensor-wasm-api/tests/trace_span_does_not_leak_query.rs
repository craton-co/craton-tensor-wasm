// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for api S-17: tracing-span attribute sanitisation.
//!
//! The `trace_layer_with_propagation` middleware in
//! `tensor-wasm-api::middleware` opens an `http.request` span for every
//! inbound HTTP request and records two attacker-controlled inputs on
//! it: the request URI and the `traceparent` header. Both were
//! historically captured verbatim, which gave a hostile client two
//! cheap exfiltration / log-injection channels:
//!
//! 1. **Query-string leak.** `uri = %req.uri()` recorded the *full*
//!    URI including the query string. A request like
//!    `GET /healthz?secret=ATTACKER_SECRET` would plant the secret in
//!    every log line tagged with that span — and `/healthz` is the
//!    one route a third-party load balancer is most likely to hit
//!    with arbitrary query parameters during probing. The fix records
//!    only `path = %req.uri().path()` (no query string ever).
//! 2. **traceparent injection / unbounded growth.** `traceparent`
//!    was recorded raw. A 64 KiB header value would land in every
//!    span attribute; a `\r\n`-containing value would forge a fake
//!    log line in JSON sinks that escape per-line rather than
//!    per-field. The fix sanitises via `sanitise_traceparent`: clamp
//!    to 64 bytes, strip non-printable, collapse CR/LF/NUL to
//!    `<invalid>`.
//!
//! This test drives a real router with a real tracing subscriber
//! installed and asserts that NO recorded span attribute contains
//! the planted secrets or unbounded header content. It is the
//! integration-level proof that the middleware sanitisation actually
//! takes effect end-to-end (the unit tests in `middleware.rs` cover
//! the helper in isolation).

#![allow(clippy::expect_used)]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;
use tracing::field::{Field, Visit};
use tracing::span;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// A single captured field on a span: the field name and its
/// debug-formatted value. We capture by debug because the production
/// middleware records `path = %req.uri().path()` and `traceparent =
/// %sanitised_tp`, both of which surface to a Visitor through
/// `record_debug` (the `%` sigil uses `fmt::Display`, which the
/// tracing crate funnels through `record_debug`).
#[derive(Debug, Clone)]
struct CapturedField {
    span_name: &'static str,
    field_name: &'static str,
    value: String,
}

/// `tracing::field::Visit` impl that drains every field on a span into
/// our shared `Vec<CapturedField>`. We capture `&dyn Debug` (the
/// generic case, which `record_debug` is called for) and also
/// implement the typed methods to be safe across tracing versions.
struct CapturingVisitor<'a> {
    span_name: &'static str,
    out: &'a Mutex<Vec<CapturedField>>,
}

impl<'a> Visit for CapturingVisitor<'a> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let mut s = String::new();
        let _ = write!(&mut s, "{value:?}");
        if let Ok(mut guard) = self.out.lock() {
            guard.push(CapturedField {
                span_name: self.span_name,
                field_name: field.name(),
                value: s,
            });
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if let Ok(mut guard) = self.out.lock() {
            guard.push(CapturedField {
                span_name: self.span_name,
                field_name: field.name(),
                value: value.to_owned(),
            });
        }
    }
}

/// `tracing_subscriber::Layer` that captures every field on every
/// newly-opened span. The two channels we care about
/// (`http.request.path` and `http.request.traceparent`) are recorded
/// at span construction time via `info_span!(...)`, which lands them
/// on `on_new_span`. Capturing only `on_new_span` is therefore
/// sufficient — we do not need to instrument `on_record`.
struct CapturingLayer {
    captured: Arc<Mutex<Vec<CapturedField>>>,
}

impl<S> Layer<S> for CapturingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &span::Id, _ctx: Context<'_, S>) {
        // Translate the dynamic span name into one of the few names we
        // care about so the captured records are filterable without
        // allocating per observation. Spans outside the allowlist get
        // tagged `"__other__"` so the count surfaces in any debug
        // dump but the assertions below ignore them.
        let raw_name = attrs.metadata().name();
        let span_name: &'static str = match raw_name {
            "http.request" => "http.request",
            _ => "__other__",
        };
        let mut visitor = CapturingVisitor {
            span_name,
            out: &self.captured,
        };
        attrs.record(&mut visitor);
    }
}

/// Install our capturing subscriber as the **thread-local** default for
/// the duration of the returned guard.
///
/// We deliberately use [`tracing::subscriber::set_default`] (thread-local)
/// rather than `set_global_default` / `try_init` (process-global). The
/// global path is set-once-per-process: under the shared test harness only
/// the first test to install wins, and every other test silently captures
/// nothing — which used to force a vacuous "soft skip" that let the leak
/// assertions pass without ever running. A thread-local default is private
/// to the calling test's thread, so EVERY test reliably observes its own
/// spans and the assertions are never skipped.
///
/// `#[tokio::test]` defaults to the current-thread runtime, so the whole
/// test future runs on the thread that holds the guard; the thread-local
/// subscriber therefore stays active across every `.await` in the test.
/// The caller MUST hold the returned guard for the entire request flow.
#[must_use]
fn install_capturing_subscriber(
    captured: Arc<Mutex<Vec<CapturedField>>>,
) -> tracing::subscriber::DefaultGuard {
    let layer = CapturingLayer { captured };
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::set_default(subscriber)
}

/// Spin up the production router with default state — same helper
/// pattern as `trace_propagation_test::router`. Default config means
/// no bearer auth, no required tenant header, so `/healthz` is
/// reachable without ceremony.
fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

/// Drain the captured-field vector and return the snapshot. Splitting
/// this out keeps the assertion blocks tidy and ensures the lock is
/// released before any `assert!` panics (poisoning the mutex would
/// otherwise hide the real failure behind a secondary panic in
/// another test).
fn snapshot(captured: &Mutex<Vec<CapturedField>>) -> Vec<CapturedField> {
    captured.lock().expect("captured lock not poisoned").clone()
}

#[tokio::test]
async fn span_attributes_do_not_contain_query_string_secrets() {
    // The two well-known sentinel values we plant in the request URI.
    // Any occurrence of either in a captured span field is a leak.
    const ATTACKER_SECRET: &str = "ATTACKER_SECRET";
    const TOKEN_VALUE: &str = "ABC123";

    let captured: Arc<Mutex<Vec<CapturedField>>> = Arc::new(Mutex::new(Vec::new()));
    // Hold the thread-local subscriber guard for the whole test — see
    // `install_capturing_subscriber`. Dropping it restores the previous
    // default, so we keep it bound until the assertions finish.
    let _guard = install_capturing_subscriber(Arc::clone(&captured));

    let router = dev_router();
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!(
            "/healthz?secret={ATTACKER_SECRET}&token={TOKEN_VALUE}"
        ))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.expect("router runs");
    assert_eq!(resp.status(), StatusCode::OK, "healthz must succeed");

    // The thread-local subscriber is private to this test, so everything
    // captured belongs to this request — no `pre_len` windowing needed.
    let load_obs: Vec<CapturedField> = snapshot(&captured);
    // Non-vacuous: with our own subscriber active we MUST have captured at
    // least one span field. A regression that drops the trace layer (or a
    // future change that breaks the thread-local install) surfaces here
    // rather than letting the leak assertions pass over an empty set.
    assert!(
        !load_obs.is_empty(),
        "no spans captured under our own thread-local subscriber — the \
         trace layer or the subscriber install regressed",
    );
    let load_obs: Vec<&CapturedField> = load_obs.iter().collect();

    // Walk every captured field. NONE of them, on any span, on any
    // field name, may contain the secret tokens we planted. This is
    // the load-bearing assertion: a regression that puts the full
    // URI back on a span fails here loudly.
    for obs in &load_obs {
        assert!(
            !obs.value.contains(ATTACKER_SECRET),
            "span `{}` field `{}` leaked the query secret `{ATTACKER_SECRET}`: {:?}",
            obs.span_name,
            obs.field_name,
            obs.value,
        );
        assert!(
            !obs.value.contains(TOKEN_VALUE),
            "span `{}` field `{}` leaked the query token `{TOKEN_VALUE}`: {:?}",
            obs.span_name,
            obs.field_name,
            obs.value,
        );
    }

    // Defensive: confirm we actually exercised the `http.request`
    // span on this request (otherwise the loop above could pass
    // vacuously if the middleware were silently disabled).
    let saw_http_request_path = load_obs
        .iter()
        .any(|o| o.span_name == "http.request" && o.field_name == "path");
    assert!(
        saw_http_request_path,
        "no `http.request` span recorded a `path` field — \
         middleware sanitisation may have been removed entirely",
    );
}

#[tokio::test]
async fn span_attributes_bound_oversized_traceparent_header() {
    // 64 KiB of attacker-supplied "traceparent". A pre-fix middleware
    // would record all 64 KiB verbatim into every log line that
    // touches the span — a cheap DoS on the log pipeline and a
    // delicious oracle if the value gets stored anywhere durable.
    // The CRLF-injection sibling attack is exercised at the unit-test
    // level in `middleware.rs::sanitise_traceparent_rejects_crlf_*`
    // because `HeaderValue::try_from` refuses to ship `\r\n`-bearing
    // values across hyper, which makes the CRLF case impossible to
    // drive through an integration request.
    let attacker_tp: String = "A".repeat(65_536);

    let captured: Arc<Mutex<Vec<CapturedField>>> = Arc::new(Mutex::new(Vec::new()));
    // Thread-local subscriber guard, held for the whole test (see the first
    // test for the rationale on `set_default` vs the global path).
    let _guard = install_capturing_subscriber(Arc::clone(&captured));

    let router = dev_router();
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .header("traceparent", attacker_tp.as_str())
        .body(Body::empty())
        .expect("64 KiB all-'A' header is a valid HeaderValue");

    // `axum::Router::oneshot` cannot fail (Service::Error =
    // Infallible). The handler MAY return a non-200 status if a
    // downstream limit kicks in on a huge header; both outcomes
    // prove the attack does not land in a span, so we record the
    // status only for debugging context and proceed to inspect the
    // captured spans regardless.
    let resp = router
        .oneshot(req)
        .await
        .expect("router::oneshot infallible");
    let _ = resp.status();

    // Private thread-local subscriber: every captured field is ours.
    let snap = snapshot(&captured);
    let load_obs: Vec<&CapturedField> = snap.iter().collect();
    assert!(
        !load_obs.is_empty(),
        "no spans captured under our own thread-local subscriber — the \
         trace layer or the subscriber install regressed",
    );

    // Find every captured `traceparent` field and assert the bound.
    let tp_obs: Vec<&CapturedField> = load_obs
        .iter()
        .copied()
        .filter(|o| o.span_name == "http.request" && o.field_name == "traceparent")
        .collect();

    // If we saw any, every one of them must be inside the bound.
    // We size the bound generously (96 bytes) because the captured
    // value goes through the tracing field-formatting layer which
    // may add surrounding quotes / escape sequences; the underlying
    // sanitiser caps the *content* at 64 bytes.
    for obs in &tp_obs {
        assert!(
            obs.value.len() <= 96,
            "captured traceparent `{}` exceeded length cap ({} bytes) — \
             sanitise_traceparent did not clamp",
            obs.value,
            obs.value.len(),
        );
        assert!(
            !obs.value.contains('\r') && !obs.value.contains('\n'),
            "captured traceparent contained CR/LF — sanitise_traceparent \
             must collapse control chars to `<invalid>`: {:?}",
            obs.value,
        );
        // The 64 KiB attack string was all-'A'. After clamping, at
        // most 64 'A's may remain; any field that contains a long
        // run of 'A's beyond what the cap allows is a regression.
        assert!(
            !obs.value.contains(&"A".repeat(100)),
            "captured traceparent retained a 100+ run of attacker padding: {:?}",
            obs.value,
        );
    }
}
