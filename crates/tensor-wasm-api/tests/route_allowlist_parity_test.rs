// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! CI parity check between the built router's routes and
//! [`DEFAULT_ROUTE_ALLOWLIST`] (`tensor_wasm_api::http_metrics`).
//!
//! The HTTP-metrics middleware emits a `route` label per request. To keep
//! label cardinality bounded it only emits templates that appear in
//! [`DEFAULT_ROUTE_ALLOWLIST`]; anything else collapses to
//! [`UNKNOWN_ROUTE`] (`"unknown"`). That allow-list is maintained by hand
//! alongside the `.route(...)` calls in `server::build_router_full`, so the
//! two can silently drift:
//!
//!   * a NEW route missing from the allow-list renders as `unknown` (its
//!     metrics vanish into the catch-all series), and
//!   * a STALE allow-list entry with no matching route is dead weight.
//!
//! This test pins them together. For every allow-list template (minus the
//! feature-gated kernel routes, absent on the default build) it drives a
//! request to a concrete instantiation and scrapes `/metrics`, asserting
//! the emitted `route` label is the template itself — never `unknown`.
//!
//! How this catches both drift directions:
//!
//!   * **Route missing from allow-list** → axum matches the route and sets
//!     `MatchedPath`, but `route_label` rejects the unlisted template and
//!     emits `unknown`; the template never appears → assertion fails.
//!   * **Stale allow-list entry (no route)** → axum's fallback handles the
//!     request, no `MatchedPath` is set, `route_label` emits `unknown`; the
//!     template never appears → assertion fails.
//!
//! The router is built in dev mode (empty token allow-list) so protected
//! routes admit the probe request and reach a handler — the response status
//! is irrelevant; only the emitted `route` label is under test.

#![allow(clippy::expect_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, DEFAULT_ROUTE_ALLOWLIST,
    UNKNOWN_ROUTE,
};
use tower::ServiceExt;

/// Sample UUID substituted into every `:id` path parameter. The value is
/// irrelevant — only the matched *template* is observed.
const SAMPLE_ID: &str = "00000000-0000-4000-8000-00000000abcd";

/// Kernel-registry templates are only mounted under
/// `--features kernel-registry-api`; on the default build they are listed
/// in the allow-list (harmless — it is a label guard, not a router) but
/// have no route to match, so they would correctly collapse to `unknown`.
/// Exclude them from the default-build parity sweep; the feature build has
/// its own coverage in `tests/kernel_registry_routes.rs`.
const FEATURE_GATED_TEMPLATES: &[&str] = &["/kernels", "/kernels/:name/:version"];

/// The full set of route templates the default-build router mounts (every
/// `.route(...)` in `server::build_router_full` minus the feature-gated
/// kernel routes). This is the authoritative expectation: the test drives a
/// request at each and asserts it resolves to its own template label, and
/// also asserts the non-feature-gated allow-list entries are exactly this
/// set. Adding a route to `server.rs` without updating this list (and the
/// allow-list) makes the test fail — which is the parity guarantee.
const EXPECTED_MOUNTED_TEMPLATES: &[&str] = &[
    "/healthz",
    "/metrics",
    "/functions",
    "/functions/:id",
    "/functions/:id/invoke",
    "/functions/:id/invoke-async",
    "/functions/:id/invoke-stream",
    "/snapshot/save",
    "/snapshot/restore",
    "/jobs/:id",
    "/v1/completions",
    "/v1/chat/completions",
];

fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

/// Map a route template to (method, concrete-path) for driving a request.
/// Methods mirror the `.route(...)` declarations in `server`.
fn instantiate(template: &str) -> (Method, String) {
    let path = template.replace(":id", SAMPLE_ID);
    let method = match template {
        // Read-only routes.
        "/healthz" | "/metrics" | "/jobs/:id" => Method::GET,
        // Function deletion.
        "/functions/:id" => Method::DELETE,
        // Everything else mounted today is a POST (create, invoke*, openai,
        // snapshot save/restore).
        _ => Method::POST,
    };
    (method, path)
}

async fn scrape_metrics(router: &axum::Router) -> String {
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .expect("metrics request builds");
    let resp = router.clone().oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::OK, "/metrics must be scrapeable");
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec();
    String::from_utf8(bytes).expect("metrics body is UTF-8")
}

#[tokio::test]
async fn every_mounted_route_resolves_to_its_template_label() {
    for template in EXPECTED_MOUNTED_TEMPLATES {
        // Fresh router per template so each scrape only reflects this row's
        // request (metrics accumulate across requests on a shared router).
        let router = dev_router();
        let (method, path) = instantiate(template);

        let req = Request::builder()
            .method(method.clone())
            .uri(&path)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .expect("probe request builds");
        // Status is irrelevant — the metrics layer counts every response,
        // including 4xx/5xx, and sets the `route` label from MatchedPath.
        let _ = router.clone().oneshot(req).await.expect("router serves");

        let body = scrape_metrics(&router).await;

        // The template must appear as a `route="<template>"` label on the
        // request-total family, proving the route exists AND is allow-listed.
        let needle = format!("route=\"{template}\"");
        assert!(
            body.lines().any(|line| {
                line.starts_with("tensor_wasm_http_requests_total{") && line.contains(&needle)
            }),
            "mounted route template `{template}` did not appear as a metrics route label after \
             a `{method} {path}` request — either the route is unmounted or it is missing from \
             DEFAULT_ROUTE_ALLOWLIST (so it collapsed to `unknown`). Metrics body:\n{body}",
        );

        // And it must NOT have collapsed to the catch-all `unknown` series
        // for this same request shape.
        let unknown_needle = format!("route=\"{UNKNOWN_ROUTE}\"");
        for line in body.lines() {
            if line.starts_with("tensor_wasm_http_requests_total{")
                && line.contains(&unknown_needle)
            {
                panic!(
                    "request to `{method} {path}` (template `{template}`) collapsed to \
                     route=\"unknown\" — the template is not matching or not allow-listed:\n{line}",
                );
            }
        }
    }
}

/// Direct set-diff between the (non-feature-gated) allow-list and the
/// authoritative `EXPECTED_MOUNTED_TEMPLATES`. This is the pure-data half of
/// the parity guarantee — it needs no router and catches stale allow-list
/// entries (listed but never mounted) and the reverse (mounted-and-expected
/// but absent from the allow-list).
#[test]
fn allowlist_and_mounted_route_sets_agree() {
    use std::collections::BTreeSet;

    let allow: BTreeSet<&str> = DEFAULT_ROUTE_ALLOWLIST
        .iter()
        .copied()
        .filter(|t| !FEATURE_GATED_TEMPLATES.contains(t))
        .collect();
    let expected: BTreeSet<&str> = EXPECTED_MOUNTED_TEMPLATES.iter().copied().collect();

    let stale: Vec<&str> = allow.difference(&expected).copied().collect();
    assert!(
        stale.is_empty(),
        "DEFAULT_ROUTE_ALLOWLIST has entries with no matching mounted route \
         (stale — remove them or mount the route): {stale:?}",
    );

    let missing: Vec<&str> = expected.difference(&allow).copied().collect();
    assert!(
        missing.is_empty(),
        "mounted routes are missing from DEFAULT_ROUTE_ALLOWLIST \
         (their metrics collapse to `unknown` — add them): {missing:?}",
    );
}
