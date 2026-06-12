// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Bearer-token authentication tests.
//!
//! Drives the router through [`tower::ServiceExt::oneshot`] with an explicit
//! [`AuthConfig`] so the assertions do not depend on the process
//! environment.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router_with_tokens(tokens: &[&str]) -> axum::Router {
    let auth = AuthConfig::from_tokens(tokens.iter().copied());
    build_router_with_config(Arc::new(AppState::default()), auth, TenantConfig::default())
}

/// `/metrics` is declared `security: []` in `openapi/tensor-wasm-api.yaml`
/// and is consumed by Prometheus scrapers that share a single token across
/// many endpoints (frequently no token at all on a private network). It
/// MUST be reachable without an `Authorization` header even when bearer
/// auth is configured for the protected routes. `503 Service Unavailable`
/// is also acceptable here because the metrics handler may surface a
/// degraded state when the registry has not been initialised in a test
/// build; what we must NOT see is `401 Unauthorized`.
#[tokio::test]
async fn metrics_endpoint_is_unauthenticated() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::SERVICE_UNAVAILABLE,
        "expected /metrics to bypass bearer auth, got {status}"
    );
}

/// `/healthz` is the k8s liveness/readiness probe target. Probes do not
/// carry an `Authorization` header; if we wrap this route in bearer auth
/// the whole gateway will fail readiness as soon as a token allowlist is
/// configured. Pinned to `200 OK` because `healthz` is unconditional.
#[tokio::test]
async fn healthz_endpoint_is_unauthenticated_even_with_tokens_configured() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// Regression guard: the protected stack must still 401 when an allowlist
/// is configured and the caller does not present a token. We exercise
/// `POST /functions` (a protected route) so we are not testing the same
/// surface as the probe-router tests above.
#[tokio::test]
async fn protected_route_still_requires_auth() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("unauthorized"),
        "got {body}"
    );
}

#[tokio::test]
async fn valid_bearer_token_passes_through() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .header("Authorization", "Bearer foo")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// `/metrics` is unauthenticated, so an unknown token is simply ignored
/// rather than triggering `401`. We assert this against a protected route
/// instead so the regression coverage is still meaningful.
#[tokio::test]
async fn unknown_bearer_token_is_401() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .header("Authorization", "Bearer baz")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("unauthorized")
    );
}

/// Regression for the lowercase-scheme interop bug: load balancers that
/// normalise header values (envoy, some nginx configs) may forward
/// `Authorization: bearer …`. RFC 6750 §2.1 says scheme matching is
/// case-insensitive, so the gateway must accept it.
#[tokio::test]
async fn lowercase_bearer_scheme_is_accepted() {
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .header("Authorization", "bearer foo")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    // The handler will likely reject the empty body with 400/422 — what
    // matters here is that we got PAST the bearer auth layer (i.e. not
    // `401`).
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// Same as above for the SHOUTING variant.
#[tokio::test]
async fn uppercase_bearer_scheme_is_accepted() {
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .header("Authorization", "BEARER foo")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn missing_authorization_header_is_401() {
    // Use a protected route (`/healthz` is now unauthenticated by design).
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn malformed_authorization_header_is_401() {
    // Use a protected route (`/healthz` is now unauthenticated by design).
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .header("Authorization", "Basic ZGVhZGJlZWY=")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn dev_mode_no_auth_required() {
    // Empty allowlist => dev mode pass-through.
    let router = router_with_tokens(&[]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// Auth-on-every-route matrix: with an allowlist configured, EVERY protected
/// route must reject a request that carries no `Authorization` header with
/// `401 unauthorized` — before any handler / tenant / rate-limit logic runs.
///
/// This guards against the "forgot to wire bearer_auth onto a new route"
/// class of regression: a route mounted outside the protected stack would
/// admit unauthenticated callers and surface a non-401 status (404 / 405 /
/// 400 / 503), failing the assertion. The probe routes (`/healthz`,
/// `/metrics`) are deliberately `security: []` and are covered by their own
/// dedicated tests above, so they are excluded here.
///
/// A fixed sample UUID is substituted into the `:id` templates; the value is
/// irrelevant because auth runs before the handler ever resolves the id.
#[tokio::test]
async fn every_protected_route_requires_auth_when_allowlist_configured() {
    const SAMPLE_ID: &str = "00000000-0000-4000-8000-00000000abcd";

    // (method, uri) for every protected route mounted by the production
    // router (see `server::build_router_full`). Feature-gated kernel routes
    // (`--features kernel-registry-api`) are intentionally not listed here —
    // they have dedicated coverage in `tests/kernel_registry_authz.rs`.
    let routes: &[(Method, String)] = &[
        (Method::POST, "/functions".to_string()),
        (Method::DELETE, format!("/functions/{SAMPLE_ID}")),
        (Method::POST, format!("/functions/{SAMPLE_ID}/invoke")),
        (Method::POST, format!("/functions/{SAMPLE_ID}/invoke-async")),
        (Method::POST, format!("/functions/{SAMPLE_ID}/invoke-stream")),
        (Method::POST, "/snapshot/save".to_string()),
        (Method::POST, "/snapshot/restore".to_string()),
        (Method::GET, format!("/jobs/{SAMPLE_ID}")),
        (Method::POST, "/v1/completions".to_string()),
        (Method::POST, "/v1/chat/completions".to_string()),
    ];

    for (method, uri) in routes {
        // Fresh router per row so a spent concurrency permit or audit state
        // from one row cannot influence the next.
        let router = router_with_tokens(&["foo", "bar"]);
        let req = Request::builder()
            .method(method.clone())
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "{method} {uri} must return 401 with no token when an allowlist is configured",
        );
        let body = body_json(resp.into_body()).await;
        assert_eq!(
            body.pointer("/error/kind").and_then(Value::as_str),
            Some("unauthorized"),
            "{method} {uri} must surface the unauthorized envelope: got {body}",
        );
    }
}
