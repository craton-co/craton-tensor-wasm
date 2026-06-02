// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tests for the `host_validate` middleware (api S-30).
//!
//! These cover the refactor that lifted the trusted-hosts allowlist out
//! of a process-wide `OnceLock` and onto an
//! `axum::Extension<TrustedHosts>`. Each test builds a tiny router that
//! wraps an `echo` handler with the production `host_validate`
//! middleware plus an explicit `TrustedHosts` extension, sends a single
//! request via `tower::ServiceExt::oneshot`, and asserts the status code.
//!
//! Coverage:
//!
//! * Empty allowlist passes through (legacy default).
//! * Non-empty allowlist accepts a listed host (exact match).
//! * Case-insensitive matching (Host: API.EXAMPLE.COM vs. allowlist
//!   entry `api.example.com`).
//! * Non-empty allowlist rejects an unlisted host with 400.
//! * Missing Host header + non-empty allowlist returns 400.
//! * HTTP/2 `:authority` fallback: when no Host header is present but
//!   the URI carries an authority (the representation hyper uses for an
//!   HTTP/2 `:authority` pseudo-header), the middleware reads the
//!   authority and admits the request if it matches.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::middleware::from_fn;
use axum::routing::get;
use axum::Router;
use tensor_wasm_api::middleware::host_validate;
use tensor_wasm_api::TrustedHosts;
use tower::ServiceExt;

/// Bare handler used as the inner service. Returns `200 OK` with an
/// empty body — every test only cares about the status code, since
/// host_validate either lets the request through (handler runs) or
/// short-circuits with a 400.
async fn ok_handler() -> &'static str {
    "ok"
}

/// Build a router that wires `host_validate` in front of `/probe`,
/// injecting the supplied `TrustedHosts` via `axum::Extension` so the
/// middleware sees it instead of falling back to the env var.
///
/// Layer order matches `crate::server::build_router_with_audit`:
/// Extension first (so it is on the request when `host_validate`
/// reads it), then `from_fn(host_validate)`.
fn router(allow: TrustedHosts) -> Router {
    let stack = tower::ServiceBuilder::new()
        .layer(axum::Extension(allow))
        .layer(from_fn(host_validate));
    Router::new().route("/probe", get(ok_handler)).layer(stack)
}

#[tokio::test]
async fn host_validate_passes_through_when_allowlist_empty() {
    // TrustedHosts::allow_all() = empty list = legacy default. The
    // middleware must not even bother looking at the Host header.
    let router = router(TrustedHosts::allow_all());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "anything.example.com")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_rejects_unlisted_host() {
    // Allowlist names api.example.com only. A request claiming to be
    // evil.com is rejected with 400 — closes api S-30's Host-spoof gap.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "evil.com")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn host_validate_accepts_listed_host() {
    // Exact-match positive case. Host header matches the single
    // allowlist entry verbatim.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "api.example.com")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_accepts_listed_host_case_insensitive() {
    // RFC 3986 §3.2.2 makes the host component case-insensitive.
    // Upstream proxies (or hostile probers) may upper-case the Host
    // header; `TrustedHosts::contains` lowercases on both sides so the
    // match still holds.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "API.EXAMPLE.COM")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_rejects_missing_host_when_configured() {
    // No Host header AND no absolute-form URI authority. With a
    // non-empty allowlist the middleware MUST reject — defaulting to
    // pass-through when both sources are absent would let a hostile
    // HTTP/1.0 client (no Host required) bypass the gate.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn host_validate_uses_http2_authority_fallback() {
    // HTTP/2 sends `:authority` instead of a `Host` header. In hyper's
    // normalised request, an absolute-form URI surfaces the authority
    // via `req.uri().authority()`. We build a request with an
    // absolute-form URI but NO Host header and assert that the
    // middleware reads the URI authority and admits the request when
    // it matches the allowlist.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("http://api.example.com/probe")
        .body(Body::empty())
        .unwrap();
    // Sanity-check our assumption about the constructed request: the
    // URI must carry an authority and the Host header must be absent.
    assert_eq!(
        req.uri().authority().map(|a| a.as_str()),
        Some("api.example.com"),
    );
    assert!(req.headers().get(axum::http::header::HOST).is_none());
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_rejects_http2_authority_when_unlisted() {
    // Negative HTTP/2 case: URI authority is the only host signal and
    // it does NOT match the allowlist.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("http://evil.com/probe")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn host_validate_default_port_strip_443() {
    // Operator listed bare `api.example.com` (no port). A client that
    // includes the default HTTPS port (`:443`) in the Host header
    // should still be admitted — RFC 7230 §5.4 lets a Host carry the
    // default port and treats it as equivalent.
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "api.example.com:443")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_default_port_strip_80() {
    let router = router(TrustedHosts::from_hosts(["api.example.com"]));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "api.example.com:80")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn host_validate_port_bound_entry_requires_exact_match() {
    // When the operator pins an entry to a non-default port
    // (`:8443`), the default-port-strip heuristic MUST NOT fire — a
    // bare `api.example.com` (or default-port variant) is not the
    // same surface as `api.example.com:8443`.
    let router = router(TrustedHosts::from_hosts(["api.example.com:8443"]));

    // Exact match on the port-bound entry: accepted.
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "api.example.com:8443")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Bare host (no port): rejected — operator was specific.
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header("host", "api.example.com")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn trusted_hosts_from_raw_parses_comma_separated_list() {
    let parsed = TrustedHosts::from_raw("api.example.com, api2.example.com ,");
    assert!(parsed.contains("api.example.com"));
    assert!(parsed.contains("api2.example.com"));
    assert!(!parsed.contains("evil.com"));
}

#[test]
fn trusted_hosts_from_raw_empty_is_allow_all() {
    let parsed = TrustedHosts::from_raw("");
    assert!(parsed.is_empty());
    assert!(parsed.contains("anything.example.com"));
}

#[test]
fn trusted_hosts_contains_is_case_insensitive() {
    let parsed = TrustedHosts::from_hosts(["api.example.com"]);
    assert!(parsed.contains("API.EXAMPLE.COM"));
    assert!(parsed.contains("Api.Example.Com"));
}
