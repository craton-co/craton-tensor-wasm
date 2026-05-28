// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the restored `POST
//! /functions/{id}/invoke-stream` route (roadmap feature #2, v0.3.7
//! scaffold).
//!
//! The route was originally mounted by B6.1 and then transiently un-routed
//! by the B6.2 merge (see `f00822b — fix(api): un-route /invoke-stream`).
//! These tests pin the wire surface a v0.4 implementation must preserve so
//! a future merge cannot silently drop the route again:
//!
//! 1. `sse_negotiation_returns_event_stream_content_type` — `Accept:
//!    text/event-stream` selects the SSE response shape; the body carries
//!    the scaffold marker so clients can detect the not-yet-wired state.
//! 2. `default_accept_returns_octet_stream` — anything else falls back to
//!    the chunked-transfer `application/octet-stream` branch.
//! 3. `unknown_function_id_returns_404` — the route shares the same
//!    not-found envelope as `/invoke` (so a probing scanner cannot
//!    fingerprint the streaming path).
//! 4. `cross_tenant_returns_403` — the route honours the per-tenant scope
//!    on the bearer token. The current `FunctionRecord` does not carry an
//!    owning tenant, so the 403 is asserted via the `tenant_scope_denied`
//!    envelope produced by `AuthContext::authorize_tenant` when the
//!    bearer's scope does not cover the request's `X-TensorWasm-Tenant`.
//! 5. `oversized_body_returns_413` — the per-route body cap installed in
//!    `build_router_with_*` rejects payloads above
//!    `MAX_REQUEST_BODY_BYTES` before the handler runs, so the streaming
//!    route inherits the same DoS-protection envelope as `/invoke`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, TokenScope, HEADER_TENANT,
    MAX_REQUEST_BODY_BYTES,
};
use tensor_wasm_core::types::TenantId;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

async fn body_json(body: Body) -> Value {
    let bytes = body_bytes(body).await;
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

/// Deploy a trivial `_start`-only Wasm module and return its id. Matches
/// the helper shape used by `streaming_invoke_scaffold.rs` and
/// `async_invoke_flow.rs` so the streaming surface is exercised against
/// the same fixture every other invoke test uses.
async fn deploy_trivial(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "stream_restored", "wasm_b64": wasm_b64 }))
                .unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK, "deploy must succeed");
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

#[tokio::test]
async fn sse_negotiation_returns_event_stream_content_type() {
    let router = router();
    let id = deploy_trivial(&router).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-stream");

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );

    let body = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8 body");
    assert!(
        body.contains("event: scaffold"),
        "scaffold event missing from SSE body: {body:?}"
    );
    assert!(
        body.contains("not_yet_wired"),
        "scaffold payload missing from SSE body: {body:?}"
    );
}

#[tokio::test]
async fn default_accept_returns_octet_stream() {
    let router = router();
    let id = deploy_trivial(&router).await;

    // No Accept header at all — we should get the chunked fallback.
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-stream");

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("application/octet-stream"),
        "expected octet-stream content-type, got {ct:?}"
    );

    let body = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8 body");
    // The chunked branch reuses the SSE event shape verbatim so existing
    // clients can detect the not-yet-wired state from either negotiation
    // outcome.
    assert!(
        body.contains("not_yet_wired"),
        "scaffold payload missing from chunked body: {body:?}"
    );
}

#[tokio::test]
async fn unknown_function_id_returns_404() {
    let router = router();
    // No deploy — the function id is bogus.
    let bogus = uuid::Uuid::new_v4();
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{bogus}/invoke-stream"))
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-stream");

    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "unknown function id must 404 before any streaming work"
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_found"),
        "expected error.kind=not_found, got {body}"
    );
}

#[tokio::test]
async fn cross_tenant_returns_403() {
    // Build a router with a token scoped to tenants {1, 2}. The streaming
    // route shares the same `bearer_auth` → `authorize_tenant` chain as
    // `/invoke`, so addressing tenant=3 must surface
    // `403 tenant_scope_denied` — mirroring the assertion in
    // `scoped_tokens_test.rs::scoped_token_denied_for_out_of_scope_tenant`.
    let mut scopes: HashMap<String, TokenScope> = HashMap::new();
    scopes.insert(
        "scoped".to_string(),
        TokenScope::from_tenants([TenantId(1), TenantId(2)]),
    );
    let auth = AuthConfig::from_scopes(scopes);
    let router = build_router_with_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
    );

    // Deploy the trivial fixture with an in-scope tenant header so the
    // bearer is accepted.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", "Bearer scoped")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "stream_403", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id");

    // Address tenant=3, which is OUTSIDE the bearer's {1,2} scope.
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        .header("authorization", "Bearer scoped")
        .header(HEADER_TENANT, "3")
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-stream");

    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "expected error.kind=tenant_scope_denied, got {body}",
    );
}

#[tokio::test]
async fn oversized_body_returns_413() {
    // Constructing a body larger than `MAX_REQUEST_BODY_BYTES` (64 MiB).
    // `DefaultBodyLimit::max` short-circuits with `413 Payload Too Large`
    // before the streaming handler runs — see `oversized_body.rs` for the
    // matching coverage on `/functions`.
    let router = router();
    // We do NOT need a deployed function: the body cap fires before the
    // handler resolves the function id. Use a syntactically-valid UUID so
    // path routing succeeds and the per-route body cap is the first guard
    // to reject.
    let bogus = uuid::Uuid::new_v4();
    let huge: Vec<u8> = vec![b'a'; MAX_REQUEST_BODY_BYTES + 1];
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{bogus}/invoke-stream"))
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(huge))
        .unwrap();

    let resp = router.oneshot(req).await.expect("invoke-stream");
    let status = resp.status();
    assert_eq!(
        status,
        StatusCode::PAYLOAD_TOO_LARGE,
        "expected 413 for oversized body on /invoke-stream, got {status}"
    );
}
