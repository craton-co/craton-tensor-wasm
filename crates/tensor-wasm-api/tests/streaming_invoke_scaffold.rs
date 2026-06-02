// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for `POST /functions/{id}/invoke-stream` (roadmap
//! feature #2). T34 (v0.4) replaced the v0.3.7 `event: scaffold`
//! frame with real end-to-end wiring through
//! [`tensor_wasm_wasi_gpu::streaming::StreamingContext`]; these tests
//! pin the surface the wiring landed:
//!
//! * The route is mounted on the protected sub-router and reachable
//!   through the production router builder.
//! * SSE negotiation works — `Accept: text/event-stream` returns a
//!   `text/event-stream` content-type whose body terminates with an
//!   `event: done` frame (the v0.3.7 `event: scaffold` placeholder is
//!   gone; see `tests/invoke_stream_real_emit.rs` for the
//!   `event: chunk` coverage on a guest that actually emits).
//! * Chunked-transfer fallback works — any other `Accept` returns
//!   `application/octet-stream` whose body carries the same
//!   `event: done` terminator so clients that don't negotiate SSE can
//!   still detect end-of-stream uniformly.
//! * 404 still applies before any streaming work happens (so a
//!   probing scanner cannot fingerprint the streaming path).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};

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

/// Deploy a trivial `_start`-only Wasm module and return its id. Mirrors
/// the helper in `async_invoke_flow.rs` so the streaming surface is
/// exercised against the same fixture shape.
async fn deploy_trivial(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "streaming", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK);
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

#[tokio::test]
async fn invoke_stream_with_sse_accept_returns_event_stream() {
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
    // T34: the v0.3.7 `event: scaffold` placeholder is replaced by
    // an `event: done` terminator. The fixture exports a no-op
    // `_start` that emits nothing, so the body carries one done
    // frame and no chunk frames.
    assert!(
        body.contains("event: done"),
        "expected terminal `event: done` frame in SSE body: {body:?}"
    );
    assert!(
        body.contains("\"status\":\"ok\""),
        "expected `status: ok` payload on done frame: {body:?}"
    );
    assert!(
        !body.contains("event: scaffold"),
        "stale `event: scaffold` placeholder must be gone from SSE body: {body:?}"
    );
}

#[tokio::test]
async fn invoke_stream_without_sse_accept_returns_chunked_octets() {
    let router = router();
    let id = deploy_trivial(&router).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        // No Accept header at all — we should get the chunked fallback.
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
    // The chunked branch uses the same terminal `event: done` line
    // as the SSE branch so clients consuming either negotiation
    // outcome detect end-of-stream uniformly.
    assert!(
        body.contains("event: done"),
        "expected terminal `event: done` line in chunked body: {body:?}"
    );
    assert!(
        !body.contains("not_yet_wired"),
        "stale `not_yet_wired` placeholder must be gone from chunked body: {body:?}"
    );
}

#[tokio::test]
async fn invoke_stream_unknown_function_returns_404() {
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
    // The 404 envelope shape must match the rest of the API surface so
    // clients can demux uniformly.
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_found"),
        "expected error.kind=not_found, got {body}"
    );
}
