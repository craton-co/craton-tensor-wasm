// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Coverage tests for every error `kind` documented in `API.md`.
//!
//! These tests assert that each documented `kind` is reachable through the
//! production router and is wrapped in the standard envelope shape so the
//! public contract stays honest as the codebase evolves.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig, HEADER_TENANT};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

fn auth_router(tokens: &[&str]) -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::from_tokens(tokens.iter().copied()),
        TenantConfig::default(),
    )
}

fn require_tenant_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig {
            require_header: true,
        },
    )
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

async fn assert_envelope(body: Value, expected_kind: &str) {
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some(expected_kind),
        "expected kind={expected_kind}, got {body}"
    );
    assert!(
        body.pointer("/error/message")
            .and_then(Value::as_str)
            .is_some(),
        "envelope missing message: {body}"
    );
}

#[tokio::test]
async fn kind_invalid_json() {
    // A POST /functions body that does not parse as the expected shape.
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "invalid_json").await;
}

#[tokio::test]
async fn kind_invalid_name() {
    // 8-byte minimal Wasm header. The name check runs before the wasm
    // bytes are inspected, so any value here is fine — empty/whitespace
    // name short-circuits.
    let req = json_post(
        "/functions",
        json!({ "name": "   ", "wasm_b64": BASE64.encode(b"\x00asm\x01\x00\x00\x00") }),
    );
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "invalid_name").await;
}

#[tokio::test]
async fn kind_invalid_wasm_short_payload() {
    // A 4-byte payload — too short to validate. Surfaced as invalid_wasm.
    let req = json_post(
        "/functions",
        json!({ "name": "tiny", "wasm_b64": BASE64.encode([0u8; 4]) }),
    );
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "invalid_wasm").await;
}

#[tokio::test]
async fn kind_invalid_wasm_bad_magic() {
    // Eight bytes but the magic header is wrong: wasmparser rejects.
    let req = json_post(
        "/functions",
        json!({ "name": "bogus", "wasm_b64": BASE64.encode([0x01u8; 16]) }),
    );
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "invalid_wasm").await;
}

#[tokio::test]
async fn kind_unauthorized() {
    // `/healthz` and `/metrics` are intentionally unauthenticated (see
    // `openapi/tensor-wasm-api.yaml`, `security: []`), so we drive a
    // protected route to exercise the bearer-auth layer.
    let router = auth_router(&["good"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{}", Uuid::nil()))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_envelope(body_json(resp.into_body()).await, "unauthorized").await;
}

#[tokio::test]
async fn kind_missing_tenant() {
    // Same rationale as `kind_unauthorized` above: the tenant-scope layer
    // only runs on protected routes after Fix 2.
    let router = require_tenant_router();
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{}", Uuid::nil()))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "missing_tenant").await;
}

#[tokio::test]
async fn kind_not_found() {
    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/functions/{}", Uuid::new_v4()))
        .body(Body::empty())
        .unwrap();
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_envelope(body_json(resp.into_body()).await, "not_found").await;
}

// NOTE: `invoke_timeout`, `wasmtime`, and `internal` envelope kinds are
// hard to trigger from an end-to-end HTTP test without bringing up a
// long-running guest that exhausts its deadline (slow on CI) or
// monkey-patching internal state. Their `ApiError`/`ExecError` mappings
// are covered by unit tests in `crate::routes::tests` —
// `exec_error_timeout_maps_to_invoke_timeout` and
// `api_error_constructors_carry_status`. Adding HTTP-driven coverage
// here would require a contrived module specifically designed to hang;
// the routes-layer assertions are the appropriate seam.

#[tokio::test]
async fn kind_tenant_header_invalid() {
    // Bad tenant value (non-u64) surfaces as missing_tenant 400 — the
    // other branch of the same envelope. Driven against a protected route
    // because `/healthz` no longer flows through `tenant_scope` (see
    // `openapi/tensor-wasm-api.yaml`, `security: []`).
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{}", Uuid::nil()))
        .header(HEADER_TENANT, "abc")
        .body(Body::empty())
        .unwrap();
    let resp = dev_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_envelope(body_json(resp.into_body()).await, "missing_tenant").await;
}

#[tokio::test]
async fn kind_body_too_large() {
    // 100 MiB > 64 MiB cap. tower_http returns 413 Payload Too Large.
    let huge: Vec<u8> = vec![b'a'; 100 * 1024 * 1024];
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(huge))
        .unwrap();
    let resp = dev_router().oneshot(req).await.unwrap();
    // tower_http renders an empty body with status 413 by default. We
    // accept either the bare 413 or our envelope variant.
    let status = resp.status();
    assert!(
        status == StatusCode::PAYLOAD_TOO_LARGE || status == StatusCode::BAD_REQUEST,
        "got {status}"
    );
}
