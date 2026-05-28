// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the OpenAI-compatible inference gateway shim
//! (B4.9 scaffold).
//!
//! v0.3.5 ships `POST /v1/completions` and `POST /v1/chat/completions` as
//! scaffolds that return `501 Not Implemented` with the OpenAI-shape error
//! envelope (`{ "error": { "message", "type", "param", "code" } }`). v0.4
//! wires the actual `model` → deployed-function translation. These tests
//! pin the wire surface so the v0.4 implementation cannot accidentally
//! reshape the error envelope or change the URL surface.
//!
//! Coverage:
//!
//! 1. `POST /v1/completions` with a well-formed OpenAI body → `501` and
//!    `error.code == "openai_not_yet_wired"`.
//! 2. `POST /v1/chat/completions` with a well-formed OpenAI body → `501`
//!    same shape.
//! 3. `POST /v1/completions` with malformed JSON → `400` and the OpenAI
//!    error envelope shape (NOT the native `{ error: { kind, message } }`
//!    shell — OpenAI SDKs will not parse the native shape).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

/// Build a dev-mode router exactly as the OpenAPI test does so we exercise
/// the same composition production callers see, minus env reads.
fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body collects").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body parses as JSON")
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

/// Assert the response body matches the OpenAI envelope shape:
/// `{ "error": { "message": "...", "type": "...", "param": ?, "code": ? } }`.
fn assert_openai_envelope(body: &Value, expected_code: &str) {
    let inner = body
        .get("error")
        .unwrap_or_else(|| panic!("response missing `error` key: {body}"));
    assert!(
        inner.get("message").and_then(Value::as_str).is_some(),
        "envelope missing `message`: {body}",
    );
    assert!(
        inner.get("type").and_then(Value::as_str).is_some(),
        "envelope missing `type`: {body}",
    );
    // `param` must be present (possibly null). Same for `code`.
    assert!(
        inner.get("param").is_some(),
        "envelope missing `param` key (must be present even when null): {body}",
    );
    assert_eq!(
        inner.get("code").and_then(Value::as_str),
        Some(expected_code),
        "expected error.code={expected_code}, got body {body}",
    );
}

#[tokio::test]
async fn completions_valid_body_returns_501_not_yet_wired() {
    let payload = json!({
        "model": "gpt-3.5-turbo",
        "prompt": "Once upon a time",
        "max_tokens": 16,
        "temperature": 0.7,
    });
    let resp = dev_router()
        .oneshot(json_post("/v1/completions", payload))
        .await
        .expect("router serves /v1/completions");
    assert_eq!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "scaffold must return 501 for /v1/completions",
    );
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "openai_not_yet_wired");
    // Pin the OpenAI `type` value too so the v0.4 wiring step can be
    // distinguished from this scaffold response by clients that branch on
    // either field.
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("not_implemented"),
    );
}

#[tokio::test]
async fn chat_completions_valid_body_returns_501_not_yet_wired() {
    let payload = json!({
        "model": "gpt-4",
        "messages": [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Hello!" },
        ],
        "max_tokens": 64,
    });
    let resp = dev_router()
        .oneshot(json_post("/v1/chat/completions", payload))
        .await
        .expect("router serves /v1/chat/completions");
    assert_eq!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "scaffold must return 501 for /v1/chat/completions",
    );
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "openai_not_yet_wired");
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("not_implemented"),
    );
}

#[tokio::test]
async fn completions_malformed_body_returns_400_openai_envelope() {
    // Body is not valid JSON. The handler accepts the payload via
    // `Result<Json<_>, JsonRejection>`, so the malformed body is
    // converted into a 400 with the OpenAI envelope (NOT the native
    // `{ error: { kind, message } }` shell that the rest of the API
    // emits) — see `crates/tensor-wasm-api/src/openai.rs`.
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("this is not json"))
        .expect("request builds");
    let resp = dev_router()
        .oneshot(req)
        .await
        .expect("router serves /v1/completions");
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "malformed body must surface as 400, not 501",
    );
    let body = body_json(resp.into_body()).await;
    // OpenAI envelope shape. `code` should be the invalid-request code,
    // not the not-yet-wired code, so clients can tell the two apart.
    assert!(
        body.get("error").is_some(),
        "400 must use the OpenAI envelope, not the native one: {body}",
    );
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("invalid_request_error"),
    );
    // No `kind` field — that would be the native envelope leaking.
    assert!(
        body.pointer("/error/kind").is_none(),
        "400 response must NOT use the native `kind` field: {body}",
    );
}

#[tokio::test]
async fn chat_completions_malformed_body_returns_400_openai_envelope() {
    // Symmetry check: the chat endpoint follows the same rejection path.
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{ this is not valid json"))
        .expect("request builds");
    let resp = dev_router()
        .oneshot(req)
        .await
        .expect("router serves /v1/chat/completions");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("invalid_request_error"),
    );
}

#[tokio::test]
async fn completions_minimal_body_still_returns_501() {
    // Every field is `#[serde(default)]` on the request struct so an
    // empty object must parse and reach the 501 — not get rejected at the
    // extractor. This pins the "fields are optional at the wire layer"
    // contract documented in the OpenAPI yaml.
    let resp = dev_router()
        .oneshot(json_post("/v1/completions", json!({})))
        .await
        .expect("router serves /v1/completions");
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "openai_not_yet_wired");
}

#[tokio::test]
async fn chat_completions_empty_messages_array_returns_501() {
    // The scaffold does not validate semantics; an empty `messages` array
    // still parses. The v0.4 wiring step will tighten validation; this
    // test is allowed to start failing then (replace with a 400
    // assertion).
    let resp = dev_router()
        .oneshot(json_post(
            "/v1/chat/completions",
            json!({ "model": "m", "messages": [] }),
        ))
        .await
        .expect("router serves /v1/chat/completions");
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}
