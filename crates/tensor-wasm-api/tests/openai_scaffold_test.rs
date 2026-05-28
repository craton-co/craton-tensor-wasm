// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the OpenAI-compatible inference gateway shim
//! after T41 wired translation through the internal invoke protocol.
//!
//! v0.3.5 shipped `POST /v1/completions` and `POST /v1/chat/completions`
//! as scaffolds returning `501 Not Implemented`. T41 (v0.4) replaced the
//! 501 path with a real translation step: an empty model map yields
//! `404 model_not_found`; a populated map dispatches the call.
//!
//! Coverage in this file pins the wire-format invariants that survived
//! the rewire — the OpenAI error envelope shape, the malformed-body
//! rejection path, the optional-field schema — without taking a
//! dependency on the executor (the happy path lives in the new
//! `openai_completions_*` integration tests).
//!
//! Coverage:
//!
//! 1. Unknown model on `POST /v1/completions` → `404 model_not_found`
//!    with the OpenAI envelope.
//! 2. Same on `POST /v1/chat/completions`.
//! 3. Malformed JSON on either endpoint → `400 invalid_request_error`
//!    with the OpenAI envelope.
//! 4. Empty body still parses (no required fields) and reaches the
//!    model-resolution step.

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
async fn completions_unknown_model_returns_404_model_not_found() {
    // Default dev router carries an empty `openai_model_map`; every
    // model id misses and the handler must surface
    // `404 model_not_found` with the OpenAI envelope (`type:
    // invalid_request_error`, `code: model_not_found`).
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
        StatusCode::NOT_FOUND,
        "unknown model must surface as 404 model_not_found",
    );
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "model_not_found");
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("invalid_request_error"),
    );
    assert_eq!(
        body.pointer("/error/param").and_then(Value::as_str),
        Some("model"),
    );
}

#[tokio::test]
async fn chat_completions_unknown_model_returns_404_model_not_found() {
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
        StatusCode::NOT_FOUND,
        "unknown model must surface as 404 model_not_found on /chat too",
    );
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "model_not_found");
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("invalid_request_error"),
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
        "malformed body must surface as 400, not 404 / 501",
    );
    let body = body_json(resp.into_body()).await;
    // OpenAI envelope shape. `code` should be the invalid-request code,
    // not the model_not_found code, so clients can tell the two apart.
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
async fn completions_minimal_body_reaches_model_resolution() {
    // Every field is `#[serde(default)]` on the request struct so an
    // empty object must parse and reach model resolution — not get
    // rejected at the extractor. With an empty model map this surfaces
    // as `404 model_not_found` (the empty `model` field is also
    // unknown).
    let resp = dev_router()
        .oneshot(json_post("/v1/completions", json!({})))
        .await
        .expect("router serves /v1/completions");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_openai_envelope(&body, "model_not_found");
}

#[tokio::test]
async fn chat_completions_empty_messages_array_reaches_model_resolution() {
    // The handler does not gate on `messages` cardinality; an empty
    // array still reaches the model-resolution step. With an empty map
    // this surfaces as 404 model_not_found.
    let resp = dev_router()
        .oneshot(json_post(
            "/v1/chat/completions",
            json!({ "model": "m", "messages": [] }),
        ))
        .await
        .expect("router serves /v1/chat/completions");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
