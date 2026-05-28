// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T41 coverage for the `model_not_found` rejection path.
//!
//! Asserts that an unknown OpenAI `model` identifier — both with an
//! empty model map and with a populated map that does not list the
//! requested id — surfaces as `404 Not Found` with the OpenAI envelope:
//!
//! ```json
//! { "error": {
//!     "message": "...",
//!     "type":    "invalid_request_error",
//!     "param":   "model",
//!     "code":    "model_not_found"
//! }}
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router_with_map(map: HashMap<String, Uuid>) -> axum::Router {
    let state = AppState::default().with_openai_model_map(Arc::new(map));
    build_router_with_config(
        Arc::new(state),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

fn assert_model_not_found(body: &Value) {
    assert_eq!(
        body.pointer("/error/type").and_then(Value::as_str),
        Some("invalid_request_error"),
        "type must be invalid_request_error: {body}",
    );
    assert_eq!(
        body.pointer("/error/code").and_then(Value::as_str),
        Some("model_not_found"),
        "code must be model_not_found: {body}",
    );
    assert_eq!(
        body.pointer("/error/param").and_then(Value::as_str),
        Some("model"),
        "param must be `model`: {body}",
    );
    // No native `kind` field leaking through.
    assert!(
        body.pointer("/error/kind").is_none(),
        "OpenAI envelope must NOT carry the native `kind` field: {body}",
    );
}

#[tokio::test]
async fn completions_with_empty_map_returns_404_model_not_found() {
    let router = router_with_map(HashMap::new());
    let req = json_post(
        "/v1/completions",
        json!({ "model": "anything", "prompt": "hi" }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_model_not_found(&body);
}

#[tokio::test]
async fn completions_with_populated_map_missing_model_returns_404() {
    let mut map = HashMap::new();
    // Populated with an unrelated model so we exercise the lookup miss
    // rather than the empty-map short-circuit.
    map.insert("known".to_owned(), Uuid::new_v4());
    let router = router_with_map(map);
    let req = json_post(
        "/v1/completions",
        json!({ "model": "unknown", "prompt": "hi" }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_model_not_found(&body);
}

#[tokio::test]
async fn chat_completions_with_empty_map_returns_404_model_not_found() {
    let router = router_with_map(HashMap::new());
    let req = json_post(
        "/v1/chat/completions",
        json!({
            "model": "anything",
            "messages": [{ "role": "user", "content": "hi" }],
        }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_model_not_found(&body);
}

#[tokio::test]
async fn completions_with_stale_function_id_in_map_still_returns_404() {
    // Map points at a UUID that has no matching FunctionRecord in
    // AppState. The handler must still surface `404 model_not_found`
    // (not 500 / panic) so the caller can refresh their alias.
    let stale_uuid = Uuid::new_v4();
    let mut map = HashMap::new();
    map.insert("stale".to_owned(), stale_uuid);
    let router = router_with_map(map);
    let req = json_post(
        "/v1/completions",
        json!({ "model": "stale", "prompt": "hi" }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_model_not_found(&body);
}
