// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T41 end-to-end coverage for `POST /v1/completions` (non-streaming).
//!
//! Deploys a WAT module that emits the bytes "hello" via
//! `wasi:tensor/host.emit-chunk`, configures the gateway's
//! `openai_model_map` to point a model alias at the deployed function,
//! POSTs an OpenAI completions request, and asserts the response body
//! matches the documented OpenAI `text_completion` envelope:
//!
//! ```json
//! {
//!   "id": "cmpl-<uuid>",
//!   "object": "text_completion",
//!   "created": <unix-seconds>,
//!   "model": "<echoed>",
//!   "choices": [{ "text": "hello", "index": 0, "finish_reason": "stop", "logprobs": null }],
//!   "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, FunctionRecord, TenantConfig,
};

/// WAT exporting `_start` and calling `wasi:tensor/host.emit-chunk`
/// once with the payload "hello".
const HELLO_EMIT_WAT: &str = r#"
(module
  (import "wasi:tensor/host" "emit-chunk"
    (func $emit (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "hello")
  (func (export "_start")
    (drop (call $emit (i32.const 0) (i32.const 5)))
  )
)
"#;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect().await.expect("collect body").to_bytes().to_vec()
}

async fn body_json(body: Body) -> Value {
    let bytes = body_bytes(body).await;
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Build an AppState that pre-deploys the "hello" emitter under a
/// known UUID and wires `openai_model_map["hello-model"]` to it.
fn router_with_model() -> (axum::Router, Uuid) {
    let state = AppState::default();
    let wasm = wat::parse_str(HELLO_EMIT_WAT).expect("WAT parses");
    let function_id = Uuid::parse_str("00000000-0000-4000-8000-00000000abcd").unwrap();
    state.functions.insert(
        function_id,
        FunctionRecord {
            id: function_id,
            name: "hello-emitter".to_string(),
            wasm_bytes: Arc::from(wasm),
            created_unix_ms: 0,
        },
    );
    let mut map = HashMap::new();
    map.insert("hello-model".to_owned(), function_id);
    let state = state.with_openai_model_map(Arc::new(map));
    let router = build_router_with_config(
        Arc::new(state),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    (router, function_id)
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

#[tokio::test]
async fn completions_returns_openai_text_completion_envelope() {
    let (router, _id) = router_with_model();

    let req = json_post(
        "/v1/completions",
        json!({
            "model": "hello-model",
            "prompt": "Say hello",
            "stream": false,
        }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::OK, "happy path must be 200");

    let body = body_json(resp.into_body()).await;

    // Top-level shape: id, object, created, model, choices, usage.
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing `id`: {body}"));
    assert!(id.starts_with("cmpl-"), "id must start with cmpl-: {id}");
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("text_completion"),
        "object must be text_completion, got {body}",
    );
    assert!(
        body.get("created").and_then(Value::as_u64).is_some(),
        "created must be a u64 unix timestamp: {body}",
    );
    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some("hello-model"),
        "model must echo the request: {body}",
    );

    // choices[0].text == "hello"
    let choices = body
        .get("choices")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("missing choices array: {body}"));
    assert_eq!(choices.len(), 1, "expected one choice, got {choices:?}");
    let choice = &choices[0];
    assert_eq!(
        choice.get("text").and_then(Value::as_str),
        Some("hello"),
        "expected `text: \"hello\"`, got {choice}",
    );
    assert_eq!(choice.get("index").and_then(Value::as_u64), Some(0));
    assert_eq!(
        choice.get("finish_reason").and_then(Value::as_str),
        Some("stop"),
    );

    // usage block (zeros until v0.5 wires a tokenizer).
    let usage = body
        .get("usage")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("missing usage object: {body}"));
    assert_eq!(usage.get("prompt_tokens").and_then(Value::as_u64), Some(0));
    assert_eq!(
        usage.get("completion_tokens").and_then(Value::as_u64),
        Some(0),
    );
    assert_eq!(usage.get("total_tokens").and_then(Value::as_u64), Some(0));
}

#[tokio::test]
async fn chat_completions_returns_openai_chat_envelope() {
    let (router, _id) = router_with_model();

    let req = json_post(
        "/v1/chat/completions",
        json!({
            "model": "hello-model",
            "messages": [
                { "role": "user", "content": "say hello" }
            ],
            "stream": false,
        }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp.into_body()).await;

    let id = body
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing id: {body}"));
    assert!(
        id.starts_with("chatcmpl-"),
        "id must start with chatcmpl-: {id}",
    );
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("chat.completion"),
    );

    // choices[0].message = { role: "assistant", content: "hello" }
    let choice = body
        .pointer("/choices/0")
        .unwrap_or_else(|| panic!("missing choices[0]: {body}"));
    let message = choice
        .get("message")
        .unwrap_or_else(|| panic!("missing message: {choice}"));
    assert_eq!(
        message.get("role").and_then(Value::as_str),
        Some("assistant"),
    );
    assert_eq!(
        message.get("content").and_then(Value::as_str),
        Some("hello"),
    );
    assert_eq!(
        choice.get("finish_reason").and_then(Value::as_str),
        Some("stop"),
    );
}
