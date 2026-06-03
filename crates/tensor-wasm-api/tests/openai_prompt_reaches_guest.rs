// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage that the OpenAI completions shim delivers the
//! request prompt to the guest via the `wasi:tensor/host` pull-model
//! input channel (`input-len` / `read-input`).
//!
//! Deploys a WAT module whose `_start` reads the staged input into linear
//! memory via `read-input` and echoes it straight back through
//! `emit-chunk`. The gateway buffers the emitted bytes into the
//! completion text, so a successful round-trip means the prompt the
//! caller sent came back as the completion — proving the prompt reached
//! the guest (it was dropped on the floor before this feature landed).

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
use tensor_wasm_core::types::TenantId;

/// WAT exporting `_start`. Reads the entire staged input into linear
/// memory at offset 1024 via `read-input(1024, input-len())`, then echoes
/// exactly the bytes written back out via `emit-chunk`.
const ECHO_PROMPT_WAT: &str = r#"
(module
  (import "wasi:tensor/host@0.1.0" "input-len" (func $len (result i32)))
  (import "wasi:tensor/host@0.1.0" "read-input" (func $read (param i32 i32) (result i32)))
  (import "wasi:tensor/host@0.1.0" "emit-chunk" (func $emit (param i32 i32) (result i32)))
  (memory (export "memory") 2)
  (func (export "_start")
    (local $written i32)
    (local.set $written (call $read (i32.const 1024) (call $len)))
    (drop (call $emit (i32.const 1024) (local.get $written)))
  )
)
"#;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

async fn body_json(body: Body) -> Value {
    serde_json::from_slice(&body_bytes(body).await).expect("body is JSON")
}

fn router_with_echo_model() -> axum::Router {
    let state = AppState::default();
    let wasm = wat::parse_str(ECHO_PROMPT_WAT).expect("WAT parses");
    let function_id = Uuid::parse_str("00000000-0000-4000-8000-0000000ec40a").unwrap();
    state.functions.insert(
        function_id,
        FunctionRecord {
            id: function_id,
            name: "echo-prompt".to_string(),
            wasm_bytes: Arc::from(wasm),
            created_unix_ms: 0,
            tenant_id: TenantId(0),
        },
    );
    let mut map = HashMap::new();
    map.insert("echo-model".to_owned(), function_id);
    let state = state.with_openai_model_map(Arc::new(map));
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

#[tokio::test]
async fn completions_prompt_round_trips_through_guest() {
    let router = router_with_echo_model();
    let prompt = "Say exactly this back to me, verbatim.";

    let req = json_post(
        "/v1/completions",
        json!({ "model": "echo-model", "prompt": prompt, "stream": false }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::OK, "happy path must be 200");

    let body = body_json(resp.into_body()).await;
    let text = body
        .pointer("/choices/0/text")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing choices[0].text: {body}"));
    assert_eq!(
        text, prompt,
        "the guest must echo back the prompt the gateway staged on the input channel",
    );
}

#[tokio::test]
async fn chat_completions_prompt_round_trips_through_guest() {
    let router = router_with_echo_model();

    let req = json_post(
        "/v1/chat/completions",
        json!({
            "model": "echo-model",
            "messages": [{ "role": "user", "content": "ping" }],
            "stream": false,
        }),
    );
    let resp = router.oneshot(req).await.expect("router serves");
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp.into_body()).await;
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing choices[0].message.content: {body}"));
    // The chat translator assembles a role-tagged prompt; the guest
    // echoes the whole assembled prompt, which must therefore carry the
    // user line and the trailing assistant turn marker.
    assert!(
        content.contains("user: ping"),
        "echoed completion must contain the assembled user line: {content:?}",
    );
    assert!(
        content.trim_end().ends_with("assistant:"),
        "echoed completion must carry the trailing assistant turn marker: {content:?}",
    );
}
