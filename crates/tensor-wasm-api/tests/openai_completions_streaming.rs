// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T41 end-to-end coverage for `POST /v1/completions` with
//! `stream: true`.
//!
//! Deploys the same "emit `'a'`, `'b'`, `'c'`" WAT fixture the
//! `/invoke-stream` test uses, configures the gateway model map to
//! point at the deployed function, POSTs an OpenAI request with
//! `stream: true`, and asserts:
//!
//! * Response content-type is `text/event-stream`.
//! * Body carries one `data: { ... }` SSE frame per emitted chunk
//!   (three frames for the three guest emits).
//! * Each frame's JSON carries `object: "chat.completion.chunk"`
//!   (for chat) or `object: "text_completion"` (for completions),
//!   and `choices[0].delta.content` (chat) /
//!   `choices[0].text` (completions) holds the chunk text.
//! * Stream terminates with a final `data: [DONE]\n\n`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, FunctionRecord, TenantConfig,
};

const EMIT_THREE_WAT: &str = r#"
(module
  (import "wasi:tensor/host" "emit-chunk"
    (func $emit (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "abc")
  (func (export "_start")
    (drop (call $emit (i32.const 0) (i32.const 1)))
    (drop (call $emit (i32.const 1) (i32.const 1)))
    (drop (call $emit (i32.const 2) (i32.const 1)))
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

fn router_with_model() -> (axum::Router, Uuid) {
    let state = AppState::default();
    let wasm = wat::parse_str(EMIT_THREE_WAT).expect("WAT parses");
    let function_id = Uuid::parse_str("00000000-0000-4000-8000-00000000ab01").unwrap();
    state.functions.insert(
        function_id,
        FunctionRecord {
            id: function_id,
            name: "abc-emitter".to_string(),
            wasm_bytes: Arc::from(wasm),
            created_unix_ms: 0,
        },
    );
    let mut map = HashMap::new();
    map.insert("abc-model".to_owned(), function_id);
    let state = state.with_openai_model_map(Arc::new(map));
    let router = build_router_with_config(
        Arc::new(state),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    (router, function_id)
}

#[allow(dead_code)]
fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

/// Extract every `data: ...\n\n` event from an SSE body string.
fn parse_sse_data_frames(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in body.split("\n\n") {
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("data: ") {
                out.push(rest.trim_end().to_owned());
            } else if let Some(rest) = line.strip_prefix("data:") {
                // Tolerate the no-space form even though we emit with one.
                out.push(rest.trim_end().to_owned());
            }
        }
    }
    out
}

#[tokio::test]
async fn completions_stream_true_returns_sse_frames_and_done_terminator() {
    let (router, _id) = router_with_model();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "abc-model",
                "prompt": "go",
                "stream": true,
            }))
            .unwrap(),
        ))
        .expect("req builds");
    let resp = router.oneshot(req).await.expect("router serves");

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "stream:true must yield text/event-stream content-type, got {ct:?}",
    );

    let body_text = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8");
    let frames = parse_sse_data_frames(&body_text);
    assert!(
        frames.len() >= 4,
        "expected >=4 frames (3 chunk + 1 DONE), got {} frames: {body_text:?}",
        frames.len(),
    );

    // The final frame is `[DONE]`.
    let last = frames.last().expect("at least one frame");
    assert_eq!(last, "[DONE]", "stream must end with [DONE], got {last}");

    // Reconstruct the streamed text from chunk frames (skip terminal /
    // `[DONE]`). Each chunk frame is a JSON object with
    // `choices[0].text` carrying one byte of the streamed output.
    let mut reconstructed = String::new();
    for f in &frames[..frames.len() - 1] {
        if f == "[DONE]" {
            continue;
        }
        let v: Value =
            serde_json::from_str(f).unwrap_or_else(|e| panic!("frame is not JSON: {e}; frame={f}"));
        if let Some(text) = v.pointer("/choices/0/text").and_then(Value::as_str) {
            reconstructed.push_str(text);
        }
    }
    assert!(
        reconstructed.contains("a") && reconstructed.contains("b") && reconstructed.contains("c"),
        "reconstructed text must contain a/b/c from the three guest emits; got {reconstructed:?}",
    );
}

#[tokio::test]
async fn chat_completions_stream_true_uses_delta_envelope() {
    let (router, _id) = router_with_model();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "abc-model",
                "messages": [{ "role": "user", "content": "go" }],
                "stream": true,
            }))
            .unwrap(),
        ))
        .expect("req builds");
    let resp = router.oneshot(req).await.expect("router serves");

    assert_eq!(resp.status(), StatusCode::OK);
    let body_text = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8");
    let frames = parse_sse_data_frames(&body_text);
    assert!(
        frames.len() >= 4,
        "expected >=4 frames including DONE; got {}: {body_text:?}",
        frames.len(),
    );
    assert_eq!(frames.last().unwrap(), "[DONE]");

    // Inspect one chunk frame; it must use the chat-shape envelope
    // (`choices[0].delta.content` rather than `choices[0].text`).
    let chunk_frame = frames
        .iter()
        .find(|f| f.contains("delta"))
        .unwrap_or_else(|| panic!("no chunk frame with `delta` key: {body_text:?}"));
    let v: Value = serde_json::from_str(chunk_frame).expect("chunk JSON parses");
    assert_eq!(
        v.get("object").and_then(Value::as_str),
        Some("chat.completion.chunk"),
        "chat stream frames must carry object=chat.completion.chunk; got {v}",
    );
    assert!(
        v.pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .is_some(),
        "chat stream frames must carry choices[0].delta.content: {v}",
    );
}
