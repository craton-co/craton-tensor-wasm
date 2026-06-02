// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the end-to-end `/invoke-stream` wiring
//! landed by T34 (roadmap feature #2, v0.4).
//!
//! Builds a minimal WAT module that imports
//! `wasi:tensor/host.emit-chunk` and calls it three times with the
//! payloads `"a"`, `"b"`, `"c"` from linear memory, then returns. The
//! gateway-side handler should surface those three calls as three
//! `event: chunk` SSE frames followed by a single `event: done` frame.
//!
//! The scaffold tests at `tests/streaming_invoke_scaffold.rs` and
//! `tests/invoke_stream_restored.rs` were retargeted in the same
//! landing to assert the new `event: chunk` / `event: done` shape
//! instead of the v0.3.7 `event: scaffold` placeholder.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};

/// WAT module exporting `_start` and importing
/// `wasi:tensor/host.emit-chunk`. The `_start` body writes three
/// single-byte payloads `'a'`, `'b'`, `'c'` into linear memory at
/// offsets 0/1/2 and invokes `emit-chunk(ptr, 1)` once per byte.
///
/// `(param i32 i32) (result i32)` matches the WIT lowering of
/// `emit-chunk: func(bytes: list<u8>) -> s32` — a `list<u8>` lowers to
/// a `(ptr, len)` pair on the host ABI surface registered by
/// [`tensor_wasm_wasi_gpu::streaming::add_streaming_to_linker`].
const EMIT_THREE_WAT: &str = r#"
(module
  (import "wasi:tensor/host@0.1.0" "emit-chunk"
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

async fn deploy_emit_module(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(EMIT_THREE_WAT).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "emit_three", "wasm_b64": wasm_b64 })).unwrap(),
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
async fn emit_chunk_three_times_surfaces_three_sse_chunk_events() {
    let router = router();
    let id = deploy_emit_module(&router).await;

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

    // Three `event: chunk` lines must appear, in order: a, b, c.
    let chunk_lines: Vec<&str> = body
        .split("\n\n")
        .filter(|frame| frame.contains("event: chunk"))
        .collect();
    assert_eq!(
        chunk_lines.len(),
        3,
        "expected 3 chunk frames; got body:\n{body}"
    );
    // The first frame's data should be "a", second "b", third "c".
    for (i, expected) in ["a", "b", "c"].iter().enumerate() {
        let frame = chunk_lines[i];
        let data_line = frame
            .lines()
            .find(|l| l.starts_with("data:"))
            .unwrap_or_else(|| panic!("no data: line in frame {i}:\n{frame}"));
        let payload = data_line.trim_start_matches("data:").trim();
        assert_eq!(
            payload, *expected,
            "chunk {i} payload mismatch in body:\n{body}"
        );
    }

    // The terminal `event: done` frame must follow with `status: ok`.
    assert!(
        body.contains("event: done"),
        "expected terminal `event: done` frame in body:\n{body}"
    );
    assert!(
        body.contains("\"status\":\"ok\""),
        "expected `status: ok` in terminal frame in body:\n{body}"
    );
}

#[tokio::test]
async fn emit_chunk_three_times_surfaces_three_chunks_on_octet_branch() {
    // Chunked-transfer fallback: no SSE prefix, raw bytes for each
    // chunk, then a final `event: done` line for end-of-stream.
    let router = router();
    let id = deploy_emit_module(&router).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        // No `Accept` header → chunked-transfer branch.
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
    // The chunked branch concatenates raw chunk bytes back-to-back —
    // "abc" — followed by the terminal SSE-style done line.
    assert!(
        body.starts_with("abc"),
        "expected chunked body to start with 'abc', got: {body:?}"
    );
    assert!(
        body.contains("event: done"),
        "expected terminal `event: done` line in chunked body: {body:?}"
    );
}
