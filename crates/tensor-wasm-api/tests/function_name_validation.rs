// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Boundary tests for `POST /functions` `name` validation (api S-10).
//!
//! Without per-field bounds a caller could submit a multi-MiB `name` and
//! pin the bytes in the in-memory `FunctionRecord` registry, where the
//! field is echoed back on every read. The handler now rejects empty /
//! whitespace-only names, names longer than `MAX_FUNCTION_NAME_BYTES`
//! (256), and names carrying control characters, each with
//! `400 invalid_name`.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;
use uuid::Uuid;

/// Minimal but legal empty Wasm module: `\0asm` magic + version 1, no
/// sections. Exactly 8 bytes — matches the constant used by
/// `tests/http_integration.rs`.
const WASM_MIN_MODULE: [u8; 8] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

async fn body_json(body: Body) -> Value {
    let bytes = body
        .collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
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
        .unwrap()
}

async fn assert_invalid_name(name: Value) {
    let wasm_b64 = BASE64.encode(WASM_MIN_MODULE);
    let req = json_post("/functions", json!({ "name": name, "wasm_b64": wasm_b64 }));
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "expected 400 for name {name:?}"
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_name"),
        "got {body}"
    );
}

#[tokio::test]
async fn empty_name_is_rejected() {
    assert_invalid_name(json!("")).await;
}

#[tokio::test]
async fn whitespace_only_name_is_rejected() {
    assert_invalid_name(json!("  ")).await;
}

#[tokio::test]
async fn oversized_name_is_rejected() {
    // 257 ASCII bytes — one over the 256-byte cap. ASCII so `.len()` (byte
    // length) lines up with the character count, exercising the boundary
    // exactly.
    let name = "a".repeat(257);
    assert_invalid_name(json!(name)).await;
}

#[tokio::test]
async fn embedded_nul_name_is_rejected() {
    // U+0000 NUL is a control character; the validator rejects any
    // `char::is_control()` codepoint, of which NUL is the canonical case.
    assert_invalid_name(json!("hello\u{0}world")).await;
}

#[tokio::test]
async fn well_formed_name_is_accepted() {
    // The happy-path counterpart to the four rejection tests above: a
    // plain ASCII name paired with the minimal valid Wasm module from
    // `tests/http_integration.rs` produces a 200 with a UUID body.
    let wasm_b64 = BASE64.encode(WASM_MIN_MODULE);
    let req = json_post(
        "/functions",
        json!({ "name": "ok_name", "wasm_b64": wasm_b64 }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "expected 200 OK");
    let body = body_json(resp.into_body()).await;
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .expect("id field present and string");
    Uuid::parse_str(id).expect("id is a valid UUID");
}
