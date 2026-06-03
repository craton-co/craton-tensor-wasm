// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! In-process HTTP integration tests driving the tensor-wasm-api router via
//! `tower::ServiceExt::oneshot`. No real socket is bound.

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
/// sections. Exactly 8 bytes — `wasmparser::validate` accepts this as a
/// well-formed module with no functions, memories, or other declarations.
const WASM_MIN_MODULE: [u8; 8] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

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
    // Use the explicit-config builder so tests do not depend on the
    // ambient process environment (`TENSOR_WASM_API_TOKENS`, etc.). Empty token
    // list = dev mode (auth disabled).
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

#[tokio::test]
async fn healthz_returns_ok() {
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body, json!({ "status": "ok" }));
}

#[tokio::test]
async fn metrics_returns_text() {
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/plain"), "content-type was {ct}");
    let bytes = body_bytes(resp.into_body()).await;
    let text = std::str::from_utf8(&bytes).unwrap();
    // Real Prometheus exposition from the shared TensorWasmMetrics registry.
    // Counters are registered without the `_total` suffix in tensor-wasm-core; the
    // prometheus-client text encoder appends it per the OpenMetrics rules.
    for needle in [
        "tensor_wasm_active_instances",
        "tensor_wasm_kernel_dispatches_total",
        "tensor_wasm_offload_success_total",
    ] {
        assert!(text.contains(needle), "missing {needle} in:\n{text}");
    }
}

#[tokio::test]
async fn create_function_with_valid_wasm() {
    let wasm_b64 = BASE64.encode(WASM_MIN_MODULE);
    let req = json_post(
        "/functions",
        json!({ "name": "hello", "wasm_b64": wasm_b64 }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .expect("id field present and string");
    Uuid::parse_str(id).expect("id is a valid UUID");
}

#[tokio::test]
async fn create_function_rejects_short_payload() {
    // < 8 bytes triggers the early length check, surfaced as invalid_wasm.
    let req = json_post(
        "/functions",
        json!({ "name": "tiny", "wasm_b64": BASE64.encode([0u8; 4]) }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_wasm"),
        "got {body}"
    );
}

#[tokio::test]
async fn create_function_rejects_non_wasm() {
    // 16 bytes with a bogus magic. `wasmparser::validate` rejects with
    // invalid_wasm.
    let req = json_post(
        "/functions",
        json!({ "name": "bogus", "wasm_b64": BASE64.encode([0x01u8; 16]) }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_wasm"),
        "got {body}"
    );
}

#[tokio::test]
async fn create_function_rejects_invalid_base64() {
    let req = json_post(
        "/functions",
        // `!` is not in the standard base64 alphabet, padded or not.
        json!({ "name": "garbage", "wasm_b64": "not!valid!!base64!!" }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_base64"),
        "got {body}"
    );
}

#[tokio::test]
async fn delete_unknown_returns_404() {
    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/functions/{}", Uuid::new_v4()))
        .body(Body::empty())
        .unwrap();
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_found")
    );
}

#[tokio::test]
async fn invoke_unknown_returns_404() {
    let req = json_post(&format!("/functions/{}/invoke", Uuid::new_v4()), json!({}));
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_found")
    );
}

#[tokio::test]
async fn deploy_then_invoke_round_trip() {
    // A minimal WASI command: a module that exports `_start` taking no args
    // and returning nothing. `TensorWasmExecutor` will spawn, call, and terminate.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    let router = router();

    // Deploy.
    let deploy_req = json_post(
        "/functions",
        json!({ "name": "round_trip", "wasm_b64": wasm_b64 }),
    );
    let deploy_resp = router
        .clone()
        .oneshot(deploy_req)
        .await
        .expect("deploy oneshot");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let deploy_body = body_json(deploy_resp.into_body()).await;
    let id = deploy_body
        .get("id")
        .and_then(Value::as_str)
        .expect("deploy response has id");

    // Invoke.
    let invoke_req = json_post(&format!("/functions/{id}/invoke"), json!({}));
    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    assert_eq!(
        invoke_resp.status(),
        StatusCode::OK,
        "expected 200, got {:?}",
        invoke_resp.status()
    );
    let body = body_json(invoke_resp.into_body()).await;
    assert!(
        body.get("result").is_some(),
        "response missing `result`: {body}"
    );
    assert_eq!(
        body.get("function_id").and_then(Value::as_str),
        Some(id),
        "function_id mismatch: {body}"
    );
}

#[tokio::test]
async fn invoke_module_without_start_or_main_is_400() {
    // A module whose only export has an unrelated name — neither `_start`
    // nor `main` is present. The executor's MissingExport bubbles up as a
    // 400 with `kind = "missing_export"`.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "noop")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    let router = router();

    let deploy_req = json_post(
        "/functions",
        json!({ "name": "no_entry", "wasm_b64": wasm_b64 }),
    );
    let deploy_resp = router
        .clone()
        .oneshot(deploy_req)
        .await
        .expect("deploy oneshot");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("deploy response has id");

    let invoke_req = json_post(&format!("/functions/{id}/invoke"), json!({}));
    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    assert_eq!(invoke_resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(invoke_resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("missing_export"),
        "got {body}"
    );
}

#[tokio::test]
async fn get_unknown_job_returns_404() {
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{}", Uuid::new_v4()))
        .body(Body::empty())
        .unwrap();
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_found"),
        "got {body}"
    );
}
