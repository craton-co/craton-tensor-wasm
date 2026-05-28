// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage for argument passing through `POST /functions/{id}/invoke`.
//!
//! Deploys an `(i32, i32) -> i32` adder built inline from WAT, posts
//! `{"export": "add", "args": [1, 2]}`, and asserts the response carries
//! `result: 3`. Complements the executor-level `call_export_with_args`
//! coverage by exercising the full HTTP transport — request parsing,
//! WasmArg conversion, executor call, response shaping — so a regression
//! in any of those layers surfaces here.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

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

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

async fn deploy_adder(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (func (export "add") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add)
        )
        "#,
    )
    .expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = json_post(
        "/functions",
        json!({ "name": "adder", "wasm_b64": wasm_b64 }),
    );
    let resp = router.clone().oneshot(req).await.expect("deploy oneshot");
    assert_eq!(resp.status(), StatusCode::OK, "deploy failed");
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("deploy response has id")
}

#[tokio::test]
async fn invoke_with_args_returns_sum() {
    let router = router();
    let id = deploy_adder(&router).await;

    let body = json!({ "export": "add", "args": [1, 2] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;

    // The handler returns the result list verbatim for non-empty arrays;
    // for a single-i32 export this is `[3]`. Unwrap one level and check
    // the integer.
    let result = body.get("result").expect("result field present");
    let arr = result.as_array().expect("result is array");
    assert_eq!(arr.len(), 1, "expected single-element result; got {arr:?}");
    assert_eq!(
        arr[0].as_i64(),
        Some(3),
        "adder returned wrong value: {body}"
    );
}

#[tokio::test]
async fn invoke_with_args_rejects_non_numeric() {
    let router = router();
    let id = deploy_adder(&router).await;

    // String element → `400 invalid_args`. The error envelope's kind
    // is stable; the message includes the offending index and value.
    let body = json!({ "export": "add", "args": ["one", 2] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_args"),
    );
}

#[tokio::test]
async fn invoke_without_export_falls_back_to_start() {
    // Body omits `export` entirely → defaults to `_start` → `main`
    // discovery. Deploy a WASI command and confirm the legacy path
    // still works after wiring args through.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let router = router();
    let deploy = router
        .clone()
        .oneshot(json_post(
            "/functions",
            json!({ "name": "wasi_cmd", "wasm_b64": wasm_b64 }),
        ))
        .await
        .expect("deploy");
    assert_eq!(deploy.status(), StatusCode::OK);
    let id = body_json(deploy.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("id");

    let resp = router
        .oneshot(json_post(&format!("/functions/{id}/invoke"), json!({})))
        .await
        .expect("invoke");
    assert_eq!(resp.status(), StatusCode::OK);
    // Empty result list collapses to the literal `"ok"` for back-compat.
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("result").and_then(Value::as_str),
        Some("ok"),
        "expected legacy 'ok' envelope: {body}"
    );
}
