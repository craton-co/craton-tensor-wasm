// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Body-handling coverage for `/invoke` and `/invoke-async`.
//!
//! Historically (api S-31) both handlers refused to parse the body at all,
//! to avoid a wasted-CPU DoS surface. With the args-passing wiring landed
//! the handlers now accept an [`InvokeRequest`] body (`{export?, args?}`)
//! but keep the back-compat surface intact:
//!
//! * an absent / empty body still works (treated as defaults);
//! * unknown fields are tolerated (no `deny_unknown_fields`);
//! * malformed JSON is rejected synchronously with `400 invalid_json`,
//!   matching the standard wire envelope.
//!
//! These tests pin all three behaviours so a future regression that
//! reintroduces `_args: Result<Json<serde_json::Value>, JsonRejection>`
//! (or the opposite — a strict schema that breaks legacy clients) is
//! caught at PR time.

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

/// Deploy a minimal WASI command (exports `_start`) and return its id.
async fn deploy_min_module(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = json_post(
        "/functions",
        json!({ "name": "invoke_ignores_body", "wasm_b64": wasm_b64 }),
    );
    let deploy_resp = router
        .clone()
        .oneshot(deploy_req)
        .await
        .expect("deploy oneshot");
    assert_eq!(deploy_resp.status(), StatusCode::OK, "deploy failed");
    body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("deploy response has id")
}

/// A 1 KiB body whose top-level fields are *unknown* to [`InvokeRequest`]
/// (no `export`, no `args`) must be accepted: serde tolerates unknown
/// fields by default, so the handler proceeds with all-defaults and runs
/// the deployed `_start`. Pre-args this returned 200 because the body
/// was ignored entirely; post-args it returns 200 because the body is
/// parsed into the defaults.
#[tokio::test]
async fn invoke_accepts_1kib_body_with_unknown_fields() {
    let router = router();
    let id = deploy_min_module(&router).await;

    // ~1 KiB of opaque JSON. Use a single string value so the encoded
    // byte length is predictable and well past 1024 bytes.
    let filler = "x".repeat(1024);
    let body = json!({ "filler": filler });

    let invoke_req = json_post(&format!("/functions/{id}/invoke"), body);
    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    let status = invoke_resp.status();
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "1 KiB body with unknown fields produced 400 — schema is too strict"
    );
    assert_ne!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "1 KiB body with unknown fields produced 422 — schema is too strict"
    );
    assert_eq!(status, StatusCode::OK, "expected 200 for default-args body");
    let body = body_json(invoke_resp.into_body()).await;
    assert_eq!(
        body.get("function_id").and_then(Value::as_str),
        Some(id.as_str()),
        "function_id mismatch: {body}"
    );
}

/// A deliberately-malformed JSON body must now surface as `400 invalid_json`.
/// Pre-args the handler ignored the body entirely (so a malformed payload
/// silently succeeded); post-args we have a schema to enforce, so the
/// canonical wire envelope kicks in.
#[tokio::test]
async fn invoke_rejects_malformed_json_body() {
    let router = router();
    let id = deploy_min_module(&router).await;

    // `{not valid}` is not parseable JSON — an unquoted bareword key with
    // no value. The `Json<InvokeRequest>` extractor surfaces this as a
    // `JsonRejection`, which the shared `From<JsonRejection>` impl maps
    // to the `invalid_json` envelope.
    let malformed = b"{not valid}".to_vec();
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .body(Body::from(malformed))
        .unwrap();

    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    assert_eq!(
        invoke_resp.status(),
        StatusCode::BAD_REQUEST,
        "malformed JSON must surface as 400, not silently succeed"
    );
    let body = body_json(invoke_resp.into_body()).await;
    let kind = body.pointer("/error/kind").and_then(Value::as_str);
    assert_eq!(
        kind,
        Some("invalid_json"),
        "expected invalid_json envelope, got: {body}"
    );
}

/// Empty body remains a valid back-compat shorthand for "use defaults".
/// Some legacy clients (notably the in-tree CLI invoking against older
/// servers) send `{}` or nothing at all; both must keep working.
#[tokio::test]
async fn invoke_accepts_empty_body() {
    let router = router();
    let id = deploy_min_module(&router).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "empty body must use defaults and run _start"
    );
}

/// Same coverage for the async sibling: unknown fields in a 1 KiB body
/// are tolerated and the call is dispatched.
#[tokio::test]
async fn invoke_async_accepts_1kib_body_with_unknown_fields() {
    let router = router();
    let id = deploy_min_module(&router).await;

    let filler = "x".repeat(1024);
    let body = json!({ "filler": filler });

    let req = json_post(&format!("/functions/{id}/invoke-async"), body);
    let resp = router.oneshot(req).await.expect("invoke-async oneshot");
    let status = resp.status();
    assert_ne!(status, StatusCode::BAD_REQUEST);
    assert_ne!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "expected 202 for default-args body on async path"
    );
}

#[tokio::test]
async fn invoke_async_rejects_malformed_json_body() {
    let router = router();
    let id = deploy_min_module(&router).await;

    let malformed = b"{not valid}".to_vec();
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-async"))
        .header("content-type", "application/json")
        .body(Body::from(malformed))
        .unwrap();

    let resp = router.oneshot(req).await.expect("invoke-async oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "malformed JSON on async path must surface 400, not silently dispatch"
    );
    let body = body_json(resp.into_body()).await;
    let kind = body.pointer("/error/kind").and_then(Value::as_str);
    assert_eq!(kind, Some("invalid_json"));
}
