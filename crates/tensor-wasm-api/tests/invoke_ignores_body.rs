// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression coverage for api S-31: the `/invoke` and `/invoke-async`
//! handlers must not parse the request body. Historically both handlers
//! accepted `_args: Result<Json<serde_json::Value>, JsonRejection>`, which
//! still allocated a full `serde_json::Value` tree (up to the 64 MiB
//! tower-http cap) on every request and immediately discarded it — a
//! wasted-CPU DoS surface.
//!
//! These tests post real bodies (well-formed and deliberately malformed)
//! and assert the response is not a JSON-parse error, proving the body
//! never reaches a `Json<_>` extractor.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
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

/// A 1 KiB well-formed JSON body must not influence the handler. Pre-fix,
/// the body would have been parsed into a `serde_json::Value` and thrown
/// away; post-fix the handler ignores it entirely and runs the deployed
/// `_start`, returning 200.
#[tokio::test]
async fn invoke_accepts_1kib_body_without_parsing() {
    let router = router();
    let id = deploy_min_module(&router).await;

    // ~1 KiB of opaque JSON. Use a single string value so the encoded
    // byte length is predictable and well past 1024 bytes.
    let filler = "x".repeat(1024);
    let body = json!({ "filler": filler });

    let invoke_req = json_post(&format!("/functions/{id}/invoke"), body);
    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    let status = invoke_resp.status();
    // The handler must not reject on body grounds. Pre-fix this still
    // worked because the body parsed cleanly; the stronger assertion is
    // that the response is the normal 200 success envelope.
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "1 KiB body produced 400 — handler should ignore the body entirely"
    );
    assert_ne!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "1 KiB body produced 422 — handler should ignore the body entirely"
    );
    assert_eq!(status, StatusCode::OK, "expected 200 for ignored body");
    let body = body_json(invoke_resp.into_body()).await;
    assert_eq!(
        body.get("function_id").and_then(Value::as_str),
        Some(id.as_str()),
        "function_id mismatch: {body}"
    );
}

/// A deliberately-malformed JSON body must not be touched by the handler.
/// Pre-fix this would surface as `invalid_json` from the `Json<_>` extractor
/// (the `From<JsonRejection> for ApiError` impl maps it to 400/invalid_json).
/// Post-fix the body is never inspected, so the handler proceeds to its
/// normal flow and returns 200.
#[tokio::test]
async fn invoke_ignores_malformed_json_body() {
    let router = router();
    let id = deploy_min_module(&router).await;

    // `{not valid}` is not parseable JSON — an unquoted bareword key with
    // no value. If the handler still used `Json<serde_json::Value>` this
    // would short-circuit with 400 invalid_json.
    let malformed = b"{not valid}".to_vec();
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .body(Body::from(malformed))
        .unwrap();

    let invoke_resp = router.oneshot(invoke_req).await.expect("invoke oneshot");
    let status = invoke_resp.status();
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "malformed JSON produced 400 — handler is still parsing the body"
    );
    assert_ne!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "malformed JSON produced 422 — handler is still parsing the body"
    );

    // Belt-and-braces: even if some future middleware turns this into a
    // 4xx with the canonical envelope, the `kind` must not be the
    // JSON-parse error kind. `invalid_json` would prove the handler is
    // still invoking serde_json on the body.
    if status.is_client_error() || status.is_server_error() {
        let body = body_json(invoke_resp.into_body()).await;
        let kind = body.pointer("/error/kind").and_then(Value::as_str);
        assert_ne!(
            kind,
            Some("invalid_json"),
            "got invalid_json — handler is still parsing the body: {body}"
        );
    } else {
        assert_eq!(status, StatusCode::OK, "expected 200 for ignored body");
    }
}

/// Same coverage for the async sibling: `/invoke-async` must not parse the
/// body either. The 1 KiB and malformed cases mirror the sync handler's
/// tests above.
#[tokio::test]
async fn invoke_async_accepts_1kib_body_without_parsing() {
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
        "expected 202 for ignored body on async path"
    );
}

#[tokio::test]
async fn invoke_async_ignores_malformed_json_body() {
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
    let status = resp.status();
    assert_ne!(status, StatusCode::BAD_REQUEST);
    assert_ne!(status, StatusCode::UNPROCESSABLE_ENTITY);

    if status.is_client_error() || status.is_server_error() {
        let body = body_json(resp.into_body()).await;
        let kind = body.pointer("/error/kind").and_then(Value::as_str);
        assert_ne!(
            kind,
            Some("invalid_json"),
            "got invalid_json — async handler is still parsing the body: {body}"
        );
    } else {
        assert_eq!(status, StatusCode::ACCEPTED, "expected 202 for ignored body");
    }
}
