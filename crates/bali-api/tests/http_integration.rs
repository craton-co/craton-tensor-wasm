//! In-process HTTP integration tests driving the bali-api router via
//! `tower::ServiceExt::oneshot`. No real socket is bound.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use bali_api::{build_router, AppState};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

/// Minimal but legal Wasm module header: `\0asm` magic + version 1.
const WASM_MIN_MODULE: [u8; 9] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00];

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
    build_router(Arc::new(AppState::default()))
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
    assert!(text.contains("bali metrics scaffold"), "got: {text}");
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
    let req = json_post(
        "/functions",
        json!({ "name": "tiny", "wasm_b64": BASE64.encode([0u8; 4]) }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("too_short"),
        "got {body}"
    );
}

#[tokio::test]
async fn create_function_rejects_non_wasm() {
    let req = json_post(
        "/functions",
        // Eight bytes, but the first four are not the Wasm magic.
        json!({ "name": "bogus", "wasm_b64": BASE64.encode([0x01u8; 8]) }),
    );
    let resp = router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("not_wasm"),
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
