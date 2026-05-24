// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end test of the async invocation lifecycle:
//! `deploy → invoke-async → poll job until completed → assert result`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

#[tokio::test]
async fn deploy_then_invoke_async_then_poll() {
    let router = router();

    // Deploy a `_start`-only module.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "async", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id");

    // Fire-and-forget. The endpoint returns 202 + a job id.
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-async"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.clone().oneshot(invoke_req).await.expect("invoke");
    assert_eq!(invoke_resp.status(), StatusCode::ACCEPTED);
    let job_id = body_json(invoke_resp.into_body())
        .await
        .get("job_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("job_id");

    // Poll under a 5 second hard timeout. The trivial module returns
    // immediately so this should take microseconds in practice.
    let final_body = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let req = Request::builder()
                .method(Method::GET)
                .uri(format!("/jobs/{job_id}"))
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.expect("poll");
            assert_eq!(resp.status(), StatusCode::OK);
            let body = body_json(resp.into_body()).await;
            match body.get("status").and_then(Value::as_str) {
                Some("completed") | Some("failed") => return body,
                _ => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }
    })
    .await
    .expect("job completes within 5s");

    assert_eq!(
        final_body.get("status").and_then(Value::as_str),
        Some("completed"),
        "expected completed; got {final_body}"
    );
    // The completed job carries the same shape the synchronous /invoke
    // endpoint returns.
    let result = final_body
        .get("result")
        .expect("completed job has result field");
    assert_eq!(result.get("result").and_then(Value::as_str), Some("ok"));
    assert_eq!(
        result.get("function_id").and_then(Value::as_str),
        Some(id.as_str())
    );
}

#[tokio::test]
async fn invoke_async_failure_recorded_as_failed_with_kind() {
    // A module with no `_start` and no `main` should make the executor
    // return MissingExport, which the async path records as Failed with
    // kind="missing_export".
    let wasm_bytes =
        wat::parse_str(r#"(module (func (export "noop")))"#).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    let router = router();
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "no_entry", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id");

    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-async"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.clone().oneshot(invoke_req).await.expect("invoke");
    assert_eq!(invoke_resp.status(), StatusCode::ACCEPTED);
    let job_id = body_json(invoke_resp.into_body())
        .await
        .get("job_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("job_id");

    let final_body = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let req = Request::builder()
                .method(Method::GET)
                .uri(format!("/jobs/{job_id}"))
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.expect("poll");
            let body = body_json(resp.into_body()).await;
            if matches!(
                body.get("status").and_then(Value::as_str),
                Some("completed") | Some("failed")
            ) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("job resolves within 5s");

    assert_eq!(
        final_body.get("status").and_then(Value::as_str),
        Some("failed"),
        "got {final_body}"
    );
    let result = final_body
        .get("result")
        .expect("failed job has result envelope");
    assert_eq!(
        result.get("kind").and_then(Value::as_str),
        Some("missing_export"),
        "got {result}"
    );
}
