// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage for the HTTP-metrics middleware
//! (`tensor_wasm_api::http_metrics`). Drives the router via
//! `tower::ServiceExt::oneshot` (no real socket) and asserts on the
//! Prometheus text exposition rendered by `GET /metrics`.

#![allow(clippy::expect_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::json;
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig,
};
use tower::ServiceExt;

/// Minimal but legal empty Wasm module. Matches the fixture used by
/// `http_integration.rs` so the bench/test fixtures stay aligned.
const WASM_MIN_MODULE: [u8; 8] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

fn router_dev_mode() -> (axum::Router, Arc<AppState>) {
    let state = Arc::new(AppState::default());
    let router = build_router_with_config(
        Arc::clone(&state),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    (router, state)
}

fn router_with_tokens(tokens: &[&str]) -> (axum::Router, Arc<AppState>) {
    let state = Arc::new(AppState::default());
    let router = build_router_with_config(
        Arc::clone(&state),
        AuthConfig::from_tokens(tokens.iter().copied()),
        TenantConfig::default(),
    );
    (router, state)
}

async fn scrape_metrics(router: &axum::Router) -> String {
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = body_bytes(resp.into_body()).await;
    String::from_utf8(bytes).expect("metrics body is UTF-8")
}

#[tokio::test]
async fn healthz_increments_counter_with_route_template_label() {
    let (router, _state) = router_dev_mode();

    // Hit /healthz once.
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Scrape /metrics. The /metrics scrape itself is counted on *future*
    // scrapes, not this one (the counter increment happens after the
    // handler returns the body), so /healthz must show count == 1 here.
    let body = scrape_metrics(&router).await;
    assert!(
        body.contains("tensor_wasm_http_requests_total{route=\"/healthz\",method=\"GET\",status=\"200\"} 1"),
        "missing /healthz counter sample in:\n{body}"
    );
    // Histogram sample for /healthz must exist (count series).
    assert!(
        body.contains("tensor_wasm_http_request_duration_seconds_count{route=\"/healthz\",method=\"GET\",status=\"200\"}"),
        "missing /healthz histogram count sample in:\n{body}"
    );
    // In-flight must have returned to zero after the request finished.
    assert!(
        body.contains("tensor_wasm_http_requests_in_flight{route=\"/healthz\",method=\"GET\"} 0"),
        "in-flight gauge did not return to zero in:\n{body}"
    );
}

#[tokio::test]
async fn create_function_post_emits_counter_and_histogram() {
    let (router, _state) = router_dev_mode();

    let wasm_b64 = BASE64.encode(WASM_MIN_MODULE);
    let body = json!({ "name": "metrics-fixture", "wasm_b64": wasm_b64 });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = scrape_metrics(&router).await;
    assert!(
        body.contains("tensor_wasm_http_requests_total{route=\"/functions\",method=\"POST\",status=\"200\"} 1"),
        "missing /functions counter sample in:\n{body}"
    );
    // A histogram emits both `_bucket` and `_count` series — assert the
    // count is exactly 1 for this label tuple.
    assert!(
        body.contains("tensor_wasm_http_request_duration_seconds_count{route=\"/functions\",method=\"POST\",status=\"200\"} 1"),
        "missing /functions histogram count sample in:\n{body}"
    );
    // At least one bucket line must be present for the same labels (the
    // first default bucket, 0.001 s, is virtually guaranteed to receive
    // the observation since the handler runs in-process with no I/O).
    assert!(
        body.contains("tensor_wasm_http_request_duration_seconds_bucket{route=\"/functions\",method=\"POST\",status=\"200\""),
        "missing /functions histogram bucket sample in:\n{body}"
    );
}

#[tokio::test]
async fn invoke_with_bogus_id_uses_route_template_label_not_uuid() {
    let (router, _state) = router_dev_mode();

    // Use a syntactically-valid UUID that is not deployed -> 404.
    let bogus = "00000000-0000-0000-0000-000000000000";
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{bogus}/invoke"))
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let body = scrape_metrics(&router).await;
    // Route label MUST be the axum template, NOT the substituted UUID.
    assert!(
        body.contains("tensor_wasm_http_requests_total{route=\"/functions/:id/invoke\",method=\"POST\",status=\"404\"} 1"),
        "expected route label to be the template, not the resolved id; got:\n{body}"
    );
    // Negative assertion: the UUID must not appear anywhere in the labels
    // of the HTTP metric families. (We do not search the whole body —
    // help/type comments mention the metric name only.)
    for line in body.lines() {
        if line.starts_with("tensor_wasm_http_requests_total{")
            || line.starts_with("tensor_wasm_http_request_duration_seconds_")
            || line.starts_with("tensor_wasm_http_requests_in_flight{")
        {
            assert!(
                !line.contains(bogus),
                "UUID leaked into a label space: {line}"
            );
        }
    }
}

#[tokio::test]
async fn unauthorized_401_is_still_counted() {
    let (router, _state) = router_with_tokens(&["secret-token"]);

    // No Authorization header -> 401 from bearer_auth. The metrics layer
    // sits OUTSIDE bearer_auth so the counter must still record the
    // response with status="401".
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // The /metrics scrape itself needs to bypass auth — but our router
    // applies bearer_auth to /metrics as well. Supply the bearer token
    // for the scrape so we can read the counter.
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .header("authorization", "Bearer secret-token")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf8");
    assert!(
        body.contains("tensor_wasm_http_requests_total{route=\"/healthz\",method=\"GET\",status=\"401\"} 1"),
        "401 on /healthz was not counted; got:\n{body}"
    );
}
