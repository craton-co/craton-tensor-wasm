// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Verifies the 64 MiB inbound body cap installed via
//! `axum::extract::DefaultBodyLimit::max`. Posting 100 MB of dummy data
//! must be rejected before any handler reads the body.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, MAX_REQUEST_BODY_BYTES,
};
use tower::ServiceExt;

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

#[tokio::test]
async fn limit_constant_is_64_mib() {
    assert_eq!(MAX_REQUEST_BODY_BYTES, 64 * 1024 * 1024);
}

#[tokio::test]
async fn oversized_body_is_rejected() {
    // 100 MiB > 64 MiB cap. `DefaultBodyLimit::max` short-circuits with
    // `413 Payload Too Large` before the deploy handler reads the body.
    let huge: Vec<u8> = vec![b'a'; 100 * 1024 * 1024];
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(huge))
        .unwrap();

    let resp = router().oneshot(req).await.expect("oneshot");
    // The `DefaultBodyLimit::max` guard returns `413 Payload Too Large` for
    // any body that exceeds the cap. The public contract in `API.md`
    // (`body_too_large` → `413`) pins the response to exactly that code;
    // the earlier "either 413 or 400 is acceptable" allowance has been
    // tightened — the gateway must not silently downgrade the rejection.
    let status = resp.status();
    assert!(
        status == StatusCode::PAYLOAD_TOO_LARGE,
        "expected 413 for oversized body, got {status}"
    );
}

#[tokio::test]
async fn body_within_limit_is_processed() {
    // A modest body well under the cap should reach the handler and get
    // rejected on JSON-shape grounds (invalid_json kind) — proving the
    // size guard did not block it.
    let small = b"{ \"not\": \"a valid deploy payload\" }".to_vec();
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(small))
        .unwrap();
    let resp = router().oneshot(req).await.expect("oneshot");
    // Either 400/invalid_json (missing required field) or 422 from axum's
    // JsonRejection — both prove we passed the body limit.
    assert_ne!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
