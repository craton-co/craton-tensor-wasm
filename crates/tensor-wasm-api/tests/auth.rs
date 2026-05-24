// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Bearer-token authentication tests.
//!
//! Drives the router through [`tower::ServiceExt::oneshot`] with an explicit
//! [`AuthConfig`] so the assertions do not depend on the process
//! environment.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router_with_tokens(tokens: &[&str]) -> axum::Router {
    let auth = AuthConfig::from_tokens(tokens.iter().copied());
    build_router_with_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
    )
}

#[tokio::test]
async fn unauthenticated_request_rejected_with_envelope() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("unauthorized"),
        "got {body}"
    );
}

#[tokio::test]
async fn valid_bearer_token_passes_through() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .header("Authorization", "Bearer foo")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn unknown_bearer_token_is_401() {
    let router = router_with_tokens(&["foo", "bar"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .header("Authorization", "Bearer baz")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("unauthorized")
    );
}

#[tokio::test]
async fn missing_authorization_header_is_401() {
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn malformed_authorization_header_is_401() {
    let router = router_with_tokens(&["foo"]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .header("Authorization", "Basic ZGVhZGJlZWY=")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn dev_mode_no_auth_required() {
    // Empty allowlist => dev mode pass-through.
    let router = router_with_tokens(&[]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
