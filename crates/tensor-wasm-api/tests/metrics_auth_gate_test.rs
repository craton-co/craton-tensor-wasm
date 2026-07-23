// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage for the `/metrics` bearer gate
//! (`tensor_wasm_api::server::metrics_auth_gate`, SECURITY H3).
//!
//! The scrape endpoint exposes per-tenant gauges and route/error
//! distributions. When `TENSOR_WASM_API_METRICS_TOKEN` is set the gateway
//! requires `Authorization: Bearer <token>` on `GET /metrics` and rejects
//! everything else with `401`. When the var is unset the endpoint stays
//! unauthenticated (the historical Prometheus-scrape posture covered by
//! `tests/auth.rs::metrics_endpoint_is_unauthenticated`).
//!
//! The gate's token is read at *router-build time* via
//! `MetricsAuth::from_env`, so each test sets the env var with
//! `temp_env::async_with_vars` (which serialises env mutation across the
//! test binary) and builds the router *inside* the closure.

#![allow(clippy::expect_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, ENV_METRICS_TOKEN,
};
use tower::ServiceExt;

const METRICS_TOKEN: &str = "scrape-secret-token";

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        // Bearer auth on the *protected* routes is irrelevant here — the
        // `/metrics` gate is a separate, narrower layer. Use a dev-mode
        // config so the only auth surface under test is the metrics gate.
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

async fn get_metrics(router: &axum::Router, auth_header: Option<&str>) -> StatusCode {
    let mut builder = Request::builder().method(Method::GET).uri("/metrics");
    if let Some(value) = auth_header {
        builder = builder.header("authorization", value);
    }
    let req = builder.body(Body::empty()).expect("request builds");
    router
        .clone()
        .oneshot(req)
        .await
        .expect("router serves")
        .status()
}

/// With the token configured, a scrape carrying NO `Authorization` header
/// is rejected with `401`.
#[tokio::test]
async fn metrics_requires_token_when_configured_no_header_is_401() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, Some(METRICS_TOKEN))], async {
        let router = router();
        assert_eq!(get_metrics(&router, None).await, StatusCode::UNAUTHORIZED);
    })
    .await;
}

/// With the token configured, a scrape carrying the WRONG bearer token is
/// rejected with `401`.
#[tokio::test]
async fn metrics_requires_token_when_configured_wrong_token_is_401() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, Some(METRICS_TOKEN))], async {
        let router = router();
        let status = get_metrics(&router, Some("Bearer not-the-token")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    })
    .await;
}

/// With the token configured, a non-Bearer scheme is rejected with `401`.
#[tokio::test]
async fn metrics_requires_token_when_configured_non_bearer_is_401() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, Some(METRICS_TOKEN))], async {
        let router = router();
        let status = get_metrics(&router, Some("Basic c2NyYXBlOnNlY3JldA==")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    })
    .await;
}

/// With the token configured, the correct bearer token is admitted with
/// `200` and renders the Prometheus exposition.
#[tokio::test]
async fn metrics_with_correct_token_is_200() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, Some(METRICS_TOKEN))], async {
        let router = router();
        let header = format!("Bearer {METRICS_TOKEN}");
        let status = get_metrics(&router, Some(&header)).await;
        assert_eq!(status, StatusCode::OK);
    })
    .await;
}

/// The case-insensitive scheme match (RFC 6750 §2.1) applies to the metrics
/// gate too: a lowercase `bearer` prefix with the right token is admitted.
#[tokio::test]
async fn metrics_with_lowercase_bearer_scheme_is_200() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, Some(METRICS_TOKEN))], async {
        let router = router();
        let header = format!("bearer {METRICS_TOKEN}");
        let status = get_metrics(&router, Some(&header)).await;
        assert_eq!(status, StatusCode::OK);
    })
    .await;
}

/// With the var explicitly unset the endpoint stays open: no header still
/// yields `200`. This pins the default Prometheus-scrape posture so a future
/// change to fail-closed surfaces here rather than silently.
#[tokio::test]
async fn metrics_unset_token_is_unauthenticated_200() {
    temp_env::async_with_vars([(ENV_METRICS_TOKEN, None::<&str>)], async {
        let router = router();
        assert_eq!(get_metrics(&router, None).await, StatusCode::OK);
    })
    .await;
}
