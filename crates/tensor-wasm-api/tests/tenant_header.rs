// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tests the `X-TensorWasm-Tenant` header extraction pipeline.
//!
//! We do not currently expose a per-tenant counter the test could scrape
//! to assert the value reached the executor end-to-end. Instead the test
//! installs the production `tenant_scope` middleware in front of a probe
//! handler that echoes back the extension axum populates — proving the
//! header is parsed and threaded through to handler-visible state.
//!
//! Separately we exercise the full `/invoke` path with the header set
//! to confirm nothing in the production stack rejects a tenant-scoped
//! request.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::Extension;
use axum::http::{Method, Request, StatusCode};
use axum::middleware::from_fn;
use axum::routing::get;
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, HEADER_TENANT,
};
use tensor_wasm_core::types::TenantId;
use tower::ServiceExt;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Probe handler that echoes the tenant the middleware put in the
/// request extensions. Returns `null` if no extension is present.
async fn echo_tenant(tenant: Option<Extension<TenantId>>) -> Json<Value> {
    Json(match tenant {
        Some(Extension(TenantId(v))) => json!({ "tenant": v }),
        None => json!({ "tenant": null }),
    })
}

fn probe_router(cfg: TenantConfig) -> Router {
    // Compose Extension(cfg) outside tenant_scope so the request carries
    // the config in its extensions by the time tenant_scope reads it.
    // ServiceBuilder applies layers in the order added: the first
    // `.layer()` call runs first on a request.
    let stack = tower::ServiceBuilder::new()
        .layer(axum::Extension(cfg))
        .layer(from_fn(tensor_wasm_api::middleware::tenant_scope));
    Router::new().route("/probe", get(echo_tenant)).layer(stack)
}

#[tokio::test]
async fn tenant_header_threaded_to_handler() {
    let router = probe_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(HEADER_TENANT, "7")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("tenant").and_then(Value::as_u64),
        Some(7),
        "got {body}"
    );
}

#[tokio::test]
async fn tenant_header_defaults_to_zero_when_absent() {
    let router = probe_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body.get("tenant").and_then(Value::as_u64), Some(0));
}

#[tokio::test]
async fn tenant_header_required_when_configured() {
    let router = probe_router(TenantConfig {
        require_header: true,
    });
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("missing_tenant"),
        "got {body}"
    );
}

#[tokio::test]
async fn tenant_header_garbage_is_400() {
    // The header is PRESENT but unparseable. Distinguishing this from
    // the absent-and-required case (`missing_tenant`) is important for
    // dashboards: a spike in `invalid_tenant` indicates a client bug or
    // probing attacker, whereas `missing_tenant` typically reflects a
    // misconfigured client.
    let router = probe_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(HEADER_TENANT, "not-a-u64")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_tenant"),
    );
}

#[tokio::test]
async fn tenant_header_accepted_on_full_invoke_path() {
    // Smoke-test the full router: deploy a `_start`-only module, then
    // invoke it with X-TensorWasm-Tenant: 42. We can't observe the tenant id
    // beyond the executor boundary today, but a successful 200 proves the
    // production middleware stack does not reject the header.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    let router = build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    );

    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.unwrap();
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id");

    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "42")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.oneshot(invoke_req).await.unwrap();
    assert_eq!(invoke_resp.status(), StatusCode::OK);
}
