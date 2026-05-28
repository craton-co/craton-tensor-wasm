// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for PATH-TO-V1 v0.4 scoped bearer tokens.
//!
//! Covers the four behaviours documented in `API.md`:
//!
//! 1. A scoped token addressing a tenant **outside** its scope yields
//!    `403 tenant_scope_denied`.
//! 2. A scoped token addressing a tenant **inside** its scope yields `200`.
//! 3. A wildcard token addresses any tenant.
//! 4. A legacy bare token (constructed via [`AuthConfig::from_tokens`])
//!    continues to address any tenant — the back-compat shim guarantees
//!    test suites that pre-date scoped tokens keep working without changes.
//!
//! The deprecation-warning emission itself is exercised by the
//! `parse_tokens_env` unit tests in `src/token_scope.rs` (it lifts the
//! `deprecated_count` field) and by the doc-tested behaviour of
//! [`AuthConfig::from_env`]; capturing the actual `tracing::warn!` event
//! from an integration test would require a process-global subscriber,
//! which would conflict with the other tests in this crate. We assert the
//! count via the public `AuthConfig::from_env`-equivalent path instead.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, parse_tokens_env, AppState, AuthConfig, TenantConfig, TenantScope,
    TokenScope, HEADER_TENANT,
};
use tensor_wasm_core::types::TenantId;
use tower::ServiceExt;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Build the production router with explicit token → scope mappings.
fn router_with_scopes(scopes: &[(&str, TokenScope)]) -> axum::Router {
    let mut map: HashMap<String, TokenScope> = HashMap::new();
    for (k, v) in scopes {
        map.insert((*k).to_owned(), v.clone());
    }
    let auth = AuthConfig::from_scopes(map);
    build_router_with_config(Arc::new(AppState::default()), auth, TenantConfig::default())
}

/// Deploy a trivial `_start`-only module so the synchronous `/invoke` path
/// has something to call. Returns the assigned function id.
async fn deploy_trivial_function(router: &axum::Router, bearer: &str) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK, "deploy must succeed");
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

fn invoke_with_tenant(
    _router: &axum::Router,
    function_id: &str,
    bearer: &str,
    tenant: u64,
) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{function_id}/invoke"))
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn scoped_token_denied_for_out_of_scope_tenant() {
    // Token "scoped" may only address tenants 1 and 2.
    let router = router_with_scopes(&[(
        "scoped",
        TokenScope::from_tenants([TenantId(1), TenantId(2)]),
    )]);

    let id = deploy_trivial_function(&router, "scoped").await;
    // Invoke with tenant=3 → 403.
    let req = invoke_with_tenant(&router, &id, "scoped", 3);
    let resp = router.clone().oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

#[tokio::test]
async fn scoped_token_accepted_for_in_scope_tenant() {
    let router = router_with_scopes(&[(
        "scoped",
        TokenScope::from_tenants([TenantId(1), TenantId(2)]),
    )]);

    let id = deploy_trivial_function(&router, "scoped").await;
    // Tenant 1 is in scope → 200 + standard invoke envelope.
    let req = invoke_with_tenant(&router, &id, "scoped", 1);
    let resp = router.clone().oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("result").and_then(Value::as_str),
        Some("ok"),
        "got {body}",
    );

    // Tenant 2 too.
    let req = invoke_with_tenant(&router, &id, "scoped", 2);
    let resp = router.clone().oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn wildcard_token_addresses_any_tenant() {
    let router = router_with_scopes(&[("wild", TokenScope::all())]);
    let id = deploy_trivial_function(&router, "wild").await;
    for t in [0u64, 1, 7, 42, u64::MAX - 1, u64::MAX] {
        let req = invoke_with_tenant(&router, &id, "wild", t);
        let resp = router.clone().oneshot(req).await.expect("invoke");
        assert_eq!(resp.status(), StatusCode::OK, "tenant {t} should pass");
    }
}

#[tokio::test]
async fn legacy_bare_token_still_works_as_wildcard() {
    // The classic `AuthConfig::from_tokens(...)` path is what every
    // pre-scoped-tokens integration test uses. Asserting it still accepts
    // arbitrary tenant ids without 403 is the guarantee the deprecation
    // documents.
    let router = build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::from_tokens(["legacy"]),
        TenantConfig::default(),
    );

    // Deploy.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", "Bearer legacy")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let id = body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap();

    // Invoke with an arbitrarily large tenant id; legacy token must not
    // produce 403.
    let req = invoke_with_tenant(&router, &id, "legacy", 999_999);
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "legacy bare token must address any tenant for back-compat",
    );
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn parse_env_counts_deprecated_bare_entries() {
    // The wire format is part of the public contract. Exercise the parser
    // through its public entrypoint so the warning-emission policy can
    // reason about the same `deprecated_count` the server logs at startup.
    let parsed = parse_tokens_env("legacy1,legacy2,scoped:tenant=1");
    assert_eq!(parsed.deprecated_count, 2);
    assert_eq!(parsed.token_scopes.len(), 3);
    assert!(parsed.token_scopes["legacy1"].tenants.is_all());
    assert!(parsed.token_scopes["legacy2"].tenants.is_all());
    assert_eq!(
        parsed.token_scopes["scoped"].tenants,
        TenantScope::Set([TenantId(1)].into_iter().collect()),
    );

    // Pure-scoped allowlist: no deprecation count.
    let parsed = parse_tokens_env("a:tenant=1,b:tenant=*");
    assert_eq!(parsed.deprecated_count, 0);
    assert_eq!(parsed.token_scopes.len(), 2);
}

#[tokio::test]
async fn dev_mode_allows_any_tenant() {
    // Empty allowlist → dev mode → AuthContext::dev → wildcard scope.
    // A request with no Authorization header must not be 403'd by the
    // tenant-scope check.
    let router = build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let id = body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "12345")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
