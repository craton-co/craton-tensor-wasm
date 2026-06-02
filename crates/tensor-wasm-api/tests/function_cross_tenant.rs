// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Cross-tenant authz tests for the `/functions` route family.
//!
//! These cover the three holes flagged as a B1.9 follow-up:
//!
//! 1. `POST /functions` previously skipped `authorize_tenant`, so a
//!    tenant-scoped token could deploy under a *different* tenant by
//!    setting `X-TensorWasm-Tenant` to a value outside its scope.
//! 2. `DELETE /functions/{id}` previously performed no tenant check at all
//!    — any authenticated caller could delete any tenant's function by id.
//! 3. `FunctionRecord` carried no `tenant_id` field, so even
//!    `invoke_function` / `invoke_function_async` only verified the
//!    *claimed* tenant against the bearer scope, never that the dispatched
//!    function actually belonged to that tenant. A wildcard-scoped caller
//!    from tenant B could thus invoke tenant A's record by passing
//!    `X-TensorWasm-Tenant: B`.
//!
//! Test layering mirrors `get_job_cross_tenant.rs` from the B1.9 patch:
//! each route gets a dedicated case, plus one
//! `token_scope_layer_runs_before_resource_check` case that pins the order
//! of the two 403 paths (a token-scope rejection must come *before* the
//! per-resource owner check, so a caller outside scope cannot probe id
//! existence via a `404 vs 403` split).

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, TokenScope, HEADER_TENANT,
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

/// Trivial `_start`-only WAT module; deploy/invoke target for tests that
/// only care about the auth layers, not the wasm path.
fn trivial_wasm_b64() -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    BASE64.encode(&wasm_bytes)
}

/// Deploy as `bearer`/`tenant`. Returns the new function id. Asserts 200.
async fn deploy_as(router: &axum::Router, bearer: &str, tenant: u64) -> String {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": trivial_wasm_b64() })).unwrap(),
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

// ---------------------------------------------------------------------------
// 1. POST /functions — token-scope check at deploy time
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_function_rejects_other_tenant_with_scoped_token() {
    // Token "alpha" is scoped to tenant 1 only. Deploying with
    // X-TensorWasm-Tenant: 2 must yield 403 tenant_scope_denied — pre-fix this
    // returned 200 because create_function never called authorize_tenant.
    let router = router_with_scopes(&[("alpha", TokenScope::from_tenants([TenantId(1)]))]);

    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", "Bearer alpha")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "2")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": trivial_wasm_b64() })).unwrap(),
        ))
        .unwrap();
    let resp = router.oneshot(req).await.expect("create");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 2. DELETE /functions/{id} — both token-scope AND resource-owner checks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_function_rejects_cross_tenant() {
    // Wildcard "wild" deploys under tenant 1. A second wildcard token from
    // tenant 2 must NOT be able to delete it. Pre-fix delete_function had
    // no tenant check at all so this would 204.
    let router = router_with_scopes(&[("wild", TokenScope::all())]);
    let id = deploy_as(&router, "wild", 1).await;

    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/functions/{id}"))
        .header("authorization", "Bearer wild")
        .header(HEADER_TENANT, "2")
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("delete");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );

    // And the record must still exist — a 403 must not have side-effected
    // the registry. Verify by deleting from the correct tenant.
    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/functions/{id}"))
        .header("authorization", "Bearer wild")
        .header(HEADER_TENANT, "1")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.expect("delete-correct");
    assert_eq!(
        resp.status(),
        StatusCode::NO_CONTENT,
        "owning tenant must still be able to delete",
    );
}

// ---------------------------------------------------------------------------
// 3a. POST /functions/{id}/invoke — resource-owner check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invoke_function_rejects_cross_tenant() {
    // Same wildcard-token-but-different-tenant shape: wildcard scope means
    // the token-scope layer passes; the *resource* check on FunctionRecord
    // .tenant_id is what stops the cross-tenant invoke. This is the hole
    // that survived the original tenant_scope layer.
    let router = router_with_scopes(&[("wild", TokenScope::all())]);
    let id = deploy_as(&router, "wild", 1).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("authorization", "Bearer wild")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "2")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 3b. POST /functions/{id}/invoke-async — resource-owner check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invoke_function_async_rejects_cross_tenant() {
    let router = router_with_scopes(&[("wild", TokenScope::all())]);
    let id = deploy_as(&router, "wild", 1).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-async"))
        .header("authorization", "Bearer wild")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "2")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-async");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 4. Order pinning: token-scope layer must reject BEFORE resource check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn token_scope_layer_runs_before_resource_check() {
    // Pin the order of the two 403 paths. Use a token "alpha" scoped only to
    // tenant 1. Deploy a record (as alpha + tenant 1), then attempt
    // `/invoke` while passing X-TensorWasm-Tenant: 3 — which is outside
    // alpha's scope. The response must be 403 tenant_scope_denied driven by
    // the *token-scope* layer (i.e. the bearer-auth-derived AuthContext),
    // NOT the resource-owner check.
    //
    // Why this matters: if the resource check fired first, a caller outside
    // scope could probe whether a given function id exists by comparing
    // a 403 (id exists, wrong tenant) vs 404 (id unknown). The contract is
    // that an out-of-scope token sees 403 regardless of registry state.
    let router = router_with_scopes(&[("alpha", TokenScope::from_tenants([TenantId(1)]))]);

    // Deploy a record owned by tenant 1 (the only tenant alpha is in scope
    // for).
    let id = deploy_as(&router, "alpha", 1).await;

    // Now invoke while claiming tenant 3 — outside alpha's scope.
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("authorization", "Bearer alpha")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "3")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );

    // Probe with a definitely-unknown id: the response must still be 403
    // (not 404) because the token-scope rejection fires first. This is the
    // load-bearing assertion — it proves the layers are ordered correctly.
    let unknown_id = "00000000-0000-0000-0000-000000000000";
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{unknown_id}/invoke"))
        .header("authorization", "Bearer alpha")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "3")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke unknown");
    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "out-of-scope caller must not be able to distinguish 404 from 403",
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 5. Happy paths — same-tenant operations still succeed
// ---------------------------------------------------------------------------

#[tokio::test]
async fn same_tenant_operations_still_succeed() {
    // Sanity check: with all three resource gates installed, the regular
    // deploy → invoke → delete cycle for a single tenant must still work
    // end-to-end. Catches the "we 403'd everything" regression.
    let router = router_with_scopes(&[("wild", TokenScope::all())]);
    let id = deploy_as(&router, "wild", 7).await;

    // Sync invoke.
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("authorization", "Bearer wild")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "7")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("invoke");
    assert_eq!(resp.status(), StatusCode::OK);

    // Async invoke.
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-async"))
        .header("authorization", "Bearer wild")
        .header("content-type", "application/json")
        .header(HEADER_TENANT, "7")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("invoke-async");
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    // Delete.
    let req = Request::builder()
        .method(Method::DELETE)
        .uri(format!("/functions/{id}"))
        .header("authorization", "Bearer wild")
        .header(HEADER_TENANT, "7")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.expect("delete");
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}
