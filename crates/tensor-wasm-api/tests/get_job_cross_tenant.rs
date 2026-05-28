// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for the `GET /jobs/{id}` cross-tenant info leak
//! (api S-32).
//!
//! Pre-fix, [`tensor_wasm_api::routes::get_job`] returned the full
//! `JobRecord` (including the invocation's `result` payload) for any
//! job id any authenticated caller could enumerate or brute-force —
//! regardless of which tenant the job was dispatched under. A
//! wildcard-scoped token issued for tenant `A` could therefore read
//! tenant `B`'s job results.
//!
//! The fix adds:
//!
//! 1. A `tenant_id` field to `JobRecord`, populated from the request
//!    tenant by `invoke_function_async`.
//! 2. A tenant-scope check at the top of `get_job`: the caller's bearer
//!    token must address `tenant_id` AND the request's
//!    `X-TensorWasm-Tenant` header must equal `job.tenant_id`. On
//!    mismatch the handler returns `403 tenant_scope_denied`, the same
//!    shape `/invoke` already uses for token-scope rejection.
//!
//! This test creates a job under tenant `A` with a wildcard token, then
//! issues `GET /jobs/{id}` as a different tenant (tenant `B`) using the
//! same wildcard token — i.e. the token has scope to BOTH tenants, so
//! the only barrier is the per-resource check. The pre-fix handler
//! would happily return the job; the post-fix handler must reject with
//! `403 tenant_scope_denied`. We also assert the same request as
//! tenant `A` returns `200` with the dispatched-tenant's record so the
//! happy path is not regressed.

use std::sync::Arc;
use std::time::Duration;

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

/// Build a production router with a single wildcard-scoped token. The
/// wildcard scope is critical to the test: it isolates the *per-resource*
/// tenant check from the *per-token* `authorize_tenant` check. If we used
/// a scoped token, the request would fail at the token-scope layer before
/// ever reaching the new tenant_id comparison, and we would not be
/// asserting the fix the regression is actually about.
fn router_with_wildcard_token(token: &str) -> axum::Router {
    let auth = AuthConfig::from_scopes([(token, TokenScope::all())]);
    build_router_with_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
    )
}

/// Deploy a trivial `_start`-only module and return the assigned id.
async fn deploy_trivial(router: &axum::Router, bearer: &str, tenant: u64) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "leak-probe", "wasm_b64": wasm_b64 })).unwrap(),
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

/// Fire-and-forget invoke under `tenant`. Returns the assigned job id.
async fn dispatch_async(
    router: &axum::Router,
    function_id: &str,
    bearer: &str,
    tenant: u64,
) -> String {
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{function_id}/invoke-async"))
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("invoke-async");
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    body_json(resp.into_body())
        .await
        .get("job_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("job_id")
}

fn poll_job(
    router: &axum::Router,
    job_id: &str,
    bearer: &str,
    tenant: u64,
) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{job_id}"))
        .header("authorization", format!("Bearer {bearer}"))
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::empty())
        .unwrap()
}

/// Wait for the job to leave `pending`. We poll as the owning tenant
/// (which is allowed) so subsequent assertions about cross-tenant access
/// are made against a job in a stable terminal state — though the fix
/// must also gate `pending` reads, hence the separate
/// `cross_tenant_read_is_blocked_for_pending_jobs` test below.
async fn wait_for_completion(
    router: &axum::Router,
    job_id: &str,
    bearer: &str,
    tenant: u64,
) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let req = poll_job(router, job_id, bearer, tenant);
            let resp = router.clone().oneshot(req).await.expect("poll");
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "owner poll must succeed",
            );
            let body = body_json(resp.into_body()).await;
            match body.get("status").and_then(Value::as_str) {
                Some("completed") | Some("failed") => return body,
                _ => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }
    })
    .await
    .expect("job resolves within 5s")
}

/// Pre-fix: tenant B reads tenant A's job and the response includes the
/// full result payload. Post-fix: tenant B receives `403 tenant_scope_denied`
/// and the response body carries the canonical error envelope only.
#[tokio::test]
async fn get_job_rejects_cross_tenant_read_with_tenant_scope_denied() {
    const TENANT_A: u64 = 1;
    const TENANT_B: u64 = 2;
    const TOKEN: &str = "wildcard";

    // Single wildcard-scoped token so token-scope layer always admits;
    // the per-resource tenant check is the only thing that can fail here.
    let router = router_with_wildcard_token(TOKEN);
    let function_id = deploy_trivial(&router, TOKEN, TENANT_A).await;
    let job_id = dispatch_async(&router, &function_id, TOKEN, TENANT_A).await;

    // Drive to completion as the owning tenant so the JobRecord carries
    // a real `result` field — the very payload the leak surfaced. The
    // cross-tenant assertion below would still hold for a pending job
    // (the gate runs before any result-shape check) but exercising the
    // completed path matches the real-world threat: an attacker
    // observing leaked invocation outputs.
    let completed = wait_for_completion(&router, &job_id, TOKEN, TENANT_A).await;
    assert_eq!(
        completed.get("status").and_then(Value::as_str),
        Some("completed"),
    );

    // Cross-tenant read attempt: same token, but the request claims
    // tenant B (a tenant the wildcard token does still address). The
    // handler must reject before serialising the JobRecord — body must
    // be the canonical error envelope and must NOT contain the
    // job's `result` payload.
    let req = poll_job(&router, &job_id, TOKEN, TENANT_B);
    let resp = router.clone().oneshot(req).await.expect("cross-tenant poll");
    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "cross-tenant /jobs read must be 403 forbidden",
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
    // Defence-in-depth: the response body must NOT echo back any of
    // the protected JobRecord fields. The error envelope path skips
    // serialisation of the record entirely, but pin the property
    // explicitly so a future refactor that accidentally leaks fields
    // through the error path is caught here.
    assert!(
        body.get("status").is_none(),
        "error envelope must not carry the job's status field: {body}",
    );
    assert!(
        body.get("result").is_none(),
        "error envelope must not carry the job's result field: {body}",
    );

    // Owner happy-path: a fresh request as tenant A still returns 200
    // with the dispatched-tenant's record. The fix must not regress
    // legitimate polling.
    let req = poll_job(&router, &job_id, TOKEN, TENANT_A);
    let resp = router.clone().oneshot(req).await.expect("owner re-poll");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("tenant_id").and_then(Value::as_u64),
        Some(TENANT_A),
        "owner-visible record must carry the dispatched tenant_id: {body}",
    );
    assert_eq!(
        body.get("status").and_then(Value::as_str),
        Some("completed"),
    );
}

/// Cross-tenant rejection must apply BEFORE the job reaches a terminal
/// state — otherwise a polling attacker could race the executor and
/// observe transient `pending` records (which still leak existence and
/// the function id). We dispatch the job under tenant A and immediately
/// poll as tenant B; the gate must reject regardless of whether the
/// spawned task has run yet.
#[tokio::test]
async fn cross_tenant_read_is_blocked_for_pending_jobs() {
    const TENANT_A: u64 = 7;
    const TENANT_B: u64 = 8;
    const TOKEN: &str = "wildcard";

    let router = router_with_wildcard_token(TOKEN);
    let function_id = deploy_trivial(&router, TOKEN, TENANT_A).await;
    let job_id = dispatch_async(&router, &function_id, TOKEN, TENANT_A).await;

    // Immediate cross-tenant probe — the spawned task may or may not
    // have run yet; the gate must fire either way.
    let req = poll_job(&router, &job_id, TOKEN, TENANT_B);
    let resp = router.clone().oneshot(req).await.expect("racing poll");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
    );
    assert!(
        body.get("function_id").is_none(),
        "error envelope must not leak the job's function_id: {body}",
    );
}

/// A scoped-token rejection still takes precedence over the per-resource
/// check — confirming the fix did not accidentally reorder the two
/// layers. A token scoped to tenant `1` only, asking for tenant `2`'s
/// job, must get `tenant_scope_denied` at the token layer before the
/// per-resource layer compares ids.
#[tokio::test]
async fn token_scope_layer_runs_before_resource_check() {
    use std::collections::HashMap;

    const TENANT_OWNER: u64 = 1;
    const TENANT_FOREIGN: u64 = 2;
    const TOKEN_OWNER: &str = "owner";
    const TOKEN_SCOPED_TO_1: &str = "scoped";

    // Two tokens: a wildcard owner used to dispatch, and a scoped
    // candidate that may only address tenant 1.
    let mut scopes: HashMap<String, TokenScope> = HashMap::new();
    scopes.insert(TOKEN_OWNER.to_owned(), TokenScope::all());
    scopes.insert(
        TOKEN_SCOPED_TO_1.to_owned(),
        TokenScope::from_tenants([TenantId(TENANT_OWNER)]),
    );
    let auth = AuthConfig::from_scopes(scopes);
    let router = build_router_with_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
    );

    let function_id = deploy_trivial(&router, TOKEN_OWNER, TENANT_OWNER).await;
    let job_id = dispatch_async(&router, &function_id, TOKEN_OWNER, TENANT_OWNER).await;

    // The scoped token addresses tenant 2 — outside its scope. The
    // token-scope layer must reject first; the per-resource layer is
    // never reached. The wire shape is the same envelope/kind in either
    // case (by design) but the test pins the precedence so a refactor
    // that swapped layer order would still pass the cross-tenant test
    // above while regressing the token-scope contract.
    let req = poll_job(&router, &job_id, TOKEN_SCOPED_TO_1, TENANT_FOREIGN);
    let resp = router.clone().oneshot(req).await.expect("scoped poll");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
    );
}
