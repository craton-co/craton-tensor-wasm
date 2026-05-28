// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tenant-scope gating coverage for the OpenAI-compat `/v1/*` routes
//! (T2 security fix).
//!
//! Before this fix the OpenAI scaffold handlers (`POST /v1/completions`
//! and `POST /v1/chat/completions`) skipped the `authorize_tenant` check
//! that every native route runs. The handlers themselves still return
//! `501 Not Implemented` (v0.4 wires real execution), but the URL surface
//! is locked and the security gate must be in place *before* v0.4 — a
//! retrofit onto a published route is a strictly worse posture than
//! shipping the gate from the beginning.
//!
//! Coverage:
//!
//! 1. A bearer token whose `TokenScope` does NOT cover the resolved
//!    tenant must receive `403 tenant_scope_denied` (not `501`). Because
//!    OpenAI SDKs do not send `X-TensorWasm-Tenant`, the resolved tenant
//!    under the default policy (`TENSOR_WASM_API_REQUIRE_TENANT` unset)
//!    is `TenantId(0)`; a token scoped to a non-zero tenant must
//!    therefore be rejected.
//! 2. A bearer token whose `TokenScope` DOES cover the resolved tenant
//!    (wildcard or `tenant=0` scope) must reach the scaffold and receive
//!    the documented `501 openai_not_yet_wired` response — confirming
//!    the gate runs but does not displace the existing 501 contract for
//!    in-scope callers.
//! 3. The same shape on the chat-completions endpoint, so a future
//!    refactor that only re-gates one of the two routes regresses
//!    visibly.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, TenantConfig, TokenScope,
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
    build_router_with_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
    )
}

fn openai_post(uri: &str, bearer: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

// ---------------------------------------------------------------------------
// 1. /v1/completions — out-of-scope token must be rejected before the 501
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_rejects_out_of_scope_token_before_501() {
    // Token "alpha" is scoped to tenant 1 only. OpenAI clients do not
    // send `X-TensorWasm-Tenant`, so the resolved tenant under default
    // policy is `TenantId(0)` — outside alpha's scope. The handler must
    // return 403 tenant_scope_denied; if it returned 501 the v0.4 wiring
    // step would inherit a hole on a locked URL surface.
    let router = router_with_scopes(&[(
        "alpha",
        TokenScope::from_tenants([TenantId(1)]),
    )]);

    let resp = router
        .oneshot(openai_post(
            "/v1/completions",
            "alpha",
            json!({ "model": "gpt-3.5-turbo", "prompt": "hello" }),
        ))
        .await
        .expect("router serves /v1/completions");

    // The contract is "401 or 403"; the tenant-scope path returns 403
    // with `kind: "tenant_scope_denied"` (see
    // `AuthContext::authorize_tenant`). Pin both: a regression to 501
    // would mean the gate was skipped, a regression to 200 would be even
    // worse, and a regression to 400 would mean the body parse ran
    // BEFORE the gate (also a hole because parsing is attacker-driven
    // work that should follow auth/authorization).
    assert!(
        resp.status() == StatusCode::FORBIDDEN || resp.status() == StatusCode::UNAUTHORIZED,
        "out-of-scope token must receive 401 or 403, got {} (501 = gate skipped)",
        resp.status(),
    );
    assert_ne!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "out-of-scope token must NOT reach the 501 scaffold path",
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "expected native tenant_scope_denied envelope, got {body}",
    );
}

// ---------------------------------------------------------------------------
// 2. /v1/completions — in-scope token still reaches the 501 scaffold
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_in_scope_token_reaches_501_scaffold() {
    // Wildcard token covers every tenant (including the default
    // `TenantId(0)` an OpenAI client implicitly addresses). The gate
    // must pass and the handler must reach its documented 501.
    let router = router_with_scopes(&[("wild", TokenScope::all())]);

    let resp = router
        .oneshot(openai_post(
            "/v1/completions",
            "wild",
            json!({ "model": "gpt-3.5-turbo", "prompt": "hello" }),
        ))
        .await
        .expect("router serves /v1/completions");

    assert_eq!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "in-scope token must reach the 501 scaffold (gate must not over-reject)",
    );
    let body = body_json(resp.into_body()).await;
    // Confirm the OpenAI-shape envelope is preserved on the happy path
    // through the gate.
    assert_eq!(
        body.pointer("/error/code").and_then(Value::as_str),
        Some("openai_not_yet_wired"),
        "501 must keep the OpenAI envelope, got {body}",
    );
}

// ---------------------------------------------------------------------------
// 3. /v1/chat/completions — symmetry: same gate on the chat endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_rejects_out_of_scope_token_before_501() {
    // Same shape as the completions test, on the chat endpoint. A
    // single-handler regression that re-opened the hole on only one of
    // the two routes must surface here.
    let router = router_with_scopes(&[(
        "alpha",
        TokenScope::from_tenants([TenantId(1)]),
    )]);

    let resp = router
        .oneshot(openai_post(
            "/v1/chat/completions",
            "alpha",
            json!({
                "model": "gpt-4",
                "messages": [{ "role": "user", "content": "hi" }],
            }),
        ))
        .await
        .expect("router serves /v1/chat/completions");

    assert!(
        resp.status() == StatusCode::FORBIDDEN || resp.status() == StatusCode::UNAUTHORIZED,
        "out-of-scope token must receive 401 or 403 on chat too, got {}",
        resp.status(),
    );
    assert_ne!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "expected native tenant_scope_denied envelope, got {body}",
    );
}

#[tokio::test]
async fn chat_completions_in_scope_token_reaches_501_scaffold() {
    let router = router_with_scopes(&[("wild", TokenScope::all())]);

    let resp = router
        .oneshot(openai_post(
            "/v1/chat/completions",
            "wild",
            json!({
                "model": "gpt-4",
                "messages": [{ "role": "user", "content": "hi" }],
            }),
        ))
        .await
        .expect("router serves /v1/chat/completions");

    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/code").and_then(Value::as_str),
        Some("openai_not_yet_wired"),
    );
}
