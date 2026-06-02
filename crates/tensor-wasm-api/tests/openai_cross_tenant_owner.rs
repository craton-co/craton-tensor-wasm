// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Per-resource owner-check coverage for the OpenAI-compat `/v1/*` routes
//! (security finding H-1).
//!
//! The OpenAI handlers resolve `model` against a *process-global*
//! `model → function_uuid` map and then dispatch the resolved
//! `FunctionRecord`. Because OpenAI SDKs never send `X-TensorWasm-Tenant`,
//! the resolved tenant under the default policy is `TenantId(0)`. The
//! token-scope gate (`authorize_tenant`) therefore only proves the caller
//! is in scope for tenant 0 — it says nothing about who *owns* the
//! function the model maps to. Before H-1, a wildcard-scoped (or
//! tenant-0-scoped) caller could drive a model alias that resolved to a
//! `FunctionRecord` owned by a *different* tenant: the scope gate passed
//! and the handler ran the foreign tenant's wasm.
//!
//! The fix added a per-resource owner check in `openai.rs::run_translated`
//! that compares the deployed function's `tenant_id` against the caller's
//! resolved tenant and returns `403 tenant_scope_denied` on mismatch
//! (mirroring the native `/invoke` handlers' `FunctionRecord::tenant_id`
//! check in `routes.rs`).
//!
//! This is DISTINCT from `openai_routes_tenant_gating.rs`, which exercises
//! only the token-scope gate with an EMPTY model map (out-of-scope → 403,
//! in-scope → 404 model_not_found). Here the model map DOES resolve to a
//! real deployed function owned by tenant 7, so the request passes the
//! scope gate (the token covers the resolved tenant 0) and the NEW owner
//! check is the only thing that can stop it.
//!
//! Coverage:
//!
//! 1. `/v1/completions` — model resolves to a function owned by tenant 7;
//!    a token in scope for tenant 0 must receive `403 tenant_scope_denied`.
//! 2. `/v1/chat/completions` — same shape on the chat endpoint, so a
//!    single-handler regression that only re-checks one route surfaces.
//! 3. Positive control: a model owned by the resolved tenant (0) gets PAST
//!    the owner check (it reaches guest execution and returns `200`),
//!    proving the 403 is specifically the owner mismatch — not a blanket
//!    reject of every model-mapped request.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppState, AuthConfig, FunctionRecord, TenantConfig, TokenScope,
};
use tensor_wasm_core::types::TenantId;
use tower::ServiceExt;
use uuid::Uuid;

/// Trivial `_start`-only WAT module. The owner-check fires before any
/// wasm runs for the mismatch cases; for the positive control we only
/// need the guest to run cleanly to completion (it emits nothing, so the
/// completion text is empty but the status is `200`).
const TRIVIAL_WAT: &str = r#"(module (func (export "_start")))"#;

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Build the production router with:
///   * a single function deployed under `owner_tenant`,
///   * an OpenAI model map aliasing `model_name` → that function id,
///   * a token `bearer` whose `TokenScope` is `scope`.
///
/// Returns the router. The caller chooses `owner_tenant` and `scope`
/// independently so a test can make the scope gate pass (scope covers the
/// resolved tenant 0) while the function is owned by someone else.
fn router_with_owned_model(
    model_name: &str,
    owner_tenant: u64,
    bearer: &str,
    scope: TokenScope,
) -> axum::Router {
    let state = AppState::default();

    // Deploy a function owned by `owner_tenant` directly into the registry
    // (same pattern as `openai_prompt_reaches_guest.rs`). A deterministic
    // UUID keeps the alias mapping obvious.
    let wasm = wat::parse_str(TRIVIAL_WAT).expect("WAT parses");
    let function_id = Uuid::new_v4();
    state.functions.insert(
        function_id,
        FunctionRecord {
            id: function_id,
            name: "victim-fn".to_string(),
            wasm_bytes: Arc::from(wasm),
            created_unix_ms: 0,
            tenant_id: TenantId(owner_tenant),
        },
    );

    // Alias `model_name` → the deployed function id.
    let mut map = HashMap::new();
    map.insert(model_name.to_owned(), function_id);
    let state = state.with_openai_model_map(Arc::new(map));

    // Token scope is supplied by the caller so the scope gate can be made
    // to pass independently of who owns the function.
    let mut scopes: HashMap<String, TokenScope> = HashMap::new();
    scopes.insert(bearer.to_owned(), scope);
    let auth = AuthConfig::from_scopes(scopes);

    build_router_with_config(Arc::new(state), auth, TenantConfig::default())
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
// 1. /v1/completions — model owned by a DIFFERENT tenant must 403
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_rejects_model_owned_by_other_tenant() {
    // "victim-model" resolves to a function owned by tenant 7. The caller's
    // wildcard token covers every tenant — including the resolved tenant 0
    // an OpenAI client implicitly addresses — so the token-scope gate
    // PASSES. The only thing that can stop the dispatch is the new
    // per-resource owner check (resolved tenant 0 != owner tenant 7).
    //
    // A regression to 200 = the foreign tenant's wasm ran (the H-1 hole).
    // A regression to 404 = the owner check was replaced by a record-miss
    // path (it should be a deliberate 403, not "looks deleted").
    let router = router_with_owned_model("victim-model", 7, "wild", TokenScope::all());

    let resp = router
        .oneshot(openai_post(
            "/v1/completions",
            "wild",
            json!({ "model": "victim-model", "prompt": "exfiltrate", "stream": false }),
        ))
        .await
        .expect("router serves /v1/completions");

    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "cross-tenant model must be rejected by the owner check (200 = H-1 hole reopened)",
    );
    let body = body_json(resp.into_body()).await;
    // OpenAI-shape envelope: `{ "error": { "type", "code", ... } }`. The
    // owner check stamps `code = "tenant_scope_denied"` and `into_response`
    // maps that to 403 (see `OpenAiError::tenant_scope_denied`).
    assert_eq!(
        body.pointer("/error/code").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "expected OpenAI tenant_scope_denied envelope, got {body}",
    );
}

// ---------------------------------------------------------------------------
// 2. /v1/chat/completions — same owner check on the chat endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_rejects_model_owned_by_other_tenant() {
    // Same shape as the completions case, on the chat endpoint. A
    // single-handler regression that only wired the owner check into one of
    // the two `run_translated` callers must surface here.
    let router = router_with_owned_model("victim-model", 7, "wild", TokenScope::all());

    let resp = router
        .oneshot(openai_post(
            "/v1/chat/completions",
            "wild",
            json!({
                "model": "victim-model",
                "messages": [{ "role": "user", "content": "exfiltrate" }],
                "stream": false,
            }),
        ))
        .await
        .expect("router serves /v1/chat/completions");

    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "cross-tenant model must be rejected on chat too (200 = H-1 hole reopened)",
    );
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.pointer("/error/code").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "expected OpenAI tenant_scope_denied envelope, got {body}",
    );
}

// ---------------------------------------------------------------------------
// 3. Positive control — model owned by the RESOLVED tenant passes the check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_allows_model_owned_by_resolved_tenant() {
    // Same router shape, but the function is owned by tenant 0 — the tenant
    // an OpenAI client (no `X-TensorWasm-Tenant`) implicitly resolves to.
    // The owner check (0 == 0) must therefore PASS and the request must get
    // through to guest execution. The trivial guest runs cleanly and emits
    // nothing, so the handler returns a well-formed 200 `text_completion`
    // envelope with empty text.
    //
    // This proves the 403 in test (1) is specifically the owner *mismatch*
    // and not a blanket reject of every model-mapped request.
    let router = router_with_owned_model("own-model", 0, "wild", TokenScope::all());

    let resp = router
        .oneshot(openai_post(
            "/v1/completions",
            "wild",
            json!({ "model": "own-model", "prompt": "hello", "stream": false }),
        ))
        .await
        .expect("router serves /v1/completions");

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "same-tenant model must pass the owner check and reach execution",
    );
    let body = body_json(resp.into_body()).await;
    // Got past the owner check into the success envelope: an OpenAI
    // `text_completion` object with a `choices[0].text` field present (the
    // trivial guest emits nothing, so the text is empty — its presence, not
    // its content, is what proves we reached the buffered-success path).
    assert_eq!(
        body.pointer("/object").and_then(Value::as_str),
        Some("text_completion"),
        "expected a text_completion success envelope, got {body}",
    );
    assert!(
        body.pointer("/choices/0/text")
            .and_then(Value::as_str)
            .is_some(),
        "expected choices[0].text on the success envelope, got {body}",
    );
}
