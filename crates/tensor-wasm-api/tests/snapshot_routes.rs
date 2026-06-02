// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the `/snapshot/save` and `/snapshot/restore`
//! routes (M5).
//!
//! These exercise the end-to-end HMAC wiring that turns
//! `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` from a dead knob into a live one:
//!
//!   1. save → restore round-trip under the *same* key succeeds (200) and
//!      the restored provenance matches the captured function/tenant.
//!   2. restore with the *wrong* key is rejected (403
//!      `snapshot_signature_invalid`) — the HMAC verification fires.
//!   3. restore with *no* key configured reports `503
//!      snapshot_signing_not_configured` (feature-detect shape).
//!   4. a cross-tenant restore — a wildcard-scoped caller from tenant B
//!      replaying a blob captured for tenant A — is rejected (403
//!      `tenant_scope_denied`).
//!
//! The router is built with an explicit [`AppConfig`] carrying the signing
//! key via [`AppState::with_app_config`] so the cases do not poison the
//! process environment (mirrors the `with_*` test-builder convention used
//! across this crate's other integration tests).

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_config, AppConfig, AppState, AuthConfig, TenantConfig, TokenScope,
    HEADER_TENANT,
};
use tensor_wasm_core::types::TenantId;
use tower::ServiceExt;

const KEY_A: [u8; 32] = [0x11u8; 32];
const KEY_B: [u8; 32] = [0x22u8; 32];

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Build the production router with explicit token → scope mappings and an
/// explicit [`AppConfig`] (carrying the snapshot signing key, if any).
fn router_with(scopes: &[(&str, TokenScope)], cfg: AppConfig) -> axum::Router {
    let mut map: HashMap<String, TokenScope> = HashMap::new();
    for (k, v) in scopes {
        map.insert((*k).to_owned(), v.clone());
    }
    let auth = AuthConfig::from_scopes(map);
    let state = Arc::new(AppState::default().with_app_config(cfg));
    build_router_with_config(state, auth, TenantConfig::default())
}

/// Trivial `_start`-only WAT module so deploy succeeds.
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

/// `POST /snapshot/save` as `bearer`/`tenant` for `function_id`.
async fn save(
    router: &axum::Router,
    bearer: &str,
    tenant: u64,
    function_id: &str,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/snapshot/save")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(
            serde_json::to_vec(&json!({ "function_id": function_id })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("save");
    let status = resp.status();
    (status, body_json(resp.into_body()).await)
}

/// `POST /snapshot/restore` as `bearer`/`tenant` with `snapshot_b64`.
async fn restore(
    router: &axum::Router,
    bearer: &str,
    tenant: u64,
    snapshot_b64: &str,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/snapshot/restore")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .header(HEADER_TENANT, tenant.to_string())
        .body(Body::from(
            serde_json::to_vec(&json!({ "snapshot_b64": snapshot_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("restore");
    let status = resp.status();
    (status, body_json(resp.into_body()).await)
}

// ---------------------------------------------------------------------------
// 1. save → restore round-trip under the same key
// ---------------------------------------------------------------------------

#[tokio::test]
async fn save_then_restore_round_trips_with_right_key() {
    let cfg = AppConfig::default().with_snapshot_hmac_key(KEY_A);
    let router = router_with(&[("wild", TokenScope::all())], cfg);
    let id = deploy_as(&router, "wild", 7).await;

    let (status, body) = save(&router, "wild", 7, &id).await;
    assert_eq!(status, StatusCode::OK, "save must succeed; got {body}");
    assert_eq!(
        body.get("signed").and_then(Value::as_bool),
        Some(true),
        "save must report a signed blob; got {body}",
    );
    let blob = body
        .get("snapshot_b64")
        .and_then(Value::as_str)
        .expect("snapshot_b64 present");
    assert!(!blob.is_empty(), "blob must be non-empty");

    let (status, body) = restore(&router, "wild", 7, blob).await;
    assert_eq!(status, StatusCode::OK, "restore must succeed; got {body}");
    assert_eq!(
        body.get("tenant_id").and_then(Value::as_u64),
        Some(7),
        "restored tenant must match the capturing tenant; got {body}",
    );
    // The gateway writes the signed v3 envelope.
    assert_eq!(
        body.get("version").and_then(Value::as_u64),
        Some(3),
        "restored blob must be the signed v3 wire format; got {body}",
    );
}

// ---------------------------------------------------------------------------
// 2. restore with the wrong key is rejected (HMAC mismatch)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restore_with_wrong_key_is_rejected() {
    // Capture under KEY_A.
    let saver = router_with(
        &[("wild", TokenScope::all())],
        AppConfig::default().with_snapshot_hmac_key(KEY_A),
    );
    let id = deploy_as(&saver, "wild", 1).await;
    let (status, body) = save(&saver, "wild", 1, &id).await;
    assert_eq!(status, StatusCode::OK, "save must succeed; got {body}");
    let blob = body
        .get("snapshot_b64")
        .and_then(Value::as_str)
        .expect("snapshot_b64")
        .to_owned();

    // Restore on a gateway configured with a DIFFERENT key (KEY_B).
    let restorer = router_with(
        &[("wild", TokenScope::all())],
        AppConfig::default().with_snapshot_hmac_key(KEY_B),
    );
    let (status, body) = restore(&restorer, "wild", 1, &blob).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "wrong-key restore must be 403; got {body}",
    );
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("snapshot_signature_invalid"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 3. routes report 503 when no signing key is configured
// ---------------------------------------------------------------------------

#[tokio::test]
async fn snapshot_routes_503_without_key() {
    // No key in the config.
    let router = router_with(&[("wild", TokenScope::all())], AppConfig::default());
    let id = deploy_as(&router, "wild", 1).await;

    let (status, body) = save(&router, "wild", 1, &id).await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "save without a key must be 503; got {body}",
    );
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("snapshot_signing_not_configured"),
        "got {body}",
    );

    // A made-up base64 blob — restore should still short-circuit on the
    // missing key before touching the bytes.
    let (status, body) = restore(&router, "wild", 1, "AAAA").await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "restore without a key must be 503; got {body}",
    );
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("snapshot_signing_not_configured"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 4. cross-tenant restore is rejected even with a valid signature
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cross_tenant_restore_is_rejected() {
    // One deployment, one key. A wildcard token captures a blob for tenant
    // 1, then tries to restore it while claiming tenant 2. The HMAC is
    // valid (same key), but the metadata records tenant 1, so the
    // per-resource owner check must reject with tenant_scope_denied.
    let router = router_with(
        &[("wild", TokenScope::all())],
        AppConfig::default().with_snapshot_hmac_key(KEY_A),
    );
    let id = deploy_as(&router, "wild", 1).await;
    let (status, body) = save(&router, "wild", 1, &id).await;
    assert_eq!(status, StatusCode::OK, "save must succeed; got {body}");
    let blob = body
        .get("snapshot_b64")
        .and_then(Value::as_str)
        .expect("snapshot_b64")
        .to_owned();

    let (status, body) = restore(&router, "wild", 2, &blob).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "cross-tenant restore must be 403; got {body}",
    );
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}

// ---------------------------------------------------------------------------
// 5. token-scope gate runs before the resource checks on save
// ---------------------------------------------------------------------------

#[tokio::test]
async fn save_rejects_out_of_scope_token() {
    // Token "alpha" is scoped to tenant 1 only. A save claiming tenant 2
    // must be rejected by the token-scope gate (403 tenant_scope_denied)
    // before any function lookup — and before the 503-no-key path, since
    // this config DOES have a key.
    let router = router_with(
        &[("alpha", TokenScope::from_tenants([TenantId(1)]))],
        AppConfig::default().with_snapshot_hmac_key(KEY_A),
    );
    let id = deploy_as(&router, "alpha", 1).await;

    let (status, body) = save(&router, "alpha", 2, &id).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "out-of-scope save must be 403; got {body}",
    );
    assert_eq!(
        body.pointer("/error/kind").and_then(Value::as_str),
        Some("tenant_scope_denied"),
        "got {body}",
    );
}
