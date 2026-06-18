// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for the `/kernels` HTTP routes (B6.4 — roadmap
//! feature #3 server side).
//!
//! These tests build a router with `kernel-registry-api` enabled and an
//! explicitly-installed `InMemoryRegistry` so the env-var initialisation
//! is bypassed (per-test envs would race other parallel tests). The
//! eight tests cover:
//!
//! 1. `publish_with_valid_signature_returns_201`
//! 2. `publish_with_bad_signature_returns_403`
//! 3. `publish_with_digest_mismatch_returns_400`
//! 4. `publish_duplicate_returns_409`
//! 5. `list_after_publish_returns_one_manifest`
//! 6. `resolve_existing_returns_manifest_and_ptx`
//! 7. `resolve_missing_returns_404`
//! 8. `routes_return_503_when_registry_not_configured`
//!
//! ## CI / feature matrix
//!
//! This whole file is gated behind `#![cfg(feature = "kernel-registry-api")]`,
//! so on the default build it compiles to nothing and runs zero tests
//! **without any signal that coverage was skipped**. To exercise it, CI MUST
//! include a job that runs the crate with the feature enabled, e.g.
//! `cargo test -p tensor-wasm-api --features kernel-registry-api` (or
//! `--all-features`). The companion authz coverage lives in
//! `tests/kernel_registry_authz.rs` under the same gate. See the
//! `kernel-registry-api` entry in `crates/tensor-wasm-api/Cargo.toml`.

#![cfg(feature = "kernel-registry-api")]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_kernel_publish_tokens, AppState, AuditConfig, AuthConfig, CorsConfig,
    KernelPublishTokens, RateLimitConfig, RateLimiter, TenantConfig, TrustedProxies,
};
use tensor_wasm_jit::registry::{sign_manifest, InMemoryRegistry, KernelManifest, KernelRegistry};
use tower::ServiceExt;

/// Bearer token used by the publish-success tests below. It is added to
/// both the API allowlist (so `bearer_auth` admits the request) and the
/// kernel-publish allowlist (so `publish_kernel` admits the POST).
const PUBLISH_TOKEN: &str = "kernel-publish-test-token";

/// HMAC key used by every test below. The actual byte value is
/// arbitrary; pinning it as a constant makes it easy to inspect failure
/// modes without scrolling.
const TEST_KEY: [u8; 32] = [0x42u8; 32];

/// Build an `AppState` wired with an in-memory registry under
/// [`TEST_KEY`]. The router built by [`publish_router`] supplies the
/// matching kernel-publish token so the auth gate clears before the
/// registry-level signature / digest checks fire.
fn state_with_registry() -> Arc<AppState> {
    let registry: Arc<dyn KernelRegistry> = Arc::new(InMemoryRegistry::new(TEST_KEY));
    Arc::new(AppState::default().with_kernel_registry(registry))
}

/// Build an `AppState` WITHOUT a registry — exercises the
/// `503 kernel_registry_not_configured` envelope.
fn state_without_registry() -> Arc<AppState> {
    Arc::new(AppState::default())
}

/// Build a router that admits the [`PUBLISH_TOKEN`] for both
/// bearer-auth and the kernel-publish scope. The previous test-only
/// helper drove the router in dev mode (no token allowlist) and relied
/// on the unguarded `POST /kernels` path; that path was the T1 finding
/// the kernel-publish gate now closes. Tests below pass the matching
/// `Authorization: Bearer <PUBLISH_TOKEN>` header on every publish.
fn publish_router(state: Arc<AppState>) -> axum::Router {
    build_router_with_kernel_publish_tokens(
        state,
        AuthConfig::from_tokens([PUBLISH_TOKEN]),
        TenantConfig::default(),
        RateLimiter::new(RateLimitConfig::disabled()),
        AuditConfig::disabled(),
        CorsConfig::default(),
        TrustedProxies::default(),
        KernelPublishTokens::from_tokens([PUBLISH_TOKEN]),
    )
}

async fn body_json(body: Body) -> Value {
    let bytes = body
        .collect()
        .await
        .expect("body collects")
        .to_bytes()
        .to_vec();
    serde_json::from_slice(&bytes).expect("body parses as JSON")
}

/// Build a manifest matching `ptx_text` signed under `key`.
fn signed_manifest(name: &str, version: &str, ptx_text: &str, key: &[u8; 32]) -> KernelManifest {
    let digest = *blake3::hash(ptx_text.as_bytes()).as_bytes();
    let mut m = KernelManifest::new(
        name.to_string(),
        version.to_string(),
        80,
        digest,
        [0u8; 32],
        0,
        "integration-test".to_string(),
    );
    m.signature = sign_manifest(&m, key);
    m
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("Authorization", format!("Bearer {PUBLISH_TOKEN}"))
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .expect("request builds")
}

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Authorization", format!("Bearer {PUBLISH_TOKEN}"))
        .body(Body::empty())
        .expect("request builds")
}

#[tokio::test]
async fn publish_with_valid_signature_returns_201() {
    let state = state_with_registry();
    let router = publish_router(state);

    let ptx = "// fake ptx\n";
    let manifest = signed_manifest("matmul.f32", "1.0.0", ptx, &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": ptx,
    });

    let resp = router
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["name"], "matmul.f32");
    assert_eq!(body["version"], "1.0.0");
}

#[tokio::test]
async fn publish_with_bad_signature_returns_403() {
    let state = state_with_registry();
    let router = publish_router(state);

    let ptx = "// fake ptx\n";
    let mut manifest = signed_manifest("matmul.f32", "1.0.0", ptx, &TEST_KEY);
    // Flip a byte in the signature so HMAC verification fails. The
    // server returns 403 (NOT 400 / 500) because a bad signature is an
    // authorization failure, not a malformed-request one.
    manifest.signature[0] ^= 0xff;
    let payload = json!({
        "manifest": manifest,
        "ptx_text": ptx,
    });

    let resp = router
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "bad_signature");
}

#[tokio::test]
async fn publish_with_digest_mismatch_returns_400() {
    let state = state_with_registry();
    let router = publish_router(state);

    // Sign the manifest against one PTX text and submit a DIFFERENT PTX
    // — the BLAKE3 digest check fires before the signature check, so we
    // expect 400 digest_mismatch.
    let signed_for_ptx = "// fake ptx\n";
    let manifest = signed_manifest("matmul.f32", "1.0.0", signed_for_ptx, &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": "// DIFFERENT ptx\n",
    });

    let resp = router
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "digest_mismatch");
}

#[tokio::test]
async fn publish_duplicate_returns_409() {
    let state = state_with_registry();
    let router = publish_router(state);

    let ptx = "// fake ptx\n";
    let manifest = signed_manifest("matmul.f32", "1.0.0", ptx, &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": ptx,
    });

    // First publish: 201.
    let resp = router
        .clone()
        .oneshot(json_post("/kernels", payload.clone()))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Second publish of the same (name@version): 409 already_registered.
    let resp = router
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "already_registered");
}

#[tokio::test]
async fn list_after_publish_returns_one_manifest() {
    let state = state_with_registry();
    let router = publish_router(state);

    let ptx = "// fake ptx\n";
    let manifest = signed_manifest("matmul.f32", "1.0.0", ptx, &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": ptx,
    });
    let resp = router
        .clone()
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CREATED);

    // GET /kernels: should now return exactly one manifest.
    let resp = router
        .oneshot(get("/kernels"))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let arr = body["manifests"].as_array().expect("manifests array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["name"], "matmul.f32");
    assert_eq!(arr[0]["version"], "1.0.0");
}

#[tokio::test]
async fn resolve_existing_returns_manifest_and_ptx() {
    let state = state_with_registry();
    let router = publish_router(state);

    let ptx = "// fake ptx\n";
    let manifest = signed_manifest("matmul.f32", "1.0.0", ptx, &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": ptx,
    });
    let resp = router
        .clone()
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CREATED);

    let resp = router
        .oneshot(get("/kernels/matmul.f32/1.0.0"))
        .await
        .expect("router serves /kernels/<name>/<version>");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["manifest"]["name"], "matmul.f32");
    assert_eq!(body["manifest"]["version"], "1.0.0");
    assert_eq!(body["ptx_text"], ptx);
}

#[tokio::test]
async fn resolve_missing_returns_404() {
    let state = state_with_registry();
    let router = publish_router(state);

    let resp = router
        .oneshot(get("/kernels/conv2d.f32/0.0.1"))
        .await
        .expect("router serves /kernels/<name>/<version>");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "not_found");
}

#[tokio::test]
async fn routes_return_503_when_registry_not_configured() {
    // No registry installed on the AppState — every /kernels call must
    // surface the configuration-failure envelope, NOT a 500 / panic.
    let state = state_without_registry();
    let router = publish_router(state);

    // POST /kernels with a syntactically-valid body. The body shape is
    // a minimal placeholder — the server should short-circuit on the
    // missing registry BEFORE attempting to parse / verify the manifest.
    let manifest = signed_manifest("matmul.f32", "1.0.0", "// ptx\n", &TEST_KEY);
    let payload = json!({
        "manifest": manifest,
        "ptx_text": "// ptx\n",
    });
    let resp = router
        .clone()
        .oneshot(json_post("/kernels", payload))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "kernel_registry_not_configured");

    // GET /kernels: same envelope.
    let resp = router
        .clone()
        .oneshot(get("/kernels"))
        .await
        .expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "kernel_registry_not_configured");

    // GET /kernels/<name>/<version>: same envelope.
    let resp = router
        .oneshot(get("/kernels/matmul.f32/1.0.0"))
        .await
        .expect("router serves /kernels/<name>/<version>");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"]["kind"], "kernel_registry_not_configured");
}
