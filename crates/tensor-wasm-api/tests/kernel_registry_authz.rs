// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Authorization gate coverage for the `/kernels` HTTP routes.
//!
//! Closes the T1 security finding: prior to this commit, `POST
//! /kernels` was mounted outside `tenant_scope`, accepted the tenant
//! extension as `Option<...>`, and ignored it. The handler's doc
//! comment promised a `kernel-publish` scope check that did not exist
//! in code — any allowlisted bearer (and, in dev mode, every
//! unauthenticated caller) could publish, list, and resolve every
//! kernel in the deployment.
//!
//! These tests pin the post-fix contract:
//!
//! 1. **Dev mode** (`TENSOR_WASM_API_TOKENS` empty / not configured) —
//!    `POST /kernels` is rejected with
//!    `403 kernel_publish_disabled_in_dev_mode` even with an otherwise
//!    valid body. GET routes still pass through because dev mode is a
//!    documented escape hatch for local development.
//! 2. **Production mode with a non-publish token** — `POST /kernels`
//!    is rejected with `403 kernel_publish_scope_required`; GET
//!    `/kernels` succeeds (any authenticated tenant may read).
//! 3. **Production mode with a publish token** — `POST /kernels`
//!    publishes successfully (`201 Created`).
//!
//! Each test builds its own router via
//! [`build_router_with_kernel_publish_tokens`] so the publish-tokens
//! allowlist is explicit and parallel tests do not race on
//! `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`.
//!
//! ## CI / feature matrix
//!
//! This file is gated behind `#![cfg(feature = "kernel-registry-api")]`, so
//! it runs zero tests on the default build with no skipped-coverage signal.
//! CI MUST run a feature-enabled job
//! (`cargo test -p tensor-wasm-api --features kernel-registry-api`, or
//! `--all-features`) to exercise this T1 authz contract. See the
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

/// HMAC key shared by every test below. Same constant the
/// `kernel_registry_routes` integration suite uses for parity with the
/// signature-fixture helper.
const TEST_KEY: [u8; 32] = [0x42u8; 32];

/// Bearer token that holds the kernel-publish scope.
const PUBLISH_TOKEN: &str = "publish-allowed-token";

/// Bearer token that is allowlisted for the API (clears `bearer_auth`)
/// but absent from the kernel-publish allowlist — exercises the
/// `kernel_publish_scope_required` branch.
const READ_ONLY_TOKEN: &str = "read-only-token";

/// Build an `AppState` wired with an in-memory registry under
/// [`TEST_KEY`]. The kernel-publish gate runs BEFORE the registry
/// dispatch, but each successful-publish path still needs a backing
/// registry, so we install one unconditionally.
fn state_with_registry() -> Arc<AppState> {
    let registry: Arc<dyn KernelRegistry> = Arc::new(InMemoryRegistry::new(TEST_KEY));
    Arc::new(AppState::default().with_kernel_registry(registry))
}

/// Build a `KernelManifest` matching `ptx_text` signed under `key`.
/// Lifted from `kernel_registry_routes.rs` to keep this test file
/// self-contained.
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

async fn body_json(body: Body) -> Value {
    let bytes = body
        .collect()
        .await
        .expect("body collects")
        .to_bytes()
        .to_vec();
    serde_json::from_slice(&bytes).expect("body parses as JSON")
}

/// Construct a publish-request payload for `name@version` signed under
/// [`TEST_KEY`]. The PTX bytes are a fixed placeholder — the
/// authorization gate fires before the registry verifies signatures,
/// so we only need the body to be syntactically well-formed.
fn publish_payload(name: &str, version: &str) -> Value {
    let ptx = "// fixture ptx\n";
    let manifest = signed_manifest(name, version, ptx, &TEST_KEY);
    json!({
        "manifest": manifest,
        "ptx_text": ptx,
    })
}

/// Build a router with the supplied auth-token allowlist and the
/// supplied publish-token allowlist. Every test below uses this so the
/// authorization knobs are explicit at the call site.
fn router_with(auth_tokens: &[&str], publish_tokens: &[&str]) -> axum::Router {
    let auth = if auth_tokens.is_empty() {
        AuthConfig::default()
    } else {
        AuthConfig::from_tokens(auth_tokens.iter().copied())
    };
    build_router_with_kernel_publish_tokens(
        state_with_registry(),
        auth,
        TenantConfig::default(),
        RateLimiter::new(RateLimitConfig::disabled()),
        AuditConfig::disabled(),
        CorsConfig::default(),
        TrustedProxies::default(),
        KernelPublishTokens::from_tokens(publish_tokens.iter().copied()),
    )
}

/// Dev mode (empty API token allowlist) MUST refuse `POST /kernels`
/// even when the body is otherwise valid. The previous behaviour
/// silently published under `AuthContext::dev()`; that path is the
/// core of the T1 finding.
#[tokio::test]
async fn dev_mode_post_kernels_returns_403_kernel_publish_disabled_in_dev_mode() {
    let router = router_with(&[], &[]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/kernels")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&publish_payload("matmul.f32", "1.0.0")).unwrap(),
        ))
        .expect("request builds");

    let resp = router.oneshot(req).await.expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body["error"]["kind"], "kernel_publish_disabled_in_dev_mode",
        "expected dev-mode rejection, got {body}",
    );
}

/// A bearer token that IS in the API allowlist but is NOT in the
/// kernel-publish allowlist must be rejected with
/// `kernel_publish_scope_required`. Mirrors the production scenario
/// where every tenant holds a tenant-scoped token but only a small
/// set of operator tokens may publish.
#[tokio::test]
async fn non_publish_token_post_kernels_returns_403_kernel_publish_scope_required() {
    let router = router_with(&[READ_ONLY_TOKEN], &[]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/kernels")
        .header("content-type", "application/json")
        .header("Authorization", format!("Bearer {READ_ONLY_TOKEN}"))
        .body(Body::from(
            serde_json::to_vec(&publish_payload("matmul.f32", "1.0.0")).unwrap(),
        ))
        .expect("request builds");

    let resp = router.oneshot(req).await.expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body["error"]["kind"], "kernel_publish_scope_required",
        "expected publish-scope rejection, got {body}",
    );
}

/// The same non-publish token MUST be admitted to `GET /kernels` — any
/// authenticated tenant may list manifests. Pins the asymmetry between
/// the read and publish authorization postures.
#[tokio::test]
async fn non_publish_token_get_kernels_returns_200() {
    let router = router_with(&[READ_ONLY_TOKEN], &[]);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/kernels")
        .header("Authorization", format!("Bearer {READ_ONLY_TOKEN}"))
        .body(Body::empty())
        .expect("request builds");

    let resp = router.oneshot(req).await.expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(
        body["manifests"].is_array(),
        "expected manifests array on list, got {body}",
    );
}

/// With a token allowlisted for kernel-publish AND in the API
/// allowlist, `POST /kernels` succeeds with `201 Created`. Confirms
/// the gate does not over-fire on the happy path and that the
/// downstream registry dispatch still runs.
#[tokio::test]
async fn publish_token_post_kernels_returns_201() {
    let router = router_with(&[PUBLISH_TOKEN], &[PUBLISH_TOKEN]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/kernels")
        .header("content-type", "application/json")
        .header("Authorization", format!("Bearer {PUBLISH_TOKEN}"))
        .body(Body::from(
            serde_json::to_vec(&publish_payload("matmul.f32", "1.0.0")).unwrap(),
        ))
        .expect("request builds");

    let resp = router.oneshot(req).await.expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["name"], "matmul.f32");
    assert_eq!(body["version"], "1.0.0");
}

/// Defence-in-depth: a token that is allowlisted to publish but is NOT
/// in the API token allowlist must still 401 at `bearer_auth` before
/// the publish gate ever runs. Guards against an operator
/// misconfiguration where `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`
/// contains a stale or off-list entry.
#[tokio::test]
async fn publish_token_not_in_api_allowlist_returns_401() {
    let router = router_with(&[READ_ONLY_TOKEN], &[PUBLISH_TOKEN]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/kernels")
        .header("content-type", "application/json")
        .header("Authorization", format!("Bearer {PUBLISH_TOKEN}"))
        .body(Body::from(
            serde_json::to_vec(&publish_payload("matmul.f32", "1.0.0")).unwrap(),
        ))
        .expect("request builds");

    let resp = router.oneshot(req).await.expect("router serves /kernels");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}
