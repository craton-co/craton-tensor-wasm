// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression coverage for the CLI Bug 2 envelope-shape alignment.
//!
//! `docs/CLI.md` documents the `POST /functions/{id}/invoke` body as
//! `{"export": "...", "args": [...]}`. The CLI's `tensor-wasm invoke`
//! subcommand emits exactly that envelope; this test pins the
//! server-side contract from the opposite direction. Posting the
//! envelope must produce the same successful response as posting an
//! empty `{}` body — proving the wire shape is stable today even though
//! the executor does not yet thread `export` / `args` through to the
//! call site (that lands with api S-31's argument pass-through).
//!
//! Why we care: when api S-31 wires the envelope into the executor,
//! clients written against this release MUST NOT need to change their
//! request body. Locking the shape today catches any future drift
//! between the docs, the CLI, and the server in a single test rather
//! than via a post-shipping bug report.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

async fn body_json(body: Body) -> Value {
    let bytes = body_bytes(body).await;
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

/// Deploy a module from its WAT source and return its server-assigned id.
async fn deploy_module(router: &axum::Router, wat: &str) -> String {
    let wasm_bytes = wat::parse_str(wat).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = json_post(
        "/functions",
        json!({ "name": "invoke_envelope_shape", "wasm_b64": wasm_b64 }),
    );
    let deploy_resp = router
        .clone()
        .oneshot(deploy_req)
        .await
        .expect("deploy oneshot");
    assert_eq!(deploy_resp.status(), StatusCode::OK, "deploy failed");
    body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("deploy response has id")
}

/// Deploy a minimal WASI command module (exports a 0-arg `_start`) and
/// return its server-assigned id.
async fn deploy_min_module(router: &axum::Router) -> String {
    deploy_module(router, r#"(module (func (export "_start")))"#).await
}

/// POST the documented envelope `{"export": "main", "args": [...]}` and
/// confirm the response matches what POSTing `{}` produces. The handler
/// currently ignores the envelope contents, so the two responses must be
/// byte-identical aside from any timestamp-shaped fields (the sync
/// invoke response has none today, so we compare the full JSON).
#[tokio::test]
async fn invoke_envelope_matches_empty_body_response() {
    let router = router();
    let id = deploy_min_module(&router).await;

    // The deployed module exports a 0-arg `_start`. Post the full documented
    // envelope shape pointing at that export with no args so the envelope and
    // empty-body invocations resolve to the SAME call and therefore produce
    // byte-identical responses. (Pre-T33 the executor ignored `export`/`args`
    // so any envelope matched the empty body; post-T33 the envelope is
    // honoured, so it must name the export the empty-body path defaults to —
    // `_start` — with arity-matching args, here none.)
    let envelope = json!({
        "export": "_start",
        "args": [],
    });

    let envelope_resp = router
        .clone()
        .oneshot(json_post(&format!("/functions/{id}/invoke"), envelope))
        .await
        .expect("envelope oneshot");
    let envelope_status = envelope_resp.status();
    let envelope_body = body_json(envelope_resp.into_body()).await;

    let empty_resp = router
        .clone()
        .oneshot(json_post(&format!("/functions/{id}/invoke"), json!({})))
        .await
        .expect("empty oneshot");
    let empty_status = empty_resp.status();
    let empty_body = body_json(empty_resp.into_body()).await;

    assert_eq!(
        envelope_status, empty_status,
        "envelope and empty-body responses must share a status: \
         envelope={envelope_status}, empty={empty_status}"
    );
    assert_eq!(
        envelope_status,
        StatusCode::OK,
        "envelope shape must produce 200, got {envelope_status}"
    );
    assert_eq!(
        envelope_body, empty_body,
        "envelope and empty-body responses must be JSON-identical \
         today (the executor ignores `export`/`args` until S-31 \
         wires them through). envelope={envelope_body}, empty={empty_body}"
    );
}

/// Posting `{"args": [1,2]}` (envelope with `export` omitted) must also
/// succeed — `InvokeRequest::export` is `#[serde(default)]`, so
/// missing-field bodies fall back to the server's `_start` → `main`
/// resolution. Belt-and-braces coverage that the CLI's default
/// (`--export _start` becomes `{"export":"_start"}`) and a body missing
/// `export` are both accepted.
#[tokio::test]
async fn invoke_envelope_with_only_args_succeeds() {
    let router = router();
    // Post-T33 the args ARE forwarded to the resolved export, so the deployed
    // module's default export (`_start`) must accept them. Deploy a `_start`
    // that takes two i32 params matching the `[1, 2]` body. This proves both
    // that an export-omitted envelope defaults to `_start` and that its args
    // are threaded through to the call.
    let id = deploy_module(&router, r#"(module (func (export "_start") (param i32 i32)))"#).await;

    let body = json!({ "args": [1, 2] });
    let resp = router
        .oneshot(json_post(&format!("/functions/{id}/invoke"), body))
        .await
        .expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "envelope with only `args` must succeed"
    );
}

/// Mirror coverage for the async sibling: the envelope must produce
/// `202 Accepted` and the same `job_id`-shaped body as the empty case.
/// We do not compare the full JSON (the `job_id` field is randomly
/// generated per call) — only the status and the presence of `job_id`.
#[tokio::test]
async fn invoke_async_envelope_produces_accepted() {
    let router = router();
    let id = deploy_min_module(&router).await;

    let envelope = json!({
        "export": "_start",
        "args": [],
    });
    let resp = router
        .oneshot(json_post(
            &format!("/functions/{id}/invoke-async"),
            envelope,
        ))
        .await
        .expect("envelope oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "envelope on async path must produce 202"
    );
    let body = body_json(resp.into_body()).await;
    assert!(
        body.get("job_id").and_then(Value::as_str).is_some(),
        "async envelope response must include a job_id, got: {body}"
    );
}
