// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression guard for api S-22 (host-path / pointer leak via error envelopes).
//!
//! Before the fix, `From<ExecError> for ApiError` set the wire `message` to
//! `err.to_string()`, which for `ExecError::Wasmtime` walks the full wasmtime
//! error chain (host pointer addresses, host paths, internal stack-frame
//! names). This test deploys and invokes a wasm module that traps at runtime
//! via `unreachable`, then asserts the resulting JSON error envelope's
//! `message` field is the stable opaque string we now return and does NOT
//! contain any of the known leak markers.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

/// Substrings that must never appear in a client-facing error message.
/// Matched case-insensitively because wasmtime is not consistent about
/// the casing of e.g. "Cranelift" vs "cranelift" across releases.
const FORBIDDEN_SUBSTRINGS: &[&str] = &[
    "0x",
    "/usr/",
    "\\\\?\\C:\\",
    "wasmtime/runtime",
    "__libc_",
    "cranelift",
];

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

#[tokio::test]
async fn invoke_trap_error_envelope_does_not_leak_host_internals() {
    let router = dev_router();

    // Module that traps at runtime when `_start` is called. We export
    // `_start` so the standard invoke flow (which prefers `_start`, falling
    // back to `main`) drives straight into the trap. `unreachable` is the
    // canonical wasm-spec trap opcode.
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start") (unreachable)))"#)
        .expect("trapping wat parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);

    // Deploy.
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "trap", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(
        deploy_resp.status(),
        StatusCode::OK,
        "deploy of a trapping module must succeed (the trap only fires on call)"
    );
    let id = body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("deploy returns id");

    // Synchronous invoke. The trap surfaces as a 500 "wasmtime" envelope.
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.clone().oneshot(invoke_req).await.expect("invoke");
    assert_eq!(
        invoke_resp.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "trapping invoke must map to 500"
    );

    let body = body_json(invoke_resp.into_body()).await;
    let kind = body
        .pointer("/error/kind")
        .and_then(Value::as_str)
        .expect("envelope has kind");
    let message = body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .expect("envelope has message");

    assert_eq!(kind, "wasmtime", "expected stable kind, got {body}");

    // The message must be a stable opaque string — exact match keeps the
    // contract honest. If we ever rephrase this string, callers grepping on
    // it (e.g. dashboards filtering for known internal failures) need to
    // know — bumping this string is a wire-visible change.
    assert_eq!(
        message, "internal execution error",
        "expected stable opaque message, got {body}",
    );

    // Defence in depth: even if the exact-match assertion above is
    // ever relaxed, the message must still not contain any of the known
    // leak markers. Matched case-insensitively per FORBIDDEN_SUBSTRINGS.
    let message_lower = message.to_ascii_lowercase();
    for needle in FORBIDDEN_SUBSTRINGS {
        let needle_lower = needle.to_ascii_lowercase();
        assert!(
            !message_lower.contains(&needle_lower),
            "error envelope message {message:?} leaks forbidden substring {needle:?}",
        );
    }
}
