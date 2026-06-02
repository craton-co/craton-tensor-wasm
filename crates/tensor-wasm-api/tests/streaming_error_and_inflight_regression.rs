// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression coverage for two api fixes:
//!
//! 1. **Streaming terminal-error sanitisation.** The `/invoke-stream`
//!    writer and the OpenAI-shape SSE gateway now route their terminal
//!    error frame through the per-variant
//!    `crate::routes::sanitised_exec_error_message` mapper instead of
//!    `format!("{err}")`. Before the fix a runtime trap surfaced the raw
//!    `ExecError` Display — which for `ExecError::Wasmtime` walks the full
//!    wasmtime error chain (host pointer addresses, host paths, internal
//!    stack-frame names) — straight into the client-facing SSE `data:`
//!    frame. After the fix every streaming surface emits the same fixed
//!    opaque string the synchronous `/invoke` path returns.
//!
//! 2. **In-flight gauge balance under cancellation.** The HTTP metrics
//!    middleware now manages `tensor_wasm_http_requests_in_flight{route,
//!    method}` through an RAII `InFlightGuard` whose `Drop` decrements the
//!    gauge on *every* exit path — including future cancellation when an
//!    outer `tower::timeout` layer (or a client disconnect) drops the
//!    metrics future mid-`await`. The previous manual `inc()`/`dec()` pair
//!    leaked the gauge permanently on that path (the `dec()` was never
//!    reached), leaving a stuck `1`.
//!
//! ## Harness
//!
//! Fix #1 mirrors the established api integration harness exactly: build
//! the production router with `build_router_with_config(Arc::new(AppState),
//! AuthConfig::default(), TenantConfig::default())` (dev mode, no auth),
//! drive it with `tower::ServiceExt::oneshot` over an in-memory
//! `axum::body::Body` (no real socket), and parse the SSE body. This is the
//! same shape as `tests/invoke_stream_real_emit.rs`,
//! `tests/invoke_stream_restored.rs`,
//! `tests/openai_completions_streaming.rs`, and
//! `tests/error_envelope_does_not_leak_paths.rs`. The trapping fixture (an
//! `unreachable` `_start`) is the same one
//! `error_envelope_does_not_leak_paths.rs` uses to drive the synchronous
//! path into `ExecError::Wasmtime`.
//!
//! ## Symbol-visibility note (fix #1)
//!
//! `crate::routes::sanitised_exec_error_message` is `pub(crate)` (see
//! `src/routes.rs`) and is therefore NOT reachable from this integration
//! test binary, which compiles as an external crate. Per the task's
//! fallback guidance we drive the **full HTTP stack** instead: a real
//! trapping module forced through `/invoke-stream` and the OpenAI SSE path
//! exercises the exact production wiring and asserts on the wire-visible
//! `message`. This is strictly stronger than a direct unit test of the
//! mapper because it also covers the streaming writer's error-frame
//! serialisation.

#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::routing::get;
use axum::Router;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

use tensor_wasm_api::{
    build_router_with_config, http_metrics_middleware, AppState, AuthConfig, FunctionRecord,
    HttpMetricsLayerConfig, RouteAllowList, TenantConfig,
};
use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::TenantId;

/// Substrings that must never appear in a client-facing streaming error
/// message. Mirrors the list in
/// `tests/error_envelope_does_not_leak_paths.rs` so the streaming surface
/// is held to the identical non-leakage contract as the synchronous one.
/// Matched case-insensitively because wasmtime is not consistent about
/// casing across releases.
const FORBIDDEN_SUBSTRINGS: &[&str] = &[
    "0x",
    "/usr/",
    "\\\\?\\C:\\",
    "wasmtime/runtime",
    "__libc_",
    "cranelift",
];

/// The fixed opaque string `ExecError::Wasmtime` (a runtime trap or
/// compile failure) maps to via `sanitised_exec_error_message`. Pinned by
/// exact match so a rephrase is caught as the wire-visible change it is —
/// this is the same string `error_envelope_does_not_leak_paths.rs` pins on
/// the synchronous `/invoke` surface.
const SANITISED_WASMTIME_MESSAGE: &str = "internal execution error";

/// WAT module whose `_start` traps immediately via the canonical
/// `unreachable` opcode. Deploys cleanly (the trap only fires on call) and
/// drives the invoke path into `ExecError::Wasmtime`. Same fixture as
/// `tests/error_envelope_does_not_leak_paths.rs`.
const TRAP_WAT: &str = r#"(module (func (export "_start") (unreachable)))"#;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

async fn body_json(body: Body) -> Value {
    serde_json::from_slice(&body_bytes(body).await).expect("body is JSON")
}

/// Dev-mode router (no auth, no required tenant) — the established harness
/// shared by the other streaming/error integration tests.
fn dev_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

/// Deploy the trapping module via `POST /functions` and return its id.
async fn deploy_trap(router: &axum::Router) -> String {
    let wasm_bytes = wat::parse_str(TRAP_WAT).expect("trapping wat parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "trap", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.expect("deploy");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "deploy of a trapping module must succeed (the trap only fires on call)"
    );
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("deploy returns id")
}

/// Assert a streaming error `message` is the fixed sanitised string and
/// leaks none of the known internal markers.
fn assert_sanitised(message: &str) {
    assert_eq!(
        message, SANITISED_WASMTIME_MESSAGE,
        "streaming terminal-error message must equal the fixed sanitised string",
    );
    // Defence in depth: even if the exact-match assertion is ever relaxed,
    // the message must still not contain any known leak marker.
    let lower = message.to_ascii_lowercase();
    for needle in FORBIDDEN_SUBSTRINGS {
        assert!(
            !lower.contains(&needle.to_ascii_lowercase()),
            "streaming error message {message:?} leaks forbidden substring {needle:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// Fix #1 — streaming terminal-error sanitisation
// ---------------------------------------------------------------------------

/// Native `/invoke-stream` SSE path: a runtime trap surfaces as a terminal
/// `event: error` frame whose `message` is the fixed sanitised string, NOT
/// the raw wasmtime error chain.
///
/// Pre-fix, the writer built the frame with `format!("{err}")`, so the
/// `ExecError::Wasmtime` Display (host paths / pointers / `cranelift` /
/// `wasmtime/runtime` frame names) leaked into this `data:` line. The exact
/// match plus the `FORBIDDEN_SUBSTRINGS` sweep would both have failed.
#[tokio::test]
async fn invoke_stream_terminal_error_is_sanitised() {
    let router = dev_router();
    let id = deploy_trap(&router).await;

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let resp = router.oneshot(req).await.expect("invoke-stream");

    // The SSE transport itself succeeds (200); the failure is carried as a
    // terminal `event: error` frame inside the stream body.
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );

    let body = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8 body");

    // Locate the terminal error frame and extract its `data:` JSON.
    let error_frame = body
        .split("\n\n")
        .find(|frame| frame.contains("event: error"))
        .unwrap_or_else(|| panic!("expected an `event: error` terminal frame in body:\n{body}"));
    let data_line = error_frame
        .lines()
        .find(|l| l.starts_with("data:"))
        .unwrap_or_else(|| panic!("error frame has no `data:` line:\n{error_frame}"));
    let data_json: Value = serde_json::from_str(data_line.trim_start_matches("data:").trim())
        .expect("error frame data is JSON");

    assert_eq!(
        data_json.get("reason").and_then(Value::as_str),
        Some("wasm_error"),
        "expected stable machine-readable reason, got {data_json}",
    );
    let message = data_json
        .get("message")
        .and_then(Value::as_str)
        .expect("error frame carries a message");
    assert_sanitised(message);
}

/// OpenAI-shape SSE path (`POST /v1/completions` with `stream: true`): the
/// terminal `data:` frame's `error.message` is the fixed sanitised string.
///
/// Wires a model name to a trapping function via the gateway model map (the
/// same harness as `tests/openai_completions_streaming.rs`), then drives a
/// streaming completion. Pre-fix, `make_terminal_event` formatted the raw
/// `ExecError`, leaking the wasmtime chain into the SSE error envelope.
#[tokio::test]
async fn openai_stream_terminal_error_is_sanitised() {
    let state = AppState::default();
    let wasm = wat::parse_str(TRAP_WAT).expect("trapping wat parses");
    // OpenAI requests without an `X-TensorWasm-Tenant` header resolve to
    // TenantId(0); own the record under tenant 0 so the spawn is authorised
    // (mirrors `openai_completions_streaming.rs::router_with_model`).
    let function_id = Uuid::parse_str("00000000-0000-4000-8000-00000000dead").unwrap();
    state.functions.insert(
        function_id,
        FunctionRecord {
            id: function_id,
            name: "trap-emitter".to_string(),
            wasm_bytes: Arc::from(wasm),
            created_unix_ms: 0,
            tenant_id: TenantId(0),
        },
    );
    let mut map = HashMap::new();
    map.insert("trap-model".to_owned(), function_id);
    let state = state.with_openai_model_map(Arc::new(map));
    let router = build_router_with_config(
        Arc::new(state),
        AuthConfig::default(),
        TenantConfig::default(),
    );

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "trap-model",
                "prompt": "go",
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = router.oneshot(req).await.expect("router serves");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = String::from_utf8(body_bytes(resp.into_body()).await).expect("utf-8 body");

    // Find the SSE `data:` frame that carries the OpenAI error envelope.
    let mut error_message: Option<String> = None;
    for raw in body.split("\n\n") {
        for line in raw.lines() {
            let payload = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .map(str::trim);
            let Some(payload) = payload else { continue };
            if payload == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(payload) {
                if let Some(msg) = v.pointer("/error/message").and_then(Value::as_str) {
                    error_message = Some(msg.to_owned());
                }
            }
        }
    }
    let message = error_message
        .unwrap_or_else(|| panic!("expected an SSE error envelope frame in body:\n{body}"));
    assert_sanitised(&message);
}

/// Cross-surface lock: the streaming `/invoke-stream` error `message` for a
/// trapping module equals the synchronous `/invoke` envelope `message` for
/// the identical module. Both flow through
/// `sanitised_exec_error_message`, so this pins the streaming and
/// synchronous surfaces together — a future change that re-leaks one
/// surface (or diverges the two strings) fails here.
#[tokio::test]
async fn streaming_and_sync_error_messages_agree() {
    let router = dev_router();
    let id = deploy_trap(&router).await;

    // Synchronous /invoke — the `ApiError::from(ExecError)` message.
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.clone().oneshot(invoke_req).await.expect("invoke");
    assert_eq!(invoke_resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let sync_body = body_json(invoke_resp.into_body()).await;
    let sync_message = sync_body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .expect("sync envelope carries a message")
        .to_owned();

    // Streaming /invoke-stream — the terminal SSE error frame's message.
    let stream_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{id}/invoke-stream"))
        .header(header::ACCEPT, "text/event-stream")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let stream_resp = router.oneshot(stream_req).await.expect("invoke-stream");
    assert_eq!(stream_resp.status(), StatusCode::OK);
    let stream_body =
        String::from_utf8(body_bytes(stream_resp.into_body()).await).expect("utf-8 body");
    let error_frame = stream_body
        .split("\n\n")
        .find(|frame| frame.contains("event: error"))
        .unwrap_or_else(|| panic!("expected `event: error` frame in:\n{stream_body}"));
    let data_line = error_frame
        .lines()
        .find(|l| l.starts_with("data:"))
        .expect("error frame has data line");
    let stream_message =
        serde_json::from_str::<Value>(data_line.trim_start_matches("data:").trim())
            .expect("error frame data is JSON")
            .get("message")
            .and_then(Value::as_str)
            .expect("stream error frame carries a message")
            .to_owned();

    assert_eq!(
        stream_message, sync_message,
        "streaming and synchronous error messages must agree for the same ExecError variant",
    );
    // And both are the fixed sanitised string.
    assert_sanitised(&sync_message);
    assert_sanitised(&stream_message);
}

// ---------------------------------------------------------------------------
// Fix #2 — in-flight gauge balance under cancellation
// ---------------------------------------------------------------------------

/// Slow handler that never completes within the test window. It awaits a
/// far-future sleep so the request is still in `next.run(req).await` when
/// the outer timeout fires and the metrics future is cancelled (dropped)
/// mid-`await`.
async fn slow_handler() -> StatusCode {
    tokio::time::sleep(Duration::from_secs(3600)).await;
    StatusCode::OK
}

/// Scrape the in-flight gauge sample for `(route="/healthz", method="GET")`
/// from the shared metrics registry's Prometheus text rendering.
fn in_flight_sample(metrics: &TensorWasmMetrics) -> Option<i64> {
    let text = metrics.encode_text();
    text.lines()
        .find_map(|line| {
            line.strip_prefix(
                "tensor_wasm_http_requests_in_flight{route=\"/healthz\",method=\"GET\"} ",
            )
        })
        .and_then(|v| v.trim().parse::<i64>().ok())
        .or_else(|| {
            // If the series rendered at all, return its value; otherwise
            // `None` signals the label tuple was never observed.
            text.contains("tensor_wasm_http_requests_in_flight{route=\"/healthz\",method=\"GET\"}")
                .then_some(i64::MIN)
        })
}

/// A request whose handler exceeds the timeout has its metrics future
/// cancelled mid-`await`; the `InFlightGuard`'s `Drop` must still decrement
/// the gauge so `tensor_wasm_http_requests_in_flight` returns to `0`.
///
/// This mirrors the production layer order in `src/server.rs`
/// (`build_router_full`): the per-request `timeout_layer` sits OUTSIDE
/// `http_metrics_middleware` in the `common_layers` `ServiceBuilder`, so a
/// firing timeout drops the inner metrics future between its `inc()` and
/// the point where a manual `dec()` would have run. We reproduce that exact
/// nesting with a short `tower_http::timeout::TimeoutLayer` wrapping the
/// metrics middleware — the very same layer type the production server
/// installs via `crate::middleware::timeout_layer`. When its deadline
/// elapses it drops the inner service future (returning `408`), which is
/// the cancellation that must still run `InFlightGuard::Drop`.
///
/// Pre-fix (manual `inc()`/`dec()` straddling `next.run(req).await`), the
/// cancelled future never reached `dec()` and the gauge stuck at `1`; this
/// assertion would read `1` (or `i64::MIN` via the sentinel) instead of `0`.
#[tokio::test]
async fn in_flight_gauge_balances_when_request_times_out() {
    // Dedicated registry so the assertion is not perturbed by any other
    // request traffic — this test owns the only `(route, method)` tuple it
    // scrapes.
    let metrics = Arc::new(TensorWasmMetrics::new());
    let cfg = HttpMetricsLayerConfig {
        metrics: Arc::clone(&metrics),
        // `/healthz` is on the default allow-list so the route label
        // resolves to the template rather than collapsing to "unknown".
        routes: RouteAllowList::new_default(),
    };

    // Build a minimal router that nests the layers the same way the
    // production server does: timeout OUTSIDE, http_metrics INSIDE. The
    // `MatchedPath` extension axum inserts for a real `.route(...)` lets
    // `route_label` resolve "/healthz".
    // `tower_http::timeout::TimeoutLayer` is exactly what the production
    // server uses (`crate::middleware::timeout_layer`); on deadline it
    // returns 408 and drops the inner future — the cancellation we test.
    let timeout = tower_http::timeout::TimeoutLayer::with_status_code(
        StatusCode::REQUEST_TIMEOUT,
        Duration::from_millis(50),
    );
    let app: Router = Router::new()
        .route("/healthz", get(slow_handler))
        .layer(axum::middleware::from_fn(http_metrics_middleware))
        .layer(axum::Extension(cfg))
        // Outermost: the timeout that cancels the inner metrics future when
        // the slow handler blows the deadline.
        .layer(timeout);

    let req = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();

    // The timeout layer maps the elapsed deadline to an error; `oneshot`
    // surfaces it as either an `Err` (tower timeout error) or a 408/500
    // response depending on the version's error mapping. Either way the
    // inner metrics future has been dropped mid-`await` by this point.
    let result = app.oneshot(req).await;
    // We do not assert on the specific status/err shape — the contract under
    // test is the gauge balance, not the timeout response rendering.
    drop(result);

    // The future has been cancelled. Allow any in-progress `Drop` to run by
    // yielding the scheduler a few times before scraping. (The `Drop`
    // happens synchronously when the future is dropped, but yielding makes
    // the test robust to executor scheduling.)
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    let sample = in_flight_sample(&metrics);
    assert_eq!(
        sample,
        Some(0),
        "in-flight gauge must return to 0 after a cancelled/timed-out request; \
         a pre-fix manual inc()/dec() would leave it stuck at 1 (got {sample:?})",
    );
}
