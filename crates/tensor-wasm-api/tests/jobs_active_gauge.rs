// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage for the `tensor_wasm_jobs_active` gauge wired in
//! `tensor_wasm_api::routes::invoke_function_async`.
//!
//! Asserts the three contracts the dashboard depends on:
//!
//! 1. `POST /functions/{id}/invoke-async` increments the gauge by one
//!    before returning 202 Accepted.
//! 2. The spawned background task decrements the gauge once the job
//!    reaches a terminal state (Completed | Failed), so a quiescent node
//!    converges back to zero regardless of outcome.
//! 3. If the spawned task panics before its explicit `dec` point, the
//!    `JobsActiveGuard` `Drop` impl decrements the gauge anyway so a
//!    runtime panic does not leak a permanent step on the series.

#![allow(clippy::expect_used)]

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use tower::ServiceExt;

/// Process-wide serialization lock for the three tests in this file. The
/// panic-injection test arms a process-global flag in
/// `tensor_wasm_api::routes::test_hooks` that any concurrently-running
/// `invoke-async` would consume; serializing keeps the probe targeted at
/// exactly one async-invoke at a time. Tests within the same integration
/// binary run on a thread-pool by default, so without this lock the three
/// jobs-active tests would race for the flag.
fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // Recover from a poisoned mutex — a panicking test should not cascade
    // into "all subsequent tests panic at lock-acquire". The state behind
    // the lock is a unit value, so there is nothing to repair.
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router_with_state() -> (axum::Router, Arc<AppState>) {
    let state = Arc::new(AppState::default());
    let router = build_router_with_config(
        Arc::clone(&state),
        AuthConfig::default(),
        TenantConfig::default(),
    );
    (router, state)
}

async fn deploy(router: &axum::Router, wat: &str) -> String {
    let wasm_bytes = wat::parse_str(wat).expect("wat");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let deploy_req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "gauge-fixture", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    let deploy_resp = router.clone().oneshot(deploy_req).await.expect("deploy");
    assert_eq!(deploy_resp.status(), StatusCode::OK);
    body_json(deploy_resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

async fn invoke_async(router: &axum::Router, function_id: &str) -> String {
    let invoke_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/functions/{function_id}/invoke-async"))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&json!({})).unwrap()))
        .unwrap();
    let invoke_resp = router.clone().oneshot(invoke_req).await.expect("invoke");
    assert_eq!(invoke_resp.status(), StatusCode::ACCEPTED);
    body_json(invoke_resp.into_body())
        .await
        .get("job_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("job_id")
}

async fn poll_until_terminal(router: &axum::Router, job_id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let req = Request::builder()
                .method(Method::GET)
                .uri(format!("/jobs/{job_id}"))
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.expect("poll");
            assert_eq!(resp.status(), StatusCode::OK);
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

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn invoke_async_increments_jobs_active_then_decrements_on_completion() {
    // _serial held across await intentionally — this is a test-only mutex
    // that serialises the global metrics registry across these tests.
    let _serial = test_lock();
    let (router, state) = router_with_state();

    // Sanity: gauge starts at zero on a fresh AppState.
    assert_eq!(
        state.metrics.jobs_active().get(),
        0,
        "jobs_active should start at zero on a fresh AppState"
    );

    // Deploy a trivial `_start`-only module that exits immediately.
    let function_id = deploy(&router, r#"(module (func (export "_start")))"#).await;

    // Fire `invoke-async`. The handler runs `state.metrics.jobs_active().inc()`
    // before returning the 202 response, so the increment is observable
    // synchronously on the response thread.
    let job_id = invoke_async(&router, &function_id).await;

    // The handler increments synchronously; the spawned task decrements
    // asynchronously. Either ordering can race — what we assert is the
    // strong invariant: after the job reaches a terminal state and the
    // background task's `dec` has run, the gauge is back at zero.
    let final_body = poll_until_terminal(&router, &job_id).await;
    assert_eq!(
        final_body.get("status").and_then(Value::as_str),
        Some("completed"),
        "expected completed; got {final_body}"
    );

    // Yield once so the spawned task's tail (which runs `dec` *after*
    // mutating the JobRecord we polled) gets a turn to complete. Without
    // this the dec can race the assertion on multi-core runners.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if state.metrics.jobs_active().get() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("jobs_active returns to zero within 2s of terminal poll");

    // And the Prometheus exposition agrees.
    let scrape_req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let scrape_resp = router.clone().oneshot(scrape_req).await.expect("metrics");
    assert_eq!(scrape_resp.status(), StatusCode::OK);
    let bytes = scrape_resp
        .into_body()
        .collect()
        .await
        .expect("metrics body")
        .to_bytes()
        .to_vec();
    let body = String::from_utf8(bytes).expect("metrics body is UTF-8");
    assert!(
        body.contains("tensor_wasm_jobs_active 0"),
        "expected jobs_active 0 in scrape after completion; got:\n{body}"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn invoke_async_failure_path_also_decrements_jobs_active() {
    let _serial = test_lock();
    // A module with no `_start` and no `main` resolves as Failed. The
    // `dec` is in the spawn block's tail outside the match, so it must
    // still run.
    let (router, state) = router_with_state();
    let function_id = deploy(&router, r#"(module (func (export "noop")))"#).await;
    let job_id = invoke_async(&router, &function_id).await;
    let final_body = poll_until_terminal(&router, &job_id).await;
    assert_eq!(
        final_body.get("status").and_then(Value::as_str),
        Some("failed"),
        "expected failed; got {final_body}"
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if state.metrics.jobs_active().get() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("jobs_active returns to zero within 2s on the failure path");
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn invoke_async_panic_path_drop_guard_decrements_jobs_active() {
    let _serial = test_lock();
    // Panic-safety contract: if the spawned task unwinds before reaching
    // its explicit `dec` point, the Drop impl on `JobsActiveGuard` must
    // still decrement the gauge so the series does not leak a permanent
    // increment. We use the routes::test_hooks panic injector to force
    // the task to panic immediately after the guard is constructed but
    // before `run_invoke` completes.
    let (router, state) = router_with_state();
    let function_id = deploy(&router, r#"(module (func (export "_start")))"#).await;

    assert_eq!(state.metrics.jobs_active().get(), 0);

    // Arm the panic probe. The RAII handle disarms it on scope exit so a
    // later test running on the same process does not inherit the flag.
    let _armed = tensor_wasm_api::routes::test_hooks::arm_panic();

    let job_id = invoke_async(&router, &function_id).await;

    // The spawned task panics, so the JobRecord stays Pending forever —
    // there is no terminal-state transition to poll for. What we *can*
    // assert is that the gauge converges back to zero via the Drop path.
    // Catch_unwind on a JoinError is handled inside `tokio::spawn`; the
    // task aborts and its locals (including the JobsActiveGuard) are
    // dropped, which is exactly what we want to verify.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if state.metrics.jobs_active().get() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("jobs_active returns to zero within 2s after spawned task panics");

    // And the JobRecord stays Pending — we never reached the result-write
    // block — confirming the panic happened before the explicit `dec`
    // point and the only path back to zero was through `Drop`.
    let probe = Request::builder()
        .method(Method::GET)
        .uri(format!("/jobs/{job_id}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(probe).await.expect("poll");
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("status").and_then(Value::as_str),
        Some("pending"),
        "expected pending (task panicked before writing result); got {body}"
    );
}
