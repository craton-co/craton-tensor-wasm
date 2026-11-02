// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Measures the overhead of `/invoke-stream` vs `/invoke` for the scaffold
//! response shape. The streaming route is live (T34): SSE mode emits an
//! `event: done` frame after the guest returns, chunked mode forwards guest
//! chunks verbatim then a terminal frame. This bench pins the framing noise
//! floor so v0.4's real streaming wire can be benchmarked against it.
//!
//! ## Why three groups
//!
//! 1. **`invoke_stream/baseline_invoke`** — POST `/functions/{id}/invoke`
//!    with an empty body. The reference number any stream variant has to
//!    beat (or at least not lose to by much).
//! 2. **`invoke_stream/sse`** — POST `/functions/{id}/invoke-stream` with
//!    `Accept: text/event-stream`. Drives the SSE framing path
//!    (`Content-Type: text/event-stream`).
//! 3. **`invoke_stream/chunked`** — POST `/functions/{id}/invoke-stream`
//!    with the default `Accept`. Drives the chunked-transfer fallback path
//!    (`Content-Type: application/octet-stream`).
//!
//! All three share a router with one pre-deployed `_start` no-op function.
//! The id is captured at deploy time (outside `b.iter`) and reused inside the
//! loop, so registry-growth allocator noise is never charged to the streaming
//! measurement. The guest is a single no-op `_start`, so the guest compute
//! term is negligible and the number is dominated by routing + the
//! SSE/chunked framing machinery — which is exactly the floor we want.
//!
//! Each sample drains the response body to completion (`to_bytes`), so the
//! cost of producing and serialising the terminal frame(s) is included.

use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use base64::Engine;
use criterion::{criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use tensor_wasm_api::middleware::{AuthConfig, TenantConfig};
use tensor_wasm_api::rate_limit::{RateLimitConfig, RateLimiter};
use tensor_wasm_api::{build_router_with_full_config, AppState};
use tower::ServiceExt;

/// A no-op `_start` module. The synchronous and streaming invoke paths both
/// fall back to `_start`, so deploying this makes invoke reach the executor
/// (a bare 8-byte empty module would 400 `missing_export` instead and the
/// bench would measure the error path, not the framing path).
fn start_noop_wat() -> &'static str {
    r#"(module (func (export "_start")))"#
}

fn make_router() -> axum::Router {
    let state = AppState::new();
    let auth = AuthConfig::default();
    let tenant = TenantConfig::default();
    let limiter = RateLimiter::new(RateLimitConfig::disabled());
    build_router_with_full_config(state, auth, tenant, limiter)
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

/// Deploy the no-op `_start` module and return its function id (a UUID
/// string). Panics on any non-200 so a misconfiguration fails the bench
/// loudly rather than measuring a 404 path.
fn deploy_start_fn(rt: &tokio::runtime::Runtime, router: &axum::Router) -> String {
    let wasm = wat::parse_str(start_noop_wat()).expect("assemble _start module");
    let wasm_b64 = base64::engine::general_purpose::STANDARD.encode(&wasm);
    let payload = serde_json::json!({ "name": "stream-noop", "wasm_b64": wasm_b64 });
    let body = serde_json::to_vec(&payload).unwrap();
    rt.block_on(async {
        let req = Request::builder()
            .method("POST")
            .uri("/functions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "create-function must succeed"
        );
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        v["id"]
            .as_str()
            .expect("create response carries an id")
            .to_string()
    })
}

/// Baseline `/invoke` — the reference the streaming variants are compared
/// against.
fn bench_invoke_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream/baseline_invoke");
    group.measurement_time(Duration::from_secs(3));
    let rt = rt();
    let router = make_router();
    let id = deploy_start_fn(&rt, &router);
    let uri = format!("/functions/{id}/invoke");
    group.bench_function("post", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap();
                let resp = router.clone().oneshot(req).await.unwrap();
                assert!(resp.status().is_success());
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                criterion::black_box(body);
            });
        });
    });
    group.finish();
}

/// SSE-mode `/invoke-stream` — drives the `text/event-stream` framing path.
fn bench_invoke_stream_sse(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream/sse");
    group.measurement_time(Duration::from_secs(3));
    let rt = rt();
    let router = make_router();
    let id = deploy_start_fn(&rt, &router);
    let uri = format!("/functions/{id}/invoke-stream");
    group.bench_function("post", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::ACCEPT, "text/event-stream")
                    .body(Body::from("{}"))
                    .unwrap();
                let resp = router.clone().oneshot(req).await.unwrap();
                assert!(resp.status().is_success());
                // Drain the whole SSE body so the terminal `event: done`
                // frame's production + serialisation is part of the sample.
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                criterion::black_box(body);
            });
        });
    });
    group.finish();
}

/// Chunked-mode `/invoke-stream` — drives the `application/octet-stream`
/// fallback path (no SSE negotiation).
fn bench_invoke_stream_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream/chunked");
    group.measurement_time(Duration::from_secs(3));
    let rt = rt();
    let router = make_router();
    let id = deploy_start_fn(&rt, &router);
    let uri = format!("/functions/{id}/invoke-stream");
    group.bench_function("post", |b| {
        b.iter(|| {
            rt.block_on(async {
                // No `Accept: text/event-stream`, so the handler picks the
                // chunked-transfer fallback.
                let req = Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap();
                let resp = router.clone().oneshot(req).await.unwrap();
                assert!(resp.status().is_success());
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                criterion::black_box(body);
            });
        });
    });
    group.finish();
}

criterion_group!(
    streaming,
    bench_invoke_baseline,
    bench_invoke_stream_sse,
    bench_invoke_stream_chunked,
);
criterion_main!(streaming);
