// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S19 end-to-end inference bench: full HTTP request → tensor-wasm-api → JSON
//! response, driven in-process via `tower::ServiceExt::oneshot`. No real
//! Wasm execution happens (S17 stubs it); the measurement covers axum's
//! routing + middleware + serde overhead, which is the floor on
//! per-request latency.
//!
//! The runtime is a 2-thread multi-thread tokio runtime so that the
//! per-iteration `block_on` enter/exit cost does not dominate as it
//! would with a fresh current-thread runtime on every sample.
//!
//! ## `router.clone()` bias (finding 4)
//!
//! `tower::ServiceExt::oneshot` consumes the service it is called on, so each
//! sample must hand it an owned `Router`. An axum `Router` clone is a shallow
//! `Arc`-bump of the shared route table + middleware stack (no per-route deep
//! copy), so the clone adds a small *constant* offset to every sample — it
//! does not scale with request work and shows up identically in P50 and the
//! tail, so any `p99 - p50` gap analysis cancels it out. We keep the clone
//! inside the loop (it cannot be hoisted past a consuming `oneshot`) and
//! document the bias here rather than papering over it. The
//! `create_function` group already clones via `iter_batched` for a different
//! reason (a fresh router per sample so the function table does not grow).

use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use base64::Engine;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use http_body_util::BodyExt;
use tensor_wasm_api::middleware::{AuthConfig, TenantConfig};
use tensor_wasm_api::rate_limit::{RateLimitConfig, RateLimiter};
use tensor_wasm_api::{build_router_with_full_config, AppState};
use tower::ServiceExt;

fn wasm_header_b64() -> String {
    // The 8-byte empty module: Wasm magic + version, no sections. This is
    // structurally valid per `wasmparser::validate`, which the create-function
    // route runs at deploy time.
    let bytes = [0x00u8, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    base64::engine::general_purpose::STANDARD.encode(bytes)
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

fn bench_healthz(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/healthz");
    group.measurement_time(Duration::from_secs(3));
    let router = make_router();
    let rt = rt();
    group.bench_function("get", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Constant Arc-bump bias — see module docs (finding 4). The
                // clone is unavoidable: `oneshot` consumes the service.
                let resp = router
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri("/healthz")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                criterion::black_box(body);
            });
        });
    });
    group.finish();
}

fn bench_create_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/create_function");
    group.measurement_time(Duration::from_secs(3));
    let rt = rt();
    let payload = serde_json::json!({
        "name": "noop",
        "wasm_b64": wasm_header_b64(),
    });
    let body_bytes = serde_json::to_vec(&payload).unwrap();

    // Each iteration gets a fresh router so the function table doesn't
    // grow across samples (which would bias steady-state latency upward).
    // `body_bytes` lives outside the closure; clone per-iter so the inner
    // `async move` owns a `Vec<u8>` it can hand to `Body::from`.
    group.bench_function("post", |b| {
        b.iter_batched(
            make_router,
            |router| {
                let body = body_bytes.clone();
                rt.block_on(async move {
                    let req = Request::builder()
                        .method("POST")
                        .uri("/functions")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap();
                    let resp = router.oneshot(req).await.unwrap();
                    assert_eq!(resp.status(), StatusCode::OK);
                });
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_invoke_not_found(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/invoke_not_found");
    group.measurement_time(Duration::from_secs(3));
    let router = make_router();
    let rt = rt();
    group.bench_function("post", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .method("POST")
                    .uri("/functions/00000000-0000-0000-0000-000000000000/invoke")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap();
                // Constant Arc-bump bias — see module docs (finding 4).
                let resp = router.clone().oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::NOT_FOUND);
            });
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_healthz,
    bench_create_function,
    bench_invoke_not_found
);
criterion_main!(benches);
