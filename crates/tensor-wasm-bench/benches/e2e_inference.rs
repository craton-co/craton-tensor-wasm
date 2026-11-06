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
//! ## Auth/rate-limit-enabled variant (finding 6)
//!
//! The `healthz`/`create`/`invoke_not_found` groups run with auth and rate
//! limiting *disabled* (the historical floor). The `e2e/secured/*` group
//! additionally exercises the bearer-auth + tenant-scope + token-bucket
//! middleware path with a single accepted token (`BENCH_TOKEN`) and a
//! deliberately high rate-limit ceiling (so the limiter admits every sample
//! and the measurement reflects the per-request bucket *check* cost, not a
//! 429 throttle).
//!
//! The secured group hits the `/functions/{id}/invoke` not-found path rather
//! than `/healthz`: in this gateway `/healthz` (and `/metrics`) sit on a probe
//! sub-router that *deliberately bypasses* the bearer-auth / rate-limit
//! layers, so a "secured healthz" would measure nothing extra. The invoke
//! route is on the authenticated sub-router, so a valid-token request
//! traverses `bearer_auth → tenant_scope → rate_limit` before the 404. The
//! delta between `e2e/invoke_not_found/post` and
//! `e2e/secured/invoke_not_found_authed` is the security-middleware tax.
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
use tensor_wasm_api::middleware::{AuthConfig, TenantConfig, HEADER_TENANT};
use tensor_wasm_api::rate_limit::{PerTenantRateLimitConfig, RateLimitConfig, RateLimiter};
use tensor_wasm_api::{build_router_with_full_config, AppState};
use tower::ServiceExt;

/// Bearer token the secured-variant router accepts (finding 6). Any other
/// token (or none) is rejected with 401 by the auth middleware.
const BENCH_TOKEN: &str = "bench-token";

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

/// Router with auth + rate limiting ENABLED (finding 6). One accepted token
/// and a high QPS/burst ceiling so the limiter never throttles a bench sample
/// — the sample measures the bucket *check* + bearer parse + tenant parse
/// overhead, not a 429.
fn make_router_secured() -> axum::Router {
    let state = AppState::new();
    let auth = AuthConfig::from_tokens([BENCH_TOKEN]);
    let tenant = TenantConfig::default();
    // High ceilings: the bench fires far fewer than 1e6 req/s, so the bucket
    // always admits; only the per-request check cost is measured.
    let limiter = RateLimiter::new(RateLimitConfig {
        qps: 1_000_000,
        burst: 1_000_000,
        per_tenant_default: PerTenantRateLimitConfig {
            burst: 1_000_000,
            qps: 1_000_000.0,
        },
    });
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

/// Auth + rate-limit-enabled invoke-not-found (finding 6). Sends a valid
/// bearer token + tenant header so the request clears every security layer
/// (`bearer_auth → tenant_scope → rate_limit`) before the 404. The cost over
/// `bench_invoke_not_found` is the security-middleware tax.
fn bench_secured_invoke_not_found(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/secured");
    group.measurement_time(Duration::from_secs(3));
    let router = make_router_secured();
    let rt = rt();
    group.bench_function("invoke_not_found_authed", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .method("POST")
                    .uri("/functions/00000000-0000-0000-0000-000000000000/invoke")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, format!("Bearer {BENCH_TOKEN}"))
                    .header(HEADER_TENANT, "1")
                    .body(Body::from("{}"))
                    .unwrap();
                let resp = router.clone().oneshot(req).await.unwrap();
                // 404: token + tenant cleared auth/rate-limit; the function id
                // simply does not exist. A 401/429 here would mean the secured
                // path is misconfigured and the measurement is meaningless.
                assert_eq!(resp.status(), StatusCode::NOT_FOUND);
                let body = resp.into_body().collect().await.unwrap().to_bytes();
                criterion::black_box(body);
            });
        });
    });
    group.finish();
}

/// Criterion configuration shared by every bench in this file (finding 6).
///
/// An in-process pprof flamegraph profiler hook was considered, but `pprof`'s
/// `flamegraph` feature pulls `inferno` (CDDL-1.0), which is outside this
/// workspace's Apache-2.0 license allowlist enforced by `cargo deny`. Capture
/// profiles out-of-band instead, per the quiet-host protocol in
/// `docs/BENCHMARKING.md`.
fn configured_criterion() -> Criterion {
    Criterion::default()
}

criterion_group!(
    name = benches;
    config = configured_criterion();
    targets =
        bench_healthz,
        bench_create_function,
        bench_invoke_not_found,
        bench_secured_invoke_not_found
);
criterion_main!(benches);
