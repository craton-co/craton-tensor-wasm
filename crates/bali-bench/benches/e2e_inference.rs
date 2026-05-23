//! S19 end-to-end inference bench: full HTTP request → bali-api → JSON
//! response, driven in-process via `tower::ServiceExt::oneshot`. No real
//! Wasm execution happens (S17 stubs it); the measurement covers axum's
//! routing + middleware + serde overhead, which is the floor on
//! per-request latency.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use bali_api::{build_router, AppState};
use base64::Engine;
use criterion::{criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn wasm_header_b64() -> String {
    // Wasm magic + version + a zero byte so length >= 8 and head matches.
    let bytes = [0x00u8, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x00];
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn make_router() -> axum::Router {
    build_router(Arc::new(AppState::default()))
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
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
    let router = make_router();
    let rt = rt();
    let payload = serde_json::json!({
        "name": "noop",
        "wasm_b64": wasm_header_b64(),
    });
    let body_bytes = serde_json::to_vec(&payload).unwrap();

    group.bench_function("post", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .method("POST")
                    .uri("/functions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body_bytes.clone()))
                    .unwrap();
                let resp = router.clone().oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
            });
        });
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
