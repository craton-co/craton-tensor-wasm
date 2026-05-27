// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the per-token rate limiter wired into the production
//! router. The unit tests in `src/rate_limit.rs` cover bucket math and the
//! middleware in isolation; these tests prove the layer is wired in the
//! correct position relative to bearer auth and the route handlers, and that
//! disabled / unconfigured limiters do not affect the existing surface.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{
    build_router_with_full_config, AppState, AuthConfig, RateLimitConfig, RateLimiter,
    TenantConfig,
};
use tower::ServiceExt;

/// A path on the PROTECTED router stack — `/healthz` no longer flows
/// through `bearer_auth` / `rate_limit` (see
/// `openapi/tensor-wasm-api.yaml`, `paths./healthz` `security: []`), so
/// every rate-limit test in this file drives a protected route instead.
/// `GET /jobs/{id}` is the cheapest such route: the handler does no I/O,
/// it just looks the id up in an in-memory map and returns 404 when it
/// is missing — which is exactly the "permitted by the limiter" outcome
/// these tests are looking for. The job id is a fixed nil UUID so the
/// status is deterministic across runs.
const PROTECTED_PATH: &str = "/jobs/00000000-0000-0000-0000-000000000000";
/// Status returned by `GET /jobs/{nil-uuid}` when the limiter permits the
/// request — the handler finds no entry and returns the standard
/// `not_found` envelope. Anything that is NOT this status (and not 429)
/// is a wiring failure.
const ALLOWED_STATUS: StatusCode = StatusCode::NOT_FOUND;

fn router_with_limit(tokens: &[&str], qps: u32, burst: u32) -> axum::Router {
    let auth = AuthConfig::from_tokens(tokens.iter().copied());
    let limiter = RateLimiter::new(RateLimitConfig { qps, burst });
    build_router_with_full_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
        limiter,
    )
}

#[tokio::test]
async fn burst_window_admits_then_429s_with_retry_after() {
    // burst=3 → tokens A's first 3 GETs succeed; the 4th is 429 with
    // a Retry-After header. We use qps=1 / burst=3 so refills are slow
    // enough that the 4th request cannot have refilled within the test.
    let router = router_with_limit(&["A", "B"], 1, 3);

    let mut statuses = Vec::new();
    let mut retry_after = None;
    for _ in 0..4 {
        let req = Request::builder()
            .method(Method::GET)
            .uri(PROTECTED_PATH)
            .header("Authorization", "Bearer A")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.expect("oneshot");
        let status = resp.status();
        if status == StatusCode::TOO_MANY_REQUESTS {
            retry_after = resp
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
        }
        statuses.push(status);
    }

    assert_eq!(statuses[0], ALLOWED_STATUS);
    assert_eq!(statuses[1], ALLOWED_STATUS);
    assert_eq!(statuses[2], ALLOWED_STATUS);
    assert_eq!(statuses[3], StatusCode::TOO_MANY_REQUESTS);
    assert!(
        retry_after.is_some(),
        "Retry-After header missing on 429: {statuses:?}",
    );
}

#[tokio::test]
async fn separate_tokens_get_separate_buckets_through_router() {
    let router = router_with_limit(&["A", "B"], 1, 2);

    // Drain A.
    for _ in 0..2 {
        let req = Request::builder()
            .method(Method::GET)
            .uri(PROTECTED_PATH)
            .header("Authorization", "Bearer A")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), ALLOWED_STATUS);
    }
    let drained = Request::builder()
        .method(Method::GET)
        .uri(PROTECTED_PATH)
        .header("Authorization", "Bearer A")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        router.clone().oneshot(drained).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS,
    );

    // B still has its full burst.
    for _ in 0..2 {
        let req = Request::builder()
            .method(Method::GET)
            .uri(PROTECTED_PATH)
            .header("Authorization", "Bearer B")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), ALLOWED_STATUS);
    }
}

#[tokio::test]
async fn disabled_limiter_does_not_throttle() {
    // burst=0 / qps=0 → disabled. Any number of requests should pass.
    let router = router_with_limit(&["A"], 0, 0);
    for _ in 0..50 {
        let req = Request::builder()
            .method(Method::GET)
            .uri(PROTECTED_PATH)
            .header("Authorization", "Bearer A")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), ALLOWED_STATUS);
    }
}

#[tokio::test]
async fn rate_limit_runs_after_bearer_auth() {
    // No Authorization header → auth rejects with 401 *before* the limiter
    // gets a chance to count this against any bucket. Proves the layer
    // ordering: auth must be earlier in the stack than rate-limit.
    let router = router_with_limit(&["A"], 1, 1);
    let req = Request::builder()
        .method(Method::GET)
        .uri(PROTECTED_PATH)
        .body(Body::empty())
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // After the unauthenticated request, the authenticated burst is still
    // fully available — proving the failed-auth request did not consume
    // a permit from token A's bucket.
    let ok = Request::builder()
        .method(Method::GET)
        .uri(PROTECTED_PATH)
        .header("Authorization", "Bearer A")
        .body(Body::empty())
        .unwrap();
    assert_eq!(router.oneshot(ok).await.unwrap().status(), ALLOWED_STATUS);
}

#[tokio::test]
async fn dev_mode_shares_a_single_bucket() {
    // Empty allowlist => dev mode => every request maps to TokenId::DEV.
    // With burst=2 / qps=1, the 3rd request 429s regardless of which
    // "client" sent it.
    let auth = AuthConfig::default(); // dev mode
    let limiter = RateLimiter::new(RateLimitConfig { qps: 1, burst: 2 });
    let router = build_router_with_full_config(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
        limiter,
    );

    for i in 0..2 {
        let req = Request::builder()
            .method(Method::GET)
            .uri(PROTECTED_PATH)
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), ALLOWED_STATUS, "request {i}");
    }
    let req = Request::builder()
        .method(Method::GET)
        .uri(PROTECTED_PATH)
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        router.oneshot(req).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS,
    );
}
