// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Per-token QPS + burst rate limiting (token-bucket).
//!
//! This module implements PATH-TO-V1 v0.4 exit-criterion *Rate limiting per
//! token*. It layers behind [`crate::middleware::bearer_auth`]: every request
//! that survives auth carries an [`AuthContext`] in its extensions, and the
//! [`rate_limit`] middleware uses the contained [`TokenId`] to credit a
//! per-token [`TokenBucket`].
//!
//! ## Design
//!
//! ### Token-bucket variant
//!
//! We use a **refill-on-take** bucket (sometimes called *lazy refill*) rather
//! than a background-tick refill. On every request:
//!
//! 1. Compute elapsed nanos since the bucket's `last_refill`.
//! 2. Add `elapsed * qps / 1_000_000_000` permits to `tokens`, clamped at
//!    `burst`.
//! 3. If `tokens >= 1.0` deduct one and admit; otherwise reject with
//!    `429 Too Many Requests` and the appropriate `Retry-After` value.
//!
//! **Tradeoff:** refill-on-take has zero background CPU cost and zero
//! coordination overhead (no ticker task), at the price of *cold* buckets
//! sitting in the [`DashMap`] until the process restarts. Since the
//! allowlist of bearer tokens is bounded by configuration size
//! (`TENSOR_WASM_API_TOKENS` is a finite comma-separated list), the cardinality
//! is small and bounded — a future TTL eviction sweep is a non-goal for
//! v0.4.0. A background-tick refiller would have been the wrong choice: it
//! requires a per-bucket schedule, awakens for idle tokens, and either holds
//! a global lock on the wake task or fragments scheduling per shard. The
//! refill-on-take math is two adds and a clamp; the lock is held for
//! microseconds.
//!
//! ### Sharding
//!
//! Buckets live in `Arc<DashMap<TokenId, Mutex<BucketState>>>`. DashMap
//! provides shard-level read/write locks; the inner `std::sync::Mutex`
//! serialises refill arithmetic for a single bucket. We use `std::sync::Mutex`
//! rather than `parking_lot::Mutex` to avoid pulling a new dependency into
//! `tensor-wasm-api`; the critical section is a handful of integer ops with
//! no `await` points, so OS-mutex contention behaviour is acceptable.
//!
//! ### Clock injection
//!
//! Unit tests need deterministic refill behaviour without `tokio::time::sleep`
//! (slow + flaky). The [`Clock`] trait abstracts "now". Production uses
//! [`RealClock`]; tests inject [`ManualClock`] and advance it explicitly.
//!
//! ## Wiring
//!
//! [`crate::server::build_router`] layers [`rate_limit`] after `bearer_auth`
//! and before any route handler. If [`RateLimitConfig::is_disabled`] returns
//! `true` (qps == 0 or burst == 0) the layer is installed but short-circuits
//! to a pass-through — equivalent to no rate limiting.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use serde_json::json;

/// Stable identifier for a bearer token within a single process lifetime.
///
/// We do **not** key the bucket map by the raw bearer token: doing so would
/// store secret material in the rate-limiter data structure for the lifetime
/// of the process. Instead we hash the token with the standard library's
/// SipHash (via [`std::collections::hash_map::DefaultHasher`]). SipHash is
/// keyed with process-local random state by the standard library, which is
/// sufficient as a key-derivation step here — the only consumer is a
/// [`DashMap`] lookup, never an authorization check.
///
/// In **dev mode** (empty allowlist) every request shares [`TokenId::DEV`]
/// so a single shared bucket throttles dev-mode traffic exactly the same way
/// as it would a single allowlisted production token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u64);

impl TokenId {
    /// Synthetic [`TokenId`] used in dev mode (no `TENSOR_WASM_API_TOKENS`).
    pub const DEV: TokenId = TokenId(0);

    /// Derive a [`TokenId`] from a bearer-token string. Uses the standard
    /// library's randomly-seeded SipHash; the returned value is stable for
    /// the lifetime of the process but unpredictable to outside callers.
    pub fn from_bearer(token: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        // Domain-separate from non-bearer hashes by mixing in a fixed tag.
        b"tensor-wasm-api/rate-limit/v1".hash(&mut h);
        token.hash(&mut h);
        // Force away from the dev sentinel in the (astronomically unlikely)
        // collision case so a real allowlisted token can never alias `DEV`.
        let v = h.finish();
        TokenId(if v == Self::DEV.0 { 1 } else { v })
    }
}

/// Per-request authentication context inserted into [`axum::http::Extensions`]
/// by [`crate::middleware::bearer_auth`] after a successful auth check.
///
/// Downstream middleware (rate limiting, audit logging) consume this rather
/// than re-parsing the `Authorization` header.
#[derive(Debug, Clone, Copy)]
pub struct AuthContext {
    /// Stable identifier for the authenticated bearer token. See [`TokenId`].
    pub token_id: TokenId,
}

impl AuthContext {
    /// Construct an [`AuthContext`] for a successfully-authenticated token.
    pub fn for_token(token: &str) -> Self {
        Self {
            token_id: TokenId::from_bearer(token),
        }
    }

    /// Construct the dev-mode pass-through context.
    pub fn dev() -> Self {
        Self {
            token_id: TokenId::DEV,
        }
    }
}

/// Static configuration for the per-token rate limiter.
///
/// Loaded from `TENSOR_WASM_API_RATE_LIMIT_QPS` and
/// `TENSOR_WASM_API_RATE_LIMIT_BURST` at server startup. If either knob is
/// zero (or unset) the limiter is disabled — see [`RateLimitConfig::is_disabled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitConfig {
    /// Steady-state requests-per-second admitted per token.
    pub qps: u32,
    /// Maximum burst — the bucket capacity, in permits.
    pub burst: u32,
}

impl RateLimitConfig {
    /// Environment variable carrying the steady-state QPS allowance per token.
    pub const ENV_QPS: &'static str = "TENSOR_WASM_API_RATE_LIMIT_QPS";

    /// Environment variable carrying the burst (bucket capacity) per token.
    pub const ENV_BURST: &'static str = "TENSOR_WASM_API_RATE_LIMIT_BURST";

    /// Default QPS applied when [`ENV_QPS`] is unset but [`ENV_BURST`] is set.
    pub const DEFAULT_QPS: u32 = 100;

    /// Default burst applied when [`ENV_BURST`] is unset but [`ENV_QPS`] is set.
    pub const DEFAULT_BURST: u32 = 200;

    /// Disabled config: both fields zero. The middleware is a pass-through.
    pub const fn disabled() -> Self {
        Self { qps: 0, burst: 0 }
    }

    /// `true` if the limiter is disabled (either knob is zero).
    pub const fn is_disabled(&self) -> bool {
        self.qps == 0 || self.burst == 0
    }

    /// Load from the process environment.
    ///
    /// * Both vars unset / either `"0"` / either unparseable => [`Self::disabled`].
    /// * Otherwise: missing-but-other-side-set falls back to
    ///   [`DEFAULT_QPS`](Self::DEFAULT_QPS) / [`DEFAULT_BURST`](Self::DEFAULT_BURST).
    pub fn from_env() -> Self {
        let qps_raw = std::env::var(Self::ENV_QPS).ok();
        let burst_raw = std::env::var(Self::ENV_BURST).ok();
        if qps_raw.is_none() && burst_raw.is_none() {
            return Self::disabled();
        }
        let qps = qps_raw
            .as_deref()
            .map(|s| s.trim().parse::<u32>().unwrap_or(0))
            .unwrap_or(Self::DEFAULT_QPS);
        let burst = burst_raw
            .as_deref()
            .map(|s| s.trim().parse::<u32>().unwrap_or(0))
            .unwrap_or(Self::DEFAULT_BURST);
        let cfg = Self { qps, burst };
        if cfg.is_disabled() {
            tracing::warn!(
                target: "tensor_wasm_api::rate_limit",
                qps,
                burst,
                "{} / {} parsed but yields a disabled limiter (qps==0 or burst==0)",
                Self::ENV_QPS,
                Self::ENV_BURST,
            );
            return Self::disabled();
        }
        tracing::info!(
            target: "tensor_wasm_api::rate_limit",
            qps,
            burst,
            "per-token rate limiter enabled",
        );
        cfg
    }
}

impl Default for RateLimitConfig {
    /// Default is *disabled*. Operators opt in by setting both env vars.
    fn default() -> Self {
        Self::disabled()
    }
}

/// Abstract monotonic clock. Implemented by [`RealClock`] (production) and
/// [`ManualClock`] (tests).
pub trait Clock: Send + Sync + 'static {
    /// Return the current monotonic [`Instant`].
    fn now(&self) -> Instant;
}

/// Production clock: delegates to [`Instant::now`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Test clock: holds an explicit [`Instant`] that the test advances.
#[derive(Debug, Clone)]
pub struct ManualClock {
    inner: Arc<Mutex<Instant>>,
}

impl ManualClock {
    /// Construct a [`ManualClock`] seeded at `Instant::now()`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Advance the clock by `d`.
    pub fn advance(&self, d: Duration) {
        let mut g = self.inner.lock().expect("ManualClock mutex poisoned");
        *g += d;
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        *self.inner.lock().expect("ManualClock mutex poisoned")
    }
}

/// Per-token bucket state. Protected by a `std::sync::Mutex`.
#[derive(Debug)]
struct BucketState {
    /// Current permit balance. Stored as `f64` so refill arithmetic does
    /// not lose sub-permit progress between requests at QPS values that
    /// don't divide evenly into a millisecond.
    tokens: f64,
    /// Monotonic instant of the most recent refill calculation.
    last_refill: Instant,
}

/// Outcome of an attempt to claim a permit from the bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmitResult {
    /// Request admitted; one permit was deducted.
    Admit,
    /// Request rejected; carries the suggested `Retry-After` value (in
    /// whole seconds, rounded up — HTTP `Retry-After` is integer seconds
    /// when not a date).
    Reject {
        /// Seconds the client should wait before retrying.
        retry_after_secs: u64,
    },
}

/// In-process per-token rate limiter.
///
/// Cheaply cloneable: every clone shares the same underlying [`DashMap`] and
/// [`Clock`] via [`Arc`].
#[derive(Clone)]
pub struct RateLimiter {
    cfg: RateLimitConfig,
    clock: Arc<dyn Clock>,
    buckets: Arc<DashMap<TokenId, Mutex<BucketState>>>,
}

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("cfg", &self.cfg)
            .field("buckets", &self.buckets.len())
            .finish()
    }
}

impl RateLimiter {
    /// Construct a limiter with the production [`RealClock`].
    pub fn new(cfg: RateLimitConfig) -> Self {
        Self::with_clock(cfg, Arc::new(RealClock))
    }

    /// Construct a limiter with an injected clock (for tests).
    pub fn with_clock(cfg: RateLimitConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            cfg,
            clock,
            buckets: Arc::new(DashMap::new()),
        }
    }

    /// `true` if the configured policy admits every request unconditionally.
    pub fn is_disabled(&self) -> bool {
        self.cfg.is_disabled()
    }

    /// Effective configuration this limiter was built with.
    pub fn config(&self) -> RateLimitConfig {
        self.cfg
    }

    /// Attempt to claim one permit for `token`.
    pub fn try_admit(&self, token: TokenId) -> AdmitResult {
        if self.is_disabled() {
            return AdmitResult::Admit;
        }
        let now = self.clock.now();
        // Per-bucket lookup. DashMap's entry API handles concurrent insert.
        let entry = self.buckets.entry(token).or_insert_with(|| {
            Mutex::new(BucketState {
                // Fresh bucket starts full: a new caller gets to spend the
                // whole burst before throttling kicks in.
                tokens: self.cfg.burst as f64,
                last_refill: now,
            })
        });
        let mut state = entry
            .value()
            .lock()
            .expect("RateLimiter bucket mutex poisoned");
        // Refill.
        let elapsed = now.saturating_duration_since(state.last_refill);
        if elapsed > Duration::ZERO {
            let refill = elapsed.as_secs_f64() * self.cfg.qps as f64;
            state.tokens = (state.tokens + refill).min(self.cfg.burst as f64);
            state.last_refill = now;
        }
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            AdmitResult::Admit
        } else {
            // Compute how long until one full permit refills. Tokens deficit
            // is `1.0 - state.tokens` (in (0, 1]); time = deficit / qps.
            let deficit = 1.0 - state.tokens;
            let secs = (deficit / self.cfg.qps as f64).ceil() as u64;
            AdmitResult::Reject {
                // Always suggest at least 1s so misbehaving clients back off
                // a measurable amount even when qps is very high.
                retry_after_secs: secs.max(1),
            }
        }
    }
}

/// Render the standard `{ "error": { "kind": ..., "message": ... } }`
/// envelope at `status`, attaching a `Retry-After` header.
fn rate_limited_response(retry_after_secs: u64) -> Response {
    let body = Json(json!({
        "error": {
            "kind": "rate_limited",
            "message": format!(
                "per-token rate limit exceeded; retry after {retry_after_secs}s",
            ),
        }
    }));
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, body).into_response();
    // `Retry-After` per RFC 9110 §10.2.3 may be either an HTTP-date or a
    // non-negative decimal integer of seconds. We emit the latter.
    if let Ok(hv) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        resp.headers_mut().insert(axum::http::header::RETRY_AFTER, hv);
    }
    resp
}

/// Axum middleware that enforces the per-token rate limit.
///
/// Reads [`AuthContext`] from request extensions (inserted by
/// [`crate::middleware::bearer_auth`]) and consults the [`RateLimiter`]
/// supplied via an `axum::Extension`. On bucket-empty, returns
/// `429 Too Many Requests` with a `Retry-After` header.
///
/// When no [`RateLimiter`] is in the extensions the middleware is a
/// pass-through (the operator did not configure rate limiting).
pub async fn rate_limit(req: Request, next: Next) -> Response {
    let limiter = match req.extensions().get::<RateLimiter>().cloned() {
        Some(l) => l,
        None => return next.run(req).await,
    };
    if limiter.is_disabled() {
        return next.run(req).await;
    }
    let token = req
        .extensions()
        .get::<AuthContext>()
        .copied()
        .map(|c| c.token_id)
        // Defensive: if the auth middleware was somehow bypassed, fold all
        // un-authed requests into the dev bucket so they still face the
        // configured cap. This should not happen in the production stack.
        .unwrap_or(TokenId::DEV);
    match limiter.try_admit(token) {
        AdmitResult::Admit => next.run(req).await,
        AdmitResult::Reject { retry_after_secs } => rate_limited_response(retry_after_secs),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Method, Request};
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn cfg(qps: u32, burst: u32) -> RateLimitConfig {
        RateLimitConfig { qps, burst }
    }

    #[test]
    fn config_disabled_when_either_zero() {
        assert!(cfg(0, 10).is_disabled());
        assert!(cfg(10, 0).is_disabled());
        assert!(cfg(0, 0).is_disabled());
        assert!(!cfg(1, 1).is_disabled());
    }

    #[test]
    fn token_id_dev_is_distinct_from_real_tokens() {
        assert_ne!(TokenId::from_bearer("anything").0, TokenId::DEV.0);
        // Stable within a process for the same input.
        assert_eq!(
            TokenId::from_bearer("alpha").0,
            TokenId::from_bearer("alpha").0
        );
        assert_ne!(
            TokenId::from_bearer("alpha").0,
            TokenId::from_bearer("beta").0
        );
    }

    #[test]
    fn bucket_allows_up_to_burst_immediately() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::with_clock(cfg(10, 5), clock.clone());
        let tok = TokenId::from_bearer("alpha");
        for i in 0..5 {
            assert!(
                matches!(limiter.try_admit(tok), AdmitResult::Admit),
                "burst slot {i} should admit",
            );
        }
        // 6th request in the same instant: rejected.
        assert!(matches!(
            limiter.try_admit(tok),
            AdmitResult::Reject { .. },
        ));
    }

    #[test]
    fn bucket_refills_at_qps_rate_with_manual_clock() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::with_clock(cfg(10, 5), clock.clone());
        let tok = TokenId::from_bearer("alpha");
        // Drain the bucket.
        for _ in 0..5 {
            assert!(matches!(limiter.try_admit(tok), AdmitResult::Admit));
        }
        assert!(matches!(
            limiter.try_admit(tok),
            AdmitResult::Reject { .. },
        ));
        // Advance enough wall-time to refill exactly one permit at 10 qps.
        clock.advance(Duration::from_millis(100));
        assert!(matches!(limiter.try_admit(tok), AdmitResult::Admit));
        // Immediately after: empty again.
        assert!(matches!(
            limiter.try_admit(tok),
            AdmitResult::Reject { .. },
        ));
        // Advance enough to refill the entire burst.
        clock.advance(Duration::from_secs(1));
        for _ in 0..5 {
            assert!(matches!(limiter.try_admit(tok), AdmitResult::Admit));
        }
        assert!(matches!(
            limiter.try_admit(tok),
            AdmitResult::Reject { .. },
        ));
    }

    #[test]
    fn separate_tokens_have_separate_buckets() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::with_clock(cfg(1, 2), clock.clone());
        let a = TokenId::from_bearer("alpha");
        let b = TokenId::from_bearer("beta");
        for _ in 0..2 {
            assert!(matches!(limiter.try_admit(a), AdmitResult::Admit));
        }
        // A is drained.
        assert!(matches!(limiter.try_admit(a), AdmitResult::Reject { .. }));
        // B is untouched.
        for _ in 0..2 {
            assert!(matches!(limiter.try_admit(b), AdmitResult::Admit));
        }
        assert!(matches!(limiter.try_admit(b), AdmitResult::Reject { .. }));
    }

    #[test]
    fn disabled_limiter_admits_unconditionally() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::with_clock(RateLimitConfig::disabled(), clock);
        let tok = TokenId::from_bearer("alpha");
        for _ in 0..1000 {
            assert!(matches!(limiter.try_admit(tok), AdmitResult::Admit));
        }
    }

    #[test]
    fn reject_carries_retry_after_at_least_one_second() {
        let clock = Arc::new(ManualClock::new());
        // qps=1000, burst=1 → very fast refill, but we still want >=1s back-off.
        let limiter = RateLimiter::with_clock(cfg(1000, 1), clock.clone());
        let tok = TokenId::from_bearer("alpha");
        assert!(matches!(limiter.try_admit(tok), AdmitResult::Admit));
        match limiter.try_admit(tok) {
            AdmitResult::Reject { retry_after_secs } => {
                assert!(retry_after_secs >= 1, "got {retry_after_secs}");
            }
            other => panic!("expected reject, got {other:?}"),
        }
    }

    /// Drives a minimal router that exercises just the rate-limit middleware
    /// in front of a 200-handler, so we can assert the integration layer
    /// emits 429 + `Retry-After` exactly as specified.
    async fn drive_tower(
        limiter: RateLimiter,
        auth: AuthContext,
        n: usize,
    ) -> Vec<(StatusCode, Option<String>)> {
        // Per-test inline handler: returns 204 if it ran, so any non-204
        // status had to come from the middleware short-circuit.
        async fn ok() -> StatusCode {
            StatusCode::NO_CONTENT
        }
        let router = Router::new()
            .route("/probe", get(ok))
            .layer(axum::middleware::from_fn(rate_limit))
            .layer(axum::Extension(limiter))
            .layer(axum::Extension(auth));

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let req = Request::builder()
                .method(Method::GET)
                .uri("/probe")
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.unwrap();
            let status = resp.status();
            let retry = resp
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            out.push((status, retry));
        }
        out
    }

    #[tokio::test]
    async fn middleware_admits_burst_then_rejects_with_retry_after() {
        let clock = Arc::new(ManualClock::new());
        // burst=3 → exactly 3 requests get through before a 429.
        let limiter = RateLimiter::with_clock(cfg(1, 3), clock.clone());
        let auth = AuthContext::for_token("alpha");
        let results = drive_tower(limiter, auth, 5).await;

        assert_eq!(results[0].0, StatusCode::NO_CONTENT);
        assert_eq!(results[1].0, StatusCode::NO_CONTENT);
        assert_eq!(results[2].0, StatusCode::NO_CONTENT);
        assert_eq!(results[3].0, StatusCode::TOO_MANY_REQUESTS);
        assert!(results[3].1.is_some(), "Retry-After header missing");
        assert_eq!(results[4].0, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn middleware_passthrough_when_no_limiter_in_extensions() {
        // No `Extension(RateLimiter)` layer — middleware should pass through.
        async fn ok() -> StatusCode {
            StatusCode::NO_CONTENT
        }
        let router = Router::new()
            .route("/probe", get(ok))
            .layer(axum::middleware::from_fn(rate_limit));
        for _ in 0..50 {
            let req = Request::builder()
                .method(Method::GET)
                .uri("/probe")
                .body(Body::empty())
                .unwrap();
            let resp = router.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        }
    }

    #[tokio::test]
    async fn middleware_passthrough_when_limiter_disabled() {
        let limiter = RateLimiter::new(RateLimitConfig::disabled());
        let auth = AuthContext::for_token("alpha");
        let results = drive_tower(limiter, auth, 20).await;
        for (i, (status, _)) in results.iter().enumerate() {
            assert_eq!(*status, StatusCode::NO_CONTENT, "request {i}");
        }
    }
}
