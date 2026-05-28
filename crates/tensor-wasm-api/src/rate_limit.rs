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
use serde::Serialize;
use serde_json::json;
use tensor_wasm_core::types::TenantId;

use crate::routes::ApiError;
use crate::token_scope::TokenScope;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
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
/// Downstream middleware (rate limiting, audit logging) and route handlers
/// (tenant-scope authorization) consume this rather than re-parsing the
/// `Authorization` header.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Stable identifier for the authenticated bearer token. See [`TokenId`].
    pub token_id: TokenId,
    /// Tenants this token is authorised to address. Populated by the bearer
    /// auth middleware from the parsed [`crate::token_scope::ParsedTokens`]
    /// map. Dev-mode contexts default to [`TokenScope::all`].
    ///
    /// Handlers that bind to a tenant call [`AuthContext::authorize_tenant`]
    /// before doing per-tenant work.
    pub scope: TokenScope,
}

impl AuthContext {
    /// Construct an [`AuthContext`] for a successfully-authenticated token,
    /// defaulting the scope to wildcard. Retained as a back-compat helper
    /// for tests that pre-date scoped tokens; production code goes through
    /// [`AuthContext::with_scope`].
    pub fn for_token(token: &str) -> Self {
        Self {
            token_id: TokenId::from_bearer(token),
            scope: TokenScope::all(),
        }
    }

    /// Construct an [`AuthContext`] with an explicit scope.
    pub fn with_scope(token: &str, scope: TokenScope) -> Self {
        Self {
            token_id: TokenId::from_bearer(token),
            scope,
        }
    }

    /// Construct the dev-mode pass-through context. Dev mode always grants
    /// the wildcard scope — the operator already opted out of auth by
    /// leaving the allowlist empty, so per-tenant gating would be theatre.
    pub fn dev() -> Self {
        Self {
            token_id: TokenId::DEV,
            scope: TokenScope::all(),
        }
    }

    /// Return `Ok(())` if this token may address `tenant`, otherwise an
    /// [`ApiError`] with `kind: "tenant_scope_denied"`. Routes that bind to
    /// a tenant call this before doing any per-tenant work.
    pub fn authorize_tenant(&self, tenant: TenantId) -> Result<(), ApiError> {
        if self.scope.allows(tenant) {
            Ok(())
        } else {
            Err(ApiError::forbidden(
                "tenant_scope_denied",
                format!(
                    "bearer token is not scoped to tenant {}; \
                     extend the token's tenant= clause in TENSOR_WASM_API_TOKENS",
                    tenant.0,
                ),
            ))
        }
    }
}

/// Per-tenant (secondary) rate-limit configuration.
///
/// Sits in front of the per-token bucket as the *primary* defence against a
/// noisy-neighbour tenant saturating a shared token's overall quota. The
/// per-token bucket (see [`RateLimitConfig`]) is retained as a backstop.
///
/// Composite bucket key is `(TokenId, TenantId)` — so a single token used by
/// two tenants does not let one tenant drain the other's allowance.
///
/// **Field semantics:**
/// * `burst == 0` => disabled (this layer admits unconditionally). The per-
///   token backstop still applies if it is itself configured.
/// * `qps == 0.0` => non-zero burst is a one-shot allowance with **no
///   refill**. Useful for tests; in production an operator who wants no
///   per-tenant ceiling should set `burst = 0` to disable the layer outright.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerTenantRateLimitConfig {
    /// Maximum burst — the per-tenant bucket capacity, in permits.
    pub burst: u32,
    /// Steady-state requests-per-second admitted per `(token, tenant)` pair.
    /// `0.0` disables refill (the bucket drains and stays empty until process
    /// restart).
    pub qps: f64,
}

impl PerTenantRateLimitConfig {
    /// Default per-tenant burst. Deliberately conservative so a misbehaving
    /// tenant on a shared token cannot trample neighbours; operators tune
    /// upward as their multi-tenant workload demands.
    pub const DEFAULT_BURST: u32 = 20;

    /// Default per-tenant steady-state QPS. Matches the conservative
    /// [`DEFAULT_BURST`](Self::DEFAULT_BURST) shape — sized for the small
    /// internal tenant fleet today; operators raise it as needed.
    pub const DEFAULT_QPS: f64 = 10.0;

    /// Disabled config: per-tenant layer admits unconditionally.
    pub const fn disabled() -> Self {
        Self {
            burst: 0,
            qps: 0.0,
        }
    }

    /// `true` if the per-tenant layer is disabled. Determined solely by
    /// `burst == 0`: a non-zero burst with `qps == 0.0` is a valid (no-
    /// refill) configuration, not a disabled one.
    pub const fn is_disabled(&self) -> bool {
        self.burst == 0
    }
}

impl Default for PerTenantRateLimitConfig {
    /// Default to the conservative active configuration
    /// (`burst = 20`, `qps = 10.0`). Operators reach the fully-disabled
    /// posture via [`PerTenantRateLimitConfig::disabled`].
    fn default() -> Self {
        Self {
            burst: Self::DEFAULT_BURST,
            qps: Self::DEFAULT_QPS,
        }
    }
}

/// Static configuration for the rate limiter.
///
/// Two layers, both enforced (whichever is tighter wins):
///
/// 1. **Per-tenant bucket** keyed on `(TokenId, TenantId)`
///    ([`per_tenant_default`](Self::per_tenant_default)) — primary defence.
///    Prevents one tenant from saturating a shared token's quota.
/// 2. **Per-token bucket** keyed on `TokenId` ([`qps`](Self::qps),
///    [`burst`](Self::burst)) — backstop. Caps aggregate usage by a single
///    token across all tenants.
///
/// Token-level knobs come from `TENSOR_WASM_API_RATE_LIMIT_QPS` and
/// `TENSOR_WASM_API_RATE_LIMIT_BURST` at server startup; per-tenant defaults
/// to [`PerTenantRateLimitConfig::default`]. If both knobs are zero (or
/// unset) the token-level backstop is disabled, but the per-tenant layer is
/// still in force unless explicitly cleared — see
/// [`RateLimitConfig::is_disabled`].
///
/// Note: this type is no longer `Eq` because [`PerTenantRateLimitConfig::qps`]
/// is `f64`. Use `PartialEq` for comparisons in tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimitConfig {
    /// Steady-state requests-per-second admitted per token (backstop layer).
    pub qps: u32,
    /// Maximum burst — the per-token bucket capacity, in permits.
    pub burst: u32,
    /// Default per-tenant configuration applied to every `(token, tenant)`
    /// pair. The primary defence against a single tenant exhausting a
    /// shared token's quota.
    pub per_tenant_default: PerTenantRateLimitConfig,
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

    /// Disabled config: every layer off. The middleware is a pass-through.
    pub const fn disabled() -> Self {
        Self {
            qps: 0,
            burst: 0,
            per_tenant_default: PerTenantRateLimitConfig::disabled(),
        }
    }

    /// `true` if every layer of the limiter is disabled and the middleware
    /// would unconditionally admit. Used by [`rate_limit`] to short-circuit
    /// the bucket lookup entirely.
    pub const fn is_disabled(&self) -> bool {
        self.is_token_layer_disabled() && self.per_tenant_default.is_disabled()
    }

    /// `true` if the per-token (backstop) layer is disabled.
    pub const fn is_token_layer_disabled(&self) -> bool {
        self.qps == 0 || self.burst == 0
    }

    /// Load from the process environment.
    ///
    /// * Both vars unset / either `"0"` / either unparseable => token-layer
    ///   disabled. The per-tenant layer still defaults to
    ///   [`PerTenantRateLimitConfig::default`].
    /// * Otherwise: missing-but-other-side-set falls back to
    ///   [`DEFAULT_QPS`](Self::DEFAULT_QPS) / [`DEFAULT_BURST`](Self::DEFAULT_BURST).
    pub fn from_env() -> Self {
        let per_tenant_default = PerTenantRateLimitConfig::default();
        let qps_raw = std::env::var(Self::ENV_QPS).ok();
        let burst_raw = std::env::var(Self::ENV_BURST).ok();
        if qps_raw.is_none() && burst_raw.is_none() {
            return Self {
                qps: 0,
                burst: 0,
                per_tenant_default,
            };
        }
        let qps = qps_raw
            .as_deref()
            .map(|s| s.trim().parse::<u32>().unwrap_or(0))
            .unwrap_or(Self::DEFAULT_QPS);
        let burst = burst_raw
            .as_deref()
            .map(|s| s.trim().parse::<u32>().unwrap_or(0))
            .unwrap_or(Self::DEFAULT_BURST);
        let cfg = Self {
            qps,
            burst,
            per_tenant_default,
        };
        if cfg.is_token_layer_disabled() {
            tracing::warn!(
                target: "tensor_wasm_api::rate_limit",
                qps,
                burst,
                "{} / {} parsed but yields a disabled token-layer limiter (qps==0 or burst==0); per-tenant layer still active",
                Self::ENV_QPS,
                Self::ENV_BURST,
            );
            return Self {
                qps: 0,
                burst: 0,
                per_tenant_default,
            };
        }
        tracing::info!(
            target: "tensor_wasm_api::rate_limit",
            qps,
            burst,
            per_tenant_burst = per_tenant_default.burst,
            per_tenant_qps = per_tenant_default.qps,
            "rate limiter enabled (per-token backstop + per-tenant primary)",
        );
        cfg
    }
}

impl Default for RateLimitConfig {
    /// Default is *disabled*. Operators opt in by setting both env vars (or
    /// by constructing a config explicitly).
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

/// In-process two-layer rate limiter.
///
/// Layer 1 (primary): `(TokenId, TenantId)` bucket — keeps a shared token
/// from being drained by a single tenant.
///
/// Layer 2 (backstop): `TokenId` bucket — caps aggregate usage by a single
/// token across all tenants. Inherited from the v0.4 design; kept active so
/// pre-multi-tenant operators see no behavioural regression.
///
/// Cheaply cloneable: every clone shares the same underlying [`DashMap`]s
/// and [`Clock`] via [`Arc`].
#[derive(Clone)]
pub struct RateLimiter {
    cfg: RateLimitConfig,
    clock: Arc<dyn Clock>,
    /// Per-token (backstop) buckets.
    buckets: Arc<DashMap<TokenId, Mutex<BucketState>>>,
    /// Per-(token, tenant) (primary) buckets. We use a composite key so a
    /// single shared token still gets per-tenant isolation. With the dev
    /// sentinel token, this also separates internal-cron tenants from
    /// external traffic that lands on `TokenId::DEV`.
    per_tenant_buckets: Arc<DashMap<(TokenId, TenantId), Mutex<BucketState>>>,
}

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("cfg", &self.cfg)
            .field("buckets", &self.buckets.len())
            .field("per_tenant_buckets", &self.per_tenant_buckets.len())
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
            per_tenant_buckets: Arc::new(DashMap::new()),
        }
    }

    /// `true` if every configured layer admits unconditionally.
    pub fn is_disabled(&self) -> bool {
        self.cfg.is_disabled()
    }

    /// Effective configuration this limiter was built with.
    pub fn config(&self) -> RateLimitConfig {
        self.cfg
    }

    /// Attempt to claim one permit for the `(token, tenant)` pair.
    ///
    /// Both the per-tenant (primary) and per-token (backstop) buckets must
    /// admit. If either rejects we return [`AdmitResult::Reject`] carrying
    /// the **smaller** of the two retry [`Duration`]s — per the
    /// per-tenant-bucket design note, the smaller backoff is the earliest
    /// the client could plausibly retry, even though it may still face the
    /// other bucket on the next attempt.
    ///
    /// To avoid leaking a permit on one layer when the other rejects, we
    /// hold both layers' inner mutexes across the decision and only deduct
    /// when *both* would admit. The lock-order is `(token, tenant)` then
    /// `token`; since these live in two distinct [`DashMap`]s and every
    /// caller takes them in the same order, no cycle is possible.
    pub fn try_admit(&self, token: TokenId, tenant: TenantId) -> AdmitResult {
        if self.is_disabled() {
            return AdmitResult::Admit;
        }
        let now = self.clock.now();

        // Acquire entries for whichever layers are active. We hold the
        // DashMap entries (RefMut) for the full critical section so the
        // inner Mutex guards stay valid; the underlying shards stay locked
        // only for the brief Mutex lock/unlock, not the whole decision.
        let per_tenant_burst = self.cfg.per_tenant_default.burst as f64;
        let per_tenant_qps = self.cfg.per_tenant_default.qps;
        let token_burst = self.cfg.burst as f64;
        let token_qps = self.cfg.qps as f64;

        let per_tenant_entry = if self.cfg.per_tenant_default.is_disabled() {
            None
        } else {
            Some(
                self.per_tenant_buckets
                    .entry((token, tenant))
                    .or_insert_with(|| {
                        Mutex::new(BucketState {
                            tokens: per_tenant_burst,
                            last_refill: now,
                        })
                    }),
            )
        };
        let token_entry = if self.cfg.is_token_layer_disabled() {
            None
        } else {
            Some(self.buckets.entry(token).or_insert_with(|| {
                Mutex::new(BucketState {
                    tokens: token_burst,
                    last_refill: now,
                })
            }))
        };

        // Lock both buckets, in a fixed order, for the whole decision.
        let mut per_tenant_guard = per_tenant_entry.as_ref().map(|e| {
            e.value()
                .lock()
                .expect("RateLimiter per-tenant bucket mutex poisoned")
        });
        let mut token_guard = token_entry.as_ref().map(|e| {
            e.value()
                .lock()
                .expect("RateLimiter token bucket mutex poisoned")
        });

        let per_tenant_decision = per_tenant_guard
            .as_deref_mut()
            .map(|s| refill_and_decide(s, per_tenant_burst, per_tenant_qps, now));
        let token_decision = token_guard
            .as_deref_mut()
            .map(|s| refill_and_decide(s, token_burst, token_qps, now));

        let per_tenant_admit = per_tenant_decision.as_ref().is_none_or(|d| d.admittable);
        let token_admit = token_decision.as_ref().is_none_or(|d| d.admittable);

        if per_tenant_admit && token_admit {
            if let Some(state) = per_tenant_guard.as_deref_mut() {
                state.tokens -= 1.0;
            }
            if let Some(state) = token_guard.as_deref_mut() {
                state.tokens -= 1.0;
            }
            return AdmitResult::Admit;
        }

        // At least one layer rejected. Per spec: signal with the SMALLER of
        // the two retry durations. (An admitting layer contributes nothing
        // — its implied duration is zero; we only consider durations from
        // layers that themselves rejected.)
        let mut chosen: Option<Duration> = None;
        for d in [per_tenant_decision.as_ref(), token_decision.as_ref()]
            .into_iter()
            .flatten()
        {
            if !d.admittable {
                chosen = Some(match chosen {
                    None => d.retry_after,
                    Some(prev) => prev.min(d.retry_after),
                });
            }
        }
        let retry = chosen.unwrap_or(Duration::from_secs(1));
        let secs = retry.as_secs_f64().ceil() as u64;
        AdmitResult::Reject {
            // Always suggest at least 1s so misbehaving clients back off a
            // measurable amount even when qps is very high.
            retry_after_secs: secs.max(1),
        }
    }
}

/// Per-layer decision returned by `refill_and_decide`.
struct BucketDecision {
    /// `true` if this layer alone would admit the request.
    admittable: bool,
    /// Retry hint for this layer if `!admittable`. Zero when `admittable`.
    retry_after: Duration,
}

/// Refill the bucket in place (updating `last_refill`) and report whether
/// it currently has at least one full permit. Does *not* deduct — the
/// caller subtracts one only after both layers agree to admit.
fn refill_and_decide(
    state: &mut BucketState,
    burst: f64,
    qps: f64,
    now: Instant,
) -> BucketDecision {
    let elapsed = now.saturating_duration_since(state.last_refill);
    if elapsed > Duration::ZERO && qps > 0.0 {
        state.tokens = (state.tokens + elapsed.as_secs_f64() * qps).min(burst);
    } else {
        state.tokens = state.tokens.min(burst);
    }
    state.last_refill = now;
    if state.tokens >= 1.0 {
        BucketDecision {
            admittable: true,
            retry_after: Duration::ZERO,
        }
    } else {
        let deficit = 1.0 - state.tokens;
        let retry = if qps > 0.0 {
            Duration::from_secs_f64((deficit / qps).max(0.0))
        } else {
            // No refill ever — surface a large but finite hint. We pick 1h
            // so it is visibly "go away" without being u64::MAX. Tests for
            // the no-refill case only assert reject, never the magnitude.
            Duration::from_secs(3600)
        };
        BucketDecision {
            admittable: false,
            retry_after: retry,
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
        .map(|c| c.token_id)
        // Defensive: if the auth middleware was somehow bypassed, fold all
        // un-authed requests into the dev bucket so they still face the
        // configured cap. This should not happen in the production stack.
        .unwrap_or(TokenId::DEV);
    // Per-tenant rate-limit (api S-25): the tenant_scope middleware sets
    // a tenant extension on the request. Fall back to TenantId(0) for the
    // unauthenticated / probe paths so they share a single bucket.
    let tenant = req
        .extensions()
        .get::<tensor_wasm_core::types::TenantId>()
        .copied()
        .unwrap_or(tensor_wasm_core::types::TenantId(0));
    match limiter.try_admit(token, tenant) {
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
        // Disable the per-tenant layer so these tests exercise the token
        // (backstop) layer in isolation; the per-tenant primary layer has
        // its own dedicated tests elsewhere.
        RateLimitConfig {
            qps,
            burst,
            per_tenant_default: PerTenantRateLimitConfig::disabled(),
        }
    }

    /// Convenience: the tenant every inline unit test in this module pins
    /// to. Pre-multi-tenant tests only need a single stable value here.
    const TENANT: TenantId = TenantId(1);

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
                matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit),
                "burst slot {i} should admit",
            );
        }
        // 6th request in the same instant: rejected.
        assert!(matches!(
            limiter.try_admit(tok, TENANT),
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
            assert!(matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit));
        }
        assert!(matches!(
            limiter.try_admit(tok, TENANT),
            AdmitResult::Reject { .. },
        ));
        // Advance enough wall-time to refill exactly one permit at 10 qps.
        clock.advance(Duration::from_millis(100));
        assert!(matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit));
        // Immediately after: empty again.
        assert!(matches!(
            limiter.try_admit(tok, TENANT),
            AdmitResult::Reject { .. },
        ));
        // Advance enough to refill the entire burst.
        clock.advance(Duration::from_secs(1));
        for _ in 0..5 {
            assert!(matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit));
        }
        assert!(matches!(
            limiter.try_admit(tok, TENANT),
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
            assert!(matches!(limiter.try_admit(a, TENANT), AdmitResult::Admit));
        }
        // A is drained.
        assert!(matches!(limiter.try_admit(a, TENANT), AdmitResult::Reject { .. }));
        // B is untouched.
        for _ in 0..2 {
            assert!(matches!(limiter.try_admit(b, TENANT), AdmitResult::Admit));
        }
        assert!(matches!(limiter.try_admit(b, TENANT), AdmitResult::Reject { .. }));
    }

    #[test]
    fn disabled_limiter_admits_unconditionally() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::with_clock(RateLimitConfig::disabled(), clock);
        let tok = TokenId::from_bearer("alpha");
        for _ in 0..1000 {
            assert!(matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit));
        }
    }

    #[test]
    fn reject_carries_retry_after_at_least_one_second() {
        let clock = Arc::new(ManualClock::new());
        // qps=1000, burst=1 → very fast refill, but we still want >=1s back-off.
        let limiter = RateLimiter::with_clock(cfg(1000, 1), clock.clone());
        let tok = TokenId::from_bearer("alpha");
        assert!(matches!(limiter.try_admit(tok, TENANT), AdmitResult::Admit));
        match limiter.try_admit(tok, TENANT) {
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
