// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Axum router builder and listener.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::json;
use subtle::ConstantTimeEq;
use tower::ServiceBuilder;

use crate::audit::{audit_log_middleware, AuditConfig, TrustedProxies};
use crate::http_metrics::{http_metrics_middleware, HttpMetricsLayerConfig, RouteAllowList};
use crate::middleware::{
    bearer_auth, body_limit_layer, concurrency_limit_layer, cors_layer, host_validate,
    tenant_scope, timeout_layer, trace_layer_with_propagation, AuthConfig, CorsConfig,
    KernelPublishTokens, TenantConfig, TrustedHosts, ENV_API_TOKENS, ENV_TRUSTED_HOSTS,
    INVOKE_CONCURRENCY_LIMIT, MAX_REQUEST_BODY_BYTES, PROBE_CONCURRENCY_LIMIT,
    READ_CONCURRENCY_LIMIT, WRITE_CONCURRENCY_LIMIT,
};
use crate::rate_limit::{rate_limit, RateLimitConfig, RateLimiter};
use crate::routes::{
    create_function, delete_function, get_job, healthz, invoke_function, invoke_function_async,
    invoke_function_stream, metrics, snapshot_restore, snapshot_save, AppState,
};
use crate::trace_propagation::{inject_trace_id_header, install_w3c_propagator};

/// Build the axum Router with all routes and middleware applied.
///
/// Reads [`AuthConfig`] (`$TENSOR_WASM_API_TOKENS`), [`TenantConfig`]
/// (`$TENSOR_WASM_API_REQUIRE_TENANT`), [`RateLimitConfig`]
/// (`$TENSOR_WASM_API_RATE_LIMIT_QPS` / `$TENSOR_WASM_API_RATE_LIMIT_BURST`),
/// and [`CorsConfig`] (`$TENSOR_WASM_API_CORS_ALLOWED_ORIGINS`) from the
/// process environment. Empty / unset `TENSOR_WASM_API_TOKENS` puts the
/// gateway in dev mode (auth disabled with a startup warning); unset or
/// zero rate-limit knobs disable the limiter (pass-through); empty / unset
/// CORS allowlist rejects every cross-origin request (safe default).
///
/// If any bare (unscoped) entries are present in `TENSOR_WASM_API_TOKENS`,
/// `AuthConfig::from_env` emits a one-shot deprecation warning naming the
/// count — scoped tokens (`token:tenant=...`) are the supported form going
/// forward and bare entries are scheduled for removal in v1.0.
pub fn build_router(state: Arc<AppState>) -> Router {
    let auth = AuthConfig::from_env();
    let tenant = TenantConfig::from_env();
    let limiter = RateLimiter::new(RateLimitConfig::from_env());
    let audit = AuditConfig::from_env();
    let cors = CorsConfig::from_env();
    // M5: read the top-level `AppConfig` (snapshot HMAC signing key +
    // require-signature toggle) from the environment and thread it onto
    // the shared `AppState` so the `/snapshot/save` and `/snapshot/restore`
    // handlers consume `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY`. This is the
    // production caller that turns the previously-inert key into a
    // live knob (closes finding M5). A malformed key is a hard startup
    // error — we refuse to come up serving snapshot routes under a
    // misconfigured signing key rather than silently degrade.
    let state = apply_app_config_from_env(state);
    build_router_with_audit(state, auth, tenant, limiter, audit, cors)
}

/// Read [`AppConfig::from_env`](crate::config::AppConfig::from_env) and
/// install it onto `state` so the snapshot routes pick up the operator's
/// signing key.
///
/// On a malformed snapshot key / toggle this logs the parse error and
/// **panics** — unlike the kernel registry (which degrades to `503`), a
/// snapshot signing-key misconfiguration must be fatal at startup, because
/// silently dropping the key would downgrade restore integrity (a wrong
/// or empty key would let unverified blobs through once the routes are
/// reachable). The hard-fail mirrors the contract documented on
/// `AppConfig::from_env`.
///
/// `state` arrives behind an `Arc`; we recover the inner value when we are
/// the sole owner (the normal startup case) and otherwise clone-and-replace
/// the `app_config` field so the contract holds even if a caller has
/// already shared the handle.
fn apply_app_config_from_env(state: Arc<AppState>) -> Arc<AppState> {
    let cfg = crate::config::AppConfig::from_env().unwrap_or_else(|e| {
        // Key-free: `ConfigError::Display` never prints the secret bytes.
        panic!(
            "invalid snapshot configuration (TENSOR_WASM_API_SNAPSHOT_*): {e}; \
             refusing to start with a misconfigured snapshot signing key",
        );
    });
    match Arc::try_unwrap(state) {
        Ok(inner) => Arc::new(inner.with_app_config(cfg)),
        Err(shared) => {
            // The handle was already cloned elsewhere; rebuild a fresh
            // `AppState` carrying the same registries/executor with the
            // config applied. Cloning `AppState` is cheap — the maps,
            // metrics, and executor all sit behind their own `Arc`.
            Arc::new((*shared).clone().with_app_config(cfg))
        }
    }
}

/// Build the router with explicit auth / tenant config and the rate limiter
/// disabled. Backwards-compatible shim retained for integration tests that
/// pre-date the per-token rate limiter; new tests should call
/// [`build_router_with_full_config`].
pub fn build_router_with_config(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
) -> Router {
    build_router_with_full_config(
        state,
        auth,
        tenant,
        RateLimiter::new(RateLimitConfig::disabled()),
    )
}

/// Build the router with explicitly supplied auth, tenant, and rate-limit
/// config. Used by integration tests so they can drive the gateway without
/// poisoning the process environment.
///
/// The audit log defaults to the no-op sink in this constructor: existing
/// integration tests that pre-date the audit middleware must not have
/// their stdout polluted by audit records. Tests that exercise the audit
/// path explicitly should call [`build_router_with_audit`].
pub fn build_router_with_full_config(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
    limiter: RateLimiter,
) -> Router {
    build_router_with_audit(
        state,
        auth,
        tenant,
        limiter,
        AuditConfig::disabled(),
        CorsConfig::default(),
    )
}

/// Build the router with full configuration including the audit sink and
/// the CORS allowlist.
///
/// The outer ServiceBuilder is layered top-to-bottom: `host_validate`
/// (with its `TrustedHosts` extension) is the outermost pair — T19 perf
/// — so hostile Host probes are rejected with `400` before any trace
/// span is allocated or any propagator hop runs. The trace layer wraps
/// the rest of the stack, followed by `inject_trace_id_header`, the
/// CORS layer (so cross-origin preflight short-circuits before any
/// expensive downstream work), the body limit (which guards every
/// downstream layer from oversized payloads), the per-request timeout
/// and the global concurrency cap. Auth and tenant resolution are
/// `from_fn` middleware that run after the size cap (so a request that
/// would be rejected with 413 does not consume an auth slot). The
/// per-token rate limiter runs after bearer auth so it can read the
/// AuthContext the auth layer inserts. The audit middleware sits
/// innermost, after every other layer has resolved the actor / tenant /
/// scope, so the synthesised record captures the same identity the
/// handler saw.
pub fn build_router_with_audit(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
    limiter: RateLimiter,
    audit: AuditConfig,
    cors: CorsConfig,
) -> Router {
    build_router_with_trusted_proxies(
        state,
        auth,
        tenant,
        limiter,
        audit,
        cors,
        TrustedProxies::from_env(),
    )
}

/// Build the router with full configuration *and* an explicit XFCC
/// trusted-proxy allowlist.
///
/// Mirrors [`build_router_with_audit`] but lets the caller inject an
/// explicit [`TrustedProxies`] instead of reading
/// `TENSOR_WASM_API_TRUSTED_XFCC_PROXIES` from the ambient process
/// environment. Integration tests that exercise the XFCC-gating path use
/// this constructor so parallel tests do not race on a global env var.
///
/// See the module-level comment block on
/// `crate::audit::extract_client_cert_subject_gated` for the threat
/// model.
pub fn build_router_with_trusted_proxies(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
    limiter: RateLimiter,
    audit: AuditConfig,
    cors: CorsConfig,
    trusted_proxies: TrustedProxies,
) -> Router {
    // Defer to the kernel-publish-tokens variant, reading the
    // production default from the env. Existing callers that pre-date
    // the kernel-publish gate keep their call site untouched.
    build_router_full(
        state,
        auth,
        tenant,
        limiter,
        audit,
        cors,
        trusted_proxies,
        KernelPublishTokens::from_env(),
    )
}

/// Build the router with every override exposed, including the explicit
/// [`KernelPublishTokens`] allowlist for `POST /kernels`.
///
/// This is the lowest-level public builder. Integration tests use it to
/// exercise the kernel-publish authorization gate without poisoning the
/// process environment via `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`.
/// Production code reaches it transitively from
/// [`build_router_with_trusted_proxies`], which reads
/// [`KernelPublishTokens::from_env`] internally.
///
/// All other parameters behave identically to
/// [`build_router_with_trusted_proxies`].
#[allow(clippy::too_many_arguments)]
pub fn build_router_with_kernel_publish_tokens(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
    limiter: RateLimiter,
    audit: AuditConfig,
    cors: CorsConfig,
    trusted_proxies: TrustedProxies,
    kernel_publish_tokens: KernelPublishTokens,
) -> Router {
    build_router_full(
        state,
        auth,
        tenant,
        limiter,
        audit,
        cors,
        trusted_proxies,
        kernel_publish_tokens,
    )
}

/// Process-wide latch for the T16 "tokens set but trusted_hosts unset"
/// startup warning. The warning fires at most once per process even if
/// the router builders are invoked multiple times (tests routinely
/// rebuild the router for each case; production starts the gateway once).
static T16_HOST_WARN_FIRED: AtomicBool = AtomicBool::new(false);

/// SECURITY (H3): environment variable carrying an optional bearer token
/// that gates `GET /metrics`. When set to a non-empty value the metrics
/// scrape endpoint requires `Authorization: Bearer <token>` matching this
/// value (constant-time compared); when unset or empty the endpoint stays
/// unauthenticated (the historical Prometheus-scrape posture) and a
/// one-shot `warn!` is emitted at startup. See `MetricsAuth` and
/// `metrics_auth_gate`.
pub const ENV_METRICS_TOKEN: &str = "TENSOR_WASM_API_METRICS_TOKEN";

/// Process-wide latch for the H3 "metrics endpoint is unauthenticated"
/// startup warning. Like [`T16_HOST_WARN_FIRED`] this fires at most once
/// per process so repeated `build_router*` calls in the test suite do not
/// flood the log.
static H3_METRICS_WARN_FIRED: AtomicBool = AtomicBool::new(false);

/// SECURITY (H3): resolved gate for the unauthenticated `/metrics` scrape
/// endpoint.
///
/// `/metrics` is documented `security: []` in
/// `openapi/tensor-wasm-api.yaml` so Prometheus scrapers and k8s tooling
/// can hit it without a bearer token — the default (env unset) preserves
/// that posture. Operators that expose the listener on a shared interface
/// can opt into a bearer-token gate by setting
/// [`ENV_METRICS_TOKEN`]; when set, `metrics_auth_gate` requires
/// `Authorization: Bearer <token>` and returns `401` otherwise.
#[derive(Debug, Clone, Default)]
struct MetricsAuth {
    /// The configured token, or `None` when the endpoint is left
    /// unauthenticated. `Arc<str>` so cloning into every request's
    /// extensions does not copy the secret bytes.
    token: Option<Arc<str>>,
}

impl MetricsAuth {
    /// Load the optional metrics token from [`ENV_METRICS_TOKEN`].
    ///
    /// Unset / empty (after trimming) → unauthenticated (the historical
    /// posture); the one-shot startup warning is emitted here so operators
    /// running an internet-facing listener without an ingress ACL notice
    /// the open scrape target. A non-empty value enables the bearer gate.
    fn from_env() -> Self {
        let raw = std::env::var(ENV_METRICS_TOKEN).unwrap_or_default();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            // One-shot warning: the scrape target carries per-tenant
            // operational data and is unauthenticated. Idempotent across
            // router rebuilds via the CAS latch.
            if H3_METRICS_WARN_FIRED
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                tracing::warn!(
                    target: "tensor_wasm_api::server",
                    "GET /metrics is unauthenticated (TENSOR_WASM_API_METRICS_TOKEN \
                     unset) — it exposes per-tenant gauges and route/error \
                     distributions. This is the expected Prometheus-scrape \
                     posture behind an internal interface or ingress ACL; set \
                     TENSOR_WASM_API_METRICS_TOKEN=<token> to require \
                     `Authorization: Bearer <token>` on the scrape path.",
                );
            }
            Self { token: None }
        } else {
            tracing::info!(
                target: "tensor_wasm_api::server",
                "GET /metrics requires a bearer token (TENSOR_WASM_API_METRICS_TOKEN set)",
            );
            Self {
                token: Some(Arc::from(trimmed)),
            }
        }
    }
}

/// Reset the H3 metrics startup-warning latch.
///
/// Test-only, mirroring [`__reset_host_validation_warn_for_test`]: each
/// scenario in an integration test that asserts on the warning needs a
/// clean "not yet warned" state. Production code MUST NOT call this — the
/// warning is fire-once for the life of the process by design.
#[doc(hidden)]
pub fn __reset_metrics_warn_for_test() {
    H3_METRICS_WARN_FIRED.store(false, Ordering::Release);
}

/// SECURITY (H3): bearer-token gate applied ONLY to the `/metrics` route.
///
/// The handler in `routes.rs` is left untouched (it is owned by another
/// agent and is `security: []` by contract); this middleware wraps the
/// route in `server.rs` so the gate is purely additive and opt-in:
///
/// * No [`MetricsAuth`] in the extensions, or `token == None` → pass
///   through unauthenticated (the default Prometheus-scrape posture).
/// * Token configured → require `Authorization: Bearer <token>` and reject
///   with `401 unauthorized` (standard `{ "error": { kind, message } }`
///   envelope) otherwise. The comparison is constant-time
///   ([`subtle::ConstantTimeEq`]) so a timing side-channel cannot be used
///   to recover the token byte-by-byte, mirroring the bearer allowlist
///   compare in `crate::middleware::AuthConfig::scope_for`.
async fn metrics_auth_gate(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let configured = req
        .extensions()
        .get::<MetricsAuth>()
        .and_then(|m| m.token.clone());

    let Some(expected) = configured else {
        // Unauthenticated posture (default): pass through unchanged.
        return next.run(req).await;
    };

    // A token is required. Recover the `Authorization: Bearer <token>`
    // header and constant-time compare against the configured value.
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(parse_bearer_token);

    let authorized = match presented {
        // `ct_eq` requires equal-length inputs to be meaningful; a length
        // mismatch is an immediate non-match. We still run `ct_eq` only
        // when lengths match so the byte loop does not early-exit on the
        // first differing byte.
        Some(tok) => {
            let a = tok.as_bytes();
            let b = expected.as_bytes();
            a.len() == b.len() && a.ct_eq(b).into()
        }
        None => false,
    };

    if !authorized {
        return metrics_unauthorized();
    }
    next.run(req).await
}

/// Render the `401` envelope for a rejected `/metrics` request. Mirrors the
/// `{ "error": { "kind", "message" } }` shape produced by
/// `crate::middleware::envelope` (which is private to that module, so this
/// is a local copy rather than a shared import).
fn metrics_unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "kind": "unauthorized",
                "message": "GET /metrics requires a valid Authorization: Bearer <token> header",
            }
        })),
    )
        .into_response()
}

/// Parse the credentials of a `Bearer` `Authorization` header value.
///
/// Local minimal mirror of `crate::middleware::parse_bearer_credentials`
/// (which is private to that module and lives in a file this change must
/// not touch). Splits on the first space/tab, matches the scheme
/// case-insensitively, trims surrounding BWS, and rejects control bytes so
/// a CR/LF/NUL cannot be smuggled into a downstream log line. Returns
/// `None` for a missing/empty credential so the caller treats it as a
/// non-match.
fn parse_bearer_token(value: &str) -> Option<&str> {
    if value.bytes().any(|b| b != b'\t' && (b < 0x20 || b == 0x7F)) {
        return None;
    }
    let split = value
        .as_bytes()
        .iter()
        .position(|&b| b == b' ' || b == b'\t')?;
    let (scheme, rest) = value.split_at(split);
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim_matches(|c: char| c == ' ' || c == '\t');
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// One-shot startup safety check: when the gateway is in production mode
/// (`TENSOR_WASM_API_TOKENS` set to a non-empty value) but the operator
/// has not configured a `Host` allowlist (`TENSOR_WASM_API_TRUSTED_HOSTS`
/// unset or empty), emit a single `tracing::warn!` that points at the
/// virtual-host-confusion / cache-poisoning risk. The runtime
/// `host_validate` middleware itself stays a no-op when no allowlist is
/// configured — this only adds operator-facing visibility so a
/// production deployment behind a misconfigured ingress that forwards
/// arbitrary `Host` headers is no longer silent.
///
/// Closes the T16 audit finding. The atomic latch makes the warning
/// idempotent across repeated `build_router*` calls in the same process
/// (the integration test suite alone rebuilds the router dozens of
/// times); operators see exactly one log line per process.
fn maybe_warn_host_validation_disabled() {
    let tokens_set = std::env::var(ENV_API_TOKENS).is_ok_and(|v| !v.is_empty());
    if !tokens_set {
        return;
    }
    let trusted_hosts_set =
        std::env::var(ENV_TRUSTED_HOSTS).is_ok_and(|v| !TrustedHosts::from_raw(&v).is_empty());
    if trusted_hosts_set {
        return;
    }
    // CAS so we only emit once per process. `Acquire`/`Release` is
    // overkill for a log latch but matches the pattern used by the
    // `install_w3c_propagator` `Once` elsewhere in this module.
    if T16_HOST_WARN_FIRED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        tracing::warn!(
            target: "tensor_wasm_api::server",
            "TENSOR_WASM_API_TOKENS is set (production-mode) but \
             TENSOR_WASM_API_TRUSTED_HOSTS is unset — Host header \
             validation is disabled. This is OK behind an ingress that \
             strips/validates Host, otherwise enable trusted hosts to \
             avoid virtual-host confusion. Set \
             TENSOR_WASM_API_TRUSTED_HOSTS=host1.example.com,host2.example.com \
             to enable.",
        );
    }
}

/// Reset the T16 startup-warning latch.
///
/// Used by the integration test in
/// `tests/host_validate_startup_warn.rs` so each scenario starts from a
/// clean "not yet warned" state. Production code MUST NOT call this:
/// the warning is fire-once for the life of the process by design, and
/// resetting it lets the misconfiguration warning fire repeatedly on
/// every router rebuild. The function is hidden from rustdoc and prefixed
/// with `__` to discourage external callers; it is `pub` only so the
/// integration test binary (which lives outside the crate root) can
/// reach it without depending on `#[cfg(test)]` visibility, which does
/// not extend to integration tests.
#[doc(hidden)]
pub fn __reset_host_validation_warn_for_test() {
    T16_HOST_WARN_FIRED.store(false, Ordering::Release);
}

#[allow(clippy::too_many_arguments)]
fn build_router_full(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
    limiter: RateLimiter,
    audit: AuditConfig,
    cors: CorsConfig,
    trusted_proxies: TrustedProxies,
    kernel_publish_tokens: KernelPublishTokens,
) -> Router {
    // When `kernel-registry-api` is OFF the kernel router below is not
    // built, so the parameter is unused on that build axis. Drop it
    // explicitly under cfg-off to silence the unused-variables lint
    // without poking holes in attribute placement.
    #[cfg(not(feature = "kernel-registry-api"))]
    let _ = kernel_publish_tokens;
    // T16: one-shot warning when production-mode tokens are configured
    // but the Host allowlist is not. The runtime middleware behaviour is
    // unchanged (pass-through with no allowlist); only the operator-
    // facing visibility is new.
    maybe_warn_host_validation_disabled();
    // Wire the W3C Trace Context propagator globally. Idempotent across
    // calls; safe to invoke on every router rebuild (tests do that
    // routinely). Without this, the tower `trace_layer_with_propagation`
    // would silently see an empty parent context for every inbound
    // `traceparent` header and start a fresh root span — collapsing the
    // distributed-tracing invariant the gateway documents in
    // `docs/OBSERVABILITY.md`.
    install_w3c_propagator();

    // HTTP metrics middleware sits OUTSIDE bearer_auth so 401 responses
    // are counted as well — the SLO doc (`docs/SLO.md` §2.1
    // `availability_http`) defines the SLI as a ratio over *every* HTTP
    // response, including auth rejections, and the burn-rate alerts in
    // §5 rely on that. Placing it inside the auth gate would drop ~all
    // probe traffic from the rate panels in the reference dashboard.
    let http_metrics_cfg = HttpMetricsLayerConfig {
        metrics: Arc::clone(&state.metrics),
        routes: RouteAllowList::new_default(),
    };

    // Layers that apply to EVERY route (protected and probe alike): tracing,
    // trace-id injection, the body cap, the timeout, the concurrency cap,
    // and HTTP metrics counting. Auth / tenant / rate-limit / audit are
    // intentionally NOT in this stack — `/healthz` and `/metrics` are
    // documented in `openapi/tensor-wasm-api.yaml` as `security: []` (no
    // auth) so that k8s liveness/readiness probes and Prometheus scrapers
    // can hit them without holding bearer tokens.
    //
    // T19 perf: `host_validate` (with its `TrustedHosts` extension) is
    // layered OUTSIDE `trace_layer_with_propagation` so a hostile Host
    // probe gets rejected with `400` before any trace span is allocated
    // or any `traceparent` parent context is propagated. Under a probe
    // storm against a production deployment that has the allowlist set,
    // this keeps the trace pipeline (and any downstream OTel collector
    // budget) from being burnt on rejections. Rejections short-circuit
    // before the trace layer (and before `http_metrics_middleware`
    // which sits inside the trace layer), so a rejected request gets
    // neither a span nor a metric increment — intentional tradeoff: a
    // hostile Host probe storm cannot burn trace budget or metric
    // cardinality, at the cost of losing observability on the rejected
    // 4xx itself. The Host allowlist is opt-in via
    // `TENSOR_WASM_API_TRUSTED_HOSTS`; deployments that leave it unset
    // are unaffected by the re-order (the middleware is a pass-through
    // and never short-circuits).
    let common_layers = ServiceBuilder::new()
        // api S-30: reject requests whose Host header isn't in the
        // operator-configured allowlist (TENSOR_WASM_API_TRUSTED_HOSTS).
        // Default (env unset) is permissive — the previous behaviour —
        // because most local-dev deployments don't set the env var.
        // Production behind a layered proxy should set the allowlist.
        //
        // The parsed allowlist travels as an `axum::Extension<TrustedHosts>`
        // so tests can override at build time
        // (`router.layer(axum::Extension(TrustedHosts::from_hosts([...])))`)
        // without poisoning the process environment. Inserted BEFORE the
        // `from_fn(host_validate)` layer so the middleware sees it on the
        // request extensions when it runs.
        //
        // T19 perf: this pair sits OUTERMOST so hostile Host probes are
        // rejected before the trace layer allocates a span for them.
        .layer(axum::Extension(TrustedHosts::from_env()))
        .layer(axum::middleware::from_fn(host_validate))
        .layer(trace_layer_with_propagation())
        // `inject_trace_id_header` runs inside the trace layer so the
        // current span (the one the trace layer just created with its
        // parent already attached) is the one whose `trace_id` we surface
        // back to the caller via the `x-trace-id` response header. This
        // is the operator-facing handle the `docs/runbooks/trace-id.md`
        // runbook points at; without it operators have to read journald
        // to recover the trace id of a failed request.
        .layer(axum::middleware::from_fn(inject_trace_id_header))
        // CORS sits near the outer edge so the layer can short-circuit
        // browser preflight (`OPTIONS`) without it needing to clear bearer
        // auth, rate-limit, or audit. The default config has an empty
        // origin allowlist — cross-origin browser callers get no
        // `Access-Control-Allow-Origin` header and the browser blocks the
        // request — operators widen the surface via
        // `TENSOR_WASM_API_CORS_ALLOWED_ORIGINS`. The headers and methods
        // admitted on a cross-origin request mirror the API contract in
        // `API.md`: `Authorization`, `Content-Type`, `X-TensorWasm-Tenant`,
        // and `Traceparent`; methods `GET`, `POST`, `DELETE`.
        .layer(cors_layer(&cors))
        .layer(body_limit_layer(MAX_REQUEST_BODY_BYTES))
        .layer(timeout_layer(Duration::from_secs(30)))
        // NOTE (api S-26): the global ConcurrencyLimit(64) is removed.
        // Per-route caps below isolate budgets so a probe storm cannot
        // starve /invoke, and a noisy /invoke cannot starve probes.
        // Pass the auth + tenant + limiter + audit config into the request
        // extensions so the protected stack's `from_fn` middleware can pick
        // them up without capturing through a type-erased closure. The
        // probe stack does not consume these but inserting them is cheap
        // and keeps the stacks symmetric — relevant if a future probe
        // grows an auth-aware behaviour (e.g. degraded-mode signalling).
        .layer(axum::Extension(auth))
        .layer(axum::Extension(tenant))
        .layer(axum::Extension(limiter))
        .layer(axum::Extension(audit))
        // XFCC spoofing mitigation: parsed allowlist of reverse-proxy peer
        // addresses whose `X-Forwarded-Client-Cert` headers the audit
        // middleware will trust. Empty / unset
        // (`TENSOR_WASM_API_TRUSTED_XFCC_PROXIES`) = trust nobody, drop
        // every inbound XFCC. See `crate::audit::TrustedProxies` and the
        // threat-model comment on `extract_client_cert_subject_gated`.
        .layer(axum::Extension(trusted_proxies))
        .layer(axum::Extension(http_metrics_cfg))
        // Metrics emission wraps every downstream layer (including
        // bearer_auth) so 401s, 429s, and handler responses all get
        // counted — see the comment block above.
        .layer(axum::middleware::from_fn(http_metrics_middleware));

    // Probe stack: `/healthz` and `/metrics` deliberately bypass bearer
    // auth, tenant scope, the per-token rate limiter, and the audit log.
    // OpenAPI (`openapi/tensor-wasm-api.yaml` `paths./healthz` and
    // `paths./metrics`) declares `security: []` for both, k8s probes do
    // not carry an Authorization header, and Prometheus scrapers typically
    // share a single credential across many endpoints — protecting these
    // here would break the published contract and silently disable
    // upstream health checks.
    // SECURITY (H3): `/metrics` is mounted in its own sub-router so the
    // opt-in bearer gate (`metrics_auth_gate`) can be layered onto it
    // WITHOUT touching the handler in `routes.rs` or gating `/healthz`.
    // The endpoint stays `security: []` in
    // `openapi/tensor-wasm-api.yaml` by default — Prometheus scrapers and
    // k8s tooling expect an unauthenticated scrape target, and the
    // endpoint exposes per-tenant operational data (the HTTP-metric
    // families carry a `route` label, and the registry also surfaces
    // per-tenant gauges). Operators MUST bind the listener to an internal
    // interface or place `/metrics` behind an ingress ACL whenever the
    // listener is internet-facing.
    //
    // When `TENSOR_WASM_API_METRICS_TOKEN` is set, `metrics_auth_gate`
    // requires `Authorization: Bearer <token>` (constant-time compared)
    // and returns `401` otherwise; when unset, the gate is a pass-through
    // and `MetricsAuth::from_env` emits a one-shot startup `warn!` that
    // the scrape target is unauthenticated. The `MetricsAuth` extension
    // is inserted on this sub-router only, so `/healthz` and every other
    // route are oblivious to it. The published spec documents `/metrics`
    // as `security: []` (open by default, matching the unset-env posture)
    // and its `description` notes that setting
    // `TENSOR_WASM_API_METRICS_TOKEN` opts the endpoint into a bearer gate;
    // see `openapi/tensor-wasm-api.yaml` and `crates/tensor-wasm-api/openapi.json`.
    let metrics_router = Router::new()
        .route("/metrics", get(metrics))
        .layer(axum::middleware::from_fn(metrics_auth_gate))
        .layer(axum::Extension(MetricsAuth::from_env()));

    let probe_router = Router::new()
        .route("/healthz", get(healthz))
        .merge(metrics_router)
        // api S-26: probes get their own generous budget. A k8s deployment
        // can have many replicas all scraping at once without affecting
        // invoke capacity.
        .layer(concurrency_limit_layer(PROBE_CONCURRENCY_LIMIT));

    // Protected stack: everything that operates on tenant data. Auth /
    // tenant resolution / rate limit / audit all run on top of
    // `common_layers`, in the same order as before so the audit record
    // still observes the final status and any handler-stamped outcome
    // extension.
    // api S-26: split protected routes into three sub-routers by class so
    // each gets an isolated concurrency budget. Invoke is tightest because
    // calls hold a Wasmtime instance lock; writes are middle; reads are
    // loosest.
    let invoke_router = Router::new()
        .route("/functions/:id/invoke", post(invoke_function))
        .route("/functions/:id/invoke-async", post(invoke_function_async))
        // Streaming invoke (roadmap feature #2). Restored by B7.1 and
        // wired end-to-end by T34: the handler builds an mpsc channel,
        // installs the sender on a `StreamingContext`, threads it
        // through `SpawnConfig::with_streaming` to the executor, and
        // drains the receiver into the SSE / chunked-transfer response
        // body. See `docs/STREAMING.md`.
        .route("/functions/:id/invoke-stream", post(invoke_function_stream))
        .layer(concurrency_limit_layer(INVOKE_CONCURRENCY_LIMIT));
    // M5: `/snapshot/save` and `/snapshot/restore` join the write-class
    // budget. Both are POSTs that do CPU-bound crypto + (de)compression
    // work on the blocking pool, comparable in weight to a deploy. They
    // sit under the same protected stack as the other resource routes
    // (bearer_auth + tenant_scope + rate_limit + audit applied below on
    // `protected_router`), so they inherit:
    //   * bearer auth — an unauthenticated caller gets `401`, not `503`;
    //   * tenant scope — `X-TensorWasm-Tenant` is resolved and the handler
    //     runs `authorize_tenant` + a per-resource owner check (save: the
    //     function's `tenant_id`; restore: the HMAC-authenticated snapshot
    //     metadata's `tenant_id`) so cross-tenant access is `403
    //     tenant_scope_denied`;
    //   * per-token rate limiting and audit logging.
    // When `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` is unset both handlers
    // return `503 snapshot_signing_not_configured` (feature-detect shape).
    let write_router = Router::new()
        .route("/functions", post(create_function))
        .route("/functions/:id", delete(delete_function))
        .route("/snapshot/save", post(snapshot_save))
        .route("/snapshot/restore", post(snapshot_restore))
        .layer(concurrency_limit_layer(WRITE_CONCURRENCY_LIMIT));
    let read_router = Router::new()
        .route("/jobs/:id", get(get_job))
        .layer(concurrency_limit_layer(READ_CONCURRENCY_LIMIT));

    // Desired execution order (request flows outer -> inner):
    //   bearer_auth -> tenant_scope -> rate_limit -> audit -> handler
    // so the auth layer rejects unauthenticated callers FIRST (before a
    // rate-limit permit is spent), tenant_scope then resolves the
    // `TenantId`, the per-token limiter reads the `AuthContext`, and the
    // audit middleware runs INNERMOST so the record it synthesises sees
    // the resolved actor / tenant / scope and the handler's final status.
    //
    // CRITICAL: in axum's `Router::layer`, the LAST `.layer(...)` added is
    // the OUTERMOST (it runs first) — the opposite of `ServiceBuilder`.
    // The calls below are therefore written innermost-first: `audit` is
    // added first (runs last) and `bearer_auth` is added last (runs
    // first). The `rate_limit_runs_after_bearer_auth` integration test
    // pins this ordering.
    let protected_router = invoke_router
        .merge(write_router)
        .merge(read_router)
        .layer(axum::middleware::from_fn(audit_log_middleware))
        .layer(axum::middleware::from_fn(rate_limit))
        .layer(axum::middleware::from_fn(tenant_scope))
        .layer(axum::middleware::from_fn(bearer_auth));

    // OpenAI-compat shim stack (B4.9). The two `/v1/...` routes accept
    // off-the-shelf OpenAI SDK requests; those clients send
    // `Authorization: Bearer <api_key>` and typically NO
    // `X-TensorWasm-Tenant` header. The `tenant_scope` middleware tolerates
    // the absent-header case under the default policy
    // (`TENSOR_WASM_API_REQUIRE_TENANT` unset) by inserting
    // `TenantId(0)` into the request extensions, which is exactly the
    // resolution the v0.4 wiring step will want as a fallback — the
    // handler then runs the `authorize_tenant` gate against the bearer
    // token's `TokenScope` so a scoped token that does not cover tenant 0
    // still receives `403 tenant_scope_denied` (not `501`). T2 security
    // fix: the gate must exist on these routes *before* v0.4 wires real
    // execution; otherwise the locked URL surface inherits a hole. See
    // `crates/tensor-wasm-api/src/openai.rs` and `docs/OPENAI-COMPAT.md`.
    //
    // Operators that set `TENSOR_WASM_API_REQUIRE_TENANT=1` deliberately
    // opt into the stricter posture: the OpenAI routes then require the
    // header too, and clients must use the gateway's native shim that
    // injects `X-TensorWasm-Tenant` (or migrate to scoped tokens once v0.4
    // surfaces tenant resolution from the bearer scope itself).
    //
    // Bearer auth, rate-limit, and audit still apply so an unauthenticated
    // OpenAI client receives `401` (not `501`), and the v0.4 wiring step
    // can rely on the same actor/tenant/audit pipeline as the native
    // routes once tenant inference moves out of the header layer.
    //
    // Concurrency budget: share INVOKE_CONCURRENCY_LIMIT — T41 executes
    // the same `TensorWasmExecutor` spawn path as native `/invoke`, so
    // the budgets track in lockstep. Both OpenAI handlers now take a
    // `State<Arc<AppState>>` extractor (T41) so they can read the
    // `openai_model_map` and dispatch through the shared executor; the
    // explicit `Router::<Arc<AppState>>::new()` annotation lines the
    // sub-router up with `protected_router` and `probe_router` for the
    // outer `.merge(...)` call.
    let openai_router: Router<Arc<AppState>> = Router::new()
        .route("/v1/completions", post(crate::openai::completions_handler))
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat_completions_handler),
        )
        // Innermost-first (see `protected_router`): bearer_auth added last
        // runs first (outermost), audit added first runs last (innermost),
        // concurrency stays closest to the handler.
        .layer(concurrency_limit_layer(INVOKE_CONCURRENCY_LIMIT))
        .layer(axum::middleware::from_fn(audit_log_middleware))
        .layer(axum::middleware::from_fn(rate_limit))
        .layer(axum::middleware::from_fn(tenant_scope))
        .layer(axum::middleware::from_fn(bearer_auth));

    // Kernel registry routes (B6.4 — roadmap feature #3 server-side).
    // The endpoints are write-class (publish) and read-class (list /
    // resolve); we put them behind the same `WRITE_CONCURRENCY_LIMIT`
    // budget the function-mutating endpoints use.
    //
    // **T1 security fix.** Earlier scaffolding mounted these OUTSIDE
    // `tenant_scope` on the rationale that the kernel registry is
    // operator-scope (one HMAC key per deployment). That rationale was
    // wrong for two reasons: (1) the `publish_kernel` handler took the
    // tenant extension as `Option<...>` and ignored it, so any
    // allowlisted token — including a tenant-1 token, or any caller at
    // all in dev mode — could publish; (2) the documented
    // `kernel-publish` scope check was unimplemented. Both holes are
    // now closed:
    //
    //   * The router sits under `bearer_auth` + `tenant_scope` so the
    //     handler can rely on the tenant being established and on the
    //     caller having cleared the API token allowlist.
    //   * An `axum::Extension<KernelPublishTokens>` carries the parsed
    //     `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS` allowlist into the
    //     handler. `publish_kernel` rejects dev-mode calls outright
    //     (`kernel_publish_disabled_in_dev_mode`) and any non-
    //     publish-scoped token with `kernel_publish_scope_required`.
    //     GET routes admit any authenticated tenant.
    //
    // Inserting the publish-tokens extension at the kernel router level
    // (rather than `common_layers`) keeps the surface tight — every
    // other route is oblivious to it. The list value flows in via the
    // `kernel_publish_tokens` parameter so tests can call
    // [`build_router_with_kernel_publish_tokens`] with an explicit
    // allowlist (no env poisoning); production callers reach
    // [`build_router_with_trusted_proxies`], which fills the parameter
    // from `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS` via
    // [`KernelPublishTokens::from_env`].
    //
    // The routes are gated behind the `kernel-registry-api` feature so
    // the default build keeps the dep graph lean. Operators flip
    // `--features kernel-registry-api` plus set
    // `TENSOR_WASM_API_KERNEL_HMAC_KEY` to enable them; when the env
    // var is unset the handlers themselves return
    // `503 kernel_registry_not_configured` (so adding the routes here
    // is safe even without the secret configured).
    #[cfg(feature = "kernel-registry-api")]
    let kernel_router: Router<Arc<AppState>> = Router::new()
        .route(
            "/kernels",
            post(crate::kernels::publish_kernel).get(crate::kernels::list_kernels),
        )
        .route(
            "/kernels/:name/:version",
            get(crate::kernels::resolve_kernel),
        )
        // Layer ordering mirrors the protected_router stack. In axum's
        // `Router::layer` the LAST `.layer(...)` added is the OUTERMOST
        // (runs first), so the calls below are written innermost-first:
        // bearer_auth is added last and therefore runs FIRST, putting the
        // AuthContext in the request extensions before tenant_scope,
        // rate_limit, and the innermost audit middleware run. The
        // KernelPublishTokens extension and the concurrency limit are
        // added first (inner-most) so they sit just above the handler —
        // the publish-scope check reads KernelPublishTokens via
        // `Extension<...>` in the handler signature, so the layer that
        // installs it must run before the handler but after any layer that
        // might short-circuit (auth / rate-limit). The
        // `rate_limit_runs_after_bearer_auth` integration test pins this
        // ordering.
        .layer(concurrency_limit_layer(WRITE_CONCURRENCY_LIMIT))
        .layer(axum::Extension(kernel_publish_tokens))
        .layer(axum::middleware::from_fn(audit_log_middleware))
        .layer(axum::middleware::from_fn(rate_limit))
        .layer(axum::middleware::from_fn(tenant_scope))
        .layer(axum::middleware::from_fn(bearer_auth));

    let router = protected_router.merge(probe_router).merge(openai_router);
    #[cfg(feature = "kernel-registry-api")]
    let router = router.merge(kernel_router);
    router.layer(common_layers).with_state(state)
}

/// Bind and serve the router on the given address.
///
/// The listener is wrapped with
/// [`axum::Router::into_make_service_with_connect_info`] so the audit
/// middleware can recover the immediate TCP peer's `SocketAddr` from the
/// request extensions. The XFCC trusted-proxy gate
/// (`crate::audit::TrustedProxies`) depends on that peer information to
/// decide whether to honour the `X-Forwarded-Client-Cert` header.
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> anyhow::Result<()> {
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(target: "tensor_wasm_api::server", %addr, "tensor-wasm-api listening");
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // SECURITY (H3): bearer parsing for the `/metrics` gate.
    #[test]
    fn parse_bearer_token_accepts_canonical_scheme() {
        assert_eq!(parse_bearer_token("Bearer abc123"), Some("abc123"));
        assert_eq!(parse_bearer_token("bearer abc123"), Some("abc123"));
        assert_eq!(parse_bearer_token("BEARER\tabc123"), Some("abc123"));
        assert_eq!(parse_bearer_token("Bearer   spaced  "), Some("spaced"));
    }

    #[test]
    fn parse_bearer_token_rejects_non_bearer_and_empty() {
        assert_eq!(parse_bearer_token("Basic abc123"), None);
        assert_eq!(parse_bearer_token("Bearer"), None);
        assert_eq!(parse_bearer_token("Bearer    "), None);
        assert_eq!(parse_bearer_token(""), None);
    }

    #[test]
    fn parse_bearer_token_rejects_control_bytes() {
        // CR/LF/NUL must not survive into a downstream log line.
        assert_eq!(parse_bearer_token("Bearer ab\r\nc"), None);
        assert_eq!(parse_bearer_token("Bearer ab\0c"), None);
    }

    // SECURITY (H3): an unconfigured `MetricsAuth` keeps the endpoint open.
    #[test]
    fn metrics_auth_default_is_unauthenticated() {
        let auth = MetricsAuth::default();
        assert!(auth.token.is_none());
    }

    // SECURITY (H3): the 401 envelope carries the standard error shape.
    #[test]
    fn metrics_unauthorized_envelope_shape() {
        let resp = metrics_unauthorized();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
