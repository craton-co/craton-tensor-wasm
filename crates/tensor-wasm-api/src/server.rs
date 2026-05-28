// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Axum router builder and listener.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{delete, get, post};
use axum::Router;
use tower::ServiceBuilder;

use crate::audit::{audit_log_middleware, AuditConfig};
use crate::http_metrics::{http_metrics_middleware, HttpMetricsLayerConfig, RouteAllowList};
use crate::middleware::{
    bearer_auth, body_limit_layer, concurrency_limit_layer, cors_layer, host_validate,
    tenant_scope, INVOKE_CONCURRENCY_LIMIT, PROBE_CONCURRENCY_LIMIT, READ_CONCURRENCY_LIMIT,
    WRITE_CONCURRENCY_LIMIT,
    timeout_layer, trace_layer_with_propagation, AuthConfig, CorsConfig, TenantConfig,
    TrustedHosts, MAX_REQUEST_BODY_BYTES,
};
use crate::rate_limit::{rate_limit, RateLimitConfig, RateLimiter};
use crate::routes::{
    create_function, delete_function, get_job, healthz, invoke_function, invoke_function_async,
    metrics, AppState,
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
    build_router_with_audit(state, auth, tenant, limiter, audit, cors)
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
/// The outer ServiceBuilder is layered top-to-bottom: tracing is the
/// outermost layer (so it covers timeouts and rejections), followed by
/// the CORS layer (so cross-origin preflight short-circuits before any
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
    let common_layers = ServiceBuilder::new()
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
        .layer(axum::Extension(TrustedHosts::from_env()))
        .layer(axum::middleware::from_fn(host_validate))
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
    let probe_router = Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
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
        .layer(concurrency_limit_layer(INVOKE_CONCURRENCY_LIMIT));
    let write_router = Router::new()
        .route("/functions", post(create_function))
        .route("/functions/:id", delete(delete_function))
        .layer(concurrency_limit_layer(WRITE_CONCURRENCY_LIMIT));
    let read_router = Router::new()
        .route("/jobs/:id", get(get_job))
        .layer(concurrency_limit_layer(READ_CONCURRENCY_LIMIT));

    let protected_router = invoke_router
        .merge(write_router)
        .merge(read_router)
        .layer(axum::middleware::from_fn(bearer_auth))
        .layer(axum::middleware::from_fn(tenant_scope))
        .layer(axum::middleware::from_fn(rate_limit))
        // Audit emission is the innermost middleware: it sees the final
        // status code and any AuditOutcomeExt the handler stamped.
        .layer(axum::middleware::from_fn(audit_log_middleware));

    protected_router
        .merge(probe_router)
        .layer(common_layers)
        .with_state(state)
}

/// Bind and serve the router on the given address.
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> anyhow::Result<()> {
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(target: "tensor_wasm_api::server", %addr, "tensor-wasm-api listening");
    axum::serve(listener, router).await?;
    Ok(())
}
