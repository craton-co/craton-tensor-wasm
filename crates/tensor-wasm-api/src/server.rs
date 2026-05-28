// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Axum router builder and listener.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{delete, get, post};
use axum::Router;
use tower::ServiceBuilder;

use crate::audit::{audit_log_middleware, AuditConfig, TrustedProxies};
use crate::http_metrics::{http_metrics_middleware, HttpMetricsLayerConfig, RouteAllowList};
use crate::middleware::{
    bearer_auth, body_limit_layer, concurrency_limit_layer, cors_layer, host_validate,
    tenant_scope, INVOKE_CONCURRENCY_LIMIT, PROBE_CONCURRENCY_LIMIT, READ_CONCURRENCY_LIMIT,
    WRITE_CONCURRENCY_LIMIT,
    timeout_layer, trace_layer_with_propagation, AuthConfig, CorsConfig, KernelPublishTokens,
    TenantConfig, TrustedHosts, MAX_REQUEST_BODY_BYTES,
};
use crate::rate_limit::{rate_limit, RateLimitConfig, RateLimiter};
use crate::routes::{
    create_function, delete_function, get_job, healthz, invoke_function, invoke_function_async,
    invoke_function_stream, metrics, AppState,
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
/// [`crate::audit::extract_client_cert_subject_gated`] for the threat
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
        // Streaming invoke (roadmap feature #2). Restored by B7.1; the
        // handler is the post-B6.2 scaffold (no body parsing yet, follows
        // api S-31 contract). v0.4 lands the StreamingContext wire.
        .route("/functions/:id/invoke-stream", post(invoke_function_stream))
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

    // OpenAI-compat shim stack (B4.9). The two `/v1/...` routes accept
    // off-the-shelf OpenAI SDK requests; those clients send
    // `Authorization: Bearer <api_key>` but NEVER an `X-TensorWasm-Tenant`
    // header, so the routes must be mounted OUTSIDE the `tenant_scope`
    // middleware — the layer would otherwise reject every OpenAI request
    // with `missing_tenant` 400. Tenant resolution comes from the bearer
    // token's `TokenScope` instead (wired in v0.4); see
    // `crates/tensor-wasm-api/src/openai.rs` and `docs/OPENAI-COMPAT.md`.
    //
    // Bearer auth, rate-limit, and audit still apply so an unauthenticated
    // OpenAI client receives `401` (not `501`), and the v0.4 wiring step
    // can rely on the same actor/tenant/audit pipeline as the native
    // routes once tenant inference moves out of the header layer.
    //
    // Concurrency budget: share INVOKE_CONCURRENCY_LIMIT — the v0.4
    // implementation will execute the same `TensorWasmExecutor` spawn path
    // as native `/invoke`, so the budgets should track in lockstep. While
    // the scaffold returns 501 in ~µs the cap is effectively a no-op.
    // Explicit `Router::<Arc<AppState>>::new()` because neither OpenAI
    // handler takes a `State` extractor (the scaffold has no AppState
    // dependency yet), so the compiler cannot infer the state type from
    // the handler signatures alone. The annotation lines the sub-router
    // up with `protected_router` and `probe_router` for the outer
    // `.merge(...)` call — without it `Router::merge` complains about
    // mismatched state generics.
    let openai_router: Router<Arc<AppState>> = Router::new()
        .route(
            "/v1/completions",
            post(crate::openai::completions_handler),
        )
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat_completions_handler),
        )
        .layer(concurrency_limit_layer(INVOKE_CONCURRENCY_LIMIT))
        .layer(axum::middleware::from_fn(bearer_auth))
        .layer(axum::middleware::from_fn(rate_limit))
        .layer(axum::middleware::from_fn(audit_log_middleware));

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
        // Layer ordering mirrors the protected_router stack (which the
        // `rate_limit_runs_after_bearer_auth` integration test pins as
        // first `.layer(...)` = outermost in axum's `Router::layer`).
        // bearer_auth runs first so the AuthContext is in the request
        // extensions for the rest of the stack; tenant_scope then
        // installs the TenantId. The KernelPublishTokens extension and
        // the concurrency limit are layered last (inner-most) so they
        // sit just above the handler — the publish-scope check reads
        // KernelPublishTokens directly via `Extension<...>` in the
        // handler signature, so the layer that installs it must run
        // BEFORE the handler but AFTER any layer that might short-
        // circuit (auth / rate-limit) — which is exactly where putting
        // it innermost lands.
        .layer(axum::middleware::from_fn(bearer_auth))
        .layer(axum::middleware::from_fn(tenant_scope))
        .layer(axum::middleware::from_fn(rate_limit))
        .layer(axum::middleware::from_fn(audit_log_middleware))
        .layer(axum::Extension(kernel_publish_tokens))
        .layer(concurrency_limit_layer(WRITE_CONCURRENCY_LIMIT));

    let router = protected_router
        .merge(probe_router)
        .merge(openai_router);
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
