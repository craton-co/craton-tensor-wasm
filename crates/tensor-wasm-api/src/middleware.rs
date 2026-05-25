// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tower middleware helpers: timeouts, concurrency limits, body limits,
//! authentication, tenant scoping, and tracing spans.
//!
//! Each helper returns a single `tower` layer (or middleware function) that
//! the server module composes into the axum router. Keeping the helpers thin
//! makes them easy to reuse in integration tests and benchmarks where a custom
//! stack is desired.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tensor_wasm_core::types::TenantId;
use serde_json::json;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::classify::{ServerErrorsAsFailures, SharedClassifier};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::token_scope::{parse_tokens_env, TokenScope};

/// Default per-request timeout used by [`crate::server::build_router`].
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Default process-wide cap on in-flight requests. Replaced by per-tenant
/// buckets in a follow-up release.
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 64;

/// Maximum allowed inbound request body size, in bytes. 64 MiB.
///
/// Documented in `API.md` under "Request limits". Requests larger than this
/// are rejected with `413 Payload Too Large` by axum's
/// [`DefaultBodyLimit`](axum::extract::DefaultBodyLimit) at extract time
/// (i.e., when a handler reads the body via `Bytes`, `Json`, etc.). We use
/// axum's native limit rather than `tower_http::limit::RequestBodyLimitLayer`
/// because the latter rewraps the request body in `Limited<Body>`, which
/// breaks composition with `axum::middleware::from_fn` (bearer auth, tenant
/// scope) downstream.
pub const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Environment variable carrying a comma-separated allowlist of bearer
/// tokens accepted by [`bearer_auth`]. Empty / unset = dev mode pass-through.
pub const ENV_API_TOKENS: &str = "TENSOR_WASM_API_TOKENS";

/// Environment variable that, when set to `1`, makes the `X-TensorWasm-Tenant`
/// header mandatory. Otherwise its absence defaults to tenant `0`.
pub const ENV_REQUIRE_TENANT: &str = "TENSOR_WASM_API_REQUIRE_TENANT";

/// Name of the header used to scope a request to a tenant.
pub const HEADER_TENANT: &str = "X-TensorWasm-Tenant";

/// Build a per-request timeout layer.
///
/// Requests that exceed `d` are aborted with `408 Request Timeout`.
pub fn timeout_layer(d: Duration) -> TimeoutLayer {
    TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, d)
}

/// Build the default HTTP tracing layer.
///
/// Emits a `tracing` span per request capturing method, URI, and response
/// status. The classifier treats `5xx` responses as failures.
pub fn trace_layer() -> TraceLayer<SharedClassifier<ServerErrorsAsFailures>> {
    TraceLayer::new_for_http()
}

/// Build a process-wide concurrency limit layer that allows at most `max`
/// in-flight requests.
pub fn concurrency_limit_layer(max: usize) -> ConcurrencyLimitLayer {
    ConcurrencyLimitLayer::new(max)
}

/// Build the global request-body size cap (64 MiB by default).
///
/// Returns `413 Payload Too Large` for any body that exceeds `max_bytes`
/// when a handler tries to extract the body. See [`MAX_REQUEST_BODY_BYTES`]
/// for the rationale on using axum's native limit rather than tower-http's.
pub fn body_limit_layer(max_bytes: usize) -> axum::extract::DefaultBodyLimit {
    axum::extract::DefaultBodyLimit::max(max_bytes)
}

/// Snapshot of authentication configuration loaded from the process
/// environment at server start. Cloned cheaply into each request.
///
/// `scopes` is empty in dev mode (no `TENSOR_WASM_API_TOKENS` set or env
/// empty), in which case [`bearer_auth`] passes every request through
/// unchecked. Otherwise each allowlisted bearer token maps to the
/// [`TokenScope`] that came out of [`parse_tokens_env`] at startup.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Allowlisted bearer tokens → tenant scope. Empty = dev mode
    /// (pass-through with startup warning).
    pub scopes: Arc<HashMap<String, TokenScope>>,
    /// Count of entries that used the legacy bare-token shape. The server
    /// emits a single deprecation warning at startup if this is nonzero.
    pub deprecated_count: usize,
}

impl AuthConfig {
    /// Load the allowlist from `$TENSOR_WASM_API_TOKENS`. Unset or empty
    /// means "no auth" (dev mode). Logs a one-shot warning in dev mode and
    /// a one-shot deprecation warning whenever any legacy bare entries were
    /// observed.
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_API_TOKENS).unwrap_or_default();
        let parsed = parse_tokens_env(&raw);
        if parsed.token_scopes.is_empty() {
            tracing::warn!(
                target: "tensor_wasm_api::middleware",
                env = ENV_API_TOKENS,
                "TENSOR_WASM_API_TOKENS empty; API accepts all requests (dev mode)",
            );
        }
        if parsed.deprecated_count > 0 {
            tracing::warn!(
                target: "tensor_wasm_api::middleware",
                env = ENV_API_TOKENS,
                count = parsed.deprecated_count,
                "bare bearer tokens in {} are deprecated; switch to \
                 `token:tenant=...` (or `token:tenant=*` for the current \
                 wildcard behaviour) — bare entries are scheduled for \
                 removal in v1.0",
                ENV_API_TOKENS,
            );
        }
        Self {
            scopes: Arc::new(parsed.token_scopes),
            deprecated_count: parsed.deprecated_count,
        }
    }

    /// Construct directly from an explicit allowlist. Each token gets the
    /// wildcard scope — preserves backwards-compatible behaviour for tests
    /// that pre-date scoped tokens. For tests that need a non-wildcard
    /// scope, build the map directly or use [`AuthConfig::from_scopes`].
    pub fn from_tokens<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut scopes: HashMap<String, TokenScope> = HashMap::new();
        for s in iter {
            scopes.insert(s.into(), TokenScope::all());
        }
        Self {
            scopes: Arc::new(scopes),
            deprecated_count: 0,
        }
    }

    /// Construct from an explicit token → scope map. Used by integration
    /// tests that drive scoped-token paths directly.
    pub fn from_scopes<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = (S, TokenScope)>,
        S: Into<String>,
    {
        let mut scopes: HashMap<String, TokenScope> = HashMap::new();
        for (k, v) in iter {
            scopes.insert(k.into(), v);
        }
        Self {
            scopes: Arc::new(scopes),
            deprecated_count: 0,
        }
    }

    /// `true` if the supplied bearer token is allowlisted.
    pub fn accepts(&self, token: &str) -> bool {
        self.scopes.contains_key(token)
    }

    /// Resolve `token` to its [`TokenScope`] if allowlisted.
    pub fn scope_for(&self, token: &str) -> Option<&TokenScope> {
        self.scopes.get(token)
    }

    /// `true` if no allowlist was configured (dev mode).
    pub fn is_dev_mode(&self) -> bool {
        self.scopes.is_empty()
    }
}

/// Snapshot of tenant-header policy loaded from the process environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct TenantConfig {
    /// `true` when `TENSOR_WASM_API_REQUIRE_TENANT=1` was set at startup.
    pub require_header: bool,
}

impl TenantConfig {
    /// Load policy from `$TENSOR_WASM_API_REQUIRE_TENANT` (`"1"` = required).
    pub fn from_env() -> Self {
        let require_header = std::env::var(ENV_REQUIRE_TENANT)
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        Self { require_header }
    }
}

/// Render the standard `{ "error": { "kind": ..., "message": ... } }`
/// envelope at `status`. Shared helper for middleware that cannot import
/// `crate::routes::ApiError` without a cycle.
fn envelope(status: StatusCode, kind: &str, message: &str) -> Response {
    let body = Json(json!({
        "error": { "kind": kind, "message": message }
    }));
    (status, body).into_response()
}

/// Bearer-token authentication middleware.
///
/// If the allowlist is empty the request passes through (dev mode — already
/// warned at startup) and a synthetic [`AuthContext::dev`] is inserted into
/// the request extensions. Otherwise the `Authorization: Bearer <token>`
/// header must match one of the allowlisted tokens; missing or mismatched
/// headers produce `401 Unauthorized` with the standard error envelope and
/// `kind: "unauthorized"`. On success an [`AuthContext`] keyed by the
/// stable [`crate::rate_limit::TokenId`] derived from the bearer token is
/// inserted into the request extensions for downstream middleware (rate
/// limiting, audit) to consume.
pub async fn bearer_auth(mut req: Request, next: Next) -> Response {
    let cfg = req
        .extensions()
        .get::<AuthConfig>()
        .cloned()
        .unwrap_or_default();

    if cfg.is_dev_mode() {
        req.extensions_mut().insert(crate::rate_limit::AuthContext::dev());
        return next.run(req).await;
    }

    let header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let token = match header.and_then(|h| h.strip_prefix("Bearer ").map(str::trim)) {
        Some(t) if !t.is_empty() => t.to_owned(),
        _ => {
            return envelope(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or malformed Authorization: Bearer <token> header",
            );
        }
    };

    let scope = match cfg.scope_for(&token) {
        Some(s) => s.clone(),
        None => {
            return envelope(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "bearer token is not allowlisted",
            );
        }
    };

    req.extensions_mut()
        .insert(crate::rate_limit::AuthContext::with_scope(&token, scope));
    next.run(req).await
}

/// Parse the `X-TensorWasm-Tenant` header into a `TenantId`, applying the
/// configured policy.
///
/// * Header absent + `TENSOR_WASM_API_REQUIRE_TENANT=1` => `Err(<400 response>)`.
/// * Header absent otherwise => `Ok(TenantId(0))`.
/// * Header present but not a `u64` => `Err(<400 response>)`.
pub fn extract_tenant(headers: &HeaderMap, cfg: TenantConfig) -> Result<TenantId, Response> {
    let raw = headers.get(HEADER_TENANT).and_then(|h| h.to_str().ok());
    match raw {
        None => {
            if cfg.require_header {
                Err(envelope(
                    StatusCode::BAD_REQUEST,
                    "missing_tenant",
                    "X-TensorWasm-Tenant header is required (TENSOR_WASM_API_REQUIRE_TENANT=1)",
                ))
            } else {
                Ok(TenantId(0))
            }
        }
        Some(s) => match s.trim().parse::<u64>() {
            Ok(v) => Ok(TenantId(v)),
            Err(_) => Err(envelope(
                StatusCode::BAD_REQUEST,
                "missing_tenant",
                "X-TensorWasm-Tenant must be a u64",
            )),
        },
    }
}

/// Middleware that resolves the tenant from `X-TensorWasm-Tenant` and stores it
/// in the request's [`axum::http::Extensions`] for handlers to pick up via
/// `Extension<TenantId>`. On parse failure / required-but-missing, emits
/// the standard error envelope and short-circuits the chain.
pub async fn tenant_scope(mut req: Request, next: Next) -> Response {
    let cfg = req
        .extensions()
        .get::<TenantConfig>()
        .copied()
        .unwrap_or_default();

    let tenant = match extract_tenant(req.headers(), cfg) {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    req.extensions_mut().insert(tenant);
    next.run(req).await
}

/// Returns a [`TraceLayer`] that, in addition to the per-request span,
/// reads the W3C `traceparent` header from the incoming request and uses
/// it as the parent context for the resulting span.
///
/// When a downstream service (or load test client) sends the W3C standard
/// header, traces stitch correctly across the boundary. When no header is
/// present, the span is parented to the local context as usual.
///
/// The function delegates the actual parsing to the OpenTelemetry global
/// `TextMapPropagator`. [`crate::server::build_router_with_audit`] calls
/// [`crate::trace_propagation::install_w3c_propagator`] before any
/// request can hit this layer, so the parsing path is reliably wired
/// regardless of whether the `otlp` feature is enabled on
/// `tensor-wasm-core`. If for some reason no propagator is installed —
/// e.g. in a test that bypasses the server builder — the extraction
/// returns an empty context and the span is parented locally.
pub fn trace_layer_with_propagation() -> tower_http::trace::TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    impl Fn(&axum::http::Request<axum::body::Body>) -> tracing::Span + Clone,
> {
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<axum::body::Body>| {
        // Surface the raw traceparent value as a span field for log-based
        // correlation, even when no OTel propagator is installed.
        let traceparent = req
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let span = tracing::info_span!(
            "http.request",
            method = %req.method(),
            uri = %req.uri(),
            version = ?req.version(),
            traceparent = %traceparent,
        );

        // Extract the parent `opentelemetry::Context` from the incoming
        // headers via the globally-installed `TextMapPropagator`. The
        // `set_parent` call hooks the freshly-created tracing span into
        // the upstream W3C trace, so subsequent `#[instrument]` spans on
        // tenant lookup, executor spawn, snapshot restore, and dispatch
        // all share the same `trace_id`.
        let parent_cx = crate::trace_propagation::extract_parent_context(req.headers());
        span.set_parent(parent_cx);

        span
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn timeout_layer_constructs() {
        let _ = timeout_layer(Duration::from_millis(1));
        let _ = timeout_layer(DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn trace_layer_constructs() {
        let _ = trace_layer();
    }

    #[test]
    fn concurrency_limit_layer_constructs() {
        let _ = concurrency_limit_layer(1);
        let _ = concurrency_limit_layer(DEFAULT_CONCURRENCY_LIMIT);
    }

    #[test]
    fn body_limit_layer_constructs() {
        let _ = body_limit_layer(MAX_REQUEST_BODY_BYTES);
    }

    #[test]
    fn trace_layer_with_propagation_constructs() {
        let _ = trace_layer_with_propagation();
    }

    #[test]
    fn auth_config_dev_mode_when_empty() {
        let cfg = AuthConfig::from_tokens(Vec::<String>::new());
        assert!(cfg.is_dev_mode());
        // dev-mode `accepts` is irrelevant, but should return `false` —
        // the dev gate runs in `bearer_auth`, not here.
        assert!(!cfg.accepts("anything"));
    }

    #[test]
    fn auth_config_accepts_matching_token() {
        let cfg = AuthConfig::from_tokens(["foo", "bar"]);
        assert!(!cfg.is_dev_mode());
        assert!(cfg.accepts("foo"));
        assert!(cfg.accepts("bar"));
        assert!(!cfg.accepts("baz"));
    }

    #[test]
    fn auth_config_from_tokens_defaults_to_wildcard_scope() {
        let cfg = AuthConfig::from_tokens(["foo"]);
        let scope = cfg.scope_for("foo").expect("scope present");
        assert!(scope.tenants.is_all(), "from_tokens must default to wildcard");
    }

    #[test]
    fn auth_config_from_scopes_round_trips() {
        let cfg = AuthConfig::from_scopes([
            ("foo", crate::token_scope::TokenScope::from_tenants([TenantId(1)])),
            ("bar", crate::token_scope::TokenScope::all()),
        ]);
        assert!(cfg.accepts("foo"));
        assert!(cfg.accepts("bar"));
        let foo = cfg.scope_for("foo").expect("foo");
        assert!(foo.allows(TenantId(1)));
        assert!(!foo.allows(TenantId(2)));
        let bar = cfg.scope_for("bar").expect("bar");
        assert!(bar.allows(TenantId(99)));
    }

    #[test]
    fn extract_tenant_default_zero_when_optional() {
        let headers = HeaderMap::new();
        let cfg = TenantConfig {
            require_header: false,
        };
        let tid = extract_tenant(&headers, cfg).expect("default to TenantId(0)");
        assert_eq!(tid, TenantId(0));
    }

    #[test]
    fn extract_tenant_errors_when_required_and_missing() {
        let headers = HeaderMap::new();
        let cfg = TenantConfig {
            require_header: true,
        };
        let err = extract_tenant(&headers, cfg).expect_err("required header missing");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn extract_tenant_parses_header_value() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_TENANT, HeaderValue::from_static("7"));
        let tid = extract_tenant(
            &headers,
            TenantConfig {
                require_header: false,
            },
        )
        .expect("parses");
        assert_eq!(tid, TenantId(7));
    }

    #[test]
    fn extract_tenant_rejects_garbage_header() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_TENANT, HeaderValue::from_static("not-a-number"));
        let err = extract_tenant(
            &headers,
            TenantConfig {
                require_header: false,
            },
        )
        .expect_err("rejects garbage");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }
}
