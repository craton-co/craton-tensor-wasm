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
use subtle::ConstantTimeEq;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::classify::{ServerErrorsAsFailures, SharedClassifier};
use tower_http::cors::{AllowOrigin, CorsLayer};
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
/// axum's native `DefaultBodyLimit::max` rather than
/// `tower_http::limit::RequestBodyLimitLayer`: the tower-http layer rewraps
/// the request body in `Limited<Body>`, which breaks composition with
/// `axum::middleware::from_fn` (bearer auth, tenant scope) downstream, and
/// `DefaultBodyLimit::max` gives the same 413 contract without the rewrap.
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

/// Environment variable carrying a comma-separated allowlist of origins
/// permitted for cross-origin requests. Empty / unset = reject all
/// cross-origin requests.
pub const ENV_CORS_ALLOWED_ORIGINS: &str = "TENSOR_WASM_API_CORS_ALLOWED_ORIGINS";

/// HTTP headers permitted on cross-origin requests. Covers the standard
/// `Authorization` and `Content-Type`, the TensorWasm tenant header used to
/// scope per-tenant calls, and the W3C `traceparent` header so browser
/// callers can stitch their own trace context into the gateway's spans.
const CORS_ALLOWED_HEADERS: &[&str] = &[
    "authorization",
    "content-type",
    "x-tensorwasm-tenant",
    "traceparent",
];

/// Cross-origin policy snapshot loaded from the process environment.
///
/// `allowed_origins` is the explicit allowlist of cross-origin browser
/// callers that may reach the API. The default is empty — i.e.
/// **cross-origin requests are rejected** until the operator widens the
/// allowlist. This matches the gateway's other security defaults
/// (`TENSOR_WASM_API_TOKENS` dev mode is the only opt-out).
///
/// To widen, list one origin per entry exactly as the browser sends the
/// `Origin` header (scheme + host + optional port), comma-separated:
///
/// ```text
/// TENSOR_WASM_API_CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
/// ```
#[derive(Debug, Clone, Default)]
pub struct CorsConfig {
    /// Origins permitted for cross-origin requests. Empty = reject all.
    pub allowed_origins: Vec<String>,
}

impl CorsConfig {
    /// Load the allowlist from `$TENSOR_WASM_API_CORS_ALLOWED_ORIGINS`.
    /// Unset or empty = reject all cross-origin requests.
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_CORS_ALLOWED_ORIGINS).unwrap_or_default();
        let allowed_origins: Vec<String> = raw
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        Self { allowed_origins }
    }

    /// Construct directly from an explicit list of origins. The empty list
    /// yields the safe default (no cross-origin requests admitted).
    pub fn from_origins<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed_origins: iter.into_iter().map(Into::into).collect(),
        }
    }
}

/// Build the CORS layer for the gateway router.
///
/// * Empty allowlist (`cfg.allowed_origins.is_empty()`) → returns
///   `CorsLayer::new()`, which sets no `Access-Control-Allow-Origin` header
///   and therefore rejects every cross-origin request. This is the safe
///   default for fresh installs — operators opt in by setting
///   `TENSOR_WASM_API_CORS_ALLOWED_ORIGINS`.
/// * Non-empty allowlist → returns a layer that admits exactly those
///   origins (parsed back into `HeaderValue`s; unparseable entries are
///   silently dropped — they were already rejected at startup by the
///   bearer-auth allowlist parser's stricter sibling and would never match
///   a real browser `Origin` header anyway).
///
/// The allowed methods (`GET`, `POST`, `DELETE`) and headers
/// (`Authorization`, `Content-Type`, `X-TensorWasm-Tenant`, `Traceparent`)
/// match the API's wire surface — see `API.md`.
pub fn cors_layer(cfg: &CorsConfig) -> CorsLayer {
    use axum::http::Method;
    // Cover every method the gateway routes use: `GET` (healthz, metrics,
    // job poll), `POST` (deploy, invoke, invoke-async), and `DELETE`
    // (function tear-down).
    let allowed_methods = vec![Method::GET, Method::POST, Method::DELETE];
    let base = CorsLayer::new()
        .allow_methods(allowed_methods)
        .allow_headers(
            CORS_ALLOWED_HEADERS
                .iter()
                .filter_map(|h| h.parse::<axum::http::HeaderName>().ok())
                .collect::<Vec<_>>(),
        );

    if cfg.allowed_origins.is_empty() {
        // No origins configured — `CorsLayer::new()` admits no origins, so
        // no cross-origin browser caller will see an
        // `Access-Control-Allow-Origin` header and the request is rejected
        // by the browser's preflight check.
        base
    } else {
        let parsed: Vec<axum::http::HeaderValue> = cfg
            .allowed_origins
            .iter()
            .filter_map(|origin| origin.parse::<axum::http::HeaderValue>().ok())
            .collect();
        base.allow_origin(AllowOrigin::list(parsed))
    }
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
    ///
    /// Uses a constant-time byte comparison against every allowlisted entry
    /// so the time taken to reject a bad token does not leak which prefix
    /// (if any) matched an allowlist entry. Delegates to [`scope_for`].
    pub fn accepts(&self, token: &str) -> bool {
        self.scope_for(token).is_some()
    }

    /// Resolve `token` to its [`TokenScope`] if allowlisted.
    ///
    /// Iterates the full allowlist and uses [`subtle::ConstantTimeEq`] for
    /// each entry rather than a `HashMap::get`. The hash-table lookup
    /// short-circuits on hash mismatch and then bytes-eq matching entries,
    /// which is timing-leakable for token discovery — an attacker can
    /// measure how long the gateway took to reject a candidate token and
    /// infer how close it got to a real entry. The loop runs over every
    /// allowlist entry on every call, and we deliberately do NOT `break`
    /// after a hit so the wall-clock cost is constant w.r.t. the matched
    /// entry's position. Hashing still happens internally (`scopes` is an
    /// `Arc<HashMap>` for cheap clones) but the lookup path is no longer
    /// hash-keyed; the map is used purely as a `(token, scope)` store here.
    pub fn scope_for(&self, token: &str) -> Option<&TokenScope> {
        let mut found_scope: Option<&TokenScope> = None;
        let token_bytes = token.as_bytes();
        for (allow_token, scope) in self.scopes.iter() {
            // ct_eq requires equal-length inputs; mismatched lengths cannot
            // be equal so we skip them. The length itself is not secret —
            // the operator's `TENSOR_WASM_API_TOKENS` allowlist is fixed at
            // startup and its lengths are observable through other means.
            if allow_token.len() == token_bytes.len()
                && allow_token.as_bytes().ct_eq(token_bytes).into()
            {
                found_scope = Some(scope);
                // Intentionally NOT `break` — keep iterating so the loop
                // time is constant w.r.t. the matched entry's position.
            }
        }
        found_scope
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

/// Parse the credentials portion of an `Authorization` header value when the
/// scheme matches `Bearer` case-insensitively (RFC 6750 §2.1 / RFC 7235 §2.1).
///
/// The header value is split on the first run of whitespace separating the
/// scheme token from the credentials. The scheme is compared via
/// `eq_ignore_ascii_case("bearer")` so `Bearer`, `bearer`, and `BEARER`
/// (the latter two emerge from upstream load balancers that normalise
/// header names/values) all match. Both space and horizontal tab are
/// accepted as separators (RFC 7235 BWS = bad whitespace). The trailing
/// credentials are trimmed of surrounding ASCII whitespace before being
/// returned.
///
/// Returns `None` when the value has no whitespace (i.e. is a single token
/// such as `Bearer`) or the scheme is not bearer. An empty-credential
/// case (e.g. `"Bearer   "`) returns `Some("")` so the caller can still
/// enforce its empty-token rejection rule.
fn parse_bearer_credentials(value: &str) -> Option<&str> {
    // Find the first whitespace byte (space or tab) — anything else
    // separating scheme from credentials would itself be a protocol
    // violation, so we don't bother with general Unicode whitespace.
    let split = value
        .as_bytes()
        .iter()
        .position(|&b| b == b' ' || b == b'\t')?;
    let (scheme, rest) = value.split_at(split);
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    // Trim the leading whitespace run (BWS) plus any trailing whitespace
    // around the credentials. `trim_matches` over the BWS set keeps the
    // behaviour aligned with `str::trim`'s ASCII-whitespace semantics.
    Some(rest.trim_matches(|c: char| c == ' ' || c == '\t'))
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

    let token = match header.and_then(parse_bearer_credentials) {
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
    fn cors_config_default_is_empty_allowlist() {
        let cfg = CorsConfig::default();
        assert!(cfg.allowed_origins.is_empty());
    }

    #[test]
    fn cors_config_from_origins_round_trips() {
        let cfg = CorsConfig::from_origins(["https://app.example.com"]);
        assert_eq!(cfg.allowed_origins, vec!["https://app.example.com"]);
    }

    #[test]
    fn cors_layer_constructs_empty_allowlist() {
        // The empty-allowlist branch is the safe default: it must produce
        // a layer that does not set Access-Control-Allow-Origin. We only
        // exercise construction here; the wire-level rejection check sits
        // in the integration test suite where a full router is in scope.
        let _ = cors_layer(&CorsConfig::default());
    }

    #[test]
    fn cors_layer_constructs_with_origins() {
        let _ = cors_layer(&CorsConfig::from_origins([
            "https://app.example.com",
            "https://admin.example.com",
        ]));
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
    fn parse_bearer_credentials_accepts_canonical_scheme() {
        assert_eq!(parse_bearer_credentials("Bearer abc"), Some("abc"));
    }

    #[test]
    fn parse_bearer_credentials_is_case_insensitive() {
        // Load balancers (e.g. Envoy / nginx lowercase plugins) may have
        // normalised the scheme; RFC 6750 §2.1 says scheme matching is
        // case-insensitive.
        assert_eq!(parse_bearer_credentials("bearer abc"), Some("abc"));
        assert_eq!(parse_bearer_credentials("BEARER abc"), Some("abc"));
        assert_eq!(parse_bearer_credentials("BeArEr abc"), Some("abc"));
    }

    #[test]
    fn parse_bearer_credentials_accepts_tab_separator() {
        assert_eq!(parse_bearer_credentials("Bearer\tabc"), Some("abc"));
    }

    #[test]
    fn parse_bearer_credentials_trims_surrounding_whitespace() {
        assert_eq!(parse_bearer_credentials("Bearer   abc"), Some("abc"));
        assert_eq!(parse_bearer_credentials("Bearer abc  "), Some("abc"));
        assert_eq!(parse_bearer_credentials("Bearer \t abc \t "), Some("abc"));
    }

    #[test]
    fn parse_bearer_credentials_rejects_other_schemes() {
        assert_eq!(parse_bearer_credentials("Basic ZGVhZGJlZWY="), None);
        assert_eq!(parse_bearer_credentials("Token abc"), None);
    }

    #[test]
    fn parse_bearer_credentials_returns_none_for_no_whitespace() {
        // No separator at all => not a parseable Authorization value.
        assert_eq!(parse_bearer_credentials("Bearer"), None);
        assert_eq!(parse_bearer_credentials(""), None);
    }

    #[test]
    fn parse_bearer_credentials_empty_token_is_some_empty() {
        // Caller (`bearer_auth`) is responsible for the empty-token check;
        // we surface the empty string so it can reject with the same shape
        // as a missing header.
        assert_eq!(parse_bearer_credentials("Bearer   "), Some(""));
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
