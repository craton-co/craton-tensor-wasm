// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tower middleware helpers: timeouts, concurrency limits, body limits,
//! authentication, tenant scoping, and tracing spans.
//!
//! Each helper returns a single `tower` layer (or middleware function) that
//! the server module composes into the axum router. Keeping the helpers thin
//! makes them easy to reuse in integration tests and benchmarks where a custom
//! stack is desired.

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
/// `tokens` is empty in dev mode (no `TENSOR_WASM_API_TOKENS` set or env empty),
/// in which case [`bearer_auth`] passes every request through unchecked.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Allowlisted bearer tokens. Empty = dev mode (pass-through).
    pub tokens: Arc<Vec<String>>,
}

impl AuthConfig {
    /// Load the allowlist from `$TENSOR_WASM_API_TOKENS`. Unset or empty means
    /// "no auth" (dev mode). Logs a one-shot warning in dev mode.
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_API_TOKENS).unwrap_or_default();
        let tokens: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if tokens.is_empty() {
            tracing::warn!(
                target: "tensor_wasm_api::middleware",
                env = ENV_API_TOKENS,
                "TENSOR_WASM_API_TOKENS empty; API accepts all requests (dev mode)",
            );
        }
        Self {
            tokens: Arc::new(tokens),
        }
    }

    /// Construct directly from an explicit allowlist. Used by tests.
    pub fn from_tokens<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            tokens: Arc::new(iter.into_iter().map(Into::into).collect()),
        }
    }

    /// `true` if the supplied bearer token is allowlisted.
    pub fn accepts(&self, token: &str) -> bool {
        self.tokens.iter().any(|t| t == token)
    }

    /// `true` if no allowlist was configured (dev mode).
    pub fn is_dev_mode(&self) -> bool {
        self.tokens.is_empty()
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
/// warned at startup). Otherwise the `Authorization: Bearer <token>` header
/// must match one of the allowlisted tokens; missing or mismatched headers
/// produce `401 Unauthorized` with the standard error envelope and
/// `kind: "unauthorized"`.
pub async fn bearer_auth(req: Request, next: Next) -> Response {
    let cfg = req
        .extensions()
        .get::<AuthConfig>()
        .cloned()
        .unwrap_or_default();

    if cfg.is_dev_mode() {
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

    if !cfg.accepts(&token) {
        return envelope(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "bearer token is not allowlisted",
        );
    }

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

/// Adapter that lets the OpenTelemetry `TextMapPropagator` read from an
/// `axum`/`http` [`HeaderMap`](axum::http::HeaderMap).
///
/// The propagator API is generic over an [`opentelemetry::propagation::Extractor`],
/// which expects `get(&str) -> Option<&str>` and `keys() -> Vec<&str>` semantics.
/// We provide the smallest possible bridge so we do not need the
/// `opentelemetry-http` crate as an additional dependency.
struct HeaderMapExtractor<'a>(&'a axum::http::HeaderMap);

impl<'a> opentelemetry::propagation::Extractor for HeaderMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
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
/// `TextMapPropagator` (installed by `tensor-wasm-core::telemetry` under the
/// `otlp` feature). If no propagator is installed the extraction returns
/// an empty context and the span is parented locally — i.e. behaviour
/// degrades gracefully.
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
        // headers via whichever `TextMapPropagator` was installed globally
        // (typically `TraceContextPropagator` for W3C `traceparent`). Then
        // attach it as the parent of the freshly-created tracing span via
        // the `OpenTelemetrySpanExt::set_parent` bridge.
        let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&HeaderMapExtractor(req.headers()))
        });
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
