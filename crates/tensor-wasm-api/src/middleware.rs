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

/// Default process-wide cap on in-flight requests. Retained for callers that
/// want a single number; production deployments should prefer the per-route
/// caps below.
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 64;

/// Per-route concurrency caps (api S-26). A single global semaphore lets a
/// probe storm starve `/invoke`; per-route caps isolate the budgets.
///
/// Probe routes (`/healthz`, `/metrics`) get a generous budget because they
/// are cheap and a k8s deployment may have many replicas all scraping at
/// once. Invoke is the heaviest path — keep it tight. Reads and writes get
/// asymmetric caps because writes tend to compile Wasm and allocate engine
/// resources.
pub const PROBE_CONCURRENCY_LIMIT: usize = 256;
/// Concurrent `/invoke` ceiling. Tighter than the default because invokes
/// hold a Wasmtime instance lock across `call_async`.
pub const INVOKE_CONCURRENCY_LIMIT: usize = 32;
/// Concurrent read-route ceiling (GETs that are not probes).
pub const READ_CONCURRENCY_LIMIT: usize = 64;
/// Concurrent write-route ceiling (POST/PUT/DELETE excluding invoke).
pub const WRITE_CONCURRENCY_LIMIT: usize = 16;

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

/// Maximum byte length permitted for an inbound `Authorization` header
/// value before [`bearer_auth`] will even attempt to parse it.
///
/// Hyper's default cap on the combined header block sits around 16 KiB,
/// so an attacker can still send a single `Authorization: Bearer <huge>`
/// value of several KiB. The downstream constant-time comparison in
/// [`AuthConfig::scope_for`] is `O(token_len)` per allowlisted token,
/// so unbounded oversized values let a hostile client burn CPU at
/// roughly `O(num_tokens * token_len)` per request. Capping the value
/// length here (1 KiB) keeps that cost flat — any legitimate bearer
/// token in production is well under this limit (JWTs are typically
/// 500–800 bytes, opaque tokens are far smaller).
///
/// A value longer than this returns `401 Unauthorized` with
/// `kind: "invalid_auth"`. We deliberately reject as `401` rather than
/// `400` so the response shape stays uniform with other auth failures
/// (missing header, mismatched token) and an attacker cannot use the
/// status code to probe the size cap.
pub const MAX_AUTH_HEADER_BYTES: usize = 1024;

/// Environment variable that, when set to `1`, makes the `X-TensorWasm-Tenant`
/// header mandatory. Otherwise its absence defaults to tenant `0`.
pub const ENV_REQUIRE_TENANT: &str = "TENSOR_WASM_API_REQUIRE_TENANT";

/// Name of the header used to scope a request to a tenant.
pub const HEADER_TENANT: &str = "X-TensorWasm-Tenant";

/// Environment variable that, when set, restricts the set of `Host` header
/// values the server will accept. Comma-separated list of authority strings
/// (e.g. `api.example.com,api2.example.com:8443`). Unset = accept any
/// `Host`, which is the previous behaviour but is unsafe behind a layered
/// proxy that may pass arbitrary `Host` values through.
///
/// Closes api S-30 (lack of Host validation). The check rejects requests
/// whose `Host` is not in the allowlist with `400 Bad Request`.
pub const ENV_TRUSTED_HOSTS: &str = "TENSOR_WASM_API_TRUSTED_HOSTS";

/// Parsed `Host` allowlist used by [`host_validate`].
///
/// Closes api S-30 (lack of Host validation). The previous implementation
/// cached the parsed env value in a process-wide `OnceLock`, which made
/// tests that wanted to vary the allowlist unable to do so (the first test
/// to touch the cell froze it for every later test in the same process).
///
/// Now the allowlist travels through `axum::Extension<TrustedHosts>`:
/// [`crate::server::build_router_with_audit`] inserts a `from_env()` value
/// at build time; tests can override by inserting a different value into
/// the router extensions. Tests that bypass the server builder still get
/// the env-var fallback (cached per-process in a private `OnceLock` so the
/// per-request cost stays zero), but an explicit extension always wins.
///
/// Precedence: **explicit `axum::Extension<TrustedHosts>` > env-var
/// fallback (`TENSOR_WASM_API_TRUSTED_HOSTS`)**.
#[derive(Debug, Clone, Default)]
pub struct TrustedHosts(Arc<Vec<String>>);

impl TrustedHosts {
    /// Parse the allowlist from [`ENV_TRUSTED_HOSTS`]. Splits on `,`,
    /// trims surrounding whitespace, drops empty entries, and lowercases
    /// each remaining entry for case-insensitive matching. Unset / empty
    /// env var yields an empty list (= [`Self::allow_all`]).
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_TRUSTED_HOSTS).unwrap_or_default();
        Self::from_raw(&raw)
    }

    /// Helper: parse a comma-separated string as if it were the env
    /// variable. Public for the explicit-construction path used by tests.
    pub fn from_raw(raw: &str) -> Self {
        let parsed: Vec<String> = raw
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        Self(Arc::new(parsed))
    }

    /// Explicit "no allowlist" constructor — every Host value is admitted
    /// (the legacy default when the env var is unset).
    pub fn allow_all() -> Self {
        Self(Arc::new(Vec::new()))
    }

    /// Construct from an iterator of allowlist entries. Entries are
    /// lowercased on insertion so case-insensitive matching in
    /// [`Self::contains`] is just a byte comparison.
    pub fn from_hosts<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let parsed: Vec<String> = iter
            .into_iter()
            .map(|s| s.into().trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        Self(Arc::new(parsed))
    }

    /// `true` when no entries are configured — every host is admitted.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `true` when the supplied host (raw value from `Host:` or
    /// `:authority`) matches one of the allowlist entries.
    ///
    /// Matching rules:
    ///
    /// * **Case-insensitive exact match** on the supplied host.
    /// * If the supplied host carries a default-port suffix (`:80` or
    ///   `:443`) and no allowlist entry contains a `:`, the port is
    ///   stripped before comparison. This lets operators list bare
    ///   hostnames (`api.example.com`) and still admit clients that
    ///   include the default port in the `Host` header. If any
    ///   allowlist entry contains a port, we do an exact match on the
    ///   full `host:port` string — the operator chose to be specific.
    pub fn contains(&self, host: &str) -> bool {
        if self.0.is_empty() {
            return true;
        }
        let host_lc = host.trim().to_ascii_lowercase();
        if self.0.iter().any(|allowed| allowed == &host_lc) {
            return true;
        }
        // Default-port strip: only apply when no allowlist entry carries
        // a port (otherwise the operator's port-bound entry must match
        // exactly).
        let allow_has_port = self.0.iter().any(|a| a.contains(':'));
        if allow_has_port {
            return false;
        }
        if let Some(stripped) = strip_default_port(&host_lc) {
            return self.0.iter().any(|allowed| allowed == stripped);
        }
        false
    }
}

/// Strip a trailing `:80` or `:443` from a (already-lowercased) host
/// string. Returns `None` if no default-port suffix is present.
fn strip_default_port(host_lc: &str) -> Option<&str> {
    for suffix in [":443", ":80"] {
        if let Some(stripped) = host_lc.strip_suffix(suffix) {
            return Some(stripped);
        }
    }
    None
}

/// Per-process cached env-var fallback for [`host_validate`] when no
/// `axum::Extension<TrustedHosts>` is present. Tests that drive the
/// middleware through the server builder always get an explicit
/// extension and never touch this; tests that bypass the builder
/// (e.g. wrap `host_validate` with `axum::middleware::from_fn` directly)
/// see the env value parsed once.
fn env_trusted_hosts_fallback() -> TrustedHosts {
    static ONCE: std::sync::OnceLock<TrustedHosts> = std::sync::OnceLock::new();
    ONCE.get_or_init(TrustedHosts::from_env).clone()
}

/// Middleware: reject requests whose `Host` header (or HTTP/2
/// `:authority` pseudo-header) is not in the configured allowlist.
///
/// Source of truth for the allowlist:
///
/// 1. `axum::Extension<TrustedHosts>` if present on the request — the
///    server builder inserts this from
///    [`TrustedHosts::from_env`] at startup, and tests can override.
/// 2. Otherwise, a per-process env-var fallback parsed once from
///    `TENSOR_WASM_API_TRUSTED_HOSTS`.
///
/// Empty allowlist (no entries / env unset) = pass-through.
///
/// Host extraction order:
///
/// 1. `Host:` request header.
/// 2. If absent, `req.uri().authority()` — the URI carries the HTTP/2
///    `:authority` pseudo-header in `hyper`'s normalised request form.
/// 3. If both absent and the allowlist is non-empty, respond `400`.
///
/// Closes api S-30. Should be layered AFTER trace/CORS (so the response
/// is still observable) but BEFORE bearer_auth (so an attacker probing
/// for valid hosts cannot also probe for valid tokens). The probe
/// router inherits this gate too; operators with split-Host probes can
/// simply omit the env var.
pub async fn host_validate(req: Request, next: Next) -> Response {
    let allow = req
        .extensions()
        .get::<TrustedHosts>()
        .cloned()
        .unwrap_or_else(env_trusted_hosts_fallback);
    if allow.is_empty() {
        return next.run(req).await;
    }
    // 1) Try the `Host:` header first (HTTP/1.1 canonical path).
    let host_header = req
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_owned());
    // 2) Fall back to the URI authority (HTTP/2 `:authority` pseudo-header
    //    surfaces here in hyper's normalised request form).
    let authority = req.uri().authority().map(|a| a.as_str().to_owned());
    let host = host_header.or(authority);

    match host {
        Some(h) if allow.contains(&h) => next.run(req).await,
        _ => envelope(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "Host header missing or not in TENSOR_WASM_API_TRUSTED_HOSTS allowlist",
        ),
    }
}

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
///
/// **Control-byte defence.** RFC 7230 §3.2.6 bans control characters
/// (other than horizontal tab) from header field values, and
/// [`axum::http::HeaderValue`] already rejects NUL/CR/LF at construction.
/// The `to_str()` conversion in [`bearer_auth`] further restricts the
/// input to "visible ASCII" plus tab. We nonetheless apply an explicit
/// belt-and-braces check here so any future refactor that fans this
/// helper out behind a more permissive byte source cannot silently
/// re-open a CRLF/NUL injection channel into downstream consumers of
/// the returned token (audit log fields, span attributes, etc.).
fn parse_bearer_credentials(value: &str) -> Option<&str> {
    // Defence in depth: reject the entire value if any control byte
    // (other than the horizontal tab we explicitly use as a separator)
    // is present. Covers NUL (C-string truncation), CR/LF (log-line
    // forgery), and the DEL byte (terminal-escape smuggling).
    if value.bytes().any(|b| b != b'\t' && (b < 0x20 || b == 0x7F)) {
        return None;
    }
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

    // api S-3: refuse requests with more than one `Authorization` header.
    // `HeaderMap::get` returns the first occurrence — a buggy or hostile
    // client sending two headers would silently get one accepted and the
    // other invisible. Some proxies also concatenate duplicates with
    // commas, which would round-trip through `parse_bearer_credentials`
    // unpredictably. Reject outright.
    let auth_count = req
        .headers()
        .get_all(axum::http::header::AUTHORIZATION)
        .iter()
        .count();
    if auth_count > 1 {
        return envelope(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "duplicate Authorization headers are not allowed",
        );
    }
    // api header-hardening: cap the inbound Authorization value length
    // BEFORE the constant-time compare loop runs. Hyper's default cap on
    // the header block (~16 KiB) is large enough that a single oversized
    // `Authorization: Bearer <huge>` would still burn CPU through every
    // `ct_eq` iteration. See [`MAX_AUTH_HEADER_BYTES`] for the rationale.
    // We inspect the raw bytes so a non-UTF-8 value (rejected by
    // `to_str()` below) is bounded too.
    let raw_auth = req.headers().get(axum::http::header::AUTHORIZATION);
    if let Some(value) = raw_auth {
        if value.as_bytes().len() > MAX_AUTH_HEADER_BYTES {
            return envelope(
                StatusCode::UNAUTHORIZED,
                "invalid_auth",
                "Authorization header exceeds the maximum permitted length",
            );
        }
    }

    let header = raw_auth.and_then(|h| h.to_str().ok());

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
/// Outcomes:
///
/// * **More than one `X-TensorWasm-Tenant` header present** => `Err(400
///   duplicate_tenant_header)`. `HeaderMap::get` returns only the first
///   match, so an attacker behind a permissive proxy could send
///   `X-TensorWasm-Tenant: 1, X-TensorWasm-Tenant: 999` and confuse
///   downstream observers about which tenant the request really
///   belongs to. We reject outright before any single value is read.
/// * **Header absent + `TENSOR_WASM_API_REQUIRE_TENANT=1`** => `Err(400
///   missing_tenant)`.
/// * **Header absent otherwise** => `Ok(TenantId(0))`.
/// * **Header present but not a `u64`** => `Err(400 invalid_tenant)`.
///   The distinct `invalid_tenant` kind separates a malformed value
///   from the legitimately-absent case so dashboards can alert on each
///   class independently — a spike in `invalid_tenant` typically
///   indicates a client bug or a probing attacker, whereas a spike in
///   `missing_tenant` indicates a misconfigured client.
pub fn extract_tenant(headers: &HeaderMap, cfg: TenantConfig) -> Result<TenantId, Response> {
    // Fix 1: refuse requests carrying more than one X-TensorWasm-Tenant
    // header. The single-`get` path would otherwise pick the first
    // occurrence silently while a downstream observer sees the second.
    let header_count = headers.get_all(HEADER_TENANT).iter().count();
    if header_count > 1 {
        return Err(envelope(
            StatusCode::BAD_REQUEST,
            "duplicate_tenant_header",
            "multiple X-TensorWasm-Tenant headers are not allowed",
        ));
    }
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
            // Fix 2: the header is PRESENT but unparseable — emit
            // `invalid_tenant`, distinct from the absent-and-required
            // `missing_tenant` case above.
            Err(_) => Err(envelope(
                StatusCode::BAD_REQUEST,
                "invalid_tenant",
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
        // correlation, even when no OTel propagator is installed. The
        // header is sanitised first (see `sanitise_traceparent`) so a
        // hostile client cannot smuggle CR/LF, NUL, or megabytes of
        // arbitrary text into every log line that touches this request.
        let raw_tp = req
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let sanitised_tp = sanitise_traceparent(raw_tp);

        // Record only the URI **path**, never the query string. Query
        // parameters frequently carry tokens/secrets (e.g. an attacker
        // probing `GET /healthz?secret=exfil`) and we MUST NOT plant
        // them in span attributes that flow to log sinks. Handlers that
        // legitimately need the query string read it from
        // `req.uri().query()` themselves under their own scrubbing.
        //
        // Both `path` and `method` flow through bounded sanitisers
        // (`sanitize_path` / `normalize_method`) before reaching the
        // span so that path-traversal probes (`/functions/../etc/passwd`)
        // and CRLF-injection payloads (`/foo%0d%0aevil-header:%20yes`)
        // can neither forge log lines nor smuggle terminal-escape
        // sequences into operator dashboards. `traceparent` has its
        // own dedicated sanitiser above.
        let sanitised_path = sanitize_path(req.uri().path());
        let normalised_method = normalize_method(req.method().as_str());
        let span = tracing::info_span!(
            "http.request",
            method = %normalised_method,
            path = %sanitised_path,
            version = ?req.version(),
            traceparent = %sanitised_tp,
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

/// Maximum byte length a `traceparent` header value is allowed to
/// occupy in a span attribute. The W3C Trace Context spec caps a
/// well-formed value at 55 bytes; 64 gives a tiny margin for future
/// versioned suffixes while still bounding the per-span log footprint
/// to a constant. Anything longer is truncated.
const TRACEPARENT_MAX_BYTES: usize = 64;

/// Sentinel emitted in span attributes when a `traceparent` header
/// contains characters that would corrupt log output (CR, LF, NUL).
/// Choosing a fixed token rather than the original (filtered) value
/// keeps grep/audit signatures stable across hostile inputs.
const TRACEPARENT_INVALID_SENTINEL: &str = "<invalid>";

/// Render an inbound `traceparent` header value as a bounded, log-safe
/// string for use as a tracing span attribute.
///
/// The header is attacker-controlled, so we apply three defences:
///
/// 1. **Reject control characters.** Any CR, LF, or NUL byte indicates
///    an attempt to inject a fake log line (CRLF) or terminate a C
///    string in a downstream consumer (NUL). We collapse the entire
///    value to the [`TRACEPARENT_INVALID_SENTINEL`] in that case rather
///    than try to filter — partial sanitisation is exactly the kind of
///    surface that breeds bypasses.
/// 2. **Clamp length to 64 bytes.** The W3C grammar caps a valid value
///    at 55 bytes; anything longer is either malformed or hostile
///    padding. Truncation happens at a UTF-8 char boundary so we never
///    return invalid UTF-8 to the tracing layer.
/// 3. **Strip non-printable bytes** (anything outside `0x20..=0x7E`).
///    Printable ASCII is the entire grammar the spec allows, and
///    keeping span attributes plain ASCII prevents terminal-escape
///    smuggling through log viewers.
///
/// When the input is already a clean ASCII-printable string short
/// enough to fit, we return [`Cow::Borrowed`] to avoid the allocation.
fn sanitise_traceparent(raw: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    // Defence 1: reject the whole value when control chars appear.
    // CRLF is the classic log-injection vector; NUL the classic C
    // boundary trick. Bail before doing any other processing so the
    // sentinel is stable.
    if raw.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0) {
        return Cow::Owned(TRACEPARENT_INVALID_SENTINEL.to_string());
    }

    // Fast path: already short, already printable ASCII -> borrow.
    let is_clean = raw.len() <= TRACEPARENT_MAX_BYTES
        && raw.bytes().all(|b| (0x20..=0x7E).contains(&b));
    if is_clean {
        return Cow::Borrowed(raw);
    }

    // Slow path: build a filtered, clamped owned copy. Walk by char so
    // truncation never lands mid-codepoint, and skip anything outside
    // printable ASCII.
    let mut out = String::with_capacity(raw.len().min(TRACEPARENT_MAX_BYTES));
    for ch in raw.chars() {
        let b = ch as u32;
        if !(0x20..=0x7E).contains(&b) {
            continue;
        }
        let ch_len = ch.len_utf8();
        if out.len() + ch_len > TRACEPARENT_MAX_BYTES {
            break;
        }
        out.push(ch);
    }
    Cow::Owned(out)
}

/// Maximum byte length that a request path is allowed to occupy in a
/// tracing span attribute (see [`sanitize_path`]). Anything longer is
/// truncated with a `…` suffix to keep operator log lines bounded and
/// to deny a hostile client a cheap way to balloon every log record
/// that touches their request.
///
/// 256 bytes comfortably accommodates every route template the API
/// exposes today (the longest, `/functions/{id}/invoke-async`, is far
/// under that even after path-parameter substitution) while still
/// bounding the per-span attribute footprint to a constant. Bump only
/// after auditing the dashboard layouts in `docs/OBSERVABILITY.md` —
/// a wider value widens the log-line budget linearly.
pub const MAX_PATH_LEN: usize = 256;

/// Sentinel returned by [`normalize_method`] for any HTTP method that
/// does not match the `[A-Z]{1,16}` shape. Keeping a fixed token rather
/// than the original (filtered) value preserves grep/audit signatures
/// across hostile inputs — every malformed method bucket maps to the
/// same string.
const METHOD_OTHER_SENTINEL: &str = "OTHER";

/// Upper bound on the byte length of an accepted HTTP method name.
/// Standard methods are at most 7 bytes (`OPTIONS`); 16 leaves room
/// for legitimate WebDAV-style extensions (`MKCALENDAR`, `PROPPATCH`)
/// while rejecting padding-style abuse.
const METHOD_MAX_LEN: usize = 16;

/// Render a request path as a bounded, log-safe string for use as a
/// tracing span attribute.
///
/// The path is attacker-controlled (it flows out of the request URI
/// after axum's routing layer), so we apply three defences:
///
/// 1. **Truncate to [`MAX_PATH_LEN`] bytes** with a `…` ellipsis
///    suffix when the input is longer. Truncation lands on a UTF-8
///    char boundary so we never emit invalid UTF-8 to the tracing
///    layer. The ellipsis is one Unicode code point (`U+2026`, three
///    UTF-8 bytes) so the returned `Cow::Owned` is at most
///    `MAX_PATH_LEN + 3` bytes — the test in
///    `tests/trace_sanitization_test.rs` asserts `MAX_PATH_LEN + 4`
///    to give a one-byte margin against future ellipsis tweaks.
/// 2. **Replace non-printable / non-ASCII bytes with `?`.** Anything
///    outside `0x20..=0x7E` (printable ASCII) is collapsed to a
///    single `?` so terminal-escape sequences cannot smuggle out of
///    a log viewer and multi-byte UTF-8 sequences (e.g. `é` →
///    `0xC3 0xA9`) cannot be reconstructed downstream.
/// 3. **Strip CR / LF / NUL specifically.** The printable-byte filter
///    above already catches these, but we apply the explicit strip as
///    defence in depth: a future relaxation of the printable filter
///    must NOT re-open the CRLF-injection channel (forging fake JSON
///    log lines) or the NUL channel (terminating C-string consumers).
///
/// When the input is already a clean ASCII-printable string short
/// enough to fit, we return [`Cow::Borrowed`] to avoid the allocation.
pub fn sanitize_path(raw: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    // Fast path: already short, already printable ASCII, no
    // CR/LF/NUL -> borrow. Walking once via `bytes().all` lets the
    // compiler fuse the checks; we only allocate when something
    // actually needs rewriting.
    let is_clean = raw.len() <= MAX_PATH_LEN
        && raw.bytes().all(|b| {
            (0x20..=0x7E).contains(&b) && b != b'\r' && b != b'\n' && b != 0
        });
    if is_clean {
        return Cow::Borrowed(raw);
    }

    // Slow path: build a filtered, clamped owned copy. Walk by char
    // so truncation never lands mid-codepoint; replace anything
    // outside printable ASCII with a literal `?` rather than dropping
    // it, so a path like `/café` round-trips to `/caf??` (two bytes
    // for `é` → two `?` substitutions) and the resulting attribute
    // length is observable to operators rather than silently
    // shrinking.
    //
    // We reserve `MAX_PATH_LEN` up front; the ellipsis (if any) is
    // appended afterwards so the truncation check operates on the
    // pre-ellipsis byte budget.
    let mut out = String::with_capacity(raw.len().min(MAX_PATH_LEN));
    let mut truncated = false;
    for ch in raw.chars() {
        // Defence 3: strip CR/LF/NUL explicitly even though the
        // printable filter below would catch them. A future relaxation
        // must not re-open the injection channel.
        if ch == '\r' || ch == '\n' || ch == '\0' {
            out.push('?');
            if out.len() >= MAX_PATH_LEN {
                truncated = true;
                break;
            }
            continue;
        }
        let b = ch as u32;
        let replacement = if (0x20..=0x7E).contains(&b) {
            // Printable ASCII: keep the character as-is (it occupies
            // one byte in UTF-8 by definition).
            ch
        } else {
            // Non-printable or non-ASCII: collapse to `?`. This
            // includes every byte of a multi-byte UTF-8 sequence
            // because we iterate by `char`, not by byte, so each
            // non-ASCII code point contributes exactly one `?` to
            // the output regardless of how many bytes it occupies
            // in the input.
            '?'
        };
        let ch_len = replacement.len_utf8();
        if out.len() + ch_len > MAX_PATH_LEN {
            truncated = true;
            break;
        }
        out.push(replacement);
    }
    if truncated {
        // Append a single-code-point ellipsis (`…`, U+2026, 3 bytes
        // in UTF-8) so the truncated value is visually distinguishable
        // from a non-truncated one. Total length stays bounded at
        // `MAX_PATH_LEN + 3` bytes — see the doc comment above.
        out.push('…');
    }
    Cow::Owned(out)
}

/// Normalise an HTTP method name for use as a tracing span attribute.
///
/// Returns the input borrowed when it matches `[A-Z]{1,16}` exactly
/// (every standard method — `GET`, `POST`, `PUT`, `PATCH`, `DELETE`,
/// `HEAD`, `OPTIONS`, `TRACE`, `CONNECT` — passes through verbatim).
/// Anything else — lowercase (`get`), mixed case (`Get`), control
/// characters, non-ASCII, oversized padding — collapses to the
/// [`METHOD_OTHER_SENTINEL`] (`"OTHER"`).
///
/// HTTP method names are far less risky than paths (hyper rejects
/// most malformed values before they reach this layer) but we still
/// normalise so that:
///
/// * dashboard label cardinality stays bounded (no per-request method
///   variant exploding the metrics index),
/// * a custom client cannot smuggle non-standard bytes into the
///   `method` span field that the path sanitiser would have rejected.
pub fn normalize_method(raw: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    let bytes = raw.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= METHOD_MAX_LEN
        && bytes.iter().all(|b| b.is_ascii_uppercase());
    if valid {
        Cow::Borrowed(raw)
    } else {
        Cow::Owned(METHOD_OTHER_SENTINEL.to_string())
    }
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
    fn sanitise_traceparent_passes_through_well_formed_value() {
        // A literal example from the W3C Trace Context spec. Already
        // printable ASCII, already under the 64-byte cap, so the
        // helper must return the borrowed pointer (no allocation,
        // exact byte equality).
        let sample = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let out = sanitise_traceparent(sample);
        assert_eq!(out.as_ref(), sample);
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "well-formed value must be borrowed, not allocated",
        );
    }

    #[test]
    fn sanitise_traceparent_rejects_crlf_injection() {
        // CRLF in a header value is the classic log-injection vector;
        // a hostile client could plant a forged "log line" into our
        // structured output. The helper must collapse the whole value
        // to a stable sentinel, NOT a partially-filtered string.
        let attack = "\r\nSET-COOKIE: x=y\r\n";
        let out = sanitise_traceparent(attack);
        assert_eq!(out.as_ref(), "<invalid>");
    }

    #[test]
    fn sanitise_traceparent_rejects_embedded_nul() {
        // NUL bytes truncate C strings in downstream consumers; treat
        // them as hostile and emit the sentinel.
        let attack = "00-aaaa\0bbbb";
        let out = sanitise_traceparent(attack);
        assert_eq!(out.as_ref(), "<invalid>");
    }

    #[test]
    fn sanitise_traceparent_truncates_oversized_input_to_64_bytes() {
        // A 100-byte all-'a' string is printable ASCII but exceeds
        // the 64-byte cap. The helper must clamp to exactly 64
        // bytes — the W3C maximum is 55, so 64 already accommodates
        // every legitimate value with room to spare.
        let input = "a".repeat(100);
        let out = sanitise_traceparent(&input);
        assert_eq!(out.len(), 64);
        assert!(out.chars().all(|c| c == 'a'));
    }

    #[test]
    fn sanitise_traceparent_filters_non_printable_bytes() {
        // ESC (0x1B) is outside printable ASCII and could be used to
        // smuggle terminal escape sequences through log viewers.
        // The helper must strip the byte but keep the surrounding
        // printable context.
        let attack = "00-aaaa\x1Bbbbb-cc";
        let out = sanitise_traceparent(attack);
        assert!(!out.contains('\x1B'));
        assert!(out.contains("aaaa"));
        assert!(out.contains("bbbb"));
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
