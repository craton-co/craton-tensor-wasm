// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! HTTP serverless API gateway built on axum 0.7.
//!
//! Exposes deploy/invoke/metrics/healthz endpoints with structured JSON
//! errors. Application state is a `DashMap<Uuid, FunctionRecord>` shared
//! via `Arc<AppState>`. The synchronous and async invoke paths both drive
//! `tensor_wasm_exec::executor::TensorWasmExecutor`.
//!
//! ## Security surface
//!
//! * **Body limit.** Every request is capped at 64 MiB by
//!   [`axum::extract::DefaultBodyLimit::max`]; oversized bodies are
//!   rejected with `413 Payload Too Large` before any handler runs.
//! * **CORS.** The gateway installs an explicit-allowlist
//!   [`tower_http::cors::CorsLayer`]; the allowlist is empty by default
//!   (every cross-origin request rejected). Widen by setting
//!   `TENSOR_WASM_API_CORS_ALLOWED_ORIGINS` to a comma-separated list of
//!   origins. See [`middleware::CorsConfig`].
//! * **Bearer auth.** Reads `TENSOR_WASM_API_TOKENS` (comma-separated allowlist)
//!   at startup. Empty/unset means dev mode (pass-through with warning);
//!   otherwise requests must carry `Authorization: Bearer <token>`.
//! * **Scoped tokens.** Each entry in `TENSOR_WASM_API_TOKENS` may carry a
//!   `:tenant=` scope clause (`token:tenant=1,2,3` or `token:tenant=*`).
//!   Routes that bind to a tenant return `403 tenant_scope_denied` when the
//!   caller's bearer token is not scoped to that tenant. Bare entries are
//!   treated as wildcard with a one-shot deprecation warning at startup
//!   (removal targeted for v1.0). See [`token_scope`].
//! * **Tenant scoping.** The `X-TensorWasm-Tenant: <u64>` header is parsed and
//!   threaded through to the executor. Set `TENSOR_WASM_API_REQUIRE_TENANT=1`
//!   to make the header mandatory.
//! * **Per-token rate limiting.** Configurable QPS + burst per bearer token,
//!   enforced behind bearer auth. Reads
//!   `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST`;
//!   either zero or unset disables the limiter. Rejections return
//!   `429 Too Many Requests` with a `Retry-After` header. See
//!   [`rate_limit`].
//! * **Audit log.** Every state-mutating call (`POST /functions`,
//!   `DELETE /functions/{id}`, `POST /functions/{id}/invoke[-async]`)
//!   emits a structured JSON record to the sink selected by
//!   `TENSOR_WASM_API_AUDIT_LOG` (default: stdout). Read-only routes
//!   emit nothing. See [`audit`] and `docs/AUDIT-LOG.md`.
//! * **Snapshot HMAC key (forward-looking).** When
//!   `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` is set (64-char hex, 32 bytes)
//!   the future `/snapshot/save` and `/snapshot/restore` routes will
//!   HMAC-SHA256 sign on save and verify on restore. Set
//!   `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE=true` to additionally
//!   reject unsigned v2 blobs. The routes themselves are not yet wired
//!   (see [`config`] for the schema and `crates/tensor-wasm-cli/src/cmd/snapshot.rs`
//!   for the matching CLI shim that returns `FEATURE_NOT_EXPOSED` today).
//! * **HTTP request metrics.** A tower middleware emits
//!   `tensor_wasm_http_requests_total`,
//!   `tensor_wasm_http_request_duration_seconds`, and
//!   `tensor_wasm_http_requests_in_flight` per `(route, method, status)`,
//!   labelled with the axum route template (never the substituted id).
//!   The layer sits OUTSIDE bearer auth so `401`/`429` responses are
//!   counted too — required by the `availability_http` SLI in
//!   `docs/SLO.md`. See [`http_metrics`].
//!
//! See [`API.md`](../API.md) for the wire-format contract.
#![deny(missing_docs)]

pub mod audit;
pub mod config;
pub mod http_metrics;
pub mod middleware;
pub mod rate_limit;
pub mod routes;
pub mod server;
pub mod token_scope;
pub mod trace_propagation;

pub use audit::{
    audit_log_middleware, AuditAction, AuditActor, AuditActorKind, AuditConfig, AuditOutcome,
    AuditRecord, AuditResource, AuditSink, FileJsonSink, NoopSink, StdoutJsonSink,
    TokenScopeView, ENV_AUDIT_LOG,
};
pub use config::{
    AppConfig, ConfigError, HexParseReason, ENV_SNAPSHOT_HMAC_KEY,
    ENV_SNAPSHOT_REQUIRE_SIGNATURE, SNAPSHOT_HMAC_KEY_LEN,
};
pub use http_metrics::{
    http_metrics_middleware, HttpMetricsLayerConfig, RouteAllowList, DEFAULT_ROUTE_ALLOWLIST,
    UNKNOWN_ROUTE,
};
pub use middleware::{
    normalize_method, sanitize_path, AuthConfig, CorsConfig, TenantConfig, TrustedHosts,
    ENV_API_TOKENS, ENV_CORS_ALLOWED_ORIGINS, ENV_REQUIRE_TENANT, ENV_TRUSTED_HOSTS,
    HEADER_TENANT, MAX_PATH_LEN, MAX_REQUEST_BODY_BYTES,
};
pub use rate_limit::{
    AuthContext, PerTenantRateLimitConfig, RateLimitConfig, RateLimiter, TokenId,
};
pub use routes::{ApiError, AppState, FunctionRecord, JobRecord, JobStatus};
pub use server::{
    build_router, build_router_with_audit, build_router_with_config,
    build_router_with_full_config, serve,
};
pub use token_scope::{
    parse_token_entry, parse_tokens_env, ParsedTokens, ScopeParseError, TenantScope, TokenScope,
};
pub use trace_propagation::{
    current_trace_id, extract_parent_context, inject_trace_id_header, install_w3c_propagator,
    HeaderMapExtractor, HEADER_TRACE_ID,
};
