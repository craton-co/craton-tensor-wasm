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
//! * **XFCC spoofing gate via `TENSOR_WASM_API_TRUSTED_XFCC_PROXIES`.**
//!   Comma-separated allowlist of IPv4/IPv6 addresses or CIDR ranges
//!   whose `X-Forwarded-Client-Cert` headers the audit middleware will
//!   honour. Empty / unset = trust nobody (safe default — every inbound
//!   XFCC is dropped). Deployments behind an mTLS-terminating proxy
//!   should set this to the proxy's IP(s); operators not running an
//!   XFCC-stripping proxy should leave it unset. See
//!   [`audit::TrustedProxies`].
//! * **Snapshot HMAC key.** When
//!   `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` is set (64-char hex, 32 bytes)
//!   the `/snapshot/save` route HMAC-SHA256-signs the snapshot blob it
//!   returns and `/snapshot/restore` verifies the signature (always
//!   requiring a signature on restore). Set
//!   `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE=true` for the matching
//!   strict-restore posture. The routes are wired into the protected
//!   stack (bearer auth + tenant scope + per-tenant ownership) by
//!   [`build_router`], which reads the key via
//!   [`AppConfig::from_env`](config::AppConfig::from_env); when the key is
//!   unset both routes return `503 snapshot_signing_not_configured`. See
//!   [`routes::snapshot_save`] / [`routes::snapshot_restore`] and
//!   [`config`] for the schema.
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
#[cfg(feature = "kernel-registry-api")]
pub mod kernels;
pub mod middleware;
pub mod openai;
pub mod openai_translator;
pub mod rate_limit;
pub mod routes;
pub mod server;
pub mod token_scope;
pub mod trace_propagation;

pub use audit::{
    audit_log_middleware, AuditAction, AuditActor, AuditActorKind, AuditConfig, AuditOutcome,
    AuditRecord, AuditResource, AuditSink, FileJsonSink, NoopSink, StdoutJsonSink, TokenScopeView,
    TrustedProxies, ENV_AUDIT_LOG, ENV_TRUSTED_XFCC_PROXIES, HEADER_REQUEST_ID, HEADER_XFCC,
};
pub use config::{
    AppConfig, ConfigError, HexParseReason, ENV_SNAPSHOT_HMAC_KEY, ENV_SNAPSHOT_REQUIRE_SIGNATURE,
    SNAPSHOT_HMAC_KEY_LEN,
};
pub use http_metrics::{
    http_metrics_middleware, HttpMetricsLayerConfig, RouteAllowList, DEFAULT_ROUTE_ALLOWLIST,
    UNKNOWN_ROUTE,
};
pub use middleware::{
    normalize_method, sanitize_path, AuthConfig, CorsConfig, KernelPublishTokens, TenantConfig,
    TrustedHosts, ENV_API_TOKENS, ENV_CORS_ALLOWED_ORIGINS, ENV_KERNEL_PUBLISH_TOKENS,
    ENV_REQUIRE_TENANT, ENV_TRUSTED_HOSTS, HEADER_TENANT, MAX_AUTH_HEADER_BYTES, MAX_PATH_LEN,
    MAX_REQUEST_BODY_BYTES,
};
pub use openai::{
    ChatCompletionsRequest, ChatMessage, CompletionsRequest, OpenAiError, OpenAiErrorBody,
};
pub use openai_translator::{
    assemble_chat_prompt, model_map_from_env, parse_model_map_env,
    translate_chat_completions_request, translate_completions_request, ModelMap, TranslatedRequest,
    ENV_OPENAI_MODEL_MAP,
};
pub use rate_limit::{
    AuthContext, PerTenantRateLimitConfig, RateLimitConfig, RateLimiter, TokenId,
};
pub use routes::{
    snapshot_restore, snapshot_save, ApiError, AppState, FunctionRecord, JobRecord, JobStatus,
    SnapshotRestoreRequest, SnapshotRestoreResponse, SnapshotSaveRequest, SnapshotSaveResponse,
};
pub use server::{
    build_router, build_router_with_audit, build_router_with_config, build_router_with_full_config,
    build_router_with_kernel_publish_tokens, build_router_with_trusted_proxies, serve,
};
pub use token_scope::{
    parse_token_entry, parse_tokens_env, ParsedTokens, ScopeParseError, TenantScope, TokenScope,
};
pub use trace_propagation::{
    current_trace_id, extract_parent_context, inject_trace_id_header, install_w3c_propagator,
    HeaderMapExtractor, HEADER_TRACE_ID,
};
