// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! REST route handlers (deploy, invoke, metrics, healthz).
//!
//! All handlers operate on a shared [`AppState`] containing in-memory registries
//! of deployed functions and pending jobs, plus a shared [`TensorWasmMetrics`]
//! registry and a [`TensorWasmExecutor`]. Wasm bytes are accepted as base64; the
//! deploy path runs full `wasmparser` validation and stores the bytes (as
//! `Arc<[u8]>` so concurrent invocations share a single allocation), and the
//! synchronous invoke path drives `tensor_wasm_exec::executor::TensorWasmExecutor` to
//! spawn, call `_start` / `main`, and terminate the instance. The async
//! `invoke-async` path spawns the same flow on a Tokio task and records
//! progress in the shared job registry for `GET /jobs/{id}` polling.
//!
//! ## Error envelope
//!
//! Every error response carries the JSON envelope:
//!
//! ```json
//! { "error": { "kind": "<machine-readable>", "message": "<human-readable>" } }
//! ```
//!
//! `kind` strings are stable; `message` strings are not.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{rejection::JsonRejection, Extension, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use dashmap::DashMap;
use futures::stream;
use serde::{Deserialize, Serialize};
use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor, WasmArg};
use tensor_wasm_wasi_gpu::streaming::StreamingContext;
use uuid::Uuid;

/// Default per-invocation deadline used by the synchronous `/invoke` handler.
const INVOKE_DEFAULT_DEADLINE: Duration = Duration::from_secs(30);

/// Minimum length, in bytes, of a Wasm module: the 4-byte `\0asm` magic plus
/// the 4-byte version field. Anything shorter cannot be a valid module.
pub const WASM_MIN_HEADER_BYTES: usize = 8;

/// Body-size threshold above which base64 decoding is moved to
/// [`tokio::task::spawn_blocking`]. Below this the inline decode is cheaper
/// than the spawn handoff.
// T19 perf: lowered from 256 KiB to 32 KiB so a 32-concurrent burst
// (per-tenant rate limit) of large-payload invokes can't occupy reactor
// threads on inline base64 decode.
pub const BASE64_OFFLOAD_THRESHOLD: usize = 32 * 1024;

/// Maximum byte-length of a tenant-supplied function name. Names are echoed
/// back on every read of [`FunctionRecord`], so an unchecked name field
/// would let a caller pin arbitrarily many MiB of strings in the in-memory
/// registry. 256 bytes is generous for any realistic display label while
/// keeping the worst-case footprint of a million records under ~256 MiB.
pub const MAX_FUNCTION_NAME_BYTES: usize = 256;

/// Maximum number of elements accepted in an [`InvokeRequest::args`] list.
///
/// The global body-size limit already bounds the *encoded* payload, but a
/// caller could still pack tens of thousands of tiny scalar args into a
/// small JSON body and force the executor to allocate / thread an
/// oversized [`WasmArg`] vector. 256 is far above any realistic export
/// arity while keeping the per-invocation argument footprint bounded.
/// Exceeding it is rejected with `400 too_many_args` in
/// [`parse_invoke_args`].
pub const MAX_INVOKE_ARGS: usize = 256;

// ---------------------------------------------------------------------------
// State records
// ---------------------------------------------------------------------------

/// A deployed function as held in memory by the API gateway.
///
/// `wasm_bytes` is intentionally excluded from the serialised wire form: the
/// API never echoes raw module bytes back to callers. The storage type is
/// `Arc<[u8]>` so concurrent invocations share a single allocation rather
/// than cloning the bytes on every spawn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRecord {
    /// Server-assigned identifier.
    pub id: Uuid,
    /// Tenant that deployed this function. Set from the request tenant at
    /// `POST /functions` time and never mutated. The invoke / delete handlers
    /// compare this against the request's claimed tenant so a wildcard-scoped
    /// caller from tenant B cannot drive tenant A's record (api S-IDOR).
    pub tenant_id: TenantId,
    /// Tenant-supplied display name.
    pub name: String,
    /// Decoded Wasm bytes, refcounted. Not serialised — see struct-level doc.
    #[serde(skip, default = "default_wasm_bytes")]
    pub wasm_bytes: Arc<[u8]>,
    /// Millisecond-precision Unix timestamp of deploy.
    pub created_unix_ms: u64,
}

fn default_wasm_bytes() -> Arc<[u8]> {
    Arc::from(Vec::<u8>::new())
}

/// Status of an asynchronously dispatched invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is queued or in flight.
    Pending,
    /// Job completed successfully; `result` holds the value.
    Completed,
    /// Job failed; `result` carries `{ "kind": ..., "message": ... }`.
    Failed,
}

/// An async-invocation record returned by `GET /jobs/{id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    /// Server-assigned identifier of the job.
    pub id: Uuid,
    /// Function this invocation was dispatched against.
    pub function_id: Uuid,
    /// Tenant this invocation was dispatched under. Set from the request
    /// tenant at `invoke-async` time and never mutated. `GET /jobs/{id}`
    /// compares this against the request's claimed tenant so a
    /// wildcard-scoped caller from another tenant cannot read this job's
    /// result payload (api S-32).
    pub tenant_id: TenantId,
    /// Current status.
    pub status: JobStatus,
    /// Result payload (set when `status` transitions to `completed` or `failed`).
    ///
    /// For `completed` jobs, this holds the JSON result of the invocation.
    /// For `failed` jobs, this is `{ "kind": "...", "message": "..." }`
    /// mirroring the synchronous error envelope shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Millisecond-precision Unix timestamp of dispatch.
    pub created_unix_ms: u64,
}

/// Process-global state shared by every handler.
///
/// `Arc<AppState>` is cloned into the router via `with_state`; the inner
/// `DashMap`s sit behind their own `Arc` so cheap clones of the maps remain
/// possible across the codebase.
#[derive(Clone)]
pub struct AppState {
    /// Deployed-function registry, keyed by id.
    pub functions: Arc<DashMap<Uuid, FunctionRecord>>,
    /// Async-job registry, keyed by id.
    pub jobs: Arc<DashMap<Uuid, JobRecord>>,
    /// Shared metrics registry. Cloned into the executor so spawn/terminate
    /// counters and the `/metrics` scrape view the same atomics.
    pub metrics: Arc<TensorWasmMetrics>,
    /// Wasm executor driving the synchronous `/invoke` path.
    pub executor: Arc<TensorWasmExecutor>,
    /// Optional kernel registry (B6.4 / roadmap feature #3). `None` when
    /// `TENSOR_WASM_API_KERNEL_HMAC_KEY` is unset; the `/kernels` routes
    /// return `503 kernel_registry_not_configured` in that case so
    /// client tools can detect feature availability without inspecting
    /// the URL surface. Only compiled when the `kernel-registry-api`
    /// feature is enabled — the default build keeps the lean dep graph.
    #[cfg(feature = "kernel-registry-api")]
    pub kernel_registry: Option<Arc<dyn tensor_wasm_jit::registry::KernelRegistry>>,
    /// OpenAI `model → FunctionId` resolution map (T41). Read from
    /// `TENSOR_WASM_API_OPENAI_MODEL_MAP` at startup; empty map means
    /// every OpenAI request returns `404 model_not_found`. See
    /// [`crate::openai_translator`] for the env-var grammar.
    pub openai_model_map: crate::openai_translator::ModelMap,
    /// Top-level [`AppConfig`](crate::config::AppConfig) carrying the
    /// snapshot HMAC signing key and the `require_signature` toggle (M5).
    ///
    /// Read from the environment (`TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` /
    /// `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE`) in
    /// [`build_router`](crate::server::build_router) and threaded here so
    /// the `/snapshot/save` and `/snapshot/restore` handlers can sign and
    /// verify blobs with the operator's configured key. When the key is
    /// `None` the snapshot routes return `503 snapshot_signing_not_configured`
    /// — mirroring the kernel-registry posture so a client can detect the
    /// feature being disabled without inspecting the URL surface.
    pub app_config: crate::config::AppConfig,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("functions", &self.functions.len())
            .field("jobs", &self.jobs.len())
            .finish()
    }
}

impl AppState {
    /// Build the inner struct (without the outer `Arc`). Returns
    /// [`TensorWasmError::WasmCompile`] if `TensorWasmEngine::new` fails — typically a
    /// fatal host misconfiguration (unsupported wasmtime strategy, etc.).
    ///
    /// When the `kernel-registry-api` feature is enabled, also reads
    /// `TENSOR_WASM_API_KERNEL_HMAC_KEY` (64-hex chars / 32 bytes) and
    /// constructs an `InMemoryRegistry` keyed off it. An unset env var
    /// leaves `kernel_registry` at `None` so the `/kernels` routes
    /// return `503` rather than silently accepting publishes under a
    /// zero key.
    fn try_build() -> Result<Self, TensorWasmError> {
        let metrics = Arc::new(TensorWasmMetrics::new());
        let engine = Arc::new(
            TensorWasmEngine::new()
                .map_err(|e| TensorWasmError::WasmCompile(format!("{e:#}").into()))?,
        );
        let executor = Arc::new(TensorWasmExecutor::with_metrics(engine, (*metrics).clone()));
        Ok(Self {
            functions: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            metrics,
            executor,
            #[cfg(feature = "kernel-registry-api")]
            kernel_registry: build_kernel_registry_from_env(),
            openai_model_map: crate::openai_translator::model_map_from_env(),
            // Default to the empty config (no snapshot key). Production
            // wiring flows the parsed `AppConfig::from_env()` in through
            // [`AppState::with_app_config`] from `build_router`; tests that
            // construct `AppState::default()` get the keyless config and
            // the `/snapshot/*` routes then report
            // `503 snapshot_signing_not_configured`.
            app_config: crate::config::AppConfig::default(),
        })
    }

    /// Install the top-level [`AppConfig`](crate::config::AppConfig),
    /// carrying the snapshot HMAC key + require-signature toggle (M5).
    ///
    /// Called by [`build_router`](crate::server::build_router) with the
    /// result of [`AppConfig::from_env`](crate::config::AppConfig::from_env)
    /// so the `/snapshot/*` handlers consume the operator-configured key.
    /// Public so integration tests can drive the snapshot routes with an
    /// explicit key without poisoning the process environment.
    #[must_use]
    pub fn with_app_config(mut self, cfg: crate::config::AppConfig) -> Self {
        self.app_config = cfg;
        self
    }

    /// Install an explicit OpenAI model map, bypassing the env-var
    /// initialisation in [`Self::try_build`].
    ///
    /// Test-only convenience so integration tests can drive the OpenAI
    /// shim without poisoning the process environment.
    #[doc(hidden)]
    pub fn with_openai_model_map(mut self, map: crate::openai_translator::ModelMap) -> Self {
        self.openai_model_map = map;
        self
    }

    /// Fallible constructor. Returns a [`TensorWasmError`] if the underlying
    /// wasmtime engine cannot be initialised (e.g. unsupported strategy
    /// on the host). Production callers should use this and surface the
    /// failure through their own startup path.
    pub fn try_new() -> Result<Arc<Self>, TensorWasmError> {
        Self::try_build().map(Arc::new)
    }

    /// Construct an `AppState` wrapped in `Arc` for use with
    /// `Router::with_state`. Panics on engine initialisation failure.
    ///
    /// Prefer [`AppState::try_new`] in production code; this convenience
    /// wrapper exists for parity with [`Default`] and for tests.
    pub fn new() -> Arc<Self> {
        Self::try_new().expect("TensorWasmEngine::new must succeed for AppState::new()")
    }

    /// Override the metrics registry, rebuilding the executor so its
    /// spawn/terminate counters share the supplied handle.
    ///
    /// Useful for tests that need to inspect counter values, or for embedders
    /// who construct a process-wide registry separately. Panics on engine
    /// initialisation failure for ergonomic parity with [`Self::new`].
    pub fn with_metrics(mut self, metrics: Arc<TensorWasmMetrics>) -> Self {
        let engine = Arc::new(
            TensorWasmEngine::new()
                .expect("TensorWasmEngine::new must succeed for AppState::with_metrics"),
        );
        self.executor = Arc::new(TensorWasmExecutor::with_metrics(engine, (*metrics).clone()));
        self.metrics = metrics;
        self
    }

    /// Install an explicit kernel registry, bypassing the env-var
    /// initialisation in [`Self::try_build`].
    ///
    /// Test-only convenience so integration tests can drive `/kernels`
    /// without poisoning the process environment with a hex-encoded
    /// HMAC key. `#[doc(hidden)]` because production code should always
    /// flow through `try_build`'s env-var read.
    #[doc(hidden)]
    #[cfg(feature = "kernel-registry-api")]
    pub fn with_kernel_registry(
        mut self,
        registry: Arc<dyn tensor_wasm_jit::registry::KernelRegistry>,
    ) -> Self {
        self.kernel_registry = Some(registry);
        self
    }
}

/// Read `TENSOR_WASM_API_KERNEL_HMAC_KEY` and build an
/// `Arc<dyn KernelRegistry>` from it.
///
/// Returns `None` when the variable is unset / empty, and also when the
/// value is malformed (not 64 hex chars). Malformed values are logged
/// at `warn` so an operator who typoed a key sees a startup signal — we
/// deliberately do NOT panic, because the gateway should still come up
/// and serve the non-kernel routes; the `/kernels` endpoints will then
/// return `503 kernel_registry_not_configured` and clients can correct
/// the deploy without a full restart loop.
///
/// ## T35: disk-backed registry selection
///
/// If `TENSOR_WASM_API_KERNEL_REGISTRY_DIR` is also set (and non-empty),
/// the gateway constructs a [`tensor_wasm_jit::registry::DiskRegistry`]
/// rooted at that path instead of the historical
/// [`tensor_wasm_jit::registry::InMemoryRegistry`]. The disk-backed path
/// survives process restarts; the in-memory path is now considered
/// dev-only. An `open` failure on the disk path falls back to the
/// in-memory registry and warns rather than dropping the routes
/// entirely — operators who relied on the env-var contract still get a
/// working `/kernels` surface; they just lose the persistence on
/// restart until the disk-path issue is resolved.
///
/// The behaviour mirrors `AppConfig::from_env` for the snapshot HMAC key
/// (see `crate::config`) except that the snapshot path returns a hard
/// error: the kernel registry is a non-critical add-on whereas a snapshot
/// signing key misconfiguration would silently downgrade integrity.
#[cfg(feature = "kernel-registry-api")]
fn build_kernel_registry_from_env() -> Option<Arc<dyn tensor_wasm_jit::registry::KernelRegistry>> {
    /// Environment variable carrying the hex-encoded 32-byte HMAC-SHA256
    /// key the kernel registry uses to verify inbound manifests.
    const ENV_KERNEL_HMAC_KEY: &str = "TENSOR_WASM_API_KERNEL_HMAC_KEY";
    /// T35: when set, the gateway uses a disk-persisted registry
    /// rooted at this path. Unset = legacy in-memory registry.
    const ENV_KERNEL_REGISTRY_DIR: &str = "TENSOR_WASM_API_KERNEL_REGISTRY_DIR";

    let raw = match std::env::var(ENV_KERNEL_HMAC_KEY) {
        Ok(s) => s,
        Err(_) => return None,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() != 64 {
        tracing::warn!(
            target: "tensor_wasm_api::routes",
            var = ENV_KERNEL_HMAC_KEY,
            actual_len = trimmed.len(),
            "kernel registry HMAC key must be exactly 64 hex characters; \
             leaving registry unconfigured (the /kernels routes will return \
             503 kernel_registry_not_configured)",
        );
        return None;
    }
    let mut key = [0u8; 32];
    for (i, slot) in key.iter_mut().enumerate() {
        let bytes = trimmed.as_bytes();
        let hi = match hex_nibble_local(bytes[i * 2]) {
            Some(v) => v,
            None => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    var = ENV_KERNEL_HMAC_KEY,
                    "kernel registry HMAC key contains non-hex characters; \
                     leaving registry unconfigured",
                );
                return None;
            }
        };
        let lo = match hex_nibble_local(bytes[i * 2 + 1]) {
            Some(v) => v,
            None => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    var = ENV_KERNEL_HMAC_KEY,
                    "kernel registry HMAC key contains non-hex characters; \
                     leaving registry unconfigured",
                );
                return None;
            }
        };
        *slot = (hi << 4) | lo;
    }

    // T35: prefer the disk-backed registry when the env var points at a
    // directory. A failed open warns and falls back to the in-memory
    // backend rather than refusing to wire `/kernels` at all — the
    // operator surface is the same on failure, just non-persistent.
    if let Ok(dir_raw) = std::env::var(ENV_KERNEL_REGISTRY_DIR) {
        let dir_trimmed = dir_raw.trim();
        if !dir_trimmed.is_empty() {
            let dir = std::path::PathBuf::from(dir_trimmed);
            match tensor_wasm_jit::registry::DiskRegistry::open(dir.clone(), key) {
                Ok(reg) => {
                    tracing::info!(
                        target: "tensor_wasm_api::routes",
                        dir = %dir.display(),
                        "kernel registry disk-backed (T35); /kernels routes live",
                    );
                    return Some(Arc::new(reg));
                }
                Err(e) => {
                    tracing::warn!(
                        target: "tensor_wasm_api::routes",
                        dir = %dir.display(),
                        error = %e,
                        var = ENV_KERNEL_REGISTRY_DIR,
                        "disk-backed kernel registry open failed; falling back \
                         to in-memory registry (manifests will NOT survive restart)",
                    );
                }
            }
        }
    }

    tracing::info!(
        target: "tensor_wasm_api::routes",
        "kernel registry HMAC key configured (64 chars hex); /kernels routes live (in-memory)",
    );
    Some(Arc::new(tensor_wasm_jit::registry::InMemoryRegistry::new(
        key,
    )))
}

/// Local hex-nibble decoder for the kernel registry env-var path. We do
/// NOT route this through `crate::config::parse_hex_key` because that
/// helper returns a typed [`crate::config::ConfigError`] which the
/// kernel-registry initialiser deliberately discards (it logs and
/// degrades rather than failing startup).
#[cfg(feature = "kernel-registry-api")]
fn hex_nibble_local(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl Default for AppState {
    /// Default `AppState`. Calls the internal builder and unwraps for the
    /// same reasons documented on [`AppState::new`]: in practice
    /// `TensorWasmEngine::new` only fails on host misconfiguration, which is a
    /// fatal startup condition. Used by tests via `AppState::default()`.
    fn default() -> Self {
        Self::try_build().expect("TensorWasmEngine::new must succeed for AppState::default")
    }
}

fn now_unix_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_millis() as u64,
        Err(e) => {
            // System clock is set before 1970-01-01 — almost certainly a
            // misconfigured container. Log loud and return 0; a bogus
            // timestamp is better than panicking a request handler.
            tracing::warn!(
                target: "tensor_wasm_api::routes",
                error = %e,
                "system clock is before UNIX_EPOCH; returning 0 timestamp",
            );
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Error envelope
// ---------------------------------------------------------------------------

/// Inner body of the JSON error envelope.
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    /// Machine-readable, stable identifier.
    pub kind: String,
    /// Human-readable description (not stable across versions).
    pub message: String,
}

/// Top-level JSON error envelope as serialised to the wire:
/// `{"error":{"kind":..., "message":...}}`.
#[derive(Debug, Serialize)]
pub struct ApiErrorEnvelope {
    /// Inner error.
    pub error: ApiErrorBody,
}

/// Error type returned by every fallible handler.
#[derive(Debug)]
pub struct ApiError {
    /// HTTP status code to send.
    pub status: StatusCode,
    /// Stable machine-readable identifier (the `kind` field on the wire).
    pub kind: String,
    /// Human-readable description.
    pub message: String,
}

impl ApiError {
    /// Construct a `400 Bad Request` with the given `kind` and `message`.
    pub fn bad_request(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: kind.into(),
            message: message.into(),
        }
    }

    /// Construct a `403 Forbidden` with the given `kind` and `message`. Used
    /// by per-tenant authorization to return `kind = "tenant_scope_denied"`
    /// when the caller's bearer token does not cover the bound tenant.
    pub fn forbidden(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            kind: kind.into(),
            message: message.into(),
        }
    }

    /// Construct a `404 Not Found` with `kind = "not_found"`.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "not_found".to_string(),
            message: message.into(),
        }
    }

    /// Construct a `409 Conflict` with the given `kind` and `message`.
    ///
    /// Used by the kernel registry's `POST /kernels` path to surface
    /// `kind = "already_registered"` when a manifest with the same
    /// `name@version` has already been published. The `(409, kind)`
    /// pair is the documented contract for "the request is well-formed
    /// but would violate a uniqueness invariant"; clients should NOT
    /// retry without changing the request (no `Retry-After` header).
    pub fn conflict(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            kind: kind.into(),
            message: message.into(),
        }
    }

    /// Construct a `503 Service Unavailable` with the given `kind` and
    /// `message`.
    ///
    /// Used by the kernel registry routes to surface
    /// `kind = "kernel_registry_not_configured"` when the gateway is
    /// running without `TENSOR_WASM_API_KERNEL_HMAC_KEY` set. Distinct
    /// from `capacity_exhausted` (also 503) because the failure mode is
    /// configuration, not load — a client should NOT retry, it should
    /// surface the error to an operator who can flip the env knob.
    pub fn service_unavailable(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: kind.into(),
            message: message.into(),
        }
    }

    /// Construct a `500 Internal Server Error` with `kind = "internal"`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal".to_string(),
            message: message.into(),
        }
    }

    /// Construct a `501 Not Implemented` with the given `kind` and `message`.
    ///
    /// Used by the snapshot routes (M5) for the parts of full
    /// live-instance snapshot / restore that still need executor support:
    /// `/snapshot/save` signs and returns a snapshot blob built from the
    /// function's deployed bytes (the HMAC envelope layer is fully wired),
    /// but capturing a *running* instance's linear / GPU memory needs a
    /// `TensorWasmExecutor` capture hook that does not exist yet — that
    /// capability surfaces `501 not_implemented` rather than silently
    /// returning an empty or misleading capture. The `(501, kind)` pair is
    /// the documented contract; clients should NOT retry.
    pub fn not_implemented(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: kind.into(),
            message: message.into(),
        }
    }

    /// Construct a `413 Payload Too Large` with `kind = "body_too_large"`.
    ///
    /// Returned when an inbound request body exceeds the global
    /// [`MAX_REQUEST_BODY_BYTES`](crate::MAX_REQUEST_BODY_BYTES) cap. The
    /// rejection is surfaced through axum's
    /// [`DefaultBodyLimit::max`](axum::extract::DefaultBodyLimit::max) at
    /// extract time — see [`From<JsonRejection>`](#impl-From%3CJsonRejection%3E-for-ApiError)
    /// for the routing that translates the underlying
    /// `JsonRejection::BytesRejection(LengthLimitError)` into this variant
    /// instead of the generic `invalid_json` 400.
    ///
    /// The `body_too_large` kind is pinned in [API.md] and
    /// [openapi.json]; clients can rely on the (kind, status) pair without
    /// inspecting `message`.
    pub fn body_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            kind: "body_too_large".to_string(),
            message: message.into(),
        }
    }

    /// Render this error into the canonical `(kind, message)` pair so the
    /// async job recorder can persist the same shape callers see from the
    /// synchronous path.
    pub fn to_kind_message(&self) -> (String, String) {
        (self.kind.clone(), self.message.clone())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Snapshot the kind before moving it into the envelope so the
        // audit middleware can recover it from the response extensions
        // without re-parsing the JSON body.
        let audit_kind = self.kind.clone();
        let body = ApiErrorEnvelope {
            error: ApiErrorBody {
                kind: self.kind,
                message: self.message,
            },
        };
        let mut response = (self.status, Json(body)).into_response();
        response
            .extensions_mut()
            .insert(crate::audit::AuditOutcomeExt {
                error_kind: audit_kind,
            });
        response
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rej: JsonRejection) -> Self {
        // When the inbound body exceeded the
        // [`DefaultBodyLimit::max`](axum::extract::DefaultBodyLimit::max)
        // cap, axum's `Json` extractor wraps the underlying
        // `LengthLimitError` as `JsonRejection::BytesRejection`, whose
        // `status()` is `413 PAYLOAD_TOO_LARGE`. Route those through the
        // dedicated `body_too_large` envelope rather than the generic
        // `invalid_json` 400 — the public contract in `API.md` (and the
        // `oversized_body_is_rejected` test) pins this kind/status pair.
        // Other JsonRejection variants (syntax errors, missing
        // content-type, schema validation) remain `invalid_json` / 400.
        if rej.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::body_too_large(rej.body_text())
        } else {
            ApiError::bad_request("invalid_json", rej.body_text())
        }
    }
}

impl From<ExecError> for ApiError {
    fn from(err: ExecError) -> Self {
        // SECURITY (api S-22, api T3): pre-W4.x this impl used
        // `err.to_string()` verbatim as the wire `message`. That leaked
        // server-internal state to untrusted callers in two ways:
        //
        //   * `ExecError::Wasmtime(_)` surfaced the full wasmtime error
        //     chain (host pointer addresses, host file paths, internal
        //     stack-frame names).
        //   * The structured variants (`NotFound`, `MissingExport`,
        //     `Timeout`, `ModuleMemoryTooLarge`, `ModuleTooLarge`,
        //     `CapacityExhausted`, `EpochTickerNotRunning`) embedded
        //     internal instance IDs, deadline figures, declared memory
        //     sizes, capacity counters, and export names.
        //
        // We now branch per-variant and emit a fixed, stable wire
        // message for every variant. The original (verbose) error is
        // logged server-side via `tracing::warn!` / `tracing::error!`
        // with structured fields so operators retain forensics without
        // leaking the same state into client responses.
        match &err {
            ExecError::NotFound(id) => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    instance_id = %id,
                    "exec error: instance not found",
                );
                ApiError {
                    status: StatusCode::NOT_FOUND,
                    kind: "instance_not_found".to_string(),
                    message: "function not found".to_string(),
                }
            }
            ExecError::MissingExport(name) => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    export = %name,
                    "exec error: requested export missing from module",
                );
                ApiError {
                    status: StatusCode::BAD_REQUEST,
                    kind: "missing_export".to_string(),
                    message: "requested export not found in module".to_string(),
                }
            }
            ExecError::Timeout(ctx) => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    instance_id = %ctx.id,
                    elapsed_ms = ctx.elapsed_ms,
                    deadline_ms = ctx.deadline_ms,
                    "exec error: invocation deadline exceeded",
                );
                ApiError {
                    status: StatusCode::GATEWAY_TIMEOUT,
                    kind: "invoke_timeout".to_string(),
                    message: "invocation deadline exceeded".to_string(),
                }
            }
            // ExecError::Wasmtime collapses both runtime traps and compile
            // failures; the executor distinguishes them only when converting
            // to TensorWasmError. For the API surface we keep a single 500
            // with a stable opaque message — the real chain is logged.
            ExecError::Wasmtime(inner) => {
                tracing::error!(
                    target: "tensor_wasm_api::routes",
                    error = ?inner,
                    error_chain = %format!("{inner:#}"),
                    "execution error mapped to 500",
                );
                ApiError {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    kind: "wasmtime".to_string(),
                    message: "internal execution error".to_string(),
                }
            }
            // Per mem H5 + exec S-2: module's declared linear memory exceeds
            // the engine's per-tenant cap. Surface as 413 so clients can
            // distinguish quota rejection from a generic compile failure.
            // The requested / limit byte figures are operator-only state.
            ExecError::ModuleMemoryTooLarge {
                requested_bytes,
                limit_bytes,
            } => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    requested_bytes = *requested_bytes,
                    limit_bytes = *limit_bytes,
                    "exec error: module declares memory above per-instance cap",
                );
                ApiError {
                    status: StatusCode::PAYLOAD_TOO_LARGE,
                    kind: "module_memory_too_large".to_string(),
                    message: "module declares memory above per-instance cap".to_string(),
                }
            }
            // Per exec S-10: the engine-wide live-instance cap is
            // saturated. 503 is the right code (the request is well-
            // formed and would succeed once load drops) so clients with
            // retry-with-backoff handling recover cleanly. The active /
            // limit counters are server-internal capacity state.
            ExecError::CapacityExhausted { active, limit } => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    active = *active,
                    limit = *limit,
                    "exec error: engine instance capacity exhausted",
                );
                ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    kind: "capacity_exhausted".to_string(),
                    message: "engine instance capacity exhausted; retry later".to_string(),
                }
            }
            // Per B3.2: adversarial Wasm bytes that exceed the pre-compile
            // size cap. 413 mirrors the body-too-large family. The
            // observed length and configured cap are operator-only state.
            ExecError::ModuleTooLarge { len, max } => {
                tracing::warn!(
                    target: "tensor_wasm_api::routes",
                    len = *len,
                    max = *max,
                    "exec error: module bytes above per-tenant cap",
                );
                ApiError {
                    status: StatusCode::PAYLOAD_TOO_LARGE,
                    kind: "module_too_large".to_string(),
                    message: "module bytes above per-tenant cap".to_string(),
                }
            }
            // Per B3.2: spawn refused because the epoch ticker is down and
            // a deadline-class bound would otherwise apply. 500 — the
            // executor is mis-configured. The remediation hint embedded
            // in the underlying error is operator-only.
            ExecError::EpochTickerNotRunning => {
                tracing::error!(
                    target: "tensor_wasm_api::routes",
                    "exec error: engine deadline ticker not running",
                );
                ApiError {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    kind: "epoch_ticker_not_running".to_string(),
                    message: "engine deadline ticker not running".to_string(),
                }
            }
        }
    }
}

/// Convenience alias.
pub type ApiResult<T> = Result<T, ApiError>;

// ---------------------------------------------------------------------------
// Request / response payloads
// ---------------------------------------------------------------------------

/// Body of `POST /functions`.
#[derive(Debug, Deserialize)]
pub struct CreateFunctionRequest {
    /// Tenant-supplied display name. Free-form, must be non-empty.
    pub name: String,
    /// Base64-encoded Wasm module bytes (standard alphabet, padded).
    pub wasm_b64: String,
}

/// Response body of `POST /functions`.
#[derive(Debug, Serialize)]
pub struct CreateFunctionResponse {
    /// Server-assigned identifier of the newly deployed function.
    pub id: Uuid,
}

/// Body of `POST /functions/{id}/invoke` and `POST /functions/{id}/invoke-async`.
///
/// Both fields are optional so callers that just want the default `_start`
/// → `main` fallback can omit them entirely (an empty `{}` body remains
/// valid). When `args` is supplied each element is converted into a
/// [`WasmArg`] via [`WasmArg::from_json`] before being threaded into
/// [`TensorWasmExecutor::call_export_with_args`].
///
/// `#[serde(default)]` plus `deny_unknown_fields = false` (the default)
/// keeps the schema forward-compatible: adding new optional fields later
/// will not break clients that send the legacy `{}` body, and clients
/// sending arbitrary extra fields are tolerated rather than rejected with
/// 400.
#[derive(Debug, Default, Deserialize)]
pub struct InvokeRequest {
    /// Optional export name override. When `None`, the handler tries
    /// `_start` first and falls back to `main`, matching the historical
    /// behaviour for WASI command modules.
    #[serde(default)]
    pub export: Option<String>,
    /// Optional JSON-array argument list. Each element is parsed into a
    /// [`WasmArg`]; non-numeric elements surface as `400 invalid_args`.
    #[serde(default)]
    pub args: Vec<serde_json::Value>,
}

/// Body of `POST /snapshot/save`.
///
/// Identifies the deployed function whose state is captured into a
/// signed snapshot blob. The captured blob's metadata is stamped with the
/// resolved tenant so a later `/snapshot/restore` can enforce that the
/// snapshot is replayed under the same tenant it was taken for.
#[derive(Debug, Deserialize)]
pub struct SnapshotSaveRequest {
    /// Server-assigned identifier of the function to snapshot. Must name a
    /// function owned by the caller's resolved tenant (cross-tenant ids are
    /// rejected with `403 tenant_scope_denied`).
    pub function_id: Uuid,
}

/// Response body of `POST /snapshot/save`.
///
/// The `snapshot_b64` field carries the full signed snapshot blob
/// (base64, standard alphabet, padded) — the exact bytes a caller hands
/// back to `POST /snapshot/restore`. The HMAC trailer is part of those
/// bytes; the gateway never returns the signing key.
#[derive(Debug, Serialize)]
pub struct SnapshotSaveResponse {
    /// Function the snapshot was taken for (echoed for caller correlation).
    pub function_id: Uuid,
    /// Base64-encoded signed snapshot blob (HMAC-SHA256, v3 envelope).
    pub snapshot_b64: String,
    /// Whether the blob carries an HMAC signature. Always `true` on this
    /// path — the route is only reachable when a key is configured.
    pub signed: bool,
}

/// Body of `POST /snapshot/restore`.
///
/// Carries a base64-encoded snapshot blob previously produced by
/// `POST /snapshot/save`. The handler verifies the blob's HMAC-SHA256
/// signature against the gateway's configured key before decoding any
/// inner payload.
#[derive(Debug, Deserialize)]
pub struct SnapshotRestoreRequest {
    /// Base64-encoded signed snapshot blob (standard alphabet, padded).
    pub snapshot_b64: String,
}

/// Response body of `POST /snapshot/restore`.
///
/// Reports the verified provenance of the restored blob. Full
/// reconstitution of a *running* instance from the captured memory is not
/// yet wired (the executor exposes no restore hook); this response
/// confirms the HMAC-verified envelope and its metadata so the caller can
/// observe the round-trip succeeded.
#[derive(Debug, Serialize)]
pub struct SnapshotRestoreResponse {
    /// Tenant the snapshot was originally captured for, recovered from the
    /// HMAC-authenticated metadata.
    pub tenant_id: u64,
    /// Instance id stamped into the snapshot metadata at capture time.
    pub instance_id: String,
    /// Total uncompressed payload bytes recorded in the snapshot metadata.
    pub total_uncompressed_bytes: u64,
    /// Snapshot wire-format version that verified (`3` for the signed
    /// envelope this gateway writes).
    pub version: u32,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /healthz` — liveness probe. Returns `{"status":"ok"}`.
pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `GET /metrics` — Prometheus text exposition.
///
/// Renders the shared [`TensorWasmMetrics`] registry into the standard text-format
/// (`text/plain; version=0.0.4`). Every metric registered in
/// `tensor_wasm_core::metrics` is exposed; counter names carry the `_total` suffix
/// per Prometheus convention.
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.encode_text();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
}

/// Validate a tenant-supplied function name before it is committed to the
/// in-memory registry.
///
/// Three classes of input are rejected with `400 invalid_name`:
///
/// * empty / whitespace-only names — there is nothing for an operator to
///   recognise the record by;
/// * names longer than [`MAX_FUNCTION_NAME_BYTES`] — the name is echoed back
///   on every `FunctionRecord` read, so an unbounded field would let a
///   caller anchor arbitrary memory in the registry by submitting many
///   records with multi-MiB names;
/// * names containing ASCII / Unicode control characters — these break log
///   readability and would let a caller smuggle NULs or escape sequences
///   into downstream consumers.
fn validate_function_name(name: &str) -> Result<(), ApiError> {
    if name.trim().is_empty() {
        return Err(ApiError::bad_request(
            "invalid_name",
            "function name must not be empty",
        ));
    }
    if name.len() > MAX_FUNCTION_NAME_BYTES {
        return Err(ApiError::bad_request(
            "invalid_name",
            "function name exceeds 256 bytes",
        ));
    }
    if name.chars().any(|c| c.is_control()) {
        return Err(ApiError::bad_request(
            "invalid_name",
            "function name contains control characters",
        ));
    }
    Ok(())
}

/// Decode `wasm_b64`, offloading to a blocking pool when the encoded payload
/// is large enough that the spawn handoff is cheaper than blocking the I/O
/// thread. The threshold is [`BASE64_OFFLOAD_THRESHOLD`] *encoded* bytes.
async fn decode_wasm_b64(wasm_b64: String) -> Result<Vec<u8>, ApiError> {
    if wasm_b64.len() < BASE64_OFFLOAD_THRESHOLD {
        return BASE64
            .decode(wasm_b64.as_bytes())
            .map_err(base64_decode_error);
    }
    tokio::task::spawn_blocking(move || BASE64.decode(wasm_b64.as_bytes()))
        .await
        .map_err(|e| ApiError::internal(format!("base64 decoder panicked: {e}")))?
        .map_err(base64_decode_error)
}

/// Map a base64 decode failure to a fixed-message `400 invalid_base64`.
///
/// L8: the underlying `base64::DecodeError` text echoes the offending
/// byte offset / alphabet detail of attacker-supplied input. We log the
/// detail server-side via `tracing` for forensics and return only a
/// stable, content-free message to the caller.
fn base64_decode_error(e: base64::DecodeError) -> ApiError {
    tracing::warn!(
        target: "tensor_wasm_api::routes",
        error = %e,
        "rejected create_function: base64 decode failed",
    );
    ApiError::bad_request("invalid_base64", "invalid base64 payload")
}

/// `POST /functions` — deploy a new Wasm module.
///
/// Decodes the supplied base64, runs full `wasmparser` validation, and stores
/// the bytes refcounted via `Arc<[u8]>` so concurrent invocations do not
/// reallocate. Returns `400 invalid_wasm` if validation fails.
#[tracing::instrument(
    name = "http.create_function",
    skip(state, auth, payload),
    fields(
        function_id = tracing::field::Empty,
        tenant = tracing::field::Empty,
    ),
)]
pub async fn create_function(
    State(state): State<Arc<AppState>>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    payload: Result<Json<CreateFunctionRequest>, JsonRejection>,
) -> ApiResult<Json<CreateFunctionResponse>> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    // Tenant-scope check: a tenant-scoped token must not deploy under a
    // tenant outside its scope (api B1.9). Mirrors the invoke handlers'
    // ordering — token-scope check before any per-tenant work. Absent
    // AuthContext only happens in routers that bypass `bearer_auth`
    // entirely (ad-hoc test routers); we degrade to dev-mode wildcard.
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }
    let Json(req) = payload?;
    validate_function_name(&req.name)?;
    let bytes = decode_wasm_b64(req.wasm_b64).await?;
    if bytes.len() < WASM_MIN_HEADER_BYTES {
        return Err(ApiError::bad_request(
            "invalid_wasm",
            format!(
                "module too short: {} bytes (minimum {WASM_MIN_HEADER_BYTES})",
                bytes.len()
            ),
        ));
    }
    // Full structural validation. `wasmparser::validate` walks every section
    // and rejects modules wasmtime would later refuse to compile, surfacing
    // the failure at deploy time rather than first invoke.
    //
    // For a 250 KiB module the walk takes 5-20ms of CPU; running it inline
    // on a Tokio reactor thread would block every other connection multiplexed
    // onto that worker. Offload to the blocking pool. We move `bytes` into the
    // closure and recover ownership via the result tuple so the downstream
    // `Arc::from(bytes)` does not need a second allocation.
    let bytes = tokio::task::spawn_blocking(move || {
        let validate_result = wasmparser::validate(&bytes);
        (bytes, validate_result)
    })
    .await
    .map_err(|e| ApiError::internal(format!("wasm validator panicked: {e}")))?;
    let (bytes, validate_result) = bytes;
    if let Err(e) = validate_result {
        // L8: the wasmparser error text describes the offending section /
        // offset of attacker-supplied bytes. Log it server-side for
        // forensics and return only a stable, content-free message.
        tracing::warn!(
            target: "tensor_wasm_api::routes",
            error = %e,
            "rejected create_function: wasm validation failed",
        );
        return Err(ApiError::bad_request("invalid_wasm", "invalid wasm module"));
    }
    let id = Uuid::new_v4();
    tracing::Span::current().record("function_id", tracing::field::display(id));
    state.functions.insert(
        id,
        FunctionRecord {
            id,
            tenant_id: tenant,
            name: req.name,
            wasm_bytes: Arc::from(bytes),
            created_unix_ms: now_unix_ms(),
        },
    );
    Ok(Json(CreateFunctionResponse { id }))
}

/// `DELETE /functions/{id}` — remove a deployed function.
///
/// Returns `204 No Content` on success and `404 Not Found` if the id is
/// unknown.
#[tracing::instrument(
    name = "http.delete_function",
    skip(state, auth),
    fields(
        function_id = %id,
        tenant = tracing::field::Empty,
    ),
)]
pub async fn delete_function(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
) -> ApiResult<StatusCode> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    // Token-scope check first (api B1.9): a token outside the claimed
    // tenant's scope must see `403 tenant_scope_denied` regardless of
    // registry state, so it cannot probe id existence via a 403-vs-404
    // split. The per-resource owner check runs only after.
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }
    // Per-resource owner check: peek the record under the shard lock and
    // verify ownership BEFORE removing it, so a cross-tenant 403 does not
    // side-effect the registry.
    match state.functions.get(&id) {
        Some(entry) => {
            if entry.value().tenant_id != tenant {
                return Err(ApiError::forbidden(
                    "tenant_scope_denied",
                    "function belongs to a different tenant",
                ));
            }
        }
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    }
    if state.functions.remove(&id).is_some() {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format!("function {id} not found")))
    }
}

/// Parse a JSON-array argument list into the executor's [`WasmArg`] form.
///
/// Surfaces a `400 invalid_args` envelope on the first element that fails
/// conversion. The envelope's `message` includes the array index and the
/// offending value so a caller debugging a malformed payload can pinpoint
/// the problem without trial-and-error.
fn parse_invoke_args(raw: &[serde_json::Value]) -> Result<Vec<WasmArg>, ApiError> {
    // L9: cap the element count before any per-element conversion work. The
    // body-size limit bounds the encoded bytes but not the element count, so
    // an attacker could otherwise force an oversized `WasmArg` allocation
    // with a compact payload of many tiny scalars.
    if raw.len() > MAX_INVOKE_ARGS {
        return Err(ApiError::bad_request(
            "too_many_args",
            format!(
                "args list has {} elements (maximum {MAX_INVOKE_ARGS})",
                raw.len()
            ),
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(i, v)| {
            WasmArg::from_json(v).map_err(|msg| {
                ApiError::bad_request("invalid_args", format!("args[{i}]: {msg} (value: {v})"))
            })
        })
        .collect()
}

/// Drive the spawn → call(`_start`|`main`|custom) → terminate flow against
/// the supplied bytes/tenant. Shared by the synchronous and async invoke
/// paths so any future fix (telemetry, retries, etc.) lands in one place.
///
/// When `export_override` is `Some`, that export is invoked directly with
/// no fallback — the caller asked for a specific function, so a missing
/// export surfaces as the usual `400 missing_export`. When `None`, the
/// legacy WASI-command discovery applies: try `_start`, then `main`.
#[tracing::instrument(
    name = "invoke.run",
    skip(executor, wasm_bytes, args),
    fields(
        tenant = %tenant,
        function_id = %function_id,
        wasm_bytes_len = wasm_bytes.len(),
        export = tracing::field::Empty,
        args_len = args.len(),
    ),
)]
async fn run_invoke(
    executor: &TensorWasmExecutor,
    wasm_bytes: &[u8],
    tenant: TenantId,
    function_id: Uuid,
    export_override: Option<&str>,
    args: &[WasmArg],
) -> ApiResult<serde_json::Value> {
    // T33 (v0.4): attach the typed argument list to the SpawnConfig as
    // well as passing it explicitly to `call_export_with_args_then_terminate`.
    // The `SpawnConfig::args` field is the canonical carrier for the upcoming
    // call (see `SpawnConfig::args` doc) — wiring it here keeps the API surface
    // honest even though the historical explicit-pass path remains for
    // back-compat with embedders that drive multi-call flows.
    let cfg = SpawnConfig::for_tenant(tenant)
        .with_deadline(INVOKE_DEFAULT_DEADLINE)
        .with_args(args.to_vec());
    let instance_id = executor.spawn_instance(cfg, wasm_bytes).await?;

    // api S-20 / exec orphan-instance: use `call_export_with_args_then_terminate`
    // so the instance is cleaned up even if our future is dropped mid-await
    // (e.g. by tower's `TimeoutLayer`). The previous `call_export` +
    // explicit `terminate` flow leaked the registry entry into
    // `instances` on outer cancellation, holding the wasmtime `Store`
    // and counting against `max_instances` until process restart.
    let result_value: serde_json::Value = if let Some(name) = export_override {
        tracing::Span::current().record("export", tracing::field::display(name));
        executor
            .call_export_with_args_then_terminate(instance_id, name, args)
            .await?
    } else {
        // Try `_start` (WASI command convention) first, then `main`. Anything
        // other than `MissingExport` from the first attempt bubbles up directly;
        // a missing `_start` falls through to `main`. If neither exists the
        // `MissingExport` from the second attempt is returned (mapped to 400).
        tracing::Span::current().record("export", tracing::field::display("_start|main"));
        match executor
            .call_export_with_args_then_terminate(instance_id, "_start", args)
            .await
        {
            Ok(v) => v,
            Err(ExecError::MissingExport(_)) => {
                // `_start` was missing AND the instance was already terminated
                // by the first guard. Re-spawn to try `main` — slightly more
                // expensive than the old "reuse the instance" flow but only
                // when `_start` is genuinely absent, and keeps the auto-
                // terminate invariant intact.
                let cfg = SpawnConfig::for_tenant(tenant)
                    .with_deadline(INVOKE_DEFAULT_DEADLINE)
                    .with_args(args.to_vec());
                let retry_id = executor.spawn_instance(cfg, wasm_bytes).await?;
                executor
                    .call_export_with_args_then_terminate(retry_id, "main", args)
                    .await?
            }
            Err(other) => return Err(other.into()),
        }
    };

    // For back-compat with the historical `{ "result": "ok" }` envelope,
    // collapse an empty result array to the string `"ok"`. Non-empty
    // result lists pass through verbatim so callers consuming an `(i32,
    // i32) -> i32` adder see the JSON array shape.
    let payload_result = match &result_value {
        serde_json::Value::Array(items) if items.is_empty() => {
            serde_json::Value::String("ok".to_string())
        }
        _ => result_value,
    };

    Ok(serde_json::json!({
        "function_id": function_id.to_string(),
        "result": payload_result,
    }))
}

/// Extract an [`InvokeRequest`] from the inbound HTTP body, treating an
/// absent / empty body as the default (no export override, no args).
///
/// `/invoke` historically accepted no body (api S-31); now that argument
/// passing is wired through we accept a body but keep the empty-body path
/// cheap — an empty payload short-circuits before the JSON allocator is
/// touched, mirroring the previous behaviour.
async fn read_invoke_request(
    payload: Result<Json<InvokeRequest>, JsonRejection>,
) -> ApiResult<InvokeRequest> {
    match payload {
        Ok(Json(req)) => Ok(req),
        Err(rej) => {
            // Two soft-failure cases get rewritten as the all-defaults
            // request so the legacy "fire-and-forget with no body" wire
            // contract still works:
            //
            //  * `415 Unsupported Media Type` — body present but
            //    `content-type` missing or wrong. The pre-args /invoke
            //    accepted any body shape (it never parsed it); we keep
            //    that surface working by treating it as defaults.
            //  * empty inbound bytes (any rejection whose
            //    `body_text()` is empty) — `curl -X POST /invoke`
            //    with no `-d` body falls in here. Matches the
            //    pre-args silent-no-op behaviour.
            //
            // Every other rejection — `400` (parse error, type error,
            // missing required field) and `413` (body too large) — is
            // forwarded through the existing `From<JsonRejection>`
            // mapping so the canonical envelope kicks in.
            if rej.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
                || rej.body_text().trim().is_empty()
            {
                Ok(InvokeRequest::default())
            } else {
                Err(rej.into())
            }
        }
    }
}

/// `POST /functions/{id}/invoke` — synchronous invocation.
///
/// Looks up the function's stored Wasm bytes, spawns a fresh instance via
/// the shared [`TensorWasmExecutor`] with a 30-second deadline, calls the
/// requested `export` (defaulting to `_start` → `main` discovery) with the
/// supplied `args`, then terminates the instance. Returns the
/// `function_id` and a JSON `result` on success; structured `ApiError`
/// otherwise.
///
/// The tenant id is sourced from the `X-TensorWasm-Tenant` middleware
/// extension; absent it defaults to `TenantId(0)`.
///
/// Body schema is [`InvokeRequest`] (both fields optional). An empty body
/// is treated as the all-defaults case for client compatibility with the
/// pre-args wire contract.
#[tracing::instrument(
    name = "http.invoke_function",
    skip(state, auth, payload),
    fields(
        function_id = %id,
        tenant = tracing::field::Empty,
    ),
)]
pub async fn invoke_function(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    payload: Result<Json<InvokeRequest>, JsonRejection>,
) -> ApiResult<Json<serde_json::Value>> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    // Tenant-scope check: reject before doing any per-tenant work (function
    // lookup, executor spawn, …). Absent AuthContext only happens in
    // configurations that bypass `bearer_auth` entirely (e.g. ad-hoc test
    // routers); we degrade to dev-mode wildcard there.
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }

    let req = read_invoke_request(payload).await?;
    let args = parse_invoke_args(&req.args)?;

    // Snapshot the Wasm bytes under the DashMap shard lock, then drop the
    // guard before we hit any `.await`. `Arc::clone` is a single refcount
    // bump regardless of payload size. We also read the record's owning
    // tenant so the per-resource check below runs against the same guard.
    let (wasm_bytes, owner) = match state.functions.get(&id) {
        Some(entry) => (Arc::clone(&entry.value().wasm_bytes), entry.value().tenant_id),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };
    // Per-resource owner check (api S-IDOR): authorize_tenant above only
    // proves the token may address the *claimed* tenant, not that the
    // looked-up record belongs to it. A wildcard-scoped caller from
    // tenant B must not invoke tenant A's record.
    if owner != tenant {
        return Err(ApiError::forbidden(
            "tenant_scope_denied",
            "function belongs to a different tenant",
        ));
    }

    let value = run_invoke(
        &state.executor,
        &wasm_bytes,
        tenant,
        id,
        req.export.as_deref(),
        &args,
    )
    .await?;
    Ok(Json(value))
}

/// RAII guard pairing one `jobs_active().inc()` with exactly one `.dec()`.
///
/// Constructed at the moment a job is accepted (synchronously, before the
/// spawn); released on the happy path via [`Self::release`] once the job has
/// reached a terminal state. If the owning task instead panics before
/// `release` is called the `Drop` impl decrements the gauge and logs a
/// warning — so an unwinding spawned task does not leak a permanent
/// increment in the `tensor_wasm_jobs_active` series.
///
/// The handle stored is an `Arc<TensorWasmMetrics>` (not a borrowed reference)
/// because the guard outlives the request handler frame: it is moved into
/// the `tokio::spawn` body, which has a `'static` bound. The `Arc` clone is
/// a single refcount bump regardless of registry size.
struct JobsActiveGuard {
    metrics: Arc<TensorWasmMetrics>,
    decremented: bool,
}

impl JobsActiveGuard {
    /// Increment `jobs_active` and return a guard that owns the matching
    /// `dec()`. Always paired one-to-one with a single `release` (happy path)
    /// or `Drop` (panic / early return path).
    fn new(metrics: Arc<TensorWasmMetrics>) -> Self {
        metrics.jobs_active().inc();
        Self {
            metrics,
            decremented: false,
        }
    }

    /// Happy-path release. Consumes the guard, decrements the gauge, and
    /// marks the guard so the subsequent `Drop` is a no-op.
    fn release(mut self) {
        self.metrics.jobs_active().dec();
        self.decremented = true;
    }
}

impl Drop for JobsActiveGuard {
    /// Panic / early-return safety net. Runs only when `release` was *not*
    /// called — i.e. the holding future was cancelled or the spawned task
    /// panicked before reaching its tail. Emits a `warn!` so an operator
    /// scraping logs can correlate a `jobs_active` step-down with the
    /// originating panic; the production happy path uses `release` and
    /// stays silent.
    fn drop(&mut self) {
        if !self.decremented {
            self.metrics.jobs_active().dec();
            tracing::warn!(
                target: "tensor_wasm_api::routes",
                "jobs_active gauge decremented via Drop (likely task panic or cancellation)",
            );
        }
    }
}

/// `POST /functions/{id}/invoke-async` — fire-and-forget invocation.
///
/// Records a `Pending` [`JobRecord`], spawns the spawn/call/terminate flow
/// onto a Tokio task, and returns `202 Accepted` with the job id. The
/// task updates the registry to `Completed` (with the JSON result) or
/// `Failed` (with `{kind, message}`) on conclusion. Callers poll via
/// `GET /jobs/{id}`.
///
/// Body schema mirrors [`invoke_function`]: optional `export` /
/// `args` ([`InvokeRequest`]). The body is parsed synchronously before
/// the Tokio task spawn so `400 invalid_args` surfaces synchronously
/// (rather than as a `JobStatus::Failed` poll result).
#[tracing::instrument(
    name = "http.invoke_function_async",
    skip(state, auth, payload),
    fields(
        function_id = %id,
        tenant = tracing::field::Empty,
        job_id = tracing::field::Empty,
    ),
)]
pub async fn invoke_function_async(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    payload: Result<Json<InvokeRequest>, JsonRejection>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }
    let req = read_invoke_request(payload).await?;
    let args = parse_invoke_args(&req.args)?;
    let export_override = req.export;
    let (wasm_bytes, owner) = match state.functions.get(&id) {
        Some(entry) => (Arc::clone(&entry.value().wasm_bytes), entry.value().tenant_id),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };
    // Per-resource owner check (api S-IDOR): see `invoke_function`. Runs
    // before the JobRecord is inserted so a cross-tenant dispatch leaves
    // no trace in the jobs registry.
    if owner != tenant {
        return Err(ApiError::forbidden(
            "tenant_scope_denied",
            "function belongs to a different tenant",
        ));
    }

    let job_id = Uuid::new_v4();
    tracing::Span::current().record("job_id", tracing::field::display(job_id));
    state.jobs.insert(
        job_id,
        JobRecord {
            id: job_id,
            function_id: id,
            tenant_id: tenant,
            status: JobStatus::Pending,
            result: None,
            created_unix_ms: now_unix_ms(),
        },
    );
    // Account the new pending job in the gauge via a Drop-implementing
    // guard. The matching `.dec()` happens either through `guard.release()`
    // at the end of the spawned task (happy path) or through `Drop` if the
    // task unwinds first (panic safety net). v0.3.x emits a single series;
    // the v0.4 follow-up to break out per tenant lands as a Family swap in
    // `tensor-wasm-core/src/metrics.rs` and a label tuple here.
    let jobs_active_guard = JobsActiveGuard::new(Arc::clone(&state.metrics));

    // Spawn the real invocation. The executor is cheap to clone (it's an
    // `Arc` internally) and the jobs map is `Arc<DashMap>`.
    //
    // The spawned task is wrapped in a dedicated `async_invoke.job` span
    // and instrumented so the active trace context (parent: the
    // `http.invoke_function_async` span we are in right now) carries
    // across the `tokio::spawn` boundary. Without `.instrument(...)` the
    // task would start a fresh root span and the executor /
    // snapshot-restore / dispatch spans it produces would appear
    // disconnected from the inbound HTTP request in the OTLP backend.
    let executor = Arc::clone(&state.executor);
    let jobs = Arc::clone(&state.jobs);
    let job_span = tracing::info_span!(
        "async_invoke.job",
        job_id = %job_id,
        function_id = %id,
        tenant = %tenant,
    );
    tokio::spawn(tracing::Instrument::instrument(
        async move {
            // The guard is moved into the spawned task so its lifetime
            // covers the entire async invocation, including any panic
            // unwind from `run_invoke` or the result-write block.
            let guard = jobs_active_guard;

            // Test-only panic injection point: lets the gauge test
            // exercise the Drop-based dec without needing a wasm
            // module that actually traps an unwind. The probe is a
            // single `Relaxed` atomic load in steady state — see
            // [`test_hooks`] for the rationale.
            test_hooks::maybe_panic_for_test();

            let outcome = run_invoke(
                &executor,
                &wasm_bytes,
                tenant,
                id,
                export_override.as_deref(),
                &args,
            )
            .await;
            if let Some(mut entry) = jobs.get_mut(&job_id) {
                match outcome {
                    Ok(value) => {
                        entry.status = JobStatus::Completed;
                        entry.result = Some(value);
                    }
                    Err(api_err) => {
                        let (kind, message) = api_err.to_kind_message();
                        entry.status = JobStatus::Failed;
                        entry.result = Some(serde_json::json!({
                            "kind": kind,
                            "message": message,
                        }));
                    }
                }
            }
            // Balanced release: paired with the `JobsActiveGuard::new`
            // before the spawn above. Runs once per terminal-state
            // transition regardless of outcome (Completed | Failed) so
            // the gauge converges back to zero on a quiescent node.
            // NOTE: if the jobs map no longer contains the entry (e.g.
            // an admin purge between insert and resolution) we still
            // decrement — the contract is "one `dec` per `inc`", not
            // "one `dec` per surviving JobRecord". If this task panics
            // before reaching `release`, the guard's `Drop` impl
            // decrements the gauge with a warn-level log instead.
            guard.release();
        },
        job_span,
    ));

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id.to_string() })),
    ))
}

/// `Accept` header value that selects the Server-Sent Events response
/// shape on [`invoke_function_stream`]. Anything else (or absence) falls
/// through to the raw chunked-transfer (`application/octet-stream`)
/// branch.
pub const SSE_MIME: &str = "text/event-stream";

/// Content-Type emitted by the chunked-transfer branch of
/// [`invoke_function_stream`] when the request did not negotiate SSE.
pub const CHUNKED_MIME: &str = "application/octet-stream";

/// Returns `true` when the supplied `Accept` header value asks for
/// `text/event-stream`. Tolerates the standard `accept: a, b; q=…`
/// shape: any comma-separated token that starts (after trim) with
/// `text/event-stream` is taken as a match. Quality-value scoring is
/// out of scope for the scaffold; if a client lists multiple types
/// including SSE we serve SSE.
fn accept_wants_sse(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| {
            s.split(',')
                .any(|part| part.trim().to_ascii_lowercase().starts_with(SSE_MIME))
        })
}

/// Channel buffer size for the gateway-side
/// `mpsc::channel::<Vec<u8>>` paired with each `/invoke-stream`
/// invocation.
///
/// 32 frames in flight is enough to absorb a short burst from the
/// guest while the SSE / chunked writer drains its TCP send buffer,
/// but small enough to apply natural back-pressure: once the writer
/// stalls (slow / blocked client), the channel fills, the guest's
/// next `emit-chunk` blocks on `Sender::send`, and the cooperative
/// yield path observes the back-pressure window. See
/// `docs/STREAMING.md` for the threat model.
pub const STREAMING_CHANNEL_BUFFER: usize = 32;

/// `POST /functions/{id}/invoke-stream` — streaming invocation
/// (roadmap feature #2; T34 wired this through end-to-end in v0.4).
///
/// Mirrors the body / auth surface of [`invoke_function`]. The
/// response shape is chosen from the request's `Accept` header:
///
/// * `Accept: text/event-stream` — Server-Sent Events. Each chunk the
///   guest emits via `wasi:tensor/host.emit-chunk` becomes one
///   `event: chunk` frame. A `keep-alive` comment is injected on idle
///   so intermediate proxies don't reap the connection. The stream
///   terminates with an `event: done` frame (`{"status":"ok"}`) on
///   guest success or an `event: error` frame on guest failure.
/// * Anything else — `Content-Type: application/octet-stream`,
///   chunked-transfer encoding. Each guest chunk is forwarded
///   verbatim as one HTTP chunk frame, followed by the same `done` /
///   `error` framing for terminal status.
///
/// ## Wiring
///
/// The handler builds a `tokio::sync::mpsc::channel::<Vec<u8>>` of
/// depth [`STREAMING_CHANNEL_BUFFER`], wraps the sender in a
/// [`StreamingContext`] via [`StreamingContext::with_channel`], and
/// passes it to the executor through
/// [`SpawnConfig::with_streaming`]. The executor's
/// `spawn_instance` path then builds a wasmtime `Linker` registering
/// `wasi:tensor/host.emit-chunk` / `flush` against the context so
/// guest emits land on the matching receiver.
///
/// The receiver is then converted into a `futures::stream::Stream`
/// via `stream::unfold` and either:
///   * wrapped in `axum::response::sse::Sse` (SSE branch) — each
///     `Vec<u8>` becomes one `event: chunk` frame, terminated by a
///     final `event: done` / `event: error`,
///   * collected into a `Body::from_stream` (chunked branch) — each
///     `Vec<u8>` becomes one HTTP chunk frame.
///
/// The guest call runs concurrently with the SSE writer via
/// `tokio::spawn`. A `oneshot::channel` carries the terminal status
/// (success / error / deadline-elapsed) so the writer can emit the
/// final `done` / `error` event.
///
/// ## Cancellation
///
/// If the HTTP client disconnects, axum drops the response future
/// which drops the SSE writer which drops the `mpsc::Receiver`. The
/// guest's next `emit-chunk` then returns `-3` (receiver dropped) and
/// the existing deadline / epoch interrupt tears the instance down.
/// Per `docs/STREAMING.md`, this is the documented disconnect path.
///
/// ## Security
///
/// Per `docs/STREAMING.md`, the host does NOT sanitise chunk
/// payloads — the bytes flow guest→client verbatim. Sanitisation
/// (control-byte / ANSI-escape stripping) is the client's
/// responsibility; the CLI's T18 sanitisation handles received text.
/// The host's contribution is the per-stream byte cap
/// ([`MAX_TOTAL_STREAM_BYTES`](tensor_wasm_wasi_gpu::streaming::MAX_TOTAL_STREAM_BYTES))
/// enforced inside [`StreamingContext::emit_chunk`], which bounds the
/// per-invocation memory footprint independent of the guest's intent.
///
/// ## Body handling
///
/// The request body matches [`InvokeRequest`] (optional `export` and
/// `args`), parsed the same way the synchronous [`invoke_function`]
/// route does. An empty body is treated as the all-defaults case so
/// the historical "fire-and-forget no body" wire contract still
/// works.
#[tracing::instrument(
    name = "http.invoke_function_stream",
    skip(state, auth, headers, payload),
    fields(
        function_id = %id,
        tenant = tracing::field::Empty,
        sse = tracing::field::Empty,
    ),
)]
pub async fn invoke_function_stream(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    headers: HeaderMap,
    payload: Result<Json<InvokeRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }

    // 404 before any negotiation work, mirroring `/invoke`.
    let (wasm_bytes, owner) = match state.functions.get(&id) {
        Some(entry) => (Arc::clone(&entry.value().wasm_bytes), entry.value().tenant_id),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };
    // Per-resource owner check (api S-IDOR): see `invoke_function`. A
    // wildcard-scoped caller from another tenant must not stream tenant
    // A's record.
    if owner != tenant {
        return Err(ApiError::forbidden(
            "tenant_scope_denied",
            "function belongs to a different tenant",
        ));
    }

    let req = read_invoke_request(payload).await?;
    let args = parse_invoke_args(&req.args)?;
    let export_override = req.export;

    let wants_sse = accept_wants_sse(&headers);
    tracing::Span::current().record("sse", tracing::field::display(wants_sse));

    // Build the (sender, receiver) pair. The sender side wraps in a
    // `StreamingContext` that the executor will plumb into the guest
    // store via `SpawnConfig::with_streaming`; the receiver side stays
    // here and feeds the SSE / chunked-transfer response body.
    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(STREAMING_CHANNEL_BUFFER);
    let streaming = StreamingContext::with_channel(chunk_tx);

    // `oneshot` carrying the terminal status so the SSE writer can emit
    // the final `event: done` / `event: error` frame after the guest
    // returns. `Result<(), StreamTerminalError>` distinguishes success
    // from failure without forcing the writer to box every error
    // variant.
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), StreamTerminalError>>();

    // Spawn the executor call concurrently with the SSE writer. The
    // guest emits land on `chunk_rx` while the writer drains; the
    // spawn's terminal status lands on `done_rx`.
    let executor = state.executor.clone();
    tokio::spawn(async move {
        let cfg = SpawnConfig::for_tenant(tenant)
            .with_deadline(INVOKE_DEFAULT_DEADLINE)
            .with_streaming(streaming);
        let outcome = match executor.spawn_instance(cfg, &wasm_bytes).await {
            Ok(instance_id) => {
                let call_result = match export_override.as_deref() {
                    Some(name) => {
                        executor
                            .call_export_with_args_then_terminate(instance_id, name, &args)
                            .await
                    }
                    None => {
                        // Try `_start` then `main`, matching the
                        // synchronous /invoke fallback.
                        match executor
                            .call_export_with_args_then_terminate(instance_id, "_start", &args)
                            .await
                        {
                            Ok(v) => Ok(v),
                            Err(ExecError::MissingExport(_)) => {
                                let cfg2 = SpawnConfig::for_tenant(tenant)
                                    .with_deadline(INVOKE_DEFAULT_DEADLINE);
                                // No streaming on the retry: the first
                                // spawn consumed the StreamingContext.
                                // Guest emits on the retry would `-1`,
                                // but `main` is rare on streaming
                                // workloads — they almost always export
                                // a custom entry point.
                                match executor.spawn_instance(cfg2, &wasm_bytes).await {
                                    Ok(retry_id) => {
                                        executor
                                            .call_export_with_args_then_terminate(
                                                retry_id, "main", &args,
                                            )
                                            .await
                                    }
                                    Err(e) => Err(e),
                                }
                            }
                            Err(other) => Err(other),
                        }
                    }
                };
                call_result.map(|_| ())
            }
            Err(e) => Err(e),
        };
        let terminal = match outcome {
            Ok(()) => Ok(()),
            Err(ExecError::Timeout(ctx)) => {
                // T36 cooperative-yield ↔ deadline integration: a
                // wall-clock deadline elapse surfaces as a structured
                // `deadline_elapsed` error event so SSE clients can
                // distinguish it from a generic trap.
                Err(StreamTerminalError {
                    kind: "deadline_elapsed",
                    message: format!(
                        "instance exceeded deadline ({} ms / {} ms)",
                        ctx.elapsed_ms, ctx.deadline_ms,
                    ),
                })
            }
            Err(e) => Err(StreamTerminalError {
                kind: "wasm_error",
                message: format!("{e}"),
            }),
        };
        // Best-effort: if the receiver is gone (client disconnected)
        // we have nothing to deliver — the response future already
        // dropped.
        let _ = done_tx.send(terminal);
    });

    // Build the body stream. The shape is the same for both SSE and
    // chunked branches: drain `chunk_rx` until empty AND the
    // `done_rx` has fired, then emit one final terminal event.
    let metrics_for_writer = state.metrics.clone();
    let initial = StreamWriterState::Streaming(chunk_rx, done_rx);
    let body_stream = stream::unfold(initial, move |st| {
        let metrics = metrics_for_writer.clone();
        async move {
            match st {
                StreamWriterState::Streaming(mut rx, mut done_rx) => {
                    // Race the next chunk against the terminal signal.
                    // `biased` so a ready chunk is delivered before
                    // the terminal frame fires — important when the
                    // guest emits N chunks and immediately returns,
                    // since both branches are then ready at once.
                    tokio::select! {
                        biased;
                        maybe_chunk = rx.recv() => {
                            match maybe_chunk {
                                Some(c) => {
                                    metrics.streaming_chunks_emitted_total().inc();
                                    Some((
                                        StreamFrame::Chunk(c),
                                        StreamWriterState::Streaming(rx, done_rx),
                                    ))
                                }
                                None => {
                                    // Channel closed (sender dropped =
                                    // guest finished + executor task
                                    // completed). Pick up the terminal
                                    // status; if the oneshot is gone
                                    // too, surface a wasm_error.
                                    let terminal = done_rx.await.unwrap_or_else(|_| {
                                        Err(StreamTerminalError {
                                            kind: "wasm_error",
                                            message: "executor task dropped without signalling".to_string(),
                                        })
                                    });
                                    Some((StreamFrame::Done(terminal), StreamWriterState::Done))
                                }
                            }
                        }
                        done = &mut done_rx => {
                            // Guest finished. Drain any in-flight
                            // chunks before emitting the terminal
                            // frame so the client sees every chunk
                            // the guest successfully emitted.
                            let terminal = done.unwrap_or_else(|_| {
                                Err(StreamTerminalError {
                                    kind: "wasm_error",
                                    message: "executor task dropped without signalling".to_string(),
                                })
                            });
                            // Try to pop one more chunk synchronously
                            // — `try_recv` returns Empty if the buffer
                            // is drained, in which case we emit the
                            // terminal frame immediately.
                            match rx.try_recv() {
                                Ok(c) => {
                                    metrics.streaming_chunks_emitted_total().inc();
                                    Some((
                                        StreamFrame::Chunk(c),
                                        StreamWriterState::DrainOnly(rx, Some(terminal)),
                                    ))
                                }
                                Err(_) => Some((
                                    StreamFrame::Done(terminal),
                                    StreamWriterState::Done,
                                )),
                            }
                        }
                    }
                }
                StreamWriterState::DrainOnly(mut rx, terminal) => match rx.try_recv() {
                    Ok(c) => {
                        metrics.streaming_chunks_emitted_total().inc();
                        Some((
                            StreamFrame::Chunk(c),
                            StreamWriterState::DrainOnly(rx, terminal),
                        ))
                    }
                    Err(_) => Some((
                        StreamFrame::Done(terminal.unwrap_or(Ok(()))),
                        StreamWriterState::Done,
                    )),
                },
                StreamWriterState::Done => None,
            }
        }
    });

    use futures::StreamExt;
    if wants_sse {
        let sse_stream = body_stream.map(|item| {
            let ev = match item {
                StreamFrame::Chunk(bytes) => {
                    // SSE `data:` requires UTF-8; for arbitrary bytes
                    // we lossy-decode so the wire stays valid. Strict
                    // byte-preservation is the chunked-transfer
                    // branch's job. The CLI's T18 sanitisation is
                    // responsible for control-byte handling on the
                    // receive side.
                    let s = String::from_utf8_lossy(&bytes).into_owned();
                    Event::default().event("chunk").data(s)
                }
                StreamFrame::Done(Ok(())) => Event::default()
                    .event("done")
                    .data(serde_json::json!({"status":"ok"}).to_string()),
                StreamFrame::Done(Err(err)) => Event::default().event("error").data(
                    serde_json::json!({
                        "reason": err.kind,
                        "message": err.message,
                    })
                    .to_string(),
                ),
            };
            Ok::<Event, std::convert::Infallible>(ev)
        });
        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Chunked-transfer branch. Each chunk is forwarded verbatim;
        // the terminal `done` / `error` frame is formatted as an
        // SSE-style line so existing clients that parse the chunked
        // body uniformly can detect end-of-stream without inspecting
        // headers.
        let byte_stream = body_stream.map(|item| {
            let bytes: axum::body::Bytes = match item {
                // M6 / control-byte hazard: guest chunk bytes are
                // forwarded verbatim here — NOT sanitised. Per the
                // streaming contract (see the `## Security` section on
                // this handler and `docs/STREAMING.md`) the host does
                // not strip control characters or ANSI escape
                // sequences, so a malicious guest can emit escape
                // sequences that reach a non-sanitising consumer (a raw
                // terminal, a log tail, a downstream pipe) and alter its
                // display state or smuggle in injected output. This is a
                // deliberate contract decision (byte-exact passthrough);
                // sanitisation is the client's responsibility (the CLI's
                // T18 layer handles received text). Do not change the
                // byte stream here without revisiting that contract.
                StreamFrame::Chunk(c) => axum::body::Bytes::from(c),
                StreamFrame::Done(Ok(())) => axum::body::Bytes::from(format!(
                    "event: done\ndata: {}\n\n",
                    serde_json::json!({"status":"ok"})
                )),
                StreamFrame::Done(Err(err)) => axum::body::Bytes::from(format!(
                    "event: error\ndata: {}\n\n",
                    serde_json::json!({"reason": err.kind, "message": err.message})
                )),
            };
            Ok::<axum::body::Bytes, std::io::Error>(bytes)
        });
        let mut resp = Response::new(Body::from_stream(byte_stream));
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            CHUNKED_MIME.parse().expect("static mime parses"),
        );
        Ok(resp)
    }
}

/// Terminal status carried from the executor task to the SSE writer
/// via the `oneshot` channel. `kind` is the stable machine-readable
/// identifier (`"deadline_elapsed"`, `"wasm_error"`); `message` is
/// the human-readable detail.
#[derive(Debug, Clone)]
struct StreamTerminalError {
    kind: &'static str,
    message: String,
}

/// One frame in the body stream the [`invoke_function_stream`] writer
/// emits. Lives at module scope so the `unfold` closure can name it
/// across `.await` points.
enum StreamFrame {
    Chunk(Vec<u8>),
    Done(Result<(), StreamTerminalError>),
}

/// State machine driving the [`invoke_function_stream`] body
/// `stream::unfold`.
///
/// `Streaming` is the active phase — both the guest's chunk channel
/// AND the executor's terminal-status oneshot are live. `DrainOnly`
/// is the post-completion phase where the guest has returned but
/// chunks may still be buffered in the channel; we drain them before
/// emitting the final `done` frame so the client sees every chunk
/// the guest successfully forwarded. `Done` ends the stream.
enum StreamWriterState {
    Streaming(
        tokio::sync::mpsc::Receiver<Vec<u8>>,
        tokio::sync::oneshot::Receiver<Result<(), StreamTerminalError>>,
    ),
    DrainOnly(
        tokio::sync::mpsc::Receiver<Vec<u8>>,
        Option<Result<(), StreamTerminalError>>,
    ),
    Done,
}

/// `GET /jobs/{id}` — poll an async invocation.
///
/// Tenant-gated (api S-32): the bearer token must address the claimed
/// tenant AND the job's recorded `tenant_id` must equal that tenant. The
/// token-scope check runs first so a caller whose token is outside the
/// claimed tenant's scope sees `403 tenant_scope_denied` regardless of
/// registry state — it cannot probe job-id existence via a 403-vs-404
/// split. The per-resource owner check then rejects a wildcard-scoped
/// caller reading another tenant's job (and its `result` payload) with
/// the same `403 tenant_scope_denied` shape, returning before the
/// `JobRecord` is serialised.
#[tracing::instrument(
    name = "http.get_job",
    skip(state, auth),
    fields(
        job_id = %id,
        tenant = tracing::field::Empty,
    ),
)]
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
) -> ApiResult<Json<JobRecord>> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    // Token-scope check BEFORE the per-resource owner check (api S-32):
    // an out-of-scope token must not be able to distinguish a 403 (job
    // exists, wrong tenant) from a 404 (job unknown).
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }
    match state.jobs.get(&id) {
        Some(rec) => {
            // Per-resource owner check: reject before serialising the
            // record so a cross-tenant caller never observes the job's
            // status / result / function_id.
            if rec.value().tenant_id != tenant {
                return Err(ApiError::forbidden(
                    "tenant_scope_denied",
                    "job belongs to a different tenant",
                ));
            }
            Ok(Json(rec.clone()))
        }
        None => Err(ApiError::not_found(format!("job {id} not found"))),
    }
}

// ---------------------------------------------------------------------------
// Snapshot save / restore (M5)
// ---------------------------------------------------------------------------

/// `POST /snapshot/save` — capture a deployed function into a signed
/// snapshot blob.
///
/// ## Auth posture
///
/// Mounted on the protected stack (bearer auth + tenant scope + rate
/// limit + audit), exactly like the function-mutating routes. The handler
/// runs the same two-stage tenant gate the other resource handlers use:
///
/// 1. `authorize_tenant` against the bearer token's scope — an out-of-scope
///    token sees `403 tenant_scope_denied` before any registry lookup, so
///    it cannot probe function-id existence via a 403-vs-404 split.
/// 2. a per-resource owner check against the looked-up
///    [`FunctionRecord::tenant_id`] — a wildcard-scoped caller from another
///    tenant cannot snapshot tenant A's function.
///
/// ## HMAC wiring (M5)
///
/// The snapshot blob is signed with the operator-configured
/// `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` (threaded in via
/// [`AppState::app_config`]). When no key is configured the route returns
/// `503 snapshot_signing_not_configured`, mirroring the kernel-registry
/// posture — the gateway will not silently emit an unsigned blob under a
/// route that promises a signed one.
///
/// ## What is captured
///
/// The function's deployed Wasm module bytes are captured into the
/// snapshot's `wasm_memory` blob via
/// [`tensor_wasm_snapshot::writer::SnapshotWriter`]. This is the working
/// save/restore-of-bytes layer with end-to-end HMAC signing. Capturing a
/// *live running instance's* linear / GPU memory needs a
/// `TensorWasmExecutor` capture hook that does not exist yet; that
/// capability is intentionally out of scope here (see the module status in
/// [`crate::config`]). The blob produced here round-trips through
/// `/snapshot/restore`.
#[tracing::instrument(
    name = "http.snapshot_save",
    skip(state, auth, payload),
    fields(
        function_id = tracing::field::Empty,
        tenant = tracing::field::Empty,
    ),
)]
pub async fn snapshot_save(
    State(state): State<Arc<AppState>>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    payload: Result<Json<SnapshotSaveRequest>, JsonRejection>,
) -> ApiResult<Json<SnapshotSaveResponse>> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    // Token-scope check first (mirrors create/delete/invoke): an
    // out-of-scope token must be rejected before any registry lookup.
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }

    // The route is only useful with a signing key; refuse early with the
    // same 503-config shape the kernel routes use so a client can detect
    // the feature being disabled without a key leak or a misleading 200.
    let key = match state.app_config.snapshot_hmac_key {
        Some(k) => k,
        None => {
            return Err(ApiError::service_unavailable(
                "snapshot_signing_not_configured",
                "snapshot signing key is not configured; set \
                 TENSOR_WASM_API_SNAPSHOT_HMAC_KEY to enable /snapshot/save",
            ));
        }
    };

    let Json(req) = payload?;
    tracing::Span::current().record("function_id", tracing::field::display(req.function_id));

    // Snapshot the Wasm bytes + owning tenant under the shard lock, then
    // drop the guard before any blocking / await work.
    let (wasm_bytes, owner) = match state.functions.get(&req.function_id) {
        Some(entry) => (Arc::clone(&entry.value().wasm_bytes), entry.value().tenant_id),
        None => {
            return Err(ApiError::not_found(format!(
                "function {} not found",
                req.function_id
            )))
        }
    };
    // Per-resource owner check (api S-IDOR): the token-scope gate above
    // only proves the caller may address the *claimed* tenant, not that
    // the looked-up record belongs to it.
    if owner != tenant {
        return Err(ApiError::forbidden(
            "tenant_scope_denied",
            "function belongs to a different tenant",
        ));
    }

    // Stamp the instance id from the function id's low 64 bits so the
    // restore side can correlate the blob back to a function. The capture
    // itself is CPU-bound (bincode + zstd) over potentially large module
    // bytes, so run it on the blocking pool to keep the reactor free.
    let instance_id = InstanceId(req.function_id.as_u128() & u128::from(u64::MAX));
    let snapshot_bytes = tokio::task::spawn_blocking(move || {
        use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};
        // `with_legacy_envelope` pins the inline v3 (HMAC-SHA256-signed)
        // wire format so the api crate does not depend on the artifact
        // store; the reader on `/snapshot/restore` is configured the same
        // way (`with_hmac_sha256_key`).
        SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .with_legacy_envelope()
            .capture(InstanceState {
                tenant_id: tenant,
                instance_id,
                wasm_memory: &wasm_bytes,
                gpu_memory: &[],
                registers: &[],
            })
    })
    .await
    .map_err(|e| ApiError::internal(format!("snapshot capture task panicked: {e}")))?
    .map_err(|e| {
        // The snapshot crate's `Display` (Serialization variant) is
        // operator-facing and key-free; log server-side and return a
        // stable, content-light message to the caller.
        tracing::error!(
            target: "tensor_wasm_api::routes",
            error = %e,
            "snapshot capture failed",
        );
        ApiError::internal("snapshot capture failed")
    })?;

    let snapshot_b64 = BASE64.encode(&snapshot_bytes);
    Ok(Json(SnapshotSaveResponse {
        function_id: req.function_id,
        snapshot_b64,
        signed: true,
    }))
}

/// `POST /snapshot/restore` — verify and decode a signed snapshot blob.
///
/// ## Auth posture
///
/// Same protected-stack posture as [`snapshot_save`]: bearer auth + tenant
/// scope + rate limit + audit. The handler runs `authorize_tenant` against
/// the caller's token scope, then — after HMAC verification recovers the
/// snapshot's captured tenant — enforces a per-resource owner check: the
/// blob's `metadata.tenant_id` must equal the caller's resolved tenant, so
/// a wildcard-scoped caller from tenant B cannot restore a blob captured
/// for tenant A (`403 tenant_scope_denied`).
///
/// ## HMAC wiring (M5)
///
/// Verification uses the operator-configured
/// `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` via
/// [`SnapshotReader::with_hmac_sha256_key`](tensor_wasm_snapshot::reader::SnapshotReader::with_hmac_sha256_key)
/// and [`SnapshotReader::require_signature`](tensor_wasm_snapshot::reader::SnapshotReader::require_signature):
/// the reader rejects any blob whose HMAC does not verify, and — because
/// the gateway always opts into `require_signature` on this path — refuses
/// unsigned v2 blobs outright. A wrong / missing key or a tampered blob
/// surfaces as `403 snapshot_signature_invalid`. When no key is configured
/// the route returns `503 snapshot_signing_not_configured`.
///
/// The reader's [`with_max_decompressed`](tensor_wasm_snapshot::reader::SnapshotReader::with_max_decompressed)
/// hardening uses the crate default (256 MiB), bounding restore-time memory
/// pressure from a zip-bomb payload.
///
/// ## What is restored
///
/// This handler restores and authenticates the snapshot *envelope* and
/// returns its verified provenance. Reconstituting a *running* instance
/// from the captured memory needs a `TensorWasmExecutor` restore hook that
/// does not exist yet; that capability surfaces `501 not_implemented` when
/// requested. Verifying the blob and recovering its metadata — the part
/// that exercises the HMAC key end-to-end — works today.
#[tracing::instrument(
    name = "http.snapshot_restore",
    skip(state, auth, payload),
    fields(tenant = tracing::field::Empty),
)]
pub async fn snapshot_restore(
    State(state): State<Arc<AppState>>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<crate::rate_limit::AuthContext>>,
    payload: Result<Json<SnapshotRestoreRequest>, JsonRejection>,
) -> ApiResult<Json<SnapshotRestoreResponse>> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }

    let key = match state.app_config.snapshot_hmac_key {
        Some(k) => k,
        None => {
            return Err(ApiError::service_unavailable(
                "snapshot_signing_not_configured",
                "snapshot signing key is not configured; set \
                 TENSOR_WASM_API_SNAPSHOT_HMAC_KEY to enable /snapshot/restore",
            ));
        }
    };

    let Json(req) = payload?;
    // Decode the base64 envelope. Reuse the dedicated invalid_base64
    // mapping so the byte-offset detail of attacker input is logged, not
    // echoed.
    let snapshot_bytes = BASE64
        .decode(req.snapshot_b64.as_bytes())
        .map_err(base64_decode_error)?;

    // Hardening posture (consumes `snapshot_require_signature`):
    //
    // `require_signature()` is applied unconditionally on this path. The
    // gateway only ever *writes* the signed v3 envelope (see
    // [`snapshot_save`]), so accepting an unsigned v2 blob on restore would
    // be a strip-the-trailer downgrade primitive. The operator-facing
    // `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE` knob is the documented
    // surface for this stricter posture; here it is enforced as the
    // hardened default. We still read the configured flag so that an
    // operator who explicitly set it gets a confirming log line (and so a
    // future relaxation of the unconditional require has the value in
    // hand) — the knob is genuinely consumed, not inert.
    if state.app_config.snapshot_require_signature {
        tracing::debug!(
            target: "tensor_wasm_api::routes",
            "snapshot restore: require-signature explicitly enabled by operator",
        );
    }
    // `with_max_decompressed` is left at the crate default
    // (`limits::MAX_DECOMPRESSED_BYTES`, 256 MiB) — the standard zip-bomb
    // ceiling that bounds restore-time memory pressure. Verify HMAC +
    // decode on the blocking pool since zstd + bincode are CPU-bound.
    let restored = tokio::task::spawn_blocking(move || {
        use tensor_wasm_snapshot::reader::SnapshotReader;
        SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .require_signature()
            .restore(&snapshot_bytes)
    })
    .await
    .map_err(|e| ApiError::internal(format!("snapshot restore task panicked: {e}")))?;

    let snapshot = match restored {
        Ok(s) => s,
        Err(e) => {
            // Any verification / structural failure (wrong key, missing
            // signature, tampered bytes, oversized decompress) is surfaced
            // as a single 403 so a caller cannot use the error text as a
            // decode oracle. The detailed snapshot-crate message is logged
            // server-side for operators.
            tracing::warn!(
                target: "tensor_wasm_api::routes",
                error = %e,
                "snapshot restore rejected (signature / structural verification failed)",
            );
            return Err(ApiError::forbidden(
                "snapshot_signature_invalid",
                "snapshot signature verification failed or blob is malformed",
            ));
        }
    };

    // Per-resource owner check: the HMAC-authenticated metadata records the
    // tenant the blob was captured for. A caller must not restore another
    // tenant's blob, even with a valid signature (the key is per-deployment,
    // not per-tenant), so reject a cross-tenant restore with the same
    // `tenant_scope_denied` shape the other handlers use.
    if snapshot.metadata.tenant_id != tenant {
        return Err(ApiError::forbidden(
            "tenant_scope_denied",
            "snapshot was captured for a different tenant",
        ));
    }

    Ok(Json(SnapshotRestoreResponse {
        tenant_id: snapshot.metadata.tenant_id.0,
        instance_id: snapshot.metadata.instance_id.to_string(),
        total_uncompressed_bytes: snapshot.metadata.total_uncompressed_bytes,
        version: snapshot.version,
    }))
}

// ---------------------------------------------------------------------------
// Test-only hooks
// ---------------------------------------------------------------------------

/// Test-only fault-injection probes for [`invoke_function_async`].
///
/// Exposed as `#[doc(hidden)] pub` so the in-tree integration test
/// (`tests/jobs_active_gauge.rs`) can arm the probes. The steady-state cost
/// is one `Relaxed` atomic load per async-invoke dispatch — negligible next
/// to the executor `.spawn_instance` walk that immediately follows — so the
/// probes are compiled into all builds rather than feature-gated. Outside
/// the crate's test surface there is no public API to *set* the flag, so
/// the production code path is unreachable.
#[doc(hidden)]
pub mod test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};

    /// When `true`, the next entry to `maybe_panic_for_test` panics and
    /// resets the flag.
    static PANIC_NEXT_INVOKE: AtomicBool = AtomicBool::new(false);

    /// Arm the panic-injection probe. Returns a drop-guard that disarms it
    /// on scope exit so a failing assertion does not leak the flag into a
    /// neighbouring test.
    pub fn arm_panic() -> ArmedPanic {
        PANIC_NEXT_INVOKE.store(true, Ordering::SeqCst);
        ArmedPanic
    }

    /// RAII disarm. Independent of whether the panic actually fired.
    pub struct ArmedPanic;

    impl Drop for ArmedPanic {
        fn drop(&mut self) {
            PANIC_NEXT_INVOKE.store(false, Ordering::SeqCst);
        }
    }

    /// Probe called from the spawned async-invoke task. Panics (and clears
    /// the flag) when armed; otherwise a noop. Steady-state production
    /// cost: one `Relaxed` atomic load.
    #[inline]
    pub(crate) fn maybe_panic_for_test() {
        // `Relaxed` is the right ordering here: there is no other state
        // we need to synchronise against the flag flip. The
        // compare-exchange upgrades to `SeqCst` only on the cold path
        // when the probe actually fires, which is fine for tests.
        if PANIC_NEXT_INVOKE.load(Ordering::Relaxed)
            && PANIC_NEXT_INVOKE
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            panic!(
                "test_hooks::maybe_panic_for_test: deliberate panic for JobsActiveGuard Drop test"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_constructors_carry_status() {
        let bad = ApiError::bad_request("invalid_name", "x");
        assert_eq!(bad.status, StatusCode::BAD_REQUEST);
        assert_eq!(bad.kind, "invalid_name");

        let nf = ApiError::not_found("missing");
        assert_eq!(nf.status, StatusCode::NOT_FOUND);
        assert_eq!(nf.kind, "not_found");

        let oops = ApiError::internal("boom");
        assert_eq!(oops.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(oops.kind, "internal");
    }

    #[test]
    fn app_state_default_is_empty() {
        let s = AppState::default();
        assert_eq!(s.functions.len(), 0);
        assert_eq!(s.jobs.len(), 0);
        // The default metrics registry exposes every TensorWasm metric name.
        let m = s.metrics.encode_text();
        assert!(m.contains("tensor_wasm_active_instances"), "got: {m}");
    }

    #[test]
    fn app_state_with_metrics_swaps_handle() {
        let custom = Arc::new(TensorWasmMetrics::new());
        custom.kernel_dispatches_total().inc();
        let s = AppState::default().with_metrics(custom.clone());
        // The state's metrics handle reflects the externally observed
        // counter increment — proving the handle was actually swapped in.
        let text = s.metrics.encode_text();
        assert!(
            text.contains("tensor_wasm_kernel_dispatches_total 1"),
            "got: {text}"
        );
    }

    #[test]
    fn app_state_try_new_constructs() {
        let s = AppState::try_new().expect("default engine builds on this host");
        assert_eq!(s.functions.len(), 0);
        assert_eq!(s.jobs.len(), 0);
    }

    #[test]
    fn function_record_skips_wasm_bytes_on_wire() {
        let rec = FunctionRecord {
            id: Uuid::nil(),
            tenant_id: TenantId(0),
            name: "n".to_string(),
            wasm_bytes: Arc::from(vec![1u8, 2, 3]),
            created_unix_ms: 0,
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert!(v.get("wasm_bytes").is_none(), "wasm_bytes leaked: {v}");
    }

    #[tokio::test]
    async fn exec_error_compile_maps_to_wasmtime_500() {
        // Compile failures surface from the executor as `ExecError::Wasmtime`
        // (the executor does not split traps vs. compile errors at the
        // `ExecError` layer — that distinction is only made when converting
        // into `TensorWasmError`). The API layer therefore maps them to a generic
        // 500 `wasmtime` envelope; the real defence against invalid Wasm is
        // `wasmparser::validate` at deploy time, which returns 400 long
        // before this path is reached.
        let engine = std::sync::Arc::new(TensorWasmEngine::new().expect("engine"));
        let exec = TensorWasmExecutor::new(engine);
        let err = exec
            .spawn_instance(SpawnConfig::for_tenant(TenantId(0)), b"not wasm")
            .await
            .expect_err("compile fails");
        assert!(matches!(err, ExecError::Wasmtime(_)));
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api.kind, "wasmtime");
    }

    #[test]
    fn jobs_active_guard_release_decrements_exactly_once() {
        // Happy path: `new` + `release` is a balanced inc/dec, no warn.
        let metrics = Arc::new(TensorWasmMetrics::new());
        assert_eq!(metrics.jobs_active().get(), 0);
        let g = JobsActiveGuard::new(Arc::clone(&metrics));
        assert_eq!(metrics.jobs_active().get(), 1);
        g.release();
        assert_eq!(metrics.jobs_active().get(), 0);
    }

    #[test]
    fn jobs_active_guard_drop_decrements_on_panic_path() {
        // Panic-safety path: a guard dropped without `release` (i.e. the
        // task unwound) still decrements the gauge so the series does not
        // leak a permanent increment. We simulate the unwind by letting
        // the guard fall out of scope without calling `release`.
        let metrics = Arc::new(TensorWasmMetrics::new());
        {
            let _g = JobsActiveGuard::new(Arc::clone(&metrics));
            assert_eq!(metrics.jobs_active().get(), 1);
            // No `release()` here — simulates a panic before the explicit
            // dec point. The `Drop` impl is the safety net.
        }
        assert_eq!(
            metrics.jobs_active().get(),
            0,
            "Drop must decrement the gauge when release was not called"
        );
    }

    #[test]
    fn exec_error_timeout_maps_to_invoke_timeout() {
        use tensor_wasm_core::types::InstanceId;
        use tensor_wasm_exec::executor::TimeoutContext;
        let err = ExecError::Timeout(TimeoutContext {
            id: InstanceId(7),
            elapsed_ms: 1000,
            deadline_ms: 500,
        });
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(api.kind, "invoke_timeout");
        // SECURITY (api T3): message must NOT leak the instance id or
        // the elapsed / deadline figures — those are server-internal.
        assert_eq!(api.message, "invocation deadline exceeded");
        assert!(
            !api.message.contains("I#7"),
            "leaked instance id: {}",
            api.message,
        );
        assert!(
            !api.message.contains("1000") && !api.message.contains("500"),
            "leaked timing figures: {}",
            api.message,
        );
    }

    // -----------------------------------------------------------------
    // SECURITY (api T3): `From<ExecError> for ApiError` previously used
    // `err.to_string()` verbatim for several variants, leaking internal
    // instance IDs, deadlines, quotas, capacity counters, and export
    // names to untrusted callers. The tests below pin the per-variant
    // wire `message` to a fixed, stable string and assert that none of
    // the structured fields appear in the response body.
    // -----------------------------------------------------------------

    #[test]
    fn exec_error_not_found_maps_to_stable_404_message() {
        use tensor_wasm_core::types::InstanceId;
        let err = ExecError::NotFound(InstanceId(42));
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::NOT_FOUND);
        assert_eq!(api.kind, "instance_not_found");
        assert_eq!(api.message, "function not found");
        assert!(
            !api.message.contains("42") && !api.message.contains("I#"),
            "leaked instance id: {}",
            api.message,
        );
    }

    #[test]
    fn exec_error_missing_export_maps_to_stable_400_message() {
        let err = ExecError::MissingExport("super_secret_internal_symbol".to_string());
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::BAD_REQUEST);
        assert_eq!(api.kind, "missing_export");
        assert_eq!(api.message, "requested export not found in module");
        // The export name is attacker-controlled but echoing it back
        // expands the attack surface (XSS-via-error, info-leak about
        // which symbols the module exports). The wire message must
        // be content-free.
        assert!(
            !api.message.contains("super_secret_internal_symbol"),
            "leaked export name: {}",
            api.message,
        );
    }

    #[test]
    fn exec_error_module_memory_too_large_maps_to_stable_413_message() {
        let err = ExecError::ModuleMemoryTooLarge {
            requested_bytes: 4_294_967_296,
            limit_bytes: 67_108_864,
        };
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(api.kind, "module_memory_too_large");
        assert_eq!(api.message, "module declares memory above per-instance cap",);
        assert!(
            !api.message.contains("4294967296") && !api.message.contains("67108864"),
            "leaked byte figures: {}",
            api.message,
        );
    }

    #[test]
    fn exec_error_capacity_exhausted_maps_to_stable_503_message() {
        let err = ExecError::CapacityExhausted {
            active: 257,
            limit: 256,
        };
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(api.kind, "capacity_exhausted");
        assert_eq!(
            api.message,
            "engine instance capacity exhausted; retry later",
        );
        assert!(
            !api.message.contains("257") && !api.message.contains("256"),
            "leaked capacity figures: {}",
            api.message,
        );
    }

    #[test]
    fn exec_error_module_too_large_maps_to_stable_413_message() {
        let err = ExecError::ModuleTooLarge {
            len: 16_777_217,
            max: 16_777_216,
        };
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(api.kind, "module_too_large");
        assert_eq!(api.message, "module bytes above per-tenant cap");
        assert!(
            !api.message.contains("16777217") && !api.message.contains("16777216"),
            "leaked size figures: {}",
            api.message,
        );
    }

    #[test]
    fn exec_error_epoch_ticker_not_running_maps_to_stable_500_message() {
        let err = ExecError::EpochTickerNotRunning;
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api.kind, "epoch_ticker_not_running");
        assert_eq!(api.message, "engine deadline ticker not running");
        // The underlying Display includes a remediation hint
        // ("call `engine.spawn_epoch_ticker()` first"); that hint is
        // operator-only and must not appear on the wire.
        assert!(
            !api.message.contains("spawn_epoch_ticker"),
            "leaked operator hint: {}",
            api.message,
        );
    }
}
