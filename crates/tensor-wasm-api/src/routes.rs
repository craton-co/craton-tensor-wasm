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
    extract::{rejection::JsonRejection, Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::metrics::TensorWasmMetrics;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{TensorWasmExecutor, ExecError, SpawnConfig};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Default per-invocation deadline used by the synchronous `/invoke` handler.
const INVOKE_DEFAULT_DEADLINE: Duration = Duration::from_secs(30);

/// Minimum length, in bytes, of a Wasm module: the 4-byte `\0asm` magic plus
/// the 4-byte version field. Anything shorter cannot be a valid module.
pub const WASM_MIN_HEADER_BYTES: usize = 8;

/// Body-size threshold above which base64 decoding is moved to
/// [`tokio::task::spawn_blocking`]. Below this the inline decode is cheaper
/// than the spawn handoff.
pub const BASE64_OFFLOAD_THRESHOLD: usize = 256 * 1024;

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
    fn try_build() -> Result<Self, TensorWasmError> {
        let metrics = Arc::new(TensorWasmMetrics::new());
        let engine = Arc::new(
            TensorWasmEngine::new().map_err(|e| TensorWasmError::WasmCompile(format!("{e:#}").into()))?,
        );
        let executor = Arc::new(TensorWasmExecutor::with_metrics(engine, (*metrics).clone()));
        Ok(Self {
            functions: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            metrics,
            executor,
        })
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
            TensorWasmEngine::new().expect("TensorWasmEngine::new must succeed for AppState::with_metrics"),
        );
        self.executor = Arc::new(TensorWasmExecutor::with_metrics(engine, (*metrics).clone()));
        self.metrics = metrics;
        self
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

    /// Construct a `500 Internal Server Error` with `kind = "internal"`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal".to_string(),
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
        // SECURITY (api S-22): pre-W4.x this impl used `err.to_string()`
        // verbatim as the wire `message`. For `ExecError::Wasmtime(_)` that
        // surfaces the full wasmtime error chain (including host pointer
        // addresses, host file paths, and internal stack-frame names) to
        // untrusted callers. We now branch per-variant so structured
        // variants (NotFound, MissingExport, Timeout) keep their safe,
        // already-stable messages while the wasmtime variant is replaced
        // with an opaque string and the full error is logged server-side.
        match &err {
            ExecError::NotFound(_) => ApiError {
                status: StatusCode::NOT_FOUND,
                kind: "instance_not_found".to_string(),
                message: err.to_string(),
            },
            ExecError::MissingExport(_) => ApiError {
                status: StatusCode::BAD_REQUEST,
                kind: "missing_export".to_string(),
                message: err.to_string(),
            },
            ExecError::Timeout(_) => ApiError {
                status: StatusCode::GATEWAY_TIMEOUT,
                kind: "invoke_timeout".to_string(),
                message: err.to_string(),
            },
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
            ExecError::ModuleMemoryTooLarge { .. } => ApiError {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                kind: "module_memory_too_large".to_string(),
                message: err.to_string(),
            },
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

/// Decode `wasm_b64`, offloading to a blocking pool when the encoded payload
/// is large enough that the spawn handoff is cheaper than blocking the I/O
/// thread. The threshold is [`BASE64_OFFLOAD_THRESHOLD`] *encoded* bytes.
async fn decode_wasm_b64(wasm_b64: String) -> Result<Vec<u8>, ApiError> {
    if wasm_b64.len() < BASE64_OFFLOAD_THRESHOLD {
        return BASE64
            .decode(wasm_b64.as_bytes())
            .map_err(|e| ApiError::bad_request("invalid_base64", e.to_string()));
    }
    tokio::task::spawn_blocking(move || BASE64.decode(wasm_b64.as_bytes()))
        .await
        .map_err(|e| ApiError::internal(format!("base64 decoder panicked: {e}")))?
        .map_err(|e| ApiError::bad_request("invalid_base64", e.to_string()))
}

/// `POST /functions` — deploy a new Wasm module.
///
/// Decodes the supplied base64, runs full `wasmparser` validation, and stores
/// the bytes refcounted via `Arc<[u8]>` so concurrent invocations do not
/// reallocate. Returns `400 invalid_wasm` if validation fails.
#[tracing::instrument(
    name = "http.create_function",
    skip(state, payload),
    fields(function_id = tracing::field::Empty),
)]
pub async fn create_function(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<CreateFunctionRequest>, JsonRejection>,
) -> ApiResult<Json<CreateFunctionResponse>> {
    let Json(req) = payload?;
    if req.name.trim().is_empty() {
        return Err(ApiError::bad_request(
            "invalid_name",
            "name must be non-empty",
        ));
    }
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
        return Err(ApiError::bad_request(
            "invalid_wasm",
            format!("wasm validation failed: {e}"),
        ));
    }
    let id = Uuid::new_v4();
    tracing::Span::current().record("function_id", tracing::field::display(id));
    state.functions.insert(
        id,
        FunctionRecord {
            id,
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
#[tracing::instrument(name = "http.delete_function", skip(state), fields(function_id = %id))]
pub async fn delete_function(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<StatusCode> {
    if state.functions.remove(&id).is_some() {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format!("function {id} not found")))
    }
}

/// Drive the spawn → call(`_start`|`main`) → terminate flow against the
/// supplied bytes/tenant. Shared by the synchronous and async invoke paths
/// so any future fix (telemetry, retries, etc.) lands in one place.
#[tracing::instrument(
    name = "invoke.run",
    skip(executor, wasm_bytes),
    fields(
        tenant = %tenant,
        function_id = %function_id,
        wasm_bytes_len = wasm_bytes.len(),
    ),
)]
async fn run_invoke(
    executor: &TensorWasmExecutor,
    wasm_bytes: &[u8],
    tenant: TenantId,
    function_id: Uuid,
) -> ApiResult<serde_json::Value> {
    let cfg = SpawnConfig::for_tenant(tenant).with_deadline(INVOKE_DEFAULT_DEADLINE);
    let instance_id = executor.spawn_instance(cfg, wasm_bytes).await?;

    // Try `_start` (WASI command convention) first, then `main`. Anything
    // other than `MissingExport` from the first attempt bubbles up directly;
    // a missing `_start` falls through to `main`. If neither exists the
    // `MissingExport` from the second attempt is returned (mapped to 400).
    let call_result = match executor.call_export(instance_id, "_start").await {
        Ok(()) => Ok(()),
        Err(ExecError::MissingExport(_)) => executor.call_export(instance_id, "main").await,
        Err(other) => Err(other),
    };

    // Always terminate, even on call failure. Surface the call error in
    // preference to the terminate error — the call result is what the
    // operator wants to see.
    let terminate_result = executor.terminate(instance_id).await;

    call_result?;
    // If the call succeeded but terminate failed we still surface the
    // terminate error: leaking instances is a real bug we want loud.
    terminate_result?;

    Ok(serde_json::json!({
        "function_id": function_id.to_string(),
        "result": "ok",
    }))
}

/// `POST /functions/{id}/invoke` — synchronous invocation.
///
/// Looks up the function's stored Wasm bytes, spawns a fresh instance via
/// the shared [`TensorWasmExecutor`] with a 30-second deadline, calls `_start`
/// (falling back to `main`), then terminates the instance. Returns the
/// `function_id` and a string `result` on success; structured `ApiError`
/// otherwise.
///
/// The tenant id is sourced from the `X-TensorWasm-Tenant` middleware
/// extension; absent it defaults to `TenantId(0)`.
#[tracing::instrument(
    name = "http.invoke_function",
    skip(state, auth, _args),
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
    _args: Result<Json<serde_json::Value>, JsonRejection>,
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

    // Snapshot the Wasm bytes under the DashMap shard lock, then drop the
    // guard before we hit any `.await`. `Arc::clone` is a single refcount
    // bump regardless of payload size.
    let wasm_bytes = match state.functions.get(&id) {
        Some(entry) => Arc::clone(&entry.value().wasm_bytes),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };

    let value = run_invoke(&state.executor, &wasm_bytes, tenant, id).await?;
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
#[tracing::instrument(
    name = "http.invoke_function_async",
    skip(state, auth, _args),
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
    _args: Result<Json<serde_json::Value>, JsonRejection>,
) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    tracing::Span::current().record("tenant", tracing::field::display(tenant));
    if let Some(Extension(ctx)) = auth.as_ref() {
        ctx.authorize_tenant(tenant)?;
    }
    let wasm_bytes = match state.functions.get(&id) {
        Some(entry) => Arc::clone(&entry.value().wasm_bytes),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };

    let job_id = Uuid::new_v4();
    tracing::Span::current().record("job_id", tracing::field::display(job_id));
    state.jobs.insert(
        job_id,
        JobRecord {
            id: job_id,
            function_id: id,
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
    tokio::spawn(
        tracing::Instrument::instrument(
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

                let outcome = run_invoke(&executor, &wasm_bytes, tenant, id).await;
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
        ),
    );

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id.to_string() })),
    ))
}

/// `GET /jobs/{id}` — poll an async invocation.
#[tracing::instrument(name = "http.get_job", skip(state), fields(job_id = %id))]
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<JobRecord>> {
    match state.jobs.get(&id) {
        Some(rec) => Ok(Json(rec.clone())),
        None => Err(ApiError::not_found(format!("job {id} not found"))),
    }
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
            panic!("test_hooks::maybe_panic_for_test: deliberate panic for JobsActiveGuard Drop test");
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
        use tensor_wasm_exec::executor::TimeoutContext;
        use tensor_wasm_core::types::InstanceId;
        let err = ExecError::Timeout(TimeoutContext {
            id: InstanceId(7),
            elapsed_ms: 1000,
            deadline_ms: 500,
        });
        let api: ApiError = err.into();
        assert_eq!(api.status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(api.kind, "invoke_timeout");
    }
}
