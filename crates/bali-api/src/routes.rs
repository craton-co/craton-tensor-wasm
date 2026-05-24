//! REST route handlers (deploy, invoke, metrics, healthz).
//!
//! All handlers operate on a shared [`AppState`] containing in-memory registries
//! of deployed functions and pending jobs, plus a shared [`BaliMetrics`]
//! registry and a [`BaliExecutor`]. Wasm bytes are accepted as base64; the
//! deploy path validates the magic-number header and stores the bytes, and
//! the synchronous invoke path drives `bali_exec::executor::BaliExecutor` to
//! spawn, call `_start` / `main`, and terminate the instance.
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
    extract::{rejection::JsonRejection, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bali_core::metrics::BaliMetrics;
use bali_core::types::TenantId;
use bali_exec::engine::BaliEngine;
use bali_exec::executor::{BaliExecutor, ExecError, SpawnConfig};
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

/// The 4-byte `\0asm` magic that prefixes every Wasm module.
const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d];

// ---------------------------------------------------------------------------
// State records
// ---------------------------------------------------------------------------

/// A deployed function as held in memory by the API gateway.
///
/// `wasm_bytes` is intentionally excluded from the serialised wire form: the
/// API never echoes raw module bytes back to callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRecord {
    /// Server-assigned identifier.
    pub id: Uuid,
    /// Tenant-supplied display name.
    pub name: String,
    /// Decoded Wasm bytes. Not serialised — see struct-level doc.
    #[serde(skip)]
    pub wasm_bytes: Vec<u8>,
    /// Millisecond-precision Unix timestamp of deploy.
    pub created_unix_ms: u64,
}

/// Status of an asynchronously dispatched invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is queued or in flight.
    Pending,
    /// Job completed successfully; `result` holds the value.
    Completed,
    /// Job failed; `result` holds the failure message.
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
    pub metrics: Arc<BaliMetrics>,
    /// Wasm executor driving the synchronous `/invoke` path.
    pub executor: Arc<BaliExecutor>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("functions", &self.functions.len())
            .field("jobs", &self.jobs.len())
            .finish()
    }
}

impl Default for AppState {
    fn default() -> Self {
        let metrics = Arc::new(BaliMetrics::new());
        let engine =
            Arc::new(BaliEngine::new().expect("BaliEngine::new must succeed for default AppState"));
        let executor = Arc::new(BaliExecutor::with_metrics(engine, (*metrics).clone()));
        Self {
            functions: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            metrics,
            executor,
        }
    }
}

impl AppState {
    /// Construct an empty `AppState` wrapped in `Arc` for use with
    /// `Router::with_state`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Override the metrics registry, rebuilding the executor so its
    /// spawn/terminate counters share the supplied handle.
    ///
    /// Useful for tests that need to inspect counter values, or for embedders
    /// who construct a process-wide registry separately.
    pub fn with_metrics(mut self, metrics: Arc<BaliMetrics>) -> Self {
        let engine =
            Arc::new(BaliEngine::new().expect("BaliEngine::new must succeed for with_metrics"));
        self.executor = Arc::new(BaliExecutor::with_metrics(engine, (*metrics).clone()));
        self.metrics = metrics;
        self
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ApiErrorEnvelope {
            error: ApiErrorBody {
                kind: self.kind,
                message: self.message,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rej: JsonRejection) -> Self {
        ApiError::bad_request("invalid_json", rej.body_text())
    }
}

impl From<ExecError> for ApiError {
    fn from(err: ExecError) -> Self {
        match err {
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
            ExecError::Wasmtime(_) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                kind: "wasmtime".to_string(),
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
/// Renders the shared [`BaliMetrics`] registry into the standard text-format
/// (`text/plain; version=0.0.4`). Every metric registered in
/// `bali_core::metrics` is exposed; counter names carry the `_total` suffix
/// per Prometheus convention.
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.encode_text();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
}

/// `POST /functions` — deploy a new Wasm module.
///
/// Validates that the supplied base64 decodes to at least
/// [`WASM_MIN_HEADER_BYTES`] bytes whose first four bytes are the Wasm magic
/// `\0asm` header. Real instantiation (and the corresponding registry
/// integration with `bali_exec::executor::BaliExecutor`) lands in S20.
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
    let bytes = BASE64
        .decode(req.wasm_b64.as_bytes())
        .map_err(|e| ApiError::bad_request("invalid_base64", e.to_string()))?;
    if bytes.len() < WASM_MIN_HEADER_BYTES {
        return Err(ApiError::bad_request(
            "too_short",
            format!(
                "module too short: {} bytes (minimum {WASM_MIN_HEADER_BYTES})",
                bytes.len()
            ),
        ));
    }
    if bytes[0..4] != WASM_MAGIC {
        return Err(ApiError::bad_request(
            "not_wasm",
            "first four bytes are not the Wasm magic \\0asm",
        ));
    }
    let id = Uuid::new_v4();
    state.functions.insert(
        id,
        FunctionRecord {
            id,
            name: req.name,
            wasm_bytes: bytes,
            created_unix_ms: now_unix_ms(),
        },
    );
    Ok(Json(CreateFunctionResponse { id }))
}

/// `DELETE /functions/{id}` — remove a deployed function.
///
/// Returns `204 No Content` on success and `404 Not Found` if the id is
/// unknown.
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

/// `POST /functions/{id}/invoke` — synchronous invocation.
///
/// Looks up the function's stored Wasm bytes, spawns a fresh instance via
/// the shared [`BaliExecutor`] with a 30-second deadline, calls `_start`
/// (falling back to `main`), then terminates the instance. Returns the
/// `function_id` and a string `result` on success; structured `ApiError`
/// otherwise.
pub async fn invoke_function(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    _args: Result<Json<serde_json::Value>, JsonRejection>,
) -> ApiResult<Json<serde_json::Value>> {
    // Snapshot the Wasm bytes under the DashMap shard lock, then drop the
    // guard before we hit any `.await`. The function record's stored bytes
    // are immutable post-deploy so a one-shot clone is correct.
    let wasm_bytes = match state.functions.get(&id) {
        Some(entry) => entry.value().wasm_bytes.clone(),
        None => return Err(ApiError::not_found(format!("function {id} not found"))),
    };

    let cfg = SpawnConfig::for_tenant(TenantId(0)).with_deadline(INVOKE_DEFAULT_DEADLINE);
    let instance_id = state.executor.spawn_instance(cfg, &wasm_bytes).await?;

    // Try `_start` (WASI command convention) first, then `main`. Anything
    // other than `MissingExport` from the first attempt bubbles up directly;
    // a missing `_start` falls through to `main`. If neither exists the
    // `MissingExport` from the second attempt is returned (mapped to 400).
    let call_result = match state.executor.call_export(instance_id, "_start").await {
        Ok(()) => Ok(()),
        Err(ExecError::MissingExport(_)) => state.executor.call_export(instance_id, "main").await,
        Err(other) => Err(other),
    };

    // Always terminate, even on call failure. Swallow a terminate error in
    // favour of surfacing the original call error — the call result is what
    // the operator wants to see.
    let terminate_result = state.executor.terminate(instance_id).await;

    call_result?;
    // If the call succeeded but terminate failed we still surface the
    // terminate error: leaking instances is a real bug we want loud.
    terminate_result?;

    Ok(Json(serde_json::json!({
        "function_id": id.to_string(),
        "result": "ok",
    })))
}

/// `POST /functions/{id}/invoke-async` — fire-and-forget invocation.
///
/// Inserts a Pending [`JobRecord`] and returns its id. Real dispatch wires in
/// S20.
pub async fn invoke_function_async(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    _args: Result<Json<serde_json::Value>, JsonRejection>,
) -> ApiResult<Json<serde_json::Value>> {
    if !state.functions.contains_key(&id) {
        return Err(ApiError::not_found(format!("function {id} not found")));
    }
    let job_id = Uuid::new_v4();
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
    Ok(Json(serde_json::json!({ "job_id": job_id.to_string() })))
}

/// `GET /jobs/{id}` — poll an async invocation.
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
        // The default metrics registry exposes every Bali metric name.
        let m = s.metrics.encode_text();
        assert!(m.contains("bali_active_instances"), "got: {m}");
    }

    #[test]
    fn app_state_with_metrics_swaps_handle() {
        let custom = Arc::new(BaliMetrics::new());
        custom.kernel_dispatches_total().inc();
        let s = AppState::default().with_metrics(custom.clone());
        // The state's metrics handle reflects the externally observed
        // counter increment — proving the handle was actually swapped in.
        let text = s.metrics.encode_text();
        assert!(
            text.contains("bali_kernel_dispatches_total 1"),
            "got: {text}"
        );
    }

    #[test]
    fn function_record_skips_wasm_bytes_on_wire() {
        let rec = FunctionRecord {
            id: Uuid::nil(),
            name: "n".to_string(),
            wasm_bytes: vec![1, 2, 3],
            created_unix_ms: 0,
        };
        let v = serde_json::to_value(&rec).unwrap();
        assert!(v.get("wasm_bytes").is_none(), "wasm_bytes leaked: {v}");
    }
}
