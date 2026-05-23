//! REST route handlers (deploy, invoke, metrics, healthz).
//!
//! All handlers operate on a shared [`AppState`] containing in-memory registries
//! of deployed functions and pending jobs. Wasm bytes are accepted as base64;
//! S17 only validates the magic-number header and stores the bytes — actual
//! instantiation is wired through `bali_exec::executor::BaliExecutor` in S20.
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
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
#[derive(Debug, Clone)]
pub struct AppState {
    /// Deployed-function registry, keyed by id.
    pub functions: Arc<DashMap<Uuid, FunctionRecord>>,
    /// Async-job registry, keyed by id.
    pub jobs: Arc<DashMap<Uuid, JobRecord>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            functions: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
        }
    }
}

impl AppState {
    /// Construct an empty `AppState` wrapped in `Arc` for use with
    /// `Router::with_state`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
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

/// `GET /metrics` — Prometheus text exposition scaffold.
///
/// S17 returns a stub; real exposition wires in S20.
pub async fn metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        "# bali metrics scaffold\n# real exposition wires in S20\n",
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
/// S17 returns a placeholder result; real execution is routed through
/// `bali_exec::executor::BaliExecutor` in S20.
pub async fn invoke_function(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    _args: Result<Json<serde_json::Value>, JsonRejection>,
) -> ApiResult<Json<serde_json::Value>> {
    if !state.functions.contains_key(&id) {
        return Err(ApiError::not_found(format!("function {id} not found")));
    }
    Ok(Json(serde_json::json!({
        "result": "ok",
        "function_id": id.to_string(),
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
