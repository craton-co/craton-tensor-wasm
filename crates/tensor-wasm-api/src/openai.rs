// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAI-compatible inference gateway shim.
//!
//! v0.3.5 landed the *scaffold* (request types, error envelope, two
//! handlers returning `501 openai_not_yet_wired`); **T41 (v0.4)** wires
//! the handlers through to the internal invoke protocol. The handlers
//! accept the standard OpenAI request bodies (so off-the-shelf SDKs
//! send valid JSON) and now resolve the `model` field against an
//! env-configured `model → function_uuid` map, dispatch the call via
//! the shared [`tensor_wasm_exec::executor::TensorWasmExecutor`], and
//! wrap the guest output in the OpenAI response envelope (or stream it
//! as OpenAI-shape `data:` SSE frames when `stream: true`).
//!
//! Legacy `501 Not Implemented` error envelope shape (still emitted by
//! the deprecated [`OpenAiError::not_yet_wired`] helper for routes that
//! deliberately remain stubs):
//!
//! ```json
//! { "error": {
//!     "message": "...",
//!     "type":    "not_implemented",
//!     "param":   null,
//!     "code":    "openai_not_yet_wired"
//! }}
//! ```
//!
//! The v0.4 follow-up wires the actual translation step (resolve the
//! requested `model` to a deployed `FunctionRecord`, marshal the prompt
//! / messages into the wasm guest's `_start` argv, stream tokens out as
//! OpenAI-shape `data:` SSE chunks). The scaffold exists so:
//!
//! * the route shape (path, method, request body) is locked in early —
//!   clients can begin integrating against the gateway's URL surface
//!   without depending on the translator's readiness;
//! * the error envelope shape (the four-field OpenAI object, distinct
//!   from the gateway's native `{ error: { kind, message } }` shell) is
//!   committed to the public contract and exercised by integration
//!   tests;
//! * the OpenAPI spec at `openapi/tensor-wasm-api.yaml` documents the
//!   surface up front, so downstream API-doc tooling renders both
//!   surfaces from a single source.
//!
//! ## Security: tenant resolution
//!
//! OpenAI clients send `Authorization: Bearer <api_key>` but never an
//! `X-TensorWasm-Tenant` header. The gateway's native routes derive the
//! tenant from that header (via the `tenant_scope` middleware); the
//! OpenAI routes cannot, because the header is absent on the wire.
//!
//! The v0.4 implementation will derive the tenant from the bearer
//! token's [`TokenScope`](crate::token_scope::TokenScope): a scoped
//! token (`mykey:tenant=7`) implies tenant 7; a wildcard token implies
//! the default tenant (0) with a one-shot warning. Clients should
//! provision one bearer token per tenant in
//! `$TENSOR_WASM_API_TOKENS`. This is why the OpenAI routes are mounted
//! *outside* the `tenant_scope` middleware in `server.rs` — the layer
//! would reject every OpenAI request as `missing_tenant` 400 otherwise.
//!
//! Bearer auth itself still runs on these routes (mounted inside the
//! `bearer_auth` middleware): an unauthenticated OpenAI client must
//! receive `401`, not `501`. The current scaffold leaves the auth /
//! rate-limit / audit composition for the server module to wire; this
//! file owns only the request type definitions, the error envelope, and
//! the two handlers.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{extract::rejection::JsonRejection, Json};
use serde::{Deserialize, Serialize};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::executor::{ExecError, SpawnConfig};
use tensor_wasm_wasi_gpu::streaming::StreamingContext;
use uuid::Uuid;

use crate::openai_translator::{
    translate_chat_completions_request, translate_completions_request, TranslatedRequest,
};
use crate::rate_limit::AuthContext;
use crate::routes::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Body of `POST /v1/completions`.
///
/// Mirrors the public OpenAI REST contract documented at
/// <https://platform.openai.com/docs/api-reference/completions/create>.
///
/// Every field is `#[serde(default)]` so an SDK that omits an optional
/// knob still deserialises cleanly. We do not (yet) validate the values
/// — the scaffold's only contract is "the request parses; we then
/// reject with 501". The v0.4 wiring step adds:
///
/// * `model` → `FunctionRecord` lookup;
/// * `prompt` → guest argv marshalling;
/// * `max_tokens` / `temperature` / `stream` → executor knobs.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[non_exhaustive]
pub struct CompletionsRequest {
    /// Model identifier. In v0.4 this will resolve to a deployed
    /// `FunctionRecord` via a `model` → `function_id` map (env-driven
    /// allowlist + alias table).
    #[serde(default)]
    pub model: String,
    /// Prompt text. Accepts either a single string or an array of
    /// strings on the wire — represented here as `serde_json::Value`
    /// so the scaffold does not commit to one shape ahead of the
    /// translator.
    #[serde(default)]
    pub prompt: serde_json::Value,
    /// Maximum tokens to generate. Optional in the OpenAI contract;
    /// defaults to 16 if absent on the wire (v0.4 will mirror).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature in `[0.0, 2.0]`. Optional; defaults to 1.0.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Stream the response as SSE if true. v0.4 wires; scaffold ignores.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Echo of the input plus completion (OpenAI compat knob).
    #[serde(default)]
    pub echo: Option<bool>,
    /// Number of completions to generate per prompt.
    #[serde(default)]
    pub n: Option<u32>,
    /// Caller-supplied request id, surfaced back on the response in
    /// OpenAI's own logs. We accept and ignore.
    #[serde(default)]
    pub user: Option<String>,
}

/// One entry in the `messages` array of `POST /v1/chat/completions`.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[non_exhaustive]
pub struct ChatMessage {
    /// One of `system`, `user`, `assistant`, `tool` (OpenAI shape).
    /// Free-form on the wire; the scaffold does not validate.
    #[serde(default)]
    pub role: String,
    /// Message content. May be a string or a content-array on the wire
    /// (OpenAI supports multimodal messages); we accept either shape
    /// via `serde_json::Value`.
    #[serde(default)]
    pub content: serde_json::Value,
    /// Optional speaker name for the `system` / `user` roles.
    #[serde(default)]
    pub name: Option<String>,
}

/// Body of `POST /v1/chat/completions`.
///
/// Mirrors the public OpenAI REST contract documented at
/// <https://platform.openai.com/docs/api-reference/chat/create>.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[non_exhaustive]
pub struct ChatCompletionsRequest {
    /// Model identifier. See [`CompletionsRequest::model`].
    #[serde(default)]
    pub model: String,
    /// Conversation history. Required by OpenAI; the scaffold rejects
    /// at the 501 step so an empty vector still parses.
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate per response.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature in `[0.0, 2.0]`.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Stream the response as SSE if true.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions to generate per prompt.
    #[serde(default)]
    pub n: Option<u32>,
    /// Optional `tools` array (OpenAI tool-calling). v0.4 wires.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Caller-supplied opaque user identifier.
    #[serde(default)]
    pub user: Option<String>,
}

// ---------------------------------------------------------------------------
// Error envelope (OpenAI-shape, distinct from the native ApiErrorEnvelope)
// ---------------------------------------------------------------------------

/// Inner OpenAI error body. The shape is:
///
/// ```json
/// { "message": "...", "type": "...", "param": null, "code": "..." }
/// ```
///
/// This intentionally does **not** match the gateway's native
/// `{ error: { kind, message } }` envelope: OpenAI SDKs parse the
/// four-field shape verbatim and will not look at our native shell.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiErrorBody {
    /// Human-readable error description.
    pub message: String,
    /// OpenAI-conventional error category (e.g. `invalid_request_error`,
    /// `not_implemented`). String, not enum, because the OpenAI contract
    /// itself adds new types over time.
    #[serde(rename = "type")]
    pub kind: String,
    /// Name of the request field that triggered the error, if any.
    /// `null` for whole-request errors (the scaffold's 501).
    pub param: Option<String>,
    /// Stable machine-readable code that callers branch on. Scaffold
    /// returns `openai_not_yet_wired`.
    pub code: Option<String>,
}

/// Top-level OpenAI error envelope: `{ "error": { ... } }`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiError {
    /// Inner error body.
    pub error: OpenAiErrorBody,
}

impl OpenAiError {
    /// Construct a not-implemented envelope. Used by both scaffold
    /// handlers to keep the wire output identical.
    pub fn not_yet_wired(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                kind: "not_implemented".to_string(),
                param: None,
                code: Some("openai_not_yet_wired".to_string()),
            },
        }
    }

    /// Construct an `invalid_request_error` envelope for malformed input.
    /// `param` should name the field that triggered the error, if known.
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                kind: "invalid_request_error".to_string(),
                param,
                code: Some("openai_invalid_request".to_string()),
            },
        }
    }

    /// Override the machine-readable `code` on an existing envelope.
    ///
    /// Used by the translator to stamp a more specific code
    /// (e.g. `unsupported_parameter`) onto an `invalid_request` envelope
    /// while keeping its `type` / `param` / status mapping intact — an
    /// `invalid_request_error` with any code other than `model_not_found`
    /// still maps to `400` in [`OpenAiError::into_response`].
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.error.code = Some(code.to_string());
        self
    }

    /// Construct a `model_not_found` envelope for an unknown model id.
    ///
    /// Mirrors the OpenAI `404 model_not_found` response shape so SDKs
    /// surface a clean "this model is not available on this endpoint"
    /// error to the caller. `param` is fixed to `Some("model")` —
    /// every `model_not_found` is caused by the same field.
    pub fn model_not_found(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                kind: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
        }
    }

    /// Construct a generic server-side `internal` envelope.
    ///
    /// Used when guest execution fails in a way that doesn't map to a
    /// caller-correctable category (e.g. a wasm trap, an executor
    /// configuration error). The OpenAI `type` is set to
    /// `server_error` and `code` to `wasm_error` so callers can branch
    /// on either the OpenAI-conventional category or the
    /// implementation-specific code.
    pub fn server_error(message: impl Into<String>, code: &'static str) -> Self {
        Self {
            error: OpenAiErrorBody {
                message: message.into(),
                kind: "server_error".to_string(),
                param: None,
                code: Some(code.to_string()),
            },
        }
    }
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        // Status is chosen from the (kind, code) pair. The OpenAI public
        // contract reuses `invalid_request_error` for both "your request
        // body was malformed" (400) and "the requested model does not
        // exist" (404); SDKs branch on the `code` to disambiguate. We
        // mirror that wire shape here.
        let status = match (self.error.kind.as_str(), self.error.code.as_deref()) {
            ("invalid_request_error", Some("model_not_found")) => StatusCode::NOT_FOUND,
            ("invalid_request_error", _) => StatusCode::BAD_REQUEST,
            ("server_error", _) => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::NOT_IMPLEMENTED,
        };
        (status, Json(self)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Wired handlers (T41 v0.4)
// ---------------------------------------------------------------------------

/// Per-invocation deadline applied to every OpenAI-routed call. Matches
/// the native `/invoke` default so the two surfaces share a budget.
const OPENAI_INVOKE_DEADLINE: Duration = Duration::from_secs(30);

/// Channel buffer for streaming chunks (mirrors the
/// `STREAMING_CHANNEL_BUFFER` constant on `/invoke-stream`). Enough to
/// absorb a short burst from the guest while the SSE writer drains its
/// TCP send buffer; small enough to apply natural back-pressure once
/// the writer stalls.
const OPENAI_STREAM_BUFFER: usize = 32;

/// Generate a millisecond-precision Unix `created` timestamp for the
/// response envelope. OpenAI's contract emits seconds; we emit seconds
/// here too so SDKs that compute `time.time() - created` work without
/// scaling.
fn unix_seconds_now() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs(),
        Err(_) => 0,
    }
}

/// Build the OpenAI `usage` block. v0.4 ships with zeros because the
/// gateway does not yet wire a tokenizer; v0.5 lands a real counter
/// (see `docs/OPENAI-COMPAT.md`).
fn empty_usage() -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
}

/// `POST /v1/completions` — OpenAI completions shim, wired through to
/// the internal invoke protocol (T41, v0.4).
///
/// Flow:
///
/// 1. Apply the tenant-scope gate (T2) on the bearer token's
///    [`crate::token_scope::TokenScope`] against the resolved tenant
///    (`TenantId(0)` for OpenAI SDKs that don't send
///    `X-TensorWasm-Tenant`).
/// 2. Parse the request body. Malformed JSON → `400
///    openai_invalid_request`.
/// 3. Resolve `req.model` against
///    [`AppState::openai_model_map`](crate::routes::AppState::openai_model_map).
///    Miss → `404 model_not_found`.
/// 4. Look up the resolved function id in
///    [`AppState::functions`](crate::routes::AppState::functions). Miss
///    → `404 model_not_found` (the map points at a deleted record).
/// 5. If `stream: true`, build a
///    [`StreamingContext`](tensor_wasm_wasi_gpu::streaming::StreamingContext)
///    + receiver pair (the T34 plumbing), spawn the call, and stream
///    chunks back as OpenAI `data: { ... }` SSE frames terminated by
///    `data: [DONE]\n\n`.
/// 6. If `stream: false`, run the call synchronously, collect any
///    emitted chunks (or fall back to a host-stamped placeholder), and
///    respond with the OpenAI `text_completion` envelope.
#[tracing::instrument(
    name = "http.openai.completions",
    skip(state, payload, auth),
    fields(model = tracing::field::Empty, stream = tracing::field::Empty),
)]
pub async fn completions_handler(
    State(state): State<Arc<AppState>>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<AuthContext>>,
    payload: Result<Json<CompletionsRequest>, JsonRejection>,
) -> Response {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    if let Some(Extension(ctx)) = auth.as_ref() {
        if let Err(err) = ctx.authorize_tenant(tenant) {
            return err.into_response();
        }
    }

    let req = match payload {
        Ok(Json(r)) => r,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenAiError::invalid_request(rej.body_text(), None)),
            )
                .into_response();
        }
    };
    tracing::Span::current().record("model", tracing::field::display(&req.model));
    let model_echo = req.model.clone();

    let translated = match translate_completions_request(&req, state.openai_model_map.as_ref()) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    tracing::Span::current().record("stream", tracing::field::display(translated.stream));

    run_translated(
        state,
        tenant,
        translated,
        model_echo,
        OpenAiObject::TextCompletion,
    )
    .await
}

/// `POST /v1/chat/completions` — OpenAI chat-completions shim, wired
/// through to the internal invoke protocol (T41, v0.4).
///
/// Same flow as [`completions_handler`] but with the chat envelope
/// (response `object: "chat.completion"`, choices carry a
/// `{role, content}` message object) and the `messages` array
/// concatenated into a single prompt string via
/// [`crate::openai_translator::assemble_chat_prompt`].
#[tracing::instrument(
    name = "http.openai.chat_completions",
    skip(state, payload, auth),
    fields(model = tracing::field::Empty, stream = tracing::field::Empty),
)]
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    tenant: Option<Extension<TenantId>>,
    auth: Option<Extension<AuthContext>>,
    payload: Result<Json<ChatCompletionsRequest>, JsonRejection>,
) -> Response {
    let tenant = tenant.map(|Extension(t)| t).unwrap_or(TenantId(0));
    if let Some(Extension(ctx)) = auth.as_ref() {
        if let Err(err) = ctx.authorize_tenant(tenant) {
            return err.into_response();
        }
    }

    let req = match payload {
        Ok(Json(r)) => r,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenAiError::invalid_request(rej.body_text(), None)),
            )
                .into_response();
        }
    };
    tracing::Span::current().record("model", tracing::field::display(&req.model));
    let model_echo = req.model.clone();

    let translated = match translate_chat_completions_request(&req, state.openai_model_map.as_ref())
    {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    tracing::Span::current().record("stream", tracing::field::display(translated.stream));

    run_translated(
        state,
        tenant,
        translated,
        model_echo,
        OpenAiObject::ChatCompletion,
    )
    .await
}

/// Which OpenAI response envelope to wrap the guest output in.
///
/// `TextCompletion` is the `/v1/completions` shape:
/// `choices[i].text` carries the generated string. `ChatCompletion` is
/// `/v1/chat/completions`: `choices[i].message = {"role":"assistant","content":...}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenAiObject {
    /// `object: "text_completion"`.
    TextCompletion,
    /// `object: "chat.completion"`.
    ChatCompletion,
}

impl OpenAiObject {
    fn object_kind(self) -> &'static str {
        match self {
            OpenAiObject::TextCompletion => "text_completion",
            OpenAiObject::ChatCompletion => "chat.completion",
        }
    }

    fn chunk_object_kind(self) -> &'static str {
        match self {
            OpenAiObject::TextCompletion => "text_completion",
            OpenAiObject::ChatCompletion => "chat.completion.chunk",
        }
    }

    fn id_prefix(self) -> &'static str {
        match self {
            OpenAiObject::TextCompletion => "cmpl-",
            OpenAiObject::ChatCompletion => "chatcmpl-",
        }
    }
}

/// Drive a translated request through the executor. Branches on
/// `translated.stream` to either return a buffered JSON envelope
/// (non-streaming) or a `text/event-stream` response (streaming).
async fn run_translated(
    state: Arc<AppState>,
    tenant: TenantId,
    translated: TranslatedRequest,
    model_echo: String,
    object_kind: OpenAiObject,
) -> Response {
    // Look up the resolved function id in the deployment registry.
    // Missing record = the map points at something that has been
    // deleted; surface as model_not_found so the caller knows to
    // refresh their model alias.
    let wasm_bytes = match state.functions.get(&translated.function_id) {
        Some(entry) => Arc::clone(&entry.value().wasm_bytes),
        None => {
            return OpenAiError::model_not_found(format!(
                "model maps to function {} which is no longer deployed",
                translated.function_id,
            ))
            .into_response();
        }
    };

    // The prompt now reaches the guest via the `wasi:tensor/host`
    // pull-model input channel: `run_buffered` / `run_streaming` stage
    // `translated.prompt` bytes on `SpawnConfig::input`, and the guest
    // copies them into its own linear memory via
    // `wasi:tensor/host.read-input` (sizing the read from
    // `input-len()`). A guest that ignores the input channel still runs
    // argument-less and produces its output via
    // `wasi:tensor/host.emit-chunk` as before — staging input is
    // non-breaking for those guests (`input-len()` is simply unread).

    if translated.stream {
        run_streaming(
            state,
            tenant,
            wasm_bytes,
            translated,
            model_echo,
            object_kind,
        )
        .await
    } else {
        run_buffered(
            state,
            tenant,
            wasm_bytes,
            translated,
            model_echo,
            object_kind,
        )
        .await
    }
}

/// Non-streaming branch. Spawns the call with a `StreamingContext`
/// attached, buffers every chunk the guest emits into a single string,
/// and returns the standard OpenAI JSON envelope.
async fn run_buffered(
    state: Arc<AppState>,
    tenant: TenantId,
    wasm_bytes: Arc<[u8]>,
    translated: TranslatedRequest,
    model_echo: String,
    object_kind: OpenAiObject,
) -> Response {
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(OPENAI_STREAM_BUFFER);
    let streaming = StreamingContext::with_channel(chunk_tx);

    let executor = state.executor.clone();
    let args = translated.args.clone();
    let function_id = translated.function_id;
    // Stage the assembled prompt so the guest can pull it via
    // `wasi:tensor/host.read-input`. Empty prompts stage nothing (the
    // guest sees `input-len() == 0`).
    let input = translated.prompt.into_bytes();

    let exec_handle = tokio::spawn(async move {
        let cfg = SpawnConfig::for_tenant(tenant)
            .with_deadline(OPENAI_INVOKE_DEADLINE)
            .with_streaming(streaming)
            .with_input(input);
        let instance_id = executor.spawn_instance(cfg, &wasm_bytes).await?;
        executor
            .call_export_with_args_then_terminate(instance_id, "_start", &args)
            .await
            .map(|_| ())
    });

    let mut collected: Vec<u8> = Vec::new();
    while let Some(chunk) = chunk_rx.recv().await {
        // Cheap upper bound check against pathologically large guest
        // output; the per-stream MAX_TOTAL_STREAM_BYTES cap enforced
        // inside StreamingContext::emit_chunk is the authoritative
        // limit, but a second host-side ceiling keeps the buffered
        // branch from OOMing on a misbehaving guest.
        if collected.len() + chunk.len() > 16 * 1024 * 1024 {
            tracing::warn!(
                target: "tensor_wasm_api::openai",
                function_id = %function_id,
                "guest output exceeded 16 MiB on the buffered (non-streaming) branch; \
                 truncating",
            );
            break;
        }
        collected.extend_from_slice(&chunk);
    }

    let exec_result = match exec_handle.await {
        Ok(r) => r,
        Err(e) => {
            return OpenAiError::server_error(format!("executor task panicked: {e}"), "wasm_error")
                .into_response();
        }
    };
    if let Err(e) = exec_result {
        return map_exec_error_to_openai(e).into_response();
    }

    let text = String::from_utf8_lossy(&collected).into_owned();
    let response_id = format!("{}{}", object_kind.id_prefix(), Uuid::new_v4());
    let created = unix_seconds_now();

    let envelope = match object_kind {
        OpenAiObject::TextCompletion => serde_json::json!({
            "id": response_id,
            "object": object_kind.object_kind(),
            "created": created,
            "model": model_echo,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop",
                "logprobs": serde_json::Value::Null,
            }],
            "usage": empty_usage(),
        }),
        OpenAiObject::ChatCompletion => serde_json::json!({
            "id": response_id,
            "object": object_kind.object_kind(),
            "created": created,
            "model": model_echo,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": empty_usage(),
        }),
    };

    (StatusCode::OK, Json(envelope)).into_response()
}

/// Streaming branch. Builds the same StreamingContext + executor
/// spawn, then wraps the receiver in a futures::Stream that emits one
/// OpenAI `data:` SSE frame per guest chunk and a final
/// `data: [DONE]\n\n` terminator.
async fn run_streaming(
    state: Arc<AppState>,
    tenant: TenantId,
    wasm_bytes: Arc<[u8]>,
    translated: TranslatedRequest,
    model_echo: String,
    object_kind: OpenAiObject,
) -> Response {
    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(OPENAI_STREAM_BUFFER);
    let streaming = StreamingContext::with_channel(chunk_tx);
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), ExecError>>();

    let executor = state.executor.clone();
    let args = translated.args.clone();
    // Stage the assembled prompt for the guest's `wasi:tensor/host`
    // pull-model input channel (`read-input`); empty prompts stage
    // nothing.
    let input = translated.prompt.into_bytes();

    tokio::spawn(async move {
        let cfg = SpawnConfig::for_tenant(tenant)
            .with_deadline(OPENAI_INVOKE_DEADLINE)
            .with_streaming(streaming)
            .with_input(input);
        let outcome = match executor.spawn_instance(cfg, &wasm_bytes).await {
            Ok(instance_id) => executor
                .call_export_with_args_then_terminate(instance_id, "_start", &args)
                .await
                .map(|_| ()),
            Err(e) => Err(e),
        };
        let _ = done_tx.send(outcome);
    });

    let response_id = format!("{}{}", object_kind.id_prefix(), Uuid::new_v4());
    let created = unix_seconds_now();

    // Build the SSE body as an `async_stream`-style unfold so each
    // emitted chunk becomes one OpenAI-shape `data: {...}` line, and
    // the terminal `data: [DONE]` line lands after the channel closes.
    let initial = OpenAiStreamState::Streaming {
        rx: chunk_rx,
        done_rx,
    };
    let sse_stream = futures::stream::unfold(initial, move |st| {
        let response_id = response_id.clone();
        let model_echo = model_echo.clone();
        async move {
            match st {
                OpenAiStreamState::Streaming {
                    mut rx,
                    mut done_rx,
                } => {
                    tokio::select! {
                        biased;
                        maybe = rx.recv() => match maybe {
                            Some(chunk) => {
                                let ev = make_chunk_event(
                                    &response_id,
                                    created,
                                    &model_echo,
                                    object_kind,
                                    &chunk,
                                );
                                Some((
                                    Ok::<Event, std::convert::Infallible>(ev),
                                    OpenAiStreamState::Streaming { rx, done_rx },
                                ))
                            }
                            None => {
                                // Channel closed; pick up terminal status.
                                let terminal = done_rx.await.unwrap_or(Ok(()));
                                Some((
                                    Ok(make_terminal_event(
                                        &response_id,
                                        created,
                                        &model_echo,
                                        object_kind,
                                        terminal.as_ref().err(),
                                    )),
                                    OpenAiStreamState::EmitDone,
                                ))
                            }
                        },
                        done = &mut done_rx => {
                            // Guest finished before we drained. Try
                            // popping any remaining buffered chunks
                            // synchronously, then emit the terminal
                            // frame.
                            let terminal = done.unwrap_or(Ok(()));
                            match rx.try_recv() {
                                Ok(chunk) => {
                                    let ev = make_chunk_event(
                                        &response_id,
                                        created,
                                        &model_echo,
                                        object_kind,
                                        &chunk,
                                    );
                                    Some((
                                        Ok(ev),
                                        OpenAiStreamState::DrainOnly {
                                            rx,
                                            terminal: Some(terminal),
                                        },
                                    ))
                                }
                                Err(_) => Some((
                                    Ok(make_terminal_event(
                                        &response_id,
                                        created,
                                        &model_echo,
                                        object_kind,
                                        terminal.as_ref().err(),
                                    )),
                                    OpenAiStreamState::EmitDone,
                                )),
                            }
                        }
                    }
                }
                OpenAiStreamState::DrainOnly { mut rx, terminal } => match rx.try_recv() {
                    Ok(chunk) => {
                        let ev = make_chunk_event(
                            &response_id,
                            created,
                            &model_echo,
                            object_kind,
                            &chunk,
                        );
                        Some((Ok(ev), OpenAiStreamState::DrainOnly { rx, terminal }))
                    }
                    Err(_) => Some((
                        Ok(make_terminal_event(
                            &response_id,
                            created,
                            &model_echo,
                            object_kind,
                            terminal.as_ref().and_then(|r| r.as_ref().err()),
                        )),
                        OpenAiStreamState::EmitDone,
                    )),
                },
                OpenAiStreamState::EmitDone => Some((
                    Ok(Event::default().data("[DONE]")),
                    OpenAiStreamState::Closed,
                )),
                OpenAiStreamState::Closed => None,
            }
        }
    });

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

enum OpenAiStreamState {
    Streaming {
        rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
        done_rx: tokio::sync::oneshot::Receiver<Result<(), ExecError>>,
    },
    DrainOnly {
        rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
        terminal: Option<Result<(), ExecError>>,
    },
    EmitDone,
    Closed,
}

/// Build one `data: { ... }` SSE event carrying a single OpenAI
/// `chat.completion.chunk` (or `text_completion`) delta.
///
/// SECURITY (control-byte / escape-sequence hazard): the guest's emitted
/// bytes are forwarded verbatim — they are decoded with
/// `String::from_utf8_lossy` and embedded directly in the JSON `text` /
/// `delta.content` field without sanitisation. Per `docs/STREAMING.md`,
/// the host does NOT strip control bytes or ANSI / terminal escape
/// sequences from guest output; a malicious or buggy guest can therefore
/// emit NULs, ANSI escapes, or other control characters that flow
/// straight through to the client. JSON string-escaping (applied by
/// `serde_json` when the payload is serialised) neutralises structural
/// injection into the SSE frame, but does not neutralise terminal escape
/// sequences once a client un-escapes and prints the content. Sanitising
/// for display is the client's responsibility (mirrors the native
/// `/invoke-stream` chunked branch in `routes.rs`, where the CLI's T18
/// sanitisation handles received text). We deliberately do not alter the
/// stream bytes here so byte-exact, non-text payloads survive the trip.
fn make_chunk_event(
    id: &str,
    created: u64,
    model: &str,
    object_kind: OpenAiObject,
    chunk_bytes: &[u8],
) -> Event {
    let text = String::from_utf8_lossy(chunk_bytes).into_owned();
    let payload = match object_kind {
        OpenAiObject::TextCompletion => serde_json::json!({
            "id": id,
            "object": object_kind.chunk_object_kind(),
            "created": created,
            "model": model,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": serde_json::Value::Null,
                "logprobs": serde_json::Value::Null,
            }],
        }),
        OpenAiObject::ChatCompletion => serde_json::json!({
            "id": id,
            "object": object_kind.chunk_object_kind(),
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": text },
                "finish_reason": serde_json::Value::Null,
            }],
        }),
    };
    Event::default().data(payload.to_string())
}

/// Build the terminal SSE event. On success this is a `finish_reason:
/// "stop"` frame; on error we emit an OpenAI error envelope as a
/// `data:` frame so clients can branch on it. The `[DONE]` sentinel is
/// emitted as a separate event after this one.
fn make_terminal_event(
    id: &str,
    created: u64,
    model: &str,
    object_kind: OpenAiObject,
    err: Option<&ExecError>,
) -> Event {
    let payload = if let Some(err) = err {
        serde_json::json!({
            "id": id,
            "object": object_kind.chunk_object_kind(),
            "created": created,
            "model": model,
            "error": {
                "message": format!("{err}"),
                "type": "server_error",
                "code": exec_error_code(err),
            },
        })
    } else {
        match object_kind {
            OpenAiObject::TextCompletion => serde_json::json!({
                "id": id,
                "object": object_kind.chunk_object_kind(),
                "created": created,
                "model": model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": serde_json::Value::Null,
                }],
            }),
            OpenAiObject::ChatCompletion => serde_json::json!({
                "id": id,
                "object": object_kind.chunk_object_kind(),
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }),
        }
    };
    Event::default().data(payload.to_string())
}

/// Surface an [`ExecError`] as an OpenAI envelope. Keeps the wire
/// errors aligned with the native `/invoke` surface's error categories
/// even though the OpenAI SDK only sees the OpenAI shape.
fn map_exec_error_to_openai(err: ExecError) -> OpenAiError {
    let code = exec_error_code(&err);
    OpenAiError::server_error(format!("{err}"), code)
}

fn exec_error_code(err: &ExecError) -> &'static str {
    match err {
        ExecError::Timeout(_) => "deadline_elapsed",
        ExecError::CapacityExhausted { .. } => "capacity_exhausted",
        ExecError::TenantCapacityExhausted { .. } => "tenant_capacity_exhausted",
        ExecError::ModuleMemoryTooLarge { .. } => "module_memory_too_large",
        ExecError::ModuleTooLarge { .. } => "module_too_large",
        ExecError::MissingExport(_) => "missing_export",
        ExecError::EpochTickerNotRunning => "epoch_ticker_not_running",
        ExecError::NotFound(_) => "instance_not_found",
        ExecError::Wasmtime(_) => "wasm_error",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_request_deserialises_minimal_body() {
        let raw = r#"{"model":"gpt-3.5-turbo","prompt":"hello"}"#;
        let parsed: CompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert_eq!(parsed.model, "gpt-3.5-turbo");
        assert_eq!(parsed.prompt, serde_json::json!("hello"));
        assert!(parsed.max_tokens.is_none());
    }

    #[test]
    fn completions_request_accepts_array_prompt() {
        // OpenAI accepts `prompt: ["a","b"]` — must parse without error.
        let raw = r#"{"model":"m","prompt":["a","b","c"]}"#;
        let parsed: CompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert!(parsed.prompt.is_array());
    }

    #[test]
    fn chat_completions_request_deserialises_minimal_body() {
        let raw = r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#;
        let parsed: ChatCompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.messages[0].role, "user");
    }

    #[test]
    fn chat_completions_request_accepts_empty_messages() {
        // The scaffold does not validate semantics; an empty messages
        // array still deserialises so the 501 shape is observable from
        // a maximally-stripped request.
        let raw = r#"{"model":"m","messages":[]}"#;
        let parsed: ChatCompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert!(parsed.messages.is_empty());
    }

    #[test]
    fn openai_error_envelope_serialises_to_openai_shape() {
        let env = OpenAiError::not_yet_wired("nope");
        let v = serde_json::to_value(&env).expect("serialises");
        // Top-level key is `error`.
        let inner = v.get("error").expect("error key present");
        assert_eq!(inner.get("message").and_then(|x| x.as_str()), Some("nope"));
        assert_eq!(
            inner.get("type").and_then(|x| x.as_str()),
            Some("not_implemented"),
        );
        assert!(inner.get("param").is_some_and(|x| x.is_null()));
        assert_eq!(
            inner.get("code").and_then(|x| x.as_str()),
            Some("openai_not_yet_wired"),
        );
    }

    #[test]
    fn openai_error_invalid_request_carries_param() {
        let env = OpenAiError::invalid_request("bad", Some("model".to_string()));
        let v = serde_json::to_value(&env).expect("serialises");
        let inner = v.get("error").unwrap();
        assert_eq!(
            inner.get("type").and_then(|x| x.as_str()),
            Some("invalid_request_error"),
        );
        assert_eq!(inner.get("param").and_then(|x| x.as_str()), Some("model"),);
    }
}
