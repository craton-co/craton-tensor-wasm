// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAI-compatible inference gateway shim (scaffold).
//!
//! v0.3.5 lands a *scaffold* exposing the two highest-traffic OpenAI
//! REST surfaces — `POST /v1/completions` and `POST /v1/chat/completions`
//! — at the same `tensor-wasm-api` axum router as the native
//! `/functions/{id}/invoke` endpoints. The handlers accept the standard
//! OpenAI request bodies (so off-the-shelf SDKs send valid JSON) and
//! return `501 Not Implemented` with the OpenAI-shape error envelope:
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

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::rejection::JsonRejection, Json};
use serde::{Deserialize, Serialize};

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
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        // Default status is 501. Handlers that need a different status
        // (e.g. malformed JSON → 400) construct a tuple `(status,
        // OpenAiError)` and `.into_response()` it via the (StatusCode, T)
        // blanket impl.
        let status = match self.error.kind.as_str() {
            "invalid_request_error" => StatusCode::BAD_REQUEST,
            _ => StatusCode::NOT_IMPLEMENTED,
        };
        (status, Json(self)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /v1/completions` — OpenAI completions shim.
///
/// Parses the OpenAI [`CompletionsRequest`] and returns `501` with the
/// OpenAI-shape error envelope. Malformed JSON is rejected at the
/// extractor with `400` and the OpenAI `invalid_request_error` shape.
///
/// The handler intentionally does no work beyond shape validation —
/// every other behaviour (model lookup, tenant resolution, streaming)
/// lands in v0.4. The scaffold's only job is to lock the URL surface
/// and the error envelope.
#[tracing::instrument(name = "http.openai.completions", skip(payload))]
pub async fn completions_handler(
    payload: Result<Json<CompletionsRequest>, JsonRejection>,
) -> Response {
    let Json(_req) = match payload {
        Ok(j) => j,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenAiError::invalid_request(rej.body_text(), None)),
            )
                .into_response();
        }
    };
    OpenAiError::not_yet_wired(
        "OpenAI-compatible /v1/completions endpoint is a scaffold; the v0.4 release wires \
         model → function translation. See docs/OPENAI-COMPAT.md for the timeline.",
    )
    .into_response()
}

/// `POST /v1/chat/completions` — OpenAI chat-completions shim.
///
/// Parses the OpenAI [`ChatCompletionsRequest`] and returns `501` with
/// the OpenAI-shape error envelope. Malformed JSON is rejected at the
/// extractor with `400` and the OpenAI `invalid_request_error` shape.
///
/// See [`completions_handler`] for the rationale on why the v0.3.5
/// scaffold only validates shape; the v0.4 wiring step lands model
/// resolution, tenant inference from the bearer token's
/// [`TokenScope`](crate::token_scope::TokenScope), and SSE streaming.
#[tracing::instrument(name = "http.openai.chat_completions", skip(payload))]
pub async fn chat_completions_handler(
    payload: Result<Json<ChatCompletionsRequest>, JsonRejection>,
) -> Response {
    let Json(_req) = match payload {
        Ok(j) => j,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenAiError::invalid_request(rej.body_text(), None)),
            )
                .into_response();
        }
    };
    OpenAiError::not_yet_wired(
        "OpenAI-compatible /v1/chat/completions endpoint is a scaffold; the v0.4 release wires \
         model → function translation. See docs/OPENAI-COMPAT.md for the timeline.",
    )
    .into_response()
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
        assert!(inner.get("param").map(|x| x.is_null()).unwrap_or(false));
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
        assert_eq!(
            inner.get("param").and_then(|x| x.as_str()),
            Some("model"),
        );
    }
}
