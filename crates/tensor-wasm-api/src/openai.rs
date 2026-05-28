// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAI-compatible inference gateway (roadmap feature #10).
//!
//! Exposes `/v1/completions` and `/v1/chat/completions` routes that mirror
//! the OpenAI wire format. Today, both routes return `501 Not Implemented`
//! with a structured envelope explaining the feature is in scaffold mode.
//! v0.4 wires translation to the internal `invoke` path.
//!
//! Why this matters: the bigger addressable market is Modal/Replicate/Beam
//! users with Python OpenAI clients, not Wasmtime/Wasmer migrators. A
//! drop-in OpenAI shim lets a wider set of users try TensorWasm without
//! rewriting their client code. See
//! `docs/PATH-TO-V1.md#post-v036-strategic-features`.
//!
//! ## Auth + tenant resolution
//!
//! These routes are mounted OUTSIDE the `X-TensorWasm-Tenant`
//! header-resolution middleware: stock OpenAI clients (the Python SDK,
//! `curl` recipes copied from the OpenAI cookbook, the JS SDK, …) do not
//! send a tenant header. Bearer authentication still runs — the OpenAI
//! SDK already sets `Authorization: Bearer <key>` for every call — so
//! the tenant the call belongs to MUST be derived from the bearer
//! token's `:tenant=` scope (see [`crate::token_scope`]). A token with
//! a wildcard scope falls back to `TenantId(0)` today; v0.4's
//! translation glue makes the resolution explicit (single-tenant
//! scopes → that tenant; multi-tenant scopes → reject with a clear
//! error rather than guess).
//!
//! ## Scaffold contract
//!
//! Both handlers parse the request body into a strongly-typed envelope
//! ([`CompletionsRequest`] / [`ChatCompletionsRequest`]) so the wire
//! shape is fixed today and a v0.4 translation layer can consume the
//! same struct without changing the public surface. The envelopes are
//! marked `#[non_exhaustive]` so adding new optional fields (top_p,
//! frequency_penalty, …) is a backwards-compatible change. A malformed
//! body still returns the standard `400 invalid_json` envelope via
//! axum's `JsonRejection`; a well-formed body returns `501` with
//! `error.code == "openai_not_yet_wired"`.

use axum::{
    extract::{Extension, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::routes::{ApiError, AppState};
use tensor_wasm_core::types::TenantId;

/// OpenAI `POST /v1/completions` request (legacy completion API).
///
/// Mirrors the public OpenAI completion schema. `prompt` is typed as
/// `serde_json::Value` because the OpenAI surface accepts EITHER a
/// single string OR a JSON array of strings (and, in the token-ids
/// variant, an array of integer arrays); pinning a stricter type here
/// would reject valid OpenAI client traffic. v0.4's translation layer
/// will normalise the variants into a single shape.
///
/// `#[non_exhaustive]` so adding optional fields (top_p, n, stop,
/// presence_penalty, frequency_penalty, …) does not break SemVer.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct CompletionsRequest {
    /// Model identifier (`gpt-3.5-turbo-instruct`, `text-davinci-003`,
    /// …). v0.4 maps known names onto deployed TensorWasm functions; an
    /// unknown model returns `400 unknown_model`.
    pub model: String,
    /// Prompt text. String OR array — see the struct-level comment for
    /// the rationale.
    pub prompt: serde_json::Value,
    /// Maximum number of tokens to generate. Optional per the OpenAI
    /// schema; absent means model default.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature in `[0.0, 2.0]`. Optional.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// When `true`, the response is a Server-Sent Events stream of
    /// `data: ...\n\n` frames terminated by `data: [DONE]\n\n`. Not yet
    /// supported by the scaffold; v0.4 wires this through the executor's
    /// async invoke path.
    #[serde(default)]
    pub stream: Option<bool>,
    // Additional fields (top_p, n, stop, presence_penalty,
    // frequency_penalty, user, …) will be added without breaking
    // SemVer thanks to `#[non_exhaustive]`.
}

/// OpenAI `POST /v1/chat/completions` request (chat-style messages).
///
/// `#[non_exhaustive]` for the same SemVer rationale as
/// [`CompletionsRequest`]. The `messages` array is required by the
/// OpenAI schema; an empty array is accepted by the parser today but
/// will be rejected with a clear validation error once v0.4 wires
/// translation.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct ChatCompletionsRequest {
    /// Model identifier — see [`CompletionsRequest::model`].
    pub model: String,
    /// Ordered conversation history. Each element carries a `role`
    /// (`system` / `user` / `assistant` / `tool`) and a `content`
    /// string. Tool-call extensions land in v0.4.
    pub messages: Vec<ChatMessage>,
    /// Maximum number of tokens to generate. Optional.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature. Optional.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// SSE streaming flag — see [`CompletionsRequest::stream`].
    #[serde(default)]
    pub stream: Option<bool>,
}

/// A single chat-completion message.
///
/// Kept minimal in the scaffold: just the OpenAI required tuple
/// `(role, content)`. Tool-call / function-call fields land with the
/// v0.4 wiring.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct ChatMessage {
    /// Speaker role. OpenAI defines `system`, `user`, `assistant`,
    /// `tool`; the scaffold accepts any string so unknown roles flow
    /// through to the translation layer for validation.
    pub role: String,
    /// Message text content.
    pub content: String,
}

impl ChatMessage {
    /// Construct a chat message with the given role and content. Useful
    /// for downstream consumers (and v0.4's translation tests) that need
    /// to assemble messages programmatically without going through the
    /// `#[non_exhaustive]` struct-literal restriction.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// OpenAI-shaped error envelope returned today.
///
/// Public so downstream client libraries can deserialise the response
/// against the same type the server emits. The shape matches the
/// `{ "error": { "message": ..., "type": ..., "code": ... } }` envelope
/// the OpenAI HTTP API uses (modulo the `param` field, which is
/// omitted by the scaffold and added back in v0.4 when validation
/// errors surface).
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiError {
    /// Inner body — see [`OpenAiErrorBody`].
    pub error: OpenAiErrorBody,
}

/// Inner body of [`OpenAiError`].
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiErrorBody {
    /// Human-readable description. Not stable across versions.
    pub message: String,
    /// OpenAI-style error class (`invalid_request_error`,
    /// `server_error`, `not_implemented_error`, …). Stable.
    pub r#type: String,
    /// Machine-readable, stable identifier (e.g.
    /// `openai_not_yet_wired`). Clients should branch on this.
    pub code: String,
}

/// Marker code returned by both scaffold handlers. Clients can branch
/// on this constant to detect the "feature is scaffolded but not yet
/// wired" state and degrade gracefully (e.g. by failing over to a
/// different inference backend) without parsing the human-readable
/// `message`.
pub const SCAFFOLD_ERROR_CODE: &str = "openai_not_yet_wired";

/// `POST /v1/completions` handler (scaffold).
///
/// Returns `501 Not Implemented` with the OpenAI error envelope. The
/// route is wired so clients can probe the surface; v0.4 lands
/// translation to the internal invoke path.
///
/// Tenant is sourced from the bearer-token scope (see the module-level
/// "Auth + tenant resolution" section) — the OpenAI Python SDK does
/// not send `X-TensorWasm-Tenant` so we cannot rely on the header
/// middleware here. The handler accepts `Option<Extension<TenantId>>`
/// purely so tests that exercise the route with a header-aware router
/// keep working; the scaffold itself never consumes the value.
pub async fn completions_handler(
    State(_state): State<std::sync::Arc<AppState>>,
    _tenant: Option<Extension<TenantId>>,
    Json(_req): Json<CompletionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Err(ApiError::not_implemented(
        SCAFFOLD_ERROR_CODE,
        "OpenAI-compatible /v1/completions is scaffold-only in v0.3.6; \
         v0.4 wires translation to internal invoke.",
    ))
}

/// `POST /v1/chat/completions` handler (scaffold). See
/// [`completions_handler`] for the full contract.
pub async fn chat_completions_handler(
    State(_state): State<std::sync::Arc<AppState>>,
    _tenant: Option<Extension<TenantId>>,
    Json(_req): Json<ChatCompletionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Err(ApiError::not_implemented(
        SCAFFOLD_ERROR_CODE,
        "OpenAI-compatible /v1/chat/completions is scaffold-only in v0.3.6; \
         v0.4 wires translation to internal invoke.",
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_request_parses_string_prompt() {
        let raw = r#"{"model":"gpt-3.5-turbo-instruct","prompt":"hello"}"#;
        let req: CompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert_eq!(req.model, "gpt-3.5-turbo-instruct");
        assert_eq!(req.prompt, serde_json::json!("hello"));
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.stream.is_none());
    }

    #[test]
    fn completions_request_parses_array_prompt() {
        // OpenAI allows `prompt` to be a JSON array of strings — confirm
        // the Value-typed field accepts that shape without bespoke
        // handling.
        let raw = r#"{"model":"text-davinci-003","prompt":["a","b","c"]}"#;
        let req: CompletionsRequest = serde_json::from_str(raw).expect("parses array");
        assert_eq!(req.prompt, serde_json::json!(["a", "b", "c"]));
    }

    #[test]
    fn chat_completions_request_parses_minimal_body() {
        let raw = r#"{
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let req: ChatCompletionsRequest = serde_json::from_str(raw).expect("parses");
        assert_eq!(req.model, "gpt-4o-mini");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].content, "hi");
    }

    #[test]
    fn openai_error_serialises_with_lowercase_type_field() {
        // Pin the wire shape: the `type` field must serialise as `type`
        // (not `r#type`) so the response matches OpenAI's documented
        // envelope. `r#type` is purely the Rust raw-identifier syntax;
        // serde renders it as `type` by default — this test guards
        // against a future `#[serde(rename = "...")]` accidentally
        // breaking that property.
        let env = OpenAiError {
            error: OpenAiErrorBody {
                message: "scaffold".to_string(),
                r#type: "not_implemented_error".to_string(),
                code: SCAFFOLD_ERROR_CODE.to_string(),
            },
        };
        let json = serde_json::to_value(&env).expect("serialises");
        let err = json.get("error").expect("envelope wraps inner");
        assert_eq!(err.get("type").and_then(|v| v.as_str()), Some("not_implemented_error"));
        assert_eq!(err.get("code").and_then(|v| v.as_str()), Some(SCAFFOLD_ERROR_CODE));
    }

    #[test]
    fn chat_message_new_round_trips() {
        let m = ChatMessage::new("user", "hi");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hi");
        let json = serde_json::to_value(&m).expect("serialises");
        assert_eq!(json, serde_json::json!({"role": "user", "content": "hi"}));
    }
}
