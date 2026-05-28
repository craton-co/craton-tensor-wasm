// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAI request → internal invoke translation (T41, v0.4 wire-up).
//!
//! The v0.3.7 scaffold (`crates/tensor-wasm-api/src/openai.rs`) parsed
//! OpenAI request bodies and returned `501 openai_not_yet_wired`. T41
//! lands the translation step that resolves the `model` field against
//! a deploy-time configured `model → FunctionId` map and prepares an
//! invocation against the existing `tensor-wasm-exec` executor.
//!
//! ## Configuration
//!
//! The map is read from the env var
//! `TENSOR_WASM_API_OPENAI_MODEL_MAP`. Format:
//!
//! ```text
//! model_id_1:function_uuid_1,model_id_2:function_uuid_2,...
//! ```
//!
//! Example:
//!
//! ```text
//! gpt-3.5-turbo:00000000-0000-4000-8000-000000000001,gpt-4:00000000-0000-4000-8000-000000000002
//! ```
//!
//! Empty / unset env var produces an empty map. With an empty map every
//! request fails the `model` lookup and returns `404 model_not_found`
//! (per the OpenAI contract). A YAML config file alternative is deferred
//! to v0.5.
//!
//! ## Prompt → guest argv
//!
//! The translator returns a [`TranslatedRequest`] carrying the resolved
//! function id, the assembled prompt text, the prompt's byte length
//! (`prompt_len_hint`), and an `args` vector. For v0.4 the args vector
//! is empty so the guest's standard WASI command `_start () -> ()`
//! export links cleanly; the prompt length is preserved on the struct
//! for a future revision that promotes it to a typed `i32` argument
//! once the host-pre-fills-guest-memory plumbing lands. v0.4 guests
//! generate their response via the T34
//! `wasi:tensor/host.emit-chunk` host function; the handler drains
//! the receiver and surfaces the emitted bytes as the completion
//! text.
//!
//! ## Chat → prompt
//!
//! For `/v1/chat/completions` the translator concatenates the
//! `messages` array into a single prompt string with role-tagged
//! lines:
//!
//! ```text
//! system: You are a helpful assistant.
//! user: Hello!
//! assistant:
//! ```
//!
//! Empty content fields are tolerated; the role line is still emitted
//! so a guest that splits on `:\n` sees a stable shape. The trailing
//! `assistant:` line signals "complete this turn" to the guest.

use std::collections::HashMap;
use std::sync::Arc;

use tensor_wasm_exec::executor::WasmArg;
use uuid::Uuid;

use crate::openai::{ChatCompletionsRequest, ChatMessage, CompletionsRequest, OpenAiError};

/// Environment variable carrying the comma-separated
/// `model:function_uuid` map. See module-level docs.
pub const ENV_OPENAI_MODEL_MAP: &str = "TENSOR_WASM_API_OPENAI_MODEL_MAP";

/// Resolved `model → FunctionId` map, shared across handler invocations
/// via `Arc`.
pub type ModelMap = Arc<HashMap<String, Uuid>>;

/// Parse the `TENSOR_WASM_API_OPENAI_MODEL_MAP` env-var value into a
/// `HashMap<String, Uuid>`.
///
/// Empty / unset value yields an empty map. Malformed entries (missing
/// colon, unparseable UUID) are skipped with a `tracing::warn!` so a
/// single bad row does not block startup — the remaining valid entries
/// stay live.
pub fn parse_model_map_env(raw: &str) -> HashMap<String, Uuid> {
    let mut out: HashMap<String, Uuid> = HashMap::new();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return out;
    }
    for entry in trimmed.split(',') {
        let frag = entry.trim();
        if frag.is_empty() {
            continue;
        }
        let (model, uuid_str) = match frag.split_once(':') {
            Some((m, u)) => (m.trim(), u.trim()),
            None => {
                tracing::warn!(
                    target: "tensor_wasm_api::openai_translator",
                    entry = %frag,
                    var = ENV_OPENAI_MODEL_MAP,
                    "ignoring malformed entry: missing ':' separator (expected model:function_uuid)",
                );
                continue;
            }
        };
        if model.is_empty() {
            tracing::warn!(
                target: "tensor_wasm_api::openai_translator",
                entry = %frag,
                var = ENV_OPENAI_MODEL_MAP,
                "ignoring malformed entry: empty model id",
            );
            continue;
        }
        let id = match Uuid::parse_str(uuid_str) {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(
                    target: "tensor_wasm_api::openai_translator",
                    entry = %frag,
                    var = ENV_OPENAI_MODEL_MAP,
                    error = %e,
                    "ignoring malformed entry: function uuid does not parse",
                );
                continue;
            }
        };
        out.insert(model.to_owned(), id);
    }
    out
}

/// Read `TENSOR_WASM_API_OPENAI_MODEL_MAP` from the ambient process env
/// and return an [`Arc`]-wrapped `HashMap` for installation into
/// [`crate::AppState`]. Empty / unset env var produces an empty map.
pub fn model_map_from_env() -> ModelMap {
    let raw = std::env::var(ENV_OPENAI_MODEL_MAP).unwrap_or_default();
    let map = parse_model_map_env(&raw);
    if !map.is_empty() {
        tracing::info!(
            target: "tensor_wasm_api::openai_translator",
            entries = map.len(),
            "OpenAI model map configured",
        );
    }
    Arc::new(map)
}

/// Translation output: the resolved [`Uuid`] function id, the
/// fully-assembled prompt text, and the synthetic [`WasmArg`] vector to
/// pass into `call_export_with_args`.
///
/// **v0.4 args policy:** the `args` vector is empty so the standard
/// WASI command `_start () -> ()` export signature works out of the
/// box. The prompt length is preserved in `prompt_len_hint` so a
/// future revision that lands the host-pre-fills-guest-memory
/// plumbing can promote it to an `i32` arg without churning the
/// translator API surface. v0.4 guests should either be fixed-output
/// or stream their reply via `wasi:tensor/host.emit-chunk` (T34) — the
/// host buffers / forwards every emitted chunk.
#[derive(Debug, Clone)]
pub struct TranslatedRequest {
    /// Resolved function id.
    pub function_id: Uuid,
    /// Fully-assembled prompt text (concatenated messages array for
    /// `/v1/chat/completions`).
    pub prompt: String,
    /// Byte length of the assembled prompt, clamped to `i32::MAX`.
    /// Reserved for a future revision that passes the prompt length
    /// into the guest as a typed argument; not used by v0.4.
    pub prompt_len_hint: i32,
    /// Args to pass into the executor. v0.4 ships empty.
    pub args: Vec<WasmArg>,
    /// `true` if the original request had `stream: true`.
    pub stream: bool,
}

/// Extract a single prompt string from the OpenAI `prompt` field. The
/// OpenAI wire contract accepts either a JSON string or an array of
/// strings; we accept both, concatenating the array with newlines.
/// Any other shape (number, null, object) is treated as an empty
/// prompt with no error — the scaffold deliberately stays permissive
/// on this surface.
fn extract_prompt_text(prompt: &serde_json::Value) -> String {
    match prompt {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Extract the textual content of a single chat message. The OpenAI
/// contract accepts either a plain string or an array of content parts
/// (for multimodal). For v0.4 we only consume the text parts; binary
/// / image parts are deferred to v0.5.
fn extract_message_content(msg: &ChatMessage) -> String {
    match &msg.content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Assemble the chat-messages array into a single prompt string with
/// role-tagged lines. Each entry becomes one `role: content` line; a
/// final `assistant:` line signals the guest that it should generate
/// the next turn.
///
/// Public so tests can pin the assembly contract without driving the
/// full HTTP surface.
pub fn assemble_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = if msg.role.is_empty() {
            "user"
        } else {
            msg.role.as_str()
        };
        let content = extract_message_content(msg);
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&content);
        out.push('\n');
    }
    // Trailing turn marker so the guest sees a stable "generate the
    // assistant response now" signal.
    out.push_str("assistant:");
    out
}

/// Translate a [`CompletionsRequest`] into a [`TranslatedRequest`].
///
/// Returns `Err(OpenAiError)` with the OpenAI 404 envelope
/// (`type: "invalid_request_error"`, `code: "model_not_found"`) when
/// `req.model` is not present in the `model_map`.
///
/// Args policy (v0.4): an empty `Vec<WasmArg>` is returned because
/// the typical `_start () -> ()` WASI command export takes no
/// arguments. The prompt length is preserved in the
/// [`TranslatedRequest::prompt_len_hint`] field so a future revision
/// that lands the host-pre-fills-guest-memory plumbing can promote it
/// to an `i32` arg without breaking existing guests.
pub fn translate_completions_request(
    req: &CompletionsRequest,
    model_map: &HashMap<String, Uuid>,
) -> Result<TranslatedRequest, OpenAiError> {
    let function_id = lookup_model(&req.model, model_map)?;
    let prompt = extract_prompt_text(&req.prompt);
    let prompt_len_hint = clamp_len_to_i32(prompt.len());
    Ok(TranslatedRequest {
        function_id,
        prompt,
        prompt_len_hint,
        args: Vec::new(),
        stream: req.stream.unwrap_or(false),
    })
}

/// Translate a [`ChatCompletionsRequest`] into a [`TranslatedRequest`].
///
/// See [`assemble_chat_prompt`] for the message-array concatenation
/// contract. Same `model_not_found` envelope as
/// [`translate_completions_request`].
pub fn translate_chat_completions_request(
    req: &ChatCompletionsRequest,
    model_map: &HashMap<String, Uuid>,
) -> Result<TranslatedRequest, OpenAiError> {
    let function_id = lookup_model(&req.model, model_map)?;
    let prompt = assemble_chat_prompt(&req.messages);
    let prompt_len_hint = clamp_len_to_i32(prompt.len());
    Ok(TranslatedRequest {
        function_id,
        prompt,
        prompt_len_hint,
        args: Vec::new(),
        stream: req.stream.unwrap_or(false),
    })
}

/// Common model-lookup helper. Returns a `model_not_found` OpenAI
/// envelope on miss.
fn lookup_model(
    model: &str,
    model_map: &HashMap<String, Uuid>,
) -> Result<Uuid, OpenAiError> {
    match model_map.get(model) {
        Some(id) => Ok(*id),
        None => Err(OpenAiError::model_not_found(format!(
            "model `{model}` is not configured in TENSOR_WASM_API_OPENAI_MODEL_MAP; \
             ask your operator to add a `{model}:<function_uuid>` entry",
        ))),
    }
}

/// Clamp a `usize` byte length to `i32`. Prompts above `i32::MAX`
/// bytes are saturated to `i32::MAX`; that's a 2 GiB prompt and any
/// real prompt is many orders of magnitude smaller. We deliberately
/// do not return an error here — the body-size cap
/// (`MAX_REQUEST_BODY_BYTES`, 64 MiB) at the middleware layer is the
/// authoritative bound. This saturation is just defensive arithmetic
/// against the `as i32` cast.
fn clamp_len_to_i32(len: usize) -> i32 {
    if len > i32::MAX as usize {
        i32::MAX
    } else {
        len as i32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_uuid_1() -> Uuid {
        Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap()
    }

    fn fixture_uuid_2() -> Uuid {
        Uuid::parse_str("00000000-0000-4000-8000-000000000002").unwrap()
    }

    #[test]
    fn parse_empty_yields_empty_map() {
        assert!(parse_model_map_env("").is_empty());
        assert!(parse_model_map_env("   ").is_empty());
    }

    #[test]
    fn parse_single_entry() {
        let raw = "gpt-3.5-turbo:00000000-0000-4000-8000-000000000001";
        let map = parse_model_map_env(raw);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("gpt-3.5-turbo"), Some(&fixture_uuid_1()));
    }

    #[test]
    fn parse_multiple_entries() {
        let raw = "gpt-3.5-turbo:00000000-0000-4000-8000-000000000001,gpt-4:00000000-0000-4000-8000-000000000002";
        let map = parse_model_map_env(raw);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("gpt-3.5-turbo"), Some(&fixture_uuid_1()));
        assert_eq!(map.get("gpt-4"), Some(&fixture_uuid_2()));
    }

    #[test]
    fn parse_skips_malformed_entries() {
        let raw = "good:00000000-0000-4000-8000-000000000001,bad-no-colon,empty:,:no-model,gpt-4:not-a-uuid";
        let map = parse_model_map_env(raw);
        // Only `good` survives.
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("good"));
    }

    #[test]
    fn parse_tolerates_whitespace() {
        let raw = " gpt-3.5-turbo : 00000000-0000-4000-8000-000000000001 , gpt-4:00000000-0000-4000-8000-000000000002 ";
        let map = parse_model_map_env(raw);
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("gpt-3.5-turbo"));
        assert!(map.contains_key("gpt-4"));
    }

    #[test]
    fn translate_completions_returns_function_id_on_hit() {
        let mut map = HashMap::new();
        map.insert("m1".to_owned(), fixture_uuid_1());
        let req = CompletionsRequest {
            model: "m1".to_owned(),
            prompt: json!("hello"),
            stream: Some(false),
            ..Default::default()
        };
        let out = translate_completions_request(&req, &map).expect("translates");
        assert_eq!(out.function_id, fixture_uuid_1());
        assert_eq!(out.prompt, "hello");
        assert!(!out.stream);
        // v0.4 args policy: empty vec, prompt length preserved in hint.
        assert!(out.args.is_empty(), "args must be empty in v0.4");
        assert_eq!(out.prompt_len_hint, 5);
    }

    #[test]
    fn translate_completions_returns_model_not_found_on_miss() {
        let map = HashMap::new();
        let req = CompletionsRequest {
            model: "unknown".to_owned(),
            prompt: json!("hello"),
            ..Default::default()
        };
        let err = translate_completions_request(&req, &map).expect_err("misses");
        assert_eq!(err.error.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn translate_completions_array_prompt_joined_with_newlines() {
        let mut map = HashMap::new();
        map.insert("m1".to_owned(), fixture_uuid_1());
        let req = CompletionsRequest {
            model: "m1".to_owned(),
            prompt: json!(["a", "b", "c"]),
            ..Default::default()
        };
        let out = translate_completions_request(&req, &map).expect("translates");
        assert_eq!(out.prompt, "a\nb\nc");
    }

    #[test]
    fn translate_chat_completions_assembles_role_tagged_prompt() {
        let mut map = HashMap::new();
        map.insert("m2".to_owned(), fixture_uuid_2());
        let req = ChatCompletionsRequest {
            model: "m2".to_owned(),
            messages: vec![
                ChatMessage {
                    role: "system".to_owned(),
                    content: json!("You are helpful."),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_owned(),
                    content: json!("Hi."),
                    name: None,
                },
            ],
            stream: Some(true),
            ..Default::default()
        };
        let out = translate_chat_completions_request(&req, &map).expect("translates");
        assert_eq!(out.function_id, fixture_uuid_2());
        assert!(out.stream);
        assert!(
            out.prompt.contains("system: You are helpful."),
            "missing system role tag: {}",
            out.prompt,
        );
        assert!(
            out.prompt.contains("user: Hi."),
            "missing user role tag: {}",
            out.prompt,
        );
        assert!(
            out.prompt.trim_end().ends_with("assistant:"),
            "must end with trailing `assistant:` turn marker: {}",
            out.prompt,
        );
    }

    #[test]
    fn translate_chat_completions_empty_messages_yields_only_marker() {
        let mut map = HashMap::new();
        map.insert("m".to_owned(), fixture_uuid_1());
        let req = ChatCompletionsRequest {
            model: "m".to_owned(),
            messages: vec![],
            ..Default::default()
        };
        let out = translate_chat_completions_request(&req, &map).expect("translates");
        assert_eq!(out.prompt, "assistant:");
    }

    #[test]
    fn translate_chat_completions_default_role_is_user() {
        let mut map = HashMap::new();
        map.insert("m".to_owned(), fixture_uuid_1());
        let req = ChatCompletionsRequest {
            model: "m".to_owned(),
            messages: vec![ChatMessage {
                role: String::new(),
                content: json!("hi"),
                name: None,
            }],
            ..Default::default()
        };
        let out = translate_chat_completions_request(&req, &map).expect("translates");
        assert!(out.prompt.contains("user: hi"), "got {}", out.prompt);
    }
}
