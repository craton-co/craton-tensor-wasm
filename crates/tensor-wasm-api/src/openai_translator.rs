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
//! ## Prompt → guest input channel (pull model)
//!
//! The translator returns a [`TranslatedRequest`] carrying the resolved
//! function id, the assembled prompt text, the prompt's byte length
//! (`prompt_len_hint`), and an `args` vector.
//!
//! **How the prompt reaches the guest.** The handler in `openai.rs`
//! stages the assembled `prompt` bytes on
//! [`SpawnConfig::input`](tensor_wasm_exec::executor::SpawnConfig::input)
//! at spawn time. The shared
//! [`tensor_wasm_exec::executor::TensorWasmExecutor`] installs those
//! bytes on the per-instance host state and wires the `wasi:tensor/host`
//! pull-model input channel — `input-len() -> u32` and `read-input(ptr,
//! len) -> s32` — so the guest copies the prompt into its own linear
//! memory at the start of the invocation (size the buffer from
//! `input-len()`, then drain it with `read-input`). This is the input
//! counterpart to the output-only
//! [`StreamingContext`](tensor_wasm_wasi_gpu::streaming::StreamingContext)
//! `emit-chunk` channel.
//!
//! Note this is a *bytes* channel, distinct from the numeric
//! [`WasmArg`](tensor_wasm_exec::executor::WasmArg) call args:
//!
//! * The assembled prompt and its byte length are preserved on
//!   [`TranslatedRequest`] (`prompt` / `prompt_len_hint`); the handler
//!   moves `prompt` into `SpawnConfig::input` so the guest can pull it.
//! * [`TranslatedRequest::args`] is left **empty**, deliberately, and
//!   for a load-bearing reason: the executor's `call_export_with_args`
//!   uses wasmtime's *dynamic* `Func::call_async` path for any non-empty
//!   arg slice, which validates the param arity against the export's
//!   declared signature **exactly**. The standard WASI command export is
//!   `_start () -> ()`; handing it even a single `i32` would make
//!   wasmtime reject the call (`ExecError::Wasmtime`) and break every
//!   off-the-shelf guest. The empty-args path takes the typed
//!   `func.typed::<(), ()>()` fast path that those guests rely on —
//!   identical to how `routes.rs`'s `/invoke` runs an argument-less body.
//!   The prompt travels through the `wasi:tensor/host` input channel
//!   instead of the numeric argv, so lowering it into `args` is neither
//!   needed nor safe.
//!
//! A guest that does not import `read-input` still runs argument-less and
//! produces its response via the
//! `wasi:tensor/host.emit-chunk` host function; the handler drains the
//! receiver and surfaces the emitted bytes as the completion text.
//! Staging input is therefore non-breaking for such guests — the
//! staged bytes simply go unread.
//!
//! ## Unsupported sampling knobs
//!
//! `max_tokens`, `temperature`, `n`, `echo`, and `tools` are parsed off
//! the wire but the executor exposes no knob for any of them. Rather
//! than silently ignore a caller's explicit setting (which would make
//! the gateway lie about honouring it), the translator returns a clear
//! `400 invalid_request_error` (`code: "unsupported_parameter"`,
//! `param` naming the field) whenever one of these is set to a value the
//! gateway cannot satisfy.
//!
//! A value that coincides with a no-op default is accepted so common
//! SDK boilerplate still works: `temperature: 1.0`, `n: 1`,
//! `echo: false`, and an empty / null / absent `tools` all translate
//! cleanly. `max_tokens` has no honourable value — we cannot cap
//! generation — so any explicit `max_tokens` is rejected. This is
//! revisited once the executor grows the corresponding sampling
//! controls.
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
/// **args policy:** the `args` vector is empty, deliberately. The
/// standard WASI command export is `_start () -> ()`, and the executor's
/// `call_export_with_args` switches to wasmtime's dynamic call path for
/// any non-empty arg slice — which validates param arity against the
/// export signature exactly and would reject a `()`-arity guest handed
/// an argument. The prompt is a *bytes* payload, not a numeric arg, so
/// it travels through the `wasi:tensor/host` pull-model input channel
/// (`input-len` / `read-input`) rather than `args`: the handler moves
/// `prompt` into
/// [`SpawnConfig::input`](tensor_wasm_exec::executor::SpawnConfig::input)
/// at spawn time. See the module-level "Prompt → guest input channel"
/// section for the full rationale. Guests stream their reply via
/// `wasi:tensor/host.emit-chunk` — the host buffers / forwards every
/// emitted chunk.
#[derive(Debug, Clone)]
pub struct TranslatedRequest {
    /// Resolved function id.
    pub function_id: Uuid,
    /// Fully-assembled prompt text (concatenated messages array for
    /// `/v1/chat/completions`). The handler stages these bytes on
    /// [`SpawnConfig::input`](tensor_wasm_exec::executor::SpawnConfig::input)
    /// so the guest can pull them via `wasi:tensor/host.read-input`.
    pub prompt: String,
    /// Byte length of the assembled prompt, clamped to `i32::MAX`.
    /// Preserved as an observability hint (e.g. for tracing the staged
    /// prompt size); the prompt itself is delivered to the guest via the
    /// `wasi:tensor/host` input channel.
    pub prompt_len_hint: i32,
    /// Args to pass into the executor. Empty in v0.4 so the standard
    /// `_start () -> ()` guest links via the typed fast path; see the
    /// struct-level note.
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
/// `req.model` is not present in the `model_map`, or the OpenAI 400
/// envelope (`code: "unsupported_parameter"`) when an unsupported
/// sampling knob (`max_tokens` / `temperature` / `n` / `echo`) is set
/// to a non-default value — see [`reject_unsupported_knob`].
///
/// Args policy (v0.4): an empty `Vec<WasmArg>` is returned — see
/// [`TranslatedRequest`] for why the prompt is preserved on the struct
/// but not lowered into `args`.
pub fn translate_completions_request(
    req: &CompletionsRequest,
    model_map: &HashMap<String, Uuid>,
) -> Result<TranslatedRequest, OpenAiError> {
    let function_id = lookup_model(&req.model, model_map)?;
    // Honest-rejection policy: the executor exposes no sampling knobs,
    // so a caller that explicitly sets one gets a 400 rather than a
    // response that silently ignored it.
    reject_unsupported_knob("max_tokens", req.max_tokens.map(|v| v as f64), None)?;
    reject_unsupported_knob("temperature", req.temperature.map(|v| v as f64), Some(1.0))?;
    reject_unsupported_knob("n", req.n.map(|v| v as f64), Some(1.0))?;
    reject_unsupported_bool("echo", req.echo)?;
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
    // Honest-rejection policy: see translate_completions_request.
    reject_unsupported_knob("max_tokens", req.max_tokens.map(|v| v as f64), None)?;
    reject_unsupported_knob("temperature", req.temperature.map(|v| v as f64), Some(1.0))?;
    reject_unsupported_knob("n", req.n.map(|v| v as f64), Some(1.0))?;
    reject_unsupported_tools(req.tools.as_ref())?;
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
fn lookup_model(model: &str, model_map: &HashMap<String, Uuid>) -> Result<Uuid, OpenAiError> {
    match model_map.get(model) {
        Some(id) => Ok(*id),
        None => Err(OpenAiError::model_not_found(format!(
            "model `{model}` is not configured in TENSOR_WASM_API_OPENAI_MODEL_MAP; \
             ask your operator to add a `{model}:<function_uuid>` entry",
        ))),
    }
}

/// Reject a numeric sampling knob the executor cannot honour, *unless*
/// the caller's value coincides with a no-op default the gateway can
/// satisfy without doing anything.
///
/// The executor exposes no sampling controls, so honouring a non-default
/// value is impossible today. Per the module's honest-rejection policy
/// we return a `400 invalid_request_error`
/// (`code: "unsupported_parameter"`, `param: <field>`) whenever the
/// caller sets a value the gateway would otherwise silently ignore.
///
/// `honored_default` names the one value (if any) that is a no-op for
/// us and therefore accepted:
/// * `temperature` → `Some(1.0)` (OpenAI's default sampling temperature
///   is a no-op relative to "no sampling control");
/// * `n` → `Some(1.0)` (the single-choice envelope only ever produces
///   one completion, which is also OpenAI's default);
/// * `max_tokens` → `None` (there is no value we can honour — we cannot
///   cap generation at all — so any explicit `max_tokens` is rejected).
// The honored defaults (`1.0`) are exactly representable, so the direct
// equality check is intentional and correct here.
#[allow(clippy::float_cmp)]
fn reject_unsupported_knob(
    field: &'static str,
    value: Option<f64>,
    honored_default: Option<f64>,
) -> Result<(), OpenAiError> {
    let Some(v) = value else {
        return Ok(());
    };
    if Some(v) == honored_default {
        return Ok(());
    }
    Err(OpenAiError::invalid_request(
        format!(
            "`{field}` is not supported by this gateway: the underlying executor exposes no \
             sampling controls. Omit `{field}` and retry.",
        ),
        Some(field.to_string()),
    )
    .with_code("unsupported_parameter"))
}

/// Reject a boolean knob (`echo`) when set to a non-default `true`.
/// `false` / absent is the default and translates cleanly.
fn reject_unsupported_bool(field: &'static str, value: Option<bool>) -> Result<(), OpenAiError> {
    if value == Some(true) {
        return Err(OpenAiError::invalid_request(
            format!(
                "`{field}` is not supported by this gateway: the underlying executor cannot \
                 echo the prompt. Omit `{field}` (or send `{field}: false`) and retry.",
            ),
            Some(field.to_string()),
        )
        .with_code("unsupported_parameter"));
    }
    Ok(())
}

/// Reject a non-empty `tools` array. The executor has no tool-calling
/// dispatch, so a populated `tools` field would be silently ignored;
/// an absent / null / empty-array value translates cleanly.
fn reject_unsupported_tools(tools: Option<&serde_json::Value>) -> Result<(), OpenAiError> {
    let populated = match tools {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Array(a)) => !a.is_empty(),
        // Any other non-null shape counts as "the caller set it".
        Some(_) => true,
    };
    if populated {
        return Err(OpenAiError::invalid_request(
            "`tools` is not supported by this gateway: the underlying executor has no \
             tool-calling dispatch. Omit `tools` and retry."
                .to_string(),
            Some("tools".to_string()),
        )
        .with_code("unsupported_parameter"));
    }
    Ok(())
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
        // v0.4 args policy: empty vec (the standard `_start () -> ()`
        // guest links via the typed fast path; the executor has no
        // channel to deliver the prompt bytes), length preserved in hint.
        assert!(out.args.is_empty(), "args must be empty in v0.4");
        assert_eq!(out.prompt_len_hint, 5);
    }

    #[test]
    fn translate_completions_rejects_unsupported_knobs() {
        let mut map = HashMap::new();
        map.insert("m1".to_owned(), fixture_uuid_1());
        let base = CompletionsRequest {
            model: "m1".to_owned(),
            prompt: json!("hi"),
            ..Default::default()
        };

        // max_tokens set → 400 unsupported_parameter, param names the field.
        let req = CompletionsRequest {
            max_tokens: Some(64),
            ..base.clone()
        };
        let err = translate_completions_request(&req, &map).expect_err("rejects max_tokens");
        assert_eq!(err.error.code.as_deref(), Some("unsupported_parameter"));
        assert_eq!(err.error.param.as_deref(), Some("max_tokens"));

        // temperature set → rejected.
        let req = CompletionsRequest {
            temperature: Some(0.7),
            ..base.clone()
        };
        assert!(translate_completions_request(&req, &map).is_err());

        // echo: true → rejected; echo: false is the default and passes.
        let req = CompletionsRequest {
            echo: Some(true),
            ..base.clone()
        };
        assert!(translate_completions_request(&req, &map).is_err());
        let req = CompletionsRequest {
            echo: Some(false),
            ..base.clone()
        };
        assert!(translate_completions_request(&req, &map).is_ok());

        // n: 1 is the honoured default; n: 2 is rejected.
        let req = CompletionsRequest {
            n: Some(1),
            ..base.clone()
        };
        assert!(translate_completions_request(&req, &map).is_ok());
        let req = CompletionsRequest { n: Some(2), ..base };
        assert!(translate_completions_request(&req, &map).is_err());
    }

    #[test]
    fn translate_chat_completions_rejects_nonempty_tools() {
        let mut map = HashMap::new();
        map.insert("m".to_owned(), fixture_uuid_1());
        let base = ChatCompletionsRequest {
            model: "m".to_owned(),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: json!("hi"),
                name: None,
            }],
            ..Default::default()
        };

        // Populated tools array → 400 unsupported_parameter.
        let req = ChatCompletionsRequest {
            tools: Some(json!([{"type": "function"}])),
            ..base.clone()
        };
        let err = translate_chat_completions_request(&req, &map).expect_err("rejects tools");
        assert_eq!(err.error.code.as_deref(), Some("unsupported_parameter"));
        assert_eq!(err.error.param.as_deref(), Some("tools"));

        // Empty array / null / absent all translate cleanly.
        let req = ChatCompletionsRequest {
            tools: Some(json!([])),
            ..base.clone()
        };
        assert!(translate_chat_completions_request(&req, &map).is_ok());
        let req = ChatCompletionsRequest {
            tools: Some(serde_json::Value::Null),
            ..base
        };
        assert!(translate_chat_completions_request(&req, &map).is_ok());
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
