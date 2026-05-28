// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T41 coverage for the `messages` → prompt assembly contract.
//!
//! Asserts that `assemble_chat_prompt` and the translator surface the
//! documented role-tagged shape:
//!
//! ```text
//! system: <content>
//! user: <content>
//! assistant: <content>
//! ...
//! assistant:
//! ```
//!
//! Drives the pure translator (no router, no executor) so the test is
//! deterministic and fast.
//!
//! `ChatMessage` / `ChatCompletionsRequest` carry `#[non_exhaustive]`,
//! so this test goes through `serde_json::from_value` to construct
//! them rather than struct literals (FRU is forbidden across crates
//! with non_exhaustive).

use std::collections::HashMap;

use serde_json::{json, Value};
use uuid::Uuid;

use tensor_wasm_api::{
    assemble_chat_prompt, translate_chat_completions_request, ChatCompletionsRequest, ChatMessage,
};

fn msg(role: &str, content: Value) -> ChatMessage {
    serde_json::from_value(json!({ "role": role, "content": content }))
        .expect("ChatMessage shape parses")
}

fn req(model: &str, messages: Vec<ChatMessage>, stream: bool) -> ChatCompletionsRequest {
    let arr: Vec<Value> = messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect();
    serde_json::from_value(json!({
        "model": model,
        "messages": arr,
        "stream": stream,
    }))
    .expect("ChatCompletionsRequest shape parses")
}

#[test]
fn assembled_prompt_carries_system_user_assistant_role_tags() {
    let messages = vec![
        msg("system", json!("You are helpful.")),
        msg("user", json!("Hello!")),
        msg("assistant", json!("Hi there.")),
        msg("user", json!("Tell me a joke.")),
    ];
    let prompt = assemble_chat_prompt(&messages);
    // Every role tag must appear as its own role-prefixed line.
    assert!(
        prompt.contains("system: You are helpful."),
        "missing system tag: {prompt}",
    );
    assert!(
        prompt.contains("user: Hello!"),
        "missing first user tag: {prompt}",
    );
    assert!(
        prompt.contains("assistant: Hi there."),
        "missing assistant tag: {prompt}",
    );
    assert!(
        prompt.contains("user: Tell me a joke."),
        "missing follow-up user tag: {prompt}",
    );
    // Trailing turn marker so the guest can identify "generate the
    // assistant reply now".
    assert!(
        prompt.trim_end().ends_with("assistant:"),
        "must end with trailing `assistant:` marker, got {prompt:?}",
    );
}

#[test]
fn assembled_prompt_preserves_message_order() {
    // Order matters for chat — the guest sees the conversation in
    // insertion order, oldest to newest.
    let messages = vec![
        msg("user", json!("first")),
        msg("user", json!("second")),
        msg("user", json!("third")),
    ];
    let prompt = assemble_chat_prompt(&messages);
    let first_idx = prompt.find("first").expect("first present");
    let second_idx = prompt.find("second").expect("second present");
    let third_idx = prompt.find("third").expect("third present");
    assert!(
        first_idx < second_idx && second_idx < third_idx,
        "messages must appear in insertion order, got {prompt:?}",
    );
}

#[test]
fn translator_assembles_full_prompt_into_translated_request() {
    let mut map = HashMap::new();
    let fid = Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap();
    map.insert("m".to_owned(), fid);
    let req = req(
        "m",
        vec![msg("system", json!("S")), msg("user", json!("U"))],
        false,
    );
    let out = translate_chat_completions_request(&req, &map).expect("translates");
    assert_eq!(out.function_id, fid);
    assert!(
        out.prompt.contains("system: S\n") && out.prompt.contains("user: U\n"),
        "translated prompt missing role-tagged lines: {prompt}",
        prompt = out.prompt,
    );
}

#[test]
fn assembled_prompt_handles_multimodal_content_array_text_only() {
    // OpenAI accepts `content: [{type: "text", text: "..."}, ...]` for
    // multimodal messages. The translator extracts text parts only;
    // image / audio parts are silently dropped for v0.4 (v0.5 lands
    // proper multimodal support per docs/OPENAI-COMPAT.md).
    let messages = vec![msg(
        "user",
        json!([
            { "type": "text", "text": "part1" },
            { "type": "image_url", "image_url": "ignored" },
            { "type": "text", "text": "part2" },
        ]),
    )];
    let prompt = assemble_chat_prompt(&messages);
    assert!(
        prompt.contains("part1"),
        "first text part dropped: {prompt}",
    );
    assert!(prompt.contains("part2"), "second text part dropped: {prompt}");
}

#[test]
fn assembled_prompt_empty_role_defaults_to_user() {
    let messages = vec![msg("", json!("hi"))];
    let prompt = assemble_chat_prompt(&messages);
    assert!(prompt.contains("user: hi"), "empty role must default to user: {prompt}");
}
