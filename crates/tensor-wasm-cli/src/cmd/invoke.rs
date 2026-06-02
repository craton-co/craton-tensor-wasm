// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm invoke` — call a previously deployed function by id.
//!
//! `POST`s `{server}/functions/{id}/invoke` with a JSON envelope of the
//! shape `{"export": "<name>", "args": [...]}`. The envelope is the wire
//! contract documented in `docs/CLI.md` and accepted by the server-side
//! `invoke_function` handler in `tensor-wasm-api`. When `--args` is
//! supplied it is parsed (and validated to be a JSON array, matching the
//! S18 contract) and embedded as the `args` field; otherwise `args` is
//! an empty array. `--export` selects the function entry point and
//! defaults to `_start` (which the server-side executor falls back to
//! `main` for, matching the local `tensor-wasm run` convention).
//!
//! The server currently ignores the body contents (api S-31 has the handler
//! deserialise the envelope strictly to limit the DoS surface, but argument
//! passing is not yet wired into the executor), but the wire shape is fixed
//! today so clients written against this CLI keep working once the
//! executor learns to consume `args`.
//!
//! The successful response body is pretty-printed to stdout. A non-2xx
//! response is rendered through the shared error-envelope helper and the
//! process exits non-zero.
//!
//! Auth/tenant headers (`Authorization: Bearer ...`, `X-TensorWasm-Tenant`) are
//! attached by [`crate::cmd::HttpContext`] when configured. See `docs/CLI.md`.

use std::time::Duration;

use anyhow::Result;
use clap::Args;

use super::{HttpContext, OutputFormat};

/// Arguments to `tensor-wasm invoke`.
#[derive(Debug, Args)]
pub struct InvokeArgs {
    /// Identifier of the deployed function to invoke.
    pub id: String,
    /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Name of the exported function to call on the deployed module. Defaults to `_start` (WASI command convention); the server falls back to `main` when `_start` is absent.
    #[arg(long, default_value = "_start")]
    pub export: String,
    /// Arguments forwarded to the function, encoded as a JSON array.
    #[arg(long)]
    pub args: Option<String>,
    /// Output format: `text` (the server response pretty-printed, default) or
    /// `json` (a stable machine-readable envelope wrapping the response for
    /// scripting / CI).
    ///
    /// `display_order` pinned so this sorts after the other local flags and
    /// before the global TLS flags, keeping the help layout stable — matching
    /// the `metrics` / `observe` / `bench` commands.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, display_order = 800)]
    pub output: OutputFormat,
}

/// Entry point for `tensor-wasm invoke`.
pub async fn run(args: InvokeArgs, ctx: &HttpContext) -> Result<()> {
    super::validate_server_url(&args.server)?;

    // sec MEDIUM (parity with `snapshot`/`kernel`): validate the function id
    // against a strict identifier charset BEFORE it is used, for a fast, clear
    // local error and parity with the other id-bearing commands. Percent-
    // encoding (below) already neutralises URL/path smuggling, but the
    // unencoded path silently accepted ids the sibling commands reject (`.`,
    // `..`, arbitrary bytes), so an operator who fat-fingered an id got an
    // opaque server round-trip instead of a clear "invalid character" message.
    //
    // The canonical implementation is `snapshot::validate_identifier_charset`
    // (snapshot.rs), but it is private to that module AND tags its errors with
    // the snapshot-specific `SnapshotExit` exit code, so it cannot be reused
    // here without leaking snapshot semantics into `invoke`. We replicate the
    // identical check (`[A-Za-z0-9._-]`, reject empty and `.`/`..`) with this
    // crate's plain `anyhow` error handling. Keep the two in lockstep.
    validate_identifier_charset(&args.id, "id")?;

    let parsed_args: serde_json::Value = match &args.args {
        Some(json) => {
            let parsed: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| anyhow::anyhow!("--args is not valid JSON: {e}"))?;
            if !parsed.is_array() {
                anyhow::bail!("--args must be a JSON array, got {}", parsed);
            }
            parsed
        }
        None => serde_json::Value::Array(Vec::new()),
    };

    // Wire envelope: `{"export": "<name>", "args": [...]}`. Documented in
    // `docs/CLI.md` and accepted (but currently ignored) by the api crate's
    // `invoke_function` handler. Locking the shape now means clients
    // written against the CLI today keep working once api S-31's argument
    // pass-through lands.
    let body = serde_json::json!({
        "export": args.export,
        "args": parsed_args,
    });

    // sec MEDIUM (URL/path injection): the function id is user-supplied and
    // was previously spliced into the path verbatim, so a value containing
    // `/`, `?`, `#`, `..`, or `%` could reshape the request target (escape
    // the `/functions/{id}/invoke` segment entirely). Percent-encode it as a
    // single path segment so it can only ever land where we intend. We encode
    // with `NON_ALPHANUMERIC`, which covers every reserved/sub-delim and dot
    // character, so `.`/`..` traversal and query/fragment smuggling are all
    // neutralised.
    let encoded_id =
        percent_encoding::utf8_percent_encode(&args.id, percent_encoding::NON_ALPHANUMERIC);
    let url = format!(
        "{}/functions/{}/invoke",
        super::server_base(&args.server),
        encoded_id
    );

    let client = ctx.build_client(Duration::from_secs(60))?;

    let resp = ctx
        .apply(client.post(&url).json(&body))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;

    let status = resp.status();
    // T17: bound the in-memory response body. A 16 MiB cap is far above
    // any legitimate invoke result envelope the API server emits.
    let text = super::bounded_text(resp)
        .await
        .map_err(|e| anyhow::anyhow!("reading response body from {url}: {e}"))?;

    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    match args.output {
        OutputFormat::Text => {
            // Pretty-print if the body is JSON; otherwise echo verbatim so the
            // user still sees what the server sent. T18: every branch outputs
            // server-controlled bytes — sanitise to strip ANSI escapes /
            // control bytes before they hit the terminal. The pretty-printed
            // JSON case is included because string *values* embedded in the
            // JSON envelope are server-controlled and
            // `serde_json::to_string_pretty` will faithfully emit any control
            // byte the server stashed in a string field.
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => match serde_json::to_string_pretty(&v) {
                    Ok(pretty) => println!("{}", super::sanitise_terminal_output(&pretty)),
                    Err(_) => println!("{}", super::sanitise_terminal_output(&text)),
                },
                Err(_) => println!("{}", super::sanitise_terminal_output(&text)),
            }
        }
        OutputFormat::Json => {
            println!("{}", invoke_to_json(&args.id, &args.export, &text));
        }
    }
    Ok(())
}

/// sec MEDIUM (URL/path injection, parity): validate the function `id` against
/// a strict charset before it is used in a request path. Accepts only
/// `[A-Za-z0-9._-]`, rejects the empty string and the traversal tokens `.` /
/// `..`. This mirrors `snapshot::validate_identifier_charset` byte-for-byte;
/// they are kept deliberately identical so every id-bearing command gives the
/// same fast, clear local error. Unlike the snapshot copy, errors here are
/// plain `anyhow` (the snapshot version tags a `SnapshotExit` exit code that
/// has no meaning in `invoke`).
fn validate_identifier_charset(id: &str, flag: &str) -> Result<()> {
    if id.is_empty() {
        anyhow::bail!("{flag} must be non-empty");
    }
    if id == "." || id == ".." {
        anyhow::bail!("{flag} must not be `.` or `..` (path traversal)");
    }
    if let Some(bad) = id
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        anyhow::bail!(
            "{flag} contains an invalid character {bad:?}; only ASCII letters, \
             digits, `.`, `_`, and `-` are allowed"
        );
    }
    Ok(())
}

/// Build the stable `--output json` envelope for an invoke response.
///
/// Shape: `{ "id", "export", "response": <parsed-or-raw> }`. When the server's
/// body parses as JSON it is embedded verbatim as `response`; otherwise the raw
/// (control-byte-sanitised) text is embedded as a JSON string so the document
/// always parses. Rendered compact (`to_string`) so the line stays greppable /
/// pipeable, matching the `bench` / `metrics` / `observe` JSON convention.
fn invoke_to_json(id: &str, export: &str, body: &str) -> String {
    let response = match serde_json::from_str::<serde_json::Value>(body) {
        // T18: even on the JSON path, string *values* are server-controlled,
        // but here the consumer is a machine (jq/CI) rather than a terminal, so
        // we embed the parsed value faithfully and leave de-escaping to the
        // reader. The raw fallback below IS sanitised because it lands as a
        // bare string a human might still eyeball.
        Ok(v) => v,
        Err(_) => serde_json::Value::String(super::sanitise_terminal_output(body)),
    };
    serde_json::json!({
        "id": id,
        "export": export,
        "response": response,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_identifier_charset_accepts_typical_ids() {
        validate_identifier_charset("fn-1", "id").unwrap();
        validate_identifier_charset("My.Func_2", "id").unwrap();
    }

    #[test]
    fn validate_identifier_charset_rejects_empty() {
        assert!(validate_identifier_charset("", "id").is_err());
    }

    #[test]
    fn validate_identifier_charset_rejects_traversal() {
        assert!(validate_identifier_charset(".", "id").is_err());
        assert!(validate_identifier_charset("..", "id").is_err());
    }

    #[test]
    fn validate_identifier_charset_rejects_path_and_control_bytes() {
        assert!(validate_identifier_charset("a/b", "id").is_err());
        assert!(validate_identifier_charset("a%2e", "id").is_err());
        assert!(validate_identifier_charset("a\nb", "id").is_err());
    }

    #[test]
    fn invoke_to_json_embeds_parsed_response() {
        let out = invoke_to_json("fn-1", "_start", r#"{"ok":true,"n":7}"#);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON envelope");
        assert_eq!(v["id"], "fn-1");
        assert_eq!(v["export"], "_start");
        assert_eq!(v["response"]["ok"], true);
        assert_eq!(v["response"]["n"], 7);
    }

    #[test]
    fn invoke_to_json_wraps_non_json_body_as_string() {
        let out = invoke_to_json("fn-1", "main", "plain text result");
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON envelope");
        assert_eq!(v["response"], "plain text result");
    }
}
