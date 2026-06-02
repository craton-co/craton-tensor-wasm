// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm deploy` — upload a Wasm module to a TensorWasm server.
//!
//! Reads the file, base64-encodes the bytes, and `POST`s
//! `{ "name": ..., "wasm_b64": ... }` to `{server}/functions`. On success the
//! server-assigned identifier is printed to stdout. On any non-2xx response
//! the structured error envelope (`{error: {kind, message}}`) is rendered as
//! a human-readable error and the process exits non-zero.
//!
//! The on-wire body is bounded so a tenant can't make the CLI silently OOM by
//! pointing it at a multi-gigabyte file: see `MAX_WASM_BYTES`.
//!
//! Auth/tenant headers (`Authorization: Bearer ...`, `X-TensorWasm-Tenant`) are
//! attached by [`crate::cmd::HttpContext`] when configured. See `docs/CLI.md`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::write::EncoderStringWriter;
use clap::Args;
use serde::Serialize;

use super::{HttpContext, OutputFormat};

/// Chunk size used when streaming the wasm file through the base64 encoder.
/// 64 KiB matches the default `BufReader` capacity and keeps the working set
/// bounded regardless of input size — the alternative (`std::fs::read` →
/// `BASE64.encode(&bytes)`) holds two transient buffers of size N and ~4N/3
/// in memory simultaneously, which at the 64 MiB deploy cap peaks at
/// ~220 MiB and trips OOMs on small CI runners. Picking a multiple of 3
/// avoids partial-group padding inside the inner loop (base64 encodes
/// 3-byte groups into 4-byte output).
const COPY_BUF_BYTES: usize = 64 * 1024 * 3;

/// Largest `.wasm` file the CLI will read in deploy. Keeps the CLI from
/// silently OOM-ing on a misconfigured `--file` path, and lines up with the
/// 64 MiB request-body cap that Batch J is enforcing in the API server.
pub(crate) const MAX_WASM_BYTES: u64 = 64 * 1024 * 1024;

/// Arguments to `tensor-wasm deploy`.
#[derive(Debug, Args)]
pub struct DeployArgs {
    /// Path to the `.wasm` file to deploy.
    pub file: PathBuf,
    /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Tenant-supplied display name. Defaults to the file stem when omitted.
    #[arg(long)]
    pub name: Option<String>,
    /// Output format: `text` (the deployed id printed bare, default) or `json`
    /// (a stable machine-readable envelope carrying the deployed id, for
    /// scripting / CI).
    ///
    /// `display_order` pinned so this sorts after the other local flags and
    /// before the global TLS flags, keeping the help layout stable — matching
    /// the `metrics` / `observe` / `bench` commands.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, display_order = 800)]
    pub output: OutputFormat,
}

/// JSON request body sent to `POST /functions`.
///
/// Mirrors `tensor_wasm_api::routes::CreateFunctionRequest`; we redefine it here so
/// the CLI does not need to depend on the server crate just to serialize one
/// payload.
#[derive(Debug, Serialize)]
struct CreateFunctionRequest<'a> {
    name: &'a str,
    wasm_b64: String,
}

/// Entry point for `tensor-wasm deploy`.
pub async fn run(args: DeployArgs, ctx: &HttpContext) -> Result<()> {
    super::validate_server_url(&args.server)?;

    let metadata = std::fs::metadata(&args.file)
        .with_context(|| format!("locating wasm file {}", args.file.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{} is not a regular file", args.file.display());
    }
    if metadata.len() > MAX_WASM_BYTES {
        anyhow::bail!(
            "wasm file {} is {} bytes; the deploy cap is {} bytes ({} MiB). \
             Strip dead code or raise the server's body limit before retrying.",
            args.file.display(),
            metadata.len(),
            MAX_WASM_BYTES,
            MAX_WASM_BYTES / (1024 * 1024)
        );
    }

    let name = match args.name {
        Some(n) => n,
        None => args
            .file
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "function".to_string()),
    };
    if name.trim().is_empty() {
        anyhow::bail!("--name must be non-empty / non-whitespace (got `{}`)", name);
    }

    // cli fix 2: encode the wasm file into base64 in chunks rather than
    // reading the whole file into a `Vec<u8>` and then base64-ing that into
    // a fresh `String`. The previous shape held two transient buffers at
    // once (raw bytes + base64 string) so at the 64 MiB cap peak memory ran
    // to roughly 64 MiB + 88 MiB ≈ 152 MiB transient, with a brief 220 MiB
    // spike during the encode allocation. Streaming through
    // `EncoderStringWriter` produces a single `String` of the encoded
    // payload (~88 MiB at the cap) and keeps the read buffer bounded to
    // [`COPY_BUF_BYTES`] regardless of input size.
    let wasm_b64 = encode_wasm_streaming(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;

    let url = format!("{}/functions", super::server_base(&args.server));
    let body = CreateFunctionRequest {
        name: &name,
        wasm_b64,
    };

    let client = ctx.build_client(Duration::from_secs(60))?;

    let resp = ctx
        .apply(client.post(&url).json(&body))
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    // T17: bound the in-memory response body. The deploy endpoint emits a
    // short `{"id": "<uuid>"}` ack or a structured error envelope on
    // failure — both well under the 16 MiB cap.
    let text = super::bounded_text(resp)
        .await
        .with_context(|| format!("reading response body from {url}"))?;

    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    // The server response is `{"id": "<uuid>"}` per `tensor-wasm-api`. We deliberately
    // do NOT honour the legacy `function_id` field — early drafts of the API
    // emitted that key but it was never shipped, and accepting it forever
    // would mask a real bug in the response shape if the server regresses.
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing JSON response from {url}: {text}"))?;
    let id = parsed
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("response missing `id` field: {text}"))?;

    match args.output {
        OutputFormat::Text => {
            // T18: `id` was extracted from a server-supplied JSON envelope, so
            // a malicious server can stuff ANSI escapes (or a literal CR) into
            // the string. Sanitise before displaying so the response cannot
            // rewrite the operator's terminal title bar or smuggle in a
            // control byte.
            println!("{}", super::sanitise_terminal_output(id));
        }
        OutputFormat::Json => {
            // Stable envelope `{ "id": "<...>" }`. The id is the same
            // server-supplied string; sanitise it before embedding so a
            // human eyeballing the JSON can't be hit by a smuggled control
            // byte. Rendered compact (`to_string`) so the line stays
            // greppable / pipeable, matching the `bench` / `metrics` /
            // `observe` JSON convention.
            println!("{}", deploy_to_json(id));
        }
    }
    Ok(())
}

/// Build the stable `--output json` envelope for a deploy response.
///
/// Shape: `{ "id": "<deployed-function-id>" }`. The id is sanitised of control
/// bytes before embedding. Rendered compact (`to_string`) to match the JSON
/// convention used by the other scriptable commands.
fn deploy_to_json(id: &str) -> String {
    serde_json::json!({ "id": super::sanitise_terminal_output(id) }).to_string()
}

/// Stream a `.wasm` file off disk through a base64 encoder into a single
/// `String`, reusing one fixed-size read buffer to keep peak memory bounded.
///
/// The output is a complete standard-alphabet, padded base64 string suitable
/// for the `wasm_b64` field of [`CreateFunctionRequest`]. Errors propagate
/// the underlying I/O failure (e.g. `EACCES`, `ENOENT`) and the caller is
/// expected to wrap them with file-path context.
///
/// Implementation note: `EncoderStringWriter` buffers an in-progress 3-byte
/// group internally between `write_all` calls and emits the final padding
/// when `into_inner()` is called, so partial reads (the inner buffer not
/// being a multiple of 3) are handled correctly without us having to align
/// the read boundaries.
fn encode_wasm_streaming(path: &std::path::Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(COPY_BUF_BYTES, file);
    let mut encoder = EncoderStringWriter::new(&BASE64);
    let mut buf = vec![0u8; COPY_BUF_BYTES];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        // `EncoderStringWriter` implements `io::Write`; the `Write::write_all`
        // contract matches what we want — buffer until the encoder has a
        // whole 3-byte group to emit, then push 4 ASCII bytes into the inner
        // `String`. Errors from `write_all` are infallible in practice (the
        // sink is a `String`), but we propagate the `io::Error` shape for
        // honest signatures.
        std::io::Write::write_all(&mut encoder, &buf[..n])?;
    }
    Ok(encoder.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deploy_to_json_carries_id() {
        let out = deploy_to_json("fn-abc123");
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON envelope");
        assert_eq!(v["id"], "fn-abc123");
    }

    #[test]
    fn deploy_to_json_sanitises_control_bytes_in_id() {
        // A malicious server stuffing an ANSI escape into the id must not
        // survive into the JSON envelope a human might eyeball.
        let out = deploy_to_json("fn\x1b[31m1");
        assert!(!out.contains('\x1b'), "ESC byte survived: {out:?}");
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON envelope");
        assert_eq!(v["id"], "fn?[31m1");
    }
}
