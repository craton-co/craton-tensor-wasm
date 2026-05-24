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
//! pointing it at a multi-gigabyte file: see [`MAX_WASM_BYTES`].
//!
//! Auth/tenant headers (`Authorization: Bearer ...`, `X-TensorWasm-Tenant`) are
//! attached by [`crate::cmd::HttpContext`] when configured. See `docs/CLI.md`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use clap::Args;
use serde::Serialize;

use super::HttpContext;

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
        anyhow::bail!(
            "--name must be non-empty / non-whitespace (got `{}`)",
            name
        );
    }

    let bytes = std::fs::read(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;
    let wasm_b64 = BASE64.encode(&bytes);

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
    let text = resp
        .text()
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

    println!("{id}");
    Ok(())
}
