//! `bali deploy` — upload a Wasm module to a Bali server.
//!
//! Reads the file, base64-encodes the bytes, and `POST`s
//! `{ "name": ..., "wasm_b64": ... }` to `{server}/functions`. On success the
//! server-assigned identifier is printed to stdout. On any non-2xx response
//! the structured error envelope (`{error: {kind, message}}`) is rendered as
//! a human-readable error and the process exits non-zero.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use clap::Args;
use serde::Serialize;

/// Arguments to `bali deploy`.
#[derive(Debug, Args)]
pub struct DeployArgs {
    /// Path to the `.wasm` file to deploy.
    pub file: PathBuf,
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Tenant-supplied display name. Defaults to the file stem when omitted.
    #[arg(long)]
    pub name: Option<String>,
}

/// JSON request body sent to `POST /functions`.
///
/// Mirrors `bali_api::routes::CreateFunctionRequest`; we redefine it here so
/// the CLI does not need to depend on the server crate just to serialize one
/// payload.
#[derive(Debug, Serialize)]
struct CreateFunctionRequest<'a> {
    name: &'a str,
    wasm_b64: String,
}

/// Entry point for `bali deploy`.
pub async fn run(args: DeployArgs) -> Result<()> {
    let metadata = std::fs::metadata(&args.file)
        .with_context(|| format!("locating wasm file {}", args.file.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{} is not a regular file", args.file.display());
    }
    super::validate_server_url(&args.server)?;

    let bytes = std::fs::read(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;
    let wasm_b64 = BASE64.encode(&bytes);

    let name = match args.name {
        Some(n) => n,
        None => args
            .file
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "function".to_string()),
    };

    let url = format!("{}/functions", super::server_base(&args.server));
    let body = CreateFunctionRequest {
        name: &name,
        wasm_b64,
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("building HTTP client")?;

    let resp = client
        .post(&url)
        .json(&body)
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

    // The server response is `{"id": "<uuid>"}` per `bali-api`. Some older
    // call-sites used `function_id`; honour either for forward compatibility.
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing JSON response from {url}: {text}"))?;
    let id = parsed
        .get("id")
        .or_else(|| parsed.get("function_id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("response missing `id` field: {text}"))?;

    println!("{id}");
    Ok(())
}
