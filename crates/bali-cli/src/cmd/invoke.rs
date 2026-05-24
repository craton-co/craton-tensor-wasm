//! `bali invoke` — call a previously deployed function by id.
//!
//! `POST`s `{server}/functions/{id}/invoke` with a JSON body. When `--args`
//! is supplied it is parsed (and validated to be a JSON array, matching the
//! S18 contract) and forwarded as the request body; otherwise an empty object
//! `{}` is sent.
//!
//! The successful response body is pretty-printed to stdout. A non-2xx
//! response is rendered through the shared error-envelope helper and the
//! process exits non-zero.

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;

/// Arguments to `bali invoke`.
#[derive(Debug, Args)]
pub struct InvokeArgs {
    /// Identifier of the deployed function to invoke.
    pub id: String,
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Arguments forwarded to the function, encoded as a JSON array.
    #[arg(long)]
    pub args: Option<String>,
}

/// Entry point for `bali invoke`.
pub async fn run(args: InvokeArgs) -> Result<()> {
    super::validate_server_url(&args.server)?;

    let body: serde_json::Value = match &args.args {
        Some(json) => {
            let parsed: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| anyhow::anyhow!("--args is not valid JSON: {e}"))?;
            if !parsed.is_array() {
                anyhow::bail!("--args must be a JSON array, got {}", parsed);
            }
            parsed
        }
        None => serde_json::json!({}),
    };

    let url = format!(
        "{}/functions/{}/invoke",
        super::server_base(&args.server),
        args.id
    );

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

    // Pretty-print if the body is JSON; otherwise echo verbatim so the user
    // still sees what the server sent.
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) => match serde_json::to_string_pretty(&v) {
            Ok(pretty) => println!("{pretty}"),
            Err(_) => println!("{text}"),
        },
        Err(_) => println!("{text}"),
    }
    Ok(())
}
