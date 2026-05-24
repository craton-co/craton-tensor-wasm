//! `bali metrics` — fetch Prometheus metrics from a Bali server.
//!
//! Issues `GET {server}/metrics` and prints the response body verbatim.
//! Prometheus text exposition is plain UTF-8 by design, so no extra
//! formatting is applied. A non-2xx response is rendered through the shared
//! error-envelope helper and the process exits non-zero.

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;

/// Arguments to `bali metrics`.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
}

/// Entry point for `bali metrics`.
pub async fn run(args: MetricsArgs) -> Result<()> {
    super::validate_server_url(&args.server)?;

    let url = format!("{}/metrics", super::server_base(&args.server));

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("building HTTP client")?;

    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .with_context(|| format!("reading response body from {url}"))?;

    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    println!("{text}");
    Ok(())
}
