// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm metrics` — fetch Prometheus metrics from a TensorWasm server.
//!
//! Issues `GET {server}/metrics` and prints the response body verbatim.
//! Prometheus text exposition is plain UTF-8 by design, so no extra
//! formatting is applied. A non-2xx response is rendered through the shared
//! error-envelope helper and the process exits non-zero.
//!
//! Auth/tenant headers (`Authorization: Bearer ...`, `X-TensorWasm-Tenant`) are
//! attached by [`crate::cmd::HttpContext`] when configured. See `docs/CLI.md`.

use std::time::Duration;

use anyhow::Result;
use clap::Args;

use super::HttpContext;

/// Arguments to `tensor-wasm metrics`.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
}

/// Entry point for `tensor-wasm metrics`.
pub async fn run(args: MetricsArgs, ctx: &HttpContext) -> Result<()> {
    super::validate_server_url(&args.server)?;

    let url = format!("{}/metrics", super::server_base(&args.server));

    let client = ctx.build_client(Duration::from_secs(30))?;

    let resp = ctx
        .apply(client.get(&url))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| anyhow::anyhow!("reading response body from {url}: {e}"))?;

    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    println!("{text}");
    Ok(())
}
