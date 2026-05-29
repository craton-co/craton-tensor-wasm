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

use super::{HttpContext, OutputFormat};

/// Arguments to `tensor-wasm metrics`.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
    /// Output format: `text` (raw Prometheus exposition, default) or `json`
    /// (a machine-readable document with the parsed samples for CI scripts).
    ///
    /// `display_order` pinned so this sorts after the other local flags and
    /// before the global TLS flags, keeping the help layout stable.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, display_order = 800)]
    pub output: OutputFormat,
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
    // T17: route the body through `bounded_text` so a malicious server
    // streaming gigabytes of `text/plain` cannot OOM the CLI. The 16 MiB
    // cap is comfortably above any legitimate Prometheus exposition.
    let text = super::bounded_text(resp)
        .await
        .map_err(|e| anyhow::anyhow!("reading response body from {url}: {e}"))?;

    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    match args.output {
        OutputFormat::Text => {
            // T18: the Prometheus text-exposition format is *mostly* US-ASCII,
            // but metric *labels* are server-controlled strings — a malicious
            // server could stash an ANSI escape inside a label value and the
            // verbatim print below would feed it straight to the terminal.
            // Strip control bytes before display; the legitimate Prometheus
            // format does not need them.
            println!("{}", super::sanitise_terminal_output(&text));
        }
        OutputFormat::Json => {
            println!("{}", metrics_to_json(&text));
        }
    }
    Ok(())
}

/// Render a Prometheus text-exposition body as a machine-readable JSON
/// document for CI scripts. Reuses the [`super::observe::parse_metrics`]
/// parser so the JSON view stays consistent with the `observe` dashboard's
/// understanding of the same payload.
///
/// Shape: `{ "samples": [ { "name", "labels": {..}, "value" }, ... ] }`.
/// `value` is emitted as a JSON number; `NaN`/inf values (which serde_json
/// cannot represent) are rendered as the string `"NaN"`/`"inf"` so the
/// document always parses.
fn metrics_to_json(body: &str) -> String {
    let parsed = super::observe::parse_metrics(body);
    let mut samples: Vec<serde_json::Value> = Vec::new();
    // Sort by name for deterministic output (HashMap iteration order is not
    // stable across runs, which would otherwise churn CI snapshots).
    let mut names: Vec<&String> = parsed.keys().collect();
    names.sort();
    for name in names {
        for s in &parsed[name] {
            let value = match serde_json::Number::from_f64(s.value) {
                Some(n) => serde_json::Value::Number(n),
                // NaN / infinite values can't be a JSON number; stringify so
                // the document stays valid.
                None => serde_json::Value::String(format!("{}", s.value)),
            };
            samples.push(serde_json::json!({
                "name": s.name,
                "labels": s.labels,
                "value": value,
            }));
        }
    }
    serde_json::json!({ "samples": samples }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_to_json_is_valid_and_carries_samples() {
        let body = "\
# HELP foo a counter
# TYPE foo counter
foo 7
bar{a=\"1\"} 2.5
";
        let json = metrics_to_json(body);
        let v: serde_json::Value =
            serde_json::from_str(&json).expect("metrics --output json must be valid JSON");
        let samples = v["samples"].as_array().expect("samples array");
        assert_eq!(samples.len(), 2);
        // Sorted by name → `bar` before `foo`.
        assert_eq!(samples[0]["name"], "bar");
        assert_eq!(samples[0]["labels"]["a"], "1");
        assert_eq!(samples[1]["name"], "foo");
        assert_eq!(samples[1]["value"], 7.0);
    }
}
