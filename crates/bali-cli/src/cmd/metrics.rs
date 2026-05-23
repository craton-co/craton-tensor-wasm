//! `bali metrics` — fetch Prometheus metrics from a Bali server.
//!
//! S18 stub: validates the server URL and prints the planned `GET /metrics`
//! request. S20 will swap the print for a real HTTP fetch via `reqwest` and
//! pretty-print the Prometheus text-format response, optionally filtered to
//! `bali_*` series.

use anyhow::Result;
use clap::Args;

/// Arguments to `bali metrics`.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Base URL of the target Bali server (e.g. `http://localhost:8080`).
    #[arg(long)]
    pub server: String,
}

/// Entry point for `bali metrics`.
pub fn run(args: MetricsArgs) -> Result<()> {
    super::validate_server_url(&args.server)?;
    println!(
        "metrics: would GET {}/metrics",
        args.server.trim_end_matches('/')
    );
    println!("HTTP transport wired in S20 with reqwest");
    Ok(())
}
