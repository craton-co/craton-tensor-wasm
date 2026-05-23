//! Project Bali developer CLI.
//!
//! The `bali` binary exposes the developer-facing surface of the Bali
//! runtime: local execution (`run`), remote deployment (`deploy`), invocation
//! of deployed functions (`invoke`), latency benchmarking (`bench`), snapshot
//! save/restore (`snapshot`), Prometheus scraping (`metrics`), and shell
//! completion (`completions`).
//!
//! Subcommand implementations live under [`cmd`]. Each command parses its
//! arguments via `clap` derive, runs to completion, and returns an
//! [`anyhow::Result`] back to `main` for unified error reporting.
#![warn(missing_docs)]

use anyhow::Result;
use clap::{Parser, Subcommand};
use clap_complete::Shell;

mod cmd;

/// Project Bali — GPU-accelerated serverless Wasm runtime CLI.
#[derive(Debug, Parser)]
#[command(
    name = "bali",
    bin_name = "bali",
    version,
    about = "Developer CLI for Project Bali",
    long_about = "Run, deploy, invoke, bench, snapshot, and inspect Bali Wasm workloads."
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level `bali` subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a Wasm module locally against the in-process Bali engine.
    Run(cmd::run::RunArgs),
    /// Upload a Wasm module to a Bali server.
    Deploy(cmd::deploy::DeployArgs),
    /// Invoke a previously deployed function by id.
    Invoke(cmd::invoke::InvokeArgs),
    /// Benchmark local invocation latency (P50/P95/P99/max).
    Bench(cmd::bench::BenchArgs),
    /// Save or restore an instance snapshot.
    Snapshot {
        /// Snapshot sub-action.
        #[command(subcommand)]
        action: cmd::snapshot::SnapshotAction,
    },
    /// Fetch and pretty-print Prometheus metrics from a Bali server.
    Metrics(cmd::metrics::MetricsArgs),
    /// Emit shell completion scripts for the named shell.
    Completions {
        /// Target shell (bash, zsh, fish, powershell, elvish).
        shell: Shell,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Best-effort tracing setup — failures are non-fatal (e.g., a subscriber
    // is already installed by an embedding harness).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd::run::run(args).await,
        Command::Deploy(args) => cmd::deploy::run(args),
        Command::Invoke(args) => cmd::invoke::run(args),
        Command::Bench(args) => cmd::bench::run(args).await,
        Command::Snapshot { action } => cmd::snapshot::run(action),
        Command::Metrics(args) => cmd::metrics::run(args),
        Command::Completions { shell } => cmd::completions::run(shell),
    }
}
