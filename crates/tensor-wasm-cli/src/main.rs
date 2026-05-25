// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Craton TensorWasm developer CLI.
//!
//! The `tensor-wasm` binary exposes the developer-facing surface of the TensorWasm
//! runtime: local execution (`run`), remote deployment (`deploy`), invocation
//! of deployed functions (`invoke`), latency benchmarking (`bench`), snapshot
//! save/restore (`snapshot`), Prometheus scraping (`metrics`), shell
//! completion (`completions`), and man-page generation (`man`).
//!
//! Subcommand implementations live under [`cmd`]. Each command parses its
//! arguments via `clap` derive, runs to completion, and returns an
//! [`anyhow::Result`] back to `main` for unified error reporting.
//!
//! # Global flags & environment
//!
//! * `--tenant <u64>` — when non-zero, attaches the `X-TensorWasm-Tenant` header to
//!   every outbound request. Defaults to `0` (header omitted; legacy behavior).
//! * `TENSOR_WASM_TOKEN` — if set, sent as `Authorization: Bearer <token>` on every
//!   outbound request. See `docs/CLI.md` for the operator guide.
//! * `TENSOR_WASM_LOG` — `tracing-subscriber` `EnvFilter` directive. Defaults to
//!   `warn`.
#![deny(missing_docs)]

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use clap_complete::Shell;

mod cmd;

/// Craton TensorWasm — GPU-accelerated serverless Wasm runtime CLI.
#[derive(Debug, Parser)]
#[command(
    name = "tensor-wasm",
    bin_name = "tensor-wasm",
    version,
    about = "Developer CLI for Craton TensorWasm",
    long_about = "Run, deploy, invoke, bench, snapshot, and inspect TensorWasm Wasm workloads."
)]
pub struct Cli {
    /// Tenant id to advertise on outbound API requests via `X-TensorWasm-Tenant`.
    /// Zero (the default) suppresses the header for backwards compatibility.
    #[arg(long, global = true, default_value_t = 0)]
    pub tenant: u64,

    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level `tensor-wasm` subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a Wasm module locally against the in-process TensorWasm engine.
    Run(cmd::run::RunArgs),
    /// Upload a Wasm module to a TensorWasm server.
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
    /// Fetch and pretty-print Prometheus metrics from a TensorWasm server.
    Metrics(cmd::metrics::MetricsArgs),
    /// Live operator dashboard over `/healthz` + `/metrics` (refreshes in place).
    Observe(cmd::observe::ObserveArgs),
    /// Emit shell completion scripts for the named shell.
    ///
    /// By default the script is written to stdout. Pass `--out-dir <dir>` to
    /// write it to a conventional filename inside `<dir>` instead — used by
    /// `crates/tensor-wasm-cli/completions/` regeneration.
    Completions {
        /// Target shell (bash, zsh, fish, powershell, elvish).
        shell: Shell,
        /// Optional output directory. When provided, the script is written to
        /// `<dir>/<conventional-name>` (e.g. `tensor-wasm.bash`,
        /// `_tensor-wasm` for zsh, `tensor-wasm.fish`).
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Generate roff(7) man pages from the clap command tree.
    Man(cmd::man::ManArgs),
}

#[tokio::main]
async fn main() {
    // Best-effort tracing setup — failures are non-fatal (e.g., a subscriber
    // is already installed by an embedding harness). Honours `TENSOR_WASM_LOG` first,
    // then `RUST_LOG`, then defaults to `warn`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("TENSOR_WASM_LOG")
                .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let cli = Cli::parse();
    let ctx = cmd::HttpContext::from_env(cli.tenant);
    let result: Result<()> = match cli.command {
        Command::Run(args) => cmd::run::run(args).await,
        Command::Deploy(args) => cmd::deploy::run(args, &ctx).await,
        Command::Invoke(args) => cmd::invoke::run(args, &ctx).await,
        Command::Bench(args) => cmd::bench::run(args).await,
        Command::Snapshot { action } => cmd::snapshot::run(action, &ctx).await,
        Command::Metrics(args) => cmd::metrics::run(args, &ctx).await,
        Command::Observe(args) => cmd::observe::run(args, &ctx).await,
        Command::Completions { shell, out_dir } => cmd::completions::run(shell, out_dir),
        Command::Man(args) => cmd::man::run(args),
    };

    if let Err(e) = result {
        // If the error was tagged with a snapshot-specific exit code, honour
        // it so callers can distinguish "feature not yet shipped" (3) from
        // "local validation failed" (2) from generic failures (1).
        let code = e
            .downcast_ref::<cmd::snapshot::SnapshotExit>()
            .map(|s| s.code)
            .unwrap_or(1);
        eprintln!("Error: {e:#}");
        std::process::exit(code);
    }
}
