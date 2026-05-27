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
//! Subcommand implementations live under [`tensor_wasm_cli::cmd`]. Each
//! command parses its arguments via `clap` derive, runs to completion, and
//! returns an [`anyhow::Result`] back to `main` for unified error
//! reporting. The clap top-level types ([`Cli`] / [`Command`]) live in
//! the sibling library (`src/lib.rs`) so the `man` subcommand can walk
//! the command tree from `Cli::command()` and integration tests under
//! `tests/` can call parser-level helpers directly.
//!
//! # Global flags & environment
//!
//! * `--tenant <u64>` — when non-zero, attaches the `X-TensorWasm-Tenant` header to
//!   every outbound request. Defaults to `0` (header omitted; legacy behavior).
//! * `TENSOR_WASM_TOKEN` — if set, sent as `Authorization: Bearer <token>` on every
//!   outbound request. See `docs/CLI.md` for the operator guide.
//! * `TENSOR_WASM_LOG` — `tracing-subscriber` `EnvFilter` directive. Defaults to
//!   `warn`. **Security note:** setting this to `trace` (or enabling
//!   `reqwest=trace` specifically) causes `reqwest` to log outbound request
//!   headers, including the `Authorization: Bearer <token>` header. Do not
//!   enable trace-level logging in production; restrict it to local debugging
//!   against a non-production token.
#![deny(missing_docs)]

use anyhow::Result;
use clap::Parser;

use tensor_wasm_cli::{cmd, Cli, Command};

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
        Command::Serve(args) => cmd::serve::run(args).await,
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
