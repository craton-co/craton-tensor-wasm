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
//! * `--ca-cert <PATH>` — trust an additional PEM-encoded private CA root for
//!   outbound HTTPS (added alongside the system trust store, not instead of
//!   it). Use this for internal/self-signed CAs rather than `--insecure`.
//! * `--insecure` — disable TLS certificate verification entirely. **Security
//!   note:** this exposes the connection to man-in-the-middle attacks and can
//!   leak the `TENSOR_WASM_TOKEN` bearer credential; a warning is logged on
//!   every invocation. Intended only for local dev against a throwaway cert.
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
        // Diagnostics/logs go to stderr so stdout stays clean for command
        // output — critical for `--output json`, where any log line on stdout
        // would corrupt the machine-readable payload. (`fmt()` otherwise
        // defaults to stdout.)
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("TENSOR_WASM_LOG")
                .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let cli = Cli::parse();

    // Resolve the global connection settings, applying the `--config` file
    // (cli feature F6) as a *defaults* source: explicit CLI flags and the
    // `TENSOR_WASM_TOKEN` env var always win, the file only fills gaps. We do
    // this in a fallible helper so a bad config / token file surfaces as a
    // clean `Error:` line rather than a panic.
    let result: Result<()> = run_cli(cli).await;

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

/// Resolve global settings (config file, token precedence, TLS, timeout),
/// build the [`cmd::HttpContext`], and dispatch the subcommand. Split out of
/// `main` so config/token-file parsing errors flow through the same unified
/// error reporter as command failures.
async fn run_cli(cli: Cli) -> Result<()> {
    use std::time::Duration;

    // Load the optional config file first; its values are defaults only.
    let config = match &cli.config {
        Some(path) => cmd::config::CliConfig::load(path)?,
        None => cmd::config::CliConfig::default(),
    };

    // Tenant precedence: an explicit non-zero `--tenant` wins; otherwise fall
    // back to the config value; otherwise 0 (header suppressed). `--tenant`
    // defaults to 0, so we cannot distinguish "operator passed --tenant 0"
    // from "operator omitted it" — treating 0 as "unset, use config" matches
    // the documented "0 == suppress header" semantics either way.
    let tenant = if cli.tenant != 0 {
        cli.tenant
    } else {
        config.tenant.unwrap_or(0)
    };

    // TLS precedence: `--ca-cert` flag wins over config; `--insecure` flag
    // (a bool that defaults false) is OR'd with the config value so either
    // source enabling it takes effect.
    let tls = cmd::TlsOptions {
        ca_cert: cli.ca_cert.clone().or_else(|| config.ca_cert.clone()),
        insecure: cli.insecure || config.insecure.unwrap_or(false),
    };

    // Token precedence: `--token-file` flag > `TENSOR_WASM_TOKEN` env var >
    // config `token_file`. The first source that yields a token wins.
    let token = if let Some(path) = &cli.token_file {
        Some(cmd::load_token_file(path)?)
    } else if let Some(t) = cmd::token_from_env() {
        Some(t)
    } else if let Some(path) = &config.token_file {
        Some(cmd::load_token_file(path)?)
    } else {
        None
    };

    // Timeout precedence: `--timeout` flag > config `timeout_secs`. Validate
    // it is >= 1 so a `--timeout 0` (which reqwest treats as "no timeout")
    // can't silently disable the bound the operator thought they set.
    let timeout_secs = cli.timeout.or(config.timeout_secs);
    let timeout_override = match timeout_secs {
        Some(0) => anyhow::bail!("--timeout must be at least 1 second (got 0)"),
        Some(s) => Some(Duration::from_secs(s)),
        None => None,
    };

    let ctx = cmd::HttpContext::from_parts(token, tenant, tls, timeout_override);

    match cli.command {
        Command::Run(args) => cmd::run::run(args).await,
        Command::Deploy(args) => cmd::deploy::run(args, &ctx).await,
        Command::Invoke(args) => cmd::invoke::run(args, &ctx).await,
        Command::Bench(args) => cmd::bench::run(args).await,
        Command::Snapshot { action } => cmd::snapshot::run(action, &ctx).await,
        Command::Kernel { action } => cmd::kernel::run(action, &ctx).await,
        Command::Metrics(args) => cmd::metrics::run(args, &ctx).await,
        Command::Observe(args) => cmd::observe::run(args, &ctx).await,
        Command::Serve(args) => cmd::serve::run(args).await,
        Command::Completions { shell, out_dir } => cmd::completions::run(shell, out_dir),
        Command::Man(args) => cmd::man::run(args),
    }
}
