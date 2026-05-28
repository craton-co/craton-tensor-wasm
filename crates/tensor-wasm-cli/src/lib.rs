// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Library surface of the `tensor-wasm` CLI.
//!
//! The CLI is delivered as a binary (`src/main.rs`), but a thin library
//! layer also exposes the subcommand modules under [`cmd`] so integration
//! tests under `tests/` can exercise parser-level helpers (URL validation,
//! credential gating, scheme/host extraction) without spawning the full
//! binary. The library is otherwise an implementation detail — external
//! consumers should depend on the `tensor-wasm` binary, not this crate.
//!
//! The [`Cli`] / [`Command`] types are exposed here (rather than living in
//! `main.rs`) because `cmd::man` needs `Cli::command()` to walk the clap
//! tree when rendering man pages, and a lib module cannot reach into a
//! binary crate.
#![deny(missing_docs)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use clap_complete::Shell;

pub mod cmd;

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
    /// Publish, list, or verify entries in the signed kernel registry.
    ///
    /// Roadmap feature #3. The B6.4 milestone wired the server-side
    /// `/kernels` HTTP route, so `publish` and `list` now perform real
    /// HTTP calls against a TensorWasm server built with the
    /// `kernel-registry-api` feature; `verify` is local-only and
    /// re-signs an on-disk manifest under the supplied key. See
    /// `docs/KERNEL-REGISTRY.md` for the manifest schema, signing
    /// envelope, and operator deployment guide.
    Kernel {
        /// Kernel sub-action.
        #[command(subcommand)]
        action: cmd::kernel::KernelAction,
    },
    /// Fetch and pretty-print Prometheus metrics from a TensorWasm server.
    Metrics(cmd::metrics::MetricsArgs),
    /// Live operator dashboard over `/healthz` + `/metrics` (refreshes in place).
    Observe(cmd::observe::ObserveArgs),
    /// Run the TensorWasm HTTP API gateway in-process (binds and serves).
    Serve(cmd::serve::ServeArgs),
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
