// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm run` — locally execute a Wasm module.
//!
//! Spawns a `TensorWasmExecutor` over a default `TensorWasmEngine`, instantiates the
//! supplied `.wasm` file under `TenantId(1)`, invokes the requested export
//! (default `main`), and prints the export's result list as JSON. Failures
//! surface through the standard `anyhow` chained-cause stack and the process
//! exits non-zero.
//!
//! Arguments destined for the guest are accepted as a JSON array via
//! `--args`. Each element is converted into a [`WasmArg`] and threaded
//! through to the executor's dynamic `call_export_with_args` path so the
//! values actually reach the guest — see `crates/tensor-wasm-exec/src/executor.rs`
//! for the parameter-marshalling rules.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};
use clap::Args;

/// Arguments to `tensor-wasm run`.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to the `.wasm` file to execute.
    pub file: PathBuf,
    /// Name of the exported function to call. Defaults to `main`.
    #[arg(long, default_value = "main")]
    pub export: String,
    /// Arguments to pass to the export, encoded as a JSON array.
    ///
    /// Example: `--args '[1, 2.0]'`. Each element is parsed into a
    /// [`WasmArg`] — integers that fit in `i32` become `I32`, larger
    /// integers become `I64`, and non-integer numerics become `F64`. The
    /// values are forwarded to the executor via the dynamic
    /// `call_export_with_args` path; the export's signature must accept
    /// the resulting parameter types or wasmtime returns an error.
    #[arg(long)]
    pub args: Option<String>,
}

/// Entry point for `tensor-wasm run`.
pub async fn run(args: RunArgs) -> Result<()> {
    let wasm_args: Vec<WasmArg> = match &args.args {
        Some(json) => {
            let parsed: serde_json::Value = serde_json::from_str(json)
                .with_context(|| format!("--args value is not valid JSON: {json}"))?;
            let array = match parsed.as_array() {
                Some(a) => a,
                None => anyhow::bail!("--args must be a JSON array, got {}", parsed),
            };
            // Convert each element with full context on failure so the
            // user knows which index tripped the parse. The error from
            // `WasmArg::from_json` is a `&'static str`, suitable for
            // direct interpolation.
            array
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    WasmArg::from_json(v)
                        .map_err(|msg| anyhow::anyhow!("--args[{i}]: {msg} (value: {v})"))
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => Vec::new(),
    };

    let wasm = std::fs::read(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;

    let engine = Arc::new(TensorWasmEngine::new().context("constructing TensorWasmEngine")?);
    let executor = TensorWasmExecutor::new(engine);

    let id = executor
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .context("spawning instance")?;

    let call_result = executor
        .call_export_with_args(id, &args.export, &wasm_args)
        .await;
    let _ = executor.terminate(id).await;
    let value = call_result.with_context(|| format!("calling export `{}`", args.export))?;

    // Render the export's result list. Empty arrays — common because most
    // legacy fixtures (and the cli_smoke test) export `() -> ()` — collapse
    // to the literal `ok` for stable scripting; non-empty arrays print as
    // compact JSON, and a single-element array unwraps to that scalar so
    // `(i32, i32) -> i32` adders print `3` rather than `[3]`.
    match &value {
        serde_json::Value::Array(items) if items.is_empty() => {
            println!("ok");
        }
        serde_json::Value::Array(items) if items.len() == 1 => {
            println!("{}", items[0]);
        }
        other => {
            println!("{other}");
        }
    }
    Ok(())
}
