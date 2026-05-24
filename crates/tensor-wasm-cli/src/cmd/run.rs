// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm run` — locally execute a Wasm module.
//!
//! Spawns a `TensorWasmExecutor` over a default `TensorWasmEngine`, instantiates the
//! supplied `.wasm` file under `TenantId(1)`, invokes the requested export
//! (default `main`), and prints `ok` on success or the error chain otherwise.
//!
//! Arguments destined for the guest are accepted as a JSON array via
//! `--args`; today the executor's `call_export` only supports the `() -> ()`
//! signature, so non-empty argument lists are validated then ignored (the
//! richer call path lands in S20 alongside the HTTP transport).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{TensorWasmExecutor, SpawnConfig};
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
    /// Example: `--args '[1.0, 2.0]'`. The current executor only invokes
    /// `() -> ()` exports, so the array is parsed and validated but the
    /// values are not yet wired into the call site.
    #[arg(long)]
    pub args: Option<String>,
}

/// Entry point for `tensor-wasm run`.
pub async fn run(args: RunArgs) -> Result<()> {
    if let Some(json) = &args.args {
        // Validate that --args is well-formed JSON; surface parse errors early.
        let parsed: serde_json::Value = serde_json::from_str(json)
            .with_context(|| format!("--args value is not valid JSON: {json}"))?;
        if !parsed.is_array() {
            anyhow::bail!("--args must be a JSON array, got {}", parsed);
        }
    }

    let wasm = std::fs::read(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;

    let engine = Arc::new(TensorWasmEngine::new().context("constructing TensorWasmEngine")?);
    let executor = TensorWasmExecutor::new(engine);

    let id = executor
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .context("spawning instance")?;

    let call_result = executor.call_export(id, &args.export).await;
    let _ = executor.terminate(id).await;
    call_result.with_context(|| format!("calling export `{}`", args.export))?;

    println!("ok");
    Ok(())
}
