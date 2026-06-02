// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Verifies that invalid wasm bytes surface as `ExecError::Wasmtime` and,
//! after conversion to `TensorWasmError`, are classified as `TensorWasmError::WasmCompile`
//! (not `WasmTrap`) — the `From` impl downcasts to `wasmtime::Trap` to make
//! the distinction. This is the "proper compile error variant" requirement
//! from the audit.

use std::sync::Arc;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

#[tokio::test]
async fn invalid_wasm_returns_wasmtime_variant() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    // Definitely not a wasm module — wrong magic and version.
    let garbage: Vec<u8> = b"not-a-wasm-module-at-all".to_vec();

    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &garbage)
        .await
        .expect_err("garbage bytes must fail to compile");

    // The executor-level variant is the catch-all wasmtime wrapper — the
    // trap-vs-compile classification happens at the `TensorWasmError` boundary.
    match err {
        ExecError::Wasmtime(_) => {}
        other => panic!("expected ExecError::Wasmtime, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_wasm_converts_to_tensor_wasm_compile_error() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    // Valid magic, bad version — wasmtime's parser rejects this before
    // anything resembling instantiation.
    let bad_version: Vec<u8> = b"\0asm\xff\xff\xff\xff".to_vec();
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &bad_version)
        .await
        .expect_err("bad wasm version must fail");

    let tensor_wasm_err: TensorWasmError = err.into();
    match tensor_wasm_err {
        TensorWasmError::WasmCompile(_) => {}
        other => panic!("expected TensorWasmError::WasmCompile, got {other:?}"),
    }
}
