// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for the Wasm proposal deny-list pinned in
//! [`tensor_wasm_exec::engine::TensorWasmEngine::with_config`].
//!
//! The hardened multi-tenant sandbox depends on the engine refusing
//! proposals we have not audited. If a future `wasmtime` bump silently
//! flips the default for one of those proposals, the test below will
//! start passing wasm bytes that our trust model never sanctioned — so
//! we assert the engine actively rejects a module declaring an
//! unsupported proposal (here: `memory64`).

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// `(memory i64 1)` is the textual form of a `memory64` declaration.
/// With `wasm_memory64(false)` pinned in the engine, compiling/instantiating
/// the resulting module must fail at the wasmtime parser/validator boundary
/// — never silently succeed.
#[tokio::test]
async fn memory64_module_is_rejected() {
    let wat = r#"(module (memory i64 1))"#;
    let wasm = wat::parse_str(wat).expect("wat assembles fine — the bytes are syntactically valid; we expect wasmtime to reject them based on the pinned proposal flags");

    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect_err(
            "memory64 must be rejected because wasm_memory64(false) is \
             pinned in TensorWasmEngine::with_config — if this assertion \
             starts failing, the deny-list has regressed and the \
             multi-tenant sandbox trust model is no longer enforced",
        );

    // The exact variant doesn't matter (wasmtime classifies this as a
    // compile/validation error); we only care that the engine refused it.
    match err {
        ExecError::Wasmtime(_) => {}
        other => panic!(
            "expected ExecError::Wasmtime rejecting the memory64 module, \
             got {other:?}"
        ),
    }
}
