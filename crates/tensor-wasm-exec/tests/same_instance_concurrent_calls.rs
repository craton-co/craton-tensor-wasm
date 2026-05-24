// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Verifies that concurrent `call_export` invocations against a *single*
//! instance serialise correctly (wasmtime stores are `!Sync`) without
//! panic or deadlock. Four parallel tasks call the same exported function;
//! all four must complete.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{TensorWasmExecutor, SpawnConfig};

fn noop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).expect("wat")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn four_concurrent_calls_on_one_instance() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &noop_wasm())
        .await
        .expect("spawn");

    let mut handles = Vec::new();
    for _ in 0..4 {
        let exec = exec.clone();
        handles.push(tokio::spawn(async move {
            exec.call_export(id, "noop").await
        }));
    }

    for h in handles {
        h.await.expect("task joined").expect("call ok");
    }

    // The instance is still live and consistent.
    assert_eq!(exec.live_count(), 1);
    exec.terminate(id).await.expect("terminate");
}
