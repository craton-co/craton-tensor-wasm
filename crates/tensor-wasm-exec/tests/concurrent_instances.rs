// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! S7 integration test: 100 concurrent instances complete without deadlock,
//! spurious timeout, or resource leak.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{TensorWasmExecutor, SpawnConfig};

fn noop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_hundred_concurrent_instances() {
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let wasm = noop_wasm();

    let mut handles = Vec::new();
    for tenant in 0u64..100 {
        let exec = exec.clone();
        let wasm = wasm.clone();
        handles.push(tokio::spawn(async move {
            let id = exec
                .spawn_instance(
                    SpawnConfig::for_tenant(TenantId(tenant)).with_deadline(Duration::from_secs(5)),
                    &wasm,
                )
                .await
                .expect("spawn");
            exec.call_export(id, "noop").await.expect("call");
            exec.terminate(id).await.expect("terminate");
        }));
    }

    for h in handles {
        h.await.expect("task joined");
    }

    assert_eq!(exec.live_count(), 0, "all instances should be terminated");
}
