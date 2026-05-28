// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Verifies that a guest hitting `unreachable` (wasm-spec trap) returns a
//! proper error from `call_export`, the executor's registry is NOT poisoned,
//! and a fresh subsequent spawn/call cycle still succeeds. This is the
//! "DashMap not poisoned by panicking guest" guarantee from the audit.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

fn trapping_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (func (export "boom") unreachable)
          (func (export "noop"))
        )
        "#,
    )
    .expect("wat")
}

#[tokio::test]
async fn trapping_guest_does_not_poison_registry() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trapping_wasm())
        .await
        .expect("spawn");

    let res = exec.call_export_with_args(id, "boom", &[]).await;
    assert!(res.is_err(), "unreachable trap must surface as Err");

    // The instance is still in the registry — the trap doesn't auto-terminate.
    // Explicit cleanup must still succeed.
    exec.terminate(id).await.expect("terminate after trap");
    assert_eq!(exec.live_count(), 0);

    // And a fresh spawn after the trap works end-to-end.
    let id2 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(2)), &trapping_wasm())
        .await
        .expect("spawn after trap");
    exec.call_export_with_args(id2, "noop", &[])
        .await
        .expect("noop after trap");
    exec.terminate(id2).await.expect("terminate second");
}
