// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Admission-control regression tests for
//! [`EngineConfig::max_instances`] (exec S-10).
//!
//! Two invariants are pinned here:
//!   1. With `max_instances = Some(2)`, the third concurrent spawn is
//!      refused with the typed [`ExecError::CapacityExhausted`] rather than
//!      an opaque wasmtime error — and the rejection happens at admission
//!      time, before any compile / instantiate work.
//!   2. A *failed* spawn (invalid wasm) must roll its admission slot back so
//!      the cap is not silently leaked. We charge the slot before compiling,
//!      so a compile failure that did not roll back would erode the cap one
//!      slot per failed spawn — a tenant could deny service to themselves by
//!      submitting garbage in a loop. After a failed spawn a fresh *valid*
//!      spawn must still be admitted.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// Trivial valid module with a single no-op export.
fn trivial_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).expect("valid wat")
}

fn make_executor(max_instances: usize) -> TensorWasmExecutor {
    let cfg = EngineConfig {
        max_instances: Some(max_instances),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    TensorWasmExecutor::new(engine)
}

#[tokio::test]
async fn third_spawn_exceeds_cap_with_typed_error() {
    let exec = make_executor(2);
    let wasm = trivial_wasm();

    let id1 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("first spawn admitted");
    let id2 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("second spawn admitted");
    assert_eq!(exec.instances_len(), 2, "two slots charged");
    assert_eq!(exec.live_count(), 2, "two instances registered");

    // Third spawn must be refused at admission time.
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect_err("third spawn must be refused at the cap");
    match err {
        ExecError::CapacityExhausted { active, limit } => {
            assert_eq!(limit, 2);
            assert!(active > limit, "active={active} must exceed limit={limit}");
        }
        other => panic!("expected CapacityExhausted, got {other:?}"),
    }

    // The refused spawn must not have leaked a slot.
    assert_eq!(
        exec.instances_len(),
        2,
        "refused spawn must not consume a slot",
    );

    // Terminating one frees a slot, re-opening admission.
    exec.terminate(id1).await.expect("terminate id1");
    assert_eq!(exec.instances_len(), 1, "slot freed after terminate");
    let id3 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("spawn admitted again after a slot was freed");

    exec.terminate(id2).await.expect("terminate id2");
    exec.terminate(id3).await.expect("terminate id3");
    assert_eq!(exec.instances_len(), 0);
}

#[tokio::test]
async fn failed_spawn_rolls_back_slot_so_cap_not_leaked() {
    // Cap of 2. Spawn one valid instance (1 slot used), then trip a FAILED
    // spawn with invalid wasm. The failure must roll the charged slot back
    // so a subsequent valid spawn is still admitted under the cap.
    let exec = make_executor(2);
    let wasm = trivial_wasm();

    let id1 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("first valid spawn admitted");
    assert_eq!(exec.instances_len(), 1);

    // Invalid wasm — the magic-byte header is wrong, so compilation fails
    // *after* the admission slot is charged. The slot guard must roll the
    // charge back on the error path.
    let garbage: &[u8] = b"\x00not-a-wasm-module";
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), garbage)
        .await
        .expect_err("invalid wasm must fail to spawn");
    assert!(
        matches!(err, ExecError::Wasmtime(_)),
        "expected a wasmtime compile error, got {err:?}",
    );

    // The failed spawn must have released its slot — back to exactly 1.
    assert_eq!(
        exec.instances_len(),
        1,
        "failed spawn must roll back its admission slot",
    );

    // Proof the cap was not leaked: a SECOND valid spawn (filling the last
    // slot) must still be admitted. If the failed spawn had leaked a slot,
    // `instances_len` would be 2 here and this spawn would be refused.
    let id2 = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(2)), &wasm)
        .await
        .expect("valid spawn after a failed one must still be admitted");
    assert_eq!(exec.instances_len(), 2);

    exec.terminate(id1).await.expect("terminate id1");
    exec.terminate(id2).await.expect("terminate id2");
    assert_eq!(exec.instances_len(), 0);
}
