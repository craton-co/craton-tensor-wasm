// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pre-compile size-cap regression test.
//!
//! Wasm modules larger than
//! [`tensor_wasm_exec::executor::MAX_MODULE_BYTES`] must be rejected with
//! [`tensor_wasm_exec::executor::ExecError::ModuleTooLarge`] *before*
//! `Module::from_binary` runs. A pathological code section can otherwise
//! force Cranelift to burn arbitrary CPU on adversarial input — even a
//! blob that ultimately fails to parse can consume the parser's CPU.
//!
//! This test deliberately submits a non-wasm blob (all zeroes) so the
//! assertion is unambiguous: a giant invalid blob must still be rejected
//! at the size gate, before any parse work happens.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor, MAX_MODULE_BYTES};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_rejects_module_above_default_cap() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    // One byte over the default cap. The blob is all zeroes — not a
    // valid wasm module — but the size gate must trip before that
    // matters.
    let oversized = vec![0u8; MAX_MODULE_BYTES + 1];
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &oversized)
        .await
        .expect_err("module above MAX_MODULE_BYTES must be rejected");

    match err {
        ExecError::ModuleTooLarge { len, max } => {
            assert_eq!(len, MAX_MODULE_BYTES + 1);
            assert_eq!(max, MAX_MODULE_BYTES);
        }
        other => panic!("expected ModuleTooLarge, got {other:?}"),
    }

    // The rejected spawn must not have leaked an admission slot.
    assert_eq!(
        exec.instances_len(),
        0,
        "failed spawn must roll back its admission slot",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_rejects_module_above_configured_cap() {
    // Tighter operator policy: 4 KiB cap. The default constant is the
    // floor (64 MiB) — embedders can shrink the budget further, and
    // this test pins that behaviour. A 5 KiB blob must trip the cap
    // even though it's well below `MAX_MODULE_BYTES`.
    let tight_cap = 4 * 1024;
    let cfg = EngineConfig {
        max_module_bytes: tight_cap,
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let oversized = vec![0u8; tight_cap + 1];
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(2)), &oversized)
        .await
        .expect_err("module above configured cap must be rejected");

    match err {
        ExecError::ModuleTooLarge { len, max } => {
            assert_eq!(len, tight_cap + 1);
            assert_eq!(max, tight_cap);
        }
        other => panic!("expected ModuleTooLarge, got {other:?}"),
    }
}

#[test]
fn module_too_large_maps_to_memory_exhausted() {
    use tensor_wasm_core::error::TensorWasmError;
    let e = ExecError::ModuleTooLarge {
        len: MAX_MODULE_BYTES + 1,
        max: MAX_MODULE_BYTES,
    };
    let b: TensorWasmError = e.into();
    match b {
        TensorWasmError::MemoryExhausted { requested, limit } => {
            assert_eq!(requested, (MAX_MODULE_BYTES + 1) as u64);
            assert_eq!(limit, MAX_MODULE_BYTES as u64);
        }
        other => panic!("expected MemoryExhausted, got {other:?}"),
    }
}
