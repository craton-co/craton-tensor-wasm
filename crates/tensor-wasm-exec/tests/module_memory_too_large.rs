// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pre-instantiation memory cap regression test (mem-H5 / exec-S-2 / exec-S-10).
//!
//! Wasmtime's [`wasmtime::ResourceLimiter::memory_growing`] fires only on
//! `memory.grow`, not on the initial allocation declared by `(memory N M)`.
//! Before the fix, a guest module declaring `(memory 65536)` (4 GiB initial
//! linear memory) was instantiated unconditionally — the per-store limiter
//! never saw the allocation, so the engine cap was silently bypassed and the
//! host was forced to back the full declaration. The executor now walks every
//! exported AND imported memory type up-front and rejects the spawn with
//! [`tensor_wasm_exec::executor::ExecError::ModuleMemoryTooLarge`] before
//! `Instance::new_async` runs.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, MemoryBackend, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// 256 MiB engine cap — chosen so the 4 GiB declaration below trips it
/// by a wide margin and the failure mode is unambiguous.
const ENGINE_CAP: usize = 256 * 1024 * 1024;

/// 4 GiB declared (65 536 pages × 64 KiB). This is the maximum a wasm32
/// linear memory can declare per spec, so the module itself is valid —
/// the rejection comes purely from the host's cap, not from a parse error.
const FOUR_GIB_PAGES: u32 = 65_536;

fn four_gib_module() -> Vec<u8> {
    // `(memory 65536)` declares an initial size of 65 536 pages = 4 GiB,
    // with no maximum. Pre-fix, this would allocate 4 GiB at instantiation
    // because `ResourceLimiter::memory_growing` does not fire on initial
    // allocations.
    let src = format!("(module (memory (export \"mem\") {FOUR_GIB_PAGES}))");
    wat::parse_str(&src).expect("valid wat")
}

fn four_gib_module_with_max() -> Vec<u8> {
    // Variant: declare a tiny initial (1 page) and a 4 GiB maximum. The
    // initial allocation would slip past a min-only check; the maximum is
    // what trips the cap. Confirms BOTH minimum and maximum are policed.
    let src = format!("(module (memory (export \"mem\") 1 {FOUR_GIB_PAGES}))");
    wat::parse_str(&src).expect("valid wat")
}

fn imported_huge_memory_module() -> Vec<u8> {
    // Variant: the offending memory is *imported*, not exported. Pre-fix
    // we only would have inspected the instance after instantiation;
    // the host was still on the hook to honour the declared maximum.
    let src = format!("(module (import \"env\" \"mem\" (memory {FOUR_GIB_PAGES})))",);
    wat::parse_str(&src).expect("valid wat")
}

fn make_executor() -> TensorWasmExecutor {
    let cfg = EngineConfig {
        max_memory_bytes: ENGINE_CAP,
        // Use the pooling backend so the test does not depend on the
        // GPU-backed `TensorWasmMemoryCreator` path. The cap check runs
        // before any backend allocator is consulted, so the choice of
        // backend does not change the outcome — pooling just keeps the
        // test deterministic on hosts without CUDA.
        backend: MemoryBackend::PoolingMpk {
            max_memories: 4,
            memory_bytes: ENGINE_CAP,
        },
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    TensorWasmExecutor::new(engine)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_rejects_exported_4gib_memory_min() {
    let exec = make_executor();
    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &four_gib_module())
        .await
        .expect_err("4 GiB initial memory must be rejected before instantiation");
    match err {
        ExecError::ModuleMemoryTooLarge {
            requested_bytes,
            limit_bytes,
        } => {
            assert_eq!(limit_bytes as usize, ENGINE_CAP);
            assert!(
                requested_bytes as usize > ENGINE_CAP,
                "requested {requested_bytes} bytes must exceed cap {ENGINE_CAP}"
            );
            // 65 536 pages × 64 KiB = 4 GiB exactly.
            assert_eq!(requested_bytes, 4 * 1024 * 1024 * 1024);
        }
        other => panic!("expected ModuleMemoryTooLarge, got {other:?}"),
    }
    assert_eq!(
        exec.live_count(),
        0,
        "no instance must have been created when the cap is tripped",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_rejects_exported_4gib_memory_max() {
    let exec = make_executor();
    let err = exec
        .spawn_instance(
            SpawnConfig::for_tenant(TenantId(1)),
            &four_gib_module_with_max(),
        )
        .await
        .expect_err("4 GiB declared maximum must be rejected");
    assert!(matches!(err, ExecError::ModuleMemoryTooLarge { .. }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_rejects_imported_huge_memory() {
    let exec = make_executor();
    let err = exec
        .spawn_instance(
            SpawnConfig::for_tenant(TenantId(1)),
            &imported_huge_memory_module(),
        )
        .await
        .expect_err("imported 4 GiB memory must be rejected at the import site");
    assert!(matches!(err, ExecError::ModuleMemoryTooLarge { .. }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_admits_module_under_cap() {
    // Sanity counterpart: a 1-page export under the cap must succeed.
    // Catches regressions that would over-tighten the check (e.g. cap
    // comparison off by one, or page-size mismatch).
    let exec = make_executor();
    let small_wasm = wat::parse_str(r#"(module (memory (export "mem") 1))"#).unwrap();
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(2)), &small_wasm)
        .await
        .expect("1-page module must spawn under any reasonable engine cap");
    exec.terminate(id).await.expect("terminate");
}

#[test]
fn module_memory_too_large_maps_to_memory_exhausted() {
    use tensor_wasm_core::error::TensorWasmError;
    let e = ExecError::ModuleMemoryTooLarge {
        requested_bytes: 4 * 1024 * 1024 * 1024,
        limit_bytes: ENGINE_CAP as u64,
    };
    let b: TensorWasmError = e.into();
    match b {
        TensorWasmError::MemoryExhausted { requested, limit } => {
            assert_eq!(requested, 4 * 1024 * 1024 * 1024);
            assert_eq!(limit, ENGINE_CAP as u64);
        }
        other => panic!("expected MemoryExhausted, got {other:?}"),
    }
}
