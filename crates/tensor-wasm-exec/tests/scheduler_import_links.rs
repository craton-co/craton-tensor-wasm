// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Wave-1 wiring regression: a guest importing the `wasi:scheduler/host`
//! surface now instantiates through [`TensorWasmExecutor::spawn_instance`].
//!
//! Before the Wave-1 linker work, `spawn_instance` instantiated against an
//! empty import set (`Instance::new_async(.., &[])`), so a guest importing
//! `wasi:scheduler/host@0.1.0` failed to link — the scheduler machinery was
//! dead on the real spawn path. `instantiate_detached` now builds a
//! `Linker<InstanceState>` and unconditionally registers the scheduler host
//! functions (the per-store `SchedulerContext` is always constructed), so a
//! scheduler-importing guest links and runs.
//!
//! This test pins that behaviour end-to-end: a minimal wat importing both
//! scheduler host functions spawns successfully and the host functions are
//! actually reachable from a guest export (an unbounded context returns the
//! CONTINUE code / the u32::MAX remaining-budget sentinel).

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};

/// Minimal guest that imports both `wasi:scheduler/host@0.1.0` functions and
/// re-exports thin wrappers so the host wiring is observable from a call.
///
/// Each export takes (and ignores) one `i32` param so the executor's
/// `call_export_with_args` dynamic path is used — the empty-args fast path
/// is hardwired to a `() -> ()` signature and cannot surface an `i32`
/// result.
const SCHEDULER_IMPORT_WAT: &str = r#"
(module
  (import "wasi:scheduler/host@0.1.0" "yield" (func $yield (result i32)))
  (import "wasi:scheduler/host@0.1.0" "deadline-remaining-ms" (func $rem (result i32)))

  (func (export "do_yield") (param i32) (result i32) (call $yield))
  (func (export "remaining") (param i32) (result i32) (call $rem))
)
"#;

#[tokio::test]
async fn guest_importing_scheduler_host_spawns_and_runs() {
    // `TensorWasmEngine::new` is constructed inside the tokio runtime, so it
    // auto-spawns the epoch ticker — required because every spawn arms the
    // implicit `MAX_START_FN_DURATION` cap.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let wasm = wat::parse_str(SCHEDULER_IMPORT_WAT).expect("valid wat");

    // The spawn itself is the load-bearing assertion: pre-Wave-1 this
    // failed to link with an opaque "unknown import" error.
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("scheduler-importing guest must link and instantiate via spawn_instance");

    // The host functions must be reachable from a guest export. No deadline
    // was configured, so the per-store SchedulerContext is unbounded:
    //   * yield() -> YIELD_CODE_CONTINUE (0)
    //   * deadline-remaining-ms() -> u32::MAX, which is -1 reinterpreted
    //     through the i32 wasm result boundary.
    let yielded = exec
        .call_export_with_args(id, "do_yield", &[WasmArg::I32(0)])
        .await
        .expect("do_yield call");
    assert_eq!(
        yielded,
        serde_json::json!([0]),
        "unbounded context yield() should return CONTINUE (0)",
    );

    let remaining = exec
        .call_export_with_args(id, "remaining", &[WasmArg::I32(0)])
        .await
        .expect("remaining call");
    assert_eq!(
        remaining,
        serde_json::json!([-1]),
        "unbounded context deadline-remaining-ms should return the u32::MAX sentinel (-1 as i32)",
    );

    exec.terminate(id).await.expect("terminate");
}

#[tokio::test]
async fn scheduler_import_links_even_with_deadline() {
    // A deadline-configured spawn exercises the bounded SchedulerContext
    // path while still linking the scheduler surface. The ticker is live
    // (engine built inside the runtime), so the deadline contract is
    // satisfiable and the spawn is not refused.
    use std::time::Duration;

    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    assert!(
        engine.is_epoch_ticker_running(),
        "ticker should auto-spawn inside a tokio runtime",
    );
    let exec = TensorWasmExecutor::new(engine);
    let wasm = wat::parse_str(SCHEDULER_IMPORT_WAT).expect("valid wat");

    let cfg = SpawnConfig::for_tenant(TenantId(7)).with_deadline(Duration::from_secs(30));
    let id = exec
        .spawn_instance(cfg, &wasm)
        .await
        .expect("scheduler-importing guest with a deadline must link and spawn");

    // A fresh 30s budget — yield() must report CONTINUE (well above the
    // approaching threshold).
    let yielded = exec
        .call_export_with_args(id, "do_yield", &[WasmArg::I32(0)])
        .await
        .expect("do_yield call");
    assert_eq!(yielded, serde_json::json!([0]));

    exec.terminate(id).await.expect("terminate");
}
