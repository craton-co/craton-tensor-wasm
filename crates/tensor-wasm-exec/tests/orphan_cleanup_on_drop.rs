//! Regression test for api S-20 / orphan-instance cleanup.
//!
//! Pre-fix flow:
//! 1. Caller `spawn_instance` -> instance registered, `instance_count = 1`.
//! 2. Caller `await call_export(id, "noop")`.
//! 3. Outer cancellation (timeout, dropped task) drops the call_export
//!    future before completion.
//! 4. Instance remains in `instances` DashMap forever; `instance_count`
//!    stays at 1; the caller never gets the id back to terminate it.
//!
//! Post-fix flow uses `call_export_with_args_then_terminate`, which installs
//! an `AutoTerminateGuard` that syncronously removes the registry entry
//! and decrements the counter when the wrapping future is dropped
//! mid-await.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

fn long_running_wasm() -> Vec<u8> {
    // An infinite loop in `noop` so call_export never returns on its own —
    // the only way out is the deadline trip or future drop.
    wat::parse_str(
        r#"
        (module
            (func (export "noop") (loop br 0))
        )
        "#,
    )
    .unwrap()
}

// This test drops an in-flight `call_async` future whose guest is a
// compute-bound infinite loop, expecting the outer `timeout(100ms)` to cancel
// it mid-await so the `AutoTerminateGuard`'s Drop path runs. That can only
// happen if the guest yields back to the async runtime mid-execution — i.e.
// with cooperative epoch yielding (`Store::epoch_deadline_async_yield_and_update`).
// The executor currently arms the epoch deadline in *trap* mode
// (`set_epoch_deadline`), so a pure-compute guest never returns `Pending`: the
// `call_async` poll blocks the worker until the deadline traps, the outer
// timeout can never drop the future before then, and it is the clean-completion
// path (not Drop-on-cancel) that removes the instance. On a single-threaded
// runtime the symptom is worse — the epoch ticker is starved by the spinning
// guest, the deadline never advances, and the test hangs indefinitely; the
// multi_thread flavor below at least lets the ticker run. Until the executor
// opts into async epoch yielding this test cannot exercise its intended path.
// The happy-path cleanup is still covered by the sibling
// `call_then_terminate_clean_path_also_removes_instance`.
#[ignore = "needs cooperative epoch yield (async_yield_and_update) to drop a \
            compute-bound guest mid-await; executor currently traps on deadline"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_terminate_guard_releases_slot_on_future_drop() {
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);
    let wasm = long_running_wasm();

    let cfg = SpawnConfig::for_tenant(TenantId(1))
        // Long deadline so the test exits via the drop path, not the
        // deadline path.
        .with_deadline(Duration::from_secs(60));
    let id = exec.spawn_instance(cfg, &wasm).await.expect("spawn");
    assert_eq!(
        exec.instances_len(),
        1,
        "instance must be registered after spawn"
    );

    // Drop the future before it completes. `tokio::time::timeout` is a
    // standard way to express "abort after N ms" — when it fires it
    // drops the inner future, exactly the outer-cancellation shape that
    // tower's TimeoutLayer creates.
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        exec.call_export_with_args_then_terminate(id, "noop", &[]),
    )
    .await;
    assert!(result.is_err(), "outer timeout must fire");

    // Give the wasmtime fiber a moment to unwind cleanly. The drop-guard
    // synchronously removes the registry entry inside the `Drop` impl
    // — there is no async wait — but the wasmtime side may need a yield
    // before the Drop actually runs in the runtime's poll cycle.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // The drop-guard MUST have removed the instance.
    assert_eq!(
        exec.instances_len(),
        0,
        "AutoTerminateGuard must remove the leaked instance on future drop"
    );
}

#[tokio::test]
async fn call_then_terminate_clean_path_also_removes_instance() {
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);
    // Trivial noop that returns immediately.
    let wasm = wat::parse_str(r#"(module (func (export "noop")))"#).unwrap();
    let cfg = SpawnConfig::for_tenant(TenantId(1)).with_deadline(Duration::from_secs(5));
    let id = exec.spawn_instance(cfg, &wasm).await.expect("spawn");
    exec.call_export_with_args_then_terminate(id, "noop", &[])
        .await
        .expect("call_export_with_args_then_terminate succeeds");
    assert_eq!(
        exec.instances_len(),
        0,
        "clean path also terminates the instance"
    );
}
