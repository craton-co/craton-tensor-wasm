// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! HIGH finding 1: `terminate` must interrupt an in-flight, DEADLINE-LESS call
//! rather than only de-registering the instance and freeing the admission slot.
//!
//! Before the fix, `terminate` removed the registry entry and decremented the
//! counters, but the per-instance mutex was held across `call_async` for the
//! whole duration of the guest call — so a compute-bound, deadline-less guest
//! kept running until it returned of its own accord, which (for an infinite
//! loop) is never. The fix adds an `Arc<AtomicBool>` cooperative-cancellation
//! flag the cooperative epoch callback consults: `terminate` flips it lock-free
//! and the next epoch tick traps the guest.
//!
//! This test spawns an infinite-loop guest with NO per-call deadline AND the
//! engine-wide `max_call_duration` ceiling disabled, so the ONLY thing that can
//! interrupt the call is `terminate`'s cancellation flag. It then drives the
//! call on one task, terminates from another, and asserts the call returns
//! (does not hang) promptly.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

fn infinite_loop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "spin") (loop (br 0))))"#).unwrap()
}

// Wasmtime's epoch-interrupt of an infinite Wasm loop triggers a fiber
// unwinding path that, on Windows, panics in a non-unwinding C-ABI frame
// (`STATUS_STACK_BUFFER_OVERRUN`). Linux/macOS CI runs the test; Windows
// developer machines skip it. SAME caveat as `epoch_timeout.rs` — this and the
// other epoch-interrupt tests MUST run on Linux CI to have any coverage value.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt; run on Linux CI"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminate_interrupts_deadline_less_running_call() {
    // Fast ticker so the cancellation flag is observed within a few ms.
    let cfg = EngineConfig {
        epoch_tick: Duration::from_millis(5),
        // CRITICAL: disable the engine-wide call ceiling so a deadline-less
        // call is genuinely unbounded — the ONLY interrupt source is
        // `terminate`'s cancellation flag, which is exactly what we're proving.
        max_call_duration: None,
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    // Spawn WITHOUT a deadline — `SpawnConfig::for_tenant` leaves `deadline:
    // None`. With `max_call_duration: None` above, the call has no wall-clock
    // bound of any kind.
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &infinite_loop_wasm())
        .await
        .expect("spawn");

    // Drive the infinite-loop call on a background task.
    let exec_call = exec.clone();
    let call = tokio::spawn(async move { exec_call.call_export_with_args(id, "spin", &[]).await });

    // Give the guest a beat to actually enter the loop, then terminate it.
    tokio::time::sleep(Duration::from_millis(50)).await;
    // `terminate` must succeed (the instance is still registered) and flip the
    // cancellation flag the epoch callback consults.
    exec.terminate(id).await.expect("terminate registered instance");

    // The call MUST return (with an error — the guest was trapped) within a
    // bounded window. If `terminate` could not interrupt it, this `timeout`
    // fires and the test fails instead of hanging forever.
    let outcome = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("terminate must interrupt the in-flight call (it hung)")
        .expect("call task must not panic");
    assert!(
        outcome.is_err(),
        "a terminated infinite-loop call must return an error, got {outcome:?}"
    );

    // The instance is gone from the registry (terminate removed it).
    assert_eq!(exec.live_count(), 0, "terminate must de-register the instance");
}
