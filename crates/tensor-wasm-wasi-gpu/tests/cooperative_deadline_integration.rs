// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T36 — cooperative-deadline / back-pressure integration tests.
//!
//! Drives the `wasi:scheduler/host@0.1.0` interface end-to-end with
//! an absolute `Instant` deadline installed via
//! [`SchedulerContext::with_bp_deadline_instant`]. The verdicts the
//! guest observes must agree with the BackPressure's acquire
//! decisions for the same `Instant`: CONTINUE > NEAR-WINDOW >
//! DEADLINE-ELAPSED.
//!
//! The executor's epoch-ticker path is intentionally NOT exercised
//! here — that machinery is heavyweight and orthogonal. We supply a
//! synthetic `Instant` directly, exactly as the production wiring
//! does (`InstanceState::with_deadline().with_deadline_duration()`
//! propagates the same `Instant` into the scheduler context).

use std::thread;
use std::time::{Duration, Instant};

use tensor_wasm_wasi_gpu::scheduler::{
    add_scheduler_to_linker, SchedulerContext, YIELD_CODE_CONTINUE,
    YIELD_CODE_DEADLINE_APPROACHING, YIELD_CODE_STOP,
};

/// Per-store payload — same shape as the unit-level scheduler tests
/// but with the new BP-aligned `Instant` deadline installed.
struct TestStore {
    scheduler: SchedulerContext,
}

fn make_engine_and_linker() -> (wasmtime::Engine, wasmtime::Linker<TestStore>) {
    let mut config = wasmtime::Config::new();
    config.async_support(true);
    let engine = wasmtime::Engine::new(&config).expect("engine");
    let mut linker: wasmtime::Linker<TestStore> = wasmtime::Linker::new(&engine);
    add_scheduler_to_linker(&mut linker, |store: &TestStore| &store.scheduler)
        .expect("add_scheduler_to_linker");
    (engine, linker)
}

/// Wasm guest with a single-shot `yield()` probe. We call it multiple
/// times from the host between sleeps to observe the verdict
/// evolving over a 100 ms window.
const PROBE_WAT: &str = r#"
(module
  (import "wasi:scheduler/host@0.1.0" "yield" (func $y (result i32)))
  (import "wasi:scheduler/host@0.1.0" "deadline-remaining-ms" (func $rem (result i32)))

  (func (export "probe") (result i32) (call $y))
  (func (export "remaining_ms") (result i32) (call $rem))
)
"#;

/// Build a store whose scheduler is driven by an absolute `Instant`
/// deadline. Returns the store plus the instantiated wasm module.
async fn make_probe_instance(
    bp_deadline: Option<Instant>,
) -> (
    wasmtime::Store<TestStore>,
    wasmtime::Instance,
) {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(PROBE_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let scheduler = SchedulerContext::new(None).with_bp_deadline_instant(bp_deadline);
    let mut store = wasmtime::Store::new(&engine, TestStore { scheduler });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    (store, instance)
}

#[tokio::test]
async fn yield_verdict_progresses_continue_near_elapsed() {
    // 100 ms deadline window. We probe at three points:
    //   t≈0 ms   → CONTINUE  (remaining > DEADLINE_NEAR_WINDOW)
    //   t≈60 ms  → APPROACHING (within the 50 ms NEAR window)
    //   t≈110 ms → STOP (past the deadline)
    //
    // The DEADLINE_NEAR_WINDOW const lives in `async_dispatch`; we
    // use 50 ms here as the contract the scheduler upholds. If the
    // const changes the timing in this test would need to track it.
    let deadline = Instant::now() + Duration::from_millis(100);
    let (mut store, instance) = make_probe_instance(Some(deadline)).await;
    let probe = instance
        .get_typed_func::<(), i32>(&mut store, "probe")
        .expect("probe");

    // t≈0 ms — must be CONTINUE.
    let code = probe.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code as u32, YIELD_CODE_CONTINUE,
        "at t≈0 (>= 50 ms remaining) must observe CONTINUE, got {code}"
    );

    // Sleep into the NEAR window. 60 ms total elapsed → ~40 ms
    // remaining, which is inside the 50 ms threshold.
    tokio::time::sleep(Duration::from_millis(60)).await;
    let code = probe.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code as u32, YIELD_CODE_DEADLINE_APPROACHING,
        "at t≈60 ms (~40 ms remaining, inside 50 ms window) must observe APPROACHING, got {code}"
    );

    // Sleep past the deadline. 110 ms total elapsed → 10 ms past.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let code = probe.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code as u32, YIELD_CODE_STOP,
        "at t≈110 ms (past deadline) must observe STOP, got {code}"
    );
}

#[tokio::test]
async fn remaining_ms_decreases_under_bp_instant() {
    // The `deadline-remaining-ms` query path must also flow from the
    // BP-aligned Instant, not from the legacy `started_at +
    // deadline_ms` computation. Confirm the readout strictly
    // decreases over a sleep.
    let deadline = Instant::now() + Duration::from_millis(500);
    let (mut store, instance) = make_probe_instance(Some(deadline)).await;
    let remaining = instance
        .get_typed_func::<(), i32>(&mut store, "remaining_ms")
        .expect("remaining_ms");

    let before = remaining.call_async(&mut store, ()).await.expect("call");
    assert!(before > 0, "initial remaining must be positive, got {before}");
    assert!(before <= 500, "initial remaining must be ≤ 500 ms, got {before}");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let after = remaining.call_async(&mut store, ()).await.expect("call");
    assert!(
        after < before,
        "remaining must strictly decrease over a 100 ms sleep: before={before}, after={after}"
    );
}

#[tokio::test]
async fn synthetic_instant_replaces_executor_epoch_path() {
    // The executor's epoch-ticker path is heavyweight; this test
    // confirms the cooperative-deadline contract holds independently
    // of it. A guest seeing a STOP code from yield() before any
    // wasmtime epoch interrupt fires is precisely the P99 win the
    // feature was added to deliver.
    let deadline = Instant::now() - Duration::from_millis(5);
    let (mut store, instance) = make_probe_instance(Some(deadline)).await;
    let probe = instance
        .get_typed_func::<(), i32>(&mut store, "probe")
        .expect("probe");
    // Without any blocking sleep — the deadline is already past at
    // construction time — the first yield must observe STOP.
    let code = probe.call_async(&mut store, ()).await.expect("call");
    assert_eq!(code as u32, YIELD_CODE_STOP);
}

#[tokio::test]
async fn no_instant_falls_back_to_legacy_ms_path() {
    // When `bp_deadline_instant` is unset, the scheduler context
    // falls back to the historical `deadline_ms` window. This is
    // critical for back-compat with embedders that have not yet
    // wired the T36 Instant.
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(PROBE_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    // 1 ms budget; sleep 15 ms (sync, before the call) burns it.
    let scheduler = SchedulerContext::new(Some(1));
    let mut store = wasmtime::Store::new(&engine, TestStore { scheduler });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    thread::sleep(Duration::from_millis(15));
    let probe = instance
        .get_typed_func::<(), i32>(&mut store, "probe")
        .expect("probe");
    let code = probe.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code as u32, YIELD_CODE_STOP,
        "legacy ms path must still STOP a burned 1 ms budget"
    );
}
