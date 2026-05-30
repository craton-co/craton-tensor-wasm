// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the `wasi:scheduler/host@0.1.0` cooperative-
//! deadline interface (roadmap feature #4).
//!
//! The pure-Rust unit tests live next to the [`SchedulerContext`] impl
//! in `src/scheduler.rs`; this file exercises the end-to-end wiring
//! through a wasmtime `Linker` — i.e. an actual Wasm guest importing
//! `yield` and observing the host's return code.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tensor_wasm_wasi_gpu::scheduler::{
    add_scheduler_to_linker, SchedulerContext, YIELD_CODE_CONTINUE, YIELD_CODE_STOP,
};

/// Per-store payload used by the scheduler integration tests. Just
/// holds a [`SchedulerContext`] — the linker getter pulls that out via
/// the closure passed to [`add_scheduler_to_linker`].
struct TestStore {
    scheduler: SchedulerContext,
}

fn make_engine_and_linker() -> (wasmtime::Engine, wasmtime::Linker<TestStore>) {
    let config = wasmtime::Config::new();
    let engine = wasmtime::Engine::new(&config).expect("engine");
    let mut linker: wasmtime::Linker<TestStore> = wasmtime::Linker::new(&engine);
    add_scheduler_to_linker(&mut linker, |store: &TestStore| &store.scheduler)
        .expect("add_scheduler_to_linker");
    (engine, linker)
}

/// Guest that tight-loops while calling `yield()` every iteration.
/// Returns the iteration counter at the point it observed a non-zero
/// yield response (or the loop's hard cap of 1_000_000 if it ran to
/// completion).
const YIELD_LOOP_WAT: &str = r#"
(module
  (import "wasi:scheduler/host@0.1.0" "yield" (func $y (result i32)))
  (import "wasi:scheduler/host@0.1.0" "deadline-remaining-ms" (func $rem (result i32)))

  (func (export "tight_loop") (result i32)
    (local $i i32)
    (loop $L
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (if (call $y) (then (return (local.get $i))))
      (br_if $L (i32.lt_s (local.get $i) (i32.const 1000000)))
    )
    (local.get $i))

  (func (export "remaining_ms") (result i32)
    (call $rem))
)
"#;

#[tokio::test]
async fn unbounded_yield_runs_to_completion() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::unbounded(),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "tight_loop")
        .expect("tight_loop");
    let n = f.call_async(&mut store, ()).await.expect("call");
    // Unbounded context — yield always returns 0, so the loop runs
    // to its hard cap.
    assert_eq!(n, 1_000_000);
}

#[tokio::test]
async fn deadline_breaks_yield_loop_early() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::new(Some(50)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "tight_loop")
        .expect("tight_loop");
    let n = f.call_async(&mut store, ()).await.expect("call");
    // 50 ms deadline — the loop body is small but each iteration
    // costs at least a host function call. Empirically the loop
    // breaks well before its hard cap of 1_000_000. Assert "broke
    // early": any count < 1_000_000 means the cooperative path
    // actually intercepted the loop. A real wedge would return
    // exactly 1_000_000.
    assert!(
        n < 1_000_000,
        "loop should break early under a 50 ms deadline, got {n} iterations"
    );
    // Sanity: at least one iteration ran.
    assert!(n >= 1, "loop should run at least one iteration, got {n}");
}

#[tokio::test]
async fn deadline_remaining_ms_visible_to_guest() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::new(Some(10_000)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "remaining_ms")
        .expect("remaining_ms");
    let remaining = f.call_async(&mut store, ()).await.expect("call");
    // The host call returns a u32 but the wasm export is `i32`. For a
    // 10s budget we expect well under 10_000 and well above 0 on any
    // reasonable runner.
    assert!(remaining > 0);
    assert!(remaining <= 10_000);
}

#[tokio::test]
async fn unbounded_remaining_ms_is_max_sentinel() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::unbounded(),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "remaining_ms")
        .expect("remaining_ms");
    let remaining = f.call_async(&mut store, ()).await.expect("call");
    // u32::MAX reinterpreted as i32 is -1.
    assert_eq!(
        remaining, -1,
        "unbounded context should report the u32::MAX sentinel"
    );
}

#[tokio::test]
async fn expired_deadline_returns_stop_to_guest() {
    let (engine, linker) = make_engine_and_linker();
    // 1 ms budget burned by a 15 ms sleep before the call — the first
    // yield in the loop must observe STOP.
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::new(Some(1)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    thread::sleep(Duration::from_millis(15));
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "tight_loop")
        .expect("tight_loop");
    let n = f.call_async(&mut store, ()).await.expect("call");
    // First yield trips → loop returns after exactly 1 iteration.
    assert_eq!(
        n, 1,
        "expired deadline must stop the loop on its first yield"
    );
}

#[tokio::test]
async fn yield_count_tracked_via_shared_arc() {
    // The yield counter lives behind an `Arc<AtomicU32>` so callers
    // that retain a clone of the context can observe how many times
    // the guest yielded during a call — useful for telemetry.
    let scheduler = SchedulerContext::new(Some(50));
    let observer = scheduler.clone();
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(YIELD_LOOP_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(&engine, TestStore { scheduler });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "tight_loop")
        .expect("tight_loop");
    let n = f.call_async(&mut store, ()).await.expect("call");
    // Every loop iteration calls yield once; the loop ran `n` times
    // (counter is incremented before the yield call so the
    // yield-count equals `n`).
    assert_eq!(observer.yield_count() as i32, n);
}

#[tokio::test]
async fn many_unbounded_yields_only_continue() {
    // Sanity check that the linker registration is wired correctly —
    // call `yield` directly via the wasmtime API to confirm the host
    // function returns 0 on an unbounded context.
    let (engine, linker) = make_engine_and_linker();
    let wat_src = r#"
        (module
          (import "wasi:scheduler/host@0.1.0" "yield" (func $y (result i32)))
          (func (export "once") (result i32) (call $y))
        )
    "#;
    let wasm = wat::parse_str(wat_src).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::unbounded(),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "once")
        .expect("once");
    for _ in 0..32 {
        let code = f.call_async(&mut store, ()).await.expect("call");
        assert_eq!(code as u32, YIELD_CODE_CONTINUE);
    }
}

/// Exercise `add_scheduler_to_linker` with a closure that walks
/// through an `Arc` indirection — the wasmtime linker captures the
/// closure by `Copy` so the getter must remain `Copy` even when
/// returning a borrow through `Arc::as_ref` chains.
#[test]
fn linker_registration_accepts_arc_indirection() {
    let engine = {
        let config = wasmtime::Config::new();
        wasmtime::Engine::new(&config).expect("engine")
    };
    struct Outer {
        inner: Arc<SchedulerContext>,
    }
    let mut linker: wasmtime::Linker<Outer> = wasmtime::Linker::new(&engine);
    add_scheduler_to_linker(&mut linker, |s: &Outer| s.inner.as_ref())
        .expect("registration through Arc must succeed");
}

#[tokio::test]
async fn stop_code_constant_round_trips_through_guest() {
    // Confirm that the numeric value of YIELD_CODE_STOP the host
    // reports through the i32 wasm boundary matches the public
    // constant — guards against accidental renumbering of the codes.
    let (engine, linker) = make_engine_and_linker();
    let wat_src = r#"
        (module
          (import "wasi:scheduler/host@0.1.0" "yield" (func $y (result i32)))
          (func (export "once") (result i32) (call $y))
        )
    "#;
    let wasm = wat::parse_str(wat_src).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            scheduler: SchedulerContext::new(Some(1)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    thread::sleep(Duration::from_millis(15));
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "once")
        .expect("once");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(code as u32, YIELD_CODE_STOP);
}
