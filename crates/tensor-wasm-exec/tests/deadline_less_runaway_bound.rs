// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! MED finding 2: a DEADLINE-LESS instance must still be wall-clock-bounded by
//! the engine-wide `EngineConfig::max_call_duration` ceiling, so `deadline:
//! None` means "engine default ceiling", never "infinite".
//!
//! Before the fix the post-start cooperative epoch callback yielded forever for
//! a deadline-less spawn — a compute-bound guest stayed cancellable on
//! future-drop, but nothing ever TRAPPED it of its own accord. This test spawns
//! an infinite-loop guest with NO per-call deadline on an engine whose
//! `max_call_duration` is set short, and asserts the call is interrupted within
//! ~2× the ceiling (rather than running unbounded).

use std::sync::Arc;
use std::time::{Duration, Instant};

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

fn infinite_loop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "spin") (loop (br 0))))"#).unwrap()
}

// SAME Windows caveat as `epoch_timeout.rs`: wasmtime's epoch-interrupt fiber
// unwinding panics on Windows in a non-unwinding C-ABI frame. Linux/macOS CI
// runs this; Windows developer machines skip it. This test MUST run on Linux
// CI to cover the deadline-less ceiling.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt; run on Linux CI"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadline_less_call_is_bounded_by_engine_ceiling() {
    let ceiling = Duration::from_millis(150);
    let cfg = EngineConfig {
        // Fast ticker so the trap lands close to the ceiling, not a tick late.
        epoch_tick: Duration::from_millis(5),
        // The engine-wide ceiling that must bound a deadline-less call.
        max_call_duration: Some(ceiling),
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    // Spawn WITHOUT a per-call deadline. The only bound is the engine ceiling.
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &infinite_loop_wasm())
        .await
        .expect("spawn");

    let start = Instant::now();
    let res = exec.call_export_with_args(id, "spin", &[]).await;
    let elapsed = start.elapsed();

    // The deadline-less call MUST be interrupted (not run forever).
    assert!(
        res.is_err(),
        "a deadline-less infinite loop must be bounded by max_call_duration"
    );
    // Interruption within ~2× the ceiling (plus CI slack), proving the engine
    // ceiling — not some other mechanism — bounded it.
    let limit = ceiling * 2 + Duration::from_millis(200);
    assert!(
        elapsed <= limit,
        "deadline-less call ran {elapsed:?}, expected ≤ {limit:?} (2× ceiling + slack)"
    );

    exec.terminate(id).await.expect("terminate");
}
