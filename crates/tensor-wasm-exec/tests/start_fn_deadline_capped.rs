//! Regression for exec S-8: a `start` function with no per-call deadline
//! must NOT burn forever inside `Instance::new_async`.
//!
//! Before the fix, `SpawnConfig { deadline: None, .. }` set the per-store
//! epoch deadline to `u64::MAX` for the duration of the instantiation
//! phase. A `start` function that spun in an infinite loop would therefore
//! hang `spawn_instance` indefinitely — and since the instance is not
//! registered with the executor until `new_async` returns, `terminate`
//! could never reach it.
//!
//! The fix caps the start-function deadline at
//! [`tensor_wasm_exec::executor::MAX_START_FN_DURATION`] (30 s) regardless
//! of the caller's per-call deadline. This test verifies that bound by
//! spawning a module whose `start` function is `(loop br 0)` and asserting
//! that `spawn_instance` returns an error within ~35 s.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

fn infinite_start_wasm() -> Vec<u8> {
    // `start` references `$loop`, which never returns. The epoch
    // interrupt is the only thing that can stop it.
    wat::parse_str(
        r#"
        (module
          (start $loop)
          (func $loop (loop br 0))
        )
    "#,
    )
    .expect("wat parse")
}

// See `epoch_timeout.rs` for the same caveat: wasmtime's fiber-unwinding
// path panics in a non-unwinding C-ABI frame on Windows when the epoch
// interrupt fires inside an infinite Wasm loop (STATUS_STACK_BUFFER_OVERRUN).
// Linux/macOS CI exercises this test; Windows developer machines skip it.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_function_is_capped_without_per_call_deadline() {
    // Use a fast-ish ticker so this test does not have to sit through the
    // full 30 s cap with the default 10 ms tick — the cap is still 30 s
    // of wall time, but a 5 ms tick gives the deadline more chances to
    // trip promptly once 30 s elapses.
    let cfg = EngineConfig {
        epoch_tick: Duration::from_millis(5),
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    // CRITICAL: without the ticker the epoch deadline never advances and
    // the cap cannot fire. The executor logs a one-shot error when this
    // is the case; here we set it up correctly so the deadline actually
    // trips.
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let start = Instant::now();
    let res = exec
        .spawn_instance(
            // No per-call deadline — the cap is the ONLY thing that can
            // stop the start function from running forever.
            SpawnConfig::for_tenant(TenantId(1)),
            &infinite_start_wasm(),
        )
        .await;
    let elapsed = start.elapsed();

    assert!(
        res.is_err(),
        "expected start-function infinite loop to be interrupted (cap = 30 s)"
    );
    // Cap is 30 s; allow 5 s slack for the ticker cadence and CI noise.
    let limit = Duration::from_secs(35);
    assert!(
        elapsed <= limit,
        "spawn_instance took {elapsed:?}, expected <= {limit:?} (MAX_START_FN_DURATION + slack)",
    );
}
