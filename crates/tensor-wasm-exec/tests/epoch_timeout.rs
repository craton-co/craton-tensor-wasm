//! S7 done-when #2: epoch timeout kills infinite-loop Wasm within 2× deadline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{TensorWasmEngine, EngineConfig};
use tensor_wasm_exec::executor::{TensorWasmExecutor, SpawnConfig};

fn infinite_loop_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (func (export "spin") (loop (br 0)))
        )
    "#,
    )
    .unwrap()
}

// Wasmtime's epoch-interrupt of an infinite Wasm loop triggers a fiber
// unwinding path that, on Windows, panics in a non-unwinding C-ABI frame
// (`STATUS_STACK_BUFFER_OVERRUN`). Linux/macOS CI runs the test; Windows
// developer machines skip it. Tracked in Wasmtime upstream; revisit at
// the next minor bump.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infinite_loop_terminates_within_2x_deadline() {
    // Faster ticker for a tight test — every 5 ms.
    let cfg = EngineConfig {
        epoch_tick: Duration::from_millis(5),
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let deadline = Duration::from_millis(100);
    let id = exec
        .spawn_instance(
            SpawnConfig::for_tenant(TenantId(1)).with_deadline(deadline),
            &infinite_loop_wasm(),
        )
        .await
        .expect("spawn");

    let start = Instant::now();
    let res = exec.call_export(id, "spin").await;
    let elapsed = start.elapsed();

    // The call MUST return an error (any of Wasmtime / Timeout flavour) within 2× deadline.
    assert!(res.is_err(), "expected the infinite loop to be interrupted");
    let twice = deadline * 2;
    // Add a 100 ms slack for CI noise.
    let limit = twice + Duration::from_millis(100);
    assert!(
        elapsed <= limit,
        "epoch interruption took {:?}, expected ≤ {:?} (2× deadline + slack)",
        elapsed,
        limit
    );

    // Cleanup — the instance still exists in the executor; terminating drops it.
    exec.terminate(id).await.expect("terminate");
}
