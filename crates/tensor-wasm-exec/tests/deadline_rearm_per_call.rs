//! Regression test for the per-call epoch-deadline re-arm.
//!
//! Before the fix, `set_epoch_deadline` was configured once at spawn time
//! and `InstanceState::deadline` was a single absolute `Instant`. A second
//! `call_export` issued after the deadline had already elapsed reported
//! `Timeout { elapsed_ms: 0, deadline_ms: 0 }` because:
//!
//! - the wasmtime epoch counter set at spawn was already consumed, so the
//!   trap fired before any real work happened (elapsed ~= 0), and
//! - the legacy `deadline_at.saturating_duration_since(started_at)` clamped
//!   to zero because `started_at` was already past `deadline_at`.
//!
//! After the fix, each `call_export` re-arms both the wall-clock deadline
//! (`InstanceState::deadline = now + d`) and the wasmtime epoch deadline,
//! so a second call gets the same window as the first and the timeout
//! report carries honest numbers.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

fn spin_wasm() -> Vec<u8> {
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
// developer machines skip it. Same caveat as `epoch_timeout.rs`.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_call_after_elapsed_deadline_reports_honest_numbers() {
    // Fast ticker so the 10 ms deadline trips promptly.
    let cfg = EngineConfig {
        epoch_tick: Duration::from_millis(2),
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let deadline = Duration::from_millis(10);
    let id = exec
        .spawn_instance(
            SpawnConfig::for_tenant(TenantId(1)).with_deadline(deadline),
            &spin_wasm(),
        )
        .await
        .expect("spawn");

    // Sleep WELL past the configured deadline. Before the fix, the
    // spawn-time `set_epoch_deadline` is now stale and the spawn-time
    // `InstanceState::deadline` is in the past.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // First call after the elapsed window. The bug would have surfaced
    // here as `Timeout { elapsed_ms: 0, deadline_ms: 0 }`. After the fix
    // the deadline is re-armed to `now + 10ms`, the spin loop runs for
    // ~10 ms, and the epoch interrupt fires with honest numbers.
    let first = exec
        .call_export(id, "spin")
        .await
        .expect_err("first call must time out");
    match first {
        ExecError::Timeout(ctx) => {
            assert!(
                ctx.elapsed_ms > 0,
                "first call: elapsed_ms must be > 0, got {ctx:?}",
            );
            assert!(
                ctx.deadline_ms > 0,
                "first call: deadline_ms must be > 0, got {ctx:?}",
            );
            // Sanity-check the deadline_ms equals the configured 10 ms
            // (this is the actionable "honest numbers" assertion — the
            // pre-fix code reported 0).
            assert_eq!(
                ctx.deadline_ms, 10,
                "first call: deadline_ms must equal the configured 10 ms, got {ctx:?}",
            );
        }
        other => panic!("first call: expected Timeout, got {other:?}"),
    }

    // Second call. The pre-fix bug repeated here for the same reason
    // (spawn-time arm consumed once, `InstanceState::deadline` is past).
    // After the fix the deadline is re-armed again and we get a second
    // round of honest numbers — not the degenerate 0/0.
    let second = exec
        .call_export(id, "spin")
        .await
        .expect_err("second call must time out");
    match second {
        ExecError::Timeout(ctx) => {
            assert!(
                ctx.elapsed_ms > 0,
                "second call: elapsed_ms must be > 0, got {ctx:?}",
            );
            assert!(
                ctx.deadline_ms > 0,
                "second call: deadline_ms must be > 0 (was 0 before the per-call \
                 re-arm fix), got {ctx:?}",
            );
            assert_eq!(
                ctx.deadline_ms, 10,
                "second call: deadline_ms must equal the configured 10 ms, got {ctx:?}",
            );
        }
        other => panic!("second call: expected Timeout, got {other:?}"),
    }

    exec.terminate(id).await.expect("terminate");
}
