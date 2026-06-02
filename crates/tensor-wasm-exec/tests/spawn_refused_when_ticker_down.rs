// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Ticker-down spawn refusal regression test.
//!
//! Pre-fix the executor logged a one-shot `tracing::error!` and proceeded
//! with the spawn when [`TensorWasmEngine::is_epoch_ticker_running`]
//! returned `false`. That meant a runaway guest — start function or
//! per-call invocation — could wedge the worker thread forever: with
//! the epoch counter frozen, neither the per-call deadline nor
//! [`tensor_wasm_exec::executor::MAX_START_FN_DURATION`] can fire.
//!
//! The fix refuses the spawn with
//! [`tensor_wasm_exec::executor::ExecError::EpochTickerNotRunning`] when
//! the ticker is down and a deadline-class bound would otherwise apply.
//! This test stands up an engine, stops its ticker, and confirms the
//! refusal.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

fn trivial_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).expect("valid wat")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_refused_when_ticker_down_with_deadline() {
    // `TensorWasmEngine::new` auto-spawns the ticker when constructed
    // inside a Tokio runtime. To exercise the ticker-down path we
    // stop it explicitly.
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.stop_epoch_ticker();
    assert!(
        !engine.is_epoch_ticker_running(),
        "ticker must be stopped before the test runs",
    );
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    // Configure a per-call deadline so the contract is unambiguous:
    // without the ticker the deadline cannot fire, so the spawn must
    // be refused.
    let cfg = SpawnConfig::for_tenant(TenantId(1)).with_deadline(Duration::from_millis(100));
    let err = exec
        .spawn_instance(cfg, &trivial_wasm())
        .await
        .expect_err("spawn must be refused when ticker is down and deadline is configured");
    assert!(
        matches!(err, ExecError::EpochTickerNotRunning),
        "expected EpochTickerNotRunning, got {err:?}",
    );

    // The refused spawn must not have leaked an admission slot.
    assert_eq!(
        exec.instances_len(),
        0,
        "refused spawn must roll back its admission slot",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_refused_when_ticker_down_without_explicit_deadline() {
    // Even without a per-call deadline, `MAX_START_FN_DURATION` always
    // applies inside `Instance::new_async`, so the ticker-required
    // contract holds. Pin that behaviour: a `SpawnConfig` with no
    // deadline must still be refused when the ticker is down.
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.stop_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trivial_wasm())
        .await
        .expect_err("spawn must be refused when ticker is down");
    assert!(
        matches!(err, ExecError::EpochTickerNotRunning),
        "expected EpochTickerNotRunning, got {err:?}",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_succeeds_when_ticker_running() {
    // Sanity counterpart: the same setup with the ticker running must
    // succeed. Catches a regression that would over-tighten the gate
    // and refuse every spawn.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    assert!(
        engine.is_epoch_ticker_running(),
        "ticker should be auto-spawned by the engine constructor inside a tokio runtime",
    );
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(
            SpawnConfig::for_tenant(TenantId(1)).with_deadline(Duration::from_millis(100)),
            &trivial_wasm(),
        )
        .await
        .expect("spawn must succeed when the ticker is running");
    exec.terminate(id).await.expect("terminate");
}
