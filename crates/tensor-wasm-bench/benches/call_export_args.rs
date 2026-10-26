// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Measures the overhead of `call_export_with_args` vs the legacy
//! no-args `call_export` shim. Confirms the typed-args path doesn't
//! regress simple `() -> ()` calls measurably.
//!
//! ## Why two groups
//!
//! Batch 6 introduced [`TensorWasmExecutor::call_export_with_args`] which
//! accepts a `&[WasmArg]` slice and reflects guest signatures via wasmtime's
//! dynamic `Func::call_async` path. The legacy `call_export` shim is a
//! `typed::<(), ()>` fast-path that bypasses the slice scan entirely. We
//! want two numbers from this bench:
//!
//! 1. **`call_export/noargs/call_export_with_args_empty`** — same export
//!    signature (`() -> ()`) called through the new args-aware entrypoint
//!    with an empty arg slice. Compared against the historical
//!    `cold_start::call_export` baseline this isolates the slice-iteration +
//!    signature-reflection overhead the typed-args path adds even when no
//!    args are passed.
//! 2. **`call_export/args/two_i32`** — a `(i32, i32) -> i32` export
//!    invoked with `[WasmArg::I32(1), WasmArg::I32(2)]`. Pins the cost of
//!    the typed-args path's *actual* job: marshalling `WasmArg` enum
//!    values into wasmtime `Val`s and back.
//!
//! ## Hot-instance vs full-envelope (finding 3)
//!
//! Each measurement comes in two shapes:
//!
//! * **`hot/*`** — the PRIMARY args-marshalling signal. The instance is
//!   spawned once *outside* `b.iter`, called in the timed loop, and
//!   terminated after the loop. Because `call_export_with_args` is `&self`,
//!   the per-sample cost is purely the invocation: slice scan + signature
//!   reflection + `WasmArg` → `Val` marshalling + the trivial wasm body.
//!   This is the number a regression on the typed-args path should move.
//! * **`envelope/*`** — spawn + invoke + terminate inside the timed loop.
//!   Instance creation dominates here, so the absolute number is anchored to
//!   the `cold_start/restore_steady_state` baseline and matches what the
//!   `/invoke` HTTP path does per request today. Kept as a separate group so
//!   the spawn/terminate cost never buries the args-marshalling delta the
//!   `hot/*` group isolates.
//!
//! Comparing `hot/two_i32` against `hot/call_export_with_args_empty` isolates
//! the *actual* marshalling cost (two `WasmArg::I32`s through the dynamic
//! path) with no spawn noise in the way.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};

/// Minimal WAT module exposing two exports the bench drives directly:
///
/// * `add (param i32 i32) (result i32)` — exercises the typed-args path
///   with a representative two-i32 signature. The body is a single
///   `i32.add` so wasm compute time is in the low nanoseconds and the
///   surrounding executor overhead dominates the measurement (which is
///   the property we want).
/// * `noop ()` — exercises the empty-args overload of
///   `call_export_with_args` and the legacy `call_export` shim alike.
fn make_adder_wat() -> &'static str {
    r#"(module
        (func (export "add") (param i32 i32) (result i32)
          (i32.add (local.get 0) (local.get 1)))
        (func (export "noop")))
    "#
}

/// Build the executor + a parsed adder module, paid once per group (outside
/// every `b.iter`), so the engine/epoch-ticker setup never lands in a timed
/// sample. The `rt.enter()` guard is held only for `spawn_epoch_ticker`, which
/// needs an active runtime context; later calls go through `rt.block_on`,
/// which enters the runtime itself.
fn setup(rt: &tokio::runtime::Runtime) -> (TensorWasmExecutor, Vec<u8>, SpawnConfig) {
    let _guard = rt.enter();
    let mut engine = TensorWasmEngine::new().unwrap();
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);
    let wasm = wat::parse_str(make_adder_wat()).unwrap();
    let spawn_cfg = SpawnConfig::for_tenant(TenantId(1));
    (exec, wasm, spawn_cfg)
}

/// PRIMARY measurement: instance spawned once outside `b.iter`, called in the
/// timed loop, terminated after. Isolates the args-marshalling cost with no
/// spawn/terminate noise (finding 3).
fn bench_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_export/hot");
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (exec, wasm, spawn_cfg) = setup(&rt);

    // Empty-args overload on a `() -> ()` export — the slice-scan +
    // signature-reflection overhead with no marshalling work.
    {
        let id = rt
            .block_on(exec.spawn_instance(spawn_cfg.clone(), &wasm))
            .unwrap();
        group.bench_function("call_export_with_args_empty", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let out = exec.call_export_with_args(id, "noop", black_box(&[])).await;
                    black_box(out.is_ok());
                });
            });
        });
        let _ = rt.block_on(exec.terminate(id));
    }

    // Two-i32 path — the actual `WasmArg` → `Val` marshalling cost.
    {
        let id = rt
            .block_on(exec.spawn_instance(spawn_cfg.clone(), &wasm))
            .unwrap();
        group.bench_function("two_i32", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let args = vec![WasmArg::I32(1), WasmArg::I32(2)];
                    let out = exec
                        .call_export_with_args(id, "add", black_box(&args))
                        .await;
                    black_box(out.is_ok());
                });
            });
        });
        let _ = rt.block_on(exec.terminate(id));
    }

    group.finish();
}

/// Full-envelope measurement: spawn + invoke + terminate inside the timed
/// loop. Instance creation dominates; matches the per-request shape of the
/// `/invoke` HTTP path. Kept separate so it never buries the `hot/*` signal.
fn bench_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_export/envelope");
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (exec, wasm, spawn_cfg) = setup(&rt);

    group.bench_function("call_export_with_args_empty", |b| {
        b.iter(|| {
            rt.block_on(async {
                let id = exec.spawn_instance(spawn_cfg.clone(), &wasm).await.unwrap();
                let out = exec.call_export_with_args(id, "noop", black_box(&[])).await;
                black_box(out.is_ok());
                let _ = exec.terminate(id).await;
            });
        });
    });

    group.bench_function("two_i32", |b| {
        b.iter(|| {
            rt.block_on(async {
                let id = exec.spawn_instance(spawn_cfg.clone(), &wasm).await.unwrap();
                let args = vec![WasmArg::I32(1), WasmArg::I32(2)];
                let out = exec
                    .call_export_with_args(id, "add", black_box(&args))
                    .await;
                black_box(out.is_ok());
                let _ = exec.terminate(id).await;
            });
        });
    });

    group.finish();
}

criterion_group!(call_export, bench_hot, bench_envelope);
criterion_main!(call_export);
