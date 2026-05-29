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
//!    `cold_start::call_export` baseline this isolates the slice-iteration
//!    + signature-reflection overhead the typed-args path adds even when no
//!    args are passed.
//! 2. **`call_export/args/two_i32`** — a `(i32, i32) -> i32` export
//!    invoked with `[WasmArg::I32(1), WasmArg::I32(2)]`. Pins the cost of
//!    the typed-args path's *actual* job: marshalling `WasmArg` enum
//!    values into wasmtime `Val`s and back.
//!
//! Both groups spawn + terminate the instance inside the timed loop. That
//! inflates the absolute numbers (instance creation dominates a single
//! `call_export_with_args` invocation on a quiet host) but is the right
//! shape for a regression gate: the deltas between the two groups isolate
//! the args-path cost while the absolute floor is anchored to the
//! `cold_start/restore` baseline. Future iterations may add a "hot-instance"
//! variant that spawns once and calls in a loop — left as a follow-up for
//! when the typed-args path has a public hot-instance entrypoint
//! (`call_export_with_args` is `&self` today which makes a per-iter loop
//! straightforward, but the spawn/terminate envelope matches what the
//! `/invoke` HTTP path actually does today, so it stays as the headline
//! number).

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

fn bench_noargs(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_export/noargs");
    // setup
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = Arc::new(TensorWasmEngine::new().unwrap());
    let exec = TensorWasmExecutor::new(engine);
    let wasm = wat::parse_str(make_adder_wat()).unwrap();
    let spawn_cfg = SpawnConfig::for_tenant(TenantId(1));
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
    group.finish();
}

fn bench_with_args(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_export/args");
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = Arc::new(TensorWasmEngine::new().unwrap());
    let exec = TensorWasmExecutor::new(engine);
    let wasm = wat::parse_str(make_adder_wat()).unwrap();
    let spawn_cfg = SpawnConfig::for_tenant(TenantId(1));
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

criterion_group!(call_export, bench_noargs, bench_with_args);
criterion_main!(call_export);
