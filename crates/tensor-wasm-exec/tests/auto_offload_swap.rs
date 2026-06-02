// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the opt-in auto-offload swap
//! ([`EngineConfig::auto_offload`]).
//!
//! Invariants pinned here:
//!   1. With `auto_offload = true` and a JIT cache attached, a vector-add
//!      style module whose hot function clears the (tuned) offload threshold
//!      is rewritten into a dispatch-trampoline module, instantiated, and
//!      runs end-to-end without trapping. The rewrite is observable two ways:
//!      the JIT cache is pre-populated for the swapped function, and the
//!      offloaded `add` returns the host dispatch stub's documented
//!      pass-through result (the first parameter echoed) rather than the
//!      CPU sum — proving Wasmtime executed the trampoline, not the original
//!      body.
//!   2. A non-offloadable module (no hot v128 loop) falls back gracefully:
//!      it runs unchanged on the CPU path and produces the real CPU result.
//!   3. With `auto_offload = false` (the default) the same hot module runs
//!      its original CPU body — the swap is strictly opt-in.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};
use tensor_wasm_jit::cache::KernelCache;
use tensor_wasm_jit::detector::DetectorConfig;

/// A hot `add(a, b) -> a + b` whose body is dominated by a v128 loop so the
/// detector flags it for offload (under the tuned threshold below). `memory`
/// is exported so the JIT host imports (dispatch / alloc / free) can reach it
/// via `Caller::get_export("memory")`. Mirrors the fixture the jit crate's
/// own e2e test uses.
const HOT_ADD_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "add") (param $a i32) (param $b i32) (result i32)
        (local $v v128)
        (loop $L
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
          (local.set $v (i32x4.add (local.get $v) (local.get $v)))
        )
        (i32.add (local.get $a) (local.get $b))
      )
    )
"#;

/// A straight-line `add(a, b) -> a + b` with no hot loop or v128 ops — the
/// detector never flags this, so it must run unchanged on the CPU path even
/// with auto-offload enabled.
const PLAIN_ADD_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "add") (param $a i32) (param $b i32) (result i32)
        (i32.add (local.get $a) (local.get $b)))
    )
"#;

/// Aggressive detector thresholds so the realistic v128-loop fixture clears
/// the offload bar (the production default of 0.8 is essentially unreachable
/// by real wasm; the jit crate's own e2e tests tune to 0.05 for the same
/// reason).
fn aggressive_detector() -> DetectorConfig {
    DetectorConfig {
        v128_ratio_threshold: 0.05,
        min_trip_count: 64,
    }
}

fn first_i32(v: &serde_json::Value) -> i64 {
    v.as_array()
        .and_then(|a| a.first())
        .and_then(|n| n.as_i64())
        .unwrap_or_else(|| panic!("expected a single-element i32 array, got {v:?}"))
}

#[tokio::test]
async fn auto_offload_rewrites_and_runs_hot_module() {
    let cache = Arc::new(KernelCache::new());
    let cfg = EngineConfig {
        auto_offload: true,
        auto_offload_detector: Some(aggressive_detector()),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine).with_jit_cache(cache.clone());

    let wasm = wat::parse_str(HOT_ADD_WAT).expect("parse hot add");
    let tenant = TenantId(11);
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(tenant), &wasm)
        .await
        .expect("spawn with auto_offload must succeed");

    // The rewrite must have pre-populated the per-tenant JIT cache for the
    // swapped function — proof the swap path actually rewrote the module.
    assert!(
        !cache.is_empty(),
        "auto_offload rewrite must pre-populate the JIT kernel cache",
    );

    // The offloaded function runs end-to-end (alloc → store args → dispatch
    // → load results → free) without trapping. The executor wires the
    // DEFAULT dispatch stub, which echoes the first `results_len` bytes of
    // the args region back as the result (documented pass-through). So for
    // `add(a, b) -> i32`, the offloaded result is `a` — distinct from the CPU
    // sum `a + b` whenever `b != 0`, which is exactly what proves the
    // trampoline (not the original body) executed.
    #[cfg(feature = "cuda")]
    {
        let out = exec
            .call_export_with_args(id, "add", &[WasmArg::I32(7), WasmArg::I32(5)])
            .await
            .expect("offloaded add must run without trapping");
        assert_eq!(
            first_i32(&out),
            7,
            "offloaded add must return the dispatch stub's echoed first arg (7), \
             not the CPU sum (12) — proving the trampoline executed",
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        // On no-CUDA builds, a cache hit deopts and traps to prevent silently
        // returning wrong (echoed) data — see the correctness stub fix in `jit_dispatch.rs`.
        let err = exec
            .call_export_with_args(id, "add", &[WasmArg::I32(7), WasmArg::I32(5)])
            .await
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("wasm trap: wasm `unreachable` instruction executed")
                || msg.contains("unreachable"),
            "no-CUDA offload must trap with unreachable, got: {msg}"
        );
    }

    exec.terminate(id).await.expect("terminate");
}

#[tokio::test]
async fn non_offloadable_module_falls_back_gracefully() {
    let cache = Arc::new(KernelCache::new());
    let cfg = EngineConfig {
        auto_offload: true,
        auto_offload_detector: Some(aggressive_detector()),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine).with_jit_cache(cache.clone());

    // A plain straight-line add: never flagged for offload, so the rewrite is
    // skipped and the original CPU body runs.
    let wasm = wat::parse_str(PLAIN_ADD_WAT).expect("parse plain add");
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(22)), &wasm)
        .await
        .expect("spawn must succeed and fall back to the original module");

    // No swap happened → cache untouched.
    assert!(
        cache.is_empty(),
        "a non-offloadable module must not pre-populate the JIT cache",
    );

    // The CPU body runs and produces the real sum.
    let out = exec
        .call_export_with_args(id, "add", &[WasmArg::I32(7), WasmArg::I32(5)])
        .await
        .expect("plain add runs on the CPU path");
    assert_eq!(
        first_i32(&out),
        12,
        "non-offloaded add must return the real CPU sum",
    );

    exec.terminate(id).await.expect("terminate");
}

#[tokio::test]
async fn auto_offload_disabled_runs_original_body() {
    // Default config: auto_offload = false. Even the hot module runs its
    // original CPU body — the swap is strictly opt-in.
    let cache = Arc::new(KernelCache::new());
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine).with_jit_cache(cache.clone());

    let wasm = wat::parse_str(HOT_ADD_WAT).expect("parse hot add");
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(33)), &wasm)
        .await
        .expect("spawn with auto_offload disabled must succeed");

    assert!(
        cache.is_empty(),
        "with auto_offload disabled the cache must stay empty (consultation-only)",
    );

    let out = exec
        .call_export_with_args(id, "add", &[WasmArg::I32(7), WasmArg::I32(5)])
        .await
        .expect("original CPU body runs");
    assert_eq!(
        first_i32(&out),
        12,
        "with the swap disabled the original CPU add must return the real sum",
    );

    exec.terminate(id).await.expect("terminate");
}

#[tokio::test]
async fn auto_offload_without_jit_cache_falls_back() {
    // auto_offload enabled but NO jit cache attached: the rewritten module's
    // trampoline imports would be unlinkable, so the executor must skip the
    // rewrite and run the original body. The spawn must still succeed.
    let cfg = EngineConfig {
        auto_offload: true,
        auto_offload_detector: Some(aggressive_detector()),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine); // no with_jit_cache

    let wasm = wat::parse_str(HOT_ADD_WAT).expect("parse hot add");
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(44)), &wasm)
        .await
        .expect("spawn must fall back and succeed without a jit cache");

    let out = exec
        .call_export_with_args(id, "add", &[WasmArg::I32(7), WasmArg::I32(5)])
        .await
        .expect("fallback CPU body runs");
    assert_eq!(
        first_i32(&out),
        12,
        "without a jit cache the original CPU add must run (no unlinkable trampoline)",
    );

    exec.terminate(id).await.expect("terminate");
}
