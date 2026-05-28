// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T37 — InstancePool wired through executor invoke path.
//!
//! Validates the per-`(tenant, module_hash)` channel, the pre-spawn loop,
//! and the reset-on-return semantics. Sister test file to the v0.3.6
//! `instance_pool_scaffold.rs` (which pins the pre-T37 fall-through
//! behaviour). Both must pass: scaffold pins the no-pool default,
//! invoke pins the warm-channel path.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};
use tensor_wasm_exec::instance_pool::{InstancePool, InstancePoolConfig};

/// A module that exports a `tick` function returning the post-increment
/// value of a mutable global `counter`. Each `tick` invocation against a
/// SHARED instance returns the next integer; a fresh instance starts at
/// 1. We use this to distinguish "same instance reused" (counter keeps
/// climbing) from "fresh instance reset" (counter returns to 1).
fn counter_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (global $counter (mut i32) (i32.const 0))
          (func (export "tick") (result i32)
            (global.set $counter
              (i32.add (global.get $counter) (i32.const 1)))
            (global.get $counter)))
        "#,
    )
    .unwrap()
}

fn noop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).unwrap()
}

#[tokio::test]
async fn invoke_without_pool_uses_spawn_path() {
    // Sanity-check the no-pool fall-through: `invoke` with no attached
    // pool must spawn fresh on every call (no warm reuse).
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let wasm = noop_wasm();

    let cfg = SpawnConfig::for_tenant(TenantId(1));
    let r = exec
        .invoke(cfg, &wasm, "noop", &[])
        .await
        .expect("invoke");
    // No-op export returns an empty result list.
    assert_eq!(r, serde_json::Value::Array(Vec::new()));
    // Instance was auto-terminated by the spawn_instance + then_terminate
    // path under the no-pool branch.
    assert_eq!(exec.live_count(), 0);
}

#[tokio::test]
async fn pool_pre_spawns_warm_instances_on_first_acquire() {
    // With `warm_instances_per_tuple = 2`, the first acquire for a
    // (tenant, module-hash) tuple must seed the channel with 2 warm
    // instances BEFORE returning a handle. The pool's `warm_count`
    // tracks live channel occupancy.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 2,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    let cfg = SpawnConfig::for_tenant(TenantId(1));
    let _r = exec
        .invoke(cfg, &wasm, "noop", &[])
        .await
        .expect("invoke");

    // After one invoke: pool was pre-warmed (2 instances), one was
    // drawn for the invoke, and one was placed back as a fresh
    // replacement post-release. Net warm count: 2 (the original
    // unused seed instance + the fresh replacement).
    assert_eq!(
        pool.warm_count(),
        2,
        "after one invoke the warm channel should still hold the unused seed instance + a fresh replacement"
    );
    // The hit_count reflects the warm draw on this invoke.
    assert_eq!(pool.hit_count(), 1);
    assert_eq!(pool.miss_count(), 0);
}

#[tokio::test]
async fn pool_reuse_resets_guest_globals_to_initial_state() {
    // The reset-on-release contract: a guest global mutated by a
    // previous invoke must NOT survive into the next invoke against
    // the same (tenant, module) tuple. The counter module increments
    // a global and returns the new value; a fresh-on-every-invoke
    // pool must therefore return 1 on every call.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 1,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = counter_wasm();
    let tenant = TenantId(1);

    for i in 0..3 {
        let cfg = SpawnConfig::for_tenant(tenant);
        let result = exec
            .invoke(cfg, &wasm, "tick", &[])
            .await
            .expect("invoke");
        // Each invoke against a freshly-reset (or freshly-spawned)
        // instance must report `1` — the global starts at 0 and the
        // function increments before returning. If the pool returned
        // a recycled-without-reset instance, iterations 1+ would
        // observe values > 1.
        let n = result.as_array().and_then(|a| a.first()).and_then(|v| v.as_i64());
        assert_eq!(
            n,
            Some(1),
            "iteration {i}: reset must scrub global state; got {result:?}",
        );
    }
}

#[tokio::test]
async fn pool_isolates_instances_per_tenant() {
    // Two different tenants invoking the SAME module must each see
    // their own channel — never serve tenant B an instance previously
    // used by tenant A.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 1,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();

    // Tenant A — first invoke seeds tenant A's channel.
    let _ = exec
        .invoke(SpawnConfig::for_tenant(TenantId(1)), &wasm, "noop", &[])
        .await
        .expect("invoke A1");
    let warm_after_a = pool.warm_count();

    // Tenant B — first invoke seeds tenant B's channel (separate
    // entry, separate channel). The warm count grows because tenant
    // B's seed instance is independent of tenant A's.
    let _ = exec
        .invoke(SpawnConfig::for_tenant(TenantId(2)), &wasm, "noop", &[])
        .await
        .expect("invoke B1");
    let warm_after_b = pool.warm_count();

    assert!(
        warm_after_b > warm_after_a,
        "tenant B's first invoke must allocate its own channel entry; \
         warm_after_a={warm_after_a}, warm_after_b={warm_after_b}",
    );
}

#[tokio::test]
async fn pool_isolates_instances_per_module() {
    // Two different modules under the SAME tenant must each get their
    // own channel — never serve module M' an instance compiled from
    // module M.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 1,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm_a = noop_wasm();
    let wasm_b = counter_wasm();
    let tenant = TenantId(7);

    // Module A — seeds channel A.
    let _ = exec
        .invoke(SpawnConfig::for_tenant(tenant), &wasm_a, "noop", &[])
        .await
        .expect("invoke A");
    let warm_after_a = pool.warm_count();

    // Module B — must seed its own channel (different hash key).
    let _ = exec
        .invoke(SpawnConfig::for_tenant(tenant), &wasm_b, "tick", &[])
        .await
        .expect("invoke B");
    let warm_after_b = pool.warm_count();

    assert!(
        warm_after_b > warm_after_a,
        "module B's first invoke must allocate its own channel entry; \
         warm_after_a={warm_after_a}, warm_after_b={warm_after_b}",
    );
}

#[tokio::test]
async fn pool_falls_back_to_spawn_on_empty_channel() {
    // With `warm_instances_per_tuple = 0`, the channel is empty on
    // first acquire. Acquire must fall through to spawn_instance
    // rather than blocking. The miss counter records the fall-through.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 0,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    let _ = exec
        .invoke(SpawnConfig::for_tenant(TenantId(1)), &wasm, "noop", &[])
        .await
        .expect("invoke");

    // First invoke missed (channel was empty post-init) and fell
    // through to spawn_instance.
    assert_eq!(pool.miss_count(), 1);
    assert_eq!(pool.hit_count(), 0);
}

#[tokio::test]
async fn pool_typed_args_pass_through_to_export() {
    // Sanity-check that T33's typed args survive the pool path
    // intact. The counter module ignores its inputs but the executor
    // must still accept them without changing the export's result
    // shape.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 1,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    // Use a module that adds two i32 args.
    let adder = wat::parse_str(
        r#"
        (module
          (func (export "add") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add))
        "#,
    )
    .unwrap();

    let cfg = SpawnConfig::for_tenant(TenantId(1));
    let result = exec
        .invoke(cfg, &adder, "add", &[WasmArg::I32(2), WasmArg::I32(40)])
        .await
        .expect("invoke add");
    let n = result.as_array().and_then(|a| a.first()).and_then(|v| v.as_i64());
    assert_eq!(n, Some(42));
}

#[tokio::test]
async fn pool_shutdown_drains_warm_channels() {
    // `InstancePool::shutdown` must drain every warm channel and
    // release the executor slots it held.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 3,
        max_total_warm: 0,
    }));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    // Drive one invoke so the pool seeds its warm channel.
    let _ = exec
        .invoke(SpawnConfig::for_tenant(TenantId(1)), &wasm, "noop", &[])
        .await
        .expect("invoke");
    assert!(pool.warm_count() > 0);
    let pre_count = exec.instances_len();

    pool.shutdown(&exec);
    assert_eq!(pool.warm_count(), 0);
    // The executor's live-instance counter dropped by the number of
    // warm instances that were drained.
    assert!(exec.instances_len() < pre_count);
}
