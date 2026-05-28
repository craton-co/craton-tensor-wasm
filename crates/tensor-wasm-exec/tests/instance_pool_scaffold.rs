// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Scaffold-level tests for roadmap feature #5 (pre-instantiated instance
//! pool). These pin the public API shape and the v0.3.6 fall-through
//! semantics. The real warm-pool behaviour (channel draw, reset on
//! return) lands in v0.4 — at that point these tests should still pass
//! unchanged, plus new tests will exercise the warm path proper.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};
use tensor_wasm_exec::instance_pool::{InstancePool, InstancePoolConfig};

fn noop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).unwrap()
}

#[test]
fn pool_default_config_is_disabled() {
    let cfg = InstancePoolConfig::default();
    assert_eq!(
        cfg.warm_instances_per_tuple, 0,
        "default config must leave the pool disabled (no warm instances)",
    );
    assert_eq!(
        cfg.max_total_warm, 0,
        "default config must leave the global warm cap unbounded (0 = unlimited)",
    );
}

#[tokio::test]
async fn pool_with_zero_warm_falls_through_to_spawn() {
    let mut engine = TensorWasmEngine::new().expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);

    let pool = InstancePool::new(InstancePoolConfig {
        warm_instances_per_tuple: 0,
        max_total_warm: 0,
    });

    // With no warm instances configured, acquire must succeed via the
    // fall-through path (i.e. it just calls spawn_instance).
    let wasm = noop_wasm();
    let cfg = SpawnConfig::for_tenant(TenantId(1));
    let pooled = pool
        .acquire(&exec, &wasm, cfg)
        .await
        .expect("acquire should succeed via spawn fall-through");

    // warm_count must remain 0 — nothing was actually pooled.
    assert_eq!(pool.warm_count(), 0);

    // The pool surfaces the assigned instance id; the underlying
    // executor sees the spawn (live count == 1).
    let id = pooled.into_inner();
    assert_eq!(exec.live_count(), 1);

    // Clean up — until v0.4 reset-on-return lands, callers terminate
    // explicitly.
    exec.terminate(id).await.expect("terminate");
    assert_eq!(exec.live_count(), 0);
}

#[tokio::test]
async fn executor_with_instance_pool_builder_wires_pool() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::default()));
    let exec = TensorWasmExecutor::new(engine).with_instance_pool(Arc::clone(&pool));

    // The accessor returns the same Arc we passed in (pointer equality).
    let attached = exec
        .instance_pool()
        .expect("pool should be attached after with_instance_pool");
    assert!(
        Arc::ptr_eq(attached, &pool),
        "instance_pool() should return the same Arc passed to with_instance_pool",
    );
}

#[test]
fn executor_without_pool_returns_none() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    assert!(exec.instance_pool().is_none());
}
