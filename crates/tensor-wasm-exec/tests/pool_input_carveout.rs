// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! TESTS finding 8 (input carve-out): a pooled invoke that stages guest input
//! must NEVER let request N+1 observe request N's staged input bytes.
//!
//! The pool recycles warm instances per `(tenant, module-hash)` tuple by
//! re-instantiating them on release. A warm instance bakes the FIRST caller's
//! `SpawnConfig::input` into its `InstanceState` at build time, and
//! `register_pooled_instance` resets only the `instance_id`, never re-stages a
//! later caller's input. So serving an input-bearing invoke from the warm
//! channel would leak the warming caller's prompt to a later, different caller
//! (same-tenant input bleed). The pool's input carve-out
//! (`instance_pool.rs::acquire`) closes this: an invoke with non-empty
//! `cfg.input` ALWAYS spawns a fresh instance (its own input staged) and is
//! NEVER recycled (`origin: None` → dropped on release).
//!
//! This test drives two input-bearing invokes through the SAME pool tuple with
//! DIFFERENT staged inputs and asserts each invoke's guest reads ITS OWN input,
//! not the other's — and that an input-bearing invoke leaves no warm instance
//! behind for a later no-input invoke to read.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};
use tensor_wasm_exec::{InstancePool, InstancePoolConfig};

/// Guest that pulls its staged input via `wasi:tensor/host@0.1.0`
/// (`input-len` / `read-input`) into linear memory at offset 0 and returns the
/// FIRST staged byte (or -1 when nothing is staged). The first byte is enough
/// to tell whose input the instance saw: input `"A.."` → 65, `"B.."` → 66.
fn input_echo_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (import "wasi:tensor/host@0.1.0" "input-len"
            (func $input_len (result i32)))
          (import "wasi:tensor/host@0.1.0" "read-input"
            (func $read_input (param i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "first_byte") (result i32)
            (local $len i32)
            (local.set $len (call $input_len))
            ;; Nothing staged => sentinel -1.
            (if (i32.eqz (local.get $len))
              (then (return (i32.const -1))))
            ;; Copy the staged bytes to offset 0, then load the first byte.
            (drop (call $read_input (i32.const 0) (local.get $len)))
            (i32.load8_u (i32.const 0)))
        )
    "#,
    )
    .unwrap()
}

/// Build an executor with a warm pool (2 warm instances per tuple) so the
/// recycling path is genuinely exercised — without warm instances the carve-out
/// would be vacuously satisfied.
fn pooled_executor() -> TensorWasmExecutor {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::new(2, 32)));
    TensorWasmExecutor::new(engine).with_instance_pool(pool)
}

/// Drive `first_byte` through the pool with the given staged input; return the
/// first byte the guest observed.
async fn invoke_first_byte(exec: &TensorWasmExecutor, wasm: &[u8], input: &[u8]) -> i64 {
    let cfg = SpawnConfig::for_tenant(TenantId(1)).with_input(input.to_vec());
    let value = exec
        .invoke(cfg, wasm, "first_byte", &[])
        .await
        .expect("invoke");
    // `() -> i32` projects to a one-element JSON array.
    value
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_i64())
        .expect("first_byte result must be a single integer")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pooled_input_does_not_bleed_across_requests() {
    let exec = pooled_executor();
    let wasm = input_echo_wasm();

    // Request 1 stages "A..." — the guest must see 'A' (65).
    let first = invoke_first_byte(&exec, &wasm, b"Apple").await;
    assert_eq!(first, 65, "request 1 must read its own input ('A')");

    // Request 2 stages "B..." on the SAME (tenant, module) tuple. If request
    // 1's instance had been recycled into the warm channel WITH its staged
    // input intact, request 2 could read 'A' (65) instead of its own 'B' (66).
    // The carve-out guarantees request 2 reads ITS input.
    let second = invoke_first_byte(&exec, &wasm, b"Banana").await;
    assert_eq!(
        second, 66,
        "request 2 read a STALE input — pool leaked request 1's staged bytes (input bleed)"
    );

    // A third request with DIFFERENT input again sees only its own bytes.
    let third = invoke_first_byte(&exec, &wasm, b"Cherry").await;
    assert_eq!(third, 67, "request 3 must read its own input ('C')");

    // No input-bearing invoke may leave a warm instance behind: input-bearing
    // spawns are stamped `origin: None` and dropped on release, never recycled.
    // A subsequent NO-input invoke must therefore see nothing staged (-1),
    // proving it did not draw a warm instance still carrying an earlier prompt.
    let no_input_cfg = SpawnConfig::for_tenant(TenantId(1));
    let no_input_value = exec
        .invoke(no_input_cfg, &wasm, "first_byte", &[])
        .await
        .expect("invoke no-input");
    let no_input_byte = no_input_value
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_i64())
        .expect("result");
    assert_eq!(
        no_input_byte, -1,
        "a no-input invoke saw staged bytes — an input-bearing instance was wrongly recycled"
    );

    // All instances cleaned up after each invoke (invoke terminates / releases).
    assert_eq!(exec.live_count(), 0, "no instance must survive the invokes");
}
