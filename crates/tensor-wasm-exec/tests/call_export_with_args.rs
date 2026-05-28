// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Coverage for [`TensorWasmExecutor::call_export_with_args`].
//!
//! Verifies that a guest export with a non-trivial signature actually
//! receives its arguments and that the result list survives the trip
//! back through the JSON-shaped public API. Pre-feature, `call_export`
//! only invoked `() -> ()` exports — anything richer was silently
//! degraded. The CLI/API smoke tests exercise the wire path; this test
//! exercises the executor surface in isolation so a future regression
//! at the boundary lights up here rather than as a flaky integration
//! failure two crates away.

use std::sync::Arc;

use serde_json::Value;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};

/// `(i32, i32) -> i32` adder, hand-rolled in WAT so the test does not
/// depend on a Cargo target build step.
fn adder_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (func (export "add") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add)
        )
        "#,
    )
    .expect("wat compiles")
}

/// `(f64) -> f64` doubler, used to confirm the `f64` JSON-shape lane.
fn doubler_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (func (export "double") (param f64) (result f64)
            local.get 0
            local.get 0
            f64.add)
        )
        "#,
    )
    .expect("wat compiles")
}

#[tokio::test]
async fn add_returns_three() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &adder_wasm())
        .await
        .expect("spawn");
    let result = exec
        .call_export_with_args(id, "add", &[WasmArg::I32(1), WasmArg::I32(2)])
        .await
        .expect("call");

    let arr = result.as_array().expect("result is an array");
    assert_eq!(arr.len(), 1, "single i32 result expected; got {arr:?}");
    assert_eq!(arr[0].as_i64(), Some(3));
}

#[tokio::test]
async fn empty_args_unchanged_for_unit_export() {
    // The fast-path branch (`args.is_empty()`) keeps the legacy
    // `() -> ()` codepath intact. The expected JSON shape is an empty
    // array — callers wanting the unit signature should keep using
    // `call_export`, which collapses this to `()`.
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let wasm =
        wat::parse_str(r#"(module (func (export "noop")))"#).expect("noop wat compiles");
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect("spawn");
    let result = exec
        .call_export_with_args(id, "noop", &[])
        .await
        .expect("call");
    assert_eq!(result, Value::Array(Vec::new()));
}

#[tokio::test]
async fn f64_arg_round_trips_through_json() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &doubler_wasm())
        .await
        .expect("spawn");
    let result = exec
        .call_export_with_args(id, "double", &[WasmArg::F64(1.5)])
        .await
        .expect("call");
    let arr = result.as_array().expect("result is array");
    assert_eq!(arr.len(), 1);
    let v = arr[0].as_f64().expect("f64 result");
    // Use direct equality — 1.5 + 1.5 is exactly representable in IEEE-754
    // so no tolerance is needed.
    assert_eq!(v, 3.0);
}

#[tokio::test]
async fn json_to_wasm_arg_conversion_picks_i32_for_small_ints() {
    // Small integers fit i32 → I32; large integers escalate to I64;
    // fractional numbers degrade to F64. This is the contract the CLI
    // and API rely on when translating `--args [1, 2.0]`.
    let cases = [
        (serde_json::json!(0), WasmArg::I32(0)),
        (serde_json::json!(2_147_483_647_i64), WasmArg::I32(i32::MAX)),
        (
            serde_json::json!(2_147_483_648_i64),
            WasmArg::I64(2_147_483_648),
        ),
        (serde_json::json!(2.5_f64), WasmArg::F64(2.5)),
    ];
    for (input, expected) in cases {
        let got = WasmArg::from_json(&input).expect("conversion succeeds");
        assert_eq!(got, expected, "input: {input}");
    }

    // Non-numeric input is rejected with a user-facing message.
    let bad = serde_json::json!("oops");
    assert!(WasmArg::from_json(&bad).is_err());
}
