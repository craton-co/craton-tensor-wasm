// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Coverage for [`SpawnConfig::with_args`] — the typed argument list
//! carried on the spawn config alongside `deadline`.
//!
//! T33 (v0.4): the API gateway and CLI both attach the parsed argument
//! list to the `SpawnConfig` via `with_args` before calling
//! `spawn_instance`. The field is documented as the canonical carrier
//! for the first `call_export_with_args` invocation against the
//! instance — this test pins the surface (default empty, builder
//! captures the supplied list, instance still spawns successfully)
//! so future refactors that touch the spawn path keep the contract
//! intact.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor, WasmArg};

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

#[test]
fn spawn_config_default_args_is_empty() {
    let cfg = SpawnConfig::for_tenant(TenantId(1));
    assert!(cfg.args.is_empty(), "args defaults to empty vec");
}

#[test]
fn with_args_captures_supplied_list() {
    let cfg =
        SpawnConfig::for_tenant(TenantId(1)).with_args(vec![WasmArg::I32(1), WasmArg::I32(2)]);
    assert_eq!(cfg.args.len(), 2);
    assert_eq!(cfg.args[0], WasmArg::I32(1));
    assert_eq!(cfg.args[1], WasmArg::I32(2));
}

/// Confirm the spawn path tolerates a SpawnConfig with `args` populated.
/// Pre-T33 the field was unused; T33 wires the CLI / API gateways to
/// populate it. The executor still consults the explicit
/// `call_export_with_args` argument slice, but a populated `cfg.args`
/// must not crash the spawn or wedge the instance.
#[tokio::test]
async fn spawn_with_args_then_call_export_succeeds() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let cfg =
        SpawnConfig::for_tenant(TenantId(1)).with_args(vec![WasmArg::I32(3), WasmArg::I32(4)]);
    let id = exec
        .spawn_instance(cfg, &adder_wasm())
        .await
        .expect("spawn");

    // Mirror the API gateway's flow: pass args explicitly to the call.
    let result = exec
        .call_export_with_args(id, "add", &[WasmArg::I32(3), WasmArg::I32(4)])
        .await
        .expect("call");
    let arr = result.as_array().expect("array");
    assert_eq!(arr[0].as_i64(), Some(7));
}

#[test]
fn with_args_chains_with_other_builders() {
    use std::time::Duration;
    // Chained builders compose cleanly — `args`, `deadline` are
    // independent fields and order does not matter.
    let cfg = SpawnConfig::for_tenant(TenantId(7))
        .with_deadline(Duration::from_secs(5))
        .with_args(vec![WasmArg::F64(2.5)]);
    assert_eq!(cfg.tenant_id, TenantId(7));
    assert_eq!(cfg.deadline, Some(Duration::from_secs(5)));
    assert_eq!(cfg.args, vec![WasmArg::F64(2.5)]);
}
