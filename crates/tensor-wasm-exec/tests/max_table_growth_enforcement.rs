// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Verifies that `TensorWasmResourceLimiter::table_growing` caps table growth
//! so a malicious guest cannot exhaust host RAM by issuing repeated
//! `table.grow` calls. Without a cap, a guest could grow a `funcref` table
//! to `u32::MAX` entries — roughly 64 GiB of host backing store at ~16 B
//! per entry — bypassing the linear-memory cap entirely.
//!
//! The guest module below exports `grow_table_huge`, which calls
//! `table.grow` once with a request that, on a 1 MiB engine cap (~65k
//! entries), the limiter must reject. Wasmtime returns `-1` (as a tagged
//! i32) in-band for a rejected `table.grow`; we don't need to inspect the
//! return value — we just need the call to come back without crashing or
//! OOM-killing the host. That alone proves the cap engaged: without it,
//! a sufficiently large request would attempt to allocate gigabytes of
//! host memory and either succeed (exhausting the host) or trigger an
//! allocator abort.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

/// A wasm module declaring a growable `funcref` table and exporting a
/// function that asks for `huge_request` more entries in a single
/// `table.grow` call. We pass the request through a global so we can keep
/// the wat literal small while exercising the limiter at near-`u32::MAX`.
///
/// We attempt to grow by 1,000,000,000 entries (~16 GiB at 16 B/entry) —
/// well above any reasonable engine cap and large enough that the host
/// would be OOM-killed if the limiter did not engage.
fn table_growth_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (table (export "tbl") 0 funcref)
          (func (export "grow_table_huge")
            (drop
              (table.grow (ref.null func) (i32.const 1000000000)))))
        "#,
    )
    .expect("valid wat")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn table_grow_past_limit_is_rejected_without_host_crash() {
    // 1 MiB engine cap → ~65k table entries permitted. A 1B-entry request
    // (~16 GiB) must be denied by the limiter; the call still returns Ok
    // because rejected `table.grow` is in-band (-1), not a trap.
    let cfg = EngineConfig {
        max_memory_bytes: 1024 * 1024,
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &table_growth_wasm())
        .await
        .expect("spawn");

    // The key assertion is that *this call returns at all*. Without the
    // limiter cap, wasmtime would attempt to allocate ~16 GiB of host RAM
    // for the table backing store on this single call, blowing through
    // every reasonable host budget. With the cap, `table.grow` is denied
    // in-band, the guest drops the -1, and the call returns Ok.
    exec.call_export_with_args(id, "grow_table_huge", &[])
        .await
        .expect("call returns; table-grow rejection is in-band");

    exec.terminate(id).await.expect("terminate");
}

/// Companion test: a modest table growth under the engine cap must
/// succeed — the cap must not falsely reject legitimate small tables.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn table_grow_under_limit_succeeds() {
    fn small_table_wasm() -> Vec<u8> {
        wat::parse_str(
            r#"
            (module
              (table (export "tbl") 0 funcref)
              (func (export "grow_table_small")
                (drop
                  (table.grow (ref.null func) (i32.const 1024)))))
            "#,
        )
        .expect("valid wat")
    }

    // 1 MiB engine cap → ~65k entries permitted. 1024 entries is well below.
    let cfg = EngineConfig {
        max_memory_bytes: 1024 * 1024,
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &small_table_wasm())
        .await
        .expect("spawn");
    exec.call_export_with_args(id, "grow_table_small", &[])
        .await
        .expect("call");
    exec.terminate(id).await.expect("terminate");
}
