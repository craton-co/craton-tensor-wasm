// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression pin for audit finding **L-3 (LOW)** — the wasmtime-version-coupled
//! English-substring matching in `executor::classify_instantiation_error`.
//!
//! ## What this guards
//!
//! `classify_instantiation_error` (private; `executor.rs` ~1032-1069) *refines*
//! an opaque [`wasmtime::Error`] raised by the pooling allocator's memory-slot
//! sizing refusal into the typed
//! [`tensor_wasm_exec::executor::ExecError::ModuleMemoryTooLarge`]. It does so
//! by lower-casing the full error chain (`format!("{err:#}")`) and testing for
//! a `"memory"` mention **and** one of the limit phrasings
//! `"exceeds the limit"` / `"exceeds the configured"`.
//!
//! That match is *substring matching on English error text* — wasmtime exposes
//! no structured error type for this condition — so it is **wasmtime-version
//! coupled**. A wasmtime bump that rewords the pooling allocator's memory-size
//! refusal would silently defeat the substring match, and the refinement would
//! quietly stop firing (the error then flows through as the generic
//! `ExecError::Wasmtime`). The degradation is safe-by-construction (the original
//! error is never dropped), but the *typed* refinement callers rely on would
//! vanish without any signal. The in-crate doc comment even still references
//! "wasmtime 25" phrasings while the workspace is pinned to wasmtime 45 — proof
//! that this coupling drifts unnoticed.
//!
//! [`test_classifier_phrasing_still_present_in_real_wasmtime_error`] reproduces
//! the **genuine** pooling-allocator memory-size error from the wasmtime version
//! this crate is actually built against (via wasmtime's own public API, which is
//! a direct dependency of this crate), then asserts the exact substring rule the
//! classifier depends on still matches that real error chain. If a future
//! wasmtime bump rewords the phrasing, this test fails LOUDLY at the source of
//! the coupling — pointing maintainers straight at
//! `classify_instantiation_error` — instead of the refinement degrading in
//! silence.
//!
//! [`spawn_oversized_module_yields_typed_variant_not_generic_wasmtime`] pins the
//! observable contract on the crate's PUBLIC API: an oversized-memory module
//! routed through [`TensorWasmExecutor::spawn_instance`] must surface the typed
//! `ModuleMemoryTooLarge`, never the opaque `ExecError::Wasmtime`. (On the spawn
//! path the static pre-instantiation check is the front-line guard and the
//! classifier is the defense-in-depth backstop; either way callers must see the
//! refined variant.)

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, MemoryBackend, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};

/// The classifier's exact recognition rule, mirrored here verbatim so this
/// test pins precisely what `classify_instantiation_error` keys off. Keep this
/// in lock-step with `executor.rs` — if the classifier's substring set is
/// changed, change it here too (and that is exactly the moment to re-verify the
/// phrasings against the current wasmtime release).
fn classifier_recognizes(err_chain_lowercased: &str) -> bool {
    err_chain_lowercased.contains("memory")
        && (err_chain_lowercased.contains("exceeds the limit")
            || err_chain_lowercased.contains("exceeds the configured"))
}

/// Build a `wasmtime::Engine` whose pooling allocator has a deliberately tiny
/// per-slot memory ceiling, so instantiating a module that declares more linear
/// memory than the slot triggers the genuine pooling memory-size refusal — the
/// very error `classify_instantiation_error` is designed to recognise.
///
/// This uses wasmtime's own public API (a direct dependency of this crate)
/// rather than the project engine, because the project engine *reconciles* the
/// pooling slot size with the module pre-check (`effective_memory_cap`), so on
/// the spawn path the static check intercepts the oversized module before the
/// allocator is ever consulted. To exercise the *allocator's* phrasing directly
/// we deliberately build an unreconciled pool here.
fn provoke_real_pooling_memory_error() -> wasmtime::Error {
    use wasmtime::{Config, Engine, InstanceAllocationStrategy, Module, PoolingAllocationConfig};

    let mut pool = PoolingAllocationConfig::default();
    pool.total_memories(1);
    // 64 KiB == exactly one wasm page: the smallest non-zero slot. Any module
    // declaring >= 2 initial pages exceeds it and the allocator must refuse.
    pool.max_memory_size(64 * 1024);

    let mut cfg = Config::new();
    // Sync instantiation keeps this test self-contained (no async runtime);
    // sync is the default, so no async_support call is needed.
    // The pooling slot ceiling (`max_memory_size`) must not exceed the
    // wasmtime-side memory reservation, or the pool refuses to *construct*
    // (a different, pool-construction-time bail). Give the reservation room.
    cfg.memory_reservation(4 * 1024 * 1024);
    cfg.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));

    let engine = Engine::new(&cfg).expect("pooling engine builds");

    // Declare 64 initial pages (4 MiB) — far above the 64 KiB slot ceiling, so
    // the pooling allocator refuses the memory with its "minimum byte size ...
    // exceeds the limit" phrasing. Depending on the wasmtime version this
    // refusal surfaces either at module *compilation* (wasmtime 45 validates
    // pool fit inside `Module::new`) or at *instantiation* (older releases).
    // Capture the genuine error wherever it occurs so the phrasing pin stays
    // robust across that internal change.
    let wasm = wat::parse_str(r#"(module (memory (export "mem") 64))"#).expect("valid wat");
    match Module::new(&engine, &wasm) {
        // Compile-time refusal (wasmtime 45 pooling allocator).
        Err(err) => err,
        // Older behavior: the module compiles and the allocator refuses at
        // instantiation time instead.
        Ok(module) => {
            let mut store = wasmtime::Store::new(&engine, ());
            match wasmtime::Instance::new(&mut store, &module, &[]) {
                Ok(_) => panic!(
                    "expected the pooling allocator to REFUSE a 4 MiB memory in a 64 KiB slot; \
                     if instantiation now succeeds the test fixture no longer provokes the \
                     memory-size refusal classify_instantiation_error keys off"
                ),
                Err(err) => err,
            }
        }
    }
}

/// LOUD pin on the wasmtime-coupled phrasing. Fails the moment a wasmtime bump
/// rewords the pooling memory-size refusal such that the classifier's substring
/// rule no longer matches — which is precisely when
/// `executor::classify_instantiation_error` would silently stop refining
/// `ModuleMemoryTooLarge`.
#[test]
fn test_classifier_phrasing_still_present_in_real_wasmtime_error() {
    let err = provoke_real_pooling_memory_error();
    // Exactly how the classifier inspects the error: the full `{:#}` chain,
    // lower-cased.
    let chain = format!("{err:#}").to_ascii_lowercase();

    assert!(
        classifier_recognizes(&chain),
        "WASMTIME PHRASING DRIFT (audit L-3): the genuine pooling-allocator \
         memory-size error no longer matches the substring rule in \
         executor::classify_instantiation_error, so that function will SILENTLY \
         stop refining wasmtime errors into the typed \
         ExecError::ModuleMemoryTooLarge. Re-read the pooling memory_pool error \
         strings for the current wasmtime release and update BOTH the substring \
         match in executor.rs AND `classifier_recognizes` in this test. \
         Real error chain was: {chain:?}"
    );

    // Sub-assertion: which of the two recognised phrasings carried it. This is
    // informational (it always passes for whichever branch matched) but makes
    // the matched signature visible in a `--nocapture` run when diagnosing a
    // future drift.
    let has_limit = chain.contains("exceeds the limit");
    let has_configured = chain.contains("exceeds the configured");
    assert!(
        has_limit || has_configured,
        "neither limit phrasing present in real wasmtime error: {chain:?}"
    );
}

/// Engine cap used by the public-API test below. 256 MiB; the oversized module
/// declares 4 GiB so it trips the cap by a wide, unambiguous margin.
const ENGINE_CAP: usize = 256 * 1024 * 1024;

/// Maximum a wasm32 linear memory may declare (65 536 pages x 64 KiB = 4 GiB).
/// Spec-valid, so the rejection is purely the host's doing, not a parse error.
const FOUR_GIB_PAGES: u32 = 65_536;

fn make_executor() -> TensorWasmExecutor {
    let cfg = EngineConfig {
        max_memory_bytes: ENGINE_CAP,
        // Pooling backend keeps the test deterministic on hosts without CUDA;
        // the choice does not change which typed variant callers observe.
        backend: MemoryBackend::PoolingMpk {
            max_memories: 4,
            memory_bytes: ENGINE_CAP,
        },
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    TensorWasmExecutor::new(engine)
}

/// PUBLIC-API contract pin: an oversized-memory module routed through
/// `spawn_instance` must surface the TYPED `ModuleMemoryTooLarge`, never the
/// generic `ExecError::Wasmtime`. This is the observable refinement the L-3
/// finding is about — whether enforced by the static pre-check (front line) or
/// by `classify_instantiation_error` (backstop), callers must always get the
/// typed variant. If a regression ever lets the opaque `Wasmtime` escape on
/// this path, this fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_oversized_module_yields_typed_variant_not_generic_wasmtime() {
    let exec = make_executor();
    let wasm = wat::parse_str(format!(
        "(module (memory (export \"mem\") {FOUR_GIB_PAGES}))"
    ))
    .expect("valid wat");

    let err = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
        .await
        .expect_err("a 4 GiB memory declaration must be rejected");

    // The whole point of L-3: callers must NOT see the opaque variant.
    assert!(
        !matches!(err, ExecError::Wasmtime(_)),
        "audit L-3: oversized-memory module surfaced the OPAQUE \
         ExecError::Wasmtime instead of the typed ModuleMemoryTooLarge — the \
         refinement contract callers depend on has broken. Got: {err:?}"
    );
    match err {
        ExecError::ModuleMemoryTooLarge {
            requested_bytes,
            limit_bytes,
        } => {
            assert_eq!(limit_bytes as usize, ENGINE_CAP);
            assert!(
                requested_bytes as usize > ENGINE_CAP,
                "requested {requested_bytes} must exceed cap {ENGINE_CAP}"
            );
        }
        other => panic!("expected typed ModuleMemoryTooLarge, got {other:?}"),
    }
    assert_eq!(
        exec.live_count(),
        0,
        "no instance must survive a rejected oversized spawn"
    );
}
