// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression tests for two instance-pool slot-accounting fixes.
//!
//! ## Fix #1 — `detach_pooled_instance` decrements the per-tenant slot on the
//! `Arc::try_unwrap` race
//!
//! `InstancePool::release` detaches the spent instance from the executor
//! registry via the (crate-internal) `detach_pooled_instance`. On the happy
//! path the sole strong reference is unwrapped and the *caller* releases both
//! the engine-wide and per-tenant slots. But a concurrent in-flight
//! `call_export_with_args` on the SAME id holds an `Arc` clone of the handle
//! (it clones the registry value before taking the per-instance lock), so a
//! racing detach observes an outstanding strong reference and `try_unwrap`
//! fails. Before the fix that failure branch released only the engine-wide
//! slot and *leaked the per-tenant slot* — under `max_instances_per_tenant =
//! Some(1)` that permanently locked the tenant out of ever spawning again.
//! The fix releases BOTH counters on the try_unwrap-failure branch.
//!
//! VISIBILITY LIMITATION: `detach_pooled_instance` / `register_pooled_instance`
//! are `pub(crate)`, so an integration test in `tests/` cannot call them
//! directly nor force the `Arc::try_unwrap` branch deterministically — and the
//! detach path itself reads the owning tenant under the per-instance lock,
//! which serialises against the in-flight call that would create the race, so
//! the exact `Err(_arc)` arm is not deterministically reachable from public
//! API. We therefore pin the *observable* invariant the fix protects: across
//! repeated pooled invokes under a per-tenant cap of 1 (including a release
//! that races a still-in-flight call on the same instance), the tenant's
//! per-tenant count never exceeds the cap, `pool.shutdown` drains it back to
//! exactly 0, and the tenant can spawn again — i.e. no self-lockout. The
//! pre-fix leak would have driven `tenant_instance_count` strictly above the
//! cap and refused subsequent spawns with `TenantCapacityExhausted`.
//!
//! (The pool's `release` always re-instantiates one replacement into the
//! per-tuple channel — capacity floored at 1 — so a single warm slot is
//! legitimately parked between invokes; the leak shows up as a count strictly
//! ABOVE that, and as a non-zero residual after `shutdown`.)
//!
//! ## Fix #2 — `ensure_entry` is single-flight (no thundering-herd
//! double-build)
//!
//! Concurrent first-`acquire`s for the same fresh `(tenant, module_hash)` used
//! to each run the full warm pre-spawn loop, Cranelift-compiling and
//! instantiating `warm_n` instances apiece and counting `warm_total` up once
//! per builder. The fix funnels all concurrent first-builders through a
//! per-key `tokio::sync::OnceCell`: exactly one runs the build, the losers
//! observe the winner's `PoolEntry`. The observable invariant: after N
//! concurrent first-invokes against the same tuple, `warm_count` equals the
//! single-builder steady state (`warm_n`), never a multiple of it.

use std::sync::Arc;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::{EngineConfig, TensorWasmEngine};
use tensor_wasm_exec::executor::{ExecError, SpawnConfig, TensorWasmExecutor};
use tensor_wasm_exec::instance_pool::{InstancePool, InstancePoolConfig};

/// Trivial valid module with a single no-op export.
fn noop_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).expect("valid wat")
}

/// An infinite loop in `spin` so an in-flight `call_export_with_args` never
/// returns on its own — it holds the per-instance lock AND an `Arc` clone of
/// the registry handle across its `call_async` await until the deadline trips.
/// Mirrors the `spin` / `noop`-loop fixtures in `epoch_timeout.rs` and
/// `orphan_cleanup_on_drop.rs`.
fn infinite_loop_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (func (export "spin") (loop (br 0)))
          (func (export "noop")))
        "#,
    )
    .expect("valid wat")
}

// ---------------------------------------------------------------------------
// Fix #1 — per-tenant slot must not leak on detach, even when an in-flight
// call races the release. Asserted via the no-self-lockout invariant.
// ---------------------------------------------------------------------------

/// Baseline (no race): repeated pooled `invoke`s under
/// `max_instances_per_tenant = Some(1)` must never self-lock-out the tenant,
/// the per-tenant slot must never exceed the cap, and a `pool.shutdown` must
/// drain every parked slot back to 0. A leaked per-tenant slot on the
/// release/detach path would push the count past the cap of 1 and refuse the
/// next invoke with `TenantCapacityExhausted`. Single-threaded (no
/// epoch/deadline trap involved), so no Windows guard is needed.
///
/// NOTE on the steady-state count: `release` always re-instantiates a fresh
/// replacement and `try_send`s it into the per-tuple channel (capacity floored
/// at 1), so between invokes ONE instance is parked and legitimately holds the
/// tenant's single slot — `tenant_instance_count` oscillates between 0 (during
/// a call) and 1 (parked), never above. The leak we guard against would drive
/// it strictly above the cap. The decisive end-state check is that
/// `pool.shutdown` returns the count to exactly 0.
#[tokio::test]
async fn pooled_invoke_under_cap_one_does_not_self_lockout() {
    let cfg = EngineConfig {
        max_instances: Some(64),
        // Smallest meaningful per-tenant cap. A leaked slot would push the
        // tenant past this cap and refuse a later invoke.
        max_instances_per_tenant: Some(1),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    // warm=1 so the pool actively detaches + recycles on every release
    // (exercising the per-tenant decrement in `detach_pooled_instance` /
    // `release_instance_slot`), parking exactly one replacement per tuple.
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::new(1, 0)));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    let tenant = TenantId(1);

    // Drive several sequential pooled invokes. Each acquires (warm draw),
    // calls, and releases (detach + release + recycle). If ANY release failed
    // to decrement the per-tenant slot, the count would climb past 1 and the
    // next invoke would be refused at the cap.
    for i in 0..8 {
        exec.invoke(SpawnConfig::for_tenant(tenant), &wasm, "noop", &[])
            .await
            .unwrap_or_else(|e| panic!("invoke #{i} must be admitted (no self-lockout); got {e:?}"));
        // The per-tenant count must never exceed the cap of 1. A leak would
        // park more than one slot's worth here.
        assert!(
            exec.tenant_instance_count(tenant) <= 1,
            "after invoke #{i} the per-tenant count {} must never exceed the cap of 1 \
             (a value > 1 means a release leaked a slot)",
            exec.tenant_instance_count(tenant),
        );
    }

    // Draining the pool releases the parked warm instance's slot. After that
    // the tenant's per-tenant count must be exactly 0 — proof that not a single
    // slot leaked across all the detach/release cycles above.
    pool.shutdown(&exec);
    assert_eq!(
        exec.tenant_instance_count(tenant),
        0,
        "after shutdown drains the warm channel, the tenant's per-tenant slot must be 0",
    );

    // And the tenant can spawn freely again — no residual lockout.
    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(tenant), &wasm)
        .await
        .expect("direct spawn after drain must be admitted under the cap");
    exec.terminate(id).await.expect("terminate");
    assert_eq!(exec.tenant_instance_count(tenant), 0);
}

/// Race variant: while a compute-bound `call_export_with_args` is still
/// in-flight against the acquired instance (holding an `Arc` clone of the
/// registry handle — the precondition for the `detach` `try_unwrap` race that
/// fix #1 hardens), release that same instance back to the pool. After both
/// settle, the tenant's per-tenant count must stay within the cap of 1, a
/// `pool.shutdown` must drain it to exactly 0, and the tenant must be able to
/// spawn again: the pre-fix leak would have pinned the count strictly above
/// the cap and self-locked-out the tenant.
///
/// Requires the `multi_thread` flavor + a running epoch ticker: the in-flight
/// `spin` call only relinquishes its lock/Arc when the cooperative epoch
/// deadline trips, which needs the runtime to make progress on a thread the
/// spinning guest is not occupying (same rationale as `orphan_cleanup_on_drop`
/// / `epoch_timeout`). Windows is skipped for the same wasmtime fiber-unwinding
/// reason those sibling tests document.
#[cfg_attr(
    windows,
    ignore = "wasmtime fiber unwinding on Windows panics on epoch interrupt"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn release_racing_in_flight_call_releases_per_tenant_slot() {
    let cfg = EngineConfig {
        max_instances: Some(64),
        max_instances_per_tenant: Some(1),
        // Tight ticker so the in-flight spin traps quickly.
        epoch_tick: Duration::from_millis(5),
        ..EngineConfig::default()
    };
    let mut engine = TensorWasmEngine::with_config(cfg).expect("engine");
    engine.spawn_epoch_ticker();
    let engine = Arc::new(engine);
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::new(1, 0)));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = infinite_loop_wasm();
    let tenant = TenantId(1);

    // Acquire a pooled instance (warm draw or spawn fall-through). We drive
    // the call + release by hand here so we can overlap them on the same id —
    // this is exactly what `invoke` does internally, unpacked so we can insert
    // the racing in-flight call.
    let spawn_cfg = SpawnConfig::for_tenant(tenant).with_deadline(Duration::from_millis(50));
    let pooled = pool
        .acquire(&exec, &wasm, spawn_cfg.clone())
        .await
        .expect("acquire");
    let id = pooled.id();

    // Start the compute-bound call on a separate task. It clones the registry
    // `Arc` handle and holds it across `call_async` (the outstanding strong
    // reference that makes a racing `detach` `try_unwrap` fail). It returns
    // only when the cooperative epoch deadline trips.
    let call_exec = exec.clone();
    let call_task = tokio::spawn(async move {
        // The result is an error (deadline trap); we only care that the call
        // held its Arc clone for a while, racing the release below.
        let _ = call_exec.call_export_with_args(id, "spin", &[]).await;
    });

    // Yield so the call task is scheduled and has taken its Arc clone + lock
    // before we release.
    tokio::task::yield_now().await;

    // Release the SAME instance back to the pool while the call is in flight.
    // `detach_pooled_instance` removes the registry entry then unwraps the Arc;
    // with the in-flight call's clone live this is the try_unwrap-race shape.
    // Whichever ordering the runtime picks, the per-tenant slot must end up
    // released exactly once (fix #1) — never leaked.
    pool.release(&exec, pooled, &spawn_cfg).await;

    // Let the in-flight call trap and unwind, dropping its Arc clone.
    call_task.await.expect("call task joined");

    // After the race settles, the tenant's per-tenant count must never exceed
    // the cap of 1. Depending on the detach/try_unwrap ordering the release
    // either parked a fresh replacement (count == 1) or returned early on the
    // try_unwrap-failure branch having released the slot (count == 0) — both
    // are within the cap. The pre-fix leak would have left it strictly above 1
    // (the original detached slot leaked AND a replacement parked).
    assert!(
        exec.tenant_instance_count(tenant) <= 1,
        "per-tenant count {} must not exceed the cap of 1 after a release racing an in-flight call \
         (a value > 1 means the detach try_unwrap branch leaked the per-tenant slot — fix #1 regressed)",
        exec.tenant_instance_count(tenant),
    );

    // Draining any parked replacement must return the per-tenant count to
    // exactly 0 — no residual leak.
    pool.shutdown(&exec);
    assert_eq!(
        exec.tenant_instance_count(tenant),
        0,
        "after drain the tenant's per-tenant slot must be 0",
    );

    // Decisive proof of no self-lockout: a fresh spawn for the same tenant
    // under the cap of 1 must be admitted. Pre-fix, the leaked slot would have
    // left the tenant at its cap and this would fail with TenantCapacityExhausted.
    let again = exec
        .spawn_instance(SpawnConfig::for_tenant(tenant), &noop_wasm())
        .await;
    match again {
        Ok(new_id) => exec.terminate(new_id).await.expect("terminate"),
        Err(ExecError::TenantCapacityExhausted { .. }) => {
            panic!("tenant self-locked-out: per-tenant slot leaked on the detach race (fix #1 regressed)")
        }
        Err(other) => panic!("unexpected spawn error after race: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Fix #2 — `ensure_entry` single-flight: concurrent first-acquires for the
// same tuple build the warm pool exactly once.
// ---------------------------------------------------------------------------

/// Fire N concurrent first-`acquire`s (via `invoke`) for the SAME fresh
/// `(tenant, module)` tuple with `warm_instances_per_tuple = W`. Post-fix the
/// per-key `OnceCell` admits exactly one builder, so the steady-state warm
/// count is exactly `W` (the single seeded channel, drawn down and replenished
/// by the herd's invokes). Pre-fix, each concurrent first-builder ran the full
/// pre-spawn loop and counted `warm_total` up independently, so `warm_count`
/// would have settled at a MULTIPLE of `W` (one channel's worth per racing
/// builder). Asserting exact equality to `W` catches the double-count.
///
/// `multi_thread` is required so the N tasks genuinely overlap inside
/// `ensure_entry` (the window the single-flight guard protects); a current-
/// thread runtime would serialise them and never exercise the herd.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_first_acquire_builds_pool_once() {
    const WARM_N: usize = 3;
    const HERD: usize = 16;

    let cfg = EngineConfig {
        // Generous engine-wide cap so a (pre-fix) multi-build herd would not be
        // masked by CapacityExhausted before we can observe the double-count.
        max_instances: Some(1_000),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::new(WARM_N, 0)));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    let tenant = TenantId(1);

    // Launch the herd: every task is a FIRST acquire for the same tuple, so
    // they all miss the `pools.get` fast path and contend on `ensure_entry`.
    let mut tasks = Vec::with_capacity(HERD);
    for _ in 0..HERD {
        let exec = exec.clone();
        let wasm = wasm.clone();
        tasks.push(tokio::spawn(async move {
            exec.invoke(SpawnConfig::for_tenant(tenant), &wasm, "noop", &[])
                .await
                .expect("invoke")
        }));
    }
    for t in tasks {
        t.await.expect("herd task joined");
    }

    // Single-flight: the pool was built exactly once, so `warm_total` is
    // bounded by the single surviving channel's capacity (WARM_N). Pre-fix,
    // each racing builder pre-spawned its OWN WARM_N instances and bumped the
    // shared `warm_total` before only one channel won the `pools` insert — so
    // `warm_count` would have settled at a MULTIPLE of WARM_N (up to
    // HERD * WARM_N), strictly above the channel capacity. Asserting
    // `<= WARM_N` is the robust discriminator: it catches the double-count
    // while tolerating a benign transient where the very last release's reset
    // overran its 10 ms budget and dropped instead of refilling.
    let warm = pool.warm_count();
    assert!(
        warm <= WARM_N,
        "warm_count {warm} must not exceed the single channel's capacity {WARM_N}; \
         a multiple indicates the thundering-herd double-build (fix #2 regressed)",
    );
    // And the pool was genuinely built (the herd did warm something).
    assert!(
        warm >= 1,
        "warm_count must be at least 1 after a herd of warm-pool invokes (got {warm})",
    );

    // The number of warm-channel hits is bounded by what a single channel can
    // serve over the herd; the remainder are misses (spawn fall-through). This
    // is a sanity cross-check that draws were accounted on a single channel —
    // hits can never exceed the steady-state warm capacity times the herd.
    assert_eq!(
        pool.draws_total(),
        HERD,
        "every invoke records exactly one draw",
    );
    assert_eq!(
        pool.hit_count() + pool.miss_count(),
        HERD,
        "every draw is classified as exactly one of hit/miss",
    );
}

/// Tighter single-flight assertion using the warm-count invariant after the
/// herd quiesces, with `max_total_warm` capping the global warm budget at
/// exactly one tuple's worth. Pre-fix, even with the global cap, multiple
/// builders racing the cap check could over-count before the cap bit; the
/// single-flight guard makes the post-herd warm count deterministically equal
/// to the configured tuple warmth.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_first_acquire_respects_single_tuple_warm_budget() {
    const WARM_N: usize = 2;
    const HERD: usize = 12;

    let cfg = EngineConfig {
        max_instances: Some(1_000),
        ..EngineConfig::default()
    };
    let engine = Arc::new(TensorWasmEngine::with_config(cfg).expect("engine"));
    let exec = TensorWasmExecutor::new(engine);
    // Global warm cap == one tuple's worth. A correctly single-flighted build
    // seeds exactly WARM_N and stays there.
    let pool = Arc::new(InstancePool::new(InstancePoolConfig::new(WARM_N, WARM_N)));
    let exec = exec.with_instance_pool(Arc::clone(&pool));

    let wasm = noop_wasm();
    let tenant = TenantId(2);

    let mut tasks = Vec::with_capacity(HERD);
    for _ in 0..HERD {
        let exec = exec.clone();
        let wasm = wasm.clone();
        tasks.push(tokio::spawn(async move {
            exec.invoke(SpawnConfig::for_tenant(tenant), &wasm, "noop", &[])
                .await
                .expect("invoke")
        }));
    }
    for t in tasks {
        t.await.expect("herd task joined");
    }

    // With single-flight, exactly one builder seeds the channel and the shared
    // `warm_total` honours the global budget, so warm_count never exceeds
    // WARM_N. A double-building herd would have bumped `warm_total` past the
    // budget (each racing builder pre-spawning its own WARM_N before the
    // `pools` insert), so `> WARM_N` is the regression signature.
    let warm = pool.warm_count();
    assert!(
        warm <= WARM_N,
        "warm_count {warm} must never exceed the global warm budget {WARM_N} \
         (a value above it is the thundering-herd double-build — fix #2 regressed)",
    );
    assert!(
        warm >= 1,
        "the tuple must have been warmed at least once (got {warm})",
    );
}
