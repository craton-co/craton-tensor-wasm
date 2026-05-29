// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Sustained-pressure quota exhaustion under concurrency.
//!
//! The existing concurrency test (`concurrent_isolation.rs`) issues
//! *matched* consume/release pairs, so the counter oscillates and never
//! sits against the cap. This test instead drives N threads that each
//! attempt to consume `quota/N + 1` bytes **without ever releasing**, so
//! the cap is genuinely exhausted and the CAS loop's rejection path is
//! the one under test.
//!
//! With `per_thread = quota/N + 1`, the cap admits at most
//! `floor(quota / per_thread)` successful consumes; every further attempt
//! must be refused with `MemoryExhausted`. The invariants we pin:
//!
//!   * the number of refusals is exactly `THREADS - admitted`, where
//!     `admitted == floor(quota / per_thread)` (deterministic regardless
//!     of interleaving, because no thread ever releases),
//!   * the counter never exceeds the quota at any observed point,
//!   * the final counter equals `admitted * per_thread` (no lost or
//!     double-counted successful consume), and
//!   * no thread panics.

#![allow(deprecated)]
// Exercises the unchecked `consume_bytes` shim deliberately: the
// sustained-pressure rejection contract must hold for the deprecated
// path until v0.4 removes it. The capability-checked variant shares the
// same `consume_bytes_inner`, so this transitively covers it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::TenantContext;

#[test]
fn n_threads_pressuring_one_cap_refuse_the_right_count() {
    const THREADS: u64 = 16;
    // A quota that is NOT an exact multiple of `per_thread`, so the
    // boundary arithmetic (floor division, the leftover headroom that is
    // too small for one more consume) is genuinely exercised.
    const QUOTA: u64 = 10_000;
    // Each thread asks for strictly more than its even share; the `+1`
    // guarantees that fewer than THREADS consumes can ever fit.
    let per_thread = QUOTA / THREADS + 1; // 626 for QUOTA=10_000, THREADS=16

    // How many of these no-release consumes the cap can admit before the
    // next one would exceed QUOTA.
    let admitted = QUOTA / per_thread; // floor: 15
    let expected_refusals = THREADS - admitted; // 1
    assert!(
        admitted < THREADS,
        "test mis-parameterised: cap must reject at least one thread",
    );

    let ctx = Arc::new(
        TenantContext::builder(TenantId(0xCAFE))
            .with_memory_quota_bytes(QUOTA)
            .build(),
    );
    let refusals = Arc::new(AtomicU64::new(0));
    // Largest counter value any thread observed via the error's `current`
    // or via a successful consume's post-state. The counter must never be
    // seen above QUOTA.
    let max_seen = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::with_capacity(THREADS as usize);
    for _ in 0..THREADS {
        let ctx = Arc::clone(&ctx);
        let refusals = Arc::clone(&refusals);
        let max_seen = Arc::clone(&max_seen);
        handles.push(thread::spawn(move || {
            match ctx.consume_bytes(per_thread) {
                Ok(()) => {
                    // Observe the counter post-consume; it must already
                    // reflect this thread's bytes and stay within the cap.
                    let now = ctx.bytes_in_use();
                    max_seen.fetch_max(now, Ordering::Relaxed);
                }
                Err(TensorWasmError::MemoryExhausted { requested, limit }) => {
                    assert_eq!(requested, per_thread, "wrong requested in refusal");
                    assert_eq!(limit, QUOTA, "refusal must report the configured quota");
                    refusals.fetch_add(1, Ordering::Relaxed);
                }
                Err(other) => panic!("unexpected error variant under pressure: {other:?}"),
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked under quota pressure");
    }

    // Exactly the right number of refusals — deterministic because no
    // thread releases, so total admitted bytes are monotonic.
    assert_eq!(
        refusals.load(Ordering::Relaxed),
        expected_refusals,
        "expected exactly {expected_refusals} refusals (admitted={admitted}, threads={THREADS})",
    );

    // The counter never exceeded the quota at any observed point.
    assert!(
        max_seen.load(Ordering::Relaxed) <= QUOTA,
        "observed in-use {} exceeded quota {QUOTA}",
        max_seen.load(Ordering::Relaxed),
    );

    // Final state: exactly the admitted consumes landed, nothing lost or
    // double-counted, and we are still at or below the cap.
    let final_in_use = ctx.bytes_in_use();
    assert_eq!(
        final_in_use,
        admitted * per_thread,
        "final in-use must equal admitted ({admitted}) * per_thread ({per_thread})",
    );
    assert!(
        final_in_use <= QUOTA,
        "final in-use {final_in_use} must not exceed quota {QUOTA}",
    );
}
