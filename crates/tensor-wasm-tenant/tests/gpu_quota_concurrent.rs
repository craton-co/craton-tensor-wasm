// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Sustained-pressure GPU-cap exhaustion under concurrency (Finding 8).
//!
//! The GPU sibling of `quota_exhaustion_concurrent.rs`: N threads each
//! attempt to reserve `cap/N + 1` GPU bytes through the
//! capability-checked `consume_gpu_bytes_with_capability` path **without
//! ever releasing**, so the per-tenant GPU cap is genuinely exhausted and
//! the CAS loop's `GpuMemoryExhausted` rejection path is the one under
//! test. The pre-existing GPU tests cover the single-threaded cap and the
//! cross-tenant capability gate; none drove the GPU rejection path under
//! contention.
//!
//! Invariants pinned (deterministic because no thread releases, so the
//! admitted-byte total is monotonic):
//!
//!   * refusals == `THREADS - floor(cap / per_thread)`,
//!   * `gpu_bytes_in_use` never exceeds the cap at any observed point,
//!   * the final counter equals `admitted * per_thread` (no lost or
//!     double-counted reservation), and
//!   * no thread panics.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{TenantContext, TenantRegistry};

#[test]
fn n_threads_pressuring_one_gpu_cap_refuse_the_right_count() {
    const THREADS: u64 = 16;
    const CAP: u64 = 10_000;
    let per_thread = CAP / THREADS + 1; // 626

    let admitted = CAP / per_thread; // floor: 15
    let expected_refusals = THREADS - admitted; // 1
    assert!(
        admitted < THREADS,
        "test mis-parameterised: cap must reject at least one thread",
    );

    // Register through the registry so we get a real `TenantCapability`
    // for the capability-checked GPU path.
    let (reg, _admin) = TenantRegistry::new();
    let (ctx, cap) = reg
        .register_with_capability(
            TenantContext::builder(TenantId(0xC0DE))
                .with_gpu_memory_bytes_cap(CAP)
                .build(),
        )
        .unwrap();
    let cap = Arc::new(cap);

    let refusals = Arc::new(AtomicU64::new(0));
    let max_seen = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::with_capacity(THREADS as usize);
    for _ in 0..THREADS {
        let ctx = Arc::clone(&ctx);
        let cap = Arc::clone(&cap);
        let refusals = Arc::clone(&refusals);
        let max_seen = Arc::clone(&max_seen);
        handles.push(thread::spawn(move || {
            match ctx.consume_gpu_bytes_with_capability(&cap, per_thread) {
                Ok(()) => {
                    let now = ctx.gpu_bytes_in_use();
                    max_seen.fetch_max(now, Ordering::Relaxed);
                }
                Err(TensorWasmError::GpuMemoryExhausted {
                    requested,
                    limit,
                    current,
                }) => {
                    assert_eq!(requested, per_thread, "wrong requested in GPU refusal");
                    assert_eq!(limit, CAP, "GPU refusal must report the configured cap");
                    assert!(
                        current <= CAP,
                        "refusal's observed current {current} must not exceed cap {CAP}",
                    );
                    refusals.fetch_add(1, Ordering::Relaxed);
                }
                Err(other) => panic!("unexpected error variant under GPU pressure: {other:?}"),
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked under GPU pressure");
    }

    assert_eq!(
        refusals.load(Ordering::Relaxed),
        expected_refusals,
        "expected exactly {expected_refusals} GPU refusals (admitted={admitted})",
    );
    assert!(
        max_seen.load(Ordering::Relaxed) <= CAP,
        "observed gpu_bytes_in_use {} exceeded cap {CAP}",
        max_seen.load(Ordering::Relaxed),
    );

    let final_in_use = ctx.gpu_bytes_in_use();
    assert_eq!(
        final_in_use,
        admitted * per_thread,
        "final gpu in-use must equal admitted ({admitted}) * per_thread ({per_thread})",
    );
    assert!(
        final_in_use <= CAP,
        "final gpu in-use {final_in_use} must not exceed cap {CAP}",
    );
}
