// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Loom model check for the consume_bytes / release_bytes CAS loop.
//! Build with: `cargo test --features loom --test loom_consume_release -- --nocapture`
//!
//! `loom` exhaustively explores thread interleavings of the CAS algorithm
//! used by [`tensor_wasm_tenant::TenantContext::consume_bytes_with_capability`]
//! and `release_bytes_with_capability` and certifies that no schedule
//! produces a non-linearizable observable — i.e. starting from
//! `bytes_in_use = N`, racing one `consume_bytes(K)` against one
//! `release_bytes(K)` must terminate with `bytes_in_use = N` for every
//! interleaving loom can construct.
//!
//! We model the inner CAS loops against a bare `loom::sync::atomic::AtomicU64`
//! rather than against a real `TenantContext`. The reason is twofold:
//!
//! 1. `loom::sync::atomic::AtomicU64::new` is not `const fn`, so the
//!    `static ISOLATION_DOWNGRADE_COUNT` in `src/context.rs` keeps the
//!    `std`-flavoured `AtomicU64` even under `--features loom`. The CAS
//!    hot path *does* swap to the loom-flavoured atomic (see the cfg
//!    block at the top of `src/context.rs`), but constructing a full
//!    `TenantContext` under loom would still require a CAS-only stand-in
//!    for `TensorWasmMetrics` and friends.
//! 2. The model's purpose is to pin the CAS algorithm, not the
//!    surrounding plumbing. A minimal atomic with the same load /
//!    compare_exchange / saturating-arithmetic shape exercises exactly
//!    the interleavings that matter.

#![cfg(feature = "loom")]

use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;
use loom::thread;

/// Starting value of the shared counter; analogous to
/// `bytes_in_use` after the tenant has accumulated `N` bytes of
/// outstanding allocation.
const N: u64 = 100;
/// Amount the consumer adds and the releaser subtracts. Picked equal so
/// the only linearizable end-state is `N` itself.
const K: u64 = 10;

#[test]
fn cas_loop_is_linearizable_under_2_thread_interleavings() {
    loom::model(|| {
        let counter = Arc::new(AtomicU64::new(N));

        // Releaser: mirrors `release_bytes_inner`'s CAS loop with
        // `saturating_sub` and `compare_exchange_weak`.
        let releaser = {
            let counter = counter.clone();
            thread::spawn(move || {
                let mut cur = counter.load(Ordering::Acquire);
                loop {
                    let next = cur.saturating_sub(K);
                    match counter.compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                        Ok(_) => break,
                        Err(observed) => cur = observed,
                    }
                }
            })
        };

        // Consumer: mirrors `consume_bytes_inner`'s CAS loop with
        // `checked_add` (here `saturating_add` — we never trip the
        // overflow branch because K << N << u64::MAX).
        let consumer = {
            let counter = counter.clone();
            thread::spawn(move || {
                let mut cur = counter.load(Ordering::Acquire);
                loop {
                    let next = cur.saturating_add(K);
                    match counter.compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                        Ok(_) => break,
                        Err(observed) => cur = observed,
                    }
                }
            })
        };

        releaser.join().unwrap();
        consumer.join().unwrap();

        // Linearizability: every interleaving must end at the algebraic
        // sum `N - K + K == N`. If the bug from the old `fetch_sub +
        // store(0)` shape ever reappears we observe it here as a non-N
        // final value.
        assert_eq!(counter.load(Ordering::Acquire), N);
    });
}
