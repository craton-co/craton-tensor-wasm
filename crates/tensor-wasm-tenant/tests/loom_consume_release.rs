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
//! TESTS (Finding 8): the model now drives the **real** CAS loops —
//! [`tensor_wasm_tenant::context::consume_cas`] and
//! [`tensor_wasm_tenant::context::release_cas`] — rather than a hand-rolled
//! copy. Under `--features loom`, the crate's `AtomicU64` / `Ordering` are
//! the `loom`-flavoured types, so importing those `#[doc(hidden)]` helpers
//! and passing them a `loom::sync::atomic::AtomicU64` model-checks the exact
//! production algorithm, including the overflow-rejection and `> limit`
//! rejection arms of `consume_cas`. If the production loop ever drifts (e.g.
//! a refactor reintroduces the old racy `fetch_sub + store(0)` shape) the
//! model fails here because it links against the real code, not a replica.

#![cfg(feature = "loom")]

use loom::sync::atomic::AtomicU64;
use loom::sync::Arc;
use loom::thread;

use tensor_wasm_tenant::context::{consume_cas, release_cas};

/// Starting value of the shared counter; analogous to
/// `bytes_in_use` after the tenant has accumulated `N` bytes of
/// outstanding allocation.
const N: u64 = 100;
/// Amount the consumer adds and the releaser subtracts. Picked equal so
/// the only linearizable end-state is `N` itself.
const K: u64 = 10;
/// Quota high enough that the consumer's add never trips the `> limit`
/// rejection arm in the linearizability model below.
const LIMIT: u64 = u64::MAX;

#[test]
fn cas_loop_is_linearizable_under_2_thread_interleavings() {
    loom::model(|| {
        let counter = Arc::new(AtomicU64::new(N));

        // Releaser: the REAL `release_cas` loop.
        let releaser = {
            let counter = counter.clone();
            thread::spawn(move || {
                let _ = release_cas(&counter, K);
            })
        };

        // Consumer: the REAL `consume_cas` loop (limit so high the
        // `> limit` arm is never taken in this model — that arm is
        // exercised separately below).
        let consumer = {
            let counter = counter.clone();
            thread::spawn(move || {
                consume_cas(&counter, K, LIMIT).expect("K within LIMIT must admit");
            })
        };

        releaser.join().unwrap();
        consumer.join().unwrap();

        // Linearizability: every interleaving must end at the algebraic
        // sum `N - K + K == N`. If the bug from the old `fetch_sub +
        // store(0)` shape ever reappears we observe it here as a non-N
        // final value.
        assert_eq!(counter.load(loom::sync::atomic::Ordering::Acquire), N);
    });
}

#[test]
fn consume_cas_over_limit_arm_never_corrupts_counter() {
    // Finding 8: model-check the `> limit` rejection arm under contention.
    // Two consumers race for the last slot of a tightly-capped counter:
    // exactly one may succeed, the other must be rejected with the counter
    // left at the cap (never wrapped, never double-counted).
    loom::model(|| {
        // Cap admits exactly one more K from a counter starting at LIMIT-K.
        const CAP: u64 = 2 * K;
        let counter = Arc::new(AtomicU64::new(CAP - K)); // one K of headroom

        let a = {
            let counter = counter.clone();
            thread::spawn(move || consume_cas(&counter, K, CAP).is_ok())
        };
        let b = {
            let counter = counter.clone();
            thread::spawn(move || consume_cas(&counter, K, CAP).is_ok())
        };

        let a_ok = a.join().unwrap();
        let b_ok = b.join().unwrap();

        // Exactly one consumer wins the single available slot.
        assert!(
            a_ok ^ b_ok,
            "exactly one of two racing consumers may take the last slot",
        );
        // The counter never exceeds the cap and reflects exactly one admit.
        assert_eq!(counter.load(loom::sync::atomic::Ordering::Acquire), CAP);
    });
}

#[test]
fn consume_cas_overflow_arm_never_corrupts_counter() {
    // Finding 8: model-check the `checked_add` overflow rejection arm under
    // contention. The counter starts near u64::MAX with an effectively
    // unbounded limit, so the cap is never the gate — the only rejection is
    // the overflow arm. Two consumers race; at most one may add without
    // overflowing, and the counter must never wrap below its start.
    loom::model(|| {
        // Headroom for exactly one K before `checked_add` overflows.
        let start = u64::MAX - K;
        let counter = Arc::new(AtomicU64::new(start));

        let a = {
            let counter = counter.clone();
            thread::spawn(move || consume_cas(&counter, K, u64::MAX).is_ok())
        };
        let b = {
            let counter = counter.clone();
            thread::spawn(move || consume_cas(&counter, K, u64::MAX).is_ok())
        };

        let a_ok = a.join().unwrap();
        let b_ok = b.join().unwrap();

        // Exactly one add fits in the remaining headroom; the other hits the
        // overflow arm and is rejected.
        assert!(
            a_ok ^ b_ok,
            "exactly one of two racing consumers may consume the last headroom",
        );
        // The successful add lands at exactly u64::MAX; the rejected one left
        // the counter untouched (never wrapped to a small value).
        assert_eq!(
            counter.load(loom::sync::atomic::Ordering::Acquire),
            u64::MAX,
        );
    });
}
