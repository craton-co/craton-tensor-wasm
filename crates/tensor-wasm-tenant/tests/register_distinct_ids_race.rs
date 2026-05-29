// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! 32 OS threads race to `register` 32 **disjoint** `TenantId`s.
//!
//! Unlike `register_contention.rs` — which pins exactly one winner for a
//! single contested id and therefore exercises only per-shard
//! linearization — this test races over disjoint keys spread across the
//! whole `DashMap` shard space. Every registration *should* succeed
//! because none of the threads is competing for the same slot. What we
//! pin here is the registry's behaviour across shards:
//!
//! * Every thread's `register` returns `Ok`. There are no spurious
//!   `AlreadyRegistered` or `OrphanStillAlive` errors (those would
//!   indicate a key-collision or tombstone bookkeeping bug under shard
//!   load).
//! * After the race, `reg.len()` is exactly 32. No registration is lost
//!   to a torn shard write, and no extra ghost entry appears.
//! * Every id `TenantId(0)..=TenantId(31)` is retrievable via
//!   `reg.get(id, &cap)` and the looked-up `Arc` is pointer-equal to the
//!   `Arc` that thread's `register` returned. The registry must not
//!   re-allocate a fresh `Arc` on lookup.
//!
//! The structure mirrors `register_contention.rs` (Barrier-aligned gate,
//! AtomicUsize counters, post-race assertions) so the two tests are
//! consistent and one can be diff-read against the other.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{RegistryError, TenantContext, TenantRegistry};

const THREADS: usize = 32;

fn make_ctx(id: u64) -> TenantContext {
    TenantContext::builder(TenantId(id))
        .with_memory_quota_bytes(1024)
        .build()
}

#[test]
fn thirty_two_threads_register_disjoint_ids_concurrently() {
    let (reg, cap) = TenantRegistry::new();
    let barrier = Arc::new(Barrier::new(THREADS));
    let successes = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    // Each thread parks its winning Arc here keyed by its own TenantId so
    // the post-race assertions can compare against the lookup result.
    type Winners = Arc<Mutex<Vec<(TenantId, Arc<TenantContext>)>>>;
    let winners: Winners = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::with_capacity(THREADS);
    for thread_idx in 0..THREADS {
        let reg = reg.clone();
        let barrier = Arc::clone(&barrier);
        let successes = Arc::clone(&successes);
        let failures = Arc::clone(&failures);
        let winners = Arc::clone(&winners);
        handles.push(thread::spawn(move || {
            // Line all threads up at the gate so the race is as tight as
            // the OS scheduler allows. Allocate the context *after* the
            // barrier so the heap allocation isn't paid before the race
            // window opens — otherwise threads scheduled first would be
            // deep inside `register` before later threads even reach the
            // gate, narrowing contention across shards.
            barrier.wait();
            let id = TenantId(thread_idx as u64);
            let ctx = make_ctx(id.get());
            match reg.register(ctx) {
                Ok(arc) => {
                    successes.fetch_add(1, Ordering::Relaxed);
                    winners.lock().unwrap().push((id, arc));
                }
                Err(RegistryError::AlreadyRegistered(other)) => {
                    // Disjoint ids: a collision here is a registry bug
                    // (e.g. a shard mis-hash or a `TenantId` hash that
                    // collapses distinct values onto one slot).
                    failures.fetch_add(1, Ordering::Relaxed);
                    panic!(
                        "unexpected AlreadyRegistered({other:?}) for disjoint id {id:?} — \
                         registry must not collapse distinct TenantIds onto one slot",
                    );
                }
                Err(RegistryError::OrphanStillAlive(other)) => {
                    // No prior unregister happened in this test, so an
                    // orphan-rejection here is a registry-invariant bug.
                    panic!(
                        "unexpected OrphanStillAlive({other:?}) for id {id:?} — \
                         no tombstone should exist before any unregister",
                    );
                }
                // H1: `CapabilityFromForeignRegistry` is now compiled
                // unconditionally (no longer gated on `strict-cap-binding`),
                // so this arm must always be present for the match to be
                // exhaustive in every feature config. `register` never mints
                // or compares an admin cap, so this variant can never occur.
                Err(RegistryError::CapabilityFromForeignRegistry) => {
                    panic!(
                        "unexpected CapabilityFromForeignRegistry for id {id:?} — \
                         ctx built by this registry",
                    );
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(
        successes.load(Ordering::Relaxed),
        THREADS,
        "every disjoint-id registration must succeed",
    );
    assert_eq!(
        failures.load(Ordering::Relaxed),
        0,
        "no thread should have observed a collision on disjoint ids",
    );

    // The registry now holds 32 distinct entries — one per thread.
    assert_eq!(
        reg.len(&cap),
        THREADS,
        "final registry size must equal the number of distinct ids registered",
    );

    // Every id is retrievable, and the lookup returns the same Arc the
    // winning registration handed back — no fresh allocations on get.
    let winners = winners.lock().unwrap();
    assert_eq!(
        winners.len(),
        THREADS,
        "every thread should have parked its winner"
    );
    for (id, winner_arc) in winners.iter() {
        let looked_up = reg
            .get(*id, &cap)
            .unwrap_or_else(|| panic!("registered tenant {id:?} must be findable"));
        assert!(
            Arc::ptr_eq(winner_arc, &looked_up),
            "lookup for {id:?} must return the same Arc as the winning registration",
        );
        assert_eq!(looked_up.id(), *id, "lookup returned the wrong tenant id");
    }
}
