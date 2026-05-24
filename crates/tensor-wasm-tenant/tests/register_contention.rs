// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! 32 OS threads race to `register` the same `TenantId`.
//!
//! The registry must be linearizable: exactly one thread observes a
//! successful insert and gets back an `Arc<TenantContext>`; the other
//! 31 must observe `RegistryError::AlreadyRegistered(_)` and never
//! overwrite the winning entry. A follow-up `lookup` must return the
//! same `Arc` (pointer-equal) as the winning registration.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{RegistryError, TenantContext, TenantRegistry};

const THREADS: usize = 32;
const TENANT: TenantId = TenantId(7);

fn make_ctx() -> TenantContext {
    TenantContext::builder(TENANT)
        .with_memory_quota_bytes(1024)
        .build()
}

#[test]
fn thirty_two_threads_contend_for_the_same_tenant_id() {
    let reg = TenantRegistry::new();
    let barrier = Arc::new(Barrier::new(THREADS));
    let successes = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    let winners: Arc<Mutex<Vec<Arc<TenantContext>>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        let reg = reg.clone();
        let barrier = Arc::clone(&barrier);
        let successes = Arc::clone(&successes);
        let failures = Arc::clone(&failures);
        let winners = Arc::clone(&winners);
        handles.push(thread::spawn(move || {
            let ctx = make_ctx();
            // Line all threads up at the gate so the race is as tight as
            // the OS scheduler allows.
            barrier.wait();
            match reg.register(ctx) {
                Ok(arc) => {
                    successes.fetch_add(1, Ordering::Relaxed);
                    winners.lock().unwrap().push(arc);
                }
                Err(RegistryError::AlreadyRegistered(id)) => {
                    assert_eq!(id, TENANT);
                    failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(
        successes.load(Ordering::Relaxed),
        1,
        "exactly one registration must succeed",
    );
    assert_eq!(
        failures.load(Ordering::Relaxed),
        THREADS - 1,
        "all other threads must report AlreadyRegistered",
    );

    // The registry now holds the winner's Arc — `get` must return the
    // same allocation, not a fresh clone.
    let winners = winners.lock().unwrap();
    assert_eq!(winners.len(), 1, "exactly one winning Arc");
    let winner = &winners[0];
    let looked_up = reg.get(TENANT).expect("registered tenant must be findable");
    assert!(
        Arc::ptr_eq(winner, &looked_up),
        "lookup must return the same Arc as the winning registration",
    );
    assert_eq!(reg.len(), 1, "no spurious registrations");
}
