// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Loom model check for the consume_bytes / release_bytes CAS loop.
//! Build with: `cargo test --features loom --test loom_consume_release -- --nocapture`
//!
//! `loom` exhaustively explores thread interleavings of the CAS algorithm
//! used by `TenantContext::{consume_bytes_inner, release_bytes_inner}` and
//! certifies that no schedule produces a non-linearizable observable —
//! i.e. starting from `bytes_in_use = N`, racing one `consume_bytes(K)`
//! against one `release_bytes(K)` must terminate with `bytes_in_use = N`
//! for every interleaving loom can construct.
//!
//! This file is the scaffold: the wiring (feature gate, `[[test]]` target,
//! `loom::model` entry point) lands now so the next patch can drop in the
//! actual model body without touching `Cargo.toml` again. See
//! `TODO(loom)` inside the model closure for the body to implement.

#![cfg(feature = "loom")]

#[test]
fn cas_loop_is_linearizable_under_2_thread_interleavings() {
    loom::model(|| {
        // TODO(loom): Build a TenantContext (or its minimal CAS-only
        // stand-in) and race two threads: T1 consumes N bytes, T2
        // releases N bytes. Assert the final bytes_in_use value is
        // exactly the starting value (linearizable).
        //
        // The actual TenantContext uses parking_lot internally which loom
        // doesn't model; if a full TenantContext can't run under loom,
        // build a minimal atomic struct with the same
        // fetch-update-compare-exchange pattern (see
        // `crates/tensor-wasm-tenant/src/context.rs::consume_bytes_inner`
        // and `release_bytes_inner`) and model that. The point is to pin
        // the algorithm, not the surrounding plumbing.
        //
        // Suggested shape once filled in:
        //
        //   use loom::sync::Arc;
        //   use loom::sync::atomic::{AtomicU64, Ordering};
        //   use loom::thread;
        //
        //   const N: u64 = 1024;
        //   const K: u64 = 256;
        //   let counter = Arc::new(AtomicU64::new(N));
        //
        //   let consumer = {
        //       let c = counter.clone();
        //       thread::spawn(move || {
        //           // mirror consume_bytes_inner's CAS loop
        //           let mut cur = c.load(Ordering::Acquire);
        //           loop {
        //               let next = cur + K;
        //               match c.compare_exchange_weak(
        //                   cur, next, Ordering::AcqRel, Ordering::Acquire,
        //               ) {
        //                   Ok(_) => break,
        //                   Err(obs) => cur = obs,
        //               }
        //           }
        //       })
        //   };
        //   let releaser = {
        //       let c = counter.clone();
        //       thread::spawn(move || {
        //           // mirror release_bytes_inner's CAS loop (saturating)
        //           let mut cur = c.load(Ordering::Acquire);
        //           loop {
        //               let next = cur.saturating_sub(K);
        //               match c.compare_exchange_weak(
        //                   cur, next, Ordering::AcqRel, Ordering::Acquire,
        //               ) {
        //                   Ok(_) => break,
        //                   Err(obs) => cur = obs,
        //               }
        //           }
        //       })
        //   };
        //   consumer.join().unwrap();
        //   releaser.join().unwrap();
        //   assert_eq!(counter.load(Ordering::Acquire), N);

        todo!("loom model body — see comment above");
    });
}
