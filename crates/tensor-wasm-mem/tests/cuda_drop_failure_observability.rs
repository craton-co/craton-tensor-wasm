// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Observability surface for CUDA driver-level `cuMemFree_v2` failures (mem H4).
//!
//! Both the `cudarc-backend` and `cuda-oxide-backend` paths expose a process-
//! global `leaked_cuda_allocations() -> Vec<u64>` accessor that returns the
//! set of raw device-pointer values orphaned by a failed free in their
//! respective `Drop` impls. These tests pin the *shape* of that surface:
//!
//! - The accessor exists, is `pub`, and returns `Vec<u64>`.
//! - On a fresh process (i.e. no allocations have failed to free yet), the
//!   accessor returns an empty `Vec`.
//! - The accessor is safe to call concurrently from multiple threads
//!   without panicking or deadlocking (it acquires a `parking_lot::Mutex`
//!   internally).
//!
//! We deliberately do **not** try to simulate a `cuMemFree_v2` failure
//! here — that would require either a live CUDA driver in a controlled
//! failure state or a fault-injection shim that does not exist in this
//! crate. The Drop-path leak recording is exercised by inspection of the
//! code; this test file exists to ensure the *audit* surface stays
//! stable for operator tooling.

#![cfg(any(feature = "cudarc-backend", feature = "cuda-oxide-backend"))]

use std::thread;

#[cfg(feature = "cudarc-backend")]
mod cudarc {
    use super::*;
    use tensor_wasm_mem::cudarc_backend::leaked_cuda_allocations;

    /// The accessor is reachable from outside the crate (i.e. `pub`, not
    /// `pub(crate)`), and has the documented signature.
    #[test]
    fn leaked_cuda_allocations_is_exported() {
        let _f: fn() -> Vec<u64> = leaked_cuda_allocations;
    }

    /// On a fresh test process no Drop has yet observed a `cuMemFree_v2`
    /// failure, so the audit set is empty. We bound the assertion to this
    /// test rather than asserting on `len() == 0` unconditionally — other
    /// tests in this binary could in principle populate the set if they
    /// allocated and then injected a failure, but no such test exists
    /// today and the test runner spawns a fresh process per binary.
    #[test]
    fn leaked_cuda_allocations_is_initially_empty() {
        let snapshot = leaked_cuda_allocations();
        assert!(
            snapshot.is_empty(),
            "expected fresh process to have no recorded CUDA leaks, \
             got {snapshot:?}"
        );
    }

    /// Two threads reading the snapshot concurrently must not panic or
    /// deadlock. The accessor holds a `parking_lot::Mutex` internally;
    /// shared-read access is the dominant call pattern (operator polling
    /// from a monitoring loop), so we exercise it explicitly.
    #[test]
    fn leaked_cuda_allocations_concurrent_reads_dont_panic() {
        let t1 = thread::spawn(|| {
            for _ in 0..100 {
                let _ = leaked_cuda_allocations();
            }
        });
        let t2 = thread::spawn(|| {
            for _ in 0..100 {
                let _ = leaked_cuda_allocations();
            }
        });
        t1.join().expect("reader thread 1 panicked");
        t2.join().expect("reader thread 2 panicked");
    }
}

#[cfg(feature = "cuda-oxide-backend")]
mod cuda_oxide {
    use super::*;
    use tensor_wasm_mem::cuda_oxide_backend::leaked_cuda_allocations;

    /// The accessor is reachable from outside the crate (i.e. `pub`, not
    /// `pub(crate)`), and has the documented signature.
    #[test]
    fn leaked_cuda_allocations_is_exported() {
        let _f: fn() -> Vec<u64> = leaked_cuda_allocations;
    }

    /// On a fresh test process no Drop has yet observed a free failure
    /// (and on the scaffold no Drop ever calls a real free), so the audit
    /// set is empty.
    #[test]
    fn leaked_cuda_allocations_is_initially_empty() {
        let snapshot = leaked_cuda_allocations();
        assert!(
            snapshot.is_empty(),
            "expected fresh process to have no recorded CUDA leaks, \
             got {snapshot:?}"
        );
    }

    /// Two threads reading the snapshot concurrently must not panic or
    /// deadlock.
    #[test]
    fn leaked_cuda_allocations_concurrent_reads_dont_panic() {
        let t1 = thread::spawn(|| {
            for _ in 0..100 {
                let _ = leaked_cuda_allocations();
            }
        });
        let t2 = thread::spawn(|| {
            for _ in 0..100 {
                let _ = leaked_cuda_allocations();
            }
        });
        t1.join().expect("reader thread 1 panicked");
        t2.join().expect("reader thread 2 panicked");
    }
}
