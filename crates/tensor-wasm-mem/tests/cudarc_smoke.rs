// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Smoke tests for the `cudarc-backend` spike.
//!
//! These tests are compiled only when the `cudarc-backend` feature is
//! enabled. Any test that touches the CUDA driver (allocate / advise /
//! prefetch) is marked `#[ignore = "requires CUDA hardware"]` so a host-only
//! CI run that does `cargo test --features cudarc-backend` still passes —
//! the unignored tests confirm that the cudarc-backend code compiles, links,
//! and that the public types from `tensor_wasm_mem::cudarc_backend` have a
//! non-trivial layout (the only thing we can prove without a GPU).
//!
//! Run with hardware:
//!
//! ```ignore
//! cargo test -p tensor-wasm-mem --features cudarc-backend --test cudarc_smoke -- --ignored
//! ```

#![cfg(feature = "cudarc-backend")]

use tensor_wasm_mem::advise::Advice;
use tensor_wasm_mem::cudarc_backend::{apply_advice, CudarcUnifiedBuffer};
use tensor_wasm_mem::unified::DeviceId;

/// The buffer type compiles and has a non-zero layout. This is the headline
/// "the spike code path is wired" assertion — if this fails the workspace
/// has a broken feature flag or a missing module declaration.
#[test]
fn cudarc_buffer_type_has_nonzero_size() {
    assert!(std::mem::size_of::<CudarcUnifiedBuffer>() > 0);
}

/// The `apply_advice` free function is reachable from outside the crate
/// (i.e. the symbol is exported, not just `pub(crate)`).
#[test]
fn cudarc_apply_advice_is_exported() {
    let _f: fn(&CudarcUnifiedBuffer, Advice) -> Result<(), tensor_wasm_mem::unified::UnifiedError> =
        apply_advice;
}

/// Zero-byte allocations are rejected without touching the driver. This
/// path runs entirely in safe Rust before any `cuMemAllocManaged` call so
/// it is safe to exercise even without CUDA hardware.
#[test]
fn cudarc_zero_size_rejected_without_driver() {
    let err = CudarcUnifiedBuffer::new(0).expect_err("zero should be rejected");
    assert!(matches!(
        err,
        tensor_wasm_mem::unified::UnifiedError::ZeroSize
    ));
}

/// Allocate a tiny managed buffer, write into it, read it back, drop it.
/// Requires a CUDA driver and at least one visible GPU.
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_round_trip_on_device() {
    let mut b = CudarcUnifiedBuffer::new(128).expect("alloc");
    assert_eq!(b.len(), 128);
    assert!(!b.is_empty());
    b.as_mut_slice().copy_from_slice(&[0x5Au8; 128]);
    assert!(b.as_slice().iter().all(|&v| v == 0x5A));
}

/// Calling `apply_advice` against a freshly allocated buffer must succeed on
/// hardware that supports unified memory advice (compute capability >= 6.0).
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_apply_advice_read_mostly_on_device() {
    let b = CudarcUnifiedBuffer::new(256).expect("alloc");
    apply_advice(&b, Advice::ReadMostly).expect("set_read_mostly should succeed on Pascal+");
}

/// Prefetch in both directions on the default stream.
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_prefetch_round_trip_on_device() {
    let b = CudarcUnifiedBuffer::new_on(64, DeviceId(0)).expect("alloc");
    b.prefetch_to_device().expect("prefetch_to_device");
    b.prefetch_to_host().expect("prefetch_to_host");
}
