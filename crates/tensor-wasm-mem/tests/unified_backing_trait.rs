// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Trait-surface integration test for [`UnifiedBacking`].
//!
//! B4.4 trait abstraction (v0.4 scaffold). The three concrete buffer types
//! in `tensor-wasm-mem` (`UnifiedBuffer`, `CudarcUnifiedBuffer`,
//! `CudaOxideUnifiedBuffer`) hand-mirror the same API. This test exercises
//! the `UnifiedBacking` trait through a `&dyn UnifiedBacking` reference
//! against the always-available `UnifiedBuffer` on its non-CUDA
//! `Box<[u8]>` fallback so the trait surface is wire-stable for v0.4
//! migrations and future ports.
//!
//! The file is intentionally gated to the no-feature build (neither
//! `unified-memory` nor `cudarc-backend`): under either CUDA backing
//! feature, `UnifiedBuffer::new` issues a real `cuMemAllocManaged` call
//! that requires a working driver. The trait surface itself is
//! exercised the same way under the CUDA-backed paths via the
//! per-backend smoke / snapshot tests already in `tests/`.
#![cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]

use tensor_wasm_mem::unified::{UnifiedBacking, UnifiedBuffer, UvmAdvice};

/// Construct a `UnifiedBuffer`, hold it through a `&dyn UnifiedBacking`
/// reference, and assert the trait surface returns the expected values
/// for the always-available methods.
#[test]
fn dyn_unified_backing_round_trip_on_default_backing() {
    let mut buf = UnifiedBuffer::new(64).expect("alloc on default backing");

    // Write through the inherent surface first so `as_slice` via the
    // trait observes the expected bytes.
    buf.as_mut_slice().copy_from_slice(&[0x5Au8; 64]);

    // Take a `&dyn UnifiedBacking` borrow and exercise the read-side
    // surface through the trait.
    let dynref: &dyn UnifiedBacking = &buf;
    assert_eq!(dynref.len(), 64);
    assert_eq!(dynref.as_slice().len(), 64);
    assert!(dynref.as_slice().iter().all(|&v| v == 0x5A));

    // `apply_advice` is a no-op on the non-CUDA backing and on the
    // legacy `UnifiedBuffer` path under the cust/cudarc features (the
    // hand-mirrored advice surface returns `Ok(())` for back-compat
    // with v0.3). Asserting `Ok(())` here pins that contract.
    dynref
        .apply_advice(UvmAdvice::SetReadMostly)
        .expect("SetReadMostly is a no-op / Ok on the default backing");
}

/// Mutable trait surface: hand a `&mut dyn UnifiedBacking` reference
/// out and confirm `as_mut_slice` produces a writable view of the
/// expected length.
#[test]
fn dyn_unified_backing_mut_slice_writable() {
    let mut buf = UnifiedBuffer::new(32).expect("alloc");
    let dynref: &mut dyn UnifiedBacking = &mut buf;
    let bytes = dynref.as_mut_slice();
    assert_eq!(bytes.len(), 32);
    bytes.fill(0xA5);
    assert!(dynref.as_slice().iter().all(|&v| v == 0xA5));
}

/// `UnifiedBacking` is `Send + Sync` (object-safe + thread-safe). Pin
/// the bound here at trait-witness scope so a regression that
/// accidentally drops the auto-trait shape fails to compile.
#[test]
fn unified_backing_is_send_sync_object_safe() {
    fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<dyn UnifiedBacking>();
}

/// Round-trip through a `Box<dyn UnifiedBacking>` — the shape the v0.4
/// migration may use to hide the concrete backing behind a stable
/// pointer.
#[test]
fn boxed_dyn_unified_backing_owns_the_buffer() {
    let buf = UnifiedBuffer::new(16).expect("alloc");
    let boxed: Box<dyn UnifiedBacking> = Box::new(buf);
    assert_eq!(boxed.len(), 16);
    assert_eq!(boxed.as_slice().len(), 16);
    // Advice still a no-op via the boxed form on the default backing.
    boxed
        .apply_advice(UvmAdvice::SetReadMostly)
        .expect("Ok on default backing");
}
