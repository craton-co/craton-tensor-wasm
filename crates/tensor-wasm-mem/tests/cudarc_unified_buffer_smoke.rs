// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Smoke test for the `cudarc-backend` wiring of [`UnifiedBuffer`].
//!
//! Mirrors `tests/cudarc_smoke.rs`'s coverage of the standalone
//! [`tensor_wasm_mem::cudarc_backend::CudarcUnifiedBuffer`] type, but exercises
//! the path where `UnifiedBuffer::new` itself selects `Backing::Cudarc`
//! because `--features cudarc-backend` is enabled and `unified-memory` is
//! NOT. This is the third branch added by the D2 wave that closes the
//! v0.5 cust-successor plan's "cudarc half" (see RFC 0001).
//!
//! The deeper "writes-are-visible-from-device" check lives in the W5.9
//! `cudarc_smoke` suite (which actually launches a kernel) and is left as
//! an ignored placeholder here so the file documents what it deliberately
//! does NOT re-verify.

#![cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]

use tensor_wasm_mem::unified::UnifiedBuffer;

/// Allocate a 64 KiB UVM buffer through `UnifiedBuffer` (which under
/// `cudarc-backend` routes to `CudarcUnifiedBuffer::new` per the precedence
/// table in `unified.rs`). The construction itself proves the cudarc lib()
/// call returned `CUDA_SUCCESS` and that the `Backing::Cudarc` variant is
/// wired up; the assertions below pin the size + pointer contract.
#[test]
fn cudarc_unified_buffer_allocates_64kib() {
    const SIZE: usize = 64 * 1024;
    let b = UnifiedBuffer::new(SIZE).expect("alloc 64 KiB under cudarc-backend");
    assert_eq!(b.len(), SIZE, "reported length must match request");
    assert!(!b.is_empty(), "64 KiB buffer is non-empty");
    assert!(
        b.is_uvm_backed(),
        "cudarc-backend path must report UVM-backed"
    );
}

/// `as_ptr` must be non-null on success; this is the host-side half of the
/// zero-copy contract that `TensorWasmLinearMemory::as_ptr` relies on under
/// the cudarc backing.
#[test]
fn cudarc_unified_buffer_as_ptr_is_non_null() {
    let b = UnifiedBuffer::new(64 * 1024).expect("alloc under cudarc-backend");
    let p = b.as_ptr();
    assert!(!p.is_null(), "cudarc UVM ptr must be non-null on success");
}

/// `as_slice` must report the full requested length (the cudarc backing is
/// fixed-size; there is no separate logical-vs-physical split at this layer).
#[test]
fn cudarc_unified_buffer_as_slice_len_matches_request() {
    const SIZE: usize = 64 * 1024;
    let b = UnifiedBuffer::new(SIZE).expect("alloc under cudarc-backend");
    let s = b.as_slice();
    assert_eq!(
        s.len(),
        SIZE,
        "as_slice() length must equal the requested allocation size"
    );
}

/// Deeper check that writes scribbled host-side become visible from a CUDA
/// kernel. The W5.9 `cudarc_smoke` integration test already covers the
/// underlying `cuMemAllocManaged` → kernel-read-back path against the
/// standalone `CudarcUnifiedBuffer` type, so re-running the same check here
/// would just double-bill the CI runner without exercising a meaningfully
/// different code path. Left as an ignored placeholder for the day the
/// `UnifiedBuffer`-routed path grows its own kernel-launch hookup.
#[test]
#[ignore = "requires CUDA hardware; covered by tests/cudarc_smoke.rs"]
fn writes_visible_from_device_via_unified_buffer() {
    // Intentionally empty: see doc-comment above.
}
