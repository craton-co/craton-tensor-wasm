// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Smoke tests for the `cuda-oxide-backend` scaffold and the
//! `cuda-oxide-host-backend` real-impl path.
//!
//! The file is compiled only when at least the dep-less
//! `cuda-oxide-backend` feature is on. Tests pivot on whether the strict-
//! superset `cuda-oxide-host-backend` feature is also enabled:
//!
//! * Without `cuda-oxide-host-backend`, the scaffold path is active and
//!   the tests assert the documented `NOT_YET_WIRED` sentinel comes back
//!   from `allocate`, that the public types are exported, and that the
//!   buffer layout is non-zero.
//!
//! * With `cuda-oxide-host-backend`, the real `cuMemAllocManaged` path is
//!   active and the tests additionally exercise zero-size rejection on
//!   the host plus the ignored hardware-gated round-trip
//!   (`#[ignore = "requires CUDA hardware"]`) that the S22 CUDA runner
//!   will pick up via `cargo test --features cuda-oxide-host-backend --
//!   --ignored`.
//!
//! Mirrors the shape of `tests/cudarc_smoke.rs` so the v0.4 PR diff is
//! easy to review.
//!
//! Run on a contributor box (no LIBCLANG / CUDA Toolkit required):
//!
//! ```ignore
//! cargo test -p tensor-wasm-mem --features cuda-oxide-backend \
//!     --test cuda_oxide_smoke
//! ```
//!
//! Run on a host with LIBCLANG_PATH + CUDA_TOOLKIT_PATH and a CUDA-
//! capable GPU:
//!
//! ```ignore
//! cargo test -p tensor-wasm-mem --features cuda-oxide-host-backend \
//!     --test cuda_oxide_smoke -- --ignored
//! ```

#![cfg(feature = "cuda-oxide-backend")]

use tensor_wasm_mem::cuda_oxide_backend::{
    apply_advice, prefetch_async, CudaOxideAdvice, CudaOxideUnifiedBuffer,
};
use tensor_wasm_mem::unified::{DeviceId, UnifiedError};

/// The buffer type compiles, links, and has a non-zero layout. Holds on
/// both the scaffold and host-backend builds — if this fails the
/// workspace has a broken feature flag or a missing module declaration
/// in `lib.rs`.
#[test]
fn cuda_oxide_buffer_type_has_nonzero_size() {
    assert!(std::mem::size_of::<CudaOxideUnifiedBuffer>() > 0);
}

/// On the dep-less scaffold build, `allocate(1024)` returns the
/// documented sentinel error string. On the host-backend build this test
/// is skipped — the real impl returns either `Ok` or a real driver
/// error, never the `NOT_YET_WIRED` sentinel.
#[cfg(not(feature = "cuda-oxide-host-backend"))]
#[test]
fn cuda_oxide_allocate_returns_not_yet_wired() {
    let err = CudaOxideUnifiedBuffer::allocate(1024)
        .expect_err("scaffold allocate must error until host-backend port");
    match err {
        UnifiedError::Cuda(msg) => {
            assert!(
                msg.contains("not yet wired"),
                "expected sentinel error string, got: {msg}"
            );
            assert!(
                msg.contains("RFC 0001"),
                "expected RFC reference in sentinel error, got: {msg}"
            );
        }
        other => panic!("expected UnifiedError::Cuda(NOT_YET_WIRED), got {other:?}"),
    }
}

/// `apply_advice` is exported as a free function with the expected
/// signature. Type-level export check — we do NOT invoke it here because
/// the scaffold cannot construct a `CudaOxideUnifiedBuffer`. Mirrors
/// `cudarc_apply_advice_is_exported` in `cudarc_smoke.rs`.
#[test]
fn cuda_oxide_apply_advice_is_exported() {
    let _f: fn(
        &CudaOxideUnifiedBuffer,
        CudaOxideAdvice,
    ) -> Result<(), UnifiedError> = apply_advice;
}

/// `prefetch_async` is exported as a free function with the expected
/// signature. Same type-level export rationale as `apply_advice` above.
#[test]
fn cuda_oxide_prefetch_async_is_exported() {
    let _f: fn(
        &CudaOxideUnifiedBuffer,
        DeviceId,
    ) -> Result<(), UnifiedError> = prefetch_async;
}

/// Under the host-backend feature, zero-byte allocations are rejected
/// without any driver call. Runs on host-only CI as long as the
/// host-backend feature is on and the build host can compile the
/// cuda-oxide crates (LIBCLANG + CUDA Toolkit). The cudarc-backend has
/// the same test under the matching name.
#[cfg(feature = "cuda-oxide-host-backend")]
#[test]
fn cuda_oxide_zero_size_rejected_without_driver() {
    let err = CudaOxideUnifiedBuffer::allocate(0).expect_err("zero should be rejected");
    assert!(
        matches!(err, UnifiedError::ZeroSize),
        "expected ZeroSize, got: {err:?}"
    );
}

/// Hardware-gated round trip: allocate, write, read, drop. Compiles only
/// when the host-backend feature is on. Requires a CUDA driver and at
/// least one visible GPU; marked `#[ignore]` per the repo convention so
/// host-only CI does not try to dlopen `libcuda`.
#[cfg(feature = "cuda-oxide-host-backend")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_oxide_round_trip_on_device() {
    let mut b = CudaOxideUnifiedBuffer::allocate(128).expect("alloc");
    assert_eq!(b.len(), 128);
    assert!(!b.is_empty());
    b.as_mut_slice().copy_from_slice(&[0x5Au8; 128]);
    assert!(b.as_slice().iter().all(|&v| v == 0x5A));
}

/// Hardware-gated advice round trip: allocate then apply
/// `CU_MEM_ADVISE_SET_READ_MOSTLY`. Compute capability ≥ 6.0 required.
#[cfg(feature = "cuda-oxide-host-backend")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_oxide_apply_advice_read_mostly_on_device() {
    let b = CudaOxideUnifiedBuffer::allocate(256).expect("alloc");
    apply_advice(&b, CudaOxideAdvice::ReadMostly)
        .expect("set_read_mostly should succeed on Pascal+");
}

/// Hardware-gated prefetch round trip: prefetch to device 0, then back
/// to the host (`DeviceId(u32::MAX)` sentinel for `CU_DEVICE_CPU`).
#[cfg(feature = "cuda-oxide-host-backend")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_oxide_prefetch_round_trip_on_device() {
    let b = CudaOxideUnifiedBuffer::allocate(64).expect("alloc");
    prefetch_async(&b, DeviceId(0)).expect("prefetch_to_device");
    prefetch_async(&b, DeviceId(u32::MAX)).expect("prefetch_to_host");
}
