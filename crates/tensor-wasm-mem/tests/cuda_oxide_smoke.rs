// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Smoke tests for the `cuda-oxide-backend` scaffold.
//!
//! These tests are compiled only when the `cuda-oxide-backend` feature is
//! enabled. As of v0.3.1 the backend is a scaffold only — the unified-memory
//! port lands at v0.4 per RFC 0001
//! (`rfcs/0001-cuda-oxide-integration.md`). The three unignored tests below
//! assert that:
//!
//! 1. The public types are reachable from outside the crate (i.e. the
//!    feature flag is plumbed and the module declaration in `lib.rs` is
//!    correct);
//! 2. The scaffold `allocate` returns the documented sentinel error string
//!    rather than panicking or succeeding silently;
//! 3. The scaffold `apply_advice` returns the same sentinel error string.
//!
//! The `#[ignore]` test below documents the hardware-and-port-gated round
//! trip that will replace tests 2 and 3 at v0.4.
//!
//! Mirrors the shape of `tests/cudarc_smoke.rs` so the v0.4 PR diff is easy
//! to review.
//!
//! Run on the cuda-oxide-pinned nightly:
//!
//! ```ignore
//! RUSTUP_TOOLCHAIN=nightly-2026-04-03 cargo test \
//!     -p tensor-wasm-mem --features cuda-oxide-backend \
//!     --test cuda_oxide_smoke
//! ```

#![cfg(feature = "cuda-oxide-backend")]

use tensor_wasm_mem::cuda_oxide_backend::{
    apply_advice, CudaOxideAdvice, CudaOxideUnifiedBuffer,
};
use tensor_wasm_mem::unified::UnifiedError;

/// The buffer type compiles, links, and has a non-zero layout. This is the
/// headline "the scaffold code path is wired" assertion — if this fails
/// the workspace has a broken feature flag or a missing module declaration
/// in `lib.rs`.
#[test]
fn cuda_oxide_buffer_type_has_nonzero_size() {
    assert!(std::mem::size_of::<CudaOxideUnifiedBuffer>() > 0);
}

/// The scaffold `allocate` returns the documented sentinel error string,
/// proving the stub is observable from outside the crate (so the v0.4
/// porting author can grep for the call sites that need updating).
#[test]
fn cuda_oxide_allocate_returns_not_yet_wired() {
    let err = CudaOxideUnifiedBuffer::allocate(1024)
        .expect_err("scaffold allocate must error until v0.4 port");
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

/// The scaffold `apply_advice` is exported as a free function with the
/// expected signature. We do NOT invoke it here — the scaffold cannot
/// construct a `CudaOxideUnifiedBuffer` (every `allocate` errors), and
/// fabricating one via `MaybeUninit` would be UB even though the stub
/// body never reads its arguments. This is a type-level export check,
/// mirroring `cudarc_apply_advice_is_exported` in `cudarc_smoke.rs`. The
/// v0.4 port replaces this test with a real allocate + apply round-trip.
#[test]
fn cuda_oxide_apply_advice_is_exported() {
    let _f: fn(
        &CudaOxideUnifiedBuffer,
        CudaOxideAdvice,
    ) -> Result<(), UnifiedError> = apply_advice;
}

/// Placeholder for the v0.4 hardware round-trip test: allocate, write,
/// read, drop. Will only pass once the v0.4 port lands AND the host has
/// cuda-oxide v0.2+ + a CUDA-capable GPU + the nightly-2026-04-03
/// toolchain. Kept here so the v0.4 PR author has an obvious un-ignore
/// target.
#[test]
#[ignore = "requires cuda-oxide v0.2+ and the v0.4 port"]
fn cuda_oxide_round_trip_on_device_v0_4() {
    let _b = CudaOxideUnifiedBuffer::allocate(128)
        .expect("v0.4 port: allocate(128) must succeed on cuda-oxide v0.2+ with hardware");
    // v0.4 port will:
    //   assert_eq!(_b.len(), 128);
    //   _b.as_mut_slice().copy_from_slice(&[0x5A; 128]);
    //   assert!(_b.as_slice().iter().all(|&v| v == 0x5A));
    //   apply_advice(&_b, CudaOxideAdvice::ReadMostly)
    //       .expect("set_read_mostly on Pascal+");
}
