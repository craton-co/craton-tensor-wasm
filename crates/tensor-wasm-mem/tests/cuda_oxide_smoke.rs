// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Smoke tests for the `cuda-oxide-backend` scaffold.
//!
//! The file is compiled only when the dep-less `cuda-oxide-backend`
//! feature is on. The scaffold path asserts the documented
//! `NOT_YET_WIRED` sentinel comes back from `allocate`, that the public
//! types are exported, and that the buffer layout is non-zero.
//!
//! NOTE: the real `cuMemAllocManaged` host-backend path (formerly gated by
//! the `experimental-cuda-oxide-host-backend` feature) was removed for
//! crates.io publishability — it relied on git-pinned cuda-oxide crates.
//! The host-only tests below were removed with it; restore them when the
//! host port returns (see RFC 0001 / docs/CUDA-OXIDE-CUTOVER.md).
//!
//! Mirrors the shape of `tests/cudarc_smoke.rs`.
//!
//! Run on a contributor box (no LIBCLANG / CUDA Toolkit required):
//!
//! ```ignore
//! cargo test -p tensor-wasm-mem --features cuda-oxide-backend \
//!     --test cuda_oxide_smoke
//! ```

#![cfg(feature = "cuda-oxide-backend")]

use tensor_wasm_mem::cuda_oxide_backend::{
    apply_advice, prefetch_async, CudaOxideAdvice, CudaOxideUnifiedBuffer,
};
use tensor_wasm_mem::unified::{DeviceId, UnifiedError};

/// The buffer type compiles, links, and has a non-zero layout. If this
/// fails the workspace has a broken feature flag or a missing module
/// declaration in `lib.rs`.
#[test]
fn cuda_oxide_buffer_type_has_nonzero_size() {
    assert!(std::mem::size_of::<CudaOxideUnifiedBuffer>() > 0);
}

/// On the dep-less scaffold build, `allocate(1024)` returns the
/// documented sentinel error string.
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
    let _f: fn(&CudaOxideUnifiedBuffer, CudaOxideAdvice) -> Result<(), UnifiedError> = apply_advice;
}

/// `prefetch_async` is exported as a free function with the expected
/// signature. Same type-level export rationale as `apply_advice` above.
#[test]
fn cuda_oxide_prefetch_async_is_exported() {
    let _f: fn(&CudaOxideUnifiedBuffer, DeviceId) -> Result<(), UnifiedError> = prefetch_async;
}

// NOTE: the host-backend tests (zero-size rejection + the three
// hardware-gated round-trip tests) were removed together with the
// `experimental-cuda-oxide-host-backend` feature and the git-pinned
// cuda-oxide crates it depended on, which blocked crates.io publishing.
// Restore them alongside the host port (RFC 0001 /
// docs/CUDA-OXIDE-CUTOVER.md).
