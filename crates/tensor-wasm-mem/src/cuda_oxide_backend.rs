// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `cuda-oxide` host-runtime adapter — **v0.3.1 scaffold only**.
//!
//! This module is the empty-stub landing site for the v0.5 cust-successor
//! migration tracked in [RFC
//! 0001](../../../../rfcs/0001-cuda-oxide-integration.md). It is gated behind
//! the opt-in `cuda-oxide-backend` feature so the default workspace build
//! stays on the workspace's pinned `nightly-2026-03-15` toolchain — cuda-oxide
//! itself pins `nightly-2026-04-03`, and enabling this feature on the
//! workspace nightly will not build. See the workspace [`Cargo.toml`] comment
//! on the `cuda-host` dependency for the rustup override required to exercise
//! this code path locally, and [`README-cuda-oxide.md`] for the contributor
//! workflow.
//!
//! # What this module IS
//!
//! - A [`CudaOxideUnifiedBuffer`] newtype with a public surface that mirrors
//!   the shape of [`crate::unified::UnifiedBuffer`] and
//!   [`crate::cudarc_backend::CudarcUnifiedBuffer`] so that downstream call
//!   sites can write to a single abstraction across all three backends.
//! - A panicking-stub [`CudaOxideUnifiedBuffer::allocate`] that returns
//!   [`UnifiedError::Cuda`] with an explicit "not yet wired — see RFC 0001
//!   v0.4 port" message, so any caller that accidentally selects this
//!   backend gets a loud actionable error instead of silently allocating
//!   host memory.
//! - A [`Drop`] impl that traces a warning matching the
//!   [`crate::cudarc_backend`] style so the v0.4 port has a working
//!   instrumentation harness to compare against.
//!
//! # What this module is NOT (yet)
//!
//! - A working unified-memory implementation. `cuMemAllocManaged`,
//!   `cuMemPrefetchAsync`, `cuMemAdvise`, and `cuMemFree_v2` all lower to
//!   `unimplemented!`-style stub paths today. The W0.4 follow-up
//!   ([RFC 0001](../../../../rfcs/0001-cuda-oxide-integration.md)
//!   "Rollout — v0.4 (parity)") is where the real port lands.
//! - A binding to cuda-oxide's `cuda-device` or `cuda-macros` crates. Those
//!   are the *kernel-authoring* side of cuda-oxide and arrive on the W4.5
//!   "Path C: Rust kernels via cuda-oxide" track, not this one.
//! - A `Stream` / `Event` plumbing. The v0.4 port needs to thread a
//!   `cuda_host::Stream` through this surface; the scaffold deliberately
//!   omits it so the v0.4 PR is a clean diff against a stable starting
//!   point.
//!
//! # Why the explicit-error stub instead of `unimplemented!()`
//!
//! `unimplemented!()` would panic, which on the [`Drop`] path could mask a
//! real bug behind a double-panic. The
//! [`UnifiedError::Cuda(_ /* not yet wired */)`] return makes the stub
//! observable to tests, traceable in production telemetry, and trivially
//! grep-able for the v0.4 follow-up author.

#![cfg(feature = "cuda-oxide-backend")]

use std::fmt;

use crate::unified::UnifiedError;

/// Sentinel error message returned by every stub call in this module.
///
/// Exposed `pub(crate)` so the unit + integration tests can assert against
/// the exact string without duplicating it. The v0.4 port will delete this
/// constant.
pub(crate) const NOT_YET_WIRED: &str =
    "cuda-oxide-backend: allocate not yet wired -- see RFC 0001 v0.4 port";

/// Placeholder advice enum so the stub `apply_advice` has a parameter shape
/// that matches the cudarc backend's free function. Mirrored from
/// [`crate::advise::Advice`] deliberately rather than re-exported — the
/// v0.4 port will replace this with the real enum once the cuda-oxide
/// `mem_advise` mapping is settled.
#[derive(Debug, Clone, Copy)]
pub enum CudaOxideAdvice {
    /// Placeholder for `CU_MEM_ADVISE_SET_READ_MOSTLY`. Carries no payload
    /// in the scaffold; the v0.4 port will replace this with the structured
    /// [`crate::advise::Advice`] enum.
    ReadMostly,
}

/// A contiguous CUDA Unified Memory region allocated via `cuda-oxide`'s
/// `cuda-host` runtime.
///
/// **Scaffold only.** The inner storage is a [`PhantomData`]-flavoured stub
/// today; the v0.4 port will replace it with the real
/// `cuda_host::DeviceBuffer<u8>` (or whatever the published v0.2 type name
/// is — see TODO below). Constructing one of these via
/// [`Self::allocate`] is the public entry point and currently always
/// returns an error.
///
/// [`PhantomData`]: std::marker::PhantomData
pub struct CudaOxideUnifiedBuffer {
    /// Size in bytes the caller asked for. Stored so [`Self::len`] is
    /// reachable from tests once a buffer is constructible (today every
    /// construction errors out, so this field is only exercised indirectly
    /// via the Send/Sync trait-bound tests).
    size: usize,
    // TODO(v0.4 port): replace with the real cuda-oxide owned allocation
    // handle. As of cuda-oxide v0.1.0 (2026-05-09) the most likely type
    // name is `cuda_host::DeviceBuffer<u8>` — the crate's README example
    // uses `DeviceBuffer::<u8>::new(size)` — but the v0.1 alpha may rename
    // it. The orchestrator that bumps the toolchain to nightly-2026-04-03
    // for the v0.4 port should verify the exact import path before
    // wiring this struct.
    //
    // The PhantomData below is what keeps the struct !Send/!Sync neutral
    // (it carries `*mut u8` semantics in the eventual real impl).
    _todo_inner: std::marker::PhantomData<*mut u8>,
}

// SAFETY: the scaffold owns no raw pointer yet — the PhantomData<*mut u8>
// is purely a placeholder for the v0.4 port's real DeviceBuffer field.
// The Send/Sync claim mirrors the contract documented on
// `cudarc_backend::CudarcUnifiedBuffer` (`Vec<u8>`-style: sendable as an
// owned value, concurrent access to the bytes requires external sync).
// The v0.4 port keeps these impls; only the inner field changes.
unsafe impl Send for CudaOxideUnifiedBuffer {}
unsafe impl Sync for CudaOxideUnifiedBuffer {}

impl CudaOxideUnifiedBuffer {
    /// Allocate `size` bytes of CUDA Unified Memory via the cuda-oxide
    /// host runtime.
    ///
    /// **Scaffold stub.** Always returns
    /// `Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`. The v0.4 port will
    /// replace the body with a real `cuda_host` managed allocation; the
    /// signature is intentionally already final so the v0.4 diff is
    /// body-only.
    pub fn allocate(size: usize) -> Result<Self, UnifiedError> {
        // Intentionally swallow the `size` argument — the v0.4 port will use
        // it. Binding to `_size` keeps the linter quiet without renaming
        // the public parameter, which would be a doc-visible churn.
        let _size = size;
        Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
    }

    /// Length in bytes of this buffer.
    ///
    /// **Scaffold:** always returns the `size` field as captured at
    /// construction. Today no construction succeeds, so this method is
    /// unreachable in practice; it exists so the v0.4 port can land
    /// without changing the public surface.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length. Always false for a successfully constructed
    /// buffer (today that means: unreachable, since no construction
    /// succeeds).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Apply a memory-advice hint against a [`CudaOxideUnifiedBuffer`].
///
/// **Scaffold stub.** Always returns
/// `Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`. Mirrors the shape of
/// [`crate::cudarc_backend::apply_advice`] so call sites can write
/// backend-agnostic code today and the v0.4 port is a body-only diff.
pub fn apply_advice(
    _buf: &CudaOxideUnifiedBuffer,
    _advice: CudaOxideAdvice,
) -> Result<(), UnifiedError> {
    Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
}

impl fmt::Debug for CudaOxideUnifiedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaOxideUnifiedBuffer")
            .field("size", &self.size)
            .field("status", &"scaffold/not-yet-wired")
            .finish()
    }
}

impl Drop for CudaOxideUnifiedBuffer {
    fn drop(&mut self) {
        // Mirrors the cudarc_backend.rs Drop style: emit a tracing event so
        // post-mortem tooling can spot the leak. Today this branch is
        // dead — no construction succeeds — but the v0.4 port will wire it
        // to a real `cuda_host` free call and this warn will only fire on
        // a genuine free failure.
        tracing::warn!(
            target: "tensor_wasm_mem::cuda_oxide_backend",
            size = self.size,
            "CudaOxideUnifiedBuffer dropped, but no real free happened -- \
             scaffold stub, see RFC 0001 v0.4 port",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer type must be `Send + Sync` so it can flow through the
    /// same downstream abstractions (`tenant`, `wasi-gpu`) as the cust and
    /// cudarc backings. This is a compile-time assertion via a trait-bound
    /// witness function.
    #[test]
    fn buffer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaOxideUnifiedBuffer>();
    }

    /// The scaffold `allocate` always errors with the documented sentinel
    /// string. The v0.4 port deletes this test (or rewrites it to assert
    /// successful allocation against a `cuda-oxide` test runner).
    #[test]
    fn allocate_returns_not_yet_wired_error() {
        let err = CudaOxideUnifiedBuffer::allocate(1024).expect_err("scaffold must error");
        match err {
            UnifiedError::Cuda(msg) => {
                assert_eq!(msg, NOT_YET_WIRED);
            }
            other => panic!("expected UnifiedError::Cuda(NOT_YET_WIRED), got {other:?}"),
        }
    }
}
