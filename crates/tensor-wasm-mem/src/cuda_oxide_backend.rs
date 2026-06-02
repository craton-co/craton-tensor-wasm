// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `cuda-oxide` host-runtime adapter (dep-less scaffold).
//!
//! This module is the v0.5 cust-successor migration tracked in [RFC
//! 0001](../../../../rfcs/0001-cuda-oxide-integration.md).
//!
//! **`cuda-oxide-backend` (scaffold, dep-less)** — [`Self::allocate`]
//! returns the documented `NOT_YET_WIRED` sentinel error. No git
//! dependency is pulled into the resolved graph; `cargo check --features
//! cuda-oxide-backend` builds on any contributor host even without a CUDA
//! Toolkit or `libclang`, and the crate stays publishable to crates.io.
//! The scaffold's only job is to keep the public surface
//! (`CudaOxideUnifiedBuffer`, `apply_advice`, `CudaOxideAdvice`) stable so
//! call-sites in `tensor-wasm-jit` / `tensor-wasm-tenant` written today
//! need no re-typing once the real host port lands.
//!
//! NOTE: the real `cuMemAllocManaged`/`cuMemAdvise`/`cuMemPrefetchAsync`/
//! `cuMemFree_v2` host port previously lived in this file behind an
//! `experimental-cuda-oxide-host-backend` feature that pulled in the
//! git-pinned cuda-oxide host/device crates (`cuda-host`, `cuda-core`,
//! `cuda-async`, `cuda-device`, `cuda-macros`). crates.io rejects any
//! manifest with a `git` dependency — even an optional, feature-gated one
//! — so both that feature and those deps were REMOVED to unblock
//! publishing. The host port never actually compiled (its module opened
//! with a `compile_error!`), so nothing that built before is lost. It can
//! be re-added once cuda-oxide ships its workspace members to crates.io;
//! see RFC 0001 and `docs/CUDA-OXIDE-CUTOVER.md` for the cutover plan.
//!
//! # What this module IS NOT (yet)
//!
//! * A binding to cuda-oxide's `dialect-mir` / `dialect-llvm` lowering
//!   pipeline. The Wasm→PTX kernel-compilation lever (per RFC 0001
//!   "Pliron lever and the auto-offload pipeline") lives in
//!   `tensor-wasm-jit::pliron_*` and is gated separately.
//! * A real driver-backed allocation surface. Every call returns the
//!   `NOT_YET_WIRED` sentinel until the host port is restored.

#![cfg(feature = "cuda-oxide-backend")]

use std::collections::HashSet;
use std::fmt;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::unified::{DeviceId, UnifiedBacking, UnifiedError, UvmAdvice};

/// Process-global set of raw CUDA device-pointer values orphaned by a failed
/// free on the cuda-oxide path. Mirrors
/// [`crate::cudarc_backend::leaked_cuda_allocations`]'s contract so the v0.4
/// port inherits an instrumentation harness when the real `cuda_host` free
/// call lands here. Today no construction succeeds, so the set stays empty
/// in practice — but the public accessor exists so operator tooling can wire
/// against a stable surface now. See `cudarc_backend` for the threat-model
/// write-up; this backend inherits the same mem H4 invariant once it grows
/// a real free path.
static LEAKED_CUDA_ALLOCATIONS: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();

fn leaked_set() -> &'static Mutex<HashSet<u64>> {
    LEAKED_CUDA_ALLOCATIONS.get_or_init(|| Mutex::new(HashSet::new()))
}

// `record_leak` will be called from the real host-backend `Drop` on every
// `cuMemFree_v2` failure (mem H4 invariant) once the cuda-oxide host port
// is restored. In the current scaffold-only build no construction
// succeeds, so the call site is unreachable and the function is dead code;
// the unconditional allow keeps `-D warnings` happy on contributor boxes.
#[allow(dead_code)]
fn record_leak(ptr: u64) {
    leaked_set().lock().insert(ptr);
}

/// Snapshot of the process-global set of CUDA pointers orphaned by a failed
/// free on the cuda-oxide path. See
/// [`crate::cudarc_backend::leaked_cuda_allocations`] for the full audit
/// contract; this accessor is the cuda-oxide-path mirror so operator
/// tooling has a single shape across backends.
pub fn leaked_cuda_allocations() -> Vec<u64> {
    leaked_set().lock().iter().copied().collect()
}

/// Sentinel error message returned by every stub call in this module while
/// the dep-less `cuda-oxide-backend` scaffold is the only path (the real
/// host port was removed for crates.io publishability — see the module NOTE).
///
/// Exposed `pub(crate)` so the unit + integration tests can assert against
/// the exact string without duplicating it. The future host port should
/// leave this constant in place because it remains observable on
/// contributor boxes that build with `--features cuda-oxide-backend`.
pub(crate) const NOT_YET_WIRED: &str =
    "cuda-oxide-backend: allocate not yet wired -- see RFC 0001 v0.4 port";

/// Memory-advice hint that the future cuda-oxide host port will pass to
/// `cuMemAdvise`. Kept on the scaffold path so the public surface is stable.
///
/// Mirrors the shape of [`crate::advise::Advice`] but is declared locally so
/// the dep-less scaffold build (without the host crates) still has a
/// concrete enum to compile against. The variants intentionally do not
/// re-export the full [`crate::advise::Advice`] enum so that
/// `cuda_oxide_backend.rs` has zero non-feature-gated dependence on the
/// `crate::advise` module — a regression in the cust path's advice table
/// cannot break a host-backend build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaOxideAdvice {
    /// `CU_MEM_ADVISE_SET_READ_MOSTLY` — pages are read on many devices.
    ReadMostly,
    /// `CU_MEM_ADVISE_SET_PREFERRED_LOCATION` for the given device ordinal.
    PreferredLocation(DeviceId),
    /// `CU_MEM_ADVISE_SET_ACCESSED_BY` for the given device ordinal.
    AccessedBy(DeviceId),
    /// `CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`.
    UnsetPreferredLocation,
    /// `CU_MEM_ADVISE_UNSET_ACCESSED_BY` for the given device ordinal.
    UnsetAccessedBy(DeviceId),
}

// =============================================================================
// Scaffold path — the only path today. The real `cuMemAllocManaged` host
// backend was removed for crates.io publishability (it relied on git-pinned
// cuda-oxide crates); see the module-level NOTE and RFC 0001 /
// docs/CUDA-OXIDE-CUTOVER.md.
// =============================================================================

mod scaffold {
    use super::*;

    /// A contiguous CUDA Unified Memory region — scaffold form.
    ///
    /// In the dep-less scaffold build this struct stores only the size
    /// the caller passed at construction time; every [`Self::allocate`]
    /// invocation returns [`UnifiedError::Cuda`] with the
    /// [`NOT_YET_WIRED`] sentinel.
    ///
    /// The future cuda-oxide host port will replace this with a real
    /// owning pointer doing the actual `cuMemAllocManaged` path; that port
    /// was removed for crates.io publishability (see the module NOTE).
    pub struct CudaOxideUnifiedBuffer {
        /// Size in bytes the caller asked for. Stored so [`Self::len`] is
        /// reachable from tests once a buffer is constructible (today
        /// every construction errors out, so this field is exercised only
        /// indirectly via the Send/Sync trait-bound tests).
        pub(super) size: usize,
        /// PhantomData keeps the struct's auto-trait shape consistent with
        /// the host-backend variant below, which holds a `*mut u8`.
        pub(super) _todo_inner: std::marker::PhantomData<*mut u8>,
    }

    // SAFETY: the scaffold owns no raw pointer yet — the PhantomData<*mut u8>
    // is purely a placeholder so this struct's Send/Sync shape matches the
    // host-backend variant (which holds an owned device pointer). The
    // Send/Sync claim mirrors the contract documented on
    // `cudarc_backend::CudarcUnifiedBuffer` (`Vec<u8>`-style: sendable as
    // an owned value, concurrent access to the bytes requires external
    // synchronisation).
    unsafe impl Send for CudaOxideUnifiedBuffer {}
    unsafe impl Sync for CudaOxideUnifiedBuffer {}

    impl CudaOxideUnifiedBuffer {
        /// Allocate `size` bytes of CUDA Unified Memory.
        ///
        /// Scaffold stub. Always returns
        /// `Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`. The signature
        /// matches the future host port so call sites do not need to be
        /// re-typed once that port is restored.
        pub fn allocate(size: usize) -> Result<Self, UnifiedError> {
            // Intentionally swallow the `size` argument — the host
            // variant uses it. Binding to `_size` keeps the linter quiet
            // without renaming the public parameter, which would be a
            // doc-visible churn.
            let _size = size;
            Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
        }

        /// Length in bytes of this buffer.
        ///
        /// Scaffold: always returns the `size` field as captured at
        /// construction. Today no construction succeeds, so this method
        /// is unreachable in practice; it exists so the host-backend
        /// build presents the same public surface.
        pub fn len(&self) -> usize {
            self.size
        }

        /// True if zero-length.
        ///
        /// Always false for a successfully constructed buffer (today that
        /// means: unreachable, since no construction succeeds).
        pub fn is_empty(&self) -> bool {
            self.size == 0
        }
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
            // Mirrors the cudarc_backend.rs Drop style: emit a tracing
            // event so post-mortem tooling can spot the leak. Today this
            // branch is dead — no construction succeeds — but the
            // host-backend build below wires it to a real `cuMemFree_v2`
            // call and this warn will only fire on a genuine free
            // failure.
            tracing::warn!(
                target: "tensor_wasm_mem::cuda_oxide_backend",
                size = self.size,
                "CudaOxideUnifiedBuffer dropped, but no real free happened -- \
                 scaffold stub, see RFC 0001 v0.4 port",
            );
        }
    }

    impl UnifiedBacking for CudaOxideUnifiedBuffer {
        fn len(&self) -> usize {
            CudaOxideUnifiedBuffer::len(self)
        }

        fn as_slice(&self) -> &[u8] {
            // The scaffold has no real allocation (every `allocate`
            // call returns `NOT_YET_WIRED`), so the byte view is
            // always empty. This branch is effectively unreachable —
            // no caller can hold a `&CudaOxideUnifiedBuffer` today
            // because construction always errors out — but the trait
            // requires a concrete return so we hand back a zero-length
            // slice on a well-aligned dangling pointer. The host-
            // backend variant below returns the real bytes.
            &[]
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            // See `as_slice`: scaffold path is empty.
            &mut []
        }

        fn apply_advice(&self, _hint: UvmAdvice) -> Result<(), UnifiedError> {
            // Mirrors the free-function scaffold stub: every method
            // that would touch the driver surfaces `NOT_YET_WIRED` so
            // the v0.4 port has a single sentinel to grep for. Callers
            // that branch on the trait surface still get a structured
            // `UnifiedError::Cuda` and can match on the message.
            Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
        }

        fn prefetch_to_device(&self, _device_ord: u32) -> Result<(), UnifiedError> {
            Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
        }

        fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
            Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
        }
    }

    /// Apply a memory-advice hint against a [`CudaOxideUnifiedBuffer`].
    ///
    /// Scaffold stub. Always returns
    /// `Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`. Mirrors the shape
    /// of [`crate::cudarc_backend::apply_advice`] so call sites can write
    /// backend-agnostic code today and the host-backend build below is a
    /// body-only diff.
    pub fn apply_advice(
        _buf: &CudaOxideUnifiedBuffer,
        _advice: CudaOxideAdvice,
    ) -> Result<(), UnifiedError> {
        Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
    }

    /// Prefetch the buffer asynchronously to a destination device.
    ///
    /// Scaffold stub. Always returns
    /// `Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))`. Signature mirrors
    /// the host-backend variant so call sites compile against both.
    pub fn prefetch_async(
        _buf: &CudaOxideUnifiedBuffer,
        _dst_device: DeviceId,
    ) -> Result<(), UnifiedError> {
        Err(UnifiedError::Cuda(NOT_YET_WIRED.into()))
    }
}

// =============================================================================
// Re-export the scaffold variant under stable names so callers do not need
// to feature-gate their imports.
//
// MEM H4 INVARIANT (contract for the future cuda-oxide host backend):
// - The Drop impl on the real `CudaOxideUnifiedBuffer` MUST never panic.
// - On any non-success return from the real free call, the raw pointer
//   MUST be passed to `record_leak()` (in this module) and the failure
//   logged at `error!` level -- see `cudarc_backend::Drop` for the
//   reference pattern. A failed free is a security-relevant event:
//   the driver may recycle the same VA for another tenant carrying
//   residual bytes.
// - After a failed free, the inner allocation handle MUST be replaced
//   with a sentinel so any hypothetical double-Drop short-circuits.
// - `leaked_cuda_allocations()` above is the operator audit surface.
//
// NOTE: the real `host_backend` module that satisfied these invariants was
// removed for crates.io publishability (it depended on git-pinned
// cuda-oxide crates). Restore it -- and re-export
// `cuda_oxide_free_failures` alongside the names below -- once cuda-oxide
// publishes to crates.io. See RFC 0001 / docs/CUDA-OXIDE-CUTOVER.md.
// =============================================================================

pub use scaffold::{apply_advice, prefetch_async, CudaOxideUnifiedBuffer};

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer type must be `Send + Sync` so it can flow through the
    /// same downstream abstractions (`tenant`, `wasi-gpu`) as the cust
    /// and cudarc backings. This is a compile-time assertion via a
    /// trait-bound witness function.
    #[test]
    fn buffer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaOxideUnifiedBuffer>();
    }

    /// The scaffold `allocate` always errors with the documented
    /// sentinel string.
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
