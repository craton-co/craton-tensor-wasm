// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `cuda-oxide` host-runtime adapter.
//!
//! This module is the v0.5 cust-successor migration tracked in [RFC
//! 0001](../../../../rfcs/0001-cuda-oxide-integration.md). It exists in two
//! flavours, selected at compile time by which superset feature is enabled:
//!
//! * **`cuda-oxide-backend` (scaffold, dep-less)** — [`Self::allocate`]
//!   returns the documented `NOT_YET_WIRED` sentinel error. No git
//!   dependency is pulled into the resolved graph; `cargo check
//!   --features cuda-oxide-backend` builds on any contributor host even
//!   without a CUDA Toolkit or `libclang`. The scaffold's only job is to
//!   keep the public surface (`CudaOxideUnifiedBuffer`, `apply_advice`,
//!   `CudaOxideAdvice`) stable so call-sites in `tensor-wasm-jit` /
//!   `tensor-wasm-tenant` written today need no re-typing once the host
//!   port lands.
//!
//! * **`cuda-oxide-host-backend` (W4.1 host port)** — strict superset that
//!   additionally pulls in the four cuda-oxide host-side crates
//!   (`cuda-host`, `cuda-core`, `cuda-device`, `cuda-macros` per the
//!   `tensor-wasm-mem` `Cargo.toml`). [`Self::allocate`] forwards to the
//!   real `cuMemAllocManaged` driver call via the
//!   `cuda_core::sys` bindgen output; [`Self::prefetch_async`] wraps
//!   `cuMemPrefetchAsync`; [`apply_advice`] wraps `cuMemAdvise`; and the
//!   `Drop` impl calls `cuMemFree_v2`. The implementation mirrors the
//!   cudarc-backend's [`crate::cudarc_backend`] code path so the two
//!   parallel backends remain easy to diff during the W4.x parity work.
//!
//! See `docs/CUDA-SETUP.md` section "Using the cuda-oxide-host-backend
//! feature" for the toolchain install matrix that the host-backend
//! requires.
//!
//! # What this module IS NOT (yet)
//!
//! * A binding to cuda-oxide's `dialect-mir` / `dialect-llvm` lowering
//!   pipeline. The Wasm→PTX kernel-compilation lever (per RFC 0001
//!   "Pliron lever and the auto-offload pipeline") lives in
//!   `tensor-wasm-jit::pliron_*` and is gated separately.
//! * A `Stream` / `Event` plumbing surface. The host port today uses the
//!   driver's null stream (the same as the cudarc-backend) so the v0.4
//!   diff is body-only against a stable starting point. The W4.x async
//!   integration will thread a `cuda_host::CudaStream` through this
//!   surface in a follow-up PR.
//!
//! # Why the dual scaffold / real impl
//!
//! Splitting the module into two `cfg`-gated implementations keeps the
//! v0.4 host port body-only against W3.x scaffold tests: the
//! `NOT_YET_WIRED` sentinel and the trait-bound (`Send + Sync`) witnesses
//! survive into the host-backend build because the public surface is the
//! same. Tests that need real hardware are isolated to
//! `tests/cuda_oxide_smoke.rs` under the host-backend feature gate.

#![cfg(feature = "cuda-oxide-backend")]

use std::fmt;

use crate::unified::{DeviceId, UnifiedError};

/// Sentinel error message returned by every stub call in this module when
/// the dep-less `cuda-oxide-backend` scaffold is active (i.e. when the
/// strict-superset `cuda-oxide-host-backend` feature is OFF).
///
/// Exposed `pub(crate)` so the unit + integration tests can assert against
/// the exact string without duplicating it. The W4.1 host port leaves this
/// constant in place because it remains observable on contributor boxes
/// that build with only `--features cuda-oxide-backend`.
pub(crate) const NOT_YET_WIRED: &str =
    "cuda-oxide-backend: allocate not yet wired -- see RFC 0001 v0.4 port";

/// Memory-advice hint passed to `cuMemAdvise` when the
/// `cuda-oxide-host-backend` feature is on.
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
// Scaffold path — active when `cuda-oxide-host-backend` is OFF.
// =============================================================================

#[cfg(not(feature = "cuda-oxide-host-backend"))]
mod scaffold {
    use super::*;

    /// A contiguous CUDA Unified Memory region — scaffold form.
    ///
    /// In the dep-less scaffold build this struct stores only the size
    /// the caller passed at construction time; every [`Self::allocate`]
    /// invocation returns [`UnifiedError::Cuda`] with the
    /// [`NOT_YET_WIRED`] sentinel.
    ///
    /// The W4.1 host-backend path replaces this with a real owning
    /// pointer; see the `cuda-oxide-host-backend`-gated `impl` block
    /// below for the actual `cuMemAllocManaged` path.
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
        /// matches the host-backend variant so call sites do not need to
        /// be re-typed once `cuda-oxide-host-backend` is enabled.
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
// Host-backend path — active when `cuda-oxide-host-backend` is ON.
// =============================================================================

#[cfg(feature = "cuda-oxide-host-backend")]
mod host_backend {
    use super::*;

    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock};

    // W4.1: cuda-host re-exports DeviceBuffer / CudaContext / CudaStream
    // from cuda-core; we go through cuda-core directly to make the
    // import surface explicit and to keep `cuda-host` reserved for the
    // kernel-launch path the W4.x dispatcher will use.
    use cuda_core::context::CudaContext;
    use cuda_core::sys as cuda_sys;

    /// Sentinel for `CU_DEVICE_CPU` in `cuMemPrefetchAsync`. CUDA defines
    /// this as `-1` in `cuda.h`; bindgen's output names drift between
    /// CUDA versions so we inline the numeric literal — same approach as
    /// the cudarc-backend.
    const CU_DEVICE_CPU: i32 = -1;

    /// `CU_MEM_ATTACH_GLOBAL` for `cuMemAllocManaged`'s `flags` argument.
    /// Documented as `1` in the CUDA driver headers; inlined for the same
    /// drift-resistance reason as `CU_DEVICE_CPU`.
    ///
    /// W4.1: API surface inferred from cuda-oxide docs; verify against
    /// `cuda-bindings`-generated `CUmemAttach_flags_enum` once a
    /// LIBCLANG-equipped CI runner is available.
    const CU_MEM_ATTACH_GLOBAL: u32 = 1;

    // Numeric values for CUmem_advise. CUDA's `cuda.h` defines:
    //   CU_MEM_ADVISE_SET_READ_MOSTLY            = 1
    //   CU_MEM_ADVISE_UNSET_READ_MOSTLY          = 2
    //   CU_MEM_ADVISE_SET_PREFERRED_LOCATION     = 3
    //   CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION   = 4
    //   CU_MEM_ADVISE_SET_ACCESSED_BY            = 5
    //   CU_MEM_ADVISE_UNSET_ACCESSED_BY          = 6
    //
    // W4.1: numeric values inferred from CUDA driver headers; the
    // cuda-bindings bindgen output names this enum
    // `CUmem_advise_enum_*` but the numeric values are stable across
    // CUDA toolkit versions (12.0 through 13.2 verified). The cuMemAdvise
    // FFI takes `CUmem_advise` (an `i32`-typedef'd enum) so we cast to
    // the bindings' enum type via `as _` at the call site rather than
    // depending on the enum path.
    const CU_MEM_ADVISE_SET_READ_MOSTLY: u32 = 1;
    const CU_MEM_ADVISE_SET_PREFERRED_LOCATION: u32 = 3;
    const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: u32 = 4;
    const CU_MEM_ADVISE_SET_ACCESSED_BY: u32 = 5;
    const CU_MEM_ADVISE_UNSET_ACCESSED_BY: u32 = 6;

    /// Process-wide counter of `cuMemFree_v2` failures observed in
    /// [`CudaOxideUnifiedBuffer`]'s `Drop`. A non-zero value indicates
    /// leaked managed allocations — the driver refused to release them.
    /// Operators can poll this via [`cuda_oxide_free_failures`] to alert
    /// on leak rate. Mirrors the cudarc-backend's `CUDARC_FREE_FAILURES`.
    static FREE_FAILURES: AtomicU64 = AtomicU64::new(0);

    /// Number of `cuMemFree_v2` failures observed since process start
    /// inside the cuda-oxide host backend.
    ///
    /// Each increment corresponds to one leaked managed-memory
    /// allocation. The monotonically increasing log is also emitted at
    /// `tracing::warn!` for forensic correlation.
    pub fn cuda_oxide_free_failures() -> u64 {
        FREE_FAILURES.load(Ordering::Relaxed)
    }

    /// Process-wide cached `Arc<CudaContext>` for device 0.
    ///
    /// `cuda_core::CudaContext::new(0)` retains the device-0 primary
    /// context. Keeping a single `Arc<CudaContext>` alive across
    /// allocations avoids re-running `cuDevicePrimaryCtxRetain` on every
    /// [`CudaOxideUnifiedBuffer::allocate`] call. The W4.x tenant-aware
    /// context cache will route through `tensor-wasm-tenant` instead;
    /// this `OnceLock` is the W4.1 placeholder.
    static DEFAULT_CONTEXT: OnceLock<Arc<CudaContext>> = OnceLock::new();

    /// Lazily fetch (or construct) the cached cuda-oxide context for
    /// device ordinal `ordinal`.
    ///
    /// Mirrors the cudarc-backend's `device_for` so a side-by-side
    /// comparison of the two backends differs only in the underlying
    /// driver call shape, never in the caching strategy.
    fn context_for(ordinal: u32) -> Result<Arc<CudaContext>, UnifiedError> {
        if ordinal == 0 {
            // `get_or_try_init` only invokes the initialiser when this
            // thread actually wins the install race, so the new
            // `Arc<CudaContext>` is never dropped *after* a winner has
            // been installed. Dropping a losing `Arc<CudaContext>` would
            // call `cuDevicePrimaryCtxRelease` on the still-held primary
            // context — invalidating every managed pointer in the
            // process. Same audit-note pattern as cudarc_backend.rs.
            let installed = DEFAULT_CONTEXT
                .get_or_try_init(|| {
                    CudaContext::new(0).map_err(|e| {
                        UnifiedError::Cuda(format!("CudaContext::new(0): {e:?}"))
                    })
                })?;
            Ok(installed.clone())
        } else {
            CudaContext::new(ordinal as usize)
                .map_err(|e| UnifiedError::Cuda(format!("CudaContext::new({ordinal}): {e:?}")))
        }
    }

    /// A contiguous CUDA Unified Memory region allocated via cuda-oxide.
    ///
    /// W4.1 host-backend variant. `ptr` is the address returned by
    /// `cuMemAllocManaged`; it is addressable from both host and device
    /// on a CUDA-capable host and the runtime page-migrates between them
    /// on demand.
    ///
    /// # Safety invariants
    ///
    /// - `ptr` is non-null and points to `size` valid bytes obtained
    ///   from `cuMemAllocManaged`.
    /// - `ctx` is kept alive for the lifetime of the allocation so the
    ///   primary context is not torn down before [`Drop`] runs
    ///   `cuMemFree_v2`.
    pub struct CudaOxideUnifiedBuffer {
        pub(super) ptr: NonNull<u8>,
        pub(super) size: usize,
        pub(super) device_id: DeviceId,
        /// Held purely to keep the primary context alive. Dropping the
        /// last reference releases the context, which would invalidate
        /// every outstanding managed pointer in the process.
        #[allow(dead_code)]
        pub(super) ctx: Arc<CudaContext>,
    }

    // SAFETY: the inner pointer is owned by this struct and not shared
    // without explicit synchronisation. Same contract as `Vec<u8>` once
    // you have a `&mut [u8]`. The `Arc<CudaContext>` is itself
    // `Send + Sync` (per the `unsafe impl` in `cuda_core::context`).
    unsafe impl Send for CudaOxideUnifiedBuffer {}
    unsafe impl Sync for CudaOxideUnifiedBuffer {}

    impl CudaOxideUnifiedBuffer {
        /// Allocate `size` bytes of CUDA Unified Memory via
        /// `cuMemAllocManaged` on the default device.
        ///
        /// Returns [`UnifiedError::ZeroSize`] for a zero-byte request
        /// without touching the driver; otherwise dispatches to
        /// [`Self::allocate_on`] with `DeviceId::default()`.
        ///
        /// W4.1: API surface inferred from cuda-oxide docs; verify
        /// against `cuda-bindings::cuMemAllocManaged` signature once a
        /// LIBCLANG-equipped CI runner is available.
        pub fn allocate(size: usize) -> Result<Self, UnifiedError> {
            Self::allocate_on(size, DeviceId::default())
        }

        /// Allocate `size` bytes of CUDA Unified Memory on the named
        /// device.
        ///
        /// Wraps `cuMemAllocManaged` with `CU_MEM_ATTACH_GLOBAL`, which
        /// matches the semantics of the cust and cudarc backends
        /// (visible to every stream on every device in the current
        /// context).
        ///
        /// W4.1: API surface inferred from cuda-oxide docs; verify
        /// against `cuda-bindings::cuMemAllocManaged` signature once a
        /// LIBCLANG-equipped CI runner is available.
        pub fn allocate_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
            if size == 0 {
                return Err(UnifiedError::ZeroSize);
            }
            let ctx = context_for(device_id.0)?;
            // Make the primary context current on this thread so the
            // driver call below is well-defined. Mirrors cuda-core's own
            // `bind_to_thread` discipline.
            ctx.bind_to_thread().map_err(|e| {
                UnifiedError::Cuda(format!("CudaContext::bind_to_thread: {e:?}"))
            })?;

            let mut raw: cuda_sys::CUdeviceptr = 0;
            // SAFETY: `raw` is a valid out-parameter; `size > 0`; the
            // ctx.bind_to_thread() call above made the primary context
            // current on this thread, which is the precondition for
            // `cuMemAllocManaged`.
            let res = unsafe {
                cuda_sys::cuMemAllocManaged(
                    &mut raw as *mut cuda_sys::CUdeviceptr,
                    size,
                    CU_MEM_ATTACH_GLOBAL,
                )
            };
            if res != cuda_sys::cudaError_enum_CUDA_SUCCESS {
                return Err(UnifiedError::Cuda(format!(
                    "cuMemAllocManaged -> {res:?}"
                )));
            }
            let ptr = NonNull::new(raw as *mut u8).ok_or_else(|| {
                UnifiedError::Allocation(
                    "cuMemAllocManaged returned null with CUDA_SUCCESS".into(),
                )
            })?;
            Ok(Self {
                ptr,
                size,
                device_id,
                ctx,
            })
        }

        /// Length in bytes of this buffer.
        pub fn len(&self) -> usize {
            self.size
        }

        /// True if zero-length. Always false for a successfully
        /// constructed buffer.
        pub fn is_empty(&self) -> bool {
            self.size == 0
        }

        /// Raw const pointer to the first byte.
        pub fn as_ptr(&self) -> *const u8 {
            self.ptr.as_ptr() as *const u8
        }

        /// Raw mutable pointer to the first byte.
        pub fn as_mut_ptr(&mut self) -> *mut u8 {
            self.ptr.as_ptr()
        }

        /// Borrow as a shared byte slice.
        ///
        /// # Safety
        ///
        /// On CUDA-capable hosts the managed memory may be migrated to
        /// the device at any moment; reading from the host side after a
        /// device kernel has written into the region without
        /// synchronising is undefined behaviour. Call sites must
        /// serialise host/device access with a stream sync or an event,
        /// exactly as they do for the cust and cudarc paths.
        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: ptr is non-null and points to `size` valid bytes by
            // the type invariant.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }

        /// Borrow as a mutable byte slice.
        ///
        /// # Safety
        ///
        /// Same caveat as [`Self::as_slice`].
        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: `&mut self` proves uniqueness; ptr/size by the type
            // invariant.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
        }

        /// Which device this buffer is anchored to.
        pub fn device_id(&self) -> DeviceId {
            self.device_id
        }
    }

    impl fmt::Debug for CudaOxideUnifiedBuffer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CudaOxideUnifiedBuffer")
                .field("ptr", &self.ptr.as_ptr())
                .field("size", &self.size)
                .field("device_id", &self.device_id)
                .finish()
        }
    }

    impl Drop for CudaOxideUnifiedBuffer {
        fn drop(&mut self) {
            // SAFETY: `ptr` was returned by `cuMemAllocManaged` and has
            // not been freed yet; the cached `Arc<CudaContext>` we hold
            // ensures the primary context is still alive when we call
            // `cuMemFree_v2`. We also re-bind the context to this
            // thread first because Drop may run on a worker thread that
            // never bound the context (cuda-core's own DeviceBuffer Drop
            // does the same).
            let _ = self.ctx.bind_to_thread();
            let res = unsafe {
                cuda_sys::cuMemFree_v2(self.ptr.as_ptr() as cuda_sys::CUdeviceptr)
            };
            if res != cuda_sys::cudaError_enum_CUDA_SUCCESS {
                let failures = FREE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::warn!(
                    target: "tensor_wasm_mem::cuda_oxide_backend",
                    ?res,
                    ptr = ?self.ptr.as_ptr(),
                    size = self.size,
                    total_failures = failures,
                    "cuMemFree_v2 failed in CudaOxideUnifiedBuffer::drop",
                );
            }
        }
    }

    /// Apply a [`CudaOxideAdvice`] hint via `cuMemAdvise` against a
    /// [`CudaOxideUnifiedBuffer`].
    ///
    /// Returns [`UnifiedError::Cuda`] on a non-`CUDA_SUCCESS` return
    /// from the driver.
    ///
    /// W4.1: API surface inferred from cuda-oxide docs; verify against
    /// `cuda-bindings::cuMemAdvise` signature once a LIBCLANG-equipped
    /// CI runner is available.
    pub fn apply_advice(
        buf: &CudaOxideUnifiedBuffer,
        advice: CudaOxideAdvice,
    ) -> Result<(), UnifiedError> {
        // Re-bind the buffer's owning context onto this thread so the
        // driver call below targets the correct primary context.
        buf.ctx.bind_to_thread().map_err(|e| {
            UnifiedError::Cuda(format!("CudaContext::bind_to_thread: {e:?}"))
        })?;
        let ptr = buf.ptr.as_ptr() as cuda_sys::CUdeviceptr;
        let size = buf.size;
        let (advice_kind, device) = match advice {
            CudaOxideAdvice::ReadMostly => (CU_MEM_ADVISE_SET_READ_MOSTLY, 0i32),
            CudaOxideAdvice::PreferredLocation(d) => {
                (CU_MEM_ADVISE_SET_PREFERRED_LOCATION, d.0 as i32)
            }
            CudaOxideAdvice::AccessedBy(d) => (CU_MEM_ADVISE_SET_ACCESSED_BY, d.0 as i32),
            CudaOxideAdvice::UnsetPreferredLocation => {
                (CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, 0i32)
            }
            CudaOxideAdvice::UnsetAccessedBy(d) => {
                (CU_MEM_ADVISE_UNSET_ACCESSED_BY, d.0 as i32)
            }
        };
        // SAFETY: ptr/size are derived from a valid live
        // CudaOxideUnifiedBuffer. `advice_kind as _` casts the numeric
        // `u32` constant into whatever the bindgen-generated enum repr is
        // (typedef'd to `i32` on every CUDA toolkit version observed).
        let res = unsafe {
            cuda_sys::cuMemAdvise(ptr, size, advice_kind as _, device)
        };
        if res == cuda_sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(UnifiedError::Cuda(format!("cuMemAdvise -> {res:?}")))
        }
    }

    /// Prefetch the buffer asynchronously to `dst_device` (or the host
    /// if `dst_device == DeviceId(u32::MAX)`, which we treat as the
    /// CU_DEVICE_CPU sentinel).
    ///
    /// Wraps `cuMemPrefetchAsync` on the null stream. Use
    /// `DeviceId(u32::MAX)` to migrate back to host memory; any other
    /// value targets the corresponding CUDA device ordinal.
    ///
    /// W4.1: API surface inferred from cuda-oxide docs; verify against
    /// `cuda-bindings::cuMemPrefetchAsync` signature once a
    /// LIBCLANG-equipped CI runner is available.
    pub fn prefetch_async(
        buf: &CudaOxideUnifiedBuffer,
        dst_device: DeviceId,
    ) -> Result<(), UnifiedError> {
        buf.ctx.bind_to_thread().map_err(|e| {
            UnifiedError::Cuda(format!("CudaContext::bind_to_thread: {e:?}"))
        })?;
        let dst = if dst_device.0 == u32::MAX {
            CU_DEVICE_CPU
        } else {
            dst_device.0 as i32
        };
        // SAFETY: ptr/size are derived from a valid live allocation;
        // passing a null stream pointer requests prefetch on the
        // default stream — same convention as the cudarc-backend.
        let res = unsafe {
            cuda_sys::cuMemPrefetchAsync(
                buf.ptr.as_ptr() as cuda_sys::CUdeviceptr,
                buf.size,
                dst,
                std::ptr::null_mut(),
            )
        };
        if res == cuda_sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(UnifiedError::Cuda(format!(
                "cuMemPrefetchAsync(dst={dst}) -> {res:?}"
            )))
        }
    }
}

// =============================================================================
// Re-export the active variant under stable names so callers do not need
// to feature-gate their imports.
// =============================================================================

#[cfg(not(feature = "cuda-oxide-host-backend"))]
pub use scaffold::{apply_advice, prefetch_async, CudaOxideUnifiedBuffer};

#[cfg(feature = "cuda-oxide-host-backend")]
pub use host_backend::{
    apply_advice, cuda_oxide_free_failures, prefetch_async, CudaOxideUnifiedBuffer,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer type must be `Send + Sync` so it can flow through the
    /// same downstream abstractions (`tenant`, `wasi-gpu`) as the cust
    /// and cudarc backings. This is a compile-time assertion via a
    /// trait-bound witness function. Holds across both the scaffold
    /// and the host-backend variants.
    #[test]
    fn buffer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaOxideUnifiedBuffer>();
    }

    /// The scaffold `allocate` always errors with the documented
    /// sentinel string. Skipped when the host-backend feature is on:
    /// in that build `allocate(1024)` either succeeds (on a CUDA box)
    /// or returns a real driver error, never the `NOT_YET_WIRED`
    /// sentinel.
    #[cfg(not(feature = "cuda-oxide-host-backend"))]
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

    /// Under the host-backend feature, `allocate(0)` must reject with
    /// `ZeroSize` before any driver call is made — same contract as the
    /// cudarc-backend. This runs on host-only CI because zero-size
    /// rejection happens entirely in safe Rust before any
    /// `cuMemAllocManaged` invocation.
    #[cfg(feature = "cuda-oxide-host-backend")]
    #[test]
    fn allocate_zero_size_rejected_without_driver() {
        let err = CudaOxideUnifiedBuffer::allocate(0).expect_err("zero should be rejected");
        assert!(
            matches!(err, UnifiedError::ZeroSize),
            "expected ZeroSize, got: {err:?}"
        );
    }
}
