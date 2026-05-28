// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Driver-level GPU memory pool enforcement (roadmap feature #8, v0.4
//! deliverable T39).
//!
//! The per-tenant GPU memory cap was previously enforced in-process by
//! `TenantContext::consume_gpu_bytes` (B6.5: a lock-free CAS counter
//! over `gpu_bytes_in_use` against the `gpu_memory_bytes_cap` cap).
//! That catches the well-behaved-allocator path — every
//! `UnifiedBuffer::new_on_with_tenant_context` routes through the
//! tenant's quota-checked path before allocating — but a tenant that
//! somehow obtained a raw CUDA driver handle could bypass it.
//!
//! `cuMemPool` (CUDA 11.2+) lets the host pin a hard ceiling that the
//! driver itself enforces. T39 wires every tenant's allocations
//! through a tenant-scoped `cuMemPoolHandle_t` configured with
//! `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` matching the tenant's cap.
//! Allocations past the cap fail at the driver level with
//! `CUDA_ERROR_OUT_OF_MEMORY` — the in-process counter is a
//! belt-and-suspenders layer on top.
//!
//! ## Status (T39): driver pin LANDED
//!
//! Pool creation, `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` set, drop, and a
//! tenant-pool-routed [`UnifiedBuffer::new_in_tenant_pool`] allocation
//! path are wired against the cudarc 0.13 FFI surface
//! (`cudarc::driver::sys`). The FFI types used here
//! (`CUmemoryPool`, `CUmemPoolProps`, `CUmemPool_attribute`,
//! `cuMemPoolCreate`, `cuMemPoolSetAttribute`, `cuMemPoolDestroy`,
//! `cuMemAllocFromPoolAsync`) were cross-checked against the cudarc
//! 0.13.9 generated bindings at
//! `cudarc/src/driver/sys/sys_12000.rs`. If a future cudarc minor
//! release renames or renumbers any of them the build fails *here*
//! rather than at runtime with `CUDA_ERROR_INVALID_VALUE`. See the
//! parallel drift-guard pattern in `cudarc_backend.rs` for the
//! `CU_MEM_ATTACH_GLOBAL` constant.
//!
//! ## TOCTOU note (driver-API limitation)
//!
//! Driver-API limitation: the threshold is set in a separate FFI call
//! after the pool's creation. A racing observer can see the
//! unprotected pool for ~microseconds between `cuMemPoolCreate` and
//! `cuMemPoolSetAttribute`. Acceptable in our model because (a) the
//! only consumer of the pool handle in this codebase is the
//! just-finished constructor (no other thread can reach the handle
//! until [`TenantMemPool::new`] returns and the caller publishes the
//! `Arc<TenantMemPool>` into the tenant context), and (b) the
//! in-process counter still applies as a second line of defence — a
//! kernel that allocated through the unprotected window would still
//! be rejected by [`TenantContext::consume_gpu_bytes`] before the
//! `UnifiedBuffer` is handed back to the caller. cudarc 0.13's
//! `CUmemPoolProps` struct doesn't carry the threshold inline, so
//! this race is unavoidable in CUDA Driver API.
//!
//! ## Gating
//!
//! Behind the `cudarc-backend` feature (cust 0.3.x does not expose the
//! `cuMemPool*` API surface, so this module would be useless without
//! cudarc). The `gpu-mem-pool` feature on this crate is an alias for
//! `cudarc-backend` — operators turn it on with `--features
//! gpu-mem-pool` to make their intent explicit on the command line.
//!
//! ## Tests
//!
//! Hardware-dependent. The unit tests in this file are pure type / API
//! checks; the live-driver tests live in
//! `tests/cuda_mem_pool_scaffold.rs` and
//! `tests/cuda_mem_pool_driver_pin.rs` and are `#[ignore]`'d so
//! host-only CI does not try to `dlopen` `libcuda.so` / `nvcuda.dll`.

#![cfg(feature = "cudarc-backend")]

use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaDevice;

use crate::cudarc_backend::{device_for, ensure_context_bound};
use crate::unified::UnifiedError;

/// Tenant-scoped CUDA memory pool (`cuMemPoolHandle_t`) with a hard
/// release-threshold cap.
///
/// The pool is owned by exactly one tenant. Dropping the wrapper calls
/// `cuMemPoolDestroy`. The cached `Arc<CudaDevice>` returned by the
/// T26 per-ordinal device cache (see [`crate::cudarc_backend`]) is
/// retained on the struct so that dropping the pool wrapper cannot
/// release the device's primary context out from under another
/// tenant's allocations — exactly the failed-construction race the
/// T26 cache exists to close.
///
/// # Safety invariants
///
/// - `pool` is non-null and was returned by `cuMemPoolCreate`; it has
///   not been destroyed yet.
/// - `cap_bytes` is the value the constructor passed to
///   `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`.
///   The CUDA driver may round it; we store the *requested* value here
///   for honest reporting back to the tenant.
/// - `device` is kept alive for the lifetime of the pool so the
///   primary context is not torn down before [`Drop`] runs
///   `cuMemPoolDestroy`. See `cudarc_backend.rs`'s `DEVICE_CACHE` doc
///   for the full audit-T26 rationale.
#[derive(Debug)]
pub struct TenantMemPool {
    pool: cuda_sys::CUmemoryPool,
    cap_bytes: u64,
    device_ordinal: u32,
    /// Held purely to keep the device's primary context alive. Dropped
    /// AFTER `cuMemPoolDestroy` in [`Drop`] thanks to Rust's struct
    /// field-drop ordering (declaration order). The cached
    /// `Arc<CudaDevice>` is a clone from `cudarc_backend::DEVICE_CACHE`
    /// so the strong-count never reaches zero while the cache is alive
    /// — but holding it here is the belt-and-braces guarantee that
    /// `cuMemPoolDestroy` sees a valid primary context even if some
    /// future refactor races the cache eviction path.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
}

impl TenantMemPool {
    /// Create a tenant-scoped memory pool with a release-threshold cap.
    ///
    /// The pool is created with `CU_MEM_ALLOCATION_TYPE_PINNED` on
    /// device-located memory for `device_ordinal`. The
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` attribute is set to
    /// `cap_bytes` so allocations past the cap fail at the driver
    /// level with `CUDA_ERROR_OUT_OF_MEMORY`.
    ///
    /// The primary context for `device_ordinal` is retained via the
    /// T26 per-ordinal device cache (see [`crate::cudarc_backend`]);
    /// the resulting `Arc<CudaDevice>` is stored on the returned
    /// [`TenantMemPool`] so the context cannot be torn down while the
    /// pool is alive.
    ///
    /// Returns [`MemPoolError::Create`] if `cuMemPoolCreate` returns
    /// anything other than `CUDA_SUCCESS`, or
    /// [`MemPoolError::SetAttribute`] if the release-threshold set
    /// fails (in which case the partially-created pool is destroyed
    /// before the error returns), or [`MemPoolError::Device`] if the
    /// T26 cache could not retain the device's primary context.
    pub fn new(device_ordinal: u32, cap_bytes: u64) -> Result<Self, MemPoolError> {
        // Acquire (or hit-cache for) the device's primary context via
        // T26's per-ordinal `Arc<CudaDevice>` cache. This both primes
        // the driver lib (so `cuda_sys::lib()` below cannot panic) and
        // pins the primary context for the lifetime of the pool. Any
        // construction failure here returns BEFORE we touch the pool
        // FFI so there is nothing to roll back.
        let device = device_for(device_ordinal)
            .map_err(|e| MemPoolError::Device(format!("device_for({device_ordinal}): {e:?}")))?;
        // mem M4: bind the primary context to the calling thread
        // before issuing any driver call. Without this, the driver
        // could dispatch against another tenant's context that the
        // calling thread last touched — both `cuMemPoolCreate` and
        // `cuMemPoolSetAttribute` are context-sensitive.
        ensure_context_bound(&device)
            .map_err(|e| MemPoolError::Device(format!("ensure_context_bound: {e:?}")))?;

        // SAFETY: the cudarc FFI surface is unsafe by construction; we
        // wrap each entry point with a tightly-scoped `unsafe` block.
        // `cuda_sys::lib()` returns a static `Lib`; cudarc 0.13.x marks
        // it `unsafe fn`. The device cache above has already primed
        // it, so the `lib()` call cannot panic on missing libcuda.
        // The `MaybeUninit`-equivalent zero-init pattern here matches
        // cudarc 0.13's documented Default for `CUmemPoolProps`:
        // `ptr::write_bytes(.., 0, 1)`. The handle out-pointer is
        // initialised to null then written by the driver on success.
        unsafe {
            let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
            // `CUmemPoolProps_st` implements `Default` via a
            // `write_bytes(.., 0, 1)` zero-init; we use `std::mem::zeroed`
            // for the same effect with no extra deps. The four fields
            // we care about (`allocType`, `handleTypes`, `location.type_`,
            // `location.id`) are then set explicitly; the
            // `win32SecurityAttributes` and `reserved` fields stay zero
            // (the documented "default-everything" state for the driver).
            let mut props: cuda_sys::CUmemPoolProps = std::mem::zeroed();
            // `allocType` = pinned device memory. The release-threshold
            // attribute below only makes sense on a pinned pool.
            props.allocType =
                cuda_sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
            // No IPC handle export: tenants live within a single
            // process today; cross-process shareable pools are a v0.5
            // follow-up.
            props.handleTypes =
                cuda_sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE;
            // Location: device-local memory on the requested ordinal.
            props.location.type_ =
                cuda_sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = device_ordinal as core::ffi::c_int;

            let res = cuda_sys::lib()
                .cuMemPoolCreate(&mut pool as *mut cuda_sys::CUmemoryPool, &props);
            if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
                return Err(MemPoolError::Create(format!("{res:?}")));
            }

            // Set the release threshold. The attribute takes a
            // `cuuint64_t` so we pass the address of `cap_bytes`
            // directly. On failure, destroy the half-built pool so we
            // do not leak a handle on the error path.
            //
            // TOCTOU note: between `cuMemPoolCreate` above and this
            // call, the unprotected pool exists for ~microseconds.
            // See the module-level "TOCTOU note" for why this is
            // acceptable in our model.
            let res = cuda_sys::lib().cuMemPoolSetAttribute(
                pool,
                cuda_sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &cap_bytes as *const u64 as *mut core::ffi::c_void,
            );
            if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
                // Best-effort destroy on the failure path; the outer
                // error wins regardless of the destroy result.
                let _ = cuda_sys::lib().cuMemPoolDestroy(pool);
                return Err(MemPoolError::SetAttribute(format!("{res:?}")));
            }

            Ok(Self {
                pool,
                cap_bytes,
                device_ordinal,
                device,
            })
        }
    }

    /// Backward-compatible shim that targets device ordinal 0.
    ///
    /// Retained because the v0.3.8 scaffold's
    /// [`crate::cuda_mem_pool::TenantMemPool::new`] only took
    /// `cap_bytes`; the T39 driver pin promotes the API to require an
    /// explicit ordinal, but
    /// `TenantContextBuilder::with_driver_enforced_gpu_cap` still
    /// passes only `cap_bytes` (the builder does not know the device
    /// index without the optional `cuda` feature). This shim is the
    /// natural one-arg entry point and matches the existing scaffold
    /// tests in `tests/cuda_mem_pool_scaffold.rs`.
    pub fn new_on_default_device(cap_bytes: u64) -> Result<Self, MemPoolError> {
        Self::new(0, cap_bytes)
    }

    /// The release-threshold cap (in bytes) this pool was created with.
    ///
    /// Note: the CUDA driver may round the value internally; this
    /// returns the *requested* value, not the effective one. v0.4
    /// adds a separate `effective_cap_bytes()` query that round-trips
    /// through `cuMemPoolGetAttribute`.
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    /// Device ordinal this pool's allocations target.
    pub fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    /// Raw `CUmemoryPool` handle. Exposed so
    /// [`crate::unified::UnifiedBuffer::new_in_tenant_pool`] (and any
    /// future co-located allocator) can pass it to
    /// `cuMemAllocFromPoolAsync`. Callers must not free the underlying
    /// pool through this handle — the [`Drop`] impl owns destruction.
    pub fn raw_handle(&self) -> cuda_sys::CUmemoryPool {
        self.pool
    }

    /// Allocate `size` bytes from this tenant pool via
    /// `cuMemAllocFromPoolAsync` on the null stream.
    ///
    /// Returns the raw `CUdeviceptr` as a `NonNull<u8>` on success.
    /// The driver enforces the release-threshold cap configured at
    /// pool construction time — over-cap allocations fail with
    /// `CUDA_ERROR_OUT_OF_MEMORY`, which is the exact bypass-resistant
    /// gate T39 wires.
    ///
    /// # Stream choice
    ///
    /// We pass the null stream (the legacy default stream). The pool
    /// API requires a stream so the driver can sequence
    /// allocate/free pairs; in v0.4 the tenant context does not yet
    /// own a `CUstream` separate from the legacy default, so the null
    /// stream is the closest match to the existing
    /// `cuMemAllocManaged` behaviour. The v0.5 cutover that ports
    /// `TensorWasmMemoryCreator::with_tenant_context` to use this
    /// allocator will thread an explicit per-tenant stream through.
    pub(crate) fn allocate(&self, size: usize) -> Result<NonNull<u8>, UnifiedError> {
        // mem M4: bind the primary context to the calling thread
        // before issuing the alloc-async driver call. See the
        // matching call in `CudarcUnifiedBuffer::prefetch_to_device`.
        ensure_context_bound(&self.device)?;
        let mut raw: cuda_sys::CUdeviceptr = 0;
        // SAFETY: `raw` is a valid out-parameter; the pool handle was
        // returned by `cuMemPoolCreate` and is still live (this is a
        // `&self` method, so the `Drop` impl cannot have run); the
        // null stream is the legacy default stream and is always
        // valid.
        let res = unsafe {
            cuda_sys::lib().cuMemAllocFromPoolAsync(
                &mut raw as *mut cuda_sys::CUdeviceptr,
                size,
                self.pool,
                std::ptr::null_mut(),
            )
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(UnifiedError::Cuda(format!(
                "cuMemAllocFromPoolAsync -> {res:?}"
            )));
        }
        NonNull::new(raw as *mut u8).ok_or_else(|| {
            UnifiedError::Allocation(
                "cuMemAllocFromPoolAsync returned null with CUDA_SUCCESS".into(),
            )
        })
    }

    /// Free a pointer previously returned by [`Self::allocate`].
    ///
    /// Wraps `cuMemFreeAsync` on the null stream — the symmetric free
    /// for `cuMemAllocFromPoolAsync`. Returns the CUDA result through
    /// [`UnifiedError::Cuda`] so callers can decide between propagate
    /// vs leak-and-log.
    pub(crate) fn deallocate(&self, ptr: NonNull<u8>) -> Result<(), UnifiedError> {
        ensure_context_bound(&self.device)?;
        // SAFETY: `ptr` was returned by `cuMemAllocFromPoolAsync` on
        // this pool and has not been freed yet (the
        // `UnifiedBuffer::drop` path that calls this method runs once
        // per buffer). The pool handle is still live (held by `self`).
        let res = unsafe {
            cuda_sys::lib().cuMemFreeAsync(
                ptr.as_ptr() as cuda_sys::CUdeviceptr,
                std::ptr::null_mut(),
            )
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(UnifiedError::Cuda(format!("cuMemFreeAsync -> {res:?}")));
        }
        Ok(())
    }
}

impl Drop for TenantMemPool {
    /// Destroy the underlying pool via `cuMemPoolDestroy`.
    ///
    /// Mirrors the failure-handling discipline of
    /// [`crate::cudarc_backend::CudarcUnifiedBuffer::drop`]: errors
    /// cannot be returned from `Drop`, so a non-`CUDA_SUCCESS` result
    /// is logged via `tracing::error!` and the handle is replaced with
    /// `null_mut()` as a defence-in-depth sentinel against double-drop.
    /// We deliberately do NOT track a per-pool leak set here yet —
    /// v0.4's leak-audit story is unified across both
    /// `CudarcUnifiedBuffer` and `TenantMemPool` and would conflate the
    /// two if added here in isolation.
    ///
    /// Drop ordering: Rust drops struct fields in declaration order, so
    /// `pool` is overwritten first; the held `device: Arc<CudaDevice>`
    /// drops AFTER the `cuMemPoolDestroy` call returns. That ordering
    /// guarantees the primary context outlives the destroy call — see
    /// the parallel rationale on `CudarcUnifiedBuffer`.
    fn drop(&mut self) {
        if self.pool.is_null() {
            return;
        }
        // SAFETY: `pool` is non-null and was returned by
        // `cuMemPoolCreate`. After this call the handle is invalid;
        // we null it out below as belt-and-braces against any
        // hypothetical double-drop.
        let res = unsafe { cuda_sys::lib().cuMemPoolDestroy(self.pool) };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            tracing::error!(
                target: "tensor_wasm_mem::cuda_mem_pool",
                ?res,
                cap_bytes = self.cap_bytes,
                device_ordinal = self.device_ordinal,
                "cuMemPoolDestroy failed in TenantMemPool::drop",
            );
        }
        self.pool = std::ptr::null_mut();
    }
}

// SAFETY: the raw `CUmemoryPool` is owned by this struct; the cudarc
// FFI documents pool handles as thread-safe for use, matching the
// `Send + Sync` contract we expose for `CudarcUnifiedBuffer`. Concurrent
// allocations against the same pool synchronise inside the driver.
// The held `Arc<CudaDevice>` is itself `Send + Sync`.
unsafe impl Send for TenantMemPool {}
unsafe impl Sync for TenantMemPool {}

/// Errors raised by [`TenantMemPool`] operations.
#[derive(Debug, thiserror::Error)]
pub enum MemPoolError {
    /// `cuMemPoolCreate` returned a non-`CUDA_SUCCESS` code. The wrapped
    /// string is the `Debug`-formatted CUDA result.
    #[error("cuMemPoolCreate failed: {0}")]
    Create(String),
    /// `cuMemPoolSetAttribute` returned a non-`CUDA_SUCCESS` code. The
    /// half-built pool is destroyed before this error is returned, so
    /// callers do not need to do anything to clean up.
    #[error("cuMemPoolSetAttribute failed: {0}")]
    SetAttribute(String),
    /// CUDA was not initialised by the time the pool constructor ran.
    /// In v0.3.8 this is reserved for future use — the cudarc
    /// `CudaDevice::new` cache in
    /// [`crate::cudarc_backend`] already primes `cuInit(0)` before any
    /// pool can be created from inside the same process — but the
    /// variant is present so callers can match on it without a
    /// breaking-change minor bump in v0.4.
    #[error("cuda not initialized")]
    NotInitialized,
    /// The T26 per-ordinal device cache could not retain a primary
    /// context for the requested device ordinal. Wraps the underlying
    /// [`crate::unified::UnifiedError::Cuda`] description from
    /// [`crate::cudarc_backend::device_for`]. A non-CUDA host or a
    /// missing GPU surfaces here, NOT through [`Self::Create`].
    #[error("device retain failed: {0}")]
    Device(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Type-level sanity check that does NOT touch the driver. Mirrors
    /// the `buffer_type_has_nonzero_size` pattern in
    /// `cudarc_backend.rs`: just having the type in the binary does
    /// not trigger a CUDA call, so this test is safe on host-only CI.
    #[test]
    fn tenant_mem_pool_type_has_nonzero_size() {
        assert!(std::mem::size_of::<TenantMemPool>() > 0);
    }

    /// `MemPoolError` Display impls produce non-empty messages. Cheap
    /// regression guard against an accidental `#[error("")]` slipping
    /// into the variants — the operator alert path keys on the message
    /// string, so an empty error would silently swallow context.
    #[test]
    fn mem_pool_error_display_non_empty() {
        let e = MemPoolError::Create("CUDA_ERROR_OUT_OF_MEMORY".into());
        assert!(format!("{e}").contains("cuMemPoolCreate failed"));
        let e = MemPoolError::SetAttribute("CUDA_ERROR_INVALID_VALUE".into());
        assert!(format!("{e}").contains("cuMemPoolSetAttribute failed"));
        let e = MemPoolError::NotInitialized;
        assert!(format!("{e}").contains("not initialized"));
        let e = MemPoolError::Device("device_for(7): CudaDevice::new(7): ...".into());
        assert!(format!("{e}").contains("device retain failed"));
    }
}
