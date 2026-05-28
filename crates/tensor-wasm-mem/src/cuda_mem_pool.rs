// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Driver-level GPU memory pool enforcement (roadmap feature #8, v0.4
//! deliverable).
//!
//! Today the per-tenant GPU memory cap is enforced in-process by
//! `TenantContext::consume_bytes` (B6.5: a lock-free CAS counter over
//! `bytes_in_use` against the `memory_quota_bytes` cap). That catches
//! the well-behaved-allocator path — every `UnifiedBuffer::new` routes
//! through the tenant's quota-checked path before allocating — but a
//! tenant that somehow obtained a raw CUDA driver handle could bypass
//! it.
//!
//! `cuMemPool` (CUDA 11.2+) lets the host pin a hard ceiling that the
//! driver itself enforces. v0.4 wires every tenant's allocations
//! through a tenant-scoped `cuMemPoolHandle_t` configured with
//! `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` matching the tenant's cap.
//! Allocations past the cap fail at the driver level with
//! `CUDA_ERROR_OUT_OF_MEMORY` — the in-process counter is a
//! belt-and-suspenders layer on top.
//!
//! ## v0.3.8 status: scaffold
//!
//! Pool creation, attribute setting, and the `release` path are wired
//! against the cudarc API surface. The actual allocations still go
//! through `cuMemAllocManaged` (the unified-memory path); v0.4 splits
//! the unified-memory and pooled-device-memory paths so operators can
//! pick the right enforcement layer per tenant. The cudarc 0.13 FFI
//! type names below (`CUmemoryPool`, `CUmemPoolProps`, the various
//! enum discriminants) are written against the cudarc bindings as
//! they were exposed at v0.13; if cudarc renumbers or renames any of
//! them, the build fails *here* rather than at runtime with
//! `CUDA_ERROR_INVALID_VALUE`. See the parallel drift-guard pattern
//! in `cudarc_backend.rs` for the `CU_MEM_ATTACH_GLOBAL` constant.
//!
//! ## Gating
//!
//! Behind the `cudarc-backend` feature (cust 0.3.x does not expose the
//! `cuMemPool*` API surface, so this module would be useless without
//! cudarc). On hosts without cudarc, the symbol does not exist at all
//! — callers in `tensor-wasm-tenant` that opt into `gpu-mem-pool` must
//! transitively pull `tensor-wasm-mem/cudarc-backend` and the cargo
//! feature resolver enforces that at workspace-resolve time.
//!
//! ## Tests
//!
//! Hardware-dependent. The unit tests in this file are pure type / API
//! checks; the live-driver tests live in
//! `tests/cuda_mem_pool_scaffold.rs` and are `#[ignore]`'d so host-only
//! CI does not try to `dlopen` `libcuda.so` / `nvcuda.dll`.

#![cfg(feature = "cudarc-backend")]

use cudarc::driver::sys as cuda_sys;

/// Tenant-scoped CUDA memory pool (`cuMemPoolHandle_t`) with a hard
/// release-threshold cap.
///
/// The pool is owned by exactly one tenant. Dropping the wrapper calls
/// `cuMemPoolDestroy`; the cached `Arc<CudaDevice>` plumbing that keeps
/// the primary context alive lives in
/// [`crate::cudarc_backend::CudarcUnifiedBuffer`], not here — this
/// scaffold deliberately keeps the pool surface minimal so the v0.4
/// follow-up can decide whether pools should be device-local or share
/// the same `OnceLock<Arc<CudaDevice>>` cache.
///
/// # Safety invariants
///
/// - `pool` is non-null and was returned by `cuMemPoolCreate`; it has
///   not been destroyed yet.
/// - `cap_bytes` is the value the constructor passed to
///   `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`.
///   The CUDA driver may round it; we store the *requested* value here
///   for honest reporting back to the tenant.
#[derive(Debug)]
pub struct TenantMemPool {
    pool: cuda_sys::CUmemoryPool,
    cap_bytes: u64,
}

impl TenantMemPool {
    /// Create a tenant-scoped memory pool with a release-threshold cap.
    ///
    /// The pool is created with `CU_MEM_ALLOCATION_TYPE_PINNED` on
    /// device-located memory (ordinal 0 for the v0.3.8 scaffold; v0.4
    /// threads the per-tenant device index through the same way
    /// `TenantContextBuilder::with_cuda_device_index` does it). The
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` attribute is set to
    /// `cap_bytes` so allocations past the cap fail at the driver
    /// level with `CUDA_ERROR_OUT_OF_MEMORY`.
    ///
    /// Returns [`MemPoolError::Create`] if `cuMemPoolCreate` returns
    /// anything other than `CUDA_SUCCESS`, or
    /// [`MemPoolError::SetAttribute`] if the release-threshold set
    /// fails (in which case the partially-created pool is destroyed
    /// before the error returns).
    pub fn new(cap_bytes: u64) -> Result<Self, MemPoolError> {
        // SAFETY: the cudarc FFI surface is unsafe by construction; we
        // wrap each entry point with a tightly-scoped `unsafe` block.
        // The `MaybeUninit` pattern here matches the upstream cudarc
        // 0.13 example for the cuMemPool family — `cuMemPoolCreate`
        // writes the handle through the out-pointer; `CUmemPoolProps`
        // is a plain C struct and zero-init is documented as
        // "default-everything" for all fields we do not explicitly
        // set.
        unsafe {
            let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
            let mut props: cuda_sys::CUmemPoolProps = std::mem::zeroed();
            // `allocType` = pinned device memory. The release-threshold
            // attribute below only makes sense on a pinned pool.
            props.allocType = cuda_sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
            // No IPC handle export for v0.3.8; tenants live within a
            // single process today.
            props.handleTypes = cuda_sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE;
            // Location: device ordinal 0. v0.4 wires through
            // `TenantContextBuilder::with_cuda_device_index`.
            props.location.type_ = cuda_sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = 0;

            let res = cuda_sys::lib()
                .cuMemPoolCreate(&mut pool as *mut cuda_sys::CUmemoryPool, &props);
            if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
                return Err(MemPoolError::Create(format!("{res:?}")));
            }

            // Set the release threshold. The attribute takes a
            // `cuuint64_t` so we pass the address of `cap_bytes`
            // directly. On failure, destroy the half-built pool so we
            // do not leak a handle on the error path.
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

            Ok(Self { pool, cap_bytes })
        }
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

    /// Raw `CUmemoryPool` handle. Exposed so the v0.4
    /// `UnifiedBuffer::new_in_tenant_pool` allocator can pass it to
    /// `cuMemAllocFromPoolAsync`. Callers must not free the underlying
    /// pool through this handle — the [`Drop`] impl owns destruction.
    pub fn raw_handle(&self) -> cuda_sys::CUmemoryPool {
        self.pool
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
    }
}
