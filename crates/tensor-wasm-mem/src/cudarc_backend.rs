// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `cudarc`-backed parallel implementation of the unified-memory path.
//!
//! This module is the v0.2 *spike* for the cust → cudarc migration tracked in
//! [`docs/CUDARC-SPIKE.md`](../../../../docs/CUDARC-SPIKE.md) and the
//! [risk register](../../../../docs/RISKS.md). It is gated behind the
//! `cudarc-backend` feature so that a developer can opt in with
//! `cargo build --features cudarc-backend` and exercise the cudarc code path,
//! while the default v0.2 builds keep using `cust`.
//!
//! Scope of the spike: a faithful mirror of [`crate::unified::UnifiedBuffer`]'s
//! public API ([`CudarcUnifiedBuffer`]), plus a thin
//! [`apply_advice`] wrapper that mirrors [`crate::advise::apply`]. Both wrap
//! the CUDA Driver API entry points (`cuMemAllocManaged`, `cuMemFree_v2`,
//! `cuMemAdvise`, `cuMemPrefetchAsync`) exposed by `cudarc::driver::sys`,
//! because cudarc 0.13's safe surface does not yet wrap `cuMemAllocManaged`
//! directly — its [`cudarc::driver::CudaDevice::alloc_zeros`] returns
//! *device-only* memory, not managed memory. See `CUDARC-SPIKE.md` for the
//! full API mapping table and the list of gaps.
//!
//! # Layering vs the cust path
//!
//! The cust path lives in [`crate::unified`]. The cudarc path is intentionally
//! a *separate* module — it does not replace or wrap the cust path. Call sites
//! continue to use [`crate::unified::UnifiedBuffer`] by default; tests and
//! benchmarks that want to exercise cudarc construct a
//! [`CudarcUnifiedBuffer`] directly. This keeps the two backends independent
//! during the spike so that a regression in one cannot mask a regression in
//! the other, and so we can compare them on the same host without recompiling.
//!
//! # Driver context
//!
//! cudarc requires an explicit `CudaDevice` (which initialises the driver and
//! a primary context). To keep the spike narrow, [`CudarcUnifiedBuffer::new`]
//! creates a device on first call and caches it in a `OnceLock`. A production
//! cutover (v0.2 or v0.3, see the spike doc) will route this through the
//! existing per-tenant context plumbing in `tensor-wasm-tenant`.

use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaDevice;

use crate::advise::Advice;
use crate::unified::{DeviceId, UnifiedError};

/// Process-wide counter of `cuMemFree_v2` failures observed in
/// [`CudarcUnifiedBuffer::drop`]. A non-zero value indicates leaked managed
/// allocations — the driver refused to release them. Operators can poll this
/// via [`cudarc_free_failures`] to alert on leak rate.
static CUDARC_FREE_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Number of `cuMemFree_v2` failures observed since process start.
///
/// Each increment corresponds to one leaked managed-memory allocation: the
/// driver returned non-`CUDA_SUCCESS` from [`Drop`] so the bytes are still
/// resident in the process's address space. The counter is a `u64` so it
/// will not realistically wrap; operators can subtract two snapshots to get
/// a leak rate over any window. The monotonically-increasing failure log is
/// also emitted at `tracing::warn!` for forensic correlation.
pub fn cudarc_free_failures() -> u64 {
    CUDARC_FREE_FAILURES.load(Ordering::Relaxed)
}

/// Process-wide cached `CudaDevice` for device 0.
///
/// cudarc holds the primary context inside the [`CudaDevice`]; keeping a
/// single `Arc<CudaDevice>` alive across allocations avoids re-running
/// `cuDevicePrimaryCtxRetain` on every [`CudarcUnifiedBuffer::new`] call.
/// Real call sites will route through the tenant-aware context cache once
/// the spike graduates — see `docs/CUDARC-SPIKE.md`.
static DEFAULT_DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();

/// Lazily construct (or fetch) the cached cudarc device for ordinal `ordinal`.
///
/// The spike caches device 0 only; any other ordinal triggers a fresh
/// `CudaDevice::new` call and is not cached. A future revision will replace
/// this with a per-ordinal cache when more than one GPU enters the test
/// matrix.
fn device_for(ordinal: u32) -> Result<Arc<CudaDevice>, UnifiedError> {
    if ordinal == 0 {
        // `get_or_try_init` only invokes the initialiser when this thread
        // actually wins the install race, so the new `Arc<CudaDevice>` is
        // never dropped *after* a winner has been installed. Dropping a
        // losing `Arc<CudaDevice>` would call `cuDevicePrimaryCtxRelease`
        // on the still-held primary context — invalidating every managed
        // pointer in the process. See the audit note attached to this
        // function's history.
        let installed = DEFAULT_DEVICE
            .get_or_try_init(|| {
                CudaDevice::new(0)
                    .map_err(|e| UnifiedError::Cuda(format!("CudaDevice::new(0): {e:?}")))
            })?;
        Ok(installed.clone())
    } else {
        CudaDevice::new(ordinal as usize)
            .map_err(|e| UnifiedError::Cuda(format!("CudaDevice::new({ordinal}): {e:?}")))
    }
}

/// A contiguous CUDA Unified Memory region allocated via `cudarc`.
///
/// Mirrors [`crate::unified::UnifiedBuffer`]'s public surface. The pointer
/// returned by [`Self::as_ptr`] is addressable from both host and device on a
/// CUDA-capable host; the runtime page-migrates between them on demand.
///
/// # Safety invariants
///
/// - `ptr` is non-null and points to `size` valid bytes obtained from
///   `cuMemAllocManaged`.
/// - `device` is kept alive for the lifetime of the allocation so the primary
///   context is not torn down before [`Drop`] runs `cuMemFree_v2`.
pub struct CudarcUnifiedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    device_id: DeviceId,
    /// Held purely to keep the primary context alive. Dropping the last
    /// reference releases the context, which would invalidate every
    /// outstanding managed pointer in the process.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
}

// SAFETY: the inner pointer is owned by this struct and not shared without
// explicit synchronisation. Same contract as `Vec<u8>` once you have a
// `&mut [u8]`. The `Arc<CudaDevice>` is itself `Send + Sync`.
unsafe impl Send for CudarcUnifiedBuffer {}
unsafe impl Sync for CudarcUnifiedBuffer {}

impl CudarcUnifiedBuffer {
    /// Allocate `size` bytes of CUDA Unified Memory on the default device.
    pub fn new(size: usize) -> Result<Self, UnifiedError> {
        Self::new_on(size, DeviceId::default())
    }

    /// Allocate `size` bytes of CUDA Unified Memory on the named device.
    ///
    /// Wraps `cuMemAllocManaged` with the `CU_MEM_ATTACH_GLOBAL` flag, which
    /// matches the semantics of `cust::memory::UnifiedBuffer::new` (visible to
    /// every stream on every device in the current context).
    pub fn new_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        let device = device_for(device_id.0)?;
        let mut raw: cuda_sys::CUdeviceptr = 0;
        // CUDA documents the `flags` argument to `cuMemAllocManaged` as one of
        // `CU_MEM_ATTACH_GLOBAL = 1` or `CU_MEM_ATTACH_HOST = 2`. cudarc 0.13's
        // bindgen output places this enum at `CUmemAttach_flags_enum` (the
        // trailing `_enum` is a bindgen convention for anonymous C enums),
        // which has drifted between cudarc minor versions. Inlining the
        // documented numeric value matches the `cust::memory::UnifiedBuffer`
        // default ("attached globally — every stream on every device") and
        // sidesteps the enum-path drift entirely.
        const CU_MEM_ATTACH_GLOBAL: u32 = 1;
        // Drift guard: if cudarc renumbers the enum in a future minor bump
        // (the path itself drifts between `CUmemAttach_flags_enum` and
        // `CUmemAttach_flags`, see the comment above), the build fails here
        // rather than at runtime with a confusing `CUDA_ERROR_INVALID_VALUE`.
        // We dodge `static_assertions` to keep the dep tree tight — a plain
        // `const _: () = assert!(...)` works on stable since 1.57.
        const _: () = assert!(
            CU_MEM_ATTACH_GLOBAL
                == cuda_sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
            "cudarc renumbered CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL; \
             update the inlined constant in cudarc_backend.rs",
        );
        // SAFETY: `raw` is a valid out-parameter; `size > 0`; the device above
        // ensures the primary context is current on this thread.
        // cudarc 0.13.x exposes CUDA driver functions as methods on a Lib
        // struct, accessed via cudarc::driver::sys::lib(). Free-function
        // imports like cust uses are not available; CudaDevice::new above
        // primed the OnceLock so this lib() call cannot panic.
        let res = unsafe {
            cuda_sys::lib().cuMemAllocManaged(
                &mut raw as *mut cuda_sys::CUdeviceptr,
                size,
                CU_MEM_ATTACH_GLOBAL,
            )
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(UnifiedError::Cuda(format!("cuMemAllocManaged -> {res:?}")));
        }
        let ptr = NonNull::new(raw as *mut u8).ok_or_else(|| {
            UnifiedError::Allocation("cuMemAllocManaged returned null with CUDA_SUCCESS".into())
        })?;
        Ok(Self {
            ptr,
            size,
            device_id,
            device,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length. Always false for a successfully constructed buffer.
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
    /// On CUDA-capable hosts the managed memory may be migrated to the device
    /// at any moment; reading from the host side after a device kernel has
    /// written into the region without synchronising is undefined behaviour.
    /// Call sites must serialise host/device access with a stream sync or an
    /// event, exactly as they do for the cust path.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null and points to `size` valid bytes by the type invariant.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Borrow as a mutable byte slice.
    ///
    /// # Safety
    ///
    /// Same caveat as [`Self::as_slice`].
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` proves uniqueness; ptr/size by the type invariant.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Which device this buffer is anchored to.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Suggest the runtime migrate the buffer to the device.
    ///
    /// Wraps `cuMemPrefetchAsync` on the device's null stream. Errors flow
    /// through as [`UnifiedError::Cuda`].
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        // SAFETY: ptr/size are derived from a valid live allocation; passing
        // the null stream (handle 0) requests prefetch on the default stream.
        let res = unsafe {
            cuda_sys::lib().cuMemPrefetchAsync(
                self.ptr.as_ptr() as cuda_sys::CUdeviceptr,
                self.size,
                self.device_id.0 as i32,
                std::ptr::null_mut(),
            )
        };
        if res == cuda_sys::cudaError_enum::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(UnifiedError::Cuda(format!(
                "cuMemPrefetchAsync(device) -> {res:?}"
            )))
        }
    }

    /// Suggest the runtime migrate the buffer back to host memory.
    ///
    /// Wraps `cuMemPrefetchAsync` with `CU_DEVICE_CPU` as the destination.
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        // `CU_DEVICE_CPU` is the sentinel for "host" in cuMemPrefetchAsync.
        // cudarc 0.13 does not export it as a named constant; CUDA defines it
        // as `-1`.
        const CU_DEVICE_CPU: i32 = -1;
        // SAFETY: see `prefetch_to_device`.
        let res = unsafe {
            cuda_sys::lib().cuMemPrefetchAsync(
                self.ptr.as_ptr() as cuda_sys::CUdeviceptr,
                self.size,
                CU_DEVICE_CPU,
                std::ptr::null_mut(),
            )
        };
        if res == cuda_sys::cudaError_enum::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(UnifiedError::Cuda(format!(
                "cuMemPrefetchAsync(host) -> {res:?}"
            )))
        }
    }
}

impl fmt::Debug for CudarcUnifiedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudarcUnifiedBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Drop for CudarcUnifiedBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` was returned by `cuMemAllocManaged` and has not been
        // freed yet; the cached `Arc<CudaDevice>` we hold ensures the primary
        // context is still alive when we call `cuMemFree_v2`.
        let res = unsafe { cuda_sys::lib().cuMemFree_v2(self.ptr.as_ptr() as cuda_sys::CUdeviceptr) };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            // Drop cannot fail; surface via a trace event so post-mortem
            // tooling can spot leaks without unwinding, and bump a
            // process-wide counter so the operator can monitor leak rate
            // without scraping logs. The counter is exposed via the
            // module-level `cudarc_free_failures()` helper.
            let failures = CUDARC_FREE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                target: "tensor_wasm_mem::cudarc_backend",
                ?res,
                ptr = ?self.ptr.as_ptr(),
                size = self.size,
                total_failures = failures,
                "cuMemFree_v2 failed in CudarcUnifiedBuffer::drop",
            );
        }
    }
}

/// Apply an [`Advice`] hint via `cuMemAdvise` against a [`CudarcUnifiedBuffer`].
///
/// Mirror of [`crate::advise::apply`] for the cudarc path. Returns
/// [`UnifiedError::Cuda`] on failure.
pub fn apply_advice(buffer: &CudarcUnifiedBuffer, advice: Advice) -> Result<(), UnifiedError> {
    let ptr = buffer.as_ptr() as cuda_sys::CUdeviceptr;
    let size = buffer.len();
    let (advice_kind, device) = match advice {
        Advice::ReadMostly => (
            cuda_sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_READ_MOSTLY,
            0i32,
        ),
        Advice::PreferredLocation(d) => (
            cuda_sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            d.0 as i32,
        ),
        Advice::AccessedBy(d) => (
            cuda_sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY,
            d.0 as i32,
        ),
        Advice::UnsetPreferredLocation => (
            cuda_sys::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
            0i32,
        ),
        Advice::UnsetAccessedBy(d) => (
            cuda_sys::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_ACCESSED_BY,
            d.0 as i32,
        ),
    };
    // SAFETY: ptr/size are derived from a valid live CudarcUnifiedBuffer.
    let res = unsafe { cuda_sys::lib().cuMemAdvise(ptr, size, advice_kind, device) };
    if res == cuda_sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(UnifiedError::Cuda(format!("cuMemAdvise -> {res:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: the buffer type carries a non-trivial layout. Confirms
    /// the cudarc-backend code compiles when the feature is on without
    /// requiring CUDA hardware (the test runs even on host-only CI as long as
    /// the `cudarc-backend` feature is enabled; cudarc 0.13's `driver` feature
    /// dlopens `libcuda` lazily, so just having the type in the binary does
    /// not trigger a driver call).
    #[test]
    fn buffer_type_has_nonzero_size() {
        assert!(std::mem::size_of::<CudarcUnifiedBuffer>() > 0);
    }

    /// Confirms `apply_advice` is callable as a free function on the cudarc
    /// path. We do not invoke it (would require a live buffer) — this is a
    /// type-level check that the public symbol is wired.
    #[test]
    fn apply_advice_is_exported() {
        let _f: fn(&CudarcUnifiedBuffer, Advice) -> Result<(), UnifiedError> = apply_advice;
    }

    /// Actually allocate + drop a tiny buffer. Requires a CUDA driver and at
    /// least one GPU; marked `#[ignore]` so host-only CI does not try to
    /// `dlopen` `libcuda.so` / `nvcuda.dll`.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn allocate_and_drop_small_buffer() {
        let mut b = CudarcUnifiedBuffer::new(64).expect("alloc");
        assert_eq!(b.len(), 64);
        b.as_mut_slice().copy_from_slice(&[0xAB; 64]);
        assert!(b.as_slice().iter().all(|&v| v == 0xAB));
    }
}
