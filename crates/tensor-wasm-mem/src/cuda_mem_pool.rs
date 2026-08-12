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
//! `cuMemPool` (CUDA 11.2+) gives each tenant its own
//! `cuMemPoolHandle_t` so allocations are pool-scoped and freed memory is
//! retained per-tenant. T39 routes every tenant allocation through a
//! tenant-scoped pool via [`UnifiedBuffer::new_in_tenant_pool`].
//!
//! The per-tenant *cap*, however, is enforced HOST-SIDE in
//! [`TenantMemPool::allocate`] (a CAS counter over `live_bytes` against
//! `cap_bytes`), NOT by the driver. `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` —
//! set at construction — is only a memory-*retention* hint (how much freed
//! memory the pool caches before returning it to the OS); it is NOT an
//! allocation ceiling, and CUDA memory pools expose no hard max-size
//! attribute. The hardware run in `docs/GPU-VALIDATION-2026-05-30.md`
//! (BUG-1) confirmed a 128 MiB request against a 64 MiB-"capped" pool
//! succeeded when the cap relied on RELEASE_THRESHOLD alone; the host-side
//! reservation in `allocate` closes that gap. Over-cap requests now fail
//! before `cuMemAllocFromPoolAsync` with a `CUDA_ERROR_OUT_OF_MEMORY`-shaped
//! `UnifiedError::Cuda`, and the in-process
//! `TenantContext::gpu_bytes_in_use` counter remains a second line of
//! defence on top.
//!
//! ## Status (T39): host-side cap enforced; pool wiring LANDED
//!
//! Pool creation, `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` set (retention hint),
//! the host-side cap reservation in [`TenantMemPool::allocate`], drop, and a
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaDevice;
use tensor_wasm_core::mem_pool::DriverMemPool;

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
    /// The *requested* release-threshold cap (in bytes). Interior-mutable
    /// so [`DriverMemPool::set_release_threshold`] can re-pin the
    /// threshold post-construction (the tenant driver-cap path) and have
    /// [`Self::cap_bytes`] / [`DriverMemPool::release_threshold`] report
    /// the new value. Plain relaxed atomics suffice: this is honest
    /// reporting, not a synchronisation point — the authoritative cap
    /// lives in the driver after the `cuMemPoolSetAttribute` call.
    cap_bytes: AtomicU64,
    /// Live (allocated-but-not-yet-freed) bytes drawn from this pool.
    ///
    /// Fix #1 (T39 real enforcement): the per-tenant cap is enforced
    /// **host-side** here, not by the driver. `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`
    /// is a memory-*retention* hint (how much freed memory the pool caches
    /// before returning it to the OS) — it is NOT an allocation ceiling, and
    /// CUDA memory pools expose no hard max-size attribute. So
    /// [`Self::allocate`] reserves against `cap_bytes` using this counter
    /// before calling `cuMemAllocFromPoolAsync`, and [`Self::release_bytes`]
    /// gives the reservation back on free. Relaxed-but-CAS'd just like the
    /// in-process `TenantContext::gpu_bytes_in_use` counter it backstops.
    live_bytes: AtomicU64,
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
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` attribute is set to `cap_bytes`
    /// as a retention hint only (it bounds cached-freed memory, NOT
    /// allocation). The per-tenant allocation cap is enforced host-side in
    /// [`Self::allocate`] — allocations past `cap_bytes` fail there with a
    /// `CUDA_ERROR_OUT_OF_MEMORY`-shaped error before the driver call. See
    /// the module docs and `docs/GPU-VALIDATION-2026-05-30.md` BUG-1.
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
            props.allocType = cuda_sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
            // No IPC handle export: tenants live within a single
            // process today; cross-process shareable pools are a v0.5
            // follow-up.
            props.handleTypes = cuda_sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE;
            // Location: device-local memory on the requested ordinal.
            props.location.type_ = cuda_sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = device_ordinal as core::ffi::c_int;

            let res =
                cuda_sys::lib().cuMemPoolCreate(&mut pool as *mut cuda_sys::CUmemoryPool, &props);
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
                cap_bytes: AtomicU64::new(cap_bytes),
                live_bytes: AtomicU64::new(0),
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
    /// returns the *requested* value, not the effective one. See
    /// [`Self::effective_cap_bytes`] for the driver-reported value that
    /// round-trips through `cuMemPoolGetAttribute`.
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes.load(Ordering::Relaxed)
    }

    /// The *effective* release-threshold the driver actually holds, read
    /// back via `cuMemPoolGetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)`
    /// (mem finding #5).
    ///
    /// [`Self::cap_bytes`] returns the value this crate *requested*; the
    /// driver may round or clamp it internally, so an operator validating a
    /// per-tenant cap against the hardware wants the authoritative figure.
    /// This issues the get-attribute query against the live pool handle.
    ///
    /// # Thread safety (mem M4)
    ///
    /// Binds the pool's primary context to the calling thread before the
    /// query — `cuMemPoolGetAttribute` is context-sensitive, exactly like
    /// the `cuMemPoolSetAttribute` calls in [`Self::new`] /
    /// [`DriverMemPool::set_release_threshold`].
    ///
    /// Returns [`UnifiedError::Cuda`] (matching [`Self::allocate`]) if the
    /// context bind or the driver query fails.
    pub fn effective_cap_bytes(&self) -> Result<u64, UnifiedError> {
        // mem M4: bind the primary context before the context-sensitive query.
        ensure_context_bound(&self.device)?;
        let mut value: u64 = 0;
        // SAFETY: `self.pool` is non-null and live (this is a `&self` method,
        // so `Drop` cannot have run); `value` is a valid `u64` out-parameter
        // and `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is documented as a
        // `cuuint64_t`-typed attribute, so the 8-byte write the driver makes
        // through the pointer is in-bounds.
        let res = unsafe {
            cuda_sys::lib().cuMemPoolGetAttribute(
                self.pool,
                cuda_sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut value as *mut u64 as *mut core::ffi::c_void,
            )
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(UnifiedError::Cuda(format!(
                "cuMemPoolGetAttribute(RELEASE_THRESHOLD) -> {res:?}"
            )));
        }
        Ok(value)
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
    /// The per-tenant cap is enforced HOST-SIDE here (see [`Self::live_bytes`]
    /// / the reservation loop below): an allocation that would push
    /// `live_bytes` past `cap_bytes` is refused before the driver call with a
    /// `CUDA_ERROR_OUT_OF_MEMORY`-shaped [`UnifiedError::Cuda`]. The driver
    /// does NOT enforce this — `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is a
    /// retention hint, not an allocation ceiling (see the module docs and
    /// `docs/GPU-VALIDATION-2026-05-30.md` BUG-1).
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
        // Fix #1 (T39 real enforcement): reserve against the per-tenant cap
        // BEFORE asking the driver. `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` does
        // not bound allocation size (it is a retention hint), and CUDA memory
        // pools expose no hard max-size attribute — so without this gate a
        // tenant with a raw pool handle could allocate past its cap. The
        // CAS-loop mirrors `TenantContext::consume_gpu_bytes`. The reservation
        // is held until the matching `release_bytes` on free; on a driver
        // failure below we roll it back so a transient CUDA error does not
        // permanently shrink the tenant's headroom.
        let cap = self.cap_bytes.load(Ordering::Acquire);
        let size_u64 = size as u64;
        let mut current = self.live_bytes.load(Ordering::Acquire);
        loop {
            let next = reserve_step(cap, current, size_u64)?;
            match self.live_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }

        // mem M4: bind the primary context to the calling thread
        // before issuing the alloc-async driver call. See the
        // matching call in `CudarcUnifiedBuffer::prefetch_to_device`.
        if let Err(e) = ensure_context_bound(&self.device) {
            self.release_bytes(size); // roll back the reservation
            return Err(e);
        }
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
            self.release_bytes(size); // roll back the reservation
            return Err(UnifiedError::Cuda(format!(
                "cuMemAllocFromPoolAsync -> {res:?}"
            )));
        }
        NonNull::new(raw as *mut u8).ok_or_else(|| {
            self.release_bytes(size); // roll back the reservation
            UnifiedError::Allocation(
                "cuMemAllocFromPoolAsync returned null with CUDA_SUCCESS".into(),
            )
        })
    }

    /// Give back `size` bytes of cap reservation previously taken by
    /// [`Self::allocate`]. Called from `TenantPoolBacking::drop` on the free
    /// path. Saturating on underflow — a bookkeeping mismatch must not wrap the
    /// counter and permanently lock the tenant out of its own pool.
    pub(crate) fn release_bytes(&self, size: usize) {
        let size_u64 = size as u64;
        let _ = self
            .live_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(size_u64))
            });
    }

    /// Live (allocated-but-not-yet-freed) bytes currently drawn from this pool.
    /// Visible for metrics and the host-side cap tests.
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Acquire)
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
            cuda_sys::lib()
                .cuMemFreeAsync(ptr.as_ptr() as cuda_sys::CUdeviceptr, std::ptr::null_mut())
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(UnifiedError::Cuda(format!("cuMemFreeAsync -> {res:?}")));
        }
        Ok(())
    }
}

/// Pure per-tenant cap-reservation decision, factored out of
/// [`TenantMemPool::allocate`] so the ceiling logic is unit-testable without a
/// CUDA driver (the `#[cfg(test)]` cases below run on a no-GPU host whenever
/// the crate is built `--features gpu-mem-pool`).
///
/// Given the pool `cap`, the `current` live byte total, and a requested
/// `size`, returns the new live total to install on success, or an
/// `OUT_OF_MEMORY`-shaped [`UnifiedError::Cuda`] if the request would exceed
/// the cap (or overflow `u64`). The error string is deliberately
/// `CUDA_ERROR_OUT_OF_MEMORY`-shaped so callers see the same failure as a
/// genuine driver OOM — the cap is a hard ceiling from the tenant's view.
fn reserve_step(cap: u64, current: u64, size: u64) -> Result<u64, UnifiedError> {
    let next = current.checked_add(size).ok_or_else(|| {
        UnifiedError::Cuda(format!(
            "CUDA_ERROR_OUT_OF_MEMORY: per-tenant pool reservation overflow \
             (live={current}, requested={size})"
        ))
    })?;
    if next > cap {
        return Err(UnifiedError::Cuda(format!(
            "CUDA_ERROR_OUT_OF_MEMORY: allocation of {size} bytes would exceed the \
             per-tenant GPU memory cap (cap={cap}, live={current}). The cap is \
             enforced host-side; CU_MEMPOOL_ATTR_RELEASE_THRESHOLD is only a \
             retention hint, not an allocation ceiling."
        )));
    }
    Ok(next)
}

impl DriverMemPool for TenantMemPool {
    /// Re-pin the pool's `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` to `bytes`.
    ///
    /// This is the cycle-breaking entry point: `tensor-wasm-tenant`
    /// drives a tenant's GPU cap through here against an
    /// `Arc<dyn DriverMemPool>` without ever naming this concrete type.
    /// On success the recorded [`Self::cap_bytes`] is updated so honest
    /// reporting (and [`Self::release_threshold`]) reflect the new value;
    /// the driver is the authoritative ceiling. Returns
    /// [`MemPoolError::SetAttribute`] if `cuMemPoolSetAttribute` returns
    /// anything other than `CUDA_SUCCESS` — in which case the recorded
    /// value is left unchanged.
    fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError> {
        // mem M4: bind the primary context before the context-sensitive
        // attribute set, mirroring `TenantMemPool::new`.
        ensure_context_bound(&self.device)
            .map_err(|e| MemPoolError::Device(format!("ensure_context_bound: {e:?}")))?;
        // SAFETY: `pool` is non-null and live (this is a `&self` method,
        // so `Drop` cannot have run); `bytes` outlives the call as the
        // attribute payload pointer.
        let res = unsafe {
            cuda_sys::lib().cuMemPoolSetAttribute(
                self.pool,
                cuda_sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &bytes as *const u64 as *mut core::ffi::c_void,
            )
        };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(MemPoolError::SetAttribute(format!("{res:?}")));
        }
        self.cap_bytes.store(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn release_threshold(&self) -> Option<u64> {
        Some(self.cap_bytes.load(Ordering::Relaxed))
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
                cap_bytes = self.cap_bytes.load(Ordering::Relaxed),
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
///
/// Re-exported from [`tensor_wasm_core::mem_pool`], which now owns the
/// type so the backend-agnostic [`DriverMemPool`] trait and this
/// concrete implementor can share one error without `tensor-wasm-tenant`
/// depending on `tensor-wasm-mem` (that edge would close the
/// `mem` <-> `tenant` cycle). The variants are unchanged from when they
/// lived here; existing `MemPoolError::Create` / `::SetAttribute` /
/// `::NotInitialized` / `::Device` match arms keep compiling against the
/// re-export.
pub use tensor_wasm_core::mem_pool::MemPoolError;

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

    /// `TenantMemPool` implements the backend-agnostic
    /// [`DriverMemPool`] trait that `tensor-wasm-tenant` drives the
    /// driver-enforced GPU cap through. A pure type-level check (no
    /// driver call): if the impl is dropped or its signature drifts from
    /// the core trait, this stops compiling — the regression guard for
    /// the cycle-break wiring. The `Arc<dyn DriverMemPool>` coercion
    /// mirrors exactly how the tenant context stores the pool.
    #[test]
    fn tenant_mem_pool_is_a_driver_mem_pool() {
        fn assert_driver_mem_pool<T: DriverMemPool>() {}
        assert_driver_mem_pool::<TenantMemPool>();
        // The object-safe coercion the tenant crate relies on must hold.
        fn _accepts_dyn(_: Arc<dyn DriverMemPool>) {}
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

    /// Fix #1: the host-side cap reservation arithmetic. Driver-free, so it
    /// runs on a no-GPU host (unlike the `#[ignore]`d driver-pin tests). This
    /// is the regression guard that the cap is a *hard ceiling*, independent of
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`.
    #[test]
    fn reserve_step_admits_up_to_the_cap() {
        const CAP: u64 = 64 * 1024 * 1024;
        // Exactly at the cap from empty: admitted.
        assert_eq!(reserve_step(CAP, 0, CAP).unwrap(), CAP);
        // Under cap: admitted, returns the new running total.
        assert_eq!(
            reserve_step(CAP, 16 * 1024 * 1024, 16 * 1024 * 1024).unwrap(),
            32 * 1024 * 1024
        );
        // Fills the remaining headroom exactly.
        assert_eq!(
            reserve_step(CAP, 48 * 1024 * 1024, 16 * 1024 * 1024).unwrap(),
            CAP
        );
    }

    #[test]
    fn reserve_step_rejects_over_cap_with_oom_shape() {
        const CAP: u64 = 64 * 1024 * 1024;
        // The BUG-1 scenario: 128 MiB against a 64 MiB cap from empty.
        let err = reserve_step(CAP, 0, 128 * 1024 * 1024).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("OUT_OF_MEMORY"),
            "over-cap rejection must be OOM-shaped so callers match the driver \
             error; got: {msg}"
        );
        // One byte past the cap is still rejected (off-by-one guard).
        assert!(reserve_step(CAP, CAP, 1).is_err());
        // No headroom left, zero-byte step is fine (degenerate, but must not
        // spuriously reject).
        assert_eq!(reserve_step(CAP, CAP, 0).unwrap(), CAP);
    }

    #[test]
    fn reserve_step_rejects_overflow_as_oom() {
        // A near-u64::MAX live total + a large request overflows; treated as
        // OOM rather than silently wrapping the counter.
        let err = reserve_step(u64::MAX, u64::MAX - 4, 16).unwrap_err();
        assert!(format!("{err}").contains("OUT_OF_MEMORY"));
    }
}
