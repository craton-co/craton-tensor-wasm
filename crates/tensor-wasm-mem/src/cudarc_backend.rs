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

use std::collections::HashSet;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaDevice;
use parking_lot::Mutex;

use crate::advise::Advice;
use crate::unified::{DeviceId, UnifiedBacking, UnifiedError, UvmAdvice};

/// Process-wide counter of `cuMemFree_v2` failures observed in
/// [`CudarcUnifiedBuffer::drop`]. Lock-free fast-path counter for alerting
/// — pair with [`leaked_cuda_allocations`] for the per-VA forensic trail.
static CUDARC_FREE_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Process-global set of raw CUDA device-pointer values orphaned by a failed
/// `cuMemFree_v2` in [`CudarcUnifiedBuffer::drop`]. See [`leaked_cuda_allocations`]
/// for the audit contract that closes mem H4.
static LEAKED_CUDA_ALLOCATIONS: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();

fn leaked_set() -> &'static Mutex<HashSet<u64>> {
    LEAKED_CUDA_ALLOCATIONS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Record that the CUDA driver pointer `ptr` was orphaned by a failed free.
///
/// Private helper so the only code path that can mutate the set is the
/// [`Drop`] impl on [`CudarcUnifiedBuffer`].
fn record_leak(ptr: u64) {
    leaked_set().lock().insert(ptr);
}

/// Number of `cuMemFree_v2` failures observed since process start.
///
/// Lock-free monotonic counter; subtract two snapshots for a leak rate.
/// For per-pointer audit data, use [`leaked_cuda_allocations`].
pub fn cudarc_free_failures() -> u64 {
    CUDARC_FREE_FAILURES.load(Ordering::Relaxed)
}

/// Snapshot the process-global set of CUDA pointers orphaned by a failed
/// `cuMemFree_v2` in [`CudarcUnifiedBuffer::drop`].
///
/// Returns a freshly allocated `Vec<u64>` — the lock is released before
/// return, so callers can iterate the snapshot without holding the leak
/// mutex. The order of the returned values is unspecified.
///
/// This is the auditing surface for mem H4: a non-empty vec means at least
/// one CUDA virtual address the driver still considers live but this crate
/// has lost the handle to. Safe remediation is to recycle the process
/// before scheduling work for a new tenant on the same device.
pub fn leaked_cuda_allocations() -> Vec<u64> {
    leaked_set().lock().iter().copied().collect()
}

/// Process-wide per-ordinal cache of `Arc<CudaDevice>`.
///
/// Caching per-ordinal `Arc<CudaDevice>` prevents the failed-construction
/// race that would otherwise release another tenant's primary context.
///
/// Audit T26 background. cudarc holds the device's primary context inside
/// the [`CudaDevice`] Arc — dropping the last `Arc<CudaDevice>` calls
/// `cuDevicePrimaryCtxRelease`, which invalidates *every* outstanding
/// managed pointer on that device process-wide. Before this cache, only
/// ordinal 0 was memoised; any non-zero ordinal allocation re-ran
/// `CudaDevice::new(ordinal)`. If that construction subsequently failed
/// downstream (e.g. before the new `Arc<CudaDevice>` was wired into a
/// surviving `CudarcUnifiedBuffer`), the freshly-built `Arc` would drop
/// and release the primary context — silently breaking every *other*
/// tenant's allocations on the same device. Keeping a per-ordinal cache
/// converts that into a single "build once, retain for the process
/// lifetime" guarantee: subsequent allocations clone the cached `Arc`, so
/// no transient holder ever owns the last reference.
///
/// The cache is populated lazily by [`device_for`] under a single-shot
/// `OnceLock`-initialised mutex. Real call sites will route through the
/// tenant-aware context cache once the spike graduates — see
/// `docs/CUDARC-SPIKE.md`.
static DEVICE_CACHE: OnceLock<Mutex<std::collections::HashMap<u32, Arc<CudaDevice>>>> =
    OnceLock::new();

fn device_cache() -> &'static Mutex<std::collections::HashMap<u32, Arc<CudaDevice>>> {
    DEVICE_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Lazily construct (or fetch) the cached cudarc device for ordinal `ordinal`.
///
/// Caching per-ordinal `Arc<CudaDevice>` prevents the failed-construction
/// race that would otherwise release another tenant's primary context.
///
/// The first call for a given ordinal runs `CudaDevice::new(ordinal)`
/// under the cache mutex, inserts the resulting `Arc<CudaDevice>` keyed by
/// ordinal, and returns a clone. Every subsequent call for the same
/// ordinal returns a fresh clone of the cached `Arc` — the cache holds a
/// process-lifetime reference so the primary context never drops to zero
/// strong-count, which is exactly the keep-alive guarantee that protects
/// other tenants' managed pointers.
pub(crate) fn device_for(ordinal: u32) -> Result<Arc<CudaDevice>, UnifiedError> {
    // Hot path: probe under the mutex; if present, hand out a clone and
    // release the lock immediately. The `Arc::clone` cost is one atomic
    // increment so the lock window is bounded by a hash lookup.
    {
        let cache = device_cache().lock();
        if let Some(cached) = cache.get(&ordinal) {
            return Ok(Arc::clone(cached));
        }
    }

    // Cold path: build outside the lock so a slow `CudaDevice::new` does
    // not block other-ordinal probes. `CudaDevice::new` already returns an
    // `Arc<CudaDevice>` — cudarc's API hands us a refcounted handle to the
    // freshly-retained primary context.
    //
    // If another thread wins the race and inserts first, we drop the
    // duplicate `Arc<CudaDevice>` we built ourselves — that drop is safe
    // because the *winning* `Arc` is now pinned in the cache (its
    // strong-count is at least 1), so the primary context is not
    // released. This is the failed-construction race the cache exists to
    // close: only a transient holder that **also** loses the install race
    // would drop the last `Arc`, and we arrange for the install-race
    // winner to always be inside the cache.
    let fresh = CudaDevice::new(ordinal as usize)
        .map_err(|e| UnifiedError::Cuda(format!("CudaDevice::new({ordinal}): {e:?}")))?;
    // Re-lock and insert-if-absent. Returning the *cached* value (which
    // may be ours OR a winner's) ensures the caller always sees the
    // single canonical handle for this ordinal.
    //
    // L2: if a winner already installed an `Arc` for this ordinal we are
    // the loser of the race and must drop our freshly-built `fresh`. We
    // must NOT let that drop run while `cache` is locked: dropping the
    // last strong ref to a `CudaDevice` triggers `cuDevicePrimaryCtxRelease`
    // under the driver, and we do not want a driver call executing under
    // the cache mutex. So we capture the canonical clone, hold onto the
    // loser `Arc` in a binding (`fresh` is moved into the entry on the
    // winning path, otherwise stays here), drop the guard explicitly, and
    // only then let the loser `Arc` fall out of scope. Race-correctness is
    // unchanged: the winner's `Arc` is pinned in the cache (strong-count
    // ≥ 1) before we release the lock.
    let mut cache = device_cache().lock();
    let (canonical, loser) = match cache.entry(ordinal) {
        std::collections::hash_map::Entry::Occupied(occupied) => {
            // We lost the race: clone the winner's pinned handle and keep
            // our now-redundant `fresh` to drop after unlocking.
            (Arc::clone(occupied.get()), Some(fresh))
        }
        std::collections::hash_map::Entry::Vacant(vacant) => {
            // We won: install `fresh` (pinning strong-count ≥ 1) and hand
            // out a clone. Nothing to drop.
            (Arc::clone(vacant.insert(fresh)), None)
        }
    };
    // Release the lock BEFORE the loser `Arc` is dropped so any
    // `cuDevicePrimaryCtxRelease` runs outside the cache mutex.
    drop(cache);
    drop(loser);
    Ok(canonical)
}

/// Bind the device's primary context to the calling thread (mem M4).
///
/// CUDA's driver API maintains a *thread-local* current context. If a
/// `CudarcUnifiedBuffer` is created on one thread and then `apply_advice`
/// / `prefetch_to_device` / `prefetch_to_host` is invoked on a *different*
/// thread (e.g. a tokio worker or a rayon worker that has never touched
/// CUDA), the driver either returns `CUDA_ERROR_INVALID_CONTEXT` or — worse,
/// for a multi-tenant host — silently uses whatever context the driver
/// happened to leave current, potentially another tenant's device. Both
/// outcomes are unacceptable; the latter is a tenancy bug.
///
/// `bind_to_thread` calls `cuCtxSetCurrent` against the cached primary
/// context for this `CudaDevice`. It is idempotent on a thread that is
/// already bound to the same context — the driver compares pointer
/// identity, so the cost is a single TLS write on the steady state — so
/// callers can invoke it unconditionally without a fast-path check.
///
/// Failure is surfaced as [`UnifiedError::Cuda`]; we deliberately do not
/// fall through to the driver call, because every downstream `cuMem*`
/// entry point would itself fail with `CUDA_ERROR_INVALID_CONTEXT` and
/// the operator's first error message would be misleadingly downstream.
pub(crate) fn ensure_context_bound(device: &Arc<CudaDevice>) -> Result<(), UnifiedError> {
    device
        .bind_to_thread()
        .map_err(|e| UnifiedError::Cuda(format!("CudaDevice::bind_to_thread: {e:?}")))
}

/// True if the device at `ordinal` reports
/// `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS != 0` — the prerequisite for
/// `cuMemPrefetchAsync` on managed memory (finding #10).
///
/// On Windows under the **WDDM** driver model this attribute is `0`: the OS
/// owns GPU memory residency, so explicit managed-memory prefetch is
/// unsupported and `cuMemPrefetchAsync` fails with
/// `CUDA_ERROR_INVALID_DEVICE`. It is `1` on Linux and on Windows/TCC. The
/// prefetch entry points use this to degrade to an advisory no-op where the
/// platform cannot honour the hint, rather than surfacing a spurious error
/// (prefetch is a performance hint — managed memory remains correct without
/// it). Verified on the RTX 2060 (Windows/WDDM, cc 7.5): the attribute is 0
/// and `cuMemPrefetchAsync` returned `INVALID_DEVICE` before this gate.
///
/// Returns `false` on any driver error querying the attribute — the safe,
/// conservative default (skip the optimisation) for a hint path.
fn supports_managed_prefetch(ordinal: u32) -> bool {
    // SAFETY: out-params are valid locals; a prior `device_for(ordinal)` (every
    // caller allocates before prefetching) has primed the driver lib + run
    // `cuInit`, so these queries cannot hit an uninitialised driver.
    unsafe {
        let mut dev: cuda_sys::CUdevice = 0;
        if cuda_sys::lib().cuDeviceGet(&mut dev as *mut cuda_sys::CUdevice, ordinal as i32)
            != cuda_sys::cudaError_enum::CUDA_SUCCESS
        {
            return false;
        }
        let mut val: i32 = 0;
        if cuda_sys::lib().cuDeviceGetAttribute(
            &mut val as *mut i32,
            cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
            dev,
        ) != cuda_sys::cudaError_enum::CUDA_SUCCESS
        {
            return false;
        }
        val != 0
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
        // mem M4 / finding #9: bind the primary context to THIS thread before
        // `cuMemAllocManaged`. `device_for` returns a cached `Arc<CudaDevice>`
        // clone on every call after the first, and `CudaDevice::new` only binds
        // the context on the thread that built it — so an allocation on any
        // other thread (a tokio/rayon worker, or simply a libtest per-test
        // thread) would otherwise fail with `CUDA_ERROR_INVALID_CONTEXT`. The
        // sibling `apply_advice` / `prefetch_*` paths already call this; `new_on`
        // was missing it (verified on the RTX 2060: cudarc_smoke round-trip tests
        // failed with INVALID_CONTEXT — see docs/GPU-VALIDATION-2026-05-30.md).
        ensure_context_bound(&device)?;
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
            CU_MEM_ATTACH_GLOBAL == cuda_sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
            "cudarc renumbered CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL; \
             update the inlined constant in cudarc_backend.rs",
        );
        // SAFETY: `raw` is a valid out-parameter; `size > 0`; the
        // `ensure_context_bound` call above made the primary context current on
        // this thread.
        // cudarc 0.13.x exposes CUDA driver functions as methods on a Lib
        // struct, accessed via cudarc::driver::sys::lib(). Free-function
        // imports like cust uses are not available; `device_for(ordinal)`
        // above (T26 per-ordinal cache) ran `CudaDevice::new` which primed
        // the driver lib, so this `lib()` call cannot panic.
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
    ///
    /// # Thread safety (mem M4)
    ///
    /// Binds the primary context on the calling thread before issuing the
    /// driver call. Each call is therefore safe to invoke from any thread
    /// without prior `CudaDevice::bind_to_thread`; the cost of the bind on
    /// an already-bound thread is a single TLS write.
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        // mem M4: bind first so the driver dispatches against *this* buffer's
        // primary context rather than whatever context the calling thread
        // last touched (potentially another tenant's device, or unset).
        ensure_context_bound(&self.device)?;
        // Finding #10: `cuMemPrefetchAsync` requires the device's
        // `CONCURRENT_MANAGED_ACCESS` attribute (0 on Windows/WDDM, where the
        // call fails with `INVALID_DEVICE`). Prefetch is an advisory hint, so on
        // platforms that cannot honour it we degrade to a no-op rather than
        // erroring — managed memory stays correct, just without the migration
        // optimisation.
        if !supports_managed_prefetch(self.device_id.0) {
            tracing::debug!(
                target: "tensor_wasm_mem::cudarc_backend",
                device = self.device_id.0,
                "prefetch_to_device: skipped (device lacks CONCURRENT_MANAGED_ACCESS, e.g. Windows/WDDM)"
            );
            return Ok(());
        }
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
    ///
    /// # Thread safety (mem M4)
    ///
    /// Binds the primary context on the calling thread before issuing the
    /// driver call. Each call is therefore safe to invoke from any thread
    /// without prior `CudaDevice::bind_to_thread`; the cost of the bind on
    /// an already-bound thread is a single TLS write.
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        // `CU_DEVICE_CPU` is the sentinel for "host" in cuMemPrefetchAsync.
        // cudarc 0.13 does not export it as a named constant; CUDA defines it
        // as `-1`.
        const CU_DEVICE_CPU: i32 = -1;
        // mem M4: bind first — see `prefetch_to_device` for the full rationale.
        ensure_context_bound(&self.device)?;
        // Finding #10: gate on CONCURRENT_MANAGED_ACCESS — see
        // `prefetch_to_device`. Migrating back to the host is the same
        // `cuMemPrefetchAsync` API and is equally unsupported on WDDM, so we
        // degrade to an advisory no-op there.
        if !supports_managed_prefetch(self.device_id.0) {
            tracing::debug!(
                target: "tensor_wasm_mem::cudarc_backend",
                device = self.device_id.0,
                "prefetch_to_host: skipped (device lacks CONCURRENT_MANAGED_ACCESS, e.g. Windows/WDDM)"
            );
            return Ok(());
        }
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

impl UnifiedBacking for CudarcUnifiedBuffer {
    fn len(&self) -> usize {
        CudarcUnifiedBuffer::len(self)
    }

    fn as_slice(&self) -> &[u8] {
        CudarcUnifiedBuffer::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        CudarcUnifiedBuffer::as_mut_slice(self)
    }

    fn apply_advice(&self, hint: UvmAdvice) -> Result<(), UnifiedError> {
        // Translate the trait-facing `UvmAdvice` to the cudarc path's
        // internal `Advice` enum. `UvmAdvice::UnsetReadMostly` has no
        // matching variant on the cudarc path today, so surface it as
        // `NotSupported` rather than papering over the gap with a
        // silent no-op.
        let advice = match hint {
            UvmAdvice::SetReadMostly => Advice::ReadMostly,
            UvmAdvice::UnsetReadMostly => {
                return Err(UnifiedError::NotSupported {
                    feature: "apply_advice(UnsetReadMostly)",
                    backing: "cudarc",
                });
            }
            UvmAdvice::SetPreferredLocation(d) => Advice::PreferredLocation(DeviceId(d)),
            UvmAdvice::UnsetPreferredLocation => Advice::UnsetPreferredLocation,
            UvmAdvice::SetAccessedBy(d) => Advice::AccessedBy(DeviceId(d)),
            UvmAdvice::UnsetAccessedBy(d) => Advice::UnsetAccessedBy(DeviceId(d)),
        };
        // Disambiguate against the trait method we are inside — the
        // free function lives at module scope and the unqualified name
        // resolution does pick it over the trait method here, but the
        // fully-qualified path makes the intent explicit for readers.
        self::apply_advice(self, advice)
    }

    fn prefetch_to_device(&self, device_ord: u32) -> Result<(), UnifiedError> {
        // The existing `CudarcUnifiedBuffer::prefetch_to_device`
        // inferred the ordinal from `self.device_id`. The trait
        // signature carries an explicit ordinal so future plumbing can
        // target a non-owning device; if the caller asks for the
        // owning device we route to the existing implementation, and
        // for any other ordinal we issue the driver call directly with
        // the supplied target. The cached `device` is the buffer's
        // owning context; `cuMemPrefetchAsync` accepts any valid
        // device ordinal so this is safe as long as the primary
        // context for the owning device is current (it is — we bind
        // it via the inherent method on the owning-ordinal path).
        if device_ord == self.device_id().0 {
            CudarcUnifiedBuffer::prefetch_to_device(self)
        } else {
            // Cross-device prefetch via the inherent method needs a
            // refactor (it currently hard-codes `self.device_id.0`).
            // Surface as `NotSupported` so the v0.4 cutover can wire
            // it explicitly rather than have us silently retarget an
            // ordinal that may not even be online.
            Err(UnifiedError::NotSupported {
                feature: "prefetch_to_device(non-owning-ordinal)",
                backing: "cudarc",
            })
        }
    }

    fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        CudarcUnifiedBuffer::prefetch_to_host(self)
    }
}

impl Drop for CudarcUnifiedBuffer {
    /// Free the underlying `cuMemAllocManaged` allocation via `cuMemFree_v2`.
    ///
    /// # Invariants
    ///
    /// - Never panics. A `Drop` panic during unwinding would abort the
    ///   process; we degrade to a recorded leak instead.
    /// - On any non-`CUDA_SUCCESS` from `cuMemFree_v2` the raw pointer is
    ///   added to the process-global [`leaked_cuda_allocations`] set so
    ///   operators can audit (mem H4). The failure is logged at `error!`
    ///   level (upgraded from `warn!` in the original implementation —
    ///   a failed free is a security-relevant event because the driver
    ///   may recycle the same VA for another tenant).
    /// - After a failed free, `self.ptr` is replaced with
    ///   `NonNull::dangling()`. This is purely defence-in-depth: if some
    ///   future refactor reaches this Drop a second time (it shouldn't —
    ///   Rust guarantees Drop runs at most once per value), the sentinel
    ///   ensures we don't re-submit the same orphaned pointer to the
    ///   driver and don't double-record the leak against the original
    ///   address.
    fn drop(&mut self) {
        let raw_ptr = self.ptr.as_ptr();
        let raw_ptr_u64 = raw_ptr as u64;
        // SAFETY: `ptr` was returned by `cuMemAllocManaged` and has not been
        // freed yet; the cached `Arc<CudaDevice>` we hold ensures the primary
        // context is still alive when we call `cuMemFree_v2`.
        let res = unsafe { cuda_sys::lib().cuMemFree_v2(raw_ptr as cuda_sys::CUdeviceptr) };
        if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            // Failed free: the driver still considers the VA mapped, so the
            // next allocator pass could hand it to another tenant carrying
            // residual bytes. Both the lock-free counter (for alerting) AND
            // the per-VA audit set (for forensics) are updated. Drop cannot
            // fail, so we surface via tracing + the audit set rather than
            // unwinding (mem H4).
            let failures = CUDARC_FREE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
            record_leak(raw_ptr_u64);
            tracing::error!(
                target: "tensor_wasm_mem::cudarc_backend",
                ?res,
                ptr = ?raw_ptr,
                size = self.size,
                total_failures = failures,
                "cuMemFree_v2 failed in CudarcUnifiedBuffer::drop -- VA leaked, \
                 recorded in leaked_cuda_allocations() for operator audit (mem H4)",
            );
            // Defence-in-depth: replace with a dangling sentinel so any
            // hypothetical double-drop short-circuits and does not re-submit
            // the same orphaned pointer to the driver. Rust guarantees Drop
            // runs at most once per value, so this is belt-and-braces.
            self.ptr = NonNull::dangling();
        }
    }
}

/// Apply an [`Advice`] hint via `cuMemAdvise` against a [`CudarcUnifiedBuffer`].
///
/// Mirror of [`crate::advise::apply`] for the cudarc path. Returns
/// [`UnifiedError::Cuda`] on failure.
///
/// # Thread safety (mem M4)
///
/// Binds the primary context on the calling thread before issuing the
/// driver call. Each call is therefore safe to invoke from any thread
/// without prior `CudaDevice::bind_to_thread`; the cost of the bind on
/// an already-bound thread is a single TLS write. Prior to mem M4 this
/// function would silently dispatch against whatever CUDA context the
/// calling thread last touched — potentially another tenant's device,
/// or no context at all (`CUDA_ERROR_INVALID_CONTEXT`).
pub fn apply_advice(buffer: &CudarcUnifiedBuffer, advice: Advice) -> Result<(), UnifiedError> {
    // mem M4: bind first so the driver dispatches against *this* buffer's
    // primary context. See `ensure_context_bound` for the tenancy rationale.
    ensure_context_bound(&buffer.device)?;
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
    // SAFETY: ptr/size are derived from a valid live CudarcUnifiedBuffer;
    // the primary context was bound above (mem M4).
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

    /// Audit T26: two `device_for(0)` calls must return clones of the same
    /// underlying `Arc<CudaDevice>`. Pointer equality on the inner
    /// `CudaDevice` payload proves the cache is wired and no transient
    /// holder ever owned the last `Arc` strong-count.
    ///
    /// Gated `#[ignore]` because the cache lookup itself unavoidably calls
    /// `CudaDevice::new` on first miss, which `dlopen`s `libcuda` — that
    /// is not available on host-only CI. A host with a CUDA driver runs
    /// this test under `cargo test --features cudarc-backend -- --ignored`.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn device_cache_returns_same_arc_for_same_ordinal() {
        let a = device_for(0).expect("first device_for(0)");
        let b = device_for(0).expect("second device_for(0)");
        // `Arc::as_ptr` returns the address of the shared payload — equal
        // iff `a` and `b` clone the same `Arc`.
        assert_eq!(
            Arc::as_ptr(&a),
            Arc::as_ptr(&b),
            "device_for(0) must return clones of the cached Arc — \
             a fresh `CudaDevice::new` on every call would re-arm the \
             failed-construction race the T26 cache exists to close"
        );
    }
}
