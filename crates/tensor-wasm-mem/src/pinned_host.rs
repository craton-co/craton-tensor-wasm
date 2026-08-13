// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `GuardedHostBuffer` — host buffer used when the `unified-memory` feature
//! is disabled.
//!
//! The previous name (`PinnedHostBuffer`) was aspirational — this type does
//! not call `cudaHostAlloc` and therefore does not produce page-locked,
//! DMA-friendly memory today. It unconditionally allocates a page-aligned
//! anonymous mapping (via the cross-platform [`region`] crate) and brackets
//! the usable region with `PROT_NONE` / `PAGE_NOACCESS` guard pages —
//! sufficient for the `--no-default-features` CI path and for developer
//! laptops without CUDA installed. The new name describes what it actually
//! is. A backwards-compatible deprecated alias (`PinnedHostBuffer`) is kept
//! for `tensor-wasm-bench` and any external callers until they migrate.
//!
//! On all hosts the usable region is bracketed by `PROT_NONE` /
//! `PAGE_NOACCESS` guard pages, catching OOB reads/writes at the OS level
//! (SIGSEGV on Unix; `EXCEPTION_ACCESS_VIOLATION` on Windows). The managed
//! memory paths ([`crate::unified::UnifiedBuffer`] on CUDA hosts) still
//! cannot use guard pages — the CUDA driver page-migrates managed memory
//! between host and device and applying host page protection would race
//! with the migration machinery. That limitation is unchanged.
//!
//! mem finding #6: with a CUDA backing feature enabled (`unified-memory` or
//! `cudarc-backend`) the usable region is additionally **page-locked** for DMA
//! via `cuMemHostRegister_v2(CU_MEMHOSTREGISTER_PORTABLE |
//! CU_MEMHOSTREGISTER_DEVICEMAP)` — the driver-API equivalent of
//! `cudaHostAlloc(cudaHostAllocPortable | cudaHostAllocMapped)`. We *register*
//! the already-allocated guarded mapping rather than allocating fresh memory
//! with `cuMemHostAlloc` precisely so the `PROT_NONE` guard pages bracketing
//! the usable region are preserved (a fresh `cuMemHostAlloc` block would have
//! no guards). On the no-CUDA `--no-default-features` build the registration is
//! compiled out and the buffer is plain guarded (unpinned) host memory, as
//! before. Registration failure degrades to unpinned memory with a `debug`
//! log — page-locking is a DMA *optimisation*, not a correctness requirement.
//!
//! The API is intentionally symmetric with [`crate::unified::UnifiedBuffer`]
//! so call sites can swap one buffer type for the other without conditional
//! compilation at the call site.

use std::fmt;
use std::ptr::NonNull;

use region::{Allocation, Protection};

use crate::unified::{DeviceId, UnifiedError};

/// Owns the full guarded region: `[guard | usable | guard]`.
///
/// The `region::Allocation` releases its mapping (and therefore the guard
/// pages) when this struct is dropped — there is no separate teardown for
/// the protections.
struct GuardedAllocation {
    /// Full anonymous mapping. Total length is `2 * page + usable_size`.
    ///
    /// Held purely for its `Drop`, which unmaps the entire region (guard
    /// pages included). Reads of this field would be unsound — the live
    /// `ptr` inside [`GuardedHostBuffer`] aliases the same memory.
    #[allow(dead_code)]
    full_alloc: Allocation,
    /// Offset (in bytes) from the start of the mapping to the usable region.
    /// Always equal to one page. Used by `#[cfg(test)]` accessors.
    #[allow(dead_code)]
    usable_offset: usize,
    /// Size in bytes of the read/write region between the guard pages,
    /// rounded up to a multiple of the page size. Used by `Debug` and the
    /// `#[cfg(test)]` `usable_size()` accessor.
    #[allow(dead_code)]
    usable_size: usize,
}

/// A guarded host buffer (anonymous mapping bracketed by `PROT_NONE` pages).
///
/// The usable region is bracketed by `PROT_NONE` / `PAGE_NOACCESS` guard
/// pages so that out-of-bounds reads or writes raise a hardware fault at the
/// OS level instead of silently corrupting adjacent allocations.
///
/// Naming note: this type used to be called `PinnedHostBuffer`; a deprecated
/// type alias is kept at the bottom of this module for backwards
/// compatibility with `tensor-wasm-bench`.
pub struct GuardedHostBuffer {
    /// Pointer to the first usable byte (one page past the start of
    /// `allocation.full_alloc`).
    ptr: NonNull<u8>,
    /// Number of bytes the caller requested (always `<= allocation.usable_size`).
    size: usize,
    device_id: DeviceId,
    allocation: GuardedAllocation,
    /// Whether the usable region was successfully page-locked via
    /// `cuMemHostRegister_v2` (mem finding #6). `true` only on a CUDA build
    /// where registration succeeded; gates the matching `cuMemHostUnregister`
    /// in [`Drop`]. Only present on CUDA builds — on the no-CUDA build there
    /// is no pinning, so the field (and the whole [`Drop`] impl) is compiled
    /// out and teardown relies on `GuardedAllocation`'s mapping `Drop`.
    #[cfg(any(feature = "unified-memory", feature = "cudarc-backend"))]
    pinned: bool,
}

// SAFETY: the inner buffer is owned by this struct, not shared. The
// `region::Allocation` mapping is `Send + Sync`; the raw `NonNull<u8>` derived
// from it is just an offset into that mapping and inherits the same guarantee.
unsafe impl Send for GuardedHostBuffer {}
unsafe impl Sync for GuardedHostBuffer {}

impl GuardedHostBuffer {
    /// Allocate `size` bytes of pinned host memory on the default device.
    pub fn new(size: usize) -> Result<Self, UnifiedError> {
        Self::new_on(size, DeviceId::default())
    }

    /// Allocate `size` bytes of pinned host memory on the named device.
    ///
    /// The returned buffer is bracketed by `PROT_NONE` guard pages. The
    /// usable region is rounded up to a multiple of the OS page size, but
    /// [`Self::len`] still reports the caller-requested `size`.
    pub fn new_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        let page = region::page::size();
        // Round usable up to a page multiple. Both the round-up and the
        // guard-page addition can overflow when `size` is close to
        // `usize::MAX`; without these checks `total` would wrap to a tiny
        // value, `region::alloc` would succeed, and the slices we hand out
        // would point outside the mapping.
        let usable = size.checked_next_multiple_of(page).ok_or_else(|| {
            UnifiedError::Allocation(format!(
                "size {size} overflows usable bytes when rounded to page size {page}"
            ))
        })?;
        let total = usable.checked_add(2 * page).ok_or_else(|| {
            UnifiedError::Allocation(format!(
                "usable {usable} overflows total bytes when adding 2 guard pages of {page}"
            ))
        })?;
        // Allocate the whole region as PROT_NONE first, then mark the middle
        // as ReadWrite. `region::alloc` returns page-aligned anonymous memory.
        let alloc = region::alloc(total, Protection::NONE)
            .map_err(|e| UnifiedError::Allocation(format!("region::alloc: {e}")))?;
        let base = alloc.as_ptr::<u8>();
        // SAFETY: `base..base + total` is the freshly allocated mapping;
        // adding one page lands strictly inside it.
        let usable_start = unsafe { base.add(page) };
        // SAFETY: `usable_start..usable_start + usable` is the middle slice
        // of the mapping (between the two guard pages).
        unsafe {
            region::protect(usable_start, usable, Protection::READ_WRITE)
                .map_err(|e| UnifiedError::Allocation(format!("region::protect: {e}")))?;
        }
        let ptr = NonNull::new(usable_start as *mut u8)
            .ok_or_else(|| UnifiedError::Allocation("region::alloc returned null".into()))?;
        // mem finding #6: page-lock the usable region for DMA on CUDA builds.
        // We register exactly the `[ptr, ptr + usable)` window (page-aligned
        // and a whole number of pages, so it satisfies `cuMemHostRegister`'s
        // alignment requirement) and leave the guard pages untouched.
        #[cfg(any(feature = "unified-memory", feature = "cudarc-backend"))]
        let pinned = Self::try_pin(ptr, usable, device_id);
        Ok(Self {
            ptr,
            size,
            device_id,
            allocation: GuardedAllocation {
                full_alloc: alloc,
                usable_offset: page,
                usable_size: usable,
            },
            #[cfg(any(feature = "unified-memory", feature = "cudarc-backend"))]
            pinned,
        })
    }

    /// Page-lock `[ptr, ptr + usable)` for DMA via `cuMemHostRegister_v2`
    /// (mem finding #6). Returns `true` if the region is now pinned.
    ///
    /// Best-effort: any failure (no driver, unsupported platform, already
    /// registered) logs at `debug` and returns `false` so the buffer falls
    /// back to ordinary (unpinned) guarded host memory — page-locking is a DMA
    /// optimisation, not a correctness requirement. The `PORTABLE` flag makes
    /// the pinning visible to every CUDA context (multi-tenant safe); `DEVICEMAP`
    /// requests a device-accessible mapping so a kernel can DMA the bytes
    /// without a staging copy.
    #[cfg(feature = "unified-memory")]
    fn try_pin(ptr: NonNull<u8>, usable: usize, _device_id: DeviceId) -> bool {
        use cust::sys as cuda_sys;
        // mem M4: bind the cust primary context before the context-sensitive
        // registration call.
        if crate::unified::ensure_cust_context_bound().is_err() {
            tracing::debug!(
                target: "tensor_wasm_mem::pinned_host",
                "try_pin: could not bind cust context; leaving host memory unpinned"
            );
            return false;
        }
        let flags = cuda_sys::CU_MEMHOSTREGISTER_PORTABLE | cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        // SAFETY: `ptr` is page-aligned and points to `usable` valid, RW,
        // host-addressable bytes (the freshly-protected middle of the guarded
        // mapping); the context was bound above.
        let res =
            unsafe { cuda_sys::cuMemHostRegister_v2(ptr.as_ptr() as *mut core::ffi::c_void, usable, flags) };
        if res == cuda_sys::CUresult::CUDA_SUCCESS {
            true
        } else {
            tracing::debug!(
                target: "tensor_wasm_mem::pinned_host",
                result = ?res,
                "try_pin: cuMemHostRegister failed; leaving host memory unpinned"
            );
            false
        }
    }

    /// cudarc-only variant of [`Self::try_pin`] (mem finding #6).
    #[cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]
    fn try_pin(ptr: NonNull<u8>, usable: usize, device_id: DeviceId) -> bool {
        use cudarc::driver::sys as cuda_sys;
        // mem M4: bind the owning device's primary context first.
        let device = match crate::cudarc_backend::device_for(device_id.0) {
            Ok(d) => d,
            Err(_) => {
                tracing::debug!(
                    target: "tensor_wasm_mem::pinned_host",
                    "try_pin: could not resolve cudarc device; leaving host memory unpinned"
                );
                return false;
            }
        };
        if crate::cudarc_backend::ensure_context_bound(&device).is_err() {
            tracing::debug!(
                target: "tensor_wasm_mem::pinned_host",
                "try_pin: could not bind cudarc context; leaving host memory unpinned"
            );
            return false;
        }
        let flags = cuda_sys::CU_MEMHOSTREGISTER_PORTABLE | cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        // SAFETY: see the cust arm — `ptr`/`usable` describe a page-aligned RW
        // host region and the context is bound.
        let res = unsafe {
            cuda_sys::lib().cuMemHostRegister_v2(
                ptr.as_ptr() as *mut core::ffi::c_void,
                usable,
                flags,
            )
        };
        if res == cuda_sys::cudaError_enum::CUDA_SUCCESS {
            true
        } else {
            tracing::debug!(
                target: "tensor_wasm_mem::pinned_host",
                result = ?res,
                "try_pin: cuMemHostRegister failed; leaving host memory unpinned"
            );
            false
        }
    }

    /// Length in bytes — matches the caller-requested size, not the page-rounded usable region.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length (never after a successful construction).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Device this buffer is tagged with.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
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
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null and points to size valid bytes by invariant.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Borrow as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` proves uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Size of the read/write region between the two guard pages, in bytes.
    ///
    /// Always a multiple of the OS page size and `>= self.len()`. Primarily
    /// useful for tests that need to address the byte one past the end of
    /// the usable region (the start of the trailing guard page).
    #[cfg(test)]
    pub(crate) fn usable_size(&self) -> usize {
        self.allocation.usable_size
    }

    /// Total size of the underlying mapping (guard pages included), in bytes.
    #[cfg(test)]
    pub(crate) fn full_mapping_size(&self) -> usize {
        self.allocation.full_alloc.len()
    }

    /// Byte offset from the mapping start to the first usable byte. Always one page.
    #[cfg(test)]
    pub(crate) fn usable_offset(&self) -> usize {
        self.allocation.usable_offset
    }

    /// No-op prefetch hint. Present for API parity with [`crate::unified::UnifiedBuffer`].
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        Ok(())
    }

    /// No-op prefetch hint. Present for API parity with [`crate::unified::UnifiedBuffer`].
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        Ok(())
    }
}

impl fmt::Debug for GuardedHostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardedHostBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("size", &self.size)
            .field("usable_size", &self.allocation.usable_size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

// mem finding #6: unregister the page-locked region before its mapping is
// unmapped. Only compiled on CUDA builds (the no-CUDA build never pins, so it
// needs no Drop — the `region::Allocation` field's own Drop unmaps everything).
//
// Drop ordering: this `Drop::drop` body runs FIRST, then Rust drops the
// struct's fields in declaration order — `allocation` (the
// `region::Allocation`) last. So `cuMemHostUnregister` always precedes the
// `munmap`/`VirtualFree`; unregistering after the unmap would target freed
// virtual addresses.
#[cfg(any(feature = "unified-memory", feature = "cudarc-backend"))]
impl Drop for GuardedHostBuffer {
    fn drop(&mut self) {
        if !self.pinned {
            return;
        }
        #[cfg(feature = "unified-memory")]
        {
            use cust::sys as cuda_sys;
            // Bind the context so the unregister dispatches against the right
            // primary context (mem M4). If the bind fails the pages are about
            // to be unmapped anyway; log and proceed.
            if crate::unified::ensure_cust_context_bound().is_ok() {
                // SAFETY: `self.ptr` is the exact pointer passed to
                // `cuMemHostRegister_v2`; the region is still mapped (the
                // `allocation` field drops after this body returns).
                let res =
                    unsafe { cuda_sys::cuMemHostUnregister(self.ptr.as_ptr() as *mut core::ffi::c_void) };
                if res != cuda_sys::CUresult::CUDA_SUCCESS {
                    tracing::debug!(
                        target: "tensor_wasm_mem::pinned_host",
                        result = ?res,
                        "drop: cuMemHostUnregister failed"
                    );
                }
            }
        }
        #[cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]
        {
            use cudarc::driver::sys as cuda_sys;
            if let Ok(device) = crate::cudarc_backend::device_for(self.device_id.0) {
                if crate::cudarc_backend::ensure_context_bound(&device).is_ok() {
                    // SAFETY: see the cust arm — `self.ptr` is the registered
                    // pointer and the region is still mapped.
                    let res = unsafe {
                        cuda_sys::lib()
                            .cuMemHostUnregister(self.ptr.as_ptr() as *mut core::ffi::c_void)
                    };
                    if res != cuda_sys::cudaError_enum::CUDA_SUCCESS {
                        tracing::debug!(
                            target: "tensor_wasm_mem::pinned_host",
                            result = ?res,
                            "drop: cuMemHostUnregister failed"
                        );
                    }
                }
            }
        }
    }
}

/// Deprecated alias for [`GuardedHostBuffer`].
///
/// Retained so external callers (notably `tensor-wasm-bench`) keep compiling while
/// they migrate. Will be removed in a future minor version.
#[deprecated(
    since = "0.1.0",
    note = "renamed to `GuardedHostBuffer`; this alias will be removed"
)]
pub type PinnedHostBuffer = GuardedHostBuffer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut b = GuardedHostBuffer::new(128).unwrap();
        b.as_mut_slice().fill(0xAB);
        assert!(b.as_slice().iter().all(|&v| v == 0xAB));
        assert_eq!(b.len(), 128);
        assert!(!b.is_empty());
    }

    #[test]
    fn zero_size_rejected() {
        assert!(matches!(
            GuardedHostBuffer::new(0).unwrap_err(),
            UnifiedError::ZeroSize
        ));
    }

    /// `size == usize::MAX` must error rather than overflow the page-rounding
    /// arithmetic and hand back a buffer pointing outside its mapping.
    #[test]
    fn usize_max_size_rejected() {
        let err = GuardedHostBuffer::new(usize::MAX).unwrap_err();
        assert!(
            matches!(err, UnifiedError::Allocation(_)),
            "expected Allocation error, got {err:?}",
        );
    }

    #[test]
    fn device_id_carried() {
        let b = GuardedHostBuffer::new_on(16, DeviceId(7)).unwrap();
        assert_eq!(b.device_id(), DeviceId(7));
    }

    #[test]
    fn prefetch_methods_are_no_ops() {
        let b = GuardedHostBuffer::new(32).unwrap();
        b.prefetch_to_device().expect("no-op should succeed");
        b.prefetch_to_host().expect("no-op should succeed");
    }

    /// Reported `len` matches the caller request, not the page-rounded usable region.
    #[test]
    fn len_reports_requested_not_rounded() {
        let page = region::page::size();
        // Pick a size that is definitely not a page multiple.
        let requested = 1234;
        assert!(requested < page, "test assumes requested < page");
        let b = GuardedHostBuffer::new(requested).unwrap();
        assert_eq!(b.len(), requested);
        assert_eq!(b.as_slice().len(), requested);
        // Usable region rounded up to one page.
        assert_eq!(b.usable_size(), page);
        // Full mapping is usable + two guard pages.
        assert_eq!(b.full_mapping_size(), 3 * page);
        assert_eq!(b.usable_offset(), page);
    }

    /// Writes across the full usable byte range succeed and read back intact.
    #[test]
    fn full_range_round_trip() {
        let page = region::page::size();
        // Choose a request that spans more than one page so we exercise multiple pages.
        let requested = page + 17;
        let mut b = GuardedHostBuffer::new(requested).unwrap();
        for (i, byte) in b.as_mut_slice().iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        for (i, &byte) in b.as_slice().iter().enumerate() {
            assert_eq!(byte, (i % 251) as u8, "mismatch at {i}");
        }
    }

    /// Confirms the bracketing pages are mapped `Protection::NONE`.
    ///
    /// This is the cross-platform stand-in for catching a SIGSEGV /
    /// `EXCEPTION_ACCESS_VIOLATION`: rather than provoke the fault (which
    /// would terminate the test process), we ask the OS to report the
    /// protection bits on those pages and assert they are inaccessible.
    #[test]
    fn guard_pages_are_inaccessible() {
        let page = region::page::size();
        let b = GuardedHostBuffer::new(1024).unwrap();
        let usable_ptr = b.as_ptr();

        // SAFETY: `usable_ptr` lies one page past the mapping start, so
        // subtracting one page lands inside the leading guard page; adding
        // the usable size lands at the start of the trailing guard page.
        let before = unsafe { usable_ptr.sub(page) };
        let after = unsafe { usable_ptr.add(b.usable_size()) };

        let prot_before = region::query(before).expect("query before-guard");
        let prot_after = region::query(after).expect("query after-guard");
        assert_eq!(
            prot_before.protection(),
            Protection::NONE,
            "before-guard must be PROT_NONE/PAGE_NOACCESS",
        );
        assert_eq!(
            prot_after.protection(),
            Protection::NONE,
            "after-guard must be PROT_NONE/PAGE_NOACCESS",
        );

        // And the usable region itself must be readable + writable.
        let prot_usable = region::query(usable_ptr).expect("query usable");
        assert_eq!(
            prot_usable.protection(),
            Protection::READ_WRITE,
            "usable region must be RW",
        );
    }

    /// Dropping a buffer must release its full mapping, including guard pages.
    #[test]
    fn drop_releases_mapping() {
        // Allocate and drop many buffers; if guards leaked we'd run out of
        // virtual address space well before this loop completes on 64-bit
        // systems, but the more realistic failure mode is the kernel's
        // per-process mapping count limit (e.g. vm.max_map_count on Linux).
        for _ in 0..256 {
            let b = GuardedHostBuffer::new(64 * 1024).unwrap();
            drop(b);
        }
    }
}
