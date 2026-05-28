// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `UnifiedBuffer` — a region of memory that is addressable from both CPU and
//! GPU when a CUDA backing feature is enabled.
//!
//! Three backings, selected at compile time:
//! - With `unified-memory`: `cudaMallocManaged` via the `cust` crate. Pages
//!   migrate on demand between host and device. This is the v0.3 default for
//!   any host that opts into a CUDA backing and remains the cust-backed path
//!   the audit-closed test matrix exercises.
//! - With `cudarc-backend` (and `unified-memory` OFF): `cuMemAllocManaged`
//!   via the W1.2 [`crate::cudarc_backend::CudarcUnifiedBuffer`] spike.
//!   This is the v0.5 cust-successor fallback per
//!   [RFC 0001](../../../rfcs/0001-cuda-oxide-integration.md): once
//!   cuda-oxide v0.2 ships its managed-memory wrapper this path can be
//!   swapped for `CudaOxideUnifiedBuffer`; until then `cudarc-backend`
//!   is the v0.5 default-flip candidate.
//! - Without either: a heap-allocated `Box<[u8]>`. This compiles on non-CUDA
//!   hosts and is what CI's no-feature build uses. It exposes the same API
//!   but the `prefetch_to_device` / `prefetch_to_host` methods become no-ops.
//!
//! # Feature precedence
//!
//! When both `unified-memory` and `cudarc-backend` are enabled simultaneously,
//! the cust path wins. This preserves the v0.3 default (every existing
//! production deployment built with `--features unified-memory` keeps the
//! exact same allocator under its feet) and lets dev boxes that have both
//! features turned on opt to cust without a separate feature gate. To force
//! the cudarc backing on such a host, build with
//! `--no-default-features --features cudarc-backend` (or otherwise omit
//! `unified-memory` from the active feature set).
//!
//! Precedence table:
//!
//! | `unified-memory` | `cudarc-backend` | Active backing             | `is_uvm_backed()` |
//! |------------------|-------------------|----------------------------|-------------------|
//! | on               | off               | cust `UnifiedBuffer<u8>`   | `true`            |
//! | on               | on                | cust `UnifiedBuffer<u8>`   | `true`            |
//! | off              | on                | `CudarcUnifiedBuffer`      | `true`            |
//! | off              | off               | `Box<[u8]>`                | `false`           |

use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use tensor_wasm_tenant::TenantContext;

/// Errors raised by `UnifiedBuffer` operations.
#[derive(Debug, thiserror::Error)]
pub enum UnifiedError {
    /// The underlying allocator failed.
    #[error("allocation failed: {0}")]
    Allocation(String),
    /// A CUDA API call failed.
    #[error("CUDA call failed: {0}")]
    Cuda(String),
    /// Zero-byte allocation requested (not supported).
    #[error("cannot allocate a zero-byte buffer")]
    ZeroSize,
    /// Requested allocation exceeds the configured / hard-coded cap.
    ///
    /// Distinct from [`UnifiedError::Allocation`] so callers can plumb
    /// structured `requested` / `limit` figures all the way through to
    /// `tensor_wasm_core::error::TensorWasmError::MemoryExhausted` without
    /// resorting to substring-matching on a message.
    #[error("requested {requested} bytes exceeds hard cap {limit}")]
    TooLarge {
        /// Bytes the caller asked for.
        requested: u64,
        /// Hard cap enforced by the host (bytes).
        limit: u64,
    },
}

/// Identifies a CUDA device. On non-CUDA hosts this is a free-form tag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda:{}", self.0)
    }
}

/// A contiguous memory region addressable from both CPU and GPU.
///
/// Safety invariant: `ptr` is non-null and points to `size` valid bytes
/// allocated by the active backing. Dropping the buffer frees the allocation.
pub struct UnifiedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    device_id: DeviceId,
    // Holds the underlying storage so it is freed on drop. Which concrete
    // variant `Backing` resolves to is determined at compile time by the
    // feature flags: cust `UnifiedBuffer<u8>` under `unified-memory`,
    // `CudarcUnifiedBuffer` under `cudarc-backend` (when `unified-memory`
    // is off), or a plain `Box<[u8]>` on the default no-feature build.
    // See the module-level precedence table for the full matrix.
    #[allow(dead_code)]
    backing: Backing,
    /// Tenant context for GPU memory accounting. When `Some`, the
    /// buffer's `Drop` impl calls
    /// [`TenantContext::release_gpu_bytes`] for `size` bytes. Set only
    /// by [`UnifiedBuffer::new_on_with_tenant_context`] — the legacy
    /// [`UnifiedBuffer::new`] / [`UnifiedBuffer::new_on`] constructors
    /// leave this `None` so existing call sites (tests, benches,
    /// untenanted use of the pool) are unaffected.
    ///
    /// Held in an `Arc` so the buffer can outlive whichever
    /// `TensorWasmMemoryCreator` constructed it — the creator's clone
    /// chain ends when the last `UnifiedBuffer` (or Wasm linear
    /// memory) finishes its `release_gpu_bytes` decrement.
    tenant_ctx: Option<Arc<TenantContext>>,
}

// SAFETY: the inner pointer is owned by this struct and not shared without
// explicit synchronisation. The wrapper type itself can be sent across threads;
// concurrent access to the underlying bytes requires external synchronisation
// (same contract as `Vec<u8>` once you have a `&mut [u8]`).
unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

#[cfg(feature = "unified-memory")]
mod backing_impl {
    use super::*;

    /// Compile-time constant exposed by [`super::UnifiedBuffer::is_uvm_backed`].
    ///
    /// `true` here: the cust path routes through `cuMemAllocManaged`. By the
    /// module-level precedence rule (see [`super`] doc), this branch wins
    /// whenever `unified-memory` is enabled even if `cudarc-backend` is also
    /// on, so dev hosts that toggle both features keep the v0.3 cust
    /// allocator. Setting this constant to `true` is part of the three-way
    /// gating documented at the module head: `unified-memory` OR
    /// `cudarc-backend` ⇒ `true`; only the default `Box<[u8]>` path ⇒ `false`.
    pub(crate) const IS_UVM_BACKED: bool = true;

    #[allow(dead_code)]
    pub(crate) enum Backing {
        Cuda(cust::memory::UnifiedBuffer<u8>),
    }

    /// Process-wide CUDA context init via cust::quick_init. cust 0.3
    /// uses an implicit primary-context model -- without this, any
    /// allocation returns `CUDA_ERROR_NOT_INITIALIZED` because `cuInit(0)`
    /// was never called. The result is held in a `OnceLock` so subsequent
    /// allocations are init-free. We only need to keep the `Context` alive
    /// for the rest of the process; nothing here ever drops it.
    fn ensure_cuda_init() -> Result<(), UnifiedError> {
        use std::sync::OnceLock;
        static CTX: OnceLock<Result<cust::context::Context, String>> = OnceLock::new();
        let r = CTX.get_or_init(|| {
            cust::quick_init().map_err(|e| format!("cust::quick_init: {e:?}"))
        });
        match r {
            Ok(_) => Ok(()),
            Err(msg) => Err(UnifiedError::Cuda(msg.clone())),
        }
    }

    impl Backing {
        pub(crate) fn allocate(
            size: usize,
            init_zero_bytes: usize,
        ) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Ensure cuInit(0) + a primary context have run before the first
            // cuMemAllocManaged. Idempotent and cheap on subsequent calls.
            ensure_cuda_init()?;
            // Allocate the managed region WITHOUT touching every byte. cust's
            // `UnifiedBuffer::new(&0u8, size)` runs a per-element write loop
            // over the whole allocation, which under `TensorWasmLinearMemory`'s
            // option-(a) preallocate-at-max strategy costs a full
            // `DEFAULT_MAX_BYTES` (256 MiB by default) memset on every Wasm
            // spawn. Wasm semantics only require the *visible* window to be
            // zero at instantiation; `memory.grow` bytes are zeroed by
            // Wasmtime itself. So we ask cust for an uninitialised allocation
            // and zero only `init_zero_bytes` ourselves.
            //
            // SAFETY: `uninitialized` only requires that the caller treats the
            // returned bytes as uninitialised memory until written. We do
            // exactly that — the slice below is the first write into the
            // region, and we cap it at `init_zero_bytes <= size`.
            let mut buf = unsafe { cust::memory::UnifiedBuffer::<u8>::uninitialized(size) }
                .map_err(|e| UnifiedError::Cuda(format!("{e:?}")))?;
            let init = init_zero_bytes.min(size);
            if init > 0 {
                buf.as_mut_slice()[..init].fill(0);
            }
            // SAFETY: cust returns a non-null aligned pointer to the allocation.
            let ptr = NonNull::new(buf.as_unified_ptr().as_raw_mut() as *mut u8)
                .ok_or_else(|| UnifiedError::Allocation("cust returned null".into()))?;
            Ok((ptr, Backing::Cuda(buf)))
        }
    }
}

#[cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]
mod backing_impl {
    use super::*;
    use crate::cudarc_backend::CudarcUnifiedBuffer;

    /// Compile-time constant exposed by [`super::UnifiedBuffer::is_uvm_backed`].
    ///
    /// `true` here: the cudarc path routes through `cuMemAllocManaged` via
    /// `cudarc::driver::sys::lib().cuMemAllocManaged` (see the W1.2 spike in
    /// [`crate::cudarc_backend`]). This branch only fires when
    /// `unified-memory` is OFF and `cudarc-backend` is ON — the module-level
    /// precedence rule lets cust win whenever both are enabled. Three-way
    /// gating recap: `unified-memory` OR `cudarc-backend` ⇒ `true`; only the
    /// default `Box<[u8]>` fallback ⇒ `false`.
    pub(crate) const IS_UVM_BACKED: bool = true;

    #[allow(dead_code)]
    pub(crate) enum Backing {
        Cudarc(CudarcUnifiedBuffer),
    }

    impl Backing {
        pub(crate) fn allocate(
            size: usize,
            init_zero_bytes: usize,
        ) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Route through the W1.2 cudarc spike. `CudarcUnifiedBuffer::new`
            // already handles `cuInit` + primary-context retention via the
            // cached `Arc<CudaDevice>` in `cudarc_backend.rs`, so there is no
            // analogue of the cust path's `ensure_cuda_init()` helper here.
            let mut buf = CudarcUnifiedBuffer::new(size)?;
            // cuMemAllocManaged does NOT zero-initialise the returned region,
            // so we zero the Wasm visible window ourselves to match the
            // initial-zero contract of `memory 1` instantiation. Bytes beyond
            // `init_zero_bytes` stay uninitialised; Wasmtime separately zeros
            // any bytes exposed by `memory.grow`.
            let init = init_zero_bytes.min(size);
            if init > 0 {
                buf.as_mut_slice()[..init].fill(0);
            }
            // SAFETY: cudarc returns a non-null aligned pointer on success;
            // `CudarcUnifiedBuffer::new` already verified this and would have
            // returned `UnifiedError::Allocation` otherwise.
            let ptr = NonNull::new(buf.as_ptr() as *mut u8)
                .ok_or_else(|| UnifiedError::Allocation("cudarc returned null".into()))?;
            // Cross-tenant data-leak mitigation (audit H2): the CUDA driver's
            // `cuMemAllocManaged` does not zero-initialise the returned region,
            // unlike `cust::memory::UnifiedBuffer::new(&0u8, size)` on the cust
            // path or `vec![0u8; size]` on the heap path. Zero the entire
            // allocation here so every backing presents the same "fresh memory
            // is zero" contract to upstream callers (notably
            // `TensorWasmLinearMemory` and `UnifiedMemoryPool`).
            //
            // SAFETY: `ptr` is non-null and points to exactly `size` valid
            // bytes of managed memory we just allocated; no other thread can
            // hold an alias because we have not yet returned the buffer to the
            // caller.
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0u8, size);
            }
            Ok((ptr, Backing::Cudarc(buf)))
        }
    }
}

#[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
mod backing_impl {
    use super::*;

    /// Compile-time constant exposed by [`super::UnifiedBuffer::is_uvm_backed`].
    ///
    /// `false` on the no-CUDA default build: [`Backing::allocate`] returns a
    /// heap `Box<[u8]>` and the prefetch/advise helpers are no-ops. This
    /// branch fires only when BOTH `unified-memory` and `cudarc-backend` are
    /// off — enabling either of the two CUDA-backing features flips the
    /// constant to `true`. See the module-level precedence table for the
    /// three-way gating.
    pub(crate) const IS_UVM_BACKED: bool = false;

    #[allow(dead_code)]
    pub(crate) enum Backing {
        Host(Box<[u8]>),
    }

    impl Backing {
        pub(crate) fn allocate(
            size: usize,
            _init_zero_bytes: usize,
        ) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Allocate a zeroed boxed slice; this serves the no-CUDA path.
            // `vec![0u8; size]` already zeroes the entire allocation in one
            // `memset`-equivalent call (and on this branch there is no GPU
            // page-fault round trip to amortise against), so the
            // `_init_zero_bytes` distinction is irrelevant: we keep the
            // whole-slab zero-init regardless. The parameter is accepted to
            // match the cust/cudarc signatures.
            let mut boxed: Box<[u8]> = vec![0u8; size].into_boxed_slice();
            let ptr = NonNull::new(boxed.as_mut_ptr())
                .ok_or_else(|| UnifiedError::Allocation("Box returned null".into()))?;
            Ok((ptr, Backing::Host(boxed)))
        }
    }
}

use backing_impl::{Backing, IS_UVM_BACKED};

impl UnifiedBuffer {
    /// Allocate a new unified buffer of `size` bytes on the default device.
    ///
    /// The full allocation is zero-initialised. For large allocations where
    /// only a subset of bytes is observed before being written by the caller,
    /// prefer [`Self::new_with_visible_window_on`], which limits the
    /// zero-fill to a caller-supplied prefix and leaves the rest uninitialised
    /// (skipping a per-element memset on the cust path).
    pub fn new(size: usize) -> Result<Self, UnifiedError> {
        Self::new_on(size, DeviceId::default())
    }

    /// Allocate a new unified buffer of `size` bytes on the named device.
    ///
    /// The full allocation is zero-initialised — see [`Self::new`] for the
    /// rationale and the partial-zero alternative.
    pub fn new_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        // Zero the whole allocation to preserve historical semantics. Callers
        // that can scope the zero-fill to a smaller prefix should use
        // `new_with_visible_window_on` directly.
        Self::new_with_visible_window_on(size, size, device_id)
    }

    /// Allocate `size` bytes on the named device, zeroing only the first
    /// `visible_bytes` (clamped to `size`).
    ///
    /// This is the per-Wasm-spawn optimisation path: under
    /// `TensorWasmLinearMemory`'s option-(a) preallocate-at-max strategy the
    /// total allocation can reach hundreds of megabytes
    /// ([`crate::wasm_memory::DEFAULT_MAX_BYTES`] is 256 MiB), but the Wasm
    /// spec only requires the initial *minimum* window to read as zero —
    /// Wasmtime separately zero-fills any bytes later exposed by
    /// `memory.grow`. Restricting the up-front memset to `visible_bytes`
    /// drops the per-spawn cost from O(cap) to O(minimum).
    ///
    /// The cust path (`unified-memory`) routes through
    /// `cust::memory::UnifiedBuffer::uninitialized` + a bounded `fill(0)`;
    /// the cudarc path zeros the same window after `cuMemAllocManaged` (which
    /// does not zero-init); the host `Box<[u8]>` fallback ignores
    /// `visible_bytes` and zero-fills via `vec![0u8; size]` because the
    /// no-CUDA build has no large-allocation cost concern.
    pub fn new_with_visible_window_on(
        size: usize,
        visible_bytes: usize,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        let (ptr, backing) = Backing::allocate(size, visible_bytes)?;
        Ok(Self {
            ptr,
            size,
            device_id,
            backing,
            tenant_ctx: None,
        })
    }

    /// Allocate `size` bytes on the named device, consulting the
    /// tenant's GPU memory cap before touching the underlying CUDA
    /// driver.
    ///
    /// Roadmap feature #8 path: this is the tenant-aware analogue of
    /// [`Self::new_on`]. The lifecycle is:
    ///
    /// 1. Call [`TenantContext::consume_gpu_bytes`] for `size` bytes.
    ///    On `Err(GpuMemoryExhausted)` return the structured error
    ///    untouched so the caller can convert it into a `4xx` response
    ///    body without scraping a message string. No CUDA driver call
    ///    happens on the rejection path — important because the only
    ///    realistic in-process recovery is to fail fast.
    /// 2. Allocate the underlying [`Backing`]. If the driver itself
    ///    fails (OOM at the cuMemAllocManaged level, ZeroSize, etc.),
    ///    we **must** undo the `consume_gpu_bytes` so the counter does
    ///    not drift above true utilisation. Failure to release here
    ///    would let a tenant's `gpu_bytes_in_use` ratchet up past the
    ///    cap on repeated driver-OOM and the cap would deny later
    ///    legitimate allocations.
    /// 3. Stash the `Arc<TenantContext>` on the buffer so `Drop` can
    ///    call [`TenantContext::release_gpu_bytes`]. This is the
    ///    release half of the accounting; the CAS-loop in
    ///    `release_gpu_bytes` saturates on underflow, so a Drop after
    ///    an extraordinary release path (e.g. process shutdown) is
    ///    bookkeeping-safe.
    ///
    /// v0.3.7 record-only contract: the CUDA driver itself never sees
    /// the cap until v0.4 wires `cuMemPoolSetAttribute`. See
    /// `docs/GPU-QUOTAS.md`.
    pub fn new_on_with_tenant_context(
        size: usize,
        device_id: DeviceId,
        tenant_ctx: Arc<TenantContext>,
    ) -> Result<Self, tensor_wasm_core::error::TensorWasmError> {
        Self::new_with_visible_window_on_with_tenant_context(size, size, device_id, tenant_ctx)
    }

    /// Tenant-aware variant of [`Self::new_with_visible_window_on`].
    ///
    /// Consults the tenant's GPU memory cap before allocating; on a cap
    /// violation returns
    /// [`tensor_wasm_core::error::TensorWasmError::GpuMemoryExhausted`]
    /// with the requested-vs-limit-vs-current triple, with no driver
    /// call performed. On a successful allocation the resulting
    /// [`UnifiedBuffer`]'s `Drop` returns `size` bytes to the tenant
    /// via [`TenantContext::release_gpu_bytes`].
    pub fn new_with_visible_window_on_with_tenant_context(
        size: usize,
        visible_bytes: usize,
        device_id: DeviceId,
        tenant_ctx: Arc<TenantContext>,
    ) -> Result<Self, tensor_wasm_core::error::TensorWasmError> {
        // Caller-bug guard: zero-byte allocations are rejected upstream
        // by `Backing::allocate`, but we also do not want to bump the
        // tenant counter for a request we are about to refuse.
        if size == 0 {
            return Err(UnifiedError::ZeroSize.into());
        }
        // Step 1: reserve against the cap (or counter-only when no cap).
        tenant_ctx.consume_gpu_bytes(size as u64)?;
        // Step 2: hand off to the legacy constructor. On driver failure
        // we must roll back the `consume_gpu_bytes` step, otherwise the
        // counter drifts above the real utilisation. Mapping
        // `UnifiedError` → `TensorWasmError` is the existing
        // `impl From<UnifiedError>` at the bottom of this module.
        match Backing::allocate(size, visible_bytes) {
            Ok((ptr, backing)) => Ok(Self {
                ptr,
                size,
                device_id,
                backing,
                tenant_ctx: Some(tenant_ctx),
            }),
            Err(e) => {
                tenant_ctx.release_gpu_bytes(size as u64);
                Err(e.into())
            }
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if zero-length. Always false for a successfully constructed buffer.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Raw pointer to the first byte. Used by FFI/`MemoryCreator` in S5.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr() as *const u8
    }

    /// Mutable raw pointer to the first byte.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Borrow the buffer as a shared byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null and points to `size` valid bytes by the type invariant.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Borrow the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null, points to `size` valid bytes, and `&mut self` proves uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Which device this buffer is anchored to.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Attempt to grow the buffer in place to `new_size` bytes.
    ///
    /// **Status: scaffolded, not yet implemented.** Always returns
    /// [`UnifiedError::Cuda`] with the documented `"in-place grow not
    /// yet wired"` sentinel until the v0.4 cutover PR lands the real
    /// `cuMemAddressReserve` + `cuMemMap` path. Until then,
    /// [`crate::TensorWasmLinearMemory`] continues to follow the B5
    /// option-(a) behavior: pre-allocate at `max-pages` and grow_to
    /// is a logical-size bump up to the cap. See B5's
    /// `grow_up_to_preallocated_cap_succeeds_beyond_fails` test for
    /// the current contract.
    ///
    /// # Background
    ///
    /// `cuMemAllocManaged` returns a fixed-size allocation; there is
    /// no in-place grow. The CUDA Driver API alternative is:
    ///
    /// 1. `cuMemAddressReserve(size = max-pages, align)` — reserve a
    ///    virtual address window large enough for any future grow.
    /// 2. `cuMemCreate(handle, initial_size, prop)` — allocate the
    ///    initial physical backing.
    /// 3. `cuMemMap(va, initial_size, handle)` — map physical to
    ///    virtual.
    /// 4. `cuMemSetAccess(va, initial_size, ReadWrite)` — grant the
    ///    current device permission.
    /// 5. To grow: `cuMemCreate(handle_more, delta, prop)` +
    ///    `cuMemMap(va + initial_size, delta, handle_more)` +
    ///    `cuMemSetAccess(va, new_size, ReadWrite)`.
    ///
    /// This is a v0.4 follow-up because:
    ///
    /// - The `cuMemAddressReserve` family is in `cust::sys` / cudarc
    ///   `sys::lib()` but neither crate has a safe wrapper, so the
    ///   implementation is ~300-500 LOC of careful `unsafe`.
    /// - The path requires the GPU's
    ///   [`concurrentManagedAccess`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-concurrent-access)
    ///   device attribute. Consumer Turing/Ampere cards in Windows
    ///   WDDM mode do NOT expose this — same limitation that makes
    ///   the W5.9 `cuMemPrefetchAsync` smoke test the one failure in
    ///   the cudarc set. The v0.4 implementation needs a Linux
    ///   datacenter GPU (the S22 self-hosted runner from C1) to
    ///   verify, which doesn't exist yet.
    /// - cuda-oxide v0.2 may ship a higher-level virtual-memory
    ///   wrapper that obviates the bare driver API entirely, per
    ///   RFC 0001's "cuda-oxide host crates" inventory; waiting one
    ///   release cycle may save us the work.
    ///
    /// # Why scaffold this now
    ///
    /// So the v0.4 author has a concrete target signature + the four
    /// known constraints listed above, rather than starting from a
    /// blank canvas. The stub also lets callers (notably
    /// `TensorWasmLinearMemory::grow_to`) feature-detect via the
    /// returned error string instead of branching on `cfg(feature =
    /// "in-place-grow")` ahead of time.
    pub fn try_grow_in_place(&mut self, _new_size: usize) -> Result<(), UnifiedError> {
        Err(UnifiedError::Cuda(
            "in-place grow not yet wired -- see UnifiedBuffer::try_grow_in_place doc + \
             RFC 0001 v0.4 follow-up. Until then TensorWasmLinearMemory uses the B5 \
             option-(a) preallocate-at-max strategy."
                .into(),
        ))
    }

    /// Whether [`Self::try_grow_in_place`] is implemented on this
    /// build. Currently always `false`; flips to `true` when the v0.4
    /// `cuMemAddressReserve` path lands.
    ///
    /// Callers (mainly `TensorWasmLinearMemory::grow_to`) probe this
    /// to pick between in-place-grow + max-preallocate strategies
    /// without scraping the error string.
    pub const fn supports_in_place_grow() -> bool {
        false
    }

    /// Whether this buffer is backed by CUDA Unified Memory (`cuMemAllocManaged`).
    ///
    /// Returns `true` when the crate was compiled with EITHER
    /// `--features unified-memory` (cust path, the v0.3 default) OR
    /// `--features cudarc-backend` (the W1.2 cudarc spike, used as the
    /// `Backing::Cudarc` variant when `unified-memory` is off). Returns
    /// `false` only on the bare default build where the backing is a heap
    /// `Box<[u8]>`. This is a compile-time property of the active backing
    /// (it does not probe the driver at runtime), and is exposed as a public
    /// probe so callers — including [`crate::wasm_memory::TensorWasmLinearMemory`]
    /// — can assert in tests that the audit-flagged "wasm linear memory not
    /// UVM-backed" gap is actually closed at build configuration time.
    ///
    /// See the module-level precedence table for the full feature-combination
    /// matrix.
    pub fn is_uvm_backed(&self) -> bool {
        IS_UVM_BACKED
    }

    /// Suggest to the runtime that the buffer should be migrated to the device.
    /// No-op when `unified-memory` is disabled.
    ///
    /// **Implementation status:** under `--features unified-memory`, this is
    /// currently an advisory no-op. cust 0.3.2's `MemoryAdvise::prefetch_to_device`
    /// requires a `&Stream` + `&Device` (rather than the bare `i32` ordinal this
    /// signature accepts), and the unified-memory subsystem does not yet thread
    /// a `Stream` through the public surface. On Windows WDDM the equivalent
    /// `cuMemPrefetchAsync` call returns `CUDA_ERROR_INVALID_DEVICE` anyway
    /// because consumer Turing cards don't expose `concurrentManagedAccess`,
    /// so the user-visible behavior is the same. The `cudarc_backend` path
    /// (see `cudarc_backend.rs`) does call the driver fn directly via
    /// `cudarc::driver::sys::lib().cuMemPrefetchAsync`, where the
    /// platform support story is the same.
    ///
    /// TODO(v0.4): thread a `Stream` through `UnifiedBuffer` and wire this
    /// against `cust::memory::MemoryAdvise::prefetch_to_device(&stream, &device)`.
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        Ok(())
    }

    /// Suggest to the runtime that the buffer should be migrated back to host
    /// memory. Currently an advisory no-op for the same reasons as
    /// [`Self::prefetch_to_device`].
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        Ok(())
    }
}

impl fmt::Debug for UnifiedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnifiedBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        // Tenant-accounting release. Only runs for buffers constructed
        // through [`Self::new_on_with_tenant_context`] (or the
        // visible-window variant); the legacy `new` / `new_on` paths
        // leave `tenant_ctx == None` so this is a single `Option` check
        // on the drop hot path — no atomic, no allocation. The
        // underlying CUDA / heap free runs unconditionally via the
        // `backing` field's own drop.
        if let Some(ctx) = self.tenant_ctx.as_ref() {
            ctx.release_gpu_bytes(self.size as u64);
        }
    }
}

/// Cross-crate conversion so `tensor-wasm-mem` errors flow into the workspace's
/// unified [`TensorWasmError`] type without manual mapping at every call site.
///
/// - `ZeroSize` → `TensorWasmError::Serialization` with a descriptive message
///   (zero-byte allocations are caller bugs, not memory exhaustion).
/// - `Allocation { .. }` → `TensorWasmError::MemoryExhausted { requested: 0, limit: 0 }`
///   when the message contains "exhausted", surfacing exhaustion as a
///   first-class error variant (placeholder numeric fields are used because
///   `UnifiedError::Allocation` carries no structured size info — the original
///   string is lost in this mapping; richer mapping is a TODO once
///   `UnifiedError::Allocation` is split into structured variants). Otherwise
///   the variant falls through to `TensorWasmError::Serialization` carrying the
///   detail string.
/// - `Cuda { .. }` → `TensorWasmError::CudaError` (1:1 mapping).
///
/// [`TensorWasmError`]: tensor_wasm_core::error::TensorWasmError
impl From<UnifiedError> for tensor_wasm_core::error::TensorWasmError {
    fn from(e: UnifiedError) -> Self {
        match e {
            UnifiedError::ZeroSize => tensor_wasm_core::error::TensorWasmError::Serialization(
                "unified buffer: zero-byte allocation rejected".into(),
            ),
            UnifiedError::Allocation(msg) => {
                if msg.contains("exhausted") {
                    // TODO: when `UnifiedError::Allocation` grows structured
                    // {requested, capacity} fields, plumb them through here
                    // instead of losing the detail in placeholder zeros.
                    tensor_wasm_core::error::TensorWasmError::MemoryExhausted {
                        requested: 0,
                        limit: 0,
                    }
                } else {
                    tensor_wasm_core::error::TensorWasmError::Serialization(
                        format!("unified buffer allocation failed: {msg}").into(),
                    )
                }
            }
            UnifiedError::Cuda(msg) => tensor_wasm_core::error::TensorWasmError::CudaError(msg.into()),
            UnifiedError::TooLarge { requested, limit } => {
                tensor_wasm_core::error::TensorWasmError::MemoryExhausted { requested, limit }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_round_trip() {
        let mut b = UnifiedBuffer::new(64).expect("alloc");
        assert_eq!(b.len(), 64);
        assert!(!b.is_empty());
        b.as_mut_slice().copy_from_slice(&[7u8; 64]);
        assert!(b.as_slice().iter().all(|&v| v == 7));
    }

    #[test]
    fn zero_size_rejected() {
        let err = UnifiedBuffer::new(0).expect_err("zero should be rejected");
        assert!(matches!(err, UnifiedError::ZeroSize));
    }

    #[test]
    fn try_grow_in_place_returns_documented_sentinel() {
        // D1 scaffold contract: until v0.4 lands the
        // cuMemAddressReserve + cuMemMap path, try_grow_in_place
        // returns the documented sentinel and supports_in_place_grow()
        // returns false. Callers that branch on the feature gate get
        // the same answer either way.
        assert!(!UnifiedBuffer::supports_in_place_grow());
        // On the no-feature build we can construct the buffer to
        // probe the API; on feature builds the same probe works
        // because allocate succeeds. The error string is the public
        // contract callers (TensorWasmLinearMemory::grow_to v0.4) match on.
        let mut b = UnifiedBuffer::new(64).expect("alloc");
        let err = b.try_grow_in_place(128).expect_err("scaffold must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("in-place grow not yet wired"),
            "sentinel string changed; v0.4 caller match site must be updated: {msg}",
        );
    }

    #[test]
    fn device_id_default_is_zero() {
        let b = UnifiedBuffer::new(8).unwrap();
        assert_eq!(b.device_id(), DeviceId(0));
    }

    #[test]
    fn device_id_display_format() {
        assert_eq!(DeviceId(3).to_string(), "cuda:3");
    }

    #[test]
    fn prefetch_no_op_without_cuda() {
        // Calling prefetch should be safe even without the unified-memory feature.
        let b = UnifiedBuffer::new(32).unwrap();
        b.prefetch_to_device().expect("no-op should succeed");
        b.prefetch_to_host().expect("no-op should succeed");
    }

    #[test]
    fn pointers_are_non_null_and_consistent() {
        let mut b = UnifiedBuffer::new(16).unwrap();
        let p1 = b.as_ptr();
        let p2 = b.as_mut_ptr() as *const u8;
        assert_eq!(p1, p2);
        assert!(!p1.is_null());
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_zero_size() {
        let e = UnifiedError::ZeroSize;
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(b, tensor_wasm_core::error::TensorWasmError::Serialization(_)));
        assert!(b.to_string().contains("zero-byte"));
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_cuda() {
        let e = UnifiedError::Cuda("ctx not current".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        match b {
            tensor_wasm_core::error::TensorWasmError::CudaError(s) => assert_eq!(&*s, "ctx not current"),
            other => panic!("expected CudaError, got {other:?}"),
        }
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_allocation_exhausted() {
        let e = UnifiedError::Allocation("pool exhausted: need 1024 bytes".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(
            b,
            tensor_wasm_core::error::TensorWasmError::MemoryExhausted { .. }
        ));
    }

    #[test]
    #[cfg(feature = "unified-memory")]
    fn is_uvm_backed_true_under_feature() {
        // Closes the v0.3.2 audit gap (Problem #5): when the `unified-memory`
        // Cargo feature is on, `UnifiedBuffer` must report it routes through
        // `cuMemAllocManaged`. This is the compile-time guarantee that the
        // `TensorWasmLinearMemory` zero-copy promise rests on.
        let b = UnifiedBuffer::new(64).expect("alloc under feature");
        assert!(b.is_uvm_backed(), "unified-memory build must use UVM backing");
    }

    #[test]
    #[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
    fn is_uvm_backed_false_without_feature() {
        // Without either CUDA backing feature, the backing is a heap
        // `Box<[u8]>`. This test pins the inverse half of the contract so a
        // future regression that accidentally turns the probe into a
        // runtime-always-true cannot sneak past CI's no-feature build.
        let b = UnifiedBuffer::new(64).expect("alloc without feature");
        assert!(!b.is_uvm_backed(), "no-feature build must use heap backing");
    }

    #[test]
    #[cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]
    fn is_uvm_backed_true_under_cudarc_backend() {
        // Mirrors `is_uvm_backed_true_under_feature` for the cudarc path.
        // When `--features cudarc-backend` is on (and `unified-memory` is
        // off), `UnifiedBuffer` routes through `cuMemAllocManaged` via
        // cudarc per the module-level precedence rule, so the probe must
        // report `true`. NOTE: this test allocates a real CUDA buffer and
        // therefore requires a working driver at test time. Use the smoke
        // test under `tests/cudarc_unified_buffer_smoke.rs` for the same
        // contract at integration-test scope.
        let b = UnifiedBuffer::new(64).expect("alloc under cudarc-backend");
        assert!(
            b.is_uvm_backed(),
            "cudarc-backend build must use UVM backing"
        );
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_allocation_non_exhausted() {
        let e = UnifiedError::Allocation("minimum 1024 > maximum 512".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(b, tensor_wasm_core::error::TensorWasmError::Serialization(_)));
        assert!(b.to_string().contains("minimum 1024"));
    }
}
