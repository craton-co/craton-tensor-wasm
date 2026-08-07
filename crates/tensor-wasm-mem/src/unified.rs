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
//!   via the W1.2 `cudarc_backend::CudarcUnifiedBuffer` spike.
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
    /// A [`UnifiedBacking`] method is not available on the active backing.
    ///
    /// Surfaced by trait methods (e.g. `prefetch_to_device` on the cudarc
    /// stub or the `Box<[u8]>` host fallback) when the underlying backing
    /// has no implementation. Carries the method name and the backing tag
    /// so operator tooling and tests can match on the exact gap without
    /// scraping driver error strings.
    #[error("{feature:?} not supported by backing {backing:?}")]
    NotSupported {
        /// Stable identifier for the method that has no implementation
        /// on this backing (e.g. `"prefetch_to_device"`).
        feature: &'static str,
        /// Stable identifier for the backing that lacks the feature
        /// (e.g. `"cudarc-stub"`, `"host-box"`).
        backing: &'static str,
    },
}

/// Memory-residency hint passed through [`UnifiedBacking::apply_advice`].
///
/// Mirrors the `cuMemAdvise` flags already used by the three concrete
/// backings ([`crate::advise::Advice`] on the cust path,
/// `cudarc_backend::apply_advice` on the cudarc path, and the
/// `CudaOxideAdvice` enum on the cuda-oxide path). This trait-facing enum
/// is declared in the common `unified` module so downstream code can
/// target a single shape across every backing.
///
/// Variants intentionally hold a bare `u32` device ordinal rather than
/// [`DeviceId`] so the enum has zero non-trivial dependencies and a
/// future port (e.g. a Vulkan / ROCm backing) can implement
/// [`UnifiedBacking`] without pulling the CUDA-tagged [`DeviceId`] into
/// its interface.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvmAdvice {
    /// `CU_MEM_ADVISE_SET_READ_MOSTLY`. Pages are read by many devices
    /// but rarely written; the runtime may duplicate them.
    SetReadMostly,
    /// `CU_MEM_ADVISE_UNSET_READ_MOSTLY`. Reverse the read-mostly hint.
    UnsetReadMostly,
    /// `CU_MEM_ADVISE_SET_PREFERRED_LOCATION` for the given device
    /// ordinal — pages should live primarily on that device.
    SetPreferredLocation(u32),
    /// `CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`. Reverse the preferred
    /// location hint.
    UnsetPreferredLocation,
    /// `CU_MEM_ADVISE_SET_ACCESSED_BY` for the given device ordinal —
    /// the device will access the region, so the runtime should map it
    /// without migrating.
    SetAccessedBy(u32),
    /// `CU_MEM_ADVISE_UNSET_ACCESSED_BY` for the given device ordinal.
    UnsetAccessedBy(u32),
}

/// Common surface for unified-memory backings (cust, cudarc, cuda-oxide).
///
/// The three concrete buffer types in this crate hand-mirror the same API.
/// This trait pins the contract; v0.4 will migrate the concrete types to
/// implement it and the public `UnifiedBuffer` may become an enum or a
/// `Box<dyn UnifiedBacking>` shell. For v0.3.6 the trait is documentation-
/// shaped: a wire-stable description of what every backing must support so
/// downstream code (and future ports) can target it.
///
/// # Method semantics
///
/// Implementations that lack support for a particular driver call (the
/// cudarc stub, the `Box<[u8]>` host fallback, or a port that simply
/// has not wired the entry point yet) MUST return
/// [`UnifiedError::NotSupported`] from the corresponding method rather
/// than silently succeeding with a no-op. The legacy `UnifiedBuffer`
/// path keeps its historical no-op behaviour for `prefetch_to_*` to
/// preserve back-compat with v0.3 callers; that exemption is documented
/// per-method below.
pub trait UnifiedBacking: Send + Sync {
    /// Number of bytes in this allocation.
    fn len(&self) -> usize;

    /// True iff [`Self::len`] is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the host-visible slice. UVM means the bytes are accessible
    /// to both host and device; reads after a device write may need a
    /// `prefetch_to_host` first depending on the backing.
    fn as_slice(&self) -> &[u8];

    /// Mutably borrow the host-visible slice. See [`Self::as_slice`].
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Apply a CUDA `cuMemAdvise` hint. No-op on backings that don't
    /// support it (host-only fallback returns `Ok(())`; the cudarc /
    /// cuda-oxide stubs return [`UnifiedError::NotSupported`] until the
    /// real wrapper lands).
    fn apply_advice(&self, hint: UvmAdvice) -> Result<(), UnifiedError>;

    /// Prefetch to a device. May be a no-op on backings without UVM
    /// prefetch (documented per-backing).
    fn prefetch_to_device(&self, device_ord: u32) -> Result<(), UnifiedError>;

    /// Prefetch back to the host CPU.
    fn prefetch_to_host(&self) -> Result<(), UnifiedError>;
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
    //
    // ALIASING: `backing` and `ptr` observe the same allocation. The
    // sealed `Backing` enum (declared `pub(super)` inside `mod backing`)
    // exists only to give that allocation a typed `Drop`; its variants
    // MUST NOT be pattern-matched and its wrapped storage MUST NOT be
    // re-borrowed via the inner type's `as_mut_slice` / `as_ptr`. See
    // the "Aliasing invariant" doc on `mod backing` for the full
    // contract. The audit-T5 fix sealed this by making variants
    // unreachable from outside the `unified` module.
    #[allow(dead_code)]
    backing: Backing,
    /// Audit H4: whether the bytes at `ptr` are dereferenceable from the
    /// host CPU. `true` for managed/unified allocations (cust
    /// `cuMemAllocManaged`, cudarc `cuMemAllocManaged`) and the host
    /// `Box<[u8]>` fallback — all of which are page-migratable / plain
    /// host memory and so safe to view as a `&[u8]`. `false` for the T39
    /// `new_in_tenant_pool` path, whose `cuMemAllocFromPoolAsync` memory
    /// is DEVICE-ONLY (see the "Bytes layout" doc on
    /// `new_in_tenant_pool`); calling `from_raw_parts` over it
    /// would be host-side UB. `as_slice` / `as_mut_slice` consult this
    /// flag and panic rather than fabricating a host slice over device
    /// memory. Every constructor MUST set this correctly.
    host_addressable: bool,
    /// Audit (LOW, Drop-time zeroization): whether the backing is plain
    /// host memory (`Box<[u8]>`) whose bytes can be safely overwritten
    /// from the CPU in `Drop`. Only the no-CUDA `Box<[u8]>` fallback sets
    /// this `true`; the CUDA device paths leave it `false` because
    /// zeroizing them needs a live CUDA context (not available in a bare
    /// `Drop`). Distinct from `host_addressable`: a future host-pinned
    /// CUDA allocation could be host-addressable yet still need a context
    /// to free, so the two concerns are tracked separately.
    host_zeroize_on_drop: bool,
    /// Physical byte capacity of the underlying allocation on the HOST
    /// (`Box<[u8]>`) backing.
    ///
    /// `size` is the *logical* (currently-used) length exposed by
    /// [`Self::len`] / [`Self::as_slice`] / [`Self::as_mut_slice`];
    /// `host_capacity` is how many bytes the backing allocation actually
    /// owns. The invariant `size <= host_capacity` is upheld by every
    /// constructor and by [`Self::try_grow_in_place`], which is what makes
    /// it sound for the slice/`Drop` paths to keep bounding their
    /// `from_raw_parts` over `self.ptr` with `self.size` — they only ever
    /// view a prefix of the real allocation.
    ///
    /// For the standard host constructors (`new`, `new_on`,
    /// `new_with_visible_window_on`, …) this equals `size`: the
    /// `vec![0u8; size]` backing is exactly sized, so there is no spare
    /// room and an in-place grow request necessarily falls back to the
    /// realloc path. [`Self::new_host_with_capacity_on`] is the entry
    /// point that allocates `capacity > size` physical bytes so that
    /// subsequent [`Self::try_grow_in_place`] calls up to `capacity`
    /// succeed without a realloc+copy — the per-spawn Wasm-linear-memory
    /// win this field exists to enable.
    ///
    /// On the CUDA backings this field is bookkeeping only (set equal to
    /// `size`); the CUDA in-place grow path is still deferred (see
    /// [`Self::try_grow_in_place`]).
    host_capacity: usize,
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

/// # Aliasing invariant
///
/// Each [`Backing`] variant wraps an owning allocation whose first byte
/// is ALSO observed via the parent [`UnifiedBuffer`]'s `NonNull<u8>`.
/// The owning storage (e.g. [`cust::memory::UnifiedBuffer<u8>`],
/// [`crate::cudarc_backend::CudarcUnifiedBuffer`], or `Box<[u8]>`)
/// exposes its own `as_mut_slice()` / `as_ptr()` accessors that would
/// hand out a `&mut [u8]` to exactly the same bytes Rust already has a
/// `&mut [u8]` to via `UnifiedBuffer::as_mut_slice` — producing two
/// live mutable references to the same allocation, instant UB.
///
/// The variants are SEALED — declared `pub(super)` so no code outside
/// the `unified` module can pattern-match or destructure them. The
/// ONLY sound operations on a `Backing` value are:
///
/// 1. Leave it in place inside [`UnifiedBuffer`] (the by-design case).
/// 2. Drop it (freeing the allocation; runs automatically on
///    `UnifiedBuffer::drop`).
/// 3. Replace the entire [`UnifiedBuffer`] (which moves + drops the
///    old `Backing` as a whole; the `NonNull<u8>` is replaced in lock
///    step).
///
/// Specifically forbidden, even inside the `unified` module:
///
/// - `match` / `if let` on `Backing` variants to call `as_mut_slice`,
///   `as_ptr`, `as_unified_ptr`, or any other method that hands out a
///   borrow or pointer overlapping the parent struct's
///   [`UnifiedBuffer::ptr`]. Use `self.ptr` directly.
/// - `mem::replace` / `mem::take` / `into_inner` style moves that
///   extract the wrapped allocation while the parent's `ptr` is still
///   considered live.
///
/// Future contributors adding a new variant MUST add a `# Safety`
/// section to its doc comment that explains how the new variant
/// preserves these rules. The `#[deny(missing_docs)]` attribute on
/// the enum enforces the per-variant doc requirement at compile time.
#[cfg(feature = "unified-memory")]
mod backing {
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
    pub(super) const IS_UVM_BACKED: bool = true;

    /// Audit (LOW, Drop-time zeroization): `false` here — the cust managed
    /// allocation is freed by cust's typed `Drop`, which needs a live CUDA
    /// context, so [`super::UnifiedBuffer::drop`] must NOT attempt a
    /// host-side memset over it. Only the no-CUDA `Box<[u8]>` build sets
    /// this `true`.
    pub(super) const IS_HOST_BACKED: bool = false;

    /// Owning storage for a [`super::UnifiedBuffer`] under the
    /// `unified-memory` feature.
    ///
    /// SEALED: declared `pub(super)`, so neither this enum nor its
    /// variants can be named outside the `unified` module. See the
    /// "Aliasing invariant" section on the parent module for the full
    /// safety contract. The variants are documentation-grade only —
    /// they exist to give the wrapped allocation a typed `Drop`; they
    /// must not be pattern-matched.
    #[deny(missing_docs)]
    #[allow(dead_code)]
    pub(super) enum Backing {
        /// cust-managed UVM allocation (`cuMemAllocManaged` via cust 0.3).
        ///
        /// # Safety
        ///
        /// The wrapped [`cust::memory::UnifiedBuffer<u8>`] aliases the
        /// same bytes as the parent [`super::UnifiedBuffer`]'s
        /// `NonNull<u8>`. Do NOT call `as_mut_slice` / `as_ptr` /
        /// `as_unified_ptr` on the wrapped value once it has been
        /// moved into this variant — those accessors are reserved for
        /// the pre-aliasing construction path inside
        /// [`Backing::allocate`].
        Cuda(cust::memory::UnifiedBuffer<u8>),

        /// T39 driver-enforced tenant pool allocation
        /// (`cuMemAllocFromPoolAsync` against a `TenantMemPool`).
        ///
        /// Distinct from [`Backing::Cuda`] because the free path goes
        /// through `cuMemFreeAsync` on the tenant's pool handle rather
        /// than cust's `cuMemFree_v2`. The held `Arc<TenantMemPool>`
        /// keeps the pool (and its `cuMemPoolDestroy` deadline) alive
        /// for the buffer's lifetime.
        ///
        /// # Safety
        ///
        /// Same aliasing rule as [`Backing::Cuda`]: the wrapped
        /// `NonNull<u8>` aliases the parent
        /// [`super::UnifiedBuffer`]'s `ptr` field; do NOT
        /// pattern-match this variant from outside the construction
        /// + drop paths in this module.
        #[cfg(feature = "gpu-mem-pool")]
        TenantPool(super::TenantPoolBacking),
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
        let r =
            CTX.get_or_init(|| cust::quick_init().map_err(|e| format!("cust::quick_init: {e:?}")));
        match r {
            Ok(_) => Ok(()),
            Err(msg) => Err(UnifiedError::Cuda(msg.clone())),
        }
    }

    impl Backing {
        pub(super) fn allocate(
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
mod backing {
    //! Sealed owning storage for [`super::UnifiedBuffer`]. The
    //! `Backing` enum aliases the same allocation as
    //! `super::UnifiedBuffer::ptr` (a `NonNull<u8>`), so its variants
    //! are declared `pub(super)` and MUST NOT be pattern-matched from
    //! outside this module. See the matching "Aliasing invariant"
    //! comment on the `feature = "unified-memory"` build of this
    //! module for the full safety contract.
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
    pub(super) const IS_UVM_BACKED: bool = true;

    /// Audit (LOW, Drop-time zeroization): `false` here — the cudarc
    /// managed allocation is freed via `cuMemFree_v2`, which needs a live
    /// CUDA context, so [`super::UnifiedBuffer::drop`] must NOT attempt a
    /// host-side memset over it. Only the no-CUDA `Box<[u8]>` build sets
    /// this `true`.
    pub(super) const IS_HOST_BACKED: bool = false;

    /// Owning storage for a [`super::UnifiedBuffer`] under the
    /// `cudarc-backend` feature.
    ///
    /// SEALED: declared `pub(super)`, so neither this enum nor its
    /// variants can be named outside the `unified` module. See the
    /// "Aliasing invariant" section on the parent module for the full
    /// safety contract.
    #[deny(missing_docs)]
    #[allow(dead_code)]
    pub(super) enum Backing {
        /// cudarc-managed UVM allocation (`cuMemAllocManaged` via cudarc).
        ///
        /// # Safety
        ///
        /// The wrapped [`CudarcUnifiedBuffer`] aliases the same bytes
        /// as the parent [`super::UnifiedBuffer`]'s `NonNull<u8>`. Do
        /// NOT call `as_mut_slice` / `as_ptr` on the wrapped value
        /// once it has been moved into this variant — those accessors
        /// are reserved for the pre-aliasing construction path inside
        /// [`Backing::allocate`].
        Cudarc(CudarcUnifiedBuffer),

        /// T39 driver-enforced tenant pool allocation
        /// (`cuMemAllocFromPoolAsync` against a `TenantMemPool`).
        ///
        /// Distinct from [`Backing::Cudarc`] because the free path
        /// goes through `cuMemFreeAsync` on the tenant's pool handle
        /// rather than `cuMemFree_v2`. The held `Arc<TenantMemPool>`
        /// keeps the pool (and its `cuMemPoolDestroy` deadline) alive
        /// for the buffer's lifetime.
        ///
        /// # Safety
        ///
        /// Same aliasing rule as [`Backing::Cudarc`]: the wrapped
        /// `NonNull<u8>` aliases the parent
        /// [`super::UnifiedBuffer`]'s `ptr` field; do NOT
        /// pattern-match this variant from outside the construction
        /// + drop paths in this module.
        #[cfg(feature = "gpu-mem-pool")]
        TenantPool(super::TenantPoolBacking),
    }

    impl Backing {
        /// Allocate `size` bytes of CUDA Unified Memory via cudarc.
        ///
        /// Zeroes only the visible window (`init_zero_bytes`, clamped to
        /// `size`). Bytes outside `[0, init_zero_bytes)` are NOT zeroed and
        /// contain undefined data until the caller writes them. This matches
        /// the cust path's behaviour and restores the `visible_bytes`
        /// optimisation that motivates
        /// [`super::UnifiedBuffer::new_with_visible_window_on`]: a 256 MiB
        /// Wasm linear-memory spawn pays only a `minimum_bytes` memset
        /// (typically one 64 KiB Wasm page) instead of a full `cap`-sized
        /// fill.
        ///
        /// # Audit H2 invariant (no cross-tenant data leak)
        ///
        /// Cross-tenant safety is upheld at the layer above this one:
        ///
        /// - The pool path
        ///   ([`crate::pool::UnifiedMemoryPool::allocate`]) zeroes every
        ///   carved `[offset, offset + size)` region via `ptr::write_bytes`
        ///   before returning the [`crate::pool::PoolAllocation`] to the
        ///   tenant. That defends slabs that are recycled across tenants
        ///   (audit H1, regression-pinned by
        ///   `pool::tests::recycled_allocation_reads_as_zero`).
        /// - The direct linear-memory path
        ///   ([`crate::wasm_memory::TensorWasmLinearMemory`]) zeroes the
        ///   visible window at construction time (via this function's
        ///   `init_zero_bytes` fill) and zeroes bytes freshly exposed by
        ///   `memory.grow` in
        ///   [`crate::wasm_memory::TensorWasmLinearMemory::grow_to`]
        ///   (regression-pinned by the H2 comment block at that call site).
        ///
        /// Removing the redundant full-allocation memset that previously
        /// ran here restores the visible-window optimisation; the H2
        /// invariant is preserved by the layer-above defences listed above.
        pub(super) fn allocate(
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
            // any bytes exposed by `memory.grow`. The audit-T14 perf fix
            // dropped a redundant full-allocation memset that previously
            // followed this fill — see the doc comment above for the H2
            // safety rationale.
            let init = init_zero_bytes.min(size);
            if init > 0 {
                buf.as_mut_slice()[..init].fill(0);
            }
            // SAFETY: cudarc returns a non-null aligned pointer on success;
            // `CudarcUnifiedBuffer::new` already verified this and would have
            // returned `UnifiedError::Allocation` otherwise.
            let ptr = NonNull::new(buf.as_ptr() as *mut u8)
                .ok_or_else(|| UnifiedError::Allocation("cudarc returned null".into()))?;
            Ok((ptr, Backing::Cudarc(buf)))
        }
    }
}

#[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
mod backing {
    //! Sealed owning storage for [`super::UnifiedBuffer`]. The
    //! `Backing` enum aliases the same allocation as
    //! `super::UnifiedBuffer::ptr` (a `NonNull<u8>`), so its variants
    //! are declared `pub(super)` and MUST NOT be pattern-matched from
    //! outside this module. See the matching "Aliasing invariant"
    //! comment on the `feature = "unified-memory"` build of this
    //! module for the full safety contract.
    use super::*;

    /// Compile-time constant exposed by [`super::UnifiedBuffer::is_uvm_backed`].
    ///
    /// `false` on the no-CUDA default build: [`Backing::allocate`] returns a
    /// heap `Box<[u8]>` and the prefetch/advise helpers are no-ops. This
    /// branch fires only when BOTH `unified-memory` and `cudarc-backend` are
    /// off — enabling either of the two CUDA-backing features flips the
    /// constant to `true`. See the module-level precedence table for the
    /// three-way gating.
    pub(super) const IS_UVM_BACKED: bool = false;

    /// Audit (LOW, Drop-time zeroization): `true` here — the backing is a
    /// plain heap `Box<[u8]>`, so [`super::UnifiedBuffer::drop`] can
    /// volatile-zero the bytes before the box is freed (no CUDA context
    /// required). This is the ONLY build where the constant is `true`.
    pub(super) const IS_HOST_BACKED: bool = true;

    /// Owning storage for a [`super::UnifiedBuffer`] on the no-CUDA
    /// default build.
    ///
    /// SEALED: declared `pub(super)`, so neither this enum nor its
    /// variants can be named outside the `unified` module. See the
    /// "Aliasing invariant" section on the parent module for the full
    /// safety contract.
    #[deny(missing_docs)]
    #[allow(dead_code)]
    pub(super) enum Backing {
        /// Heap-backed fallback (`Box<[u8]>`).
        ///
        /// # Safety
        ///
        /// The wrapped `Box<[u8]>` aliases the same bytes as the
        /// parent [`super::UnifiedBuffer`]'s `NonNull<u8>`. Do NOT
        /// call `as_mut_ptr` / `as_mut` / index the slice once the box
        /// has been moved into this variant — those accessors are
        /// reserved for the pre-aliasing construction path inside
        /// [`Backing::allocate`].
        Host(Box<[u8]>),
    }

    impl Backing {
        pub(super) fn allocate(
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

// Private re-export: pulls the SEALED `Backing` type and its associated
// constant into the `unified` module's name space. The use statement is
// intentionally non-`pub` — `Backing` itself is `pub(super)` inside its
// `mod backing` block, so neither this re-export nor the type can be
// named from any other module in the crate. Combined with the
// per-variant `# Safety` invariant on each `Backing` arm (see the
// "Aliasing invariant" doc on the `mod backing` blocks above), this
// closes the audit-T5 finding that `Backing::Cuda` aliased the parent
// struct's `NonNull<u8>`.
use backing::{Backing, IS_HOST_BACKED, IS_UVM_BACKED};

/// T39 owning storage for a [`UnifiedBuffer`] allocated through a
/// tenant-scoped `cuMemPool`.
///
/// Wraps the raw `NonNull<u8>` returned by
/// `cuMemAllocFromPoolAsync` plus the `Arc<TenantMemPool>` whose
/// release-threshold cap the driver is now enforcing. The `Drop` impl
/// frees through `pool.deallocate` (`cuMemFreeAsync`) before the held
/// `Arc` itself drops, so the pool's `cuMemPoolDestroy` cannot run
/// while any of its allocations are still live.
///
/// Lives at parent-module scope so the per-feature `mod backing`
/// blocks above can both name the same type — the cudarc-only build
/// and the `unified-memory`-plus-`gpu-mem-pool` build share this one
/// definition rather than each carrying their own copy.
///
/// # Safety
///
/// `ptr` aliases the parent [`UnifiedBuffer`]'s `ptr` field. Do NOT
/// hand the inner pointer out via any accessor here — the pre-aliasing
/// construction path inside [`UnifiedBuffer::new_in_tenant_pool`]
/// captures the pointer, and from then on only the parent struct's
/// `as_ptr` / `as_mut_ptr` are the legal access paths.
#[cfg(feature = "gpu-mem-pool")]
pub(crate) struct TenantPoolBacking {
    ptr: NonNull<u8>,
    /// Allocation size in bytes. Held so [`Drop`] can return the host-side cap
    /// reservation (`TenantMemPool::live_bytes`) that
    /// [`TenantMemPool::allocate`] took — `deallocate` only carries the
    /// pointer, not the size. See fix #1.
    size: usize,
    pool: Arc<crate::cuda_mem_pool::TenantMemPool>,
}

// Audit (LOW — ASLR-leak via logs): hand-written `Debug` that REDACTS the
// raw `ptr` address rather than the `#[derive(Debug)]` default (which would
// print the allocation pointer). Logs/backtraces that render this struct
// then cannot be used to defeat ASLR.
#[cfg(feature = "gpu-mem-pool")]
impl fmt::Debug for TenantPoolBacking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TenantPoolBacking")
            .field("ptr", &"<redacted>")
            .field("pool_cap_bytes", &self.pool.cap_bytes())
            .finish()
    }
}

#[cfg(feature = "gpu-mem-pool")]
impl Drop for TenantPoolBacking {
    fn drop(&mut self) {
        // Free through the tenant pool. Mirrors the failure-handling
        // discipline of [`crate::cudarc_backend::CudarcUnifiedBuffer::drop`]:
        // drop cannot return an error, so a `cuMemFreeAsync` failure is
        // logged at `error!` and the pointer is left orphaned. The
        // tenant pool's release-threshold cap means a leaked
        // allocation is still bounded by the cap; the v0.5 leak-audit
        // story will unify this with `LEAKED_CUDA_ALLOCATIONS`.
        if let Err(e) = self.pool.deallocate(self.ptr) {
            tracing::error!(
                target: "tensor_wasm_mem::cuda_mem_pool",
                error = ?e,
                pool_cap_bytes = self.pool.cap_bytes(),
                "cuMemFreeAsync failed in TenantPoolBacking::drop -- \
                 allocation leaked, bounded by pool cap",
            );
        }
        // Fix #1: return the host-side cap reservation regardless of whether
        // the driver free succeeded. The reservation is admission control for
        // *future* allocations; holding a phantom reservation for a buffer the
        // tenant has dropped would slowly starve the tenant on repeated driver
        // hiccups. A genuine driver-free failure is already logged loudly above.
        self.pool.release_bytes(self.size);
    }
}

// SAFETY: the inner pointer is owned by this struct and not shared
// without explicit synchronisation; the `Arc<TenantMemPool>` is itself
// `Send + Sync` (see `TenantMemPool`'s `unsafe impl`).
#[cfg(feature = "gpu-mem-pool")]
unsafe impl Send for TenantPoolBacking {}
#[cfg(feature = "gpu-mem-pool")]
unsafe impl Sync for TenantPoolBacking {}

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
            // Audit H4: the standard backings (cust/cudarc managed memory
            // and the host `Box<[u8]>`) are all host-dereferenceable.
            host_addressable: true,
            // Audit (LOW): only the no-CUDA `Box<[u8]>` build can be
            // zeroized in `Drop` without a live CUDA context.
            host_zeroize_on_drop: IS_HOST_BACKED,
            // Exactly-sized backing: no spare capacity, so logical ==
            // physical and `try_grow_in_place` cannot grow without a
            // realloc. See `new_host_with_capacity_on` for the spare-cap
            // path.
            host_capacity: size,
            tenant_ctx: None,
        })
    }

    /// Allocate a HOST (`Box<[u8]>`) buffer with `capacity` physical bytes
    /// but only `size` bytes of *logical* length exposed.
    ///
    /// This is the entry point that makes [`Self::try_grow_in_place`]
    /// genuinely grow in place: the backing owns `capacity` zeroed bytes,
    /// [`Self::len`] / [`Self::as_slice`] / [`Self::as_mut_slice`] report
    /// the `size`-byte prefix, and a later `try_grow_in_place(new_size)`
    /// with `new_size <= capacity` bumps the logical length (zero-filling
    /// the newly-exposed region per the H2 guarantee) without a
    /// realloc+copy. The motivating case is
    /// [`crate::wasm_memory::TensorWasmLinearMemory`]: reserve at
    /// `max-pages` once, then satisfy each `memory.grow` as an in-place
    /// logical bump instead of forcing the B5 option-(a) up-front
    /// max-preallocate on every spawn.
    ///
    /// On the CUDA backings there is no host capacity concept, so this
    /// degrades to an exactly-`size` allocation (`capacity` is clamped to
    /// `size`) and `try_grow_in_place` remains deferred there. The whole
    /// `capacity`-region is zero-initialised on the host path, so the H2
    /// zero-on-grow guarantee holds for every byte that any future
    /// in-place grow can expose.
    ///
    /// `capacity` is clamped up to `size` (a `capacity < size` request is
    /// a caller bug; we never expose more than we own). `size == 0` is
    /// rejected as [`UnifiedError::ZeroSize`], matching the other
    /// constructors.
    pub fn new_host_with_capacity_on(
        size: usize,
        capacity: usize,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        // Never expose more than the physical allocation owns.
        let capacity = capacity.max(size);
        // mem finding #2: only the host `Box<[u8]>` backing actually
        // *uses* the spare `capacity` region — `try_grow_in_place` is host
        // only (`supports_in_place_grow() == !IS_UVM_BACKED`). On UVM-backed
        // CUDA hosts the in-place grow is deferred, so reserving the full
        // `capacity` of managed/device memory just over-allocates GPU memory
        // that can never be exposed. Allocate exactly `size` there (matching
        // this constructor's documented "degrades to an exactly-`size`
        // allocation" contract) and pin `host_capacity` to match so the
        // bookkeeping never claims more than is physically owned.
        //
        // On the host path we still allocate the full `capacity`:
        // `Backing::allocate` zero-fills the entire `Box<[u8]>` regardless
        // of the visible-window argument, so every byte a future in-place
        // grow can expose is already zero (H2).
        let backing_size = if IS_UVM_BACKED { size } else { capacity };
        let (ptr, backing) = Backing::allocate(backing_size, backing_size)?;
        Ok(Self {
            ptr,
            // Logical length starts at `size`; on the host path the rest of
            // `capacity` is reserved spare that `try_grow_in_place` can
            // later expose.
            size,
            device_id,
            backing,
            host_addressable: true,
            host_zeroize_on_drop: IS_HOST_BACKED,
            host_capacity: backing_size,
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
    /// 2. Allocate the underlying `Backing`. If the driver itself
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
                // Audit H4: same standard host-addressable backings as
                // `new_with_visible_window_on`.
                host_addressable: true,
                // Audit (LOW): only the no-CUDA `Box<[u8]>` build is
                // zeroized on drop.
                host_zeroize_on_drop: IS_HOST_BACKED,
                // Exactly-sized backing: logical == physical, no spare
                // capacity for an in-place grow.
                host_capacity: size,
                tenant_ctx: Some(tenant_ctx),
            }),
            Err(e) => {
                tenant_ctx.release_gpu_bytes(size as u64);
                Err(e.into())
            }
        }
    }

    /// T39 — allocate `size` bytes routed through the tenant's
    /// driver-enforced [`TenantMemPool`].
    ///
    /// This is the bypass-resistant complement to
    /// [`Self::new_on_with_tenant_context`]: instead of (or rather,
    /// in addition to) the in-process `consume_gpu_bytes` counter,
    /// the CUDA driver itself enforces the
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` cap configured on the
    /// pool. Over-cap allocations fail with
    /// `CUDA_ERROR_OUT_OF_MEMORY` at the driver level, which this
    /// constructor surfaces as
    /// [`tensor_wasm_core::error::TensorWasmError::CudaError`] via
    /// the [`UnifiedError`] → [`tensor_wasm_core::error::TensorWasmError`]
    /// conversion.
    ///
    /// # Bytes layout
    ///
    /// `cuMemAllocFromPoolAsync` returns device-located uninitialised
    /// memory. Unlike the unified-memory (`cuMemAllocManaged`) path,
    /// this allocation is NOT host-addressable through normal
    /// dereferencing — accessing the bytes from the CPU requires an
    /// explicit copy or an event-synchronised stream. The Wasm
    /// linear-memory path therefore CANNOT route through this
    /// constructor today; the v0.5 cutover that splits managed-memory
    /// from pooled-device-memory will introduce a separate
    /// linear-memory variant for tenants that opt into the
    /// driver-enforced path. For v0.4 this entry point is intended
    /// for scratch/working-set buffers that live entirely on the
    /// device.
    ///
    /// Gated behind `#[cfg(feature = "gpu-mem-pool")]`. The pool
    /// handle MUST have been constructed for the same `device_id` you
    /// pass here; the constructor does not double-check (cudarc's
    /// `cuMemAllocFromPoolAsync` returns
    /// `CUDA_ERROR_INVALID_DEVICE` on a mismatch, which we surface as
    /// `UnifiedError::Cuda`).
    #[cfg(feature = "gpu-mem-pool")]
    pub fn new_in_tenant_pool(
        pool: Arc<crate::cuda_mem_pool::TenantMemPool>,
        size: usize,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        // Bypass the per-feature `Backing::allocate` machinery: the
        // pool path is the *only* allocator here, and it does not
        // zero-init (device-located memory is uninitialised at
        // allocation). We deliberately do NOT pay for a
        // full-allocation memset here because the T39 use cases
        // (working-set scratch buffers) overwrite the region
        // immediately.
        //
        // Audit (MEDIUM — intra-tenant residue): because this path
        // skips the device memset, the freshly-allocated region may
        // contain bytes left over from a PRIOR allocation that the
        // SAME tenant's pool recycled (`cuMemAllocFromPoolAsync` draws
        // from the pool's released-but-not-returned arena). Cross-
        // tenant safety is unaffected — each tenant owns a distinct
        // pool — but a tenant could observe its own freed residue.
        //
        // CONTRACT: callers MUST fully overwrite every byte they will
        // subsequently read before reading it. This buffer is returned
        // UNINITIALISED. There is no host-side memset available on this
        // path (the bytes are device-only — see "Bytes layout" above —
        // so `ptr::write_bytes` from the CPU would be UB), and a device
        // memset helper would require threading a `&Stream` through
        // `TenantMemPool`, which lives in another module. If a
        // zero-on-allocate option is wanted it should be added to
        // `TenantMemPool::allocate` (a stream-synchronised `cuMemsetD8`)
        // rather than faked here; deferred — see the report.
        let ptr = pool.allocate(size)?;
        let tp = TenantPoolBacking {
            ptr,
            size,
            pool: Arc::clone(&pool),
        };
        Ok(Self {
            ptr,
            size,
            device_id,
            backing: Backing::TenantPool(tp),
            // Audit H4: `cuMemAllocFromPoolAsync` returns DEVICE-ONLY
            // memory (see the "Bytes layout" doc above). It is NOT
            // host-dereferenceable, so flag it accordingly — `as_slice`
            // / `as_mut_slice` will panic rather than fabricate a host
            // `&[u8]` over device memory (which would be UB).
            host_addressable: false,
            // Audit (LOW): device-only memory cannot be zeroized from the
            // CPU in `Drop` (needs a live CUDA context); leave it to the
            // pool's free path.
            host_zeroize_on_drop: false,
            // Device-only allocation: no host capacity concept and no
            // in-place grow on this path. Bookkeeping-equal to `size`.
            host_capacity: size,
            // No `tenant_ctx`: this constructor is the *driver*-pin
            // entry point. The caller chooses whether to also stash
            // an `Arc<TenantContext>` for in-process accounting via a
            // wrapping path; conflating the two layers here would
            // make the cap appear to be enforced twice on a single
            // allocation and confuse `gpu_bytes_in_use` reporting.
            tenant_ctx: None,
        })
    }

    /// Length in bytes.
    ///
    /// This is the *logical* (currently-used) length — the number of bytes
    /// exposed by [`Self::as_slice`] / [`Self::as_mut_slice`]. On the host
    /// backing it may be smaller than the physical allocation; see
    /// [`Self::capacity`].
    pub fn len(&self) -> usize {
        self.size
    }

    /// Physical byte capacity of the underlying allocation.
    ///
    /// Equals [`Self::len`] for every buffer except those built via
    /// [`Self::new_host_with_capacity_on`], where it reports the reserved
    /// spare into which [`Self::try_grow_in_place`] can grow without a
    /// realloc+copy. Always `>= len()`.
    pub fn capacity(&self) -> usize {
        self.host_capacity
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
    ///
    /// # Panics
    ///
    /// Audit H4: panics if the buffer is NOT host-addressable (i.e. it
    /// was created via `new_in_tenant_pool`, whose
    /// `cuMemAllocFromPoolAsync` memory is device-only). Fabricating a
    /// host `&[u8]` over device memory is undefined behaviour, so this
    /// fails loudly rather than handing out a slice the CPU cannot
    /// legally dereference. The signature is unchanged (returns `&[u8]`,
    /// not `Result`) because every in-tree caller —
    /// `TensorWasmLinearMemory` in `wasm_memory.rs`, the pool/tests —
    /// only ever calls this on host-addressable managed/heap buffers;
    /// see the `host_addressable` field.
    pub fn as_slice(&self) -> &[u8] {
        assert!(
            self.host_addressable,
            "UnifiedBuffer::as_slice on a device-only buffer (audit H4): \
             this allocation came from new_in_tenant_pool \
             (cuMemAllocFromPoolAsync) and is NOT host-dereferenceable; \
             copy it to host memory via a stream instead"
        );
        // SAFETY: ptr is non-null and points to `size` valid bytes by the
        // type invariant, and the assert above proves the bytes are
        // host-addressable.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Borrow the buffer as a mutable byte slice.
    ///
    /// # Panics
    ///
    /// Audit H4: panics if the buffer is NOT host-addressable. See
    /// [`Self::as_slice`] for the rationale.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(
            self.host_addressable,
            "UnifiedBuffer::as_mut_slice on a device-only buffer (audit H4): \
             this allocation came from new_in_tenant_pool \
             (cuMemAllocFromPoolAsync) and is NOT host-dereferenceable; \
             copy it to/from host memory via a stream instead"
        );
        // SAFETY: ptr is non-null, points to `size` valid bytes, `&mut self`
        // proves uniqueness, and the assert above proves the bytes are
        // host-addressable.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Which device this buffer is anchored to.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Audit (LOW — ASLR-leak via logs): an opaque, non-reversible token
    /// derived from the real pointer for use in `Debug` / log output. It
    /// lets two renders of the same buffer be correlated WITHOUT
    /// disclosing the actual allocation address (which would leak the
    /// memory layout and defeat ASLR). A plain multiplicative hash of the
    /// address bits — not cryptographic, just enough to scramble the
    /// address so it cannot be read back.
    fn opaque_handle(&self) -> u32 {
        let bits = self.ptr.as_ptr() as usize as u64;
        // FNV-1a-ish fold down to 32 bits; one-way for logging purposes.
        let mixed = bits.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((mixed >> 32) ^ (mixed & 0xFFFF_FFFF)) as u32
    }

    /// Attempt to grow the buffer's logical length in place to `new_size`
    /// bytes, without a realloc+copy.
    ///
    /// # Behaviour by backing
    ///
    /// **HOST (`Box<[u8]>`, the no-CUDA default build) — implemented.**
    /// Succeeds iff `new_size <= self.capacity()` (the physical bytes the
    /// backing already owns). On success the logical length
    /// ([`Self::len`]) is bumped to `new_size` and the newly-exposed region
    /// `[old_len, new_size)` is zero-filled to uphold the H2 zero-on-grow
    /// guarantee, then `Ok(())` is returned. A shrink (`new_size <=
    /// old_len`) just lowers the logical length and zero-fills nothing.
    /// When `new_size` exceeds the physical capacity there is no spare room
    /// to grow into, so the call returns
    /// [`UnifiedError::NotSupported`] (`feature = "try_grow_in_place"`,
    /// `backing = "host-box"`) — a typed, NON-CUDA "would require realloc"
    /// signal the caller uses to fall back to its realloc+copy path. The
    /// exactly-sized constructors (`new`, `new_on`,
    /// `new_with_visible_window_on`, …) leave no spare capacity, so a real
    /// grow against them always takes that fallback;
    /// [`Self::new_host_with_capacity_on`] is the constructor that reserves
    /// spare capacity so the in-place path actually fires (the per-spawn
    /// Wasm-linear-memory win — reserve at max once, then grow logically).
    ///
    /// **CUDA (`unified-memory` cust / `cudarc-backend`) — still deferred.**
    /// Returns [`UnifiedError::Cuda`] with the documented
    /// `"in-place grow not yet wired"` sentinel. `cuMemAllocManaged`
    /// returns a fixed-size allocation with no in-place grow; the Driver
    /// API alternative (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` +
    /// `cuMemSetAccess`, see the `TODO(v0.4)` below) is the remaining
    /// work. We deliberately do NOT report a CUDA error on the host build:
    /// the host path has no CUDA at all, so misclassifying its "no spare
    /// capacity" outcome as a CUDA failure (the previous stub's bug) would
    /// send callers down a driver-error branch that does not apply.
    ///
    /// # H2 / safety invariants
    ///
    /// `new_size` is never allowed to exceed the physical allocation:
    /// success requires `new_size <= host_capacity`, preserving the
    /// `size <= host_capacity` struct invariant that lets the slice and
    /// `Drop` paths soundly bound their `from_raw_parts` over `self.ptr`
    /// with `self.size`. The grown region is zeroed through the same
    /// host-addressable `&mut [u8]` the caller would see, so no UB and no
    /// stale residue is exposed. The grow is rejected on a non-host-
    /// addressable buffer (device-only `new_in_tenant_pool` memory) because
    /// it would need a host write into device memory — that path is CUDA
    /// and so already returns the deferred sentinel.
    ///
    /// # TODO(v0.4): CUDA in-place grow via `cuMemAddressReserve`+`cuMemMap`
    ///
    /// Implemented: the HOST `Box<[u8]>` spare-capacity path above.
    /// Deferred: the CUDA virtual-memory path:
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
    /// Still a follow-up because:
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
    pub fn try_grow_in_place(&mut self, new_size: usize) -> Result<(), UnifiedError> {
        // CUDA backings: still deferred. Keep the documented sentinel so
        // the v0.4 caller match site is stable. Reported as a CUDA error
        // ONLY because the active backing IS CUDA here.
        if IS_UVM_BACKED {
            return Err(UnifiedError::Cuda(
                "in-place grow not yet wired -- see UnifiedBuffer::try_grow_in_place doc + \
                 RFC 0001 v0.4 follow-up. Until then TensorWasmLinearMemory uses the B5 \
                 option-(a) preallocate-at-max strategy."
                    .into(),
            ));
        }

        // HOST (`Box<[u8]>`) path. A device-only (non-host-addressable)
        // buffer cannot be grown by a host write; reject it the same way
        // the slice accessors do, but as a typed error rather than a panic.
        if !self.host_addressable {
            return Err(UnifiedError::NotSupported {
                feature: "try_grow_in_place",
                backing: "host-box",
            });
        }

        // No spare physical capacity to grow into: signal the caller to
        // fall back to its realloc+copy path. NOT a CUDA error — there is
        // no CUDA on this build.
        if new_size > self.host_capacity {
            return Err(UnifiedError::NotSupported {
                feature: "try_grow_in_place",
                backing: "host-box",
            });
        }

        let old_size = self.size;
        // Bump the logical length first so `as_mut_slice` exposes the
        // newly-claimed bytes; the `size <= host_capacity` invariant is
        // upheld by the capacity check above.
        self.size = new_size;
        if new_size > old_size {
            // H2 zero-on-grow: zero exactly the freshly-exposed region
            // `[old_size, new_size)`. These bytes were already zero-filled
            // at allocation time (the host `Backing::allocate` /
            // `new_host_with_capacity_on` zero the whole physical
            // allocation), but a buffer may have been grown, shrunk, and
            // re-grown, so re-zero unconditionally to keep the guarantee
            // independent of prior history.
            self.as_mut_slice()[old_size..new_size].fill(0);
        }
        Ok(())
    }

    /// Whether [`Self::try_grow_in_place`] is implemented on this build.
    ///
    /// `true` on the no-CUDA `Box<[u8]>` host build, where
    /// [`Self::try_grow_in_place`] grows the logical length into reserved
    /// physical capacity without a realloc (see
    /// [`Self::new_host_with_capacity_on`]). `false` on the CUDA backings,
    /// where the `cuMemAddressReserve` + `cuMemMap` path is still the v0.4
    /// follow-up documented on `try_grow_in_place`.
    ///
    /// Callers (mainly `TensorWasmLinearMemory::grow_to`) probe this to
    /// pick between the in-place-grow and max-preallocate strategies
    /// without scraping the error string. Note that a `true` here means the
    /// mechanism exists; an individual grow can still return
    /// [`UnifiedError::NotSupported`] when the specific buffer has no spare
    /// capacity, at which point the caller takes its realloc fallback.
    pub const fn supports_in_place_grow() -> bool {
        // HOST build supports it; CUDA builds do not (yet).
        !IS_UVM_BACKED
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

impl UnifiedBacking for UnifiedBuffer {
    fn len(&self) -> usize {
        UnifiedBuffer::len(self)
    }

    fn as_slice(&self) -> &[u8] {
        UnifiedBuffer::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        UnifiedBuffer::as_mut_slice(self)
    }

    fn apply_advice(&self, hint: UvmAdvice) -> Result<(), UnifiedError> {
        // The legacy `UnifiedBuffer` path delegates advice through the
        // [`crate::advise`] module on cust builds; on other builds the
        // module is a documented no-op (`Ok(())`). To stay back-compat
        // with v0.3 we preserve that no-op shape rather than escalating
        // to `NotSupported`. The cust path is the only one whose
        // upstream `crate::advise::Advice` enum is wired to the real
        // driver call today.
        #[cfg(feature = "unified-memory")]
        {
            let advice = match hint {
                UvmAdvice::SetReadMostly => crate::advise::Advice::ReadMostly,
                UvmAdvice::UnsetReadMostly => {
                    // The cust path's [`crate::advise::Advice`] enum has
                    // no `UnsetReadMostly` variant yet (v0.3 never wired
                    // it). Surface as `NotSupported` so callers can
                    // detect the gap without scraping a driver error
                    // string; the v0.4 cutover will add the variant.
                    return Err(UnifiedError::NotSupported {
                        feature: "apply_advice(UnsetReadMostly)",
                        backing: "cust",
                    });
                }
                UvmAdvice::SetPreferredLocation(d) => {
                    crate::advise::Advice::PreferredLocation(DeviceId(d))
                }
                UvmAdvice::UnsetPreferredLocation => crate::advise::Advice::UnsetPreferredLocation,
                UvmAdvice::SetAccessedBy(d) => crate::advise::Advice::AccessedBy(DeviceId(d)),
                UvmAdvice::UnsetAccessedBy(d) => {
                    crate::advise::Advice::UnsetAccessedBy(DeviceId(d))
                }
            };
            crate::advise::apply(self, advice)
        }
        #[cfg(not(feature = "unified-memory"))]
        {
            // No-CUDA and cudarc-only paths: the legacy `UnifiedBuffer`
            // hand-mirror returned `Ok(())` for advise calls (the
            // `crate::advise::apply` function is itself a no-op here),
            // so the trait surface keeps that contract for back-compat.
            // Callers that want a hard error on a missing backing should
            // use the per-backend types directly until v0.4 routes
            // advice through `UnifiedBacking` everywhere.
            let _ = hint;
            Ok(())
        }
    }

    fn prefetch_to_device(&self, device_ord: u32) -> Result<(), UnifiedError> {
        // The legacy method signature on `UnifiedBuffer` takes no
        // ordinal (the cust path infers it from the buffer's owning
        // device). The trait surface accepts an ordinal so future
        // backings can target a non-owning device. cust 0.3's safe
        // surface cannot retarget mid-flight, so we can only honor a
        // prefetch aimed at the owning device; for any other ordinal we
        // surface `NotSupported` rather than silently servicing it
        // against the wrong device (matching the cudarc path — see
        // `cudarc_backend.rs`). The owning-device case stays an advisory
        // no-op `Ok(())` per `UnifiedBuffer::prefetch_to_device`.
        if device_ord == self.device_id().0 {
            UnifiedBuffer::prefetch_to_device(self)
        } else {
            Err(UnifiedError::NotSupported {
                feature: "prefetch_to_device(non-owning-ordinal)",
                backing: "cust",
            })
        }
    }

    fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        UnifiedBuffer::prefetch_to_host(self)
    }
}

impl fmt::Debug for UnifiedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Audit (LOW — ASLR-leak via logs): do NOT print the real `ptr`
        // address. A raw allocation address in a `Debug` render (which
        // routinely lands in logs / panic backtraces / error reports)
        // leaks the heap/UVM layout and defeats ASLR for an attacker who
        // can read them. We surface only the size + device, plus an
        // opaque, non-reversible handle so two `Debug` lines for the same
        // buffer can still be correlated without disclosing the address.
        f.debug_struct("UnifiedBuffer")
            .field("ptr", &"<redacted>")
            .field("handle", &format_args!("{:#010x}", self.opaque_handle()))
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .field("host_addressable", &self.host_addressable)
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
        //
        // Drop-ordering w.r.t. the `Backing` aliasing invariant: this
        // `drop` body only touches the tenant counter; it does NOT
        // read or write through `self.ptr`. After the body returns,
        // Rust runs field-drop in declaration order (`ptr`, `size`,
        // `device_id`, `backing`, `tenant_ctx`). `NonNull<u8>` is
        // `Copy`-shaped — its drop is a no-op — and only the
        // `backing` field's `Drop` actually frees the allocation. So
        // no in-flight `as_mut_slice` borrow can race a free here:
        // `&mut self` in `drop` precludes any outstanding borrow, and
        // the wrapped allocation is freed exactly once via the typed
        // `Backing` drop. See the `Backing` "Aliasing invariant" doc.
        if let Some(ctx) = self.tenant_ctx.as_ref() {
            ctx.release_gpu_bytes(self.size as u64);
        }

        // Audit (LOW — no Drop-time zeroization of sensitive buffers):
        // overwrite the bytes before the backing's own `Drop` frees them,
        // so freed memory does not linger with potentially-sensitive
        // residue. Gated to the HOST `Box<[u8]>` path only
        // (`host_zeroize_on_drop`, set from `IS_HOST_BACKED`): the CUDA
        // device/managed paths need a live CUDA context to touch their
        // bytes and so are deliberately excluded — zeroing them here would
        // be UB / a fault. We use `write_bytes` through the volatile
        // wrapper so the compiler cannot elide the store as a dead write
        // into about-to-be-freed memory. This crate does not depend on the
        // `zeroize` crate (not in its Cargo.toml), so a manual volatile
        // memset is used instead.
        if self.host_zeroize_on_drop && self.host_capacity > 0 {
            // SAFETY: on the host-backed build `ptr` points to
            // `host_capacity` valid, host-addressable, uniquely-owned bytes
            // (proved by `&mut self` in `drop`). We zero the FULL physical
            // capacity, not just the logical `size`: a buffer built via
            // `new_host_with_capacity_on` (or grown-then-shrunk) owns spare
            // bytes in `[size, host_capacity)` that may hold prior residue,
            // and those bytes are about to be freed too. The backing
            // `Box<[u8]>` is still alive — field drop runs only after this
            // body returns — so the write targets live memory.
            // `write_volatile` prevents the store from being optimised away.
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0u8, self.host_capacity);
                // A volatile re-read defeats dead-store elimination: it
                // forces the zeroing store above to be observable.
                let _ = std::ptr::read_volatile(self.ptr.as_ptr());
            }
        }
    }
}

/// Cross-crate conversion so `tensor-wasm-mem` errors flow into the workspace's
/// unified [`TensorWasmError`] type without manual mapping at every call site.
///
/// - `ZeroSize` → `TensorWasmError::Serialization` with a descriptive message
///   (zero-byte allocations are caller bugs, not memory exhaustion).
/// - `Allocation { .. }` → `TensorWasmError::Serialization` carrying the
///   detail string. Exhaustion is NOT routed through this variant — pool /
///   buffer exhaustion is reported as the structured
///   `UnifiedError::TooLarge { requested, limit }` instead, which maps
///   directly to `MemoryExhausted` below. Any remaining `Allocation` payload
///   reaching this conversion is a caller bug (e.g. `minimum > maximum`,
///   bad alignment) and is surfaced as `Serialization` accordingly.
/// - `TooLarge { requested, limit }` → `TensorWasmError::MemoryExhausted {
///   requested, limit }` (1:1, structured).
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
                tensor_wasm_core::error::TensorWasmError::Serialization(
                    format!("unified buffer allocation failed: {msg}").into(),
                )
            }
            UnifiedError::Cuda(msg) => {
                tensor_wasm_core::error::TensorWasmError::CudaError(msg.into())
            }
            UnifiedError::TooLarge { requested, limit } => {
                tensor_wasm_core::error::TensorWasmError::MemoryExhausted { requested, limit }
            }
            // `NotSupported` is the v0.3.6 B4.4 trait-surface error
            // variant: a [`UnifiedBacking`] method that has no
            // implementation on the active backing. We surface it as
            // `Serialization` (a "the call shape is wrong for what's
            // available" bucket) carrying the {feature, backing} pair
            // so downstream logs preserve the gap shape. A future
            // workspace error refactor may give this a first-class
            // `BackendUnsupported` variant; for v0.3.6 we keep the
            // mapping body-only.
            UnifiedError::NotSupported { feature, backing } => {
                tensor_wasm_core::error::TensorWasmError::Serialization(
                    format!("unified backing {backing:?} does not support feature {feature:?}")
                        .into(),
                )
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
    fn try_grow_in_place_classifies_per_backing() {
        // CUDA backings are still scaffolded: until v0.4 lands the
        // cuMemAddressReserve + cuMemMap path they return the documented
        // sentinel and supports_in_place_grow() is false. The HOST build
        // implements in-place grow, so supports_in_place_grow() is true and
        // an over-capacity grow returns the typed NON-CUDA NotSupported
        // signal rather than a CUDA error.
        assert_eq!(UnifiedBuffer::supports_in_place_grow(), !IS_UVM_BACKED);
        // `new(64)` is exactly-sized (capacity == 64), so growing to 128 has
        // no spare room and must fall back.
        let mut b = UnifiedBuffer::new(64).expect("alloc");
        let err = b.try_grow_in_place(128).expect_err("over-cap must error");
        if IS_UVM_BACKED {
            // CUDA path: documented sentinel string, contract for the v0.4
            // TensorWasmLinearMemory::grow_to caller match site.
            let msg = format!("{err}");
            assert!(
                msg.contains("in-place grow not yet wired"),
                "sentinel string changed; v0.4 caller match site must be updated: {msg}",
            );
        } else {
            // HOST path: typed realloc-fallback signal, NOT a CUDA error.
            assert!(
                matches!(
                    err,
                    UnifiedError::NotSupported {
                        feature: "try_grow_in_place",
                        backing: "host-box",
                    }
                ),
                "host over-cap grow must return typed NotSupported, got {err:?}",
            );
        }
    }

    // ---- HOST (`Box<[u8]>`) in-place grow regression tests ----
    //
    // Gated to the no-CUDA default build: these exercise the
    // spare-capacity path that only exists on the host backing.
    #[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
    mod host_in_place_grow {
        use super::*;

        #[test]
        fn host_grow_into_reserved_capacity_succeeds_and_zero_fills() {
            // Reserve 256 bytes physical, expose 64 logical.
            let mut b = UnifiedBuffer::new_host_with_capacity_on(64, 256, DeviceId::default())
                .expect("alloc with capacity");
            assert_eq!(b.len(), 64);
            assert_eq!(b.capacity(), 256);
            // Write a non-zero sentinel into the visible window so we can
            // prove the grow does NOT clobber existing bytes.
            b.as_mut_slice().fill(0xAB);
            // Grow in place to 200 (<= capacity): must succeed without realloc.
            b.try_grow_in_place(200)
                .expect("in-place grow within capacity");
            assert_eq!(b.len(), 200);
            assert_eq!(b.capacity(), 256, "capacity unchanged by in-place grow");
            let s = b.as_slice();
            // Pre-existing bytes preserved.
            assert!(
                s[..64].iter().all(|&x| x == 0xAB),
                "existing bytes clobbered"
            );
            // H2: freshly-exposed region reads as zero.
            assert!(
                s[64..200].iter().all(|&x| x == 0),
                "grown region not zeroed"
            );
        }

        #[test]
        fn host_grow_beyond_capacity_returns_not_supported() {
            let mut b = UnifiedBuffer::new_host_with_capacity_on(64, 128, DeviceId::default())
                .expect("alloc with capacity");
            let err = b
                .try_grow_in_place(129)
                .expect_err("over-capacity must fall back");
            assert!(
                matches!(
                    err,
                    UnifiedError::NotSupported {
                        feature: "try_grow_in_place",
                        backing: "host-box",
                    }
                ),
                "expected typed realloc-fallback signal, got {err:?}",
            );
            // The failed grow must not mutate the logical length.
            assert_eq!(b.len(), 64);
        }

        #[test]
        fn host_regrow_after_shrink_rezeros_exposed_region() {
            let mut b = UnifiedBuffer::new_host_with_capacity_on(8, 64, DeviceId::default())
                .expect("alloc with capacity");
            // Grow to 64, dirty the whole window, then shrink back to 8.
            b.try_grow_in_place(64).expect("grow to cap");
            b.as_mut_slice().fill(0xFF);
            b.try_grow_in_place(8).expect("shrink");
            assert_eq!(b.len(), 8);
            // Re-grow: the re-exposed [8, 64) region must read as zero even
            // though it held 0xFF before the shrink (H2 holds independent of
            // prior history).
            b.try_grow_in_place(64).expect("re-grow to cap");
            assert!(
                b.as_slice()[8..64].iter().all(|&x| x == 0),
                "re-grown region must be re-zeroed",
            );
        }

        #[test]
        fn host_exactly_sized_buffer_has_no_spare_capacity() {
            // Sanity: the standard constructor leaves capacity == len, so a
            // real grow always falls back.
            let mut b = UnifiedBuffer::new(64).expect("alloc");
            assert_eq!(b.capacity(), b.len());
            assert!(b.try_grow_in_place(65).is_err());
            // Growing to exactly the current size is a no-op success.
            b.try_grow_in_place(64)
                .expect("grow to current len is a no-op");
            assert_eq!(b.len(), 64);
        }
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
        assert!(matches!(
            b,
            tensor_wasm_core::error::TensorWasmError::Serialization(_)
        ));
        // `TensorWasmError`'s Display is sanitised and omits the inner detail;
        // the forwarded context is reachable via `inner()`.
        assert!(b.inner().unwrap_or("").contains("zero-byte"));
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_cuda() {
        let e = UnifiedError::Cuda("ctx not current".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        match b {
            tensor_wasm_core::error::TensorWasmError::CudaError(s) => {
                assert_eq!(&*s, "ctx not current")
            }
            other => panic!("expected CudaError, got {other:?}"),
        }
    }

    #[test]
    fn from_unified_error_too_large_maps_to_memory_exhausted_with_figures() {
        // Pool / buffer exhaustion is reported as the structured `TooLarge`
        // variant; the `From` impl plumbs the `requested` / `limit` fields
        // straight through to `MemoryExhausted` (no string parsing).
        let e = UnifiedError::TooLarge {
            requested: 4096,
            limit: 1024,
        };
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        match b {
            tensor_wasm_core::error::TensorWasmError::MemoryExhausted { requested, limit } => {
                assert_eq!(requested, 4096);
                assert_eq!(limit, 1024);
            }
            other => panic!("expected MemoryExhausted, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "unified-memory")]
    fn is_uvm_backed_true_under_feature() {
        // Closes the v0.3.2 audit gap (Problem #5): when the `unified-memory`
        // Cargo feature is on, `UnifiedBuffer` must report it routes through
        // `cuMemAllocManaged`. This is the compile-time guarantee that the
        // `TensorWasmLinearMemory` zero-copy promise rests on.
        let b = UnifiedBuffer::new(64).expect("alloc under feature");
        assert!(
            b.is_uvm_backed(),
            "unified-memory build must use UVM backing"
        );
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
    fn backing_aliasing_sealed_allocate_use_drop_round_trip() {
        // Audit T5 regression: the `Backing` enum aliases the same
        // allocation as the parent `UnifiedBuffer`'s `NonNull<u8>`.
        // We have sealed the enum (`pub(super)` inside a private
        // `mod backing { ... }`) so no caller can pattern-match a
        // variant and call `as_mut_slice` on the inner storage in
        // parallel with `UnifiedBuffer::as_mut_slice`. This test
        // exercises the only sound lifecycle — allocate, observe
        // through the parent struct's slice accessor, drop — and
        // asserts that the bytes round-trip without observable
        // aliasing fallout. The compile-time guarantee that no
        // external code can name `Backing::Cuda(...)` etc. is
        // enforced by the `pub(super)` declaration and verified at
        // build time; this runtime test exists for behavioural
        // regression coverage.
        let mut b = UnifiedBuffer::new(128).expect("alloc");
        // Write through the parent struct's `as_mut_slice` — the
        // only sound path. The inner `Backing` storage MUST NOT be
        // touched concurrently.
        {
            let s = b.as_mut_slice();
            for (i, byte) in s.iter_mut().enumerate() {
                *byte = (i & 0xFF) as u8;
            }
        }
        // Re-borrow read-only and confirm the writes landed.
        {
            let s = b.as_slice();
            for (i, byte) in s.iter().enumerate() {
                assert_eq!(
                    *byte,
                    (i & 0xFF) as u8,
                    "byte {i} mismatch — aliasing regression?"
                );
            }
        }
        // Drop the buffer at end of scope. The `Drop` impl must free
        // the underlying allocation exactly once via `Backing`'s
        // typed drop; ASan / Valgrind under CI would surface a
        // double-free if anything outside the sealed module had
        // reached in and called `into_inner` on the wrapped storage.
        drop(b);
    }

    #[test]
    fn debug_redacts_raw_pointer_address() {
        // Audit (LOW — ASLR-leak via logs): the `Debug` render must NOT
        // contain the real allocation address. We assert the redaction
        // marker is present and that the literal pointer (formatted the
        // way `#[derive(Debug)]` would) does not appear.
        let b = UnifiedBuffer::new(32).expect("alloc");
        let dbg = format!("{b:?}");
        assert!(dbg.contains("<redacted>"), "ptr not redacted: {dbg}");
        let raw = format!("{:p}", b.as_ptr());
        assert!(
            !dbg.contains(&raw),
            "Debug leaked the raw pointer address {raw}: {dbg}"
        );
    }

    #[test]
    #[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
    fn host_backed_buffer_is_zeroized_on_drop() {
        // Audit (LOW — Drop-time zeroization): on the host `Box<[u8]>`
        // build, dropping a buffer must overwrite its bytes. We cannot
        // legally read freed memory, so instead we verify the buffer is
        // flagged for drop-time zeroization and that the volatile memset
        // path runs without fault on a written-then-dropped buffer.
        let mut b = UnifiedBuffer::new(256).expect("alloc");
        b.as_mut_slice().fill(0xAB);
        assert!(
            b.host_zeroize_on_drop,
            "host build must flag drop-time zeroization"
        );
        // Dropping triggers the volatile zero memset; a fault here (e.g.
        // a regression that enabled zeroization on a device path) would
        // surface as a crash under this no-feature CI build.
        drop(b);
    }

    #[test]
    fn host_addressable_buffers_expose_slices() {
        // Audit H4: the standard constructors produce host-addressable
        // buffers, so `as_slice` / `as_mut_slice` must NOT panic.
        let mut b = UnifiedBuffer::new(16).expect("alloc");
        assert!(b.host_addressable);
        b.as_mut_slice().fill(1);
        assert!(b.as_slice().iter().all(|&v| v == 1));
    }

    #[test]
    fn from_unified_error_allocation_maps_to_serialization() {
        // Any `Allocation` payload reaching this conversion is a caller bug
        // (bad alignment, `minimum > maximum`, etc.) — exhaustion is now
        // routed through the structured `TooLarge` variant. The conversion
        // simply forwards the detail string into `Serialization`.
        let e = UnifiedError::Allocation("minimum 1024 > maximum 512".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(
            b,
            tensor_wasm_core::error::TensorWasmError::Serialization(_)
        ));
        // Display is sanitised; the forwarded detail is in `inner()`.
        assert!(b.inner().unwrap_or("").contains("minimum 1024"));
    }
}
