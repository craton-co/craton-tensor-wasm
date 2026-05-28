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
/// [`crate::cudarc_backend::apply_advice`] on the cudarc path, and the
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
    }

    impl Backing {
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
use backing::{Backing, IS_UVM_BACKED};

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
                UvmAdvice::UnsetPreferredLocation => {
                    crate::advise::Advice::UnsetPreferredLocation
                }
                UvmAdvice::SetAccessedBy(d) => {
                    crate::advise::Advice::AccessedBy(DeviceId(d))
                }
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
        // backings can target a non-owning device; on the cust path we
        // silently discard the argument to preserve v0.3 semantics
        // (cust 0.3's safe surface cannot retarget mid-flight anyway).
        let _ = device_ord;
        UnifiedBuffer::prefetch_to_device(self)
    }

    fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        UnifiedBuffer::prefetch_to_host(self)
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
            UnifiedError::Allocation(msg) => tensor_wasm_core::error::TensorWasmError::Serialization(
                format!("unified buffer allocation failed: {msg}").into(),
            ),
            UnifiedError::Cuda(msg) => tensor_wasm_core::error::TensorWasmError::CudaError(msg.into()),
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
                    format!(
                        "unified backing {backing:?} does not support feature {feature:?}"
                    )
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
                assert_eq!(*byte, (i & 0xFF) as u8, "byte {i} mismatch — aliasing regression?");
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
    fn from_unified_error_allocation_maps_to_serialization() {
        // Any `Allocation` payload reaching this conversion is a caller bug
        // (bad alignment, `minimum > maximum`, etc.) — exhaustion is now
        // routed through the structured `TooLarge` variant. The conversion
        // simply forwards the detail string into `Serialization`.
        let e = UnifiedError::Allocation("minimum 1024 > maximum 512".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(b, tensor_wasm_core::error::TensorWasmError::Serialization(_)));
        assert!(b.to_string().contains("minimum 1024"));
    }
}
