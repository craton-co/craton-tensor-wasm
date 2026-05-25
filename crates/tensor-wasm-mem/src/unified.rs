// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `UnifiedBuffer` — a region of memory that is addressable from both CPU and
//! GPU when the `unified-memory` feature is enabled.
//!
//! Two backings:
//! - With `unified-memory`: `cudaMallocManaged` via the `cust` crate. Pages
//!   migrate on demand between host and device.
//! - Without `unified-memory`: a heap-allocated `Box<[u8]>`. This compiles on
//!   non-CUDA hosts and is what CI uses. It exposes the same API but the
//!   `prefetch_to_device` / `prefetch_to_host` methods become no-ops.

use std::fmt;
use std::ptr::NonNull;

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
    // Holds the underlying storage so it is freed on drop. When the
    // `unified-memory` feature is enabled this is a wrapper around the CUDA
    // unified pointer; otherwise it is a plain Box<[u8]>.
    #[allow(dead_code)]
    backing: Backing,
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
    /// `true` only when the `unified-memory` Cargo feature is enabled, in which
    /// case [`Backing::allocate`] routes through `cuMemAllocManaged` via cust.
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
        pub(crate) fn allocate(size: usize) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Ensure cuInit(0) + a primary context have run before the first
            // cuMemAllocManaged. Idempotent and cheap on subsequent calls.
            ensure_cuda_init()?;
            // `as_raw_mut` is a `&mut self` method on cust's UnifiedPointer,
            // so the buffer must be bound mutably even though we only read it
            // once to derive the raw pointer.
            let mut buf = cust::memory::UnifiedBuffer::new(&0u8, size)
                .map_err(|e| UnifiedError::Cuda(format!("{e:?}")))?;
            // SAFETY: cust returns a non-null aligned pointer to the allocation.
            let ptr = NonNull::new(buf.as_unified_ptr().as_raw_mut() as *mut u8)
                .ok_or_else(|| UnifiedError::Allocation("cust returned null".into()))?;
            Ok((ptr, Backing::Cuda(buf)))
        }
    }
}

#[cfg(not(feature = "unified-memory"))]
mod backing_impl {
    use super::*;

    /// Compile-time constant exposed by [`super::UnifiedBuffer::is_uvm_backed`].
    /// `false` on non-CUDA builds: [`Backing::allocate`] returns a heap
    /// `Box<[u8]>` and the prefetch/advise helpers are no-ops.
    pub(crate) const IS_UVM_BACKED: bool = false;

    #[allow(dead_code)]
    pub(crate) enum Backing {
        Host(Box<[u8]>),
    }

    impl Backing {
        pub(crate) fn allocate(size: usize) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Allocate a zeroed boxed slice; this serves the no-CUDA path.
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
    pub fn new(size: usize) -> Result<Self, UnifiedError> {
        Self::new_on(size, DeviceId::default())
    }

    /// Allocate a new unified buffer of `size` bytes on the named device.
    pub fn new_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        let (ptr, backing) = Backing::allocate(size)?;
        Ok(Self {
            ptr,
            size,
            device_id,
            backing,
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

    /// Whether this buffer is backed by CUDA Unified Memory (`cuMemAllocManaged`).
    ///
    /// Returns `true` only when the crate was compiled with
    /// `--features unified-memory`; otherwise the backing is a heap
    /// `Box<[u8]>`. This is a compile-time property of the active backing
    /// (it does not probe the driver at runtime), and is exposed as a public
    /// probe so callers — including [`crate::wasm_memory::TensorWasmLinearMemory`]
    /// — can assert in tests that the audit-flagged "wasm linear memory not
    /// UVM-backed" gap is actually closed at build configuration time.
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
    #[cfg(not(feature = "unified-memory"))]
    fn is_uvm_backed_false_without_feature() {
        // Without the feature, the backing is a heap `Box<[u8]>`. This test
        // pins the inverse half of the contract so a future regression that
        // accidentally turns the probe into a runtime-always-true cannot
        // sneak past CI's no-feature build.
        let b = UnifiedBuffer::new(64).expect("alloc without feature");
        assert!(!b.is_uvm_backed(), "no-feature build must use heap backing");
    }

    #[test]
    fn from_unified_error_to_tensor_wasm_error_allocation_non_exhausted() {
        let e = UnifiedError::Allocation("minimum 1024 > maximum 512".into());
        let b: tensor_wasm_core::error::TensorWasmError = e.into();
        assert!(matches!(b, tensor_wasm_core::error::TensorWasmError::Serialization(_)));
        assert!(b.to_string().contains("minimum 1024"));
    }
}
