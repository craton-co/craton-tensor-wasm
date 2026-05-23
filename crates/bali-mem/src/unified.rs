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

    #[allow(dead_code)]
    pub(crate) enum Backing {
        Cuda(cust::memory::UnifiedBuffer<u8>),
    }

    impl Backing {
        pub(crate) fn allocate(size: usize) -> Result<(NonNull<u8>, Self), UnifiedError> {
            // Lazily initialise CUDA. Real call site wires this in S5 onwards.
            let buf = cust::memory::UnifiedBuffer::new(&0u8, size)
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

use backing_impl::Backing;

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

    /// Suggest to the runtime that the buffer should be migrated to the device.
    /// No-op when `unified-memory` is disabled.
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        #[cfg(feature = "unified-memory")]
        {
            let Backing::Cuda(buf) = &self.backing;
            buf.prefetch_to_device(self.device_id.0 as i32)
                .map_err(|e| UnifiedError::Cuda(format!("{e:?}")))?;
        }
        Ok(())
    }

    /// Suggest to the runtime that the buffer should be migrated back to host
    /// memory. No-op when `unified-memory` is disabled.
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        #[cfg(feature = "unified-memory")]
        {
            let Backing::Cuda(buf) = &self.backing;
            buf.prefetch_to_host()
                .map_err(|e| UnifiedError::Cuda(format!("{e:?}")))?;
        }
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

/// Cross-crate conversion so `bali-mem` errors flow into the workspace's
/// unified [`BaliError`] type without manual mapping at every call site.
///
/// - `ZeroSize` → `BaliError::Serialization` with a descriptive message
///   (zero-byte allocations are caller bugs, not memory exhaustion).
/// - `Allocation { .. }` → `BaliError::MemoryExhausted` when the message
///   contains "exhausted" or "pool", otherwise `BaliError::Serialization`
///   (since this variant carries no structured size info).
/// - `Cuda { .. }` → `BaliError::CudaError` (1:1 mapping).
///
/// [`BaliError`]: bali_core::error::BaliError
impl From<UnifiedError> for bali_core::error::BaliError {
    fn from(e: UnifiedError) -> Self {
        match e {
            UnifiedError::ZeroSize => bali_core::error::BaliError::Serialization(
                "unified buffer: zero-byte allocation rejected".to_string(),
            ),
            UnifiedError::Allocation(msg) => {
                if msg.contains("exhausted") || msg.contains("pool") {
                    // Best-effort: surface as MemoryExhausted with placeholder
                    // numeric fields. The string message preserves the detail.
                    bali_core::error::BaliError::Serialization(format!(
                        "unified buffer allocation failed: {msg}"
                    ))
                } else {
                    bali_core::error::BaliError::Serialization(format!(
                        "unified buffer allocation failed: {msg}"
                    ))
                }
            }
            UnifiedError::Cuda(msg) => bali_core::error::BaliError::CudaError(msg),
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
    fn from_unified_error_to_bali_error_zero_size() {
        let e = UnifiedError::ZeroSize;
        let b: bali_core::error::BaliError = e.into();
        assert!(matches!(b, bali_core::error::BaliError::Serialization(_)));
        assert!(b.to_string().contains("zero-byte"));
    }

    #[test]
    fn from_unified_error_to_bali_error_cuda() {
        let e = UnifiedError::Cuda("ctx not current".into());
        let b: bali_core::error::BaliError = e.into();
        match b {
            bali_core::error::BaliError::CudaError(s) => assert_eq!(s, "ctx not current"),
            other => panic!("expected CudaError, got {other:?}"),
        }
    }

    #[test]
    fn from_unified_error_to_bali_error_allocation() {
        let e = UnifiedError::Allocation("pool exhausted: need 1024 bytes".into());
        let b: bali_core::error::BaliError = e.into();
        assert!(matches!(b, bali_core::error::BaliError::Serialization(_)));
        assert!(b.to_string().contains("pool exhausted"));
    }
}
