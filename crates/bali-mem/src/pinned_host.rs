//! `PinnedHostBuffer` — host buffer used when the `unified-memory` feature is
//! disabled.
//!
//! The "pinned" in the name reflects the intended end-state, not the current
//! behavior. Today this type unconditionally allocates a plain
//! `Box<[u8]>` — which is sufficient for the `--no-default-features` CI path
//! and for developer laptops without CUDA installed.
//!
//! Wiring a real `cudaHostAlloc` (with `cudaHostAllocMapped |
//! cudaHostAllocPortable`) path that produces page-locked, DMA-friendly memory
//! is deferred. When that work lands it will be gated behind the
//! `pinned-host-memory` feature flag and engaged only when the
//! `unified-memory` feature is disabled on a host that does have CUDA
//! available; the heap-`Box<[u8]>` path will remain the fallback for everyone
//! else.
//!
//! The API is intentionally symmetric with [`crate::unified::UnifiedBuffer`]
//! so call sites can swap one buffer type for the other without conditional
//! compilation at the call site.

use std::fmt;
use std::ptr::NonNull;

use crate::unified::{DeviceId, UnifiedError};

/// A page-locked host buffer (or a plain heap buffer when CUDA isn't linked).
pub struct PinnedHostBuffer {
    ptr: NonNull<u8>,
    size: usize,
    device_id: DeviceId,
    _backing: Box<[u8]>,
}

// SAFETY: the inner buffer is owned by this struct, not shared.
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    /// Allocate `size` bytes of pinned host memory on the default device.
    pub fn new(size: usize) -> Result<Self, UnifiedError> {
        Self::new_on(size, DeviceId::default())
    }

    /// Allocate `size` bytes of pinned host memory on the named device.
    pub fn new_on(size: usize, device_id: DeviceId) -> Result<Self, UnifiedError> {
        if size == 0 {
            return Err(UnifiedError::ZeroSize);
        }
        let mut backing: Box<[u8]> = vec![0u8; size].into_boxed_slice();
        let ptr = NonNull::new(backing.as_mut_ptr())
            .ok_or_else(|| UnifiedError::Allocation("Box returned null".into()))?;
        Ok(Self {
            ptr,
            size,
            device_id,
            _backing: backing,
        })
    }

    /// Length in bytes.
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

    /// No-op prefetch hint. Present for API parity with [`crate::unified::UnifiedBuffer`].
    pub fn prefetch_to_device(&self) -> Result<(), UnifiedError> {
        Ok(())
    }

    /// No-op prefetch hint. Present for API parity with [`crate::unified::UnifiedBuffer`].
    pub fn prefetch_to_host(&self) -> Result<(), UnifiedError> {
        Ok(())
    }
}

impl fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut b = PinnedHostBuffer::new(128).unwrap();
        b.as_mut_slice().fill(0xAB);
        assert!(b.as_slice().iter().all(|&v| v == 0xAB));
        assert_eq!(b.len(), 128);
        assert!(!b.is_empty());
    }

    #[test]
    fn zero_size_rejected() {
        assert!(matches!(
            PinnedHostBuffer::new(0).unwrap_err(),
            UnifiedError::ZeroSize
        ));
    }

    #[test]
    fn device_id_carried() {
        let b = PinnedHostBuffer::new_on(16, DeviceId(7)).unwrap();
        assert_eq!(b.device_id(), DeviceId(7));
    }

    #[test]
    fn prefetch_methods_are_no_ops() {
        let b = PinnedHostBuffer::new(32).unwrap();
        b.prefetch_to_device().expect("no-op should succeed");
        b.prefetch_to_host().expect("no-op should succeed");
    }
}
