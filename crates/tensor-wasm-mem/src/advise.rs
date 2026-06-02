// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `cudaMemAdvise` hint helpers.
//!
//! The CUDA runtime supports hinting the unified-memory subsystem about
//! anticipated access patterns. The three hints we expose here are the most
//! frequently useful for TensorWasm's mix of inference/training workloads:
//!
//! - [`Advice::ReadMostly`] — pages are read on many devices but rarely written.
//! - [`Advice::PreferredLocation`] — the page should live primarily on a
//!   specific device.
//! - [`Advice::AccessedBy`] — a particular device will access the page, so
//!   the runtime should map it without migrating.
//!
//! When the `unified-memory` feature is disabled every function in this module
//! is a no-op that returns `Ok(())`, so call sites do not need to feature-gate
//! their own logic.

use crate::unified::{DeviceId, UnifiedBuffer, UnifiedError};

/// A memory-access hint passed to the CUDA unified-memory subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advice {
    /// The region is mostly read; multiple devices may keep duplicated copies.
    ReadMostly,
    /// Pages should live primarily on the given device.
    PreferredLocation(DeviceId),
    /// The given device will access the region; map it without migrating.
    AccessedBy(DeviceId),
    /// Reverse a previous `PreferredLocation` hint.
    UnsetPreferredLocation,
    /// Reverse a previous `AccessedBy` hint.
    UnsetAccessedBy(DeviceId),
}

impl Advice {
    /// Stable, human-readable name (used for log fields).
    pub fn name(&self) -> &'static str {
        match self {
            Advice::ReadMostly => "read_mostly",
            Advice::PreferredLocation(_) => "preferred_location",
            Advice::AccessedBy(_) => "accessed_by",
            Advice::UnsetPreferredLocation => "unset_preferred_location",
            Advice::UnsetAccessedBy(_) => "unset_accessed_by",
        }
    }
}

/// Apply an [`Advice`] hint to a [`UnifiedBuffer`].
///
/// On non-CUDA builds this is a no-op that always returns `Ok(())`. On CUDA
/// builds it forwards to `cudaMemAdvise` and converts errors via
/// [`UnifiedError::Cuda`].
pub fn apply(_buffer: &UnifiedBuffer, _advice: Advice) -> Result<(), UnifiedError> {
    #[cfg(feature = "unified-memory")]
    {
        apply_cuda(_buffer, _advice)
    }
    #[cfg(not(feature = "unified-memory"))]
    {
        Ok(())
    }
}

#[cfg(feature = "unified-memory")]
fn apply_cuda(buffer: &UnifiedBuffer, advice: Advice) -> Result<(), UnifiedError> {
    use cust::sys as cuda_sys;

    let ptr = buffer.as_ptr() as cuda_sys::CUdeviceptr;
    let size = buffer.len();
    let (advice_kind, device) = match advice {
        Advice::ReadMostly => (cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY, 0i32),
        Advice::PreferredLocation(d) => (
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            d.0 as i32,
        ),
        Advice::AccessedBy(d) => (
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY,
            d.0 as i32,
        ),
        Advice::UnsetPreferredLocation => (
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
            0i32,
        ),
        Advice::UnsetAccessedBy(d) => (
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_ACCESSED_BY,
            d.0 as i32,
        ),
    };
    // SAFETY: ptr/size are derived from a valid live UnifiedBuffer.
    let res = unsafe { cuda_sys::cuMemAdvise(ptr, size, advice_kind, device) };
    if res == cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(UnifiedError::Cuda(format!("cuMemAdvise -> {res:?}")))
    }
}

/// Convenience: mark a buffer as read-mostly.
pub fn set_read_mostly(buffer: &UnifiedBuffer) -> Result<(), UnifiedError> {
    apply(buffer, Advice::ReadMostly)
}

/// Convenience: set preferred location.
pub fn set_preferred_location(
    buffer: &UnifiedBuffer,
    device_id: DeviceId,
) -> Result<(), UnifiedError> {
    apply(buffer, Advice::PreferredLocation(device_id))
}

/// Convenience: signal that `device_id` will access this buffer.
pub fn set_accessed_by(buffer: &UnifiedBuffer, device_id: DeviceId) -> Result<(), UnifiedError> {
    apply(buffer, Advice::AccessedBy(device_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::UnifiedBuffer;

    #[test]
    fn names_are_stable() {
        assert_eq!(Advice::ReadMostly.name(), "read_mostly");
        assert_eq!(
            Advice::PreferredLocation(DeviceId(0)).name(),
            "preferred_location"
        );
        assert_eq!(Advice::AccessedBy(DeviceId(0)).name(), "accessed_by");
        assert_eq!(
            Advice::UnsetPreferredLocation.name(),
            "unset_preferred_location"
        );
        assert_eq!(
            Advice::UnsetAccessedBy(DeviceId(0)).name(),
            "unset_accessed_by"
        );
    }

    #[test]
    fn apply_no_op_without_cuda() {
        // Without the unified-memory feature this MUST always succeed.
        let b = UnifiedBuffer::new(64).unwrap();
        apply(&b, Advice::ReadMostly).expect("no-op should succeed");
        set_read_mostly(&b).expect("convenience helper should also succeed");
        set_preferred_location(&b, DeviceId(0)).expect("set_preferred_location");
        set_accessed_by(&b, DeviceId(1)).expect("set_accessed_by");
    }

    #[test]
    fn advice_is_copy_and_eq() {
        let a = Advice::AccessedBy(DeviceId(2));
        let b = a;
        assert_eq!(a, b);
    }
}
