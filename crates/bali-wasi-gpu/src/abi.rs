//! ABI for the `wasi-cuda` extension exposed to Wasm modules.
//!
//! The host functions live in a single module named [`MODULE`]. Each function
//! takes integer / pointer-as-i32 arguments compatible with the canonical
//! WASI ABI. Pointer arguments are interpreted as offsets within the
//! caller's Wasm linear memory; on the host side they are translated to
//! native pointers via the `caller.get_export("memory")` call.

/// Wasm module name to import host functions from: `wasi:cuda/host@0.1.0`.
pub const MODULE: &str = "wasi:cuda/host@0.1.0";

/// Function name: `load_ptx(ptx_ptr, ptx_len, entry_ptr, entry_len) -> i64`
///
/// Returns a non-negative [`KernelId`](bali_core::types::KernelId) on success,
/// or a negative [`AbiError`] code on failure.
pub const FN_LOAD_PTX: &str = "wasi_cuda_load_ptx";

/// Function name: `launch(kernel_id, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, args_ptr, args_len) -> i32`
///
/// Returns 0 on success, a negative [`AbiError`] on failure.
pub const FN_LAUNCH: &str = "wasi_cuda_launch";

/// Function name: `sync() -> i32`. Returns 0 on success, negative on failure.
pub const FN_SYNC: &str = "wasi_cuda_sync";

/// Function name: `last_error_ptr() -> i32`.
///
/// **Deprecated**: superseded by [`FN_LAST_ERROR_COPY`]. The original design
/// returned a pointer into a host-allocated Wasm-memory buffer, but that
/// required coordination with the guest's allocator. The constant is kept
/// for ABI / backwards-compat reasons but the function is no longer
/// registered with the linker — callers must instead use
/// [`FN_LAST_ERROR_LEN`] to learn the size and [`FN_LAST_ERROR_COPY`] to
/// receive the bytes into a guest-supplied buffer.
pub const FN_LAST_ERROR_PTR: &str = "wasi_cuda_last_error_ptr";

/// Function name: `last_error_len() -> i32`. Length of the most recent error
/// message in bytes (excluding the trailing NUL).
pub const FN_LAST_ERROR_LEN: &str = "wasi_cuda_last_error_len";

/// Function name: `last_error_copy(dst_ptr, dst_len) -> i32`. Copies the
/// most recent error message (without NUL terminator) into `[dst_ptr,
/// dst_ptr+dst_len)` in the caller's linear memory. Returns the number of
/// bytes actually written (clamped to `min(error_len, dst_len)`), or
/// `0` if no error is recorded.
pub const FN_LAST_ERROR_COPY: &str = "wasi_cuda_last_error_copy";

/// Negative i32 status codes returned by the wasi-cuda host functions.
///
/// These are stable across Bali versions; client code may match on the
/// numeric value if it needs to.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiError {
    /// CUDA is not available on this host (no toolkit / no driver).
    NotAvailable = -1,
    /// A pointer / length pair pointed outside the caller's linear memory.
    InvalidPointer = -2,
    /// Wasi-cuda was passed a [`KernelId`](bali_core::types::KernelId) that
    /// is not registered (or that belongs to another instance).
    InvalidKernel = -3,
    /// `load_ptx` was called with bytes that `ptxas` rejected.
    MalformedPtx = -4,
    /// The CUDA driver returned an error during a launch / sync.
    LaunchFailed = -5,
    /// Resource limits (max kernels per instance, etc.) were exceeded.
    QuotaExceeded = -6,
    /// Generic internal error.
    Internal = -7,
}

impl AbiError {
    /// Convert to the wire i32 code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Stable, human-readable name (used for log fields).
    pub fn name(self) -> &'static str {
        match self {
            AbiError::NotAvailable => "not_available",
            AbiError::InvalidPointer => "invalid_pointer",
            AbiError::InvalidKernel => "invalid_kernel",
            AbiError::MalformedPtx => "malformed_ptx",
            AbiError::LaunchFailed => "launch_failed",
            AbiError::QuotaExceeded => "quota_exceeded",
            AbiError::Internal => "internal",
        }
    }
}

/// Maximum PTX module length we'll accept from a single `load_ptx` call.
pub const MAX_PTX_BYTES: usize = 8 * 1024 * 1024;

/// Maximum number of kernels a single instance may keep alive.
pub const MAX_KERNELS_PER_INSTANCE: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_error_codes_stable() {
        assert_eq!(AbiError::NotAvailable.code(), -1);
        assert_eq!(AbiError::InvalidPointer.code(), -2);
        assert_eq!(AbiError::InvalidKernel.code(), -3);
        assert_eq!(AbiError::MalformedPtx.code(), -4);
        assert_eq!(AbiError::LaunchFailed.code(), -5);
        assert_eq!(AbiError::QuotaExceeded.code(), -6);
        assert_eq!(AbiError::Internal.code(), -7);
    }

    #[test]
    fn abi_error_names_stable() {
        assert_eq!(AbiError::NotAvailable.name(), "not_available");
        assert_eq!(AbiError::MalformedPtx.name(), "malformed_ptx");
    }

    #[test]
    fn function_names_carry_wasi_cuda_prefix() {
        for name in [
            FN_LOAD_PTX,
            FN_LAUNCH,
            FN_SYNC,
            FN_LAST_ERROR_PTR,
            FN_LAST_ERROR_LEN,
            FN_LAST_ERROR_COPY,
        ] {
            assert!(name.starts_with("wasi_cuda_"));
        }
    }

    #[test]
    fn module_string_is_versioned() {
        assert!(MODULE.contains("wasi:cuda"));
        assert!(MODULE.contains("@0.1"));
    }
}
