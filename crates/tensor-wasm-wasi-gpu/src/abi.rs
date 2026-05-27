// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! ABI for the `wasi-cuda` extension exposed to Wasm modules.
//!
//! The host functions live in a single module named [`MODULE`]. Each function
//! takes integer / pointer-as-i32 arguments compatible with the canonical
//! WASI ABI. Pointer arguments are interpreted as offsets within the
//! caller's Wasm linear memory; on the host side they are translated to
//! native pointers via the `caller.get_export("memory")` call.
//!
//! See `wit/wasi-cuda.wit` at the workspace root for the equivalent
//! Component-Model interface description (`wasi:cuda/host@0.2.0`).

/// Wasm module name to import host functions from: `wasi:cuda/host@0.2.0`.
///
/// The version segment is kept in lockstep with the `package` declaration in
/// `wit/wasi-cuda.wit`; bumping one without the other will cause guests
/// generated from the WIT to fail to link against this host.
pub const MODULE: &str = "wasi:cuda/host@0.2.0";

/// Function name: `load_ptx(ptx_ptr, ptx_len, entry_ptr, entry_len) -> i64`
///
/// Returns a non-negative [`KernelId`](tensor_wasm_core::types::KernelId) on success,
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
/// dst_ptr+dst_len)` in the caller's linear memory.
///
/// Return values:
/// - `n > 0`: number of bytes written (clamped to `min(error_len, dst_len)`).
/// - `0`: no error is currently recorded, or `dst_len == 0`.
/// - `-2` ([`AbiError::InvalidPointer`]): `(dst_ptr, dst_len)` is invalid
///   (out of bounds, negative, or overflows) or the underlying memory
///   write failed. Crucially this is distinct from `0`: a guest that sees
///   `-2` knows it must fix its buffer rather than assume "no error."
pub const FN_LAST_ERROR_COPY: &str = "wasi_cuda_last_error_copy";

/// Negative i32 status codes returned by the wasi-cuda host functions.
///
/// These are stable across TensorWasm versions; client code may match on the
/// numeric value if it needs to.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiError {
    /// CUDA is not available on this host (no toolkit / no driver).
    NotAvailable = -1,
    /// A pointer / length pair pointed outside the caller's linear memory.
    InvalidPointer = -2,
    /// Wasi-cuda was passed a [`KernelId`](tensor_wasm_core::types::KernelId) that
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
    /// Launch grid / block dimensions exceeded hardware caps or were
    /// non-positive. Returned by the host *before* any CUDA driver call,
    /// allowing the guest to distinguish a launch-shape bug from a driver
    /// error.
    InvalidDimensions = -8,
    /// Caller passed a structurally invalid argument that is neither a
    /// memory-region issue nor a dimensions issue. Currently used for
    /// non-UTF8 entry names in `load_ptx`.
    InvalidArgs = -9,
    /// Caller passed a well-formed, in-bounds kernel-argument buffer
    /// that exceeds the host's sanity caps — total argv bytes greater
    /// than [`MAX_KERNEL_ARGS_BYTES`](crate::kernel_args::MAX_KERNEL_ARGS_BYTES)
    /// (4 KiB) or more than
    /// [`MAX_KERNEL_ARGS`](crate::kernel_args::MAX_KERNEL_ARGS) (128)
    /// tagged records. Since v0.2.0 (W1.1) the typed-argv lane accepts
    /// arbitrary scalar + pointer argv below those caps; this code is
    /// reserved for cap busts and is kept distinct from
    /// [`AbiError::InvalidArgs`] so a guest can tell "your input shape
    /// is too big for the host to accept" from "your input bytes are
    /// malformed." See `wit/wasi-cuda.wit` and
    /// [`crate::kernel_args::parse_argv`] for the contract.
    KernelArgsUnsupported = -10,
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
            AbiError::InvalidDimensions => "invalid_dimensions",
            AbiError::InvalidArgs => "invalid_args",
            AbiError::KernelArgsUnsupported => "kernel_args_unsupported",
        }
    }
}

/// Maximum PTX module length we'll accept from a single `load_ptx` call.
pub const MAX_PTX_BYTES: usize = 8 * 1024 * 1024;

/// Maximum number of kernels a single instance may keep alive.
pub const MAX_KERNELS_PER_INSTANCE: usize = 256;

/// Maximum threads-per-block product (`block_x * block_y * block_z`).
/// CUDA hardware cap on every device shipped since Kepler.
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// Maximum per-axis block dimension. The CUDA driver allows up to 1024 for
/// `block_x` / `block_y` and 64 for `block_z`; we cap each axis at 1024 and
/// rely on the [`MAX_THREADS_PER_BLOCK`] product check to enforce the z
/// constraint indirectly (any z > 64 paired with non-trivial x or y will
/// already exceed 1024 threads).
pub const MAX_BLOCK_DIM: u32 = 1024;

/// Maximum per-axis grid dimension. CUDA driver maximum for `grid_x` is
/// `2^31 - 1`; `grid_y` / `grid_z` are `2^16 - 1`, but we keep the cap
/// uniform at the larger value and let the driver enforce the per-axis
/// distinction (`cuLaunchKernel` returns `CUDA_ERROR_INVALID_VALUE`).
pub const MAX_GRID_DIM: u32 = i32::MAX as u32;

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
        assert_eq!(AbiError::InvalidDimensions.code(), -8);
        assert_eq!(AbiError::InvalidArgs.code(), -9);
        assert_eq!(AbiError::KernelArgsUnsupported.code(), -10);
    }

    #[test]
    fn abi_error_names_stable() {
        assert_eq!(AbiError::NotAvailable.name(), "not_available");
        assert_eq!(AbiError::MalformedPtx.name(), "malformed_ptx");
        assert_eq!(AbiError::InvalidDimensions.name(), "invalid_dimensions");
        assert_eq!(AbiError::InvalidArgs.name(), "invalid_args");
        assert_eq!(
            AbiError::KernelArgsUnsupported.name(),
            "kernel_args_unsupported"
        );
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
        assert!(MODULE.ends_with("@0.2.0"));
    }

    /// Pin the host MODULE string against drift from `wit/wasi-cuda.wit`.
    ///
    /// The WIT file is the authoritative spec; the host's import-module
    /// name has to carry the same `@x.y.z` segment so guests generated
    /// from the WIT can link. Parse the version out of the WIT
    /// `package wasi:cuda@x.y.z;` line and compare to the suffix of
    /// [`MODULE`]. If somebody bumps one without the other, this test
    /// trips before the linker error reaches downstream users.
    #[test]
    fn module_version_matches_wit_package_decl() {
        const WIT: &str = include_str!("../../../wit/wasi-cuda.wit");
        // Path is relative to this source file (`crates/tensor-wasm-wasi-gpu/src/abi.rs`):
        //   ../        -> crates/tensor-wasm-wasi-gpu/
        //   ../../     -> crates/
        //   ../../../  -> workspace root, where `wit/` lives.
        let pkg_line = WIT
            .lines()
            .find(|l| l.trim_start().starts_with("package wasi:cuda@"))
            .expect("wit/wasi-cuda.wit must declare `package wasi:cuda@x.y.z;`");
        let version = pkg_line
            .trim()
            .trim_start_matches("package wasi:cuda@")
            .trim_end_matches(';')
            .trim();
        assert!(
            !version.is_empty(),
            "could not parse a version out of: {pkg_line:?}"
        );
        let expected_suffix = format!("@{version}");
        assert!(
            MODULE.ends_with(&expected_suffix),
            "MODULE ({MODULE:?}) drifted from wit/wasi-cuda.wit \
             package version ({version:?}); update src/abi.rs::MODULE \
             or the WIT file so they agree."
        );
    }

    #[test]
    fn dimension_caps_are_plausible() {
        // Defensive sanity checks — bumping these accidentally would let
        // launch requests through that the driver will then reject in a
        // less actionable way.
        assert_eq!(MAX_THREADS_PER_BLOCK, 1024);
        assert_eq!(MAX_BLOCK_DIM, 1024);
        assert_eq!(MAX_GRID_DIM, i32::MAX as u32);
    }
}
