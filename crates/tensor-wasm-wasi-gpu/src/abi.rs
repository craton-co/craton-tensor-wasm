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
        const WIT: &str = include_str!("../wit/wasi-cuda.wit");
        // Path is relative to this source file (`crates/tensor-wasm-wasi-gpu/src/abi.rs`):
        //   ../  -> crates/tensor-wasm-wasi-gpu/, where the in-crate
        //          `wit/wasi-cuda.wit` copy lives. Kept in-crate so the
        //          published tarball is self-contained (`cargo publish`
        //          rejects paths that escape the crate root).
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

    /// Guard against stray git merge-conflict markers in the in-crate WIT
    /// contract.
    ///
    /// `wit/wasi-cuda.wit` ships inside the published crate tarball and is
    /// consumed verbatim by external `wit-bindgen` users — any leftover
    /// `<<<<<<<` / `=======` / `>>>>>>>` line will break those downstream
    /// builds long before this crate's own host tests notice. The original
    /// `module_version_matches_wit_package_decl` test only inspects the
    /// `package` line, so a marker further down the file slips through.
    ///
    /// We use byte-pattern substrings rather than full marker strings (e.g.
    /// `"<<<<<<< HEAD"`) so the test catches markers regardless of which
    /// branch label git chose. The markers themselves are constructed via
    /// `str::repeat` so this test file is safe to include in any future
    /// merge — the literal source here never contains seven of any conflict
    /// character in a row.
    #[test]
    fn wit_file_has_no_merge_conflict_markers() {
        const WIT: &str = include_str!("../wit/wasi-cuda.wit");
        let ours = "<".repeat(7);
        let sep = "=".repeat(7);
        let theirs = ">".repeat(7);
        for (label, marker) in [
            ("ours (<<<<<<<)", ours.as_str()),
            ("separator (=======)", sep.as_str()),
            ("theirs (>>>>>>>)", theirs.as_str()),
        ] {
            for (lineno, line) in WIT.lines().enumerate() {
                assert!(
                    !line.contains(marker),
                    "wit/wasi-cuda.wit contains an unresolved git merge \
                     conflict marker {label} on line {} — resolve it before \
                     publishing or downstream `wit-bindgen` consumers will \
                     fail to parse the contract. Offending line: {line:?}",
                    lineno + 1,
                );
            }
        }
    }

    /// Drift guard: the in-crate authoritative `wit/wasi-cuda.wit` and the
    /// workspace-root mirror `wit/wasi-cuda.wit` must carry a byte-identical
    /// `interface host { ... }` body.
    ///
    /// Two copies of the cuda WIT exist:
    ///   * `crates/tensor-wasm-wasi-gpu/wit/wasi-cuda.wit` — authoritative,
    ///     `include_str!`'d by the tests above and shipped in the published
    ///     crate tarball.
    ///   * `<workspace-root>/wit/wasi-cuda.wit` — mirror for tooling that
    ///     consumes WIT from the workspace root.
    ///
    /// The file *headers* legitimately differ (the in-crate copy documents
    /// both `wasi:cuda` and `wasi:tensor`; the mirror is cuda-only), so we
    /// compare only the `interface host` body — the part guests actually
    /// bind against. If the mirror drifts (a signature/error-code edit
    /// applied to one copy but not the other) this trips before downstream
    /// `wit-bindgen` consumers see two different contracts.
    ///
    /// This is a *runtime* test (reads both files via `CARGO_MANIFEST_DIR`)
    /// rather than a compile-time `include_str!("../../../wit/...")`: the
    /// root path escapes the crate root, which `cargo publish` rejects when
    /// packaging the tarball.
    #[test]
    fn cuda_wit_mirror_interface_body_matches() {
        use std::path::Path;

        /// Extract the `interface host { ... }` block (inclusive of the
        /// opening `interface host {` line through its matching `}`), with
        /// leading/trailing whitespace trimmed, so header differences and
        /// surrounding blank lines do not register as drift.
        fn interface_host_body(src: &str) -> String {
            let start = src
                .find("interface host {")
                .expect("WIT must declare `interface host {`");
            let rest = &src[start..];
            // Walk brace depth to find the matching close brace.
            let mut depth = 0usize;
            let mut end = None;
            for (idx, ch) in rest.char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(idx + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = end.expect("interface host block must be brace-balanced");
            rest[..end].trim().to_string()
        }

        let manifest = env!("CARGO_MANIFEST_DIR");
        let in_crate_path = Path::new(manifest).join("wit").join("wasi-cuda.wit");
        // CARGO_MANIFEST_DIR is `<root>/crates/tensor-wasm-wasi-gpu`; the
        // workspace root is two levels up.
        let root_path = Path::new(manifest)
            .join("..")
            .join("..")
            .join("wit")
            .join("wasi-cuda.wit");

        let in_crate = std::fs::read_to_string(&in_crate_path)
            .unwrap_or_else(|e| panic!("read {in_crate_path:?}: {e}"));
        let root = std::fs::read_to_string(&root_path)
            .unwrap_or_else(|e| panic!("read {root_path:?}: {e}"));

        assert_eq!(
            interface_host_body(&in_crate),
            interface_host_body(&root),
            "wasi-cuda.wit `interface host` body drifted between the in-crate \
             authoritative copy ({in_crate_path:?}) and the workspace-root \
             mirror ({root_path:?}); update both so guests bind one contract."
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
