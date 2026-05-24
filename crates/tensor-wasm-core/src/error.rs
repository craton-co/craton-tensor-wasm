// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Project-wide error types.
//!
//! [`TensorWasmError`] is the single, unified error returned by every public API in
//! the TensorWasm workspace. Inner error sources are preserved via `#[source]` chains;
//! `std::io::Error` is wired via `#[from]`. Wasmtime, cust, and serialisation
//! errors are converted at their crate boundaries and surface here as
//! `WasmTrap`, `WasmCompile`, `CudaError`, and `Serialization` with string
//! contexts.
//!
//! String-carrying variants store their message as `Box<str>` rather than
//! `String` — error values are typically constructed once and then propagated
//! up the call stack untouched, so we trade away the spare capacity an owned
//! `String` carries (and shrink each variant's footprint by one pointer-sized
//! word). Callers that have a `String` should pass it via `.into()`; callers
//! with a `&str` should use `.into()` likewise.

use std::io;

use thiserror::Error;

use crate::types::TenantId;

/// The unified error type for every TensorWasm crate.
///
/// Variants are deliberately broad — host-level code matches on the variant to
/// classify failures into tenant-facing vs operator-facing responses. Inner
/// error sources are preserved via `#[source]` chains.
#[derive(Debug, Error)]
pub enum TensorWasmError {
    /// A call into the CUDA driver or runtime failed.
    #[error("CUDA error: {0}")]
    CudaError(Box<str>),

    /// A Wasm trap was triggered during execution (divide-by-zero, OOB access, ...).
    #[error("Wasm trap: {0}")]
    WasmTrap(Box<str>),

    /// Compiling Wasm bytes to native code failed.
    #[error("Wasm compile error: {0}")]
    WasmCompile(Box<str>),

    /// The instance exceeded its memory quota.
    #[error("memory exhausted: requested {requested} bytes, limit {limit}")]
    MemoryExhausted {
        /// Bytes the instance attempted to allocate.
        requested: u64,
        /// Bytes the tenant is allowed.
        limit: u64,
    },

    /// A GPU kernel exceeded its deadline.
    #[error("kernel timeout after {elapsed_ms} ms (deadline {deadline_ms} ms)")]
    KernelTimeout {
        /// Time elapsed before timeout was enforced.
        elapsed_ms: u64,
        /// Configured per-kernel deadline in milliseconds.
        deadline_ms: u64,
    },

    /// An instance accessed memory or resources belonging to another tenant.
    #[error("tenant isolation violation: tenant {tenant_id} attempted to access {resource}")]
    TenantIsolationViolation {
        /// Offending tenant identifier.
        tenant_id: TenantId,
        /// Free-form description of the resource that was accessed out of scope.
        resource: Box<str>,
    },

    /// An I/O error from the host OS.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// A (de)serialisation error.
    #[error("serialization error: {0}")]
    Serialization(Box<str>),
}

impl TensorWasmError {
    /// Returns `true` if the error is plausibly transient and retrying may succeed
    /// (timeouts, certain I/O conditions). Used by the API layer to decide
    /// between `503 Service Unavailable` (retryable) and `500 Internal Server Error`.
    ///
    /// `WasmCompile` and `TenantIsolationViolation` are *not* retryable —
    /// recompiling identical bytes will fail identically, and an isolation
    /// breach is a hard policy decision rather than a transient condition.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            TensorWasmError::KernelTimeout { .. } | TensorWasmError::Io(_) | TensorWasmError::MemoryExhausted { .. }
        )
    }

    /// Returns a stable, machine-readable variant name (used in metrics labels).
    pub fn kind(&self) -> &'static str {
        match self {
            TensorWasmError::CudaError(_) => "cuda",
            TensorWasmError::WasmTrap(_) => "wasm_trap",
            TensorWasmError::WasmCompile(_) => "wasm_compile",
            TensorWasmError::MemoryExhausted { .. } => "memory_exhausted",
            TensorWasmError::KernelTimeout { .. } => "kernel_timeout",
            TensorWasmError::TenantIsolationViolation { .. } => "tenant_isolation",
            TensorWasmError::Io(_) => "io",
            TensorWasmError::Serialization(_) => "serialization",
        }
    }
}

/// Convenience alias used throughout the workspace.
pub type Result<T, E = TensorWasmError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_cuda() {
        let e = TensorWasmError::CudaError("ctx not current".into());
        assert_eq!(format!("{e}"), "CUDA error: ctx not current");
    }

    #[test]
    fn display_memory_exhausted_fields() {
        let e = TensorWasmError::MemoryExhausted {
            requested: 1024,
            limit: 512,
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn display_kernel_timeout_fields() {
        let e = TensorWasmError::KernelTimeout {
            elapsed_ms: 1500,
            deadline_ms: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("1500"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn display_isolation_violation() {
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        let s = e.to_string();
        assert!(s.contains("42"));
        assert!(s.contains("/dev/shm/other-tenant"));
    }

    #[test]
    fn display_wasm_trap() {
        let e = TensorWasmError::WasmTrap("unreachable".into());
        assert_eq!(format!("{e}"), "Wasm trap: unreachable");
    }

    #[test]
    fn display_wasm_compile() {
        let e = TensorWasmError::WasmCompile("bad opcode".into());
        assert_eq!(format!("{e}"), "Wasm compile error: bad opcode");
    }

    #[test]
    fn display_io() {
        let e = TensorWasmError::Io(io::Error::new(io::ErrorKind::NotFound, "missing"));
        let s = format!("{e}");
        assert!(s.contains("I/O error"));
        assert!(s.contains("missing"));
    }

    #[test]
    fn display_serialization() {
        let e = TensorWasmError::Serialization("bad json".into());
        assert_eq!(format!("{e}"), "serialization error: bad json");
    }

    #[test]
    fn io_from_conversion() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let e: TensorWasmError = io_err.into();
        assert!(matches!(e, TensorWasmError::Io(_)));
        assert_eq!(e.kind(), "io");
    }

    #[test]
    fn kind_stable_names() {
        assert_eq!(TensorWasmError::CudaError("x".into()).kind(), "cuda");
        assert_eq!(TensorWasmError::WasmTrap("x".into()).kind(), "wasm_trap");
        assert_eq!(TensorWasmError::WasmCompile("x".into()).kind(), "wasm_compile");
        assert_eq!(TensorWasmError::Serialization("x".into()).kind(), "serialization");
    }

    #[test]
    fn retryable_classification() {
        assert!(TensorWasmError::KernelTimeout {
            elapsed_ms: 1,
            deadline_ms: 1
        }
        .is_retryable());
        assert!(TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1
        }
        .is_retryable());
        assert!(!TensorWasmError::WasmTrap("x".into()).is_retryable());
        assert!(!TensorWasmError::CudaError("x".into()).is_retryable());
    }

    #[test]
    fn wasm_compile_is_not_retryable() {
        // Recompiling the same Wasm bytes will fail identically — never retry.
        let e = TensorWasmError::WasmCompile("invalid opcode 0xfe".into());
        assert!(
            !e.is_retryable(),
            "WasmCompile must not be flagged as retryable",
        );
    }

    #[test]
    fn tenant_isolation_violation_is_not_retryable() {
        // An isolation breach is a hard policy decision — retrying is a security
        // bug, not a recovery strategy.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(1),
            resource: "/dev/shm/foreign".into(),
        };
        assert!(
            !e.is_retryable(),
            "TenantIsolationViolation must not be flagged as retryable",
        );
    }

    #[test]
    fn string_construction_via_into() {
        // String, &str, and Box<str> should all convert into the inner Box<str>
        // via the standard library `From` impls — exercise each path so future
        // refactors that break ergonomics are caught here.
        let from_string: Box<str> = String::from("hello").into();
        let from_str: Box<str> = "hello".into();
        let from_box: Box<str> = Box::<str>::from("hello");
        let e1 = TensorWasmError::CudaError(from_string);
        let e2 = TensorWasmError::CudaError(from_str);
        let e3 = TensorWasmError::CudaError(from_box);
        assert_eq!(format!("{e1}"), "CUDA error: hello");
        assert_eq!(format!("{e2}"), "CUDA error: hello");
        assert_eq!(format!("{e3}"), "CUDA error: hello");
    }
}
