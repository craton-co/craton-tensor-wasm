//! Project-wide error types.
//!
//! [`BaliError`] is the single, unified error returned by every public API in
//! the Bali workspace. Inner error sources are preserved via `#[source]` chains;
//! `std::io::Error` is wired via `#[from]`. Wasmtime, cust, and serialisation
//! errors are converted at their crate boundaries and surface here as
//! `WasmTrap`, `WasmCompile`, `CudaError`, and `Serialization` with string
//! contexts.

use std::io;

use thiserror::Error;

use crate::types::TenantId;

/// The unified error type for every Bali crate.
///
/// Variants are deliberately broad — host-level code matches on the variant to
/// classify failures into tenant-facing vs operator-facing responses. Inner
/// error sources are preserved via `#[source]` chains.
#[derive(Debug, Error)]
pub enum BaliError {
    /// A call into the CUDA driver or runtime failed.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// A Wasm trap was triggered during execution (divide-by-zero, OOB access, ...).
    #[error("Wasm trap: {0}")]
    WasmTrap(String),

    /// Compiling Wasm bytes to native code failed.
    #[error("Wasm compile error: {0}")]
    WasmCompile(String),

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
        resource: String,
    },

    /// An I/O error from the host OS.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// A (de)serialisation error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl BaliError {
    /// Returns `true` if the error is plausibly transient and retrying may succeed
    /// (timeouts, certain I/O conditions). Used by the API layer to decide
    /// between `503 Service Unavailable` (retryable) and `500 Internal Server Error`.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            BaliError::KernelTimeout { .. } | BaliError::Io(_) | BaliError::MemoryExhausted { .. }
        )
    }

    /// Returns a stable, machine-readable variant name (used in metrics labels).
    pub fn kind(&self) -> &'static str {
        match self {
            BaliError::CudaError(_) => "cuda",
            BaliError::WasmTrap(_) => "wasm_trap",
            BaliError::WasmCompile(_) => "wasm_compile",
            BaliError::MemoryExhausted { .. } => "memory_exhausted",
            BaliError::KernelTimeout { .. } => "kernel_timeout",
            BaliError::TenantIsolationViolation { .. } => "tenant_isolation",
            BaliError::Io(_) => "io",
            BaliError::Serialization(_) => "serialization",
        }
    }
}

/// Convenience alias used throughout the workspace.
pub type Result<T, E = BaliError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_cuda() {
        let e = BaliError::CudaError("ctx not current".into());
        assert_eq!(format!("{e}"), "CUDA error: ctx not current");
    }

    #[test]
    fn display_memory_exhausted_fields() {
        let e = BaliError::MemoryExhausted {
            requested: 1024,
            limit: 512,
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn display_kernel_timeout_fields() {
        let e = BaliError::KernelTimeout {
            elapsed_ms: 1500,
            deadline_ms: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("1500"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn display_isolation_violation() {
        let e = BaliError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        let s = e.to_string();
        assert!(s.contains("42"));
        assert!(s.contains("/dev/shm/other-tenant"));
    }

    #[test]
    fn display_wasm_trap() {
        let e = BaliError::WasmTrap("unreachable".into());
        assert_eq!(format!("{e}"), "Wasm trap: unreachable");
    }

    #[test]
    fn display_wasm_compile() {
        let e = BaliError::WasmCompile("bad opcode".into());
        assert_eq!(format!("{e}"), "Wasm compile error: bad opcode");
    }

    #[test]
    fn display_io() {
        let e = BaliError::Io(io::Error::new(io::ErrorKind::NotFound, "missing"));
        let s = format!("{e}");
        assert!(s.contains("I/O error"));
        assert!(s.contains("missing"));
    }

    #[test]
    fn display_serialization() {
        let e = BaliError::Serialization("bad json".into());
        assert_eq!(format!("{e}"), "serialization error: bad json");
    }

    #[test]
    fn io_from_conversion() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let e: BaliError = io_err.into();
        assert!(matches!(e, BaliError::Io(_)));
        assert_eq!(e.kind(), "io");
    }

    #[test]
    fn kind_stable_names() {
        assert_eq!(BaliError::CudaError("x".into()).kind(), "cuda");
        assert_eq!(BaliError::WasmTrap("x".into()).kind(), "wasm_trap");
        assert_eq!(BaliError::WasmCompile("x".into()).kind(), "wasm_compile");
        assert_eq!(BaliError::Serialization("x".into()).kind(), "serialization");
    }

    #[test]
    fn retryable_classification() {
        assert!(BaliError::KernelTimeout {
            elapsed_ms: 1,
            deadline_ms: 1
        }
        .is_retryable());
        assert!(BaliError::MemoryExhausted {
            requested: 1,
            limit: 1
        }
        .is_retryable());
        assert!(!BaliError::WasmTrap("x".into()).is_retryable());
        assert!(!BaliError::CudaError("x".into()).is_retryable());
    }
}
