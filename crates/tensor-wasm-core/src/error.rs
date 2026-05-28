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
//!
//! # Display sanitisation
//!
//! The `CudaError`, `WasmTrap`, `WasmCompile`, and `Serialization` variants
//! carry inner strings produced by third-party crates (wasmtime, cust, serde,
//! ...). Those messages can leak host filesystem paths, raw pointer
//! addresses, or fragments of tenant-supplied source bytes — none of which is
//! safe to render into a response body or end-user log line. The
//! `Display` impls for those four variants therefore emit a stable, opaque
//! label only (e.g. `"cuda driver call failed"`). The inner string is still
//! reachable through `Debug` formatting for operator-side diagnostic logs and
//! through the `inner()` accessor for callers that need the raw text in a
//! trusted context. Variants whose fields are structured (`MemoryExhausted`,
//! `KernelTimeout`, `TenantIsolationViolation`) format normally — those
//! fields are integers and tenant-controlled identifiers, not opaque strings
//! from a vendor crate.

use std::io;

use thiserror::Error;

use crate::types::TenantId;

/// Returns a short, stable name for an [`std::io::ErrorKind`] suitable for
/// the sanitised `Display` of [`TensorWasmError::Io`].
///
/// Uses `Debug` formatting to surface the variant identifier
/// (`"NotFound"`, `"PermissionDenied"`, ...) rather than the prose
/// `Display` form (`"entity not found"`, ...) — the variant name is what
/// dashboards, alert rules, and operators grep for. The set of returned
/// strings is closed and tracks the upstream `std::io::ErrorKind` enum.
fn io_kind_name(err: &io::Error) -> String {
    format!("{:?}", err.kind())
}

/// Returns `true` if an [`std::io::ErrorKind`] is plausibly transient and
/// the caller should retry the I/O operation. Used by
/// [`TensorWasmError::is_retryable`] to differentiate `Io(_)` errors that
/// should map to `503 Service Unavailable` (retryable) from those that
/// should map to `500 Internal Server Error` or a tenant-side 4xx (hard
/// miss).
///
/// The retryable set is intentionally narrow — every kind not listed
/// here defaults to non-retryable so a future addition to
/// `std::io::ErrorKind` does not silently flip a hard error into a
/// `503`. Kinds in the retryable set:
///
/// * [`io::ErrorKind::WouldBlock`] — non-blocking socket / fd would
///   block; the canonical retry-now kind.
/// * [`io::ErrorKind::TimedOut`] — operation deadline expired; the
///   underlying resource may recover.
/// * [`io::ErrorKind::Interrupted`] — signal-interrupted syscall
///   (`EINTR`); restarting almost always succeeds.
/// * [`io::ErrorKind::WriteZero`] — short write; a follow-up write may
///   drain the rest.
/// * [`io::ErrorKind::ConnectionReset`], [`io::ErrorKind::ConnectionAborted`],
///   [`io::ErrorKind::BrokenPipe`] — peer dropped the connection;
///   re-establishing and retrying may succeed.
fn is_retryable_io_kind(kind: io::ErrorKind) -> bool {
    matches!(
        kind,
        io::ErrorKind::WouldBlock
            | io::ErrorKind::TimedOut
            | io::ErrorKind::Interrupted
            | io::ErrorKind::WriteZero
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
    )
}

/// The unified error type for every TensorWasm crate.
///
/// Variants are deliberately broad — host-level code matches on the variant to
/// classify failures into tenant-facing vs operator-facing responses. Inner
/// error sources are preserved via `#[source]` chains.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` arms so new variants
/// added in a future minor release do not break downstream code. The enum has
/// no `Default` impl, so the `Self { .., ..Default::default() }` pattern does
/// not apply here — construct variants explicitly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorWasmError {
    /// A call into the CUDA driver or runtime failed.
    ///
    /// The inner string is the underlying vendor message (cust / cuda-oxide /
    /// cudarc); it can contain raw pointer addresses and host paths, so it is
    /// deliberately omitted from `Display` and only surfaced via `Debug`.
    #[error("cuda driver call failed")]
    CudaError(Box<str>),

    /// A Wasm trap was triggered during execution (divide-by-zero, OOB access, ...).
    ///
    /// The inner string is wasmtime's trap message, which may include host
    /// instruction-pointer addresses; it is omitted from `Display` and only
    /// surfaced via `Debug`.
    #[error("wasm trap")]
    WasmTrap(Box<str>),

    /// Compiling Wasm bytes to native code failed.
    ///
    /// The inner string is wasmtime's compiler diagnostic, which can echo
    /// fragments of the tenant-supplied module bytes; it is omitted from
    /// `Display` and only surfaced via `Debug`.
    #[error("wasm compile failed")]
    WasmCompile(Box<str>),

    /// The instance exceeded its memory quota.
    #[error("memory exhausted: requested {requested} bytes, limit {limit}")]
    MemoryExhausted {
        /// Bytes the instance attempted to allocate.
        requested: u64,
        /// Bytes the tenant is allowed.
        limit: u64,
    },

    /// The tenant exceeded its per-tenant GPU memory cap.
    ///
    /// Distinct from [`Self::MemoryExhausted`] (the host-side / CPU
    /// quota) so dashboards, alerts, and tenant-facing error responses
    /// can distinguish "host RAM cap tripped" from "GPU memory cap
    /// tripped". The cap is the value passed to
    /// `TenantContextBuilder::with_gpu_memory_bytes_cap` (and recorded
    /// on `TenantContext::gpu_memory_bytes_cap`); `current` is the
    /// tenant's `gpu_bytes_in_use` *before* the rejected allocation
    /// would have been added.
    ///
    /// v0.3.7 enforcement is in-process — the allocator path
    /// (`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context`)
    /// consults the counter on every `UnifiedBuffer::new_on` and refuses
    /// to allocate when the cap would be exceeded. v0.4 will additionally
    /// pin a CUDA driver-level cap via
    /// `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`
    /// (CUDA 11.2+).
    #[error("gpu memory exhausted: requested {requested} bytes, limit {limit}, current {current}")]
    GpuMemoryExhausted {
        /// Bytes the tenant attempted to allocate on the GPU.
        requested: u64,
        /// Per-tenant GPU memory cap in bytes
        /// (`TenantContext::gpu_memory_bytes_cap.unwrap()`).
        limit: u64,
        /// Bytes the tenant had already reserved on the GPU at the time
        /// the allocation was rejected. The would-be new total is
        /// `current + requested`.
        current: u64,
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
    ///
    /// **`Display` is intentionally sanitised.** The HTTP layer surfaces error
    /// `Display` to tenants as part of 4xx response bodies; including
    /// `tenant_id` or `resource` here would let tenant A learn tenant B's id
    /// and on-disk path layout. The structured fields remain on the variant
    /// so server-side `tracing::error!(?err, ...)` (which uses `Debug`) still
    /// gets the full context for operator triage.
    #[error("tenant isolation violation")]
    TenantIsolationViolation {
        /// Offending tenant identifier. Logged server-side via `Debug`; never
        /// included in `Display`.
        tenant_id: TenantId,
        /// Free-form description of the resource that was accessed out of
        /// scope. Logged server-side via `Debug`; never included in `Display`.
        resource: Box<str>,
    },

    /// An I/O error from the host OS.
    ///
    /// **`Display` is intentionally sanitised.** `std::io::Error::Display`
    /// echoes the underlying OS error string and — when the error was built
    /// from a filesystem operation — frequently includes the offending path.
    /// Both are unsafe to render into a tenant-facing 4xx body. The
    /// sanitised `Display` here surfaces only the `ErrorKind` discriminant
    /// (e.g. `"NotFound"`, `"PermissionDenied"`) which is a closed,
    /// non-sensitive enum. The inner `io::Error` remains reachable through
    /// `Debug` and via [`std::error::Error::source`] for server-side
    /// operator logs.
    #[error("I/O error (kind: {})", io_kind_name(.0))]
    Io(#[from] io::Error),

    /// A (de)serialisation error.
    ///
    /// The inner string is the serde / format-crate message and may quote
    /// untrusted input bytes verbatim; it is omitted from `Display` and only
    /// surfaced via `Debug`.
    #[error("serialization error")]
    Serialization(Box<str>),

    /// A snapshot was rejected by the reader's freshness check
    /// ([`max_age`](../tensor_wasm_snapshot/reader/struct.SnapshotReader.html#method.with_max_age)).
    ///
    /// Distinct from [`Self::Serialization`] so dashboards, alerts, and
    /// audit logs can pin replay-attempt rejections separately from
    /// generic format errors. All three fields are millisecond
    /// timestamps (or a millisecond duration); the variant carries
    /// enough context for an operator to triage a clock-skew false
    /// positive without consulting any other field on the snapshot.
    ///
    /// Introduced in T9 alongside
    /// [`tensor_wasm_snapshot::reader::SnapshotReader::with_max_age`].
    /// The freshness check is **opt-in** — operators must set
    /// `max_age` on their reader to receive this error; the default
    /// reader continues to accept arbitrarily old snapshots for
    /// backward compatibility with v0.3.x captures.
    #[error(
        "snapshot too old: created {created_unix_ms} ms, now {now_unix_ms} ms, \
         max age {max_age_ms} ms"
    )]
    SnapshotTooOld {
        /// Wall-clock time the snapshot was captured, in milliseconds
        /// since the Unix epoch (from
        /// `SnapshotMetadata::created_unix_ms`).
        created_unix_ms: u64,
        /// Reader-side wall-clock time at the moment freshness was
        /// checked, in milliseconds since the Unix epoch.
        now_unix_ms: u64,
        /// Maximum age the reader was configured to accept, in
        /// milliseconds (`max_age` as passed to
        /// `SnapshotReader::with_max_age`).
        max_age_ms: u64,
    },
}

impl TensorWasmError {
    /// Returns `true` if the error is plausibly transient and retrying may succeed
    /// (timeouts, certain I/O conditions). Used by the API layer to decide
    /// between `503 Service Unavailable` (retryable) and `500 Internal Server Error`.
    ///
    /// `WasmCompile` and `TenantIsolationViolation` are *not* retryable —
    /// recompiling identical bytes will fail identically, and an isolation
    /// breach is a hard policy decision rather than a transient condition.
    ///
    /// `Io(_)` is classified by inspecting [`std::io::Error::kind`]: transient
    /// kinds (`WouldBlock`, `TimedOut`, `Interrupted`, `WriteZero`, the
    /// connection-reset family) flag as retryable, while hard-miss kinds
    /// (`NotFound`, `PermissionDenied`, `AlreadyExists`, ...) do not. The
    /// per-kind classifier is more useful than a blanket `Io(_) -> true`
    /// because the API layer otherwise returns `503` for hard 404-class
    /// failures and the CLI's retry loop spins on doomed requests.
    pub fn is_retryable(&self) -> bool {
        match self {
            TensorWasmError::KernelTimeout { .. }
            | TensorWasmError::MemoryExhausted { .. }
            | TensorWasmError::GpuMemoryExhausted { .. } => true,
            TensorWasmError::Io(err) => is_retryable_io_kind(err.kind()),
            _ => false,
        }
    }

    /// Returns the inner diagnostic string for the four variants that wrap a
    /// vendor message (`CudaError`, `WasmTrap`, `WasmCompile`,
    /// `Serialization`). For every other variant — and for `Io`, whose
    /// `std::io::Error` source is already reachable via [`std::error::Error::source`]
    /// — this returns `None`.
    ///
    /// This accessor exists so server-side operator logs can record the full
    /// vendor message even though `Display` deliberately omits it. **Never
    /// expose the returned string to end-users / response bodies**: that is
    /// precisely the leak surface the sanitised `Display` impls protect against.
    pub fn inner(&self) -> Option<&str> {
        match self {
            TensorWasmError::CudaError(s)
            | TensorWasmError::WasmTrap(s)
            | TensorWasmError::WasmCompile(s)
            | TensorWasmError::Serialization(s) => Some(s),
            _ => None,
        }
    }

    /// Returns a stable, machine-readable variant name (used in metrics labels).
    pub fn kind(&self) -> &'static str {
        match self {
            TensorWasmError::CudaError(_) => "cuda",
            TensorWasmError::WasmTrap(_) => "wasm_trap",
            TensorWasmError::WasmCompile(_) => "wasm_compile",
            TensorWasmError::MemoryExhausted { .. } => "memory_exhausted",
            TensorWasmError::GpuMemoryExhausted { .. } => "gpu_memory_exhausted",
            TensorWasmError::KernelTimeout { .. } => "kernel_timeout",
            TensorWasmError::TenantIsolationViolation { .. } => "tenant_isolation",
            TensorWasmError::Io(_) => "io",
            TensorWasmError::Serialization(_) => "serialization",
            TensorWasmError::SnapshotTooOld { .. } => "snapshot_too_old",
        }
    }
}

/// Convenience alias used throughout the workspace.
pub type Result<T, E = TensorWasmError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_cuda_is_opaque() {
        // Display must NOT echo the inner vendor string — it can carry pointer
        // addresses and host paths. The inner string is reachable via inner()
        // and Debug for operator-side logging.
        let e = TensorWasmError::CudaError("ctx not current at 0x7ffe0000".into());
        assert_eq!(format!("{e}"), "cuda driver call failed");
        assert_eq!(e.inner(), Some("ctx not current at 0x7ffe0000"));
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
    fn display_isolation_violation_is_sanitised() {
        // Display is surfaced to tenants via 4xx response bodies; it must NOT
        // leak the offending tenant id or the resource path (the latter often
        // encodes another tenant's on-disk layout). See the regression test in
        // `tests/error_display_does_not_leak.rs` for the full leak-shaped
        // assertions; this inline test pins the exact Display string.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        assert_eq!(e.to_string(), "tenant isolation violation");
    }

    #[test]
    fn debug_isolation_violation_still_carries_fields() {
        // Debug IS used for server-side `tracing::error!(?err)` and must
        // continue to expose the structured fields for operator triage —
        // only Display is sanitised.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        let dbg = format!("{e:?}");
        assert!(dbg.contains("42"), "Debug should expose tenant_id: {dbg}");
        assert!(
            dbg.contains("/dev/shm/other-tenant"),
            "Debug should expose resource: {dbg}",
        );
    }

    #[test]
    fn display_wasm_trap_is_opaque() {
        let e = TensorWasmError::WasmTrap("unreachable at ip 0x401234".into());
        assert_eq!(format!("{e}"), "wasm trap");
        assert_eq!(e.inner(), Some("unreachable at ip 0x401234"));
    }

    #[test]
    fn display_wasm_compile_is_opaque() {
        let e = TensorWasmError::WasmCompile("bad opcode 0xfe in /tmp/mod.wasm".into());
        assert_eq!(format!("{e}"), "wasm compile failed");
        assert_eq!(e.inner(), Some("bad opcode 0xfe in /tmp/mod.wasm"));
    }

    #[test]
    fn display_io() {
        // The sanitised `Display` surfaces the `ErrorKind` variant name but
        // MUST NOT echo the inner OS string (which routinely carries
        // filesystem paths and host-specific diagnostic text). See the
        // regression test in `tests/error_display_does_not_leak.rs` for
        // the leak-shaped assertions; this inline test pins the exact
        // shape of the sanitised render.
        let e = TensorWasmError::Io(io::Error::new(io::ErrorKind::NotFound, "missing"));
        let s = format!("{e}");
        assert!(s.contains("I/O error"));
        assert!(s.contains("NotFound"));
        assert!(
            !s.contains("missing"),
            "Display must not leak the inner OS string, got: {s}",
        );
    }

    #[test]
    fn display_serialization_is_opaque() {
        let e = TensorWasmError::Serialization("bad json at byte 42: 'secret-tenant-key'".into());
        assert_eq!(format!("{e}"), "serialization error");
        assert_eq!(e.inner(), Some("bad json at byte 42: 'secret-tenant-key'"),);
    }

    #[test]
    fn inner_returns_none_for_structured_variants() {
        // Variants without an inner vendor string return None — there's
        // nothing to hand to the operator log beyond the Display output.
        let mem = TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1,
        };
        let kt = TensorWasmError::KernelTimeout {
            elapsed_ms: 1,
            deadline_ms: 1,
        };
        let iso = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(1),
            resource: "x".into(),
        };
        let io = TensorWasmError::Io(io::Error::other("x"));
        assert!(mem.inner().is_none());
        assert!(kt.inner().is_none());
        assert!(iso.inner().is_none());
        assert!(io.inner().is_none());
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
        assert_eq!(
            TensorWasmError::WasmCompile("x".into()).kind(),
            "wasm_compile"
        );
        assert_eq!(
            TensorWasmError::Serialization("x".into()).kind(),
            "serialization"
        );
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
    fn io_error_is_retryable_wouldblock_true() {
        // `WouldBlock` is the canonical retry-now I/O kind — pending a
        // socket send buffer or a non-blocking file descriptor. The
        // `is_retryable()` classifier must surface it as retryable so
        // the API layer renders `503 Service Unavailable` rather than
        // `500 Internal Server Error` (and so the CLI's retry loop
        // engages instead of bailing out).
        let e = TensorWasmError::Io(io::Error::from(io::ErrorKind::WouldBlock));
        assert!(
            e.is_retryable(),
            "Io(WouldBlock) must be classified as retryable",
        );
    }

    #[test]
    fn io_error_is_retryable_notfound_false() {
        // `NotFound` is a hard miss — retrying the same path will fail
        // identically. The classifier must NOT flag it as retryable;
        // otherwise the API layer would return `503 Service
        // Unavailable` for what should be a permanent `404`-class
        // failure and the CLI's retry loop would spin on a doomed
        // request.
        let e = TensorWasmError::Io(io::Error::from(io::ErrorKind::NotFound));
        assert!(
            !e.is_retryable(),
            "Io(NotFound) must NOT be classified as retryable",
        );
    }

    #[test]
    fn string_construction_via_into() {
        // String, &str, and Box<str> should all convert into the inner Box<str>
        // via the standard library `From` impls — exercise each path so future
        // refactors that break ergonomics are caught here. Display is opaque
        // (see `display_cuda_is_opaque`), so we verify the inner string was
        // captured via `inner()` rather than re-asserting the format string.
        let from_string: Box<str> = String::from("hello").into();
        let from_str: Box<str> = "hello".into();
        let from_box: Box<str> = Box::<str>::from("hello");
        let e1 = TensorWasmError::CudaError(from_string);
        let e2 = TensorWasmError::CudaError(from_str);
        let e3 = TensorWasmError::CudaError(from_box);
        assert_eq!(e1.inner(), Some("hello"));
        assert_eq!(e2.inner(), Some("hello"));
        assert_eq!(e3.inner(), Some("hello"));
    }
}
