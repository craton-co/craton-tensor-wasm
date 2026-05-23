//! `TenantContext`: per-tenant CUDA context + stream + memory pool.
//!
//! A `TenantContext` is the runtime handle that ties a [`TenantId`] to the GPU
//! resources reserved for that tenant: a CUDA stream identifier, an isolation
//! policy ([`IsolationKind`]), and a memory quota that the scheduler enforces
//! before kernels are dispatched. Construction goes through
//! [`TenantContextBuilder`] so callers can opt into individual fields without
//! a 5-argument constructor.
//!
//! Under the `cuda` feature, each `ContextIsolated` tenant additionally owns a
//! real `cust::context::Context`; without that feature (the default on
//! CUDA-less hosts), the field collapses to a unit stub so the rest of the
//! crate compiles and tests run unchanged.

use std::sync::atomic::{AtomicU64, Ordering};

use bali_core::error::BaliError;
use bali_core::types::TenantId;

/// How aggressively a tenant's GPU work is separated from other tenants'.
///
/// The variants mirror the levels exposed by `bali-mem::isolation::IsolationLevel`,
/// but live here as a separate type so this crate can be consumed without
/// pulling in the Wasmtime-dependent memory crate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsolationKind {
    /// All tenants share the default CUDA context and stream.
    ///
    /// Cheap to spawn but unsuitable for multi-tenant untrusted workloads.
    Shared,
    /// Each tenant gets its own CUDA stream; contexts are shared.
    ///
    /// Default for multi-tenant deployments — prevents kernel-ordering
    /// accidents without paying the cost of per-tenant context creation.
    #[default]
    StreamIsolated,
    /// Each tenant gets its own CUDA context (via MPS when available, or
    /// `cuCtxCreate` otherwise).
    ContextIsolated,
}

impl IsolationKind {
    /// Stable, human-readable name (used in span attributes and metrics).
    pub fn name(self) -> &'static str {
        match self {
            IsolationKind::Shared => "shared",
            IsolationKind::StreamIsolated => "stream_isolated",
            IsolationKind::ContextIsolated => "context_isolated",
        }
    }
}

impl std::fmt::Display for IsolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-tenant runtime handle: identity, isolation level, stream, and quota.
///
/// Instances are constructed through [`TenantContextBuilder`] and then placed
/// into the [`crate::TenantRegistry`]. The byte-counter methods
/// ([`Self::consume_bytes`] / [`Self::release_bytes`]) are lock-free and safe
/// to call from any thread; quota enforcement happens at the point of
/// allocation, not asynchronously.
#[derive(Debug)]
pub struct TenantContext {
    tenant_id: TenantId,
    isolation: IsolationKind,
    stream_id: u64,
    memory_quota_bytes: u64,
    bytes_in_use: AtomicU64,

    // Real `cust::context::Context` under the `cuda` feature; otherwise a
    // unit stub so the rest of the crate compiles on CUDA-less hosts.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    cu_context: Option<cust::context::Context>,
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    cu_context: (),
}

impl TenantContext {
    /// Start a builder for a tenant with the given identifier.
    pub fn builder(tenant_id: TenantId) -> TenantContextBuilder {
        TenantContextBuilder::new(tenant_id)
    }

    /// Tenant identifier this context belongs to.
    pub fn id(&self) -> TenantId {
        self.tenant_id
    }

    /// Isolation level configured for this tenant.
    pub fn isolation(&self) -> IsolationKind {
        self.isolation
    }

    /// Stream identifier (logical handle; the actual `CUstream` lives in
    /// `bali-mem` / `bali-wasi-gpu` and is keyed by this value).
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Total bytes the tenant is permitted to allocate concurrently.
    pub fn quota(&self) -> u64 {
        self.memory_quota_bytes
    }

    /// Bytes currently accounted as in-use against the quota.
    pub fn bytes_in_use(&self) -> u64 {
        self.bytes_in_use.load(Ordering::Acquire)
    }

    /// Atomically reserve `n` bytes against the quota.
    ///
    /// Returns `Err(BaliError::MemoryExhausted)` if the allocation would push
    /// usage above the configured quota; on success, [`Self::bytes_in_use`]
    /// reflects the new total.
    pub fn consume_bytes(&self, n: u64) -> Result<(), BaliError> {
        let limit = self.memory_quota_bytes;
        let mut current = self.bytes_in_use.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(n);
            if next > limit {
                return Err(BaliError::MemoryExhausted {
                    requested: n,
                    limit,
                });
            }
            match self.bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    /// Atomically release `n` bytes back to the quota. Saturating on
    /// underflow — callers must not release more than they consumed, but a
    /// bookkeeping mismatch is not fatal.
    pub fn release_bytes(&self, n: u64) {
        let mut current = self.bytes_in_use.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(n);
            match self.bytes_in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Builder for [`TenantContext`].
///
/// Fields default to a low-overhead, multi-tenant-safe configuration:
/// [`IsolationKind::StreamIsolated`], stream id `0`, and an 8 GiB memory
/// quota. Override with the chained `with_*` methods, then call
/// [`Self::build`].
#[derive(Debug)]
pub struct TenantContextBuilder {
    tenant_id: TenantId,
    isolation: IsolationKind,
    stream_id: u64,
    memory_quota_bytes: u64,
}

impl TenantContextBuilder {
    /// Default quota: 8 GiB. Sized for a single H100 SXM partition under MPS.
    pub const DEFAULT_QUOTA_BYTES: u64 = 8 * 1024 * 1024 * 1024;

    /// Create a builder with default isolation, stream id, and quota.
    pub fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            isolation: IsolationKind::default(),
            stream_id: 0,
            memory_quota_bytes: Self::DEFAULT_QUOTA_BYTES,
        }
    }

    /// Override the isolation level.
    pub fn with_isolation(mut self, isolation: IsolationKind) -> Self {
        self.isolation = isolation;
        self
    }

    /// Override the stream identifier.
    pub fn with_stream_id(mut self, stream_id: u64) -> Self {
        self.stream_id = stream_id;
        self
    }

    /// Override the memory quota in bytes.
    pub fn with_memory_quota_bytes(mut self, memory_quota_bytes: u64) -> Self {
        self.memory_quota_bytes = memory_quota_bytes;
        self
    }

    /// Finalise into a `TenantContext`.
    pub fn build(self) -> TenantContext {
        TenantContext {
            tenant_id: self.tenant_id,
            isolation: self.isolation,
            stream_id: self.stream_id,
            memory_quota_bytes: self.memory_quota_bytes,
            bytes_in_use: AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cu_context: None,
            #[cfg(not(feature = "cuda"))]
            cu_context: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let ctx = TenantContext::builder(TenantId(1)).build();
        assert_eq!(ctx.id(), TenantId(1));
        assert_eq!(ctx.isolation(), IsolationKind::StreamIsolated);
        assert_eq!(ctx.stream_id(), 0);
        assert_eq!(ctx.quota(), TenantContextBuilder::DEFAULT_QUOTA_BYTES);
        assert_eq!(ctx.bytes_in_use(), 0);
    }

    #[test]
    fn builder_overrides() {
        let ctx = TenantContext::builder(TenantId(7))
            .with_isolation(IsolationKind::ContextIsolated)
            .with_stream_id(42)
            .with_memory_quota_bytes(1024)
            .build();
        assert_eq!(ctx.isolation(), IsolationKind::ContextIsolated);
        assert_eq!(ctx.stream_id(), 42);
        assert_eq!(ctx.quota(), 1024);
    }

    #[test]
    fn quota_consume_release_round_trip() {
        let ctx = TenantContext::builder(TenantId(2))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.consume_bytes(256).unwrap();
        assert_eq!(ctx.bytes_in_use(), 256);
        ctx.consume_bytes(512).unwrap();
        assert_eq!(ctx.bytes_in_use(), 768);
        ctx.release_bytes(256);
        assert_eq!(ctx.bytes_in_use(), 512);
    }

    #[test]
    fn quota_enforcement_rejects_over_limit() {
        let ctx = TenantContext::builder(TenantId(3))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.consume_bytes(1000).unwrap();
        let err = ctx.consume_bytes(100).unwrap_err();
        match err {
            BaliError::MemoryExhausted { requested, limit } => {
                assert_eq!(requested, 100);
                assert_eq!(limit, 1024);
            }
            other => panic!("expected MemoryExhausted, got {other:?}"),
        }
        // Failed allocation must not move the counter.
        assert_eq!(ctx.bytes_in_use(), 1000);
    }

    #[test]
    fn release_saturates_on_underflow() {
        let ctx = TenantContext::builder(TenantId(4))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.release_bytes(999); // never consumed; must not panic or wrap.
        assert_eq!(ctx.bytes_in_use(), 0);
    }

    #[test]
    fn isolation_kind_names_are_stable() {
        assert_eq!(IsolationKind::Shared.name(), "shared");
        assert_eq!(IsolationKind::StreamIsolated.name(), "stream_isolated");
        assert_eq!(IsolationKind::ContextIsolated.name(), "context_isolated");
    }

    #[test]
    fn isolation_kind_matches_each_variant() {
        for kind in [
            IsolationKind::Shared,
            IsolationKind::StreamIsolated,
            IsolationKind::ContextIsolated,
        ] {
            let ctx = TenantContext::builder(TenantId(99))
                .with_isolation(kind)
                .build();
            assert_eq!(ctx.isolation(), kind);
            // Display is the same as the name.
            assert_eq!(ctx.isolation().to_string(), kind.name());
        }
    }

    #[test]
    fn isolation_kind_default_is_stream_isolated() {
        assert_eq!(IsolationKind::default(), IsolationKind::StreamIsolated);
    }
}
