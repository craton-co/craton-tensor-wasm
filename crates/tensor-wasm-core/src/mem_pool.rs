// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Backend-agnostic driver memory-pool abstraction shared across crates.
//!
//! The per-tenant driver-enforced GPU cap (roadmap feature #8, v0.4
//! deliverable T39) needs `tensor-wasm-tenant` to push a tenant's cap
//! down to a concrete CUDA memory pool living in `tensor-wasm-mem`. But
//! `tensor-wasm-tenant` cannot depend on `tensor-wasm-mem`: `mem`
//! already depends on `tenant` (its `TensorWasmMemoryCreator::with_tenant_context`
//! builder takes an `Arc<TenantContext>` to drive `consume_gpu_bytes`),
//! so a `tenant -> mem` edge would close a dependency cycle.
//!
//! This module breaks that cycle by hoisting the *interface* into
//! `tensor-wasm-core` — a crate both `mem` and `tenant` already depend
//! on. `tensor-wasm-tenant` holds an
//! `Option<Arc<dyn DriverMemPool>>` and pushes the cap through
//! [`DriverMemPool::set_release_threshold`]; `tensor-wasm-mem`'s
//! `cuda_mem_pool::TenantMemPool` implements the trait. Neither crate
//! references the other's concrete types, so no cycle forms.
//!
//! The surface here is deliberately minimal: only what the tenant
//! driver-cap path needs (set the release threshold, read it back for
//! honest reporting). The concrete `cuMemPool*` FFI, pool creation, and
//! allocation paths stay in `tensor-wasm-mem` behind its cudarc gating.

use std::sync::Arc;

/// Errors raised by [`DriverMemPool`] operations.
///
/// Owned here in `tensor-wasm-core` (moved out of
/// `tensor-wasm-mem::cuda_mem_pool`) so both the trait and its
/// implementors share one error type without either crate depending on
/// the other. The `String` payloads carry the `Debug`-formatted
/// underlying CUDA result so the operator alert path keeps the same
/// context it had when these variants lived in `mem`.
#[derive(Debug, thiserror::Error)]
pub enum MemPoolError {
    /// `cuMemPoolCreate` returned a non-`CUDA_SUCCESS` code. The wrapped
    /// string is the `Debug`-formatted CUDA result.
    #[error("cuMemPoolCreate failed: {0}")]
    Create(String),
    /// `cuMemPoolSetAttribute` returned a non-`CUDA_SUCCESS` code. The
    /// half-built pool is destroyed before this error is returned, so
    /// callers do not need to do anything to clean up.
    #[error("cuMemPoolSetAttribute failed: {0}")]
    SetAttribute(String),
    /// CUDA was not initialised by the time the pool operation ran.
    /// Reserved for callers that need to distinguish "driver not primed"
    /// from a hard `cuMemPool*` failure without a breaking-change minor
    /// bump.
    #[error("cuda not initialized")]
    NotInitialized,
    /// The per-ordinal device cache could not retain a primary context
    /// for the requested device ordinal. Wraps the underlying CUDA
    /// description. A non-CUDA host or a missing GPU surfaces here, NOT
    /// through [`Self::Create`].
    #[error("device retain failed: {0}")]
    Device(String),
}

/// A driver-level memory pool whose release threshold can be pinned to a
/// per-tenant cap.
///
/// Implemented by `tensor-wasm-mem`'s `cuda_mem_pool::TenantMemPool`
/// (backed by `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`)
/// and by test mocks. `tensor-wasm-tenant` holds one as an
/// `Arc<dyn DriverMemPool>` and never names the concrete type — that is
/// what keeps the `mem` <-> `tenant` dependency graph acyclic.
///
/// `Send + Sync` so the pool can be shared across the threads that drive
/// a tenant's allocations (the tenant context is itself shared behind an
/// `Arc`). `Debug` so `TenantContext` (which holds an
/// `Option<Arc<dyn DriverMemPool>>`) keeps its derived `Debug` impl;
/// the concrete `tensor-wasm-mem::TenantMemPool` already derives it and
/// test mocks can too.
pub trait DriverMemPool: std::fmt::Debug + Send + Sync {
    /// Pin the pool's release threshold to `bytes`.
    ///
    /// Allocations past this ceiling fail at the driver level with
    /// `CUDA_ERROR_OUT_OF_MEMORY` — the bypass-resistant gate the
    /// in-process `consume_gpu_bytes` counter cannot enforce against a
    /// tenant that obtained a raw CUDA driver handle. Returns
    /// [`MemPoolError::SetAttribute`] if the underlying driver call
    /// fails.
    fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError>;

    /// The release-threshold cap (in bytes) this pool was last configured
    /// with, or `None` if no threshold has been pinned yet.
    ///
    /// The CUDA driver may round the value internally; implementors
    /// return the *requested* value for honest reporting back to the
    /// tenant.
    fn release_threshold(&self) -> Option<u64>;
}

impl DriverMemPool for Arc<dyn DriverMemPool> {
    fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError> {
        (**self).set_release_threshold(bytes)
    }

    fn release_threshold(&self) -> Option<u64> {
        (**self).release_threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MemPoolError` Display impls produce non-empty, distinguishable
    /// messages. The operator alert path keys on the message string, so
    /// an accidental `#[error("")]` would silently swallow context.
    #[test]
    fn mem_pool_error_display_non_empty() {
        let e = MemPoolError::Create("CUDA_ERROR_OUT_OF_MEMORY".into());
        assert!(format!("{e}").contains("cuMemPoolCreate failed"));
        let e = MemPoolError::SetAttribute("CUDA_ERROR_INVALID_VALUE".into());
        assert!(format!("{e}").contains("cuMemPoolSetAttribute failed"));
        let e = MemPoolError::NotInitialized;
        assert!(format!("{e}").contains("not initialized"));
        let e = MemPoolError::Device("device_for(7): ...".into());
        assert!(format!("{e}").contains("device retain failed"));
    }

    /// The `Arc<dyn DriverMemPool>` blanket impl forwards to the inner
    /// pool. Guards the convenience that lets `tensor-wasm-tenant` call
    /// the trait methods on its stored `Arc` without an explicit
    /// `(*arc)` deref.
    #[test]
    fn arc_forwards_to_inner() {
        use std::sync::atomic::{AtomicU64, Ordering};

        #[derive(Debug)]
        struct Spy(AtomicU64);
        impl DriverMemPool for Spy {
            fn set_release_threshold(&self, bytes: u64) -> Result<(), MemPoolError> {
                self.0.store(bytes, Ordering::SeqCst);
                Ok(())
            }
            fn release_threshold(&self) -> Option<u64> {
                Some(self.0.load(Ordering::SeqCst))
            }
        }

        let pool: Arc<dyn DriverMemPool> = Arc::new(Spy(AtomicU64::new(0)));
        pool.set_release_threshold(4096).unwrap();
        assert_eq!(pool.release_threshold(), Some(4096));
    }
}
