// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Multi-tenant CUDA context management for Craton TensorWasm.
//!
//! This crate owns the per-tenant runtime handle ([`TenantContext`]) and the
//! concurrent registry that maps a [`tensor_wasm_core::types::TenantId`] to it
//! ([`TenantRegistry`]). The registry also decides between MPS-backed shared
//! contexts and per-tenant `cuCtxCreate` fallback via a filesystem probe — see
//! `docs/MPS-SETUP.md` for daemon configuration and `SECURITY.md` for the
//! threat model that motivates the [`IsolationKind`] taxonomy.
//!
//! # Quick start
//!
//! Two distinct capability primitives:
//!
//! * [`TenantCapability`] — minted per-tenant by
//!   [`TenantRegistry::register_with_capability`], required to drive that
//!   tenant's quota counters (`consume_bytes_with_capability` /
//!   `release_bytes_with_capability`). Holding an `Arc<TenantContext>` for
//!   tenant A grants no power to mutate tenant B's accounting.
//! * [`RegistryAdminCapability`] — minted once by [`TenantRegistry::new`],
//!   required to invoke registry-wide admin operations
//!   ([`TenantRegistry::get`], [`TenantRegistry::unregister`],
//!   [`TenantRegistry::tenants`], [`TenantRegistry::len`]). Without it, any
//!   holder of an `Arc<TenantRegistry>` could enumerate or evict tenants.
//!
//! ```
//! # #[cfg(not(feature = "cuda"))]
//! # fn main() {
//! use tensor_wasm_core::types::TenantId;
//! use tensor_wasm_tenant::{TenantContext, TenantRegistry};
//!
//! let (reg, admin_cap) = TenantRegistry::new();
//! let (ctx, tenant_cap) = reg
//!     .register_with_capability(TenantContext::builder(TenantId(1)).build())
//!     .unwrap();
//! ctx.consume_bytes_with_capability(&tenant_cap, 4096).unwrap();
//! assert_eq!(ctx.bytes_in_use(), 4096);
//! // Admin cap is required to look the tenant up again by id.
//! assert!(reg.get(TenantId(1), &admin_cap).is_some());
//! # }
//! # #[cfg(feature = "cuda")]
//! # fn main() {}
//! ```
#![deny(missing_docs)]

pub mod context;
pub mod registry;

pub use context::{
    isolation_downgrade_count, IsolationKind, RateLimited, ReleaseOutcome, TenantCapability,
    TenantContext, TenantContextBuilder,
};
pub use registry::{
    foreign_cap_rejections_total, MpsDecision, RegistryAdminCapability, RegistryError,
    TenantRegistry, MPS_CONTROL_PATH, MPS_PIPE_DIRECTORY_ENV,
};
