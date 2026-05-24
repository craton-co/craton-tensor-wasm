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
//! Register a tenant and account some bytes against its quota:
//!
//! ```
//! # #[cfg(not(feature = "cuda"))]
//! # fn main() {
//! use tensor_wasm_core::types::TenantId;
//! use tensor_wasm_tenant::{TenantContext, TenantRegistry};
//!
//! let reg = TenantRegistry::new();
//! let ctx = reg
//!     .register(TenantContext::builder(TenantId(1)).build())
//!     .unwrap();
//! ctx.consume_bytes(4096).unwrap();
//! assert_eq!(ctx.bytes_in_use(), 4096);
//! # }
//! # #[cfg(feature = "cuda")]
//! # fn main() {}
//! ```
#![deny(missing_docs)]

pub mod context;
pub mod registry;

pub use context::{IsolationKind, TenantContext, TenantContextBuilder};
pub use registry::{
    MpsDecision, RegistryError, TenantRegistry, MPS_CONTROL_PATH, MPS_PIPE_DIRECTORY_ENV,
};
