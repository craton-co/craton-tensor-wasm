//! Multi-tenant CUDA context management for Project Bali.
//!
//! This crate owns the per-tenant runtime handle ([`TenantContext`]) and the
//! concurrent registry that maps a [`bali_core::types::TenantId`] to it
//! ([`TenantRegistry`]). The registry also decides between MPS-backed shared
//! contexts and per-tenant `cuCtxCreate` fallback via a filesystem probe — see
//! `docs/MPS-SETUP.md` for daemon configuration and `SECURITY.md` for the
//! threat model that motivates the [`IsolationKind`] taxonomy.
#![warn(missing_docs)]

pub mod context;
pub mod registry;

pub use context::{IsolationKind, TenantContext, TenantContextBuilder};
pub use registry::{MpsDecision, RegistryError, TenantRegistry, MPS_CONTROL_PATH};
