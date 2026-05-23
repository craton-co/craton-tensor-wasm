//! Wasmtime + Tokio async execution engine for Bali instances.
//!
//! `bali-exec` wraps [`wasmtime`] with a Bali-specific [`engine::BaliEngine`]
//! that wires in async execution, epoch-based interruption, and a custom
//! linear-memory creator backed by [`bali_mem`]. The [`instance`] module
//! manages per-tenant instance lifecycles, and [`executor`] drives async
//! invocation of guest exports against a shared engine and store pool.
#![warn(missing_docs)]

pub mod auto_offload;
pub mod engine;
pub mod executor;
pub mod instance;
