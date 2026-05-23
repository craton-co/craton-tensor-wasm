//! CUDA Unified Memory allocator and Wasmtime `MemoryCreator` integration.
//!
//! Two byte-buffer types:
//! - [`unified::UnifiedBuffer`] — CUDA Unified Memory when the
//!   `unified-memory` feature is enabled; a heap buffer otherwise.
//! - [`pinned_host::PinnedHostBuffer`] — host-pinned (or plain heap on
//!   non-CUDA hosts) memory for the explicit-fallback path used by
//!   `cargo build --no-default-features`.
//!
//! Pools (`pool::UnifiedMemoryPool`) amortise the cost of CUDA allocations
//! by carving sub-slices from a single slab. Hints (`advise`) forward to
//! `cudaMemAdvise` on CUDA hosts and are no-ops otherwise.
#![warn(missing_docs)]

pub mod advise;
pub mod isolation;
pub mod pinned_host;
pub mod pool;
pub mod unified;
pub mod wasm_memory;
