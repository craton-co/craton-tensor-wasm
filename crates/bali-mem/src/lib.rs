//! CUDA Unified Memory allocator and Wasmtime `MemoryCreator` integration.
//!
//! Two byte-buffer types:
//! - [`unified::UnifiedBuffer`] — CUDA Unified Memory when the
//!   `unified-memory` feature is enabled; a heap buffer otherwise.
//! - [`pinned_host::PinnedHostBuffer`] — host-pinned memory for the
//!   explicit-fallback path used by `cargo build --no-default-features`.
//!   On all hosts the buffer is bracketed by `PROT_NONE` / `PAGE_NOACCESS`
//!   guard pages catching OOB reads/writes at the OS level. The managed
//!   memory path (`UnifiedBuffer` on CUDA hosts) still cannot use guard
//!   pages — that limitation is unchanged.
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
