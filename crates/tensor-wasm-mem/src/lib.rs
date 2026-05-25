// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! CUDA Unified Memory allocator and Wasmtime `MemoryCreator` integration.
//!
//! Two byte-buffer types:
//! - [`unified::UnifiedBuffer`] — CUDA Unified Memory when the
//!   `unified-memory` feature is enabled; a heap buffer otherwise.
//! - [`pinned_host::GuardedHostBuffer`] — guarded host memory for the
//!   explicit-fallback path used by `cargo build --no-default-features`.
//!   On all hosts the buffer is bracketed by `PROT_NONE` / `PAGE_NOACCESS`
//!   guard pages catching OOB reads/writes at the OS level. The managed
//!   memory path (`UnifiedBuffer` on CUDA hosts) still cannot use guard
//!   pages — that limitation is unchanged. The previous name
//!   `PinnedHostBuffer` is preserved as a deprecated alias.
//!
//! Pools (`pool::UnifiedMemoryPool`) amortise the cost of CUDA allocations
//! by carving sub-slices from a single slab. Hints (`advise`) forward to
//! `cudaMemAdvise` on CUDA hosts and are no-ops otherwise.
#![deny(missing_docs)]

pub mod advise;
#[cfg(feature = "cudarc-backend")]
pub mod cudarc_backend;
pub mod isolation;
pub mod pinned_host;
pub mod pool;
pub mod unified;
pub mod wasm_memory;
