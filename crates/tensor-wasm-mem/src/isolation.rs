// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Memory isolation policy for TensorWasm Wasm instances.
//!
//! Three levels are exposed; the active level for an instance is fixed at
//! spawn time and stored on the executor's instance metadata (set up in S7).
//! The level shapes both how the memory creator allocates the buffer and
//! how the WASI-CUDA host functions in `tensor-wasm-wasi-gpu` (S8) wire kernels
//! up to streams/contexts.

use std::fmt;

/// How aggressively to isolate one tenant's memory and GPU operations from
/// another's.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsolationLevel {
    /// All instances share the default CUDA context and stream.
    ///
    /// Cheap to spawn (no context creation) and good for fully trusted
    /// workloads (e.g. a single-tenant deployment). Not safe for multi-tenant
    /// untrusted code.
    Shared,
    /// Each instance gets its own CUDA stream but contexts are shared.
    ///
    /// Prevents cross-instance kernel ordering accidents and gives per-stream
    /// scheduling. Default for multi-tenant deployments.
    #[default]
    StreamIsolated,
    /// Each tenant gets its own CUDA context (via MPS when available, or
    /// `cuCtxCreate` otherwise). Enabled in S16.
    ContextIsolated,
}

impl IsolationLevel {
    /// Stable, human-readable name (used in span attributes and metrics).
    pub fn name(&self) -> &'static str {
        match self {
            IsolationLevel::Shared => "shared",
            IsolationLevel::StreamIsolated => "stream_isolated",
            IsolationLevel::ContextIsolated => "context_isolated",
        }
    }
}

impl fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_stream_isolated() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::StreamIsolated);
    }

    #[test]
    fn names_are_stable() {
        assert_eq!(IsolationLevel::Shared.name(), "shared");
        assert_eq!(IsolationLevel::StreamIsolated.name(), "stream_isolated");
        assert_eq!(IsolationLevel::ContextIsolated.name(), "context_isolated");
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(IsolationLevel::Shared.to_string(), "shared");
        assert_eq!(
            IsolationLevel::ContextIsolated.to_string(),
            "context_isolated"
        );
    }

    #[test]
    fn levels_are_ord_independent_but_distinct() {
        assert_ne!(IsolationLevel::Shared, IsolationLevel::StreamIsolated);
        assert_ne!(
            IsolationLevel::StreamIsolated,
            IsolationLevel::ContextIsolated
        );
    }
}
