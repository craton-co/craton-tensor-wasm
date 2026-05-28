// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Differential correctness oracle (roadmap feature #6).
//!
//! For every `auto_offload` candidate that lowers to PTX, the harness
//! runs:
//!   1. The original Wasm body on the Wasmtime CPU interpreter (the
//!      ground truth).
//!   2. The JIT-emitted PTX on the cust/cudarc backend.
//! and asserts byte-equal output. Discrepancies are surfaced as
//! `OracleDivergence` records the CI gate can publish on every PR.
//!
//! ## v0.3.6 status: scaffold
//!
//! The harness types are stable; the actual two-path runner requires
//! both a Wasmtime CPU interpreter handle AND a CUDA device, which the
//! current CI runner doesn't have. v0.4 wires this against the
//! self-hosted CUDA runner. Until then, the harness validates the
//! input shape and returns `OracleVerdict::Skipped("no-cuda")` for
//! every call.
//!
//! ## How to use (v0.4 target shape)
//!
//! ```ignore
//! use tensor_wasm_jit::differential::DifferentialOracle;
//!
//! let oracle = DifferentialOracle::new();
//! let blueprint = /* TensorWasmKernelBlueprint */;
//! let inputs = /* &[u8] guest memory snapshot */;
//! match oracle.compare(&blueprint, inputs) {
//!     OracleVerdict::Match { .. } => { /* pass */ }
//!     OracleVerdict::Divergence(d) => panic!("audit-bait: {d:?}"),
//!     OracleVerdict::Skipped(reason) => eprintln!("skipped: {reason}"),
//! }
//! ```

use crate::ir::TensorWasmKernelBlueprint;

/// Configuration for the oracle. v0.3.6: defaults are sufficient.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OracleConfig {
    /// Skip the comparison and return Skipped("disabled") for every
    /// call. Used to disable the oracle in environments where one of
    /// the two paths is not available.
    pub disabled: bool,
}

/// The oracle. Drive it from proptest test bodies and CI gates.
pub struct DifferentialOracle {
    cfg: OracleConfig,
}

impl DifferentialOracle {
    /// Construct an oracle with default configuration.
    pub fn new() -> Self {
        Self { cfg: OracleConfig::default() }
    }

    /// Construct an oracle with a caller-provided configuration.
    pub fn with_config(cfg: OracleConfig) -> Self {
        Self { cfg }
    }

    /// Run the two paths and compare. v0.3.6: always returns Skipped
    /// because the v0.4 wiring against the self-hosted CUDA runner is
    /// not yet landed.
    pub fn compare(&self, _bp: &TensorWasmKernelBlueprint, _inputs: &[u8]) -> OracleVerdict {
        if self.cfg.disabled {
            return OracleVerdict::Skipped("oracle disabled by config");
        }
        // v0.4: run the Wasmtime CPU interpreter and the JIT PTX path,
        // collect outputs, return Match or Divergence.
        OracleVerdict::Skipped("no-cuda; v0.4 wires this against the S22 runner")
    }
}

impl Default for DifferentialOracle {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single oracle comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleVerdict {
    /// Both paths produced byte-equal output.
    Match {
        /// Number of output bytes that were compared.
        output_len: usize,
    },
    /// Outputs diverged. The CI gate must fail.
    Divergence(OracleDivergence),
    /// Comparison was skipped (typically: one of the two paths is not
    /// available on the current host). Not a failure; the test runner
    /// should record it as a skip.
    Skipped(&'static str),
}

/// Detailed divergence record. Logged with `tracing::error!` in CI
/// gates; the offending blueprint fingerprint is stable across runs so
/// the same divergence is triaged once, not per-CI-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleDivergence {
    /// Stable fingerprint of the blueprint that diverged. Matches
    /// `TensorWasmKernelBlueprint::fingerprint`.
    pub blueprint_fingerprint: u64,
    /// Number of bytes the Wasmtime CPU path produced.
    pub cpu_output_len: usize,
    /// Number of bytes the JIT PTX path produced.
    pub gpu_output_len: usize,
    /// Offset of the first differing byte if both outputs are non-empty
    /// and share at least one byte; `None` if the lengths differed
    /// before any byte could be compared.
    pub first_diff_offset: Option<usize>,
}
