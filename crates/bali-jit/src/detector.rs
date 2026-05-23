//! GPU-offload-candidate detector.
//!
//! Walks a [`BlockIR`] (a simplified, in-house representation modelled after
//! Cranelift CLIF basic blocks) and decides whether the block should be
//! lowered to PTX. The decision rule, per the plan, is:
//!
//! > Flag a block if **>80 %** of its instructions are v128 SIMD ops AND the
//! > loop trip count is statically known.
//!
//! The simplified IR keeps this crate independent of the Cranelift runtime
//! API surface — the real-CLIF integration is documented in
//! `docs/WASMTIME-FORK.md` and lands in a follow-up session once the
//! Wasmtime team upstreams the necessary hooks.

use std::fmt;

/// Instruction kinds the detector recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// 128-bit SIMD add (any element width).
    V128Add,
    /// 128-bit SIMD multiply.
    V128Mul,
    /// 128-bit SIMD fused-multiply-add.
    V128Fma,
    /// Scalar arithmetic add.
    ScalarAdd,
    /// Scalar arithmetic multiply.
    ScalarMul,
    /// Load from linear memory.
    Load,
    /// Store to linear memory.
    Store,
    /// Conditional branch.
    Branch,
    /// Function call (including host imports).
    Call,
}

impl Op {
    /// True if this op operates on `v128` SIMD values.
    pub fn is_v128(self) -> bool {
        matches!(self, Op::V128Add | Op::V128Mul | Op::V128Fma)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Op::V128Add => "v128.add",
            Op::V128Mul => "v128.mul",
            Op::V128Fma => "v128.fma",
            Op::ScalarAdd => "scalar.add",
            Op::ScalarMul => "scalar.mul",
            Op::Load => "load",
            Op::Store => "store",
            Op::Branch => "branch",
            Op::Call => "call",
        };
        f.write_str(s)
    }
}

/// A simplified Cranelift-style basic block.
#[derive(Debug, Clone)]
pub struct BlockIR {
    /// Linear sequence of ops in this block.
    pub ops: Vec<Op>,
    /// Static loop trip count (if the loop in which this block sits has
    /// a statically-known iteration count). `None` if unknown/dynamic.
    pub loop_trip_count: Option<u64>,
    /// Human-readable name (used in tests and trace output).
    pub name: String,
}

impl BlockIR {
    /// Construct a new block.
    pub fn new(name: impl Into<String>, ops: Vec<Op>, loop_trip_count: Option<u64>) -> Self {
        Self {
            name: name.into(),
            ops,
            loop_trip_count,
        }
    }

    /// Fraction of the block that is v128 ops (between 0.0 and 1.0).
    pub fn v128_ratio(&self) -> f32 {
        if self.ops.is_empty() {
            return 0.0;
        }
        let v128 = self.ops.iter().filter(|o| o.is_v128()).count();
        v128 as f32 / self.ops.len() as f32
    }
}

/// Annotation attached to a [`BlockIR`] after the detector runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorVerdict {
    /// Block should be GPU-offloaded.
    Offload,
    /// Block should stay on the CPU path.
    KeepOnCpu,
}

/// Configurable parameters for the detector.
#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// Minimum fraction of v128 ops to consider offloading (default 0.8).
    pub v128_ratio_threshold: f32,
    /// Minimum loop trip count to bother with offload setup overhead (default 64).
    pub min_trip_count: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            v128_ratio_threshold: 0.8,
            min_trip_count: 64,
        }
    }
}

/// Inspect a [`BlockIR`] and return a [`DetectorVerdict`].
pub fn classify(block: &BlockIR, cfg: &DetectorConfig) -> DetectorVerdict {
    let ratio = block.v128_ratio();
    let trip_ok = block
        .loop_trip_count
        .map(|n| n >= cfg.min_trip_count)
        .unwrap_or(false);
    if ratio >= cfg.v128_ratio_threshold && trip_ok {
        DetectorVerdict::Offload
    } else {
        DetectorVerdict::KeepOnCpu
    }
}

/// Convenience: classify with default config.
pub fn classify_default(block: &BlockIR) -> DetectorVerdict {
    classify(block, &DetectorConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, ops: Vec<Op>, loop_n: Option<u64>) -> BlockIR {
        BlockIR::new(name, ops, loop_n)
    }

    #[test]
    fn mixed_v128_ratio_below_threshold_is_kept_on_cpu() {
        let b = block(
            "vector_add_loop",
            vec![
                Op::Load,
                Op::Load,
                Op::V128Add,
                Op::V128Add,
                Op::V128Add,
                Op::V128Add,
                Op::Store,
            ],
            Some(128),
        );
        // 4/7 = 57% — under threshold. Need >80%.
        assert_eq!(classify_default(&b), DetectorVerdict::KeepOnCpu);
    }

    #[test]
    fn high_v128_ratio_offloaded() {
        let b = block(
            "matmul_inner",
            vec![
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Mul,
                Op::Store,
            ],
            Some(512),
        );
        // 9/10 = 90% — over threshold AND trip count 512 > 64.
        assert_eq!(classify_default(&b), DetectorVerdict::Offload);
    }

    #[test]
    fn pure_v128_matmul_tile_is_offloaded() {
        let b = block(
            "matmul_tile",
            vec![
                Op::Load,
                Op::Load,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::V128Fma,
                Op::Store,
            ],
            Some(256),
        );
        // 12/15 = 80% — meets threshold AND trip count 256 > 64.
        assert_eq!(classify_default(&b), DetectorVerdict::Offload);
    }

    #[test]
    fn pure_v128_vector_mul_loop_is_offloaded() {
        let b = block(
            "vector_mul",
            vec![
                Op::Load,
                Op::V128Mul,
                Op::V128Mul,
                Op::V128Mul,
                Op::V128Mul,
                Op::V128Add,
                Op::V128Add,
                Op::V128Add,
                Op::V128Add,
                Op::Store,
            ],
            Some(1024),
        );
        // 8/10 = 80% — meets threshold AND trip count 1024 > 64.
        assert_eq!(classify_default(&b), DetectorVerdict::Offload);
    }

    #[test]
    fn dynamic_loop_not_offloaded_even_if_v128_heavy() {
        let b = block(
            "dynamic_loop",
            vec![Op::V128Add; 16],
            None, // unknown trip count
        );
        assert_eq!(classify_default(&b), DetectorVerdict::KeepOnCpu);
    }

    #[test]
    fn tiny_loop_not_offloaded() {
        let b = block(
            "tiny",
            vec![Op::V128Add; 16],
            Some(8), // trip < threshold (64)
        );
        assert_eq!(classify_default(&b), DetectorVerdict::KeepOnCpu);
    }

    #[test]
    fn scalar_heavy_not_offloaded() {
        let b = block(
            "scalar",
            vec![
                Op::ScalarAdd,
                Op::ScalarAdd,
                Op::ScalarMul,
                Op::Branch,
                Op::Call,
                Op::Load,
            ],
            Some(1024),
        );
        assert_eq!(classify_default(&b), DetectorVerdict::KeepOnCpu);
    }

    #[test]
    fn op_is_v128_taxonomy() {
        assert!(Op::V128Add.is_v128());
        assert!(Op::V128Mul.is_v128());
        assert!(Op::V128Fma.is_v128());
        assert!(!Op::ScalarAdd.is_v128());
        assert!(!Op::Load.is_v128());
    }

    #[test]
    fn config_threshold_tunable() {
        let b = block(
            "borderline",
            vec![Op::V128Add, Op::V128Add, Op::Load],
            Some(128),
        );
        // 2/3 = 67% — below default 80% threshold.
        assert_eq!(classify_default(&b), DetectorVerdict::KeepOnCpu);
        // Lower the threshold to 60% — now it's offloaded.
        let cfg = DetectorConfig {
            v128_ratio_threshold: 0.6,
            ..DetectorConfig::default()
        };
        assert_eq!(classify(&b, &cfg), DetectorVerdict::Offload);
    }
}
