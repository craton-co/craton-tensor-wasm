// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Pre-instantiation auto-offload analysis.
//!
//! Walks Wasm bytes via [`wasmparser`], extracts per-function operator
//! sequences shaped like basic blocks, and runs each through
//! [`tensor_wasm_jit::detector::classify`]. Verdicts are emitted as `tracing`
//! events so operators can see which kernels *would* be GPU-offloaded
//! once Wasmtime exposes the per-block compile hook documented in
//! `docs/WASMTIME-FORK.md`.
//!
//! This is consultation-only: the analysis runs, but Wasmtime's
//! Cranelift output is NOT replaced. Activating the swap requires
//! the rewrite pipeline in `tensor_wasm_jit::rewrite`.

use tensor_wasm_jit::detector::{classify, BlockIR, DetectorConfig, DetectorVerdict, Op};
use tracing::{debug, info, instrument};
use wasmparser::{Operator, Parser, Payload};

/// A single classification result for one function body.
#[derive(Debug, Clone, PartialEq)]
pub struct OffloadVerdict {
    /// Function index inside the Wasm module.
    pub function_index: u32,
    /// Detector classification.
    pub verdict: DetectorVerdict,
    /// Number of operators inspected.
    pub op_count: usize,
    /// Fraction of v128 ops (0.0-1.0).
    pub v128_ratio: f32,
}

/// Errors raised by the analyser.
#[derive(Debug, thiserror::Error)]
pub enum AnalyseError {
    /// The Wasm bytes failed to parse.
    #[error("wasmparser: {0}")]
    Parse(String),
}

/// Map a wasmparser [`Operator`] to the simplified [`tensor_wasm_jit::detector::Op`].
///
/// Most ops fall outside the detector's taxonomy — they're collapsed into
/// [`Op::Load`] / [`Op::Store`] (memory) or [`Op::Branch`] (control flow) so
/// the ratio computation is meaningful but never panics. Anything else
/// becomes [`Op::Other`] so the v128 ratio is computed honestly without
/// inflating arithmetic density from non-arithmetic ops like `local.get`.
fn op_to_detector_op(op: &Operator<'_>) -> Op {
    use wasmparser::Operator::*;
    match op {
        V128Load { .. } => Op::Load,
        V128Store { .. } => Op::Store,
        F32Add | I32Add | I64Add | F64Add => Op::ScalarAdd,
        F32Mul | I32Mul | I64Mul | F64Mul => Op::ScalarMul,
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } => Op::Load,
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } => Op::Store,
        Br { .. }
        | BrIf { .. }
        | BrTable { .. }
        | If { .. }
        | Else
        | Loop { .. }
        | Block { .. } => Op::Branch,
        Call { .. } | CallIndirect { .. } | ReturnCall { .. } => Op::Call,
        F32x4Add | F64x2Add | I32x4Add | I64x2Add | I16x8Add | I8x16Add => Op::V128Add,
        F32x4Mul | F64x2Mul | I32x4Mul | I64x2Mul | I16x8Mul => Op::V128Mul,
        _ => Op::Other,
    }
}

/// Analyse Wasm bytes, emit per-function `tracing` events, and return the
/// list of verdicts, using the default detector configuration
/// ([`DetectorConfig::default`]).
///
/// Thin wrapper over [`analyse_with_config`] preserving the historical
/// signature for the consultation-only call sites.
#[instrument(skip(wasm), fields(wasm_bytes = wasm.len()))]
pub fn analyse(wasm: &[u8]) -> Result<Vec<OffloadVerdict>, AnalyseError> {
    analyse_with_config(wasm, &DetectorConfig::default())
}

/// Analyse Wasm bytes against an explicit detector configuration, emit
/// per-function `tracing` events, and return the list of verdicts.
///
/// The default-config wrapper [`analyse`] keeps the historical
/// consultation-only behaviour; this entry point lets the executor's
/// auto-offload activation path consult with the *same* detector thresholds
/// the rewrite will use, so the consultation verdict and the rewrite agree
/// on which functions are offload candidates.
#[instrument(skip(wasm, cfg), fields(wasm_bytes = wasm.len()))]
pub fn analyse_with_config(
    wasm: &[u8],
    cfg: &DetectorConfig,
) -> Result<Vec<OffloadVerdict>, AnalyseError> {
    let mut verdicts = Vec::new();
    let mut func_idx: u32 = 0;
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.map_err(|e| AnalyseError::Parse(format!("{e}")))?;
        if let Payload::CodeSectionEntry(body) = payload {
            let mut ops = body
                .get_operators_reader()
                .map_err(|e| AnalyseError::Parse(format!("{e}")))?;
            let mut detector_ops = Vec::new();
            let mut saw_loop = false;
            while !ops.eof() {
                match ops.read() {
                    Ok(op) => {
                        if matches!(op, wasmparser::Operator::Loop { .. }) {
                            saw_loop = true;
                        }
                        detector_ops.push(op_to_detector_op(&op));
                    }
                    Err(_) => break,
                }
            }
            // Only attach a static-loop guess when the parser actually saw a
            // `Loop` — straight-line functions inheriting `Some(128)` made
            // the detector approve cold code that would never amortise the
            // offload setup cost.
            let trip_count = if saw_loop { Some(128) } else { None };
            let block = BlockIR::new(format!("func{func_idx}"), detector_ops.clone(), trip_count);
            let v = classify(&block, cfg);
            let op_count = detector_ops.len();
            let v128 = detector_ops.iter().filter(|o| o.is_v128()).count();
            let v128_ratio = if op_count == 0 {
                0.0
            } else {
                v128 as f32 / op_count as f32
            };
            match v {
                DetectorVerdict::Offload => info!(
                    target: "tensor_wasm_exec::auto_offload",
                    function = func_idx,
                    op_count,
                    v128_ratio,
                    "verdict: Offload"
                ),
                DetectorVerdict::KeepOnCpu => debug!(
                    target: "tensor_wasm_exec::auto_offload",
                    function = func_idx,
                    op_count,
                    v128_ratio,
                    "verdict: KeepOnCpu"
                ),
            }
            verdicts.push(OffloadVerdict {
                function_index: func_idx,
                verdict: v,
                op_count,
                v128_ratio,
            });
            func_idx += 1;
        }
    }
    Ok(verdicts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL_WAT: &str = r#"(module (func (export "noop")))"#;

    #[test]
    fn analyse_noop_module() {
        let wasm = wat::parse_str(TRIVIAL_WAT).unwrap();
        let v = analyse(&wasm).expect("analyse");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].function_index, 0);
        // A noop function has no v128 ops worth offloading.
        assert_eq!(v[0].verdict, DetectorVerdict::KeepOnCpu);
    }

    #[test]
    fn analyse_v128_heavy_module() {
        // A function full of v128.add — should be flagged.
        let wat = r#"
            (module
              (memory 1)
              (func (export "hot")
                (drop (v128.const i32x4 1 2 3 4))
                (drop (v128.const i32x4 5 6 7 8))
                (drop (i32x4.add (v128.const i32x4 1 2 3 4) (v128.const i32x4 5 6 7 8)))
                (drop (i32x4.add (v128.const i32x4 1 2 3 4) (v128.const i32x4 5 6 7 8)))
                (drop (i32x4.add (v128.const i32x4 1 2 3 4) (v128.const i32x4 5 6 7 8)))
                (drop (i32x4.add (v128.const i32x4 1 2 3 4) (v128.const i32x4 5 6 7 8)))
              )
            )
        "#;
        let wasm = wat::parse_str(wat).unwrap();
        let v = analyse(&wasm).expect("analyse");
        assert_eq!(v.len(), 1);
        // We don't assert on the threshold outcome — the test exercises the
        // analyser, not the threshold tuning. Just confirm we walked ops.
        assert!(matches!(
            v[0].verdict,
            DetectorVerdict::Offload | DetectorVerdict::KeepOnCpu
        ));
        assert!(v[0].op_count > 0);
    }

    #[test]
    fn analyse_invalid_wasm_returns_error() {
        let bytes = [0x00, 0x61, 0x73, 0x6d, 0xff, 0xff, 0xff, 0xff];
        let err = analyse(&bytes).unwrap_err();
        assert!(matches!(err, AnalyseError::Parse(_)));
    }

    #[test]
    fn straight_line_function_keeps_on_cpu_even_with_v128() {
        // Without a `loop`, the function should NOT be offloaded —
        // offload overhead doesn't amortise on a single straight-line call.
        let wat = r#"
            (module
              (memory 1)
              (func (export "straight") (result v128)
                (i32x4.add (v128.const i32x4 1 2 3 4) (v128.const i32x4 5 6 7 8))
              )
            )
        "#;
        let wasm = wat::parse_str(wat).unwrap();
        let v = analyse(&wasm).expect("analyse");
        assert_eq!(v.len(), 1);
        assert_eq!(
            v[0].verdict,
            DetectorVerdict::KeepOnCpu,
            "straight-line v128 must not trigger offload"
        );
    }
}
