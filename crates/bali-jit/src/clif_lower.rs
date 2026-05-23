//! Lower a [`crate::detector::BlockIR`] into a
//! [`crate::ir::BaliKernelBlueprint`].

use thiserror::Error;

use crate::detector::{BlockIR, Op};
use crate::ir::{BaliKernelBlueprint, BaliOp, GridHint};

/// Errors produced by the lowering pass.
#[derive(Debug, Error, PartialEq)]
pub enum LowerError {
    /// The block contained operations the lowering pass doesn't yet handle.
    #[error("unsupported op in block `{block}`: {op:?}")]
    UnsupportedOp {
        /// Block name from `BlockIR::name`.
        block: String,
        /// The op the lowering didn't know how to handle.
        op: Op,
    },
    /// The lowering refused because of a memory-safety concern (e.g. an
    /// out-of-bounds load pattern).
    #[error("memory-reference out of bounds in block `{block}`")]
    OutOfBoundsMemory {
        /// Block name.
        block: String,
    },
}

/// Default lane width used when we don't have better info from the IR.
const DEFAULT_LANES: u32 = 4;

/// A memory reference seen in the source block: an absolute byte offset
/// into the instance's linear memory and an access width in bytes. The
/// lowering pass uses these to refuse offload candidates whose accesses
/// extend past the instance's allocated `UnifiedBuffer` extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRef {
    /// Byte offset into the instance linear memory.
    pub offset: u64,
    /// Number of bytes accessed.
    pub len: u64,
}

/// Lower a [`BlockIR`] into a [`BaliKernelBlueprint`].
///
/// The mapping is intentionally narrow:
///
/// | Source                                       | Target                          |
/// |----------------------------------------------|---------------------------------|
/// | `Op::V128Add`                                | `BaliOp::VecAdd { lanes: 4 }`   |
/// | `Op::V128Mul`                                | `BaliOp::VecMul { lanes: 4 }`   |
/// | `Op::V128Fma`                                | `BaliOp::VecFma { lanes: 4 }`   |
/// | `Op::Load`                                   | `BaliOp::LoadUnified { 4 }`     |
/// | `Op::Store`                                  | `BaliOp::StoreUnified { 4 }`    |
/// | Anything else                                | `UnsupportedOp` error           |
pub fn lower_block(block: &BlockIR) -> Result<BaliKernelBlueprint, LowerError> {
    let mut bp = BaliKernelBlueprint::new(&block.name);
    for op in &block.ops {
        let lo = match op {
            Op::V128Add => BaliOp::VecAdd {
                lanes: DEFAULT_LANES,
            },
            Op::V128Mul => BaliOp::VecMul {
                lanes: DEFAULT_LANES,
            },
            Op::V128Fma => BaliOp::VecFma {
                lanes: DEFAULT_LANES,
            },
            Op::Load => BaliOp::LoadUnified {
                lanes: DEFAULT_LANES,
            },
            Op::Store => BaliOp::StoreUnified {
                lanes: DEFAULT_LANES,
            },
            unsupported => {
                return Err(LowerError::UnsupportedOp {
                    block: block.name.clone(),
                    op: *unsupported,
                });
            }
        };
        bp.ops.push(lo);
    }

    let total_threads = block.loop_trip_count.unwrap_or(256) as u32;
    bp.grid_hint = GridHint {
        total_threads,
        preferred_block_size: 128,
    };
    Ok(bp)
}

/// Lower a [`BlockIR`] with explicit memory-reference validation.
///
/// `mem_refs` is the list of memory accesses extracted from the source
/// block by the caller's side-band analysis (e.g. `wasmparser`). Each
/// `(offset, len)` pair must satisfy `offset + len <= memory_bytes`;
/// any violation aborts with [`LowerError::OutOfBoundsMemory`] so the
/// caller falls back to CPU execution.
pub fn lower_block_checked(
    block: &BlockIR,
    mem_refs: &[MemRef],
    memory_bytes: u64,
) -> Result<BaliKernelBlueprint, LowerError> {
    for r in mem_refs {
        let end = r.offset.checked_add(r.len);
        let out_of_bounds = match end {
            Some(end) => end > memory_bytes,
            None => true, // overflow → treat as OOB
        };
        if out_of_bounds {
            return Err(LowerError::OutOfBoundsMemory {
                block: block.name.clone(),
            });
        }
    }
    lower_block(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, ops: Vec<Op>, loop_n: Option<u64>) -> BlockIR {
        BlockIR::new(name, ops, loop_n)
    }

    #[test]
    fn lower_simple_vector_add() {
        let b = block(
            "vec_add",
            vec![Op::Load, Op::Load, Op::V128Add, Op::Store],
            Some(128),
        );
        let bp = lower_block(&b).unwrap();
        assert_eq!(bp.entry, "vec_add");
        assert_eq!(bp.ops.len(), 4);
        assert!(matches!(bp.ops[2], BaliOp::VecAdd { lanes: 4 }));
        assert_eq!(bp.grid_hint.total_threads, 128);
    }

    #[test]
    fn unsupported_op_rejected() {
        let b = block("with_call", vec![Op::V128Add, Op::Call], Some(64));
        let err = lower_block(&b).unwrap_err();
        assert!(matches!(err, LowerError::UnsupportedOp { .. }));
    }

    #[test]
    fn missing_trip_count_uses_default() {
        let b = block("dyn", vec![Op::V128Add], None);
        let bp = lower_block(&b).unwrap();
        assert_eq!(bp.grid_hint.total_threads, 256);
    }

    #[test]
    fn all_v128_lower() {
        let b = block(
            "fma_loop",
            vec![Op::V128Fma, Op::V128Fma, Op::V128Mul],
            Some(64),
        );
        let bp = lower_block(&b).unwrap();
        for op in &bp.ops {
            assert!(matches!(
                op,
                BaliOp::VecFma { .. } | BaliOp::VecMul { .. } | BaliOp::VecAdd { .. }
            ));
        }
    }

    #[test]
    fn lower_block_checked_passes_with_in_bounds_refs() {
        let b = block("vec_add", vec![Op::Load, Op::V128Add, Op::Store], Some(128));
        let refs = &[
            MemRef { offset: 0, len: 16 },
            MemRef {
                offset: 1024,
                len: 16,
            },
        ];
        let bp = lower_block_checked(&b, refs, 64 * 1024).expect("in-bounds ok");
        assert_eq!(bp.ops.len(), 3);
    }

    #[test]
    fn lower_block_checked_rejects_out_of_bounds_refs() {
        let b = block("oob", vec![Op::Load], Some(64));
        // memory_bytes = 1024 but the ref reads past it.
        let refs = &[MemRef {
            offset: 1020,
            len: 16,
        }];
        let err = lower_block_checked(&b, refs, 1024).unwrap_err();
        assert!(matches!(err, LowerError::OutOfBoundsMemory { .. }));
    }

    #[test]
    fn lower_block_checked_rejects_overflow_refs() {
        let b = block("overflow", vec![Op::Load], Some(64));
        let refs = &[MemRef {
            offset: u64::MAX - 4,
            len: 16,
        }];
        let err = lower_block_checked(&b, refs, 1024).unwrap_err();
        assert!(matches!(err, LowerError::OutOfBoundsMemory { .. }));
    }
}
