//! `BaliIR` — a normalised IR easier to lower to PTX than raw CLIF/Wasm.
//!
//! The IR is the contract between the detector (S10), the lowering pass
//! (`clif_lower`, S11), the PTX emitter (`ptx_emit`, S12), and the kernel
//! cache (S13). Keeping it small and explicit means each successor stage
//! can be unit-tested without dragging in the others.

use std::fmt;

/// One coarse-grained op in a Bali-emitted GPU kernel.
///
/// Each op corresponds to a small group of PTX instructions; the emitter
/// (S12) materialises them into a register-allocated assembly block.
#[derive(Debug, Clone, PartialEq)]
pub enum BaliOp {
    /// `c = a + b` element-wise on f32 lanes.
    VecAdd {
        /// Lane count.
        lanes: u32,
    },
    /// `c = a * b` element-wise on f32 lanes.
    VecMul {
        /// Lane count.
        lanes: u32,
    },
    /// `d = a*b + c` element-wise on f32 lanes (single rounding).
    VecFma {
        /// Lane count.
        lanes: u32,
    },
    /// 16x16x16 matrix multiply-accumulate (wmma on sm_80).
    MatMul {
        /// Tile size m (currently fixed at 16).
        m: u32,
        /// Tile size n (currently fixed at 16).
        n: u32,
        /// Tile size k (currently fixed at 16).
        k: u32,
    },
    /// Load f32 lanes from unified-memory pointer `+ offset`.
    LoadUnified {
        /// Lane count.
        lanes: u32,
    },
    /// Store f32 lanes to unified-memory pointer `+ offset`.
    StoreUnified {
        /// Lane count.
        lanes: u32,
    },
    /// CTA-wide synchronisation barrier.
    Barrier,
}

impl fmt::Display for BaliOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaliOp::VecAdd { lanes } => write!(f, "vec_add[{lanes}]"),
            BaliOp::VecMul { lanes } => write!(f, "vec_mul[{lanes}]"),
            BaliOp::VecFma { lanes } => write!(f, "vec_fma[{lanes}]"),
            BaliOp::MatMul { m, n, k } => write!(f, "matmul[{m}x{n}x{k}]"),
            BaliOp::LoadUnified { lanes } => write!(f, "load_unified[{lanes}]"),
            BaliOp::StoreUnified { lanes } => write!(f, "store_unified[{lanes}]"),
            BaliOp::Barrier => f.write_str("barrier"),
        }
    }
}

/// Hint about CUDA launch geometry. Provided by the lowering pass when it
/// can infer a sensible grid/block from the source loop's trip count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridHint {
    /// Total number of threads to launch (across all blocks).
    pub total_threads: u32,
    /// Preferred CTA size; the emitter rounds the total to a multiple.
    pub preferred_block_size: u32,
}

impl Default for GridHint {
    fn default() -> Self {
        Self {
            total_threads: 256,
            preferred_block_size: 128,
        }
    }
}

impl GridHint {
    /// Compute the corresponding (grid_size, block_size) pair.
    pub fn launch_geometry(&self) -> (u32, u32) {
        let block = self.preferred_block_size.max(1);
        let grid = self.total_threads.div_ceil(block);
        (grid.max(1), block)
    }
}

/// A complete blueprint that the emitter (S12) can turn into PTX text.
#[derive(Debug, Clone, PartialEq)]
pub struct BaliKernelBlueprint {
    /// Symbolic entry name (used by `load_ptx` lookup).
    pub entry: String,
    /// Ordered ops to emit.
    pub ops: Vec<BaliOp>,
    /// Grid hint (S11 fills in; S12 honours).
    pub grid_hint: GridHint,
    /// Bytes of shared memory the kernel will request.
    pub shared_mem_bytes: u32,
}

impl BaliKernelBlueprint {
    /// Construct a new blueprint.
    pub fn new(entry: impl Into<String>) -> Self {
        Self {
            entry: entry.into(),
            ops: Vec::new(),
            grid_hint: GridHint::default(),
            shared_mem_bytes: 0,
        }
    }

    /// Append an op; returns `self` for builder-style chaining.
    pub fn push(mut self, op: BaliOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Update the grid hint.
    pub fn with_grid(mut self, hint: GridHint) -> Self {
        self.grid_hint = hint;
        self
    }

    /// Update the shared-memory requirement.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// True if the blueprint contains no ops (a no-op kernel).
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Stable hash used as the cache key by S13.
    pub fn fingerprint(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.entry.hash(&mut h);
        for op in &self.ops {
            std::mem::discriminant(op).hash(&mut h);
            // Hash lane / m / n / k metadata explicitly because Hash isn't
            // derived for the enum (lanes is on the variants).
            match op {
                BaliOp::VecAdd { lanes }
                | BaliOp::VecMul { lanes }
                | BaliOp::VecFma { lanes }
                | BaliOp::LoadUnified { lanes }
                | BaliOp::StoreUnified { lanes } => lanes.hash(&mut h),
                BaliOp::MatMul { m, n, k } => {
                    m.hash(&mut h);
                    n.hash(&mut h);
                    k.hash(&mut h);
                }
                BaliOp::Barrier => {}
            }
        }
        self.grid_hint.total_threads.hash(&mut h);
        self.grid_hint.preferred_block_size.hash(&mut h);
        self.shared_mem_bytes.hash(&mut h);
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueprint_builder() {
        let bp = BaliKernelBlueprint::new("vector_add")
            .push(BaliOp::LoadUnified { lanes: 4 })
            .push(BaliOp::VecAdd { lanes: 4 })
            .push(BaliOp::StoreUnified { lanes: 4 })
            .with_grid(GridHint {
                total_threads: 1024,
                preferred_block_size: 128,
            })
            .with_shared_mem(0);
        assert_eq!(bp.entry, "vector_add");
        assert_eq!(bp.ops.len(), 3);
        assert_eq!(bp.grid_hint.total_threads, 1024);
    }

    #[test]
    fn grid_geometry_rounds_up() {
        let g = GridHint {
            total_threads: 130,
            preferred_block_size: 64,
        };
        let (grid, block) = g.launch_geometry();
        assert_eq!(block, 64);
        assert_eq!(grid, 3); // ceil(130 / 64)
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let a = BaliKernelBlueprint::new("k").push(BaliOp::VecAdd { lanes: 4 });
        let b = BaliKernelBlueprint::new("k").push(BaliOp::VecAdd { lanes: 4 });
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_lanes() {
        let a = BaliKernelBlueprint::new("k").push(BaliOp::VecAdd { lanes: 4 });
        let b = BaliKernelBlueprint::new("k").push(BaliOp::VecAdd { lanes: 8 });
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn empty_blueprint_detectable() {
        let bp = BaliKernelBlueprint::new("empty");
        assert!(bp.is_empty());
    }

    #[test]
    fn op_display() {
        assert_eq!(BaliOp::VecAdd { lanes: 4 }.to_string(), "vec_add[4]");
        assert_eq!(
            BaliOp::MatMul {
                m: 16,
                n: 16,
                k: 16
            }
            .to_string(),
            "matmul[16x16x16]"
        );
        assert_eq!(BaliOp::Barrier.to_string(), "barrier");
    }
}
