#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use bali_jit::ir::{BaliKernelBlueprint, BaliOp, GridHint};
use bali_jit::ptx_emit::emit;

#[derive(Debug, Arbitrary)]
enum FuzzOp {
    VecAdd(u16),
    VecMul(u16),
    VecFma(u16),
    MatMul(u8, u8, u8),
    LoadUnified(u16),
    StoreUnified(u16),
    Barrier,
}

impl From<FuzzOp> for BaliOp {
    fn from(o: FuzzOp) -> Self {
        match o {
            FuzzOp::VecAdd(lanes) => BaliOp::VecAdd {
                lanes: lanes.max(1) as u32,
            },
            FuzzOp::VecMul(lanes) => BaliOp::VecMul {
                lanes: lanes.max(1) as u32,
            },
            FuzzOp::VecFma(lanes) => BaliOp::VecFma {
                lanes: lanes.max(1) as u32,
            },
            FuzzOp::MatMul(m, n, k) => BaliOp::MatMul {
                m: m.max(1) as u32,
                n: n.max(1) as u32,
                k: k.max(1) as u32,
            },
            FuzzOp::LoadUnified(lanes) => BaliOp::LoadUnified {
                lanes: lanes.max(1) as u32,
            },
            FuzzOp::StoreUnified(lanes) => BaliOp::StoreUnified {
                lanes: lanes.max(1) as u32,
            },
            FuzzOp::Barrier => BaliOp::Barrier,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(ops): Result<Vec<FuzzOp>, _> = u.arbitrary() else {
        return;
    };
    // Cap to a sane upper bound; longer streams just waste fuzzer cycles.
    if ops.len() > 1024 {
        return;
    }
    let total_threads: u32 = u.arbitrary().unwrap_or(256u32);
    let block_size: u32 = u.arbitrary().unwrap_or(128u32);
    let mut bp = BaliKernelBlueprint::new("fuzz_kernel").with_grid(GridHint {
        total_threads: total_threads.max(1),
        preferred_block_size: block_size.max(1),
    });
    for op in ops {
        bp = bp.push(op.into());
    }
    let _ = emit(&bp);
});
