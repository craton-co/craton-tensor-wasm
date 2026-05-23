//! Optionally validates the emitted PTX with the host `ptxas` tool.
//!
//! Enabled by setting the env var `BALI_PTXAS=/path/to/ptxas` (or
//! `BALI_PTXAS=ptxas` if it's on PATH). Skipped silently otherwise so the
//! test suite stays green on hosts without a CUDA toolkit.

use std::process::Command;

use bali_jit::ir::{BaliKernelBlueprint, BaliOp, GridHint};
use bali_jit::ptx_emit::emit;

fn ptxas_path() -> Option<String> {
    std::env::var("BALI_PTXAS").ok()
}

fn run_ptxas_or_skip(ptx: &str, label: &str) {
    let Some(ptxas) = ptxas_path() else {
        eprintln!("BALI_PTXAS not set; skipping ptxas validation for {label}");
        return;
    };
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("bali_ptx_{}.ptx", label));
    std::fs::write(&tmp, ptx).expect("write tmp ptx");
    let out = Command::new(&ptxas)
        .arg("--gpu-name")
        .arg("sm_80")
        .arg("-o")
        .arg(format!("{}.cubin", tmp.display()))
        .arg(&tmp)
        .output()
        .expect("spawn ptxas");
    assert!(
        out.status.success(),
        "ptxas rejected emitted PTX for {label}\nstderr:\n{}\nptx:\n{}",
        String::from_utf8_lossy(&out.stderr),
        ptx,
    );
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(format!("{}.cubin", tmp.display()));
}

#[test]
fn emitted_vector_add_validates_with_ptxas() {
    let bp = BaliKernelBlueprint::new("vector_add")
        .push(BaliOp::LoadUnified { lanes: 4 })
        .push(BaliOp::LoadUnified { lanes: 4 })
        .push(BaliOp::VecAdd { lanes: 4 })
        .push(BaliOp::StoreUnified { lanes: 4 })
        .with_grid(GridHint {
            total_threads: 1024,
            preferred_block_size: 128,
        });
    let out = emit(&bp);
    run_ptxas_or_skip(&out.text, "vector_add");
}

#[test]
fn emitted_matmul_validates_with_ptxas() {
    let bp = BaliKernelBlueprint::new("matmul_16x16x16")
        .push(BaliOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        })
        .with_grid(GridHint {
            total_threads: 256,
            preferred_block_size: 128,
        });
    let out = emit(&bp);
    run_ptxas_or_skip(&out.text, "matmul");
}
