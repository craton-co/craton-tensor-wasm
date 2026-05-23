//! PTX text emitter.
//!
//! Lowers a [`BaliKernelBlueprint`] to a PTX assembly text suitable for
//! `cust::module::Module::from_ptx` (CUDA-host path) or `ptxas` (CI
//! validation). The emitter targets sm_80 — Ampere — which is the lowest
//! deployed architecture Bali claims first-class support for.

use std::fmt::Write;

use crate::ir::{BaliKernelBlueprint, BaliOp};

/// Default PTX target architecture.
pub const DEFAULT_TARGET: &str = "sm_80";

/// Default PTX language version.
pub const DEFAULT_PTX_VERSION: &str = "8.0";

/// Configuration knobs for the emitter.
#[derive(Debug, Clone)]
pub struct EmitConfig {
    /// PTX `.target` directive value (e.g. "sm_80", "sm_89").
    pub target: String,
    /// PTX `.version` directive value (e.g. "8.0").
    pub ptx_version: String,
    /// Emit `__launch_bounds__` annotation on the entry.
    pub launch_bounds: bool,
}

impl Default for EmitConfig {
    fn default() -> Self {
        Self {
            target: DEFAULT_TARGET.to_string(),
            ptx_version: DEFAULT_PTX_VERSION.to_string(),
            launch_bounds: true,
        }
    }
}

/// Result of emitting a blueprint.
#[derive(Debug, Clone)]
pub struct EmittedPtx {
    /// The PTX text.
    pub text: String,
    /// Launch geometry the blueprint expects.
    pub launch_geometry: (u32, u32),
}

impl EmittedPtx {
    /// Byte length of the PTX text.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// True if the PTX text is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Emit PTX text for a blueprint with default config.
pub fn emit(blueprint: &BaliKernelBlueprint) -> EmittedPtx {
    emit_with(blueprint, &EmitConfig::default())
}

/// Emit PTX text for a blueprint with caller-supplied config.
pub fn emit_with(blueprint: &BaliKernelBlueprint, cfg: &EmitConfig) -> EmittedPtx {
    let mut text = String::new();
    let _ = writeln!(text, "//");
    let _ = writeln!(text, "// Auto-emitted by bali-jit::ptx_emit");
    let _ = writeln!(text, "// entry: {}", blueprint.entry);
    let _ = writeln!(text, "// ops:   {}", blueprint.ops.len());
    let _ = writeln!(text, "//");
    let _ = writeln!(text);
    let _ = writeln!(text, ".version {}", cfg.ptx_version);
    let _ = writeln!(text, ".target {}", cfg.target);
    let _ = writeln!(text, ".address_size 64");
    let _ = writeln!(text);

    let (grid_size, block_size) = blueprint.grid_hint.launch_geometry();
    let _ = writeln!(text, ".visible .entry {}(", blueprint.entry);
    let _ = writeln!(text, "    .param .u64 in_ptr,");
    let _ = writeln!(text, "    .param .u64 out_ptr,");
    let _ = writeln!(text, "    .param .u32 n");
    let _ = writeln!(text, ")");
    if cfg.launch_bounds {
        // PTX directive — block size cap is hinted to ptxas via .maxntid.
        // The runtime grid_size lives on the host side (see launch_geometry).
        let _ = writeln!(text, ".maxntid {block_size}, 1, 1");
    }
    let _ = writeln!(text, "{{");

    // Register-allocation comments and prologue.
    let _ = writeln!(text, "    .reg .pred  %p<2>;");
    let _ = writeln!(text, "    .reg .s32   %r<8>;");
    let _ = writeln!(text, "    .reg .s64   %rd<8>;");
    let _ = writeln!(text, "    .reg .f32   %f<8>;");
    let _ = writeln!(text);

    let mut reg_counter: u32 = 0;
    for op in &blueprint.ops {
        match op {
            BaliOp::VecAdd { lanes } => {
                let _ = writeln!(text, "    // vec_add[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let r = reg_counter % 4;
                    let _ = writeln!(
                        text,
                        "    add.f32 %f{}, %f{}, %f{};",
                        r,
                        (r + 1) % 4,
                        (r + 2) % 4,
                    );
                    reg_counter += 1;
                }
            }
            BaliOp::VecMul { lanes } => {
                let _ = writeln!(text, "    // vec_mul[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let r = reg_counter % 4;
                    let _ = writeln!(
                        text,
                        "    mul.f32 %f{}, %f{}, %f{};",
                        r,
                        (r + 1) % 4,
                        (r + 2) % 4,
                    );
                    reg_counter += 1;
                }
            }
            BaliOp::VecFma { lanes } => {
                let _ = writeln!(text, "    // vec_fma[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let r = reg_counter % 4;
                    let _ = writeln!(
                        text,
                        "    fma.rn.f32 %f{}, %f{}, %f{}, %f{};",
                        r,
                        (r + 1) % 4,
                        (r + 2) % 4,
                        (r + 3) % 4,
                    );
                    reg_counter += 1;
                }
            }
            BaliOp::MatMul { m, n, k } => {
                let _ = writeln!(
                    text,
                    "    // matmul[{m}x{n}x{k}] via wmma m16n16k16 (sm_80 tensor cores)",
                );
                let _ = writeln!(
                    text,
                    "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32"
                );
                let _ = writeln!(text, "        {{%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}},");
                let _ = writeln!(text, "        {{%r0, %r1, %r2, %r3}},");
                let _ = writeln!(text, "        {{%r4, %r5, %r6, %r7}},");
                let _ = writeln!(text, "        {{%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}};");
            }
            BaliOp::LoadUnified { lanes } => {
                let _ = writeln!(
                    text,
                    "    // load_unified[{}] lanes (.lu cache hint)",
                    lanes
                );
                for _ in 0..*lanes {
                    let r = reg_counter % 4;
                    let _ = writeln!(text, "    ld.global.lu.f32 %f{}, [%rd0];", r);
                    reg_counter += 1;
                }
            }
            BaliOp::StoreUnified { lanes } => {
                let _ = writeln!(
                    text,
                    "    // store_unified[{}] lanes (.cs cache hint)",
                    lanes
                );
                for _ in 0..*lanes {
                    let r = reg_counter % 4;
                    let _ = writeln!(text, "    st.global.cs.f32 [%rd1], %f{};", r);
                    reg_counter += 1;
                }
            }
            BaliOp::Barrier => {
                let _ = writeln!(text, "    bar.sync 0;");
            }
        }
    }

    let _ = writeln!(text);
    let _ = writeln!(text, "    ret;");
    let _ = writeln!(text, "}}");

    EmittedPtx {
        text,
        launch_geometry: (grid_size, block_size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BaliKernelBlueprint, BaliOp, GridHint};

    #[test]
    fn vector_add_emits_add_f32() {
        let bp = BaliKernelBlueprint::new("vector_add")
            .push(BaliOp::LoadUnified { lanes: 4 })
            .push(BaliOp::LoadUnified { lanes: 4 })
            .push(BaliOp::VecAdd { lanes: 4 })
            .push(BaliOp::StoreUnified { lanes: 4 });
        let out = emit(&bp);
        assert!(out.text.contains(".target sm_80"));
        assert!(out.text.contains(".visible .entry vector_add"));
        assert!(out.text.contains("add.f32"));
        assert!(out.text.contains("ld.global.lu.f32"));
        assert!(out.text.contains("st.global.cs.f32"));
        assert!(out.text.contains("ret;"));
    }

    #[test]
    fn matmul_emits_wmma() {
        let bp = BaliKernelBlueprint::new("matmul_16x16x16").push(BaliOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        let out = emit(&bp);
        assert!(out.text.contains("wmma.mma.sync"));
        assert!(out.text.contains("m16n16k16"));
    }

    #[test]
    fn ptx_header_includes_version_and_target() {
        let bp = BaliKernelBlueprint::new("noop");
        let out = emit(&bp);
        assert!(out.text.contains(".version 8.0"));
        assert!(out.text.contains(".target sm_80"));
        assert!(out.text.contains(".address_size 64"));
    }

    #[test]
    fn launch_geometry_in_header() {
        let bp = BaliKernelBlueprint::new("k").with_grid(GridHint {
            total_threads: 1024,
            preferred_block_size: 128,
        });
        let out = emit(&bp);
        assert_eq!(out.launch_geometry, (8, 128));
        assert!(out.text.contains(".maxntid 128, 1, 1"));
        // Negative assertion: the broken C++-annotation form must not appear.
        assert!(!out.text.contains("__launch_bounds__"));
    }

    #[test]
    fn barrier_emits_bar_sync() {
        let bp = BaliKernelBlueprint::new("k").push(BaliOp::Barrier);
        let out = emit(&bp);
        assert!(out.text.contains("bar.sync 0;"));
    }

    #[test]
    fn fma_emits_fma_rn() {
        let bp = BaliKernelBlueprint::new("k").push(BaliOp::VecFma { lanes: 4 });
        let out = emit(&bp);
        assert!(out.text.contains("fma.rn.f32"));
    }

    #[test]
    fn custom_target() {
        let bp = BaliKernelBlueprint::new("k");
        let cfg = EmitConfig {
            target: "sm_89".into(),
            ..EmitConfig::default()
        };
        let out = emit_with(&bp, &cfg);
        assert!(out.text.contains(".target sm_89"));
    }

    #[test]
    fn emitted_text_nonempty() {
        let bp = BaliKernelBlueprint::new("k");
        let out = emit(&bp);
        assert!(!out.is_empty());
    }
}
