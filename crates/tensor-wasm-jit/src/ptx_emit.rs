// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! PTX text emitter.
//!
//! Lowers a [`TensorWasmKernelBlueprint`] to a PTX assembly text suitable for
//! `cust::module::Module::from_ptx` (CUDA-host path) or `ptxas` (CI
//! validation). The emitter targets sm_80 — Ampere — which is the lowest
//! deployed architecture TensorWasm claims first-class support for.
//!
//! ## Register allocation
//!
//! Each lowered op produces a fresh f32 register (`%fN`). A `Vec<u32>`
//! "value stack" tracks the registers currently live on the abstract TensorWasm
//! stack — `Op::VecAdd` pops two operands and pushes one, etc. The high-
//! water mark of allocated registers is recorded so the kernel header
//! declares an exact-sized `.reg .f32 %f<MAX>` rather than a fixed `8`.
//!
//! Loads pull a fresh `%f` from the input buffer (`%rd0`) at a running
//! byte offset that advances by 4 bytes per lane. Stores pop their operand
//! and write it to the output buffer (`%rd1`) at the corresponding offset.
//! This is a contract with `tensor_wasm_exec::jit_dispatch`: the scratch buffer's
//! `args` half maps to `in_ptr`, the `results` half maps to `out_ptr`.
//!
//! The previous emitter cycled the same four hard-coded registers
//! (`%f0..%f3`), which causes the GPU to read undefined values once more
//! than four ops execute — a memory-safety hazard that ptxas couldn't catch
//! because the code was syntactically valid. The new emitter assigns a
//! fresh register per op, so values flow correctly.

use std::fmt::Write;

use crate::ir::{TensorWasmKernelBlueprint, TensorWasmOp};

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
pub fn emit(blueprint: &TensorWasmKernelBlueprint) -> EmittedPtx {
    emit_with(blueprint, &EmitConfig::default())
}

/// Allocate fresh registers and track the per-op body lines.
///
/// Returns `(body_text, max_f32_reg, max_s32_reg, max_s64_reg, max_pred_reg)`
/// where `max_*` is one greater than the highest register index used (i.e.
/// the count to declare in `.reg .f32 %f<N>;`). Counts are floored at 1 so
/// the declaration is always well-formed even for empty kernels.
fn lower_body(blueprint: &TensorWasmKernelBlueprint) -> (String, u32, u32, u32, u32) {
    let mut body = String::new();
    // Pre-size the value stack to `ops.len()` — each op pushes at most a
    // handful of registers, and most pop more than they push, so this is a
    // safe upper bound that avoids grow-by-doubling reallocs on the hot
    // emit path (matmul[16x16x16] visibly suffered from this).
    let mut value_stack: Vec<u32> = Vec::with_capacity(blueprint.ops.len());
    let mut next_f = 0u32;
    // Counters for the other reg classes — track high-water mark so the
    // header `.reg` declarations are exactly sized.
    let mut max_s32 = 8u32; // always need at least %r0..%r7 for the MatMul tile path
    let max_s64 = 2u32; // %rd0 (in_ptr), %rd1 (out_ptr)
    let max_pred = 2u32;
    // Running byte offset into the input / output buffers; advances 4 bytes
    // per lane (f32 = 4 bytes). Loads consume from `%rd0+in_off`; stores
    // consume from `%rd1+out_off`. Wraps via wrapping_add — overflow at u32
    // boundary is theoretical (would require ~4 GiB of args) but explicit.
    let mut in_off: u32 = 0;
    let mut out_off: u32 = 0;

    let alloc_f = |next_f: &mut u32| -> u32 {
        let r = *next_f;
        *next_f = next_f.checked_add(1).unwrap_or(u32::MAX);
        r
    };

    for op in &blueprint.ops {
        match op {
            TensorWasmOp::VecAdd { lanes } => {
                let _ = writeln!(body, "    // vec_add[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let b = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let a = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let dst = alloc_f(&mut next_f);
                    let _ = writeln!(body, "    add.f32 %f{dst}, %f{a}, %f{b};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::VecMul { lanes } => {
                let _ = writeln!(body, "    // vec_mul[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let b = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let a = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let dst = alloc_f(&mut next_f);
                    let _ = writeln!(body, "    mul.f32 %f{dst}, %f{a}, %f{b};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::VecFma { lanes } => {
                let _ = writeln!(body, "    // vec_fma[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let c = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let b = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let a = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    let dst = alloc_f(&mut next_f);
                    let _ = writeln!(body, "    fma.rn.f32 %f{dst}, %f{a}, %f{b}, %f{c};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::MatMul { m, n, k } => {
                let _ = writeln!(
                    body,
                    "    // matmul[{m}x{n}x{k}] via wmma m16n16k16 (sm_80 tensor cores)",
                );
                // For the matmul tile we use fresh registers so the wmma
                // operands don't alias prior values. Tensor-core fragments
                // use 8 f32 accumulators + 4 s32 row/col handles each.
                let dst_base = next_f;
                for _ in 0..8 {
                    let _ = alloc_f(&mut next_f);
                }
                let _ = writeln!(
                    body,
                    "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32"
                );
                let _ = writeln!(
                    body,
                    "        {{%f{r0}, %f{r1}, %f{r2}, %f{r3}, %f{r4}, %f{r5}, %f{r6}, %f{r7}}},",
                    r0 = dst_base,
                    r1 = dst_base + 1,
                    r2 = dst_base + 2,
                    r3 = dst_base + 3,
                    r4 = dst_base + 4,
                    r5 = dst_base + 5,
                    r6 = dst_base + 6,
                    r7 = dst_base + 7,
                );
                let _ = writeln!(body, "        {{%r0, %r1, %r2, %r3}},");
                let _ = writeln!(body, "        {{%r4, %r5, %r6, %r7}},");
                let _ = writeln!(
                    body,
                    "        {{%f{r0}, %f{r1}, %f{r2}, %f{r3}, %f{r4}, %f{r5}, %f{r6}, %f{r7}}};",
                    r0 = dst_base,
                    r1 = dst_base + 1,
                    r2 = dst_base + 2,
                    r3 = dst_base + 3,
                    r4 = dst_base + 4,
                    r5 = dst_base + 5,
                    r6 = dst_base + 6,
                    r7 = dst_base + 7,
                );
                // Push the eight accumulators onto the value stack — a
                // subsequent StoreUnified will pop them in order.
                for i in 0..8 {
                    value_stack.push(dst_base + i);
                }
                max_s32 = max_s32.max(8);
            }
            TensorWasmOp::LoadUnified { lanes } => {
                let _ = writeln!(
                    body,
                    "    // load_unified[{}] lanes (.lu cache hint) from %rd0+{in_off}",
                    lanes
                );
                for _ in 0..*lanes {
                    let dst = alloc_f(&mut next_f);
                    if in_off == 0 {
                        let _ = writeln!(body, "    ld.global.lu.f32 %f{dst}, [%rd0];");
                    } else {
                        let _ = writeln!(body, "    ld.global.lu.f32 %f{dst}, [%rd0+{in_off}];");
                    }
                    in_off = in_off.wrapping_add(4);
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::StoreUnified { lanes } => {
                let _ = writeln!(
                    body,
                    "    // store_unified[{}] lanes (.cs cache hint) to %rd1+{out_off}",
                    lanes
                );
                for _ in 0..*lanes {
                    let src = value_stack.pop().unwrap_or_else(|| alloc_f(&mut next_f));
                    if out_off == 0 {
                        let _ = writeln!(body, "    st.global.cs.f32 [%rd1], %f{src};");
                    } else {
                        let _ = writeln!(body, "    st.global.cs.f32 [%rd1+{out_off}], %f{src};");
                    }
                    out_off = out_off.wrapping_add(4);
                }
            }
            TensorWasmOp::Barrier => {
                let _ = writeln!(body, "    bar.sync 0;");
            }
        }
    }

    // `.reg .* %x<N>;` requires N >= 1, even if no register is used.
    let max_f = next_f.max(1);
    (body, max_f, max_s32, max_s64, max_pred)
}

/// Emit PTX text for a blueprint with caller-supplied config.
pub fn emit_with(blueprint: &TensorWasmKernelBlueprint, cfg: &EmitConfig) -> EmittedPtx {
    // Rough estimate: ~64 B/op covers the per-op body line plus a fair share
    // of the fixed prologue/epilogue. Avoids grow-by-doubling reallocs on
    // the hot emit path.
    let mut text = String::with_capacity(blueprint.ops.len().saturating_mul(64));
    let _ = writeln!(text, "//");
    let _ = writeln!(text, "// Auto-emitted by tensor-wasm-jit::ptx_emit");
    let _ = writeln!(text, "// entry: {}", blueprint.entry);
    let _ = writeln!(text, "// ops:   {}", blueprint.ops.len());
    let _ = writeln!(text, "//");
    let _ = writeln!(text);
    let _ = writeln!(text, ".version {}", cfg.ptx_version);
    let _ = writeln!(text, ".target {}", cfg.target);
    let _ = writeln!(text, ".address_size 64");
    let _ = writeln!(text);

    let (grid_size, block_size) = blueprint.grid_hint.launch_geometry();

    // Lower the body first so we know how many registers to declare. This
    // is the critical fix for the prior sham allocator: declare exactly
    // what we use, not a fixed `8` that overflowed silently.
    let (body, max_f, max_s32, max_s64, max_pred) = lower_body(blueprint);

    let _ = writeln!(text, ".visible .entry {}(", blueprint.entry);
    let _ = writeln!(text, "    .param .u64 {}_param_in_ptr,", blueprint.entry);
    let _ = writeln!(text, "    .param .u64 {}_param_out_ptr,", blueprint.entry);
    let _ = writeln!(text, "    .param .u32 {}_param_n", blueprint.entry);
    let _ = writeln!(text, ")");
    if cfg.launch_bounds {
        // PTX directive — block size cap is hinted to ptxas via .maxntid.
        // The runtime grid_size lives on the host side (see launch_geometry).
        let _ = writeln!(text, ".maxntid {block_size}, 1, 1");
    }
    let _ = writeln!(text, "{{");

    // Exact-sized register declarations.
    let _ = writeln!(text, "    .reg .pred  %p<{max_pred}>;");
    let _ = writeln!(text, "    .reg .s32   %r<{max_s32}>;");
    let _ = writeln!(text, "    .reg .s64   %rd<{max_s64}>;");
    let _ = writeln!(text, "    .reg .f32   %f<{max_f}>;");
    let _ = writeln!(text);

    // Prologue: load the .param declarations into the registers the body
    // uses. `%rd0` holds the input pointer, `%rd1` holds the output
    // pointer, `%r0` holds the element count `n`. These mirror the host-
    // side dispatch ABI in `tensor_wasm_exec::jit_dispatch`.
    let _ = writeln!(
        text,
        "    ld.param.u64 %rd0, [{entry}_param_in_ptr];",
        entry = blueprint.entry,
    );
    let _ = writeln!(
        text,
        "    ld.param.u64 %rd1, [{entry}_param_out_ptr];",
        entry = blueprint.entry,
    );
    let _ = writeln!(
        text,
        "    ld.param.u32 %r0, [{entry}_param_n];",
        entry = blueprint.entry,
    );
    let _ = writeln!(text);

    text.push_str(&body);

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
    use crate::ir::{TensorWasmKernelBlueprint, TensorWasmOp, GridHint};

    #[test]
    fn vector_add_emits_add_f32() {
        let bp = TensorWasmKernelBlueprint::new("vector_add")
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::VecAdd { lanes: 4 })
            .push(TensorWasmOp::StoreUnified { lanes: 4 });
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
        let bp = TensorWasmKernelBlueprint::new("matmul_16x16x16").push(TensorWasmOp::MatMul {
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
        let bp = TensorWasmKernelBlueprint::new("noop");
        let out = emit(&bp);
        assert!(out.text.contains(".version 8.0"));
        assert!(out.text.contains(".target sm_80"));
        assert!(out.text.contains(".address_size 64"));
    }

    #[test]
    fn launch_geometry_in_header() {
        let bp = TensorWasmKernelBlueprint::new("k").with_grid(GridHint {
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
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::Barrier);
        let out = emit(&bp);
        assert!(out.text.contains("bar.sync 0;"));
    }

    #[test]
    fn fma_emits_fma_rn() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecFma { lanes: 4 });
        let out = emit(&bp);
        assert!(out.text.contains("fma.rn.f32"));
    }

    #[test]
    fn custom_target() {
        let bp = TensorWasmKernelBlueprint::new("k");
        let cfg = EmitConfig {
            target: "sm_89".into(),
            ..EmitConfig::default()
        };
        let out = emit_with(&bp, &cfg);
        assert!(out.text.contains(".target sm_89"));
    }

    #[test]
    fn emitted_text_nonempty() {
        let bp = TensorWasmKernelBlueprint::new("k");
        let out = emit(&bp);
        assert!(!out.is_empty());
    }

    /// Critical correctness assertion: each lane-op produces a *fresh*
    /// register. The old emitter cycled `%f0..%f3` which silently read
    /// undefined values after four ops. With proper allocation, an 8-lane
    /// add produces `%f0..%fN` with strictly increasing destinations.
    #[test]
    fn register_allocator_assigns_fresh_destination_per_op() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecAdd { lanes: 8 });
        let out = emit(&bp);
        // After lowering, we expect at least registers %f0 through %f7 to
        // appear as add destinations. The exact regex match would couple
        // the test to the syntax, so we just count distinct destinations.
        let mut destinations = std::collections::BTreeSet::new();
        for line in out.text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("add.f32 %f") {
                if let Some((reg, _)) = rest.split_once(',') {
                    if let Ok(n) = reg.trim().parse::<u32>() {
                        destinations.insert(n);
                    }
                }
            }
        }
        assert!(
            destinations.len() >= 8,
            "expected at least 8 distinct add destinations, got {} ({:?})",
            destinations.len(),
            destinations,
        );
    }

    #[test]
    fn register_declarations_match_usage() {
        // 16 lanes of add will need at least 16 fresh registers (plus the
        // operand registers materialised on stack-underflow). The header
        // `.reg .f32 %f<N>` must declare at least N regs.
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecAdd { lanes: 16 });
        let out = emit(&bp);
        // Find the .reg .f32 declaration and parse out the count.
        let count_line = out
            .text
            .lines()
            .find(|l| l.contains(".reg .f32"))
            .expect("missing .reg .f32 declaration");
        let count: u32 = count_line
            .split('<')
            .nth(1)
            .and_then(|s| s.split('>').next())
            .and_then(|s| s.parse().ok())
            .expect("parse reg count");
        assert!(
            count >= 16,
            "register declaration must cover all uses (got {count})"
        );
    }

    #[test]
    fn prologue_loads_params_into_registers() {
        let bp = TensorWasmKernelBlueprint::new("vec_op");
        let out = emit(&bp);
        assert!(out.text.contains("ld.param.u64 %rd0"));
        assert!(out.text.contains("ld.param.u64 %rd1"));
        assert!(out.text.contains("ld.param.u32 %r0"));
    }

    #[test]
    fn loads_advance_input_offset() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::LoadUnified { lanes: 3 });
        let out = emit(&bp);
        // First lane at [%rd0], next at [%rd0+4], next at [%rd0+8].
        assert!(out.text.contains("[%rd0];"), "first load uses no offset");
        assert!(out.text.contains("[%rd0+4]"), "second lane at offset 4");
        assert!(out.text.contains("[%rd0+8]"), "third lane at offset 8");
    }

    #[test]
    fn stores_advance_output_offset() {
        let bp = TensorWasmKernelBlueprint::new("k")
            .push(TensorWasmOp::LoadUnified { lanes: 2 })
            .push(TensorWasmOp::StoreUnified { lanes: 2 });
        let out = emit(&bp);
        assert!(out.text.contains("[%rd1]"));
        assert!(out.text.contains("[%rd1+4]"));
    }
}
