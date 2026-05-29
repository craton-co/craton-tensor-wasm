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
//! ## What is intentionally NOT lowered
//!
//! [`TensorWasmOp::MatMul`] is currently rejected at emit time with
//! [`EmitError::NotYetImplemented`]. A real wmma lowering needs fragment-
//! handle materialisation (loading the `a`/`b` operand tiles into the
//! `%r0..%r7` row/col handles) and a paired `StoreUnified` that writes the
//! accumulator fragments back to global memory — neither of which the
//! current IR-to-PTX path encodes. Emitting a syntactically-valid-but-
//! semantically-broken wmma block would silently corrupt GPU state on
//! launch, so we refuse instead. See v0.4 roadmap.

use std::fmt::Write;

use thiserror::Error;

use crate::ir::{TensorWasmKernelBlueprint, TensorWasmOp};

/// Errors produced by the PTX emitter.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum EmitError {
    /// The blueprint contains an op the emitter does not yet lower to PTX.
    /// Callers should treat this as "keep this function on the CPU path".
    #[error("PTX emission not yet implemented: {0}")]
    NotYetImplemented(&'static str),
    /// The blueprint's `entry` field is not a valid PTX identifier.
    ///
    /// Today every legitimate caller constructs `entry` via
    /// `format!("func{idx}")` so this can never fire — but the Pliron
    /// `UserFuncName::Testcase` path can carry tenant-controlled bytes that
    /// would otherwise be interpolated unescaped into the PTX template
    /// (closing the kernel scope, emitting a second adversary kernel, etc.).
    /// Closes jit S-1.
    #[error("invalid PTX entry-name: {entry}")]
    InvalidEntryName {
        /// The rejected identifier (echoed for diagnostic purposes only —
        /// callers MUST NOT include the value in untrusted-facing errors).
        entry: String,
    },
    /// The blueprint demands more SSA registers than the `%fN` index space
    /// (`u32`) can address. Previously the allocator saturated at
    /// `u32::MAX`, which silently aliased two distinct SSA values onto the
    /// same physical register and corrupted results. We now refuse the
    /// blueprint so the caller (rewrite.rs) deopts to the CPU path rather
    /// than launch a miscompiled kernel. Closes jit L1.
    #[error("PTX register allocation overflowed the u32 index space")]
    TooManyRegisters,
}

/// Maximum length of a PTX identifier (NVIDIA PTX ISA §A.3).
pub const MAX_PTX_IDENTIFIER_LEN: usize = 1024;

/// Returns `true` iff `s` matches `[A-Za-z_][A-Za-z0-9_$]*` and is no
/// longer than [`MAX_PTX_IDENTIFIER_LEN`] bytes.
///
/// This is the validator that closes jit S-1 (PTX injection via guest-
/// controlled entry name). Every value that ends up interpolated into
/// the `.visible .entry {entry}(` template MUST be screened through this
/// function first.
#[must_use]
pub fn is_valid_ptx_identifier(s: &str) -> bool {
    if s.is_empty() || s.len() > MAX_PTX_IDENTIFIER_LEN {
        return false;
    }
    let mut bytes = s.bytes();
    let first = bytes.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return false;
    }
    bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$')
}

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
///
/// Returns [`EmitError::NotYetImplemented`] for blueprints containing ops
/// the emitter cannot lower (currently: [`TensorWasmOp::MatMul`]).
pub fn emit(blueprint: &TensorWasmKernelBlueprint) -> Result<EmittedPtx, EmitError> {
    emit_with(blueprint, &EmitConfig::default())
}

/// Allocate fresh registers and track the per-op body lines.
///
/// Returns `(body_text, max_f32_reg, max_s32_reg, max_s64_reg, max_pred_reg)`
/// where `max_*` is one greater than the highest register index used (i.e.
/// the count to declare in `.reg .f32 %f<N>;`). Counts are floored at 1 so
/// the declaration is always well-formed even for empty kernels.
fn lower_body(
    blueprint: &TensorWasmKernelBlueprint,
) -> Result<(String, u32, u32, u32, u32), EmitError> {
    // PERF (T20): pre-size the body buffer rather than letting `writeln!`
    // grow it by doubling. Each lowered op emits a comment line plus one
    // or more instruction lines (`add.f32 %f… %f… %f…;`, ~32 ASCII chars
    // each); 80 chars per op is a comfortable upper-bound average across
    // VecAdd/VecMul/VecFma/Load/Store lowering, so this single allocation
    // covers the common case without a single realloc. Empty blueprints
    // still get a usable (zero-byte) capacity.
    let mut body = String::with_capacity(blueprint.ops.len() * 80);
    // Pre-size the value stack to `ops.len()` — each lowered op pushes at
    // most one register, and most pop more than they push, so this is a
    // safe upper bound that avoids grow-by-doubling reallocs on the hot
    // emit path.
    let mut value_stack: Vec<u32> = Vec::with_capacity(blueprint.ops.len());
    let mut next_f = 0u32;
    // Counters for the other reg classes — track high-water mark so the
    // header `.reg` declarations are exactly sized. `%r0` carries the
    // element count `n` from the param load in the prologue; nothing else
    // currently uses the s32 reg class.
    let max_s32 = 1u32;
    let max_s64 = 2u32; // %rd0 (in_ptr), %rd1 (out_ptr)
    let max_pred = 2u32;
    // Running byte offset into the input / output buffers; advances 4 bytes
    // per lane (f32 = 4 bytes). Loads consume from `%rd0+in_off`; stores
    // consume from `%rd1+out_off`. Wraps via wrapping_add — overflow at u32
    // boundary is theoretical (would require ~4 GiB of args) but explicit.
    let mut in_off: u32 = 0;
    let mut out_off: u32 = 0;

    // Allocate a fresh `%fN` register. Returns `EmitError::TooManyRegisters`
    // rather than saturating at `u32::MAX` — saturation would alias two
    // distinct SSA values onto the same physical register (jit L1).
    let alloc_f = |next_f: &mut u32| -> Result<u32, EmitError> {
        let r = *next_f;
        *next_f = next_f.checked_add(1).ok_or(EmitError::TooManyRegisters)?;
        Ok(r)
    };
    // Pop an operand register off the value stack, or allocate a fresh one
    // if the stack underflows (an op consuming more than the abstract stack
    // holds). Threads the fallible allocator through the underflow path.
    let pop_or_alloc = |value_stack: &mut Vec<u32>, next_f: &mut u32| -> Result<u32, EmitError> {
        match value_stack.pop() {
            Some(r) => Ok(r),
            None => alloc_f(next_f),
        }
    };

    for op in &blueprint.ops {
        match op {
            TensorWasmOp::VecAdd { lanes } => {
                let _ = writeln!(body, "    // vec_add[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let b = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let a = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let dst = alloc_f(&mut next_f)?;
                    let _ = writeln!(body, "    add.f32 %f{dst}, %f{a}, %f{b};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::VecMul { lanes } => {
                let _ = writeln!(body, "    // vec_mul[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let b = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let a = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let dst = alloc_f(&mut next_f)?;
                    let _ = writeln!(body, "    mul.f32 %f{dst}, %f{a}, %f{b};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::VecFma { lanes } => {
                let _ = writeln!(body, "    // vec_fma[{}] lanes", lanes);
                for _ in 0..*lanes {
                    let c = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let b = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let a = pop_or_alloc(&mut value_stack, &mut next_f)?;
                    let dst = alloc_f(&mut next_f)?;
                    let _ = writeln!(body, "    fma.rn.f32 %f{dst}, %f{a}, %f{b}, %f{c};");
                    value_stack.push(dst);
                }
            }
            TensorWasmOp::MatMul { .. } => {
                // Lowering wmma m16n16k16 requires fragment-handle
                // materialisation (a/b operand tiles into `%r0..%r7`) and a
                // paired store that writes the accumulator fragments back
                // to global memory. Neither is encoded by the current IR-
                // to-PTX path. The previous emitter wrote a wmma.mma.sync
                // line referencing undefined `%r1..%r7` and never emitted
                // the matching store, so the kernel would have read
                // garbage and discarded its results. Refuse instead — the
                // caller (rewrite.rs) treats this as a deopt and keeps
                // the function on the CPU path. Deferred to v0.4.
                return Err(EmitError::NotYetImplemented(
                    "MatMul lowering deferred to v0.4",
                ));
            }
            TensorWasmOp::LoadUnified { lanes } => {
                let _ = writeln!(
                    body,
                    "    // load_unified[{}] lanes (.lu cache hint) from %rd0+{in_off}",
                    lanes
                );
                for _ in 0..*lanes {
                    let dst = alloc_f(&mut next_f)?;
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
                    let src = pop_or_alloc(&mut value_stack, &mut next_f)?;
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
    Ok((body, max_f, max_s32, max_s64, max_pred))
}

/// Emit PTX text for a blueprint with caller-supplied config.
///
/// Returns [`EmitError::NotYetImplemented`] for blueprints containing ops
/// the emitter cannot lower (currently: [`TensorWasmOp::MatMul`]).
pub fn emit_with(
    blueprint: &TensorWasmKernelBlueprint,
    cfg: &EmitConfig,
) -> Result<EmittedPtx, EmitError> {
    // SECURITY (jit S-1): every `blueprint.entry` value reaches the PTX
    // template unescaped through several `writeln!` sites below
    // (`.visible .entry {entry}(`, `{entry}_param_in_ptr`, etc.). A
    // tenant-controlled entry like `myfunc) … \n.entry attacker(...)`
    // would close the kernel scope and emit a second kernel. Reject
    // anything that isn't a well-formed PTX identifier BEFORE writing
    // a byte of output.
    if !is_valid_ptx_identifier(&blueprint.entry) {
        return Err(EmitError::InvalidEntryName {
            entry: blueprint.entry.clone(),
        });
    }
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
    let (body, max_f, max_s32, max_s64, max_pred) = lower_body(blueprint)?;

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

    Ok(EmittedPtx {
        text,
        launch_geometry: (grid_size, block_size),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{GridHint, TensorWasmKernelBlueprint, TensorWasmOp};

    #[test]
    fn vector_add_emits_add_f32() {
        let bp = TensorWasmKernelBlueprint::new("vector_add")
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::VecAdd { lanes: 4 })
            .push(TensorWasmOp::StoreUnified { lanes: 4 });
        let out = emit(&bp).expect("emit");
        assert!(out.text.contains(".target sm_80"));
        assert!(out.text.contains(".visible .entry vector_add"));
        assert!(out.text.contains("add.f32"));
        assert!(out.text.contains("ld.global.lu.f32"));
        assert!(out.text.contains("st.global.cs.f32"));
        assert!(out.text.contains("ret;"));
    }

    /// MatMul lowering is deferred to v0.4. The emitter must refuse to
    /// produce PTX for blueprints containing a `MatMul` op rather than
    /// emit a syntactically-valid-but-semantically-broken wmma block
    /// (the prior emitter referenced undefined `%r1..%r7` and never paired
    /// the accumulators with a store, so launched kernels would silently
    /// corrupt state). Callers should treat this as a deopt signal.
    #[test]
    fn matmul_emission_is_not_yet_implemented() {
        let bp = TensorWasmKernelBlueprint::new("matmul_16x16x16").push(TensorWasmOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        let err = emit(&bp).expect_err("MatMul emission must fail until v0.4");
        assert!(matches!(err, EmitError::NotYetImplemented(_)));
    }

    /// MatMul anywhere in the op stream taints the whole blueprint —
    /// not just blueprints whose only op is MatMul.
    #[test]
    fn matmul_in_mixed_stream_also_refused() {
        let bp = TensorWasmKernelBlueprint::new("mixed")
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::MatMul {
                m: 16,
                n: 16,
                k: 16,
            })
            .push(TensorWasmOp::StoreUnified { lanes: 4 });
        assert!(matches!(emit(&bp), Err(EmitError::NotYetImplemented(_))));
    }

    #[test]
    fn ptx_header_includes_version_and_target() {
        let bp = TensorWasmKernelBlueprint::new("noop");
        let out = emit(&bp).expect("emit");
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
        let out = emit(&bp).expect("emit");
        assert_eq!(out.launch_geometry, (8, 128));
        assert!(out.text.contains(".maxntid 128, 1, 1"));
        // Negative assertion: the broken C++-annotation form must not appear.
        assert!(!out.text.contains("__launch_bounds__"));
    }

    #[test]
    fn barrier_emits_bar_sync() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::Barrier);
        let out = emit(&bp).expect("emit");
        assert!(out.text.contains("bar.sync 0;"));
    }

    #[test]
    fn fma_emits_fma_rn() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecFma { lanes: 4 });
        let out = emit(&bp).expect("emit");
        assert!(out.text.contains("fma.rn.f32"));
    }

    #[test]
    fn custom_target() {
        let bp = TensorWasmKernelBlueprint::new("k");
        let cfg = EmitConfig {
            target: "sm_89".into(),
            ..EmitConfig::default()
        };
        let out = emit_with(&bp, &cfg).expect("emit");
        assert!(out.text.contains(".target sm_89"));
    }

    #[test]
    fn emitted_text_nonempty() {
        let bp = TensorWasmKernelBlueprint::new("k");
        let out = emit(&bp).expect("emit");
        assert!(!out.is_empty());
    }

    /// Critical correctness assertion: each lane-op produces a *fresh*
    /// register. The old emitter cycled `%f0..%f3` which silently read
    /// undefined values after four ops. With proper allocation, an 8-lane
    /// add produces `%f0..%fN` with strictly increasing destinations.
    #[test]
    fn register_allocator_assigns_fresh_destination_per_op() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::VecAdd { lanes: 8 });
        let out = emit(&bp).expect("emit");
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
        let out = emit(&bp).expect("emit");
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
        let out = emit(&bp).expect("emit");
        assert!(out.text.contains("ld.param.u64 %rd0"));
        assert!(out.text.contains("ld.param.u64 %rd1"));
        assert!(out.text.contains("ld.param.u32 %r0"));
    }

    #[test]
    fn loads_advance_input_offset() {
        let bp = TensorWasmKernelBlueprint::new("k").push(TensorWasmOp::LoadUnified { lanes: 3 });
        let out = emit(&bp).expect("emit");
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
        let out = emit(&bp).expect("emit");
        assert!(out.text.contains("[%rd1]"));
        assert!(out.text.contains("[%rd1+4]"));
    }
}
