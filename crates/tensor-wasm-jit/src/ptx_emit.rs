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
//! ## What is intentionally NOT lowered (default)
//!
//! [`TensorWasmOp::MatMul`] is **by default** rejected at emit time with
//! [`EmitError::NotYetImplemented`]. A real wmma lowering needs fragment-
//! handle materialisation (loading the `a`/`b` operand tiles into the
//! fragment register handles) and a paired store that writes the
//! accumulator fragments back to global memory — neither of which the
//! default IR-to-PTX path encodes. Emitting a syntactically-valid-but-
//! semantically-broken wmma block would silently corrupt GPU state on
//! launch, so we refuse instead. See v0.4 roadmap.
//!
//! ## EXPERIMENTAL / UNVERIFIED ON HARDWARE: opt-in wmma MatMul
//!
//! Setting [`EmitConfig::enable_experimental_matmul`] to `true` flips the
//! emitter into lowering [`TensorWasmOp::MatMul`] with `m == n == k == 16`
//! to a wmma `m16n16k16` fragment-load → `wmma.mma.sync` → fragment-store
//! sequence. This flag defaults to `false` and the safe refusal above
//! remains the default behaviour; nothing in the auto-offload pipeline
//! sets it. The generated PTX is **NOT verified on real GPU hardware** in
//! this environment — its structural correctness (fragment shapes, mma.sync
//! count, accumulator chaining, register declarations) is gated behind the
//! differential oracle in [`crate::differential`], which MUST pass before
//! anyone enables this flag in a shipping path.
//!
//! ### wmma m16n16k16 tile / fragment contract
//!
//! The lowering targets the sm_80 `wmma.mma.sync.aligned.row.col.m16n16k16`
//! shape with f16 operand fragments and an f32 accumulator (`.f32.f32`):
//!
//! * **Tile shape.** One warp cooperatively computes a single
//!   `C[16x16] += A[16x16] * B[16x16]` tile. A is loaded `row` major, B is
//!   loaded `col` major (the canonical `.row.col` operand layout).
//! * **Operand fragments.** Each of the A and B operand fragments is held
//!   in [`WMMA_OPERAND_FRAG_REGS`] (`8`) packed `.b32` registers per warp
//!   (two f16 lanes per `.b32`), materialised with
//!   `wmma.load.a`/`wmma.load.b`.
//! * **Accumulator fragment.** The C accumulator is held in
//!   [`WMMA_ACC_FRAG_REGS`] (`8`) `.f32` registers per warp, initialised
//!   from global memory with `wmma.load.c` and written back with
//!   `wmma.store.d`. The accumulator is *chained*: the `wmma.mma.sync`
//!   reads the same `%fa…` accumulator registers it writes (paired
//!   accumulator — `D = A*B + C` with `C` and `D` aliasing the same
//!   fragment), so back-to-back tiles over a longer `k` would accumulate
//!   correctly.
//! * **Alignment / leading dimension.** `wmma.load.*`/`wmma.store.*`
//!   require the base pointer to be **128-bit aligned** and the supplied
//!   leading-dimension (stride, in elements) to be a multiple of 8 for the
//!   f16 operands and a multiple of 4 for the f32 accumulator. We assume a
//!   tightly-packed row-major tile, so the stride equals the tile width
//!   (`16`) and the caller guarantees the `in_ptr`/`out_ptr` buffers are
//!   16-byte aligned (the unified-memory allocator already over-aligns to
//!   256 bytes, so this holds for every legitimate caller).

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

/// Number of `.b32` registers in one wmma `m16n16k16` operand (A or B)
/// fragment, per warp. f16 operands pack two lanes per `.b32`, so a
/// 16-wide row spread across the warp's 32 lanes lands in 8 registers.
pub const WMMA_OPERAND_FRAG_REGS: u32 = 8;

/// Number of `.f32` registers in the wmma `m16n16k16` f32 accumulator
/// fragment, per warp. The 16x16 = 256 accumulator elements distributed
/// over 32 lanes give 8 elements (registers) per lane.
pub const WMMA_ACC_FRAG_REGS: u32 = 8;

/// The only MatMul tile dimension the experimental wmma lowering accepts.
/// `m == n == k == 16` is the single sm_80 wmma shape this emitter models;
/// any other dimensions fall back to [`EmitError::NotYetImplemented`] even
/// when the experimental flag is on.
pub const WMMA_TILE_DIM: u32 = 16;

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
    /// EXPERIMENTAL / UNVERIFIED ON HARDWARE.
    ///
    /// When `true`, [`emit_with`] lowers a `MatMul { m: 16, n: 16, k: 16 }`
    /// op to a wmma `m16n16k16` fragment-load → `wmma.mma.sync` →
    /// fragment-store sequence (see the module docs for the tile/fragment
    /// contract). When `false` — the default — `MatMul` is refused with
    /// [`EmitError::NotYetImplemented`], the safe behaviour the
    /// auto-offload pipeline relies on.
    ///
    /// This MUST stay `false` outside of explicitly opted-in experiments:
    /// the emitted PTX cannot be validated on a GPU here, so its
    /// correctness is gated solely behind the differential oracle's
    /// structural assertions (see [`crate::differential`]).
    pub enable_experimental_matmul: bool,
}

impl Default for EmitConfig {
    fn default() -> Self {
        Self {
            target: DEFAULT_TARGET.to_string(),
            ptx_version: DEFAULT_PTX_VERSION.to_string(),
            launch_bounds: true,
            // SAFETY DEFAULT: MatMul stays refused unless a caller flips
            // this on for an experiment. Never default this to `true`.
            enable_experimental_matmul: false,
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
/// Returns `(body_text, max_f32_reg, max_s32_reg, max_s64_reg,
/// max_pred_reg, max_b32_reg)` where `max_*` is one greater than the
/// highest register index used (i.e. the count to declare in
/// `.reg .f32 %f<N>;`). Counts are floored at 1 so the declaration is
/// always well-formed even for empty kernels. `max_b32_reg` is the wmma
/// operand-fragment `.b32 %rb<N>` count and is `0` unless an experimental
/// MatMul was lowered.
fn lower_body(
    blueprint: &TensorWasmKernelBlueprint,
    cfg: &EmitConfig,
) -> Result<(String, u32, u32, u32, u32, u32), EmitError> {
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
    let mut max_s64 = 2u32; // %rd0 (in_ptr), %rd1 (out_ptr)
    let max_pred = 2u32;
    // EXPERIMENTAL: high-water mark of the `.b32` wmma operand-fragment
    // register class (`%rb<N>`). Stays 0 unless an experimental MatMul is
    // lowered, in which case the header declares exactly the fragment
    // registers the `wmma.load.a`/`wmma.load.b` ops use.
    let mut max_b32 = 0u32;
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
            TensorWasmOp::MatMul { m, n, k } => {
                // SAFETY DEFAULT: unless the caller explicitly opted into
                // the experimental wmma lowering, MatMul is refused so the
                // auto-offload pipeline deopts to the CPU path rather than
                // launch a kernel that cannot be verified on hardware here.
                if !cfg.enable_experimental_matmul {
                    return Err(EmitError::NotYetImplemented(
                        "MatMul lowering deferred to v0.4",
                    ));
                }
                // The experimental lowering models exactly one sm_80 wmma
                // shape: m16n16k16. Any other tile dimension is still
                // refused even with the flag on — emitting an unmodelled
                // shape would be the same silent-miscompile hazard the
                // refusal exists to prevent.
                if (*m, *n, *k) != (WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM) {
                    return Err(EmitError::NotYetImplemented(
                        "experimental wmma lowering only models m16n16k16",
                    ));
                }
                // EXPERIMENTAL / UNVERIFIED ON HARDWARE.
                //
                // wmma m16n16k16 fragment-load → wmma.mma.sync →
                // fragment-store. See the module docs for the full
                // tile/fragment + alignment contract. The PTX produced
                // here is validated ONLY by the differential oracle's
                // structural assertions; it has NOT been run on a GPU.
                lower_wmma_m16n16k16(
                    &mut body,
                    &mut next_f,
                    &mut max_b32,
                    &mut max_s64,
                )?;
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
    Ok((body, max_f, max_s32, max_s64, max_pred, max_b32))
}

/// EXPERIMENTAL / UNVERIFIED ON HARDWARE.
///
/// Lower one `MatMul { m: 16, n: 16, k: 16 }` op to a wmma `m16n16k16`
/// fragment-load → `wmma.mma.sync` → fragment-store sequence targeting
/// sm_80 (`.row.col`, f16 operands, f32 accumulator).
///
/// The emitted sequence is, per warp:
///
/// 1. Materialise the A operand fragment ([`WMMA_OPERAND_FRAG_REGS`]
///    `.b32` regs) from `%rd0` with `wmma.load.a.sync.aligned.row`.
/// 2. Materialise the B operand fragment (same width) from the second
///    tile in `%rd0` with `wmma.load.b.sync.aligned.col`.
/// 3. Load the C accumulator fragment ([`WMMA_ACC_FRAG_REGS`] `.f32`
///    regs) from `%rd0`'s third tile with `wmma.load.c.sync.aligned.row`.
/// 4. `wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32` — the accumulator
///    is **paired**: the same `%f` registers appear as both the `c`
///    operand and the `d` result, so `D = A*B + C` chains in place.
/// 5. Write the accumulator back to `%rd1` with
///    `wmma.store.d.sync.aligned.row`.
///
/// Register accounting:
/// * A and B fragments are allocated in the `.b32 %rb` class
///   (`max_b32` high-water mark) — `2 * WMMA_OPERAND_FRAG_REGS` registers.
/// * The accumulator is allocated in the existing `.f32 %f` class
///   (`next_f`) — [`WMMA_ACC_FRAG_REGS`] registers.
/// * The load/store base pointers are derived into the `.s64 %rd` class;
///   the highest `%rd` index used is recorded in `max_s64`.
///
/// Alignment / leading-dimension assumptions are documented on the module
/// (`row` stride == tile width == 16, 128-bit aligned base pointers).
fn lower_wmma_m16n16k16(
    body: &mut String,
    next_f: &mut u32,
    max_b32: &mut u32,
    max_s64: &mut u32,
) -> Result<(), EmitError> {
    // Allocate the contiguous operand-fragment register windows. Each
    // operand fragment occupies `WMMA_OPERAND_FRAG_REGS` `.b32` regs;
    // `wmma.load`/`mma.sync` name the window by its base index `%rbN`.
    let a_frag_base = *max_b32;
    *max_b32 = max_b32
        .checked_add(WMMA_OPERAND_FRAG_REGS)
        .ok_or(EmitError::TooManyRegisters)?;
    let b_frag_base = *max_b32;
    *max_b32 = max_b32
        .checked_add(WMMA_OPERAND_FRAG_REGS)
        .ok_or(EmitError::TooManyRegisters)?;

    // Accumulator fragment in the f32 class, chained through mma.sync.
    let acc_frag_base = *next_f;
    *next_f = next_f
        .checked_add(WMMA_ACC_FRAG_REGS)
        .ok_or(EmitError::TooManyRegisters)?;

    // Derive the three tile base pointers into the s64 class. `%rd0`
    // holds the input pointer (A tile); B and C tiles follow it at the
    // tile-stride offset (16x16 f16 = 512 B for A/B, 16x16 f32 = 1024 B
    // for C). `%rd1` holds the output (D tile) pointer. We materialise
    // %rd2 (B base) and %rd3 (C base) so the loads name distinct bases.
    let b_base = *max_s64; // %rd2
    let c_base = b_base + 1; // %rd3
    *max_s64 = c_base
        .checked_add(1)
        .ok_or(EmitError::TooManyRegisters)?; // declare through %rd3

    // Leading dimension (stride in elements) for the row/col loads. A
    // tightly-packed 16-wide tile has stride 16.
    let stride = WMMA_TILE_DIM;
    // Byte offsets of the B and C tiles within the input buffer. A is
    // 16x16 f16 (512 B); B is 16x16 f16 (512 B); C is 16x16 f32 (1024 B).
    const A_TILE_BYTES: u32 = WMMA_TILE_DIM * WMMA_TILE_DIM * 2; // f16
    const B_TILE_BYTES: u32 = WMMA_TILE_DIM * WMMA_TILE_DIM * 2; // f16
    let b_tile_off = A_TILE_BYTES;
    let c_tile_off = A_TILE_BYTES + B_TILE_BYTES;

    let _ = writeln!(
        body,
        "    // EXPERIMENTAL / UNVERIFIED ON HARDWARE: wmma m16n16k16 \
         (row.col, f16xf16->f32)"
    );

    // Render a `{%rbB, %rbB+1, …}` fragment register list.
    let frag_list = |base: u32, count: u32, prefix: &str| -> String {
        let regs: Vec<String> = (0..count).map(|i| format!("%{prefix}{}", base + i)).collect();
        format!("{{{}}}", regs.join(", "))
    };
    let a_list = frag_list(a_frag_base, WMMA_OPERAND_FRAG_REGS, "rb");
    let b_list = frag_list(b_frag_base, WMMA_OPERAND_FRAG_REGS, "rb");
    let acc_list = frag_list(acc_frag_base, WMMA_ACC_FRAG_REGS, "f");

    // Tile base-pointer setup.
    let _ = writeln!(body, "    add.s64 %rd{b_base}, %rd0, {b_tile_off};");
    let _ = writeln!(body, "    add.s64 %rd{c_base}, %rd0, {c_tile_off};");

    // 1+2. Operand-fragment loads (A row-major, B col-major).
    let _ = writeln!(
        body,
        "    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {a_list}, [%rd0], {stride};"
    );
    let _ = writeln!(
        body,
        "    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {b_list}, [%rd{b_base}], {stride};"
    );
    // 3. Accumulator load (paired: read C into the same regs mma writes).
    let _ = writeln!(
        body,
        "    wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {acc_list}, [%rd{c_base}], {stride};"
    );
    // 4. The mma. `d` and `c` alias `acc_list` — paired accumulator.
    let _ = writeln!(
        body,
        "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 \
         {acc_list}, {a_list}, {b_list}, {acc_list};"
    );
    // 5. Store the accumulator fragment back to the D tile in %rd1.
    let _ = writeln!(
        body,
        "    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd1], {acc_list}, {stride};"
    );

    Ok(())
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
    let (body, max_f, max_s32, max_s64, max_pred, max_b32) = lower_body(blueprint, cfg)?;

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
    // EXPERIMENTAL: wmma operand-fragment registers. Only declared when a
    // wmma MatMul was lowered (max_b32 > 0); otherwise omitted entirely so
    // the default-path PTX is byte-for-byte unchanged.
    if max_b32 > 0 {
        let _ = writeln!(text, "    .reg .b32   %rb<{max_b32}>;");
    }
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

    /// SAFETY-DEFAULT PIN: the default `EmitConfig` must keep refusing
    /// MatMul. This locks in the constraint that the experimental wmma
    /// lowering is opt-in only — a regression that flips the default on
    /// would fail here before it could reach a GPU launch path.
    #[test]
    fn matmul_refused_under_default_config() {
        let bp = TensorWasmKernelBlueprint::new("matmul_16x16x16").push(TensorWasmOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        // Default config: flag is off.
        assert!(!EmitConfig::default().enable_experimental_matmul);
        let err = emit_with(&bp, &EmitConfig::default())
            .expect_err("MatMul must be refused with the flag off");
        assert!(matches!(err, EmitError::NotYetImplemented(_)));
        // The convenience `emit` wrapper uses the default config, so it
        // must refuse too.
        assert!(matches!(emit(&bp), Err(EmitError::NotYetImplemented(_))));
    }

    /// EXPERIMENTAL opt-in: with the flag on, an m16n16k16 MatMul lowers
    /// to the wmma fragment-load → mma.sync → fragment-store sequence.
    #[test]
    fn matmul_opt_in_emits_wmma_sequence() {
        let bp = TensorWasmKernelBlueprint::new("matmul").push(TensorWasmOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        });
        let cfg = EmitConfig {
            enable_experimental_matmul: true,
            ..EmitConfig::default()
        };
        let out = emit_with(&bp, &cfg).expect("opt-in emit must succeed");
        // Fragment loads for both operands + the accumulator.
        assert!(out.text.contains("wmma.load.a.sync.aligned.row.m16n16k16"));
        assert!(out.text.contains("wmma.load.b.sync.aligned.col.m16n16k16"));
        assert!(out.text.contains("wmma.load.c.sync.aligned.row.m16n16k16"));
        // Exactly one mma.sync for a single k16 tile.
        assert_eq!(out.text.matches("wmma.mma.sync.aligned").count(), 1);
        // Paired store of the accumulator fragment.
        assert!(out.text.contains("wmma.store.d.sync.aligned.row.m16n16k16"));
        // The operand-fragment register class must be declared.
        assert!(out.text.contains(".reg .b32   %rb<"));
        // The prominent experimental marker rides in the body.
        assert!(out.text.contains("EXPERIMENTAL / UNVERIFIED ON HARDWARE"));
    }

    /// Even with the flag on, only the modelled m16n16k16 shape is
    /// lowered — any other tile is still refused (never silently
    /// miscompiled).
    #[test]
    fn matmul_opt_in_rejects_non_16_shape() {
        let bp = TensorWasmKernelBlueprint::new("matmul").push(TensorWasmOp::MatMul {
            m: 32,
            n: 32,
            k: 32,
        });
        let cfg = EmitConfig {
            enable_experimental_matmul: true,
            ..EmitConfig::default()
        };
        assert!(matches!(
            emit_with(&bp, &cfg),
            Err(EmitError::NotYetImplemented(_))
        ));
    }

    /// The default (flag-off) PTX must be byte-identical to before the
    /// experimental wmma support was added for non-MatMul blueprints: the
    /// `.b32 %rb` operand-fragment declaration only appears when a wmma
    /// MatMul is actually lowered.
    #[test]
    fn no_wmma_reg_decl_without_matmul() {
        let bp = TensorWasmKernelBlueprint::new("vector_add")
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::LoadUnified { lanes: 4 })
            .push(TensorWasmOp::VecAdd { lanes: 4 })
            .push(TensorWasmOp::StoreUnified { lanes: 4 });
        let out = emit(&bp).expect("emit");
        assert!(!out.text.contains("%rb<"));
        assert!(!out.text.contains("wmma."));
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
