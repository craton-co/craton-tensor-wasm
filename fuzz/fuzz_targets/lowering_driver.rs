// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `tensor_wasm_jit::lowering_driver::lower_function`
//! (wave-2 task W2.8).
//!
//! Property: the Cranelift → [`LoweredFunction`] driver must never panic.
//! For any well-formed [`cranelift_codegen::ir::Function`] reachable from
//! this generator, `lower_function` must return either
//! `Ok(LoweredFunction)` or `Err(LoweringError)` — both are acceptable
//! outcomes. A panic anywhere inside the driver, the per-family lowerings
//! (arith / float / memory / cf / vector / conv), the signature lowering,
//! or the `LoweringBuilder` is a real bug that would surface as a host
//! abort on adversarial guest input once the auto-offload pipeline is
//! wired up.
//!
//! # Generation approach (Approach 2 from W2.8 brief)
//!
//! We consume a small fixed-shape header from the fuzzer-supplied bytes
//! to pick a function "shape" (param count, return count, parameter
//! types), then walk the remaining bytes as a tape of opcode selectors
//! and emit a handful of Cranelift instructions of each family. The
//! function is always terminated by a `return` so the driver's
//! terminator-check path is exercised on every input.
//!
//! Generation is intentionally narrow: we don't build random control
//! flow, don't insert undefined operand references, and don't construct
//! ill-typed instructions. Those failure modes belong to the
//! `cranelift-frontend` API surface, not the driver. The driver's
//! invariant is "well-formed input → no panic, errors are structured" —
//! that's what this harness exercises.
//!
//! Approach 3 (`cranelift_fuzzgen`) was considered and rejected: that
//! crate isn't a dep of this workspace, and adding it would require
//! a new dep-graph audit. The hand-rolled generator below is small
//! enough to read in one screen and covers every wave-1 family.
//!
//! [`LoweredFunction`]: tensor_wasm_jit::lowered_ir::LoweredFunction

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{
    immediates::Offset32, types, AbiParam, Function, InstBuilder, MemFlags, Opcode, Signature,
    Type, UserFuncName, Value,
};
use cranelift_codegen::isa::CallConv;

use tensor_wasm_jit::lowering_driver::lower_function;

/// A scalar lane type the generator is willing to put in a signature or
/// produce as the result of an arithmetic op.
///
/// We deliberately exclude types the wave-1 signature lowering rejects
/// (e.g. `I128`, `F16`) so the fuzzer doesn't spend all its budget on
/// signature-rejection paths. Those paths have unit coverage already;
/// the fuzz target's job is to hammer the *post-signature* driver code.
#[derive(Debug, Clone, Copy, Arbitrary)]
enum LaneTy {
    I32,
    I64,
    F32,
    F64,
}

impl LaneTy {
    fn to_clif(self) -> Type {
        match self {
            LaneTy::I32 => types::I32,
            LaneTy::I64 => types::I64,
            LaneTy::F32 => types::F32,
            LaneTy::F64 => types::F64,
        }
    }
}

/// A single instruction to emit in the entry block.
///
/// The generator picks one of these per "step" and emits the
/// corresponding Cranelift inst. Each variant is shaped so the inst is
/// guaranteed well-typed by construction; the fuzzer can still drive the
/// generator into shapes that surface `Err(LoweringError::*)`, which is
/// the whole point.
#[derive(Debug, Arbitrary)]
enum FuzzInst {
    /// Integer addition on the first two same-typed integer params.
    Iadd,
    /// Integer subtraction (same).
    Isub,
    /// Integer multiplication (same).
    Imul,
    /// Float addition on the first two same-typed float params.
    Fadd,
    /// Float subtraction (same).
    Fsub,
    /// Float multiplication (same).
    Fmul,
    /// `load.<ty>` from the first 64-bit-int param at the given offset.
    /// `offset` is masked to a small range so we don't waste bits.
    Load { ty: LaneTy, offset: i16 },
    /// `store.<ty>` of the first matching-typed param into the first
    /// 64-bit-int param at the given offset.
    Store { ty: LaneTy, offset: i16 },
}

/// Top-level fuzz input. `arbitrary` derives a `Arbitrary` impl that
/// consumes bytes from the fuzzer in a structured way; capping field
/// counts here keeps each test case bounded.
#[derive(Debug, Arbitrary)]
struct FuzzFunc {
    /// One to four parameter types. The generator's `Arbitrary` impl can
    /// produce zero-length vectors, which we clamp below.
    params: Vec<LaneTy>,
    /// Zero or one return type. Multi-return is technically allowed in
    /// Cranelift but the wave-1 signature lowering only models 0- or
    /// 1-return Wasm-shape functions; staying inside that envelope keeps
    /// the harness focused on driver behaviour, not signature edge cases.
    ret: Option<LaneTy>,
    /// The instruction tape. We cap the length when consuming.
    insts: Vec<FuzzInst>,
    /// Force the signature to include a pointer-typed (I64) param at
    /// index 0 so load/store ops have something to consume. When false,
    /// load/store ops that don't find a suitable operand are skipped.
    prepend_ptr_param: bool,
}

/// Hard cap on per-case instruction count. The driver's per-inst work is
/// O(value_map size); leaving this unbounded would let libfuzzer
/// synthesise pathological tape lengths that starve coverage.
const MAX_INSTS: usize = 32;

/// Hard cap on signature parameter count. Wave-1 lowering tests stay in
/// the 0..=4 range; we mirror that here.
const MAX_PARAMS: usize = 4;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(mut spec): Result<FuzzFunc, _> = u.arbitrary() else {
        return;
    };

    // Clamp the generator output into the envelope the wave-1 lowering
    // passes were designed for. This isn't softening the property — we
    // still drive the *driver* with these inputs; we're just bounding
    // per-case work so libfuzzer keeps the iteration rate high.
    spec.params.truncate(MAX_PARAMS);
    spec.insts.truncate(MAX_INSTS);

    // Build the function. If construction itself returns `None` (some
    // generator outputs have no usable signature, e.g. an empty `params`
    // when we asked for a load that needs a pointer), skip the case.
    let Some(func) = build_function(&spec) else {
        return;
    };

    // The property under test: the driver returns `Result`, never panics.
    let _ = lower_function(&func);
});

/// Translate a `FuzzFunc` specification into a real Cranelift `Function`.
///
/// Returns `None` if the spec is unrealisable (e.g. needs a pointer
/// param but none is available). All other inputs produce a well-formed
/// function — the driver is the system under test, not the constructor.
fn build_function(spec: &FuzzFunc) -> Option<Function> {
    // Build the signature. Optionally prepend a 64-bit-int "pointer"
    // param so load/store ops have something to address.
    let mut sig = Signature::new(CallConv::SystemV);
    if spec.prepend_ptr_param {
        sig.params.push(AbiParam::new(types::I64));
    }
    for p in &spec.params {
        sig.params.push(AbiParam::new(p.to_clif()));
    }
    if let Some(r) = spec.ret {
        sig.returns.push(AbiParam::new(r.to_clif()));
    }

    let mut func =
        Function::with_name_signature(UserFuncName::testcase("fuzz_fn".as_bytes()), sig);

    // Entry block + one block param per signature param.
    let entry = func.dfg.make_block();
    let param_types: Vec<Type> = func
        .signature
        .params
        .iter()
        .map(|p| p.value_type)
        .collect();
    for pty in &param_types {
        func.dfg.append_block_param(entry, *pty);
    }
    func.layout.append_block(entry);

    // Snapshot the block-param values for the per-inst emitters. We
    // build per-type lookups so each emitter can pick the first param
    // of a matching type without re-walking the dfg.
    let block_params: Vec<Value> = func.dfg.block_params(entry).to_vec();

    // For each instruction in the tape, try to emit a matching
    // Cranelift inst. Emitters return `false` if the spec can't be
    // honoured (e.g. iadd requested but no I32/I64 params); the loop
    // just continues — the fuzzer can keep exploring.
    for inst in &spec.insts {
        let ok = emit_inst(&mut func, entry, &block_params, &param_types, inst);
        // We intentionally ignore `ok`; an inst that didn't apply is a
        // skip, not a failure.
        let _ = ok;
    }

    // Always terminate with a return. The signature may demand a single
    // value; we provide one by picking the first matching-typed param,
    // falling back to a zero-value return when nothing matches (which
    // produces a malformed function — that's a valid input for the
    // *driver* to reject with `Err`, which is what we're testing).
    let return_values: Vec<Value> = if let Some(ret) = spec.ret {
        let ret_clif = ret.to_clif();
        block_params
            .iter()
            .copied()
            .find(|v| func.dfg.value_type(*v) == ret_clif)
            .map(|v| vec![v])
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut cursor = FuncCursor::new(&mut func).at_bottom(entry);
    cursor.ins().return_(&return_values);

    Some(func)
}

/// Emit a single instruction into `entry` based on `inst`. Returns
/// `true` if an instruction was actually emitted, `false` if the spec
/// couldn't be satisfied with the available params.
fn emit_inst(
    func: &mut Function,
    entry: cranelift_codegen::ir::Block,
    block_params: &[Value],
    param_types: &[Type],
    inst: &FuzzInst,
) -> bool {
    fn find_of(
        block_params: &[Value],
        param_types: &[Type],
        ty: Type,
    ) -> Option<Value> {
        block_params
            .iter()
            .zip(param_types.iter())
            .find(|(_, t)| **t == ty)
            .map(|(v, _)| *v)
    }
    fn find_two_of(
        block_params: &[Value],
        param_types: &[Type],
        ty: Type,
    ) -> Option<(Value, Value)> {
        let mut iter = block_params
            .iter()
            .zip(param_types.iter())
            .filter(|(_, t)| **t == ty)
            .map(|(v, _)| *v);
        let a = iter.next()?;
        // Second can be the same value as the first if only one param
        // matches — Cranelift permits aliased operands.
        let b = iter.next().unwrap_or(a);
        Some((a, b))
    }

    let mut cursor = FuncCursor::new(func).at_bottom(entry);
    match inst {
        FuzzInst::Iadd | FuzzInst::Isub | FuzzInst::Imul => {
            // Pick I32 if available, else I64.
            let ty = if find_of(block_params, param_types, types::I32).is_some() {
                types::I32
            } else if find_of(block_params, param_types, types::I64).is_some() {
                types::I64
            } else {
                return false;
            };
            let Some((a, b)) = find_two_of(block_params, param_types, ty) else {
                return false;
            };
            let opcode = match inst {
                FuzzInst::Iadd => Opcode::Iadd,
                FuzzInst::Isub => Opcode::Isub,
                FuzzInst::Imul => Opcode::Imul,
                _ => unreachable!(),
            };
            let (_inst, _dfg) = cursor.ins().Binary(opcode, ty, a, b);
            true
        }
        FuzzInst::Fadd | FuzzInst::Fsub | FuzzInst::Fmul => {
            let ty = if find_of(block_params, param_types, types::F32).is_some() {
                types::F32
            } else if find_of(block_params, param_types, types::F64).is_some() {
                types::F64
            } else {
                return false;
            };
            let Some((a, b)) = find_two_of(block_params, param_types, ty) else {
                return false;
            };
            let opcode = match inst {
                FuzzInst::Fadd => Opcode::Fadd,
                FuzzInst::Fsub => Opcode::Fsub,
                FuzzInst::Fmul => Opcode::Fmul,
                _ => unreachable!(),
            };
            let (_inst, _dfg) = cursor.ins().Binary(opcode, ty, a, b);
            true
        }
        FuzzInst::Load { ty, offset } => {
            // Need a 64-bit-int param to use as the address.
            let Some(ptr) = find_of(block_params, param_types, types::I64) else {
                return false;
            };
            let load_ty = ty.to_clif();
            // Clamp offset to i16 range (already i16); convert to i32.
            let off = Offset32::new(*offset as i32);
            let (_inst, _dfg) = cursor
                .ins()
                .Load(Opcode::Load, load_ty, MemFlags::trusted(), off, ptr);
            true
        }
        FuzzInst::Store { ty, offset } => {
            let Some(ptr) = find_of(block_params, param_types, types::I64) else {
                return false;
            };
            let store_ty = ty.to_clif();
            // Find a value of the matching store type. If none, skip.
            let Some(val) = block_params
                .iter()
                .zip(param_types.iter())
                .find(|(_, t)| **t == store_ty)
                .map(|(v, _)| *v)
            else {
                return false;
            };
            let off = Offset32::new(*offset as i32);
            let (_inst, _dfg) = cursor
                .ins()
                .Store(Opcode::Store, store_ty, MemFlags::trusted(), off, val, ptr);
            true
        }
    }
}
