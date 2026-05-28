// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W2.7 — end-to-end integration tests for the Cranelift → `LoweredFunction`
//! pipeline.
//!
//! These tests drive realistic [`cranelift_codegen::ir::Function`] shapes
//! through the full driver
//! ([`tensor_wasm_jit::lowering_driver::lower_function`]) — signature
//! lowering, per-family dispatch (L1-L6), block assembly, and entry-id
//! finalisation — and assert on the resulting [`LoweredFunction`].
//!
//! Scope vs the driver's `#[cfg(test)]` unit tests in
//! `src/lowering_driver.rs`: those tests poke individual code paths in
//! isolation (one opcode per family, one shape at a time). The cases here
//! cover *composite* shapes a real Cranelift function carries — a vector_add
//! kernel body that touches load + load + fadd + store + return in one
//! block; an fma that exercises the float family's ternary form; a
//! multi-block diamond with block-param passing on a `brif`; and a
//! reject-list candidate that should fail the lowering at the first
//! unsupported opcode. Together they form the "smoke test" the W2 wave
//! depends on for downstream consumers (the blueprint adapter and the
//! PTX emitter).
//!
//! All tests are gated on the `cuda-oxide-backend` feature because every
//! wave-2 lowering module is gated the same way.

#![cfg(feature = "cuda-oxide-backend")]

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::{
    types, AbiParam, BlockCall, Function, InstBuilder, InstructionData, MemFlags, Opcode,
    Signature, UserFuncName,
};
use cranelift_codegen::isa::CallConv;

use tensor_wasm_jit::lowered_ir::{LoweredOp, LoweredType};
use tensor_wasm_jit::lowering_driver::lower_function;
use tensor_wasm_jit::lowering_errors::LoweringError;
use tensor_wasm_jit::lowering_test_support::function_with_binary_op;

/// Type used for pointer-shaped block params in the fixtures.
///
/// Mirrors the convention in `lowering_test_support` (`PTR_TY = I64`): the
/// memory-family lowering accepts I64 as a base-pointer width, and the
/// signature lowering rounds-trips it as `LoweredType::I64` (the wave-1
/// `LoweredType::Ptr` variant is reserved for Cranelift's reference types
/// `R32`/`R64`, which Wasm-derived IR doesn't produce).
const PTR_TY: cranelift_codegen::ir::Type = types::I64;

// ---- 1. vector_add kernel shape -----------------------------------------

/// Drive a `vector_add`-shaped kernel through the full lowering pipeline.
///
/// Cranelift IR shape (one block):
///
/// ```text
/// fn(out: i64, a: i64, b: i64) -> () {
///   v_a = load.f32 a + 0
///   v_b = load.f32 b + 0
///   v_sum = fadd v_a, v_b
///   store.f32 v_sum, out + 0
///   return
/// }
/// ```
///
/// This is the smallest function shape that exercises both the memory
/// family (Load + Store) and the float family (AddF) in the same block,
/// and confirms the driver wires their `LoweredValueId`s together via the
/// shared value map.
#[test]
fn vector_add_kernel_lowers_to_load_load_addf_store_return() {
    let mut sig = Signature::new(CallConv::SystemV);
    // (out, a, b) — three pointer-shaped params, no returns.
    sig.params.push(AbiParam::new(PTR_TY));
    sig.params.push(AbiParam::new(PTR_TY));
    sig.params.push(AbiParam::new(PTR_TY));
    let mut func =
        Function::with_name_signature(UserFuncName::testcase("vector_add".as_bytes()), sig);

    let block = func.dfg.make_block();
    let out_ptr = func.dfg.append_block_param(block, PTR_TY);
    let a_ptr = func.dfg.append_block_param(block, PTR_TY);
    let b_ptr = func.dfg.append_block_param(block, PTR_TY);
    func.layout.append_block(block);

    let mut cursor = FuncCursor::new(&mut func).at_bottom(block);
    // `Load` lowers via `InstructionData::Load { opcode: Opcode::Load, .. }`.
    let (load_a_inst, dfg) = cursor.ins().Load(
        Opcode::Load,
        types::F32,
        MemFlags::trusted(),
        Offset32::new(0),
        a_ptr,
    );
    let v_a = dfg.first_result(load_a_inst);
    let (load_b_inst, dfg) = cursor.ins().Load(
        Opcode::Load,
        types::F32,
        MemFlags::trusted(),
        Offset32::new(0),
        b_ptr,
    );
    let v_b = dfg.first_result(load_b_inst);
    // `fadd` is `Opcode::Fadd` with the `Binary` form.
    let (fadd_inst, dfg) = cursor.ins().Binary(Opcode::Fadd, types::F32, v_a, v_b);
    let v_sum = dfg.first_result(fadd_inst);
    // `Store` writes `value` at `(base + offset)`.
    let _ = cursor.ins().Store(
        Opcode::Store,
        types::F32,
        MemFlags::trusted(),
        Offset32::new(0),
        v_sum,
        out_ptr,
    );
    cursor.ins().return_(&[]);

    let lowered = lower_function(&func).expect("vector_add must lower");

    // Signature: 3 I64 params, 0 returns.
    assert_eq!(
        lowered.signature.params,
        vec![LoweredType::I64, LoweredType::I64, LoweredType::I64],
        "expected 3 pointer-shaped params lowered as I64",
    );
    assert!(
        lowered.signature.returns.is_empty(),
        "vector_add returns void",
    );

    // One block, well-formed.
    assert_eq!(lowered.blocks.len(), 1, "single-block function");
    assert!(lowered.is_well_formed(), "function must be well-formed");
    let lblock = &lowered.blocks[0];
    assert!(lblock.is_well_formed(), "block must end in a terminator");
    assert_eq!(lblock.id, lowered.entry, "entry id matches the only block");

    // Block params lowered 1:1 with the signature shape.
    assert_eq!(lblock.params.len(), 3, "three block params");
    for (_, ty) in &lblock.params {
        assert_eq!(*ty, LoweredType::I64);
    }

    // The driver may insert intermediate ops between the headline ones
    // (e.g. a StackAlloc for stack-slot loads); for plain pointer
    // load/store it does not. We assert the *ordered subsequence*
    // `[Load, Load, AddF, Store, Return]` rather than strict equality so
    // future driver refinements that add diagnostic no-ops don't break
    // the test, but we also check the total count matches what we
    // currently emit (5) to catch silent extra-op regressions.
    assert_eq!(
        lblock.ops.len(),
        5,
        "expected exactly 5 ops; got {:?}",
        lblock.ops,
    );

    match &lblock.ops[0] {
        LoweredOp::Load { ty, .. } => assert_eq!(*ty, LoweredType::F32),
        other => panic!("op[0] expected Load, got {other:?}"),
    }
    match &lblock.ops[1] {
        LoweredOp::Load { ty, .. } => assert_eq!(*ty, LoweredType::F32),
        other => panic!("op[1] expected Load, got {other:?}"),
    }
    match &lblock.ops[2] {
        LoweredOp::AddF { ty, lhs, rhs, .. } => {
            assert_eq!(*ty, LoweredType::F32);
            // The fadd operands must be the result ids of the two loads
            // — confirms the driver wired the value map across families.
            let load0_result = lblock.ops[0].result().expect("load has a result");
            let load1_result = lblock.ops[1].result().expect("load has a result");
            assert_eq!(*lhs, load0_result, "fadd lhs == load0 result");
            assert_eq!(*rhs, load1_result, "fadd rhs == load1 result");
        }
        other => panic!("op[2] expected AddF, got {other:?}"),
    }
    match &lblock.ops[3] {
        LoweredOp::Store { ty, value, .. } => {
            assert_eq!(*ty, LoweredType::F32);
            let fadd_result = lblock.ops[2].result().expect("fadd has a result");
            assert_eq!(*value, fadd_result, "store value == fadd result");
        }
        other => panic!("op[3] expected Store, got {other:?}"),
    }
    match &lblock.ops[4] {
        LoweredOp::Return { values } => {
            assert!(values.is_empty(), "void return carries no values");
        }
        other => panic!("op[4] expected Return, got {other:?}"),
    }
}

// ---- 2. Scalar arith — driver-as-public-API check ------------------------

/// Drive the `fn(i32, i32) -> i32 { iadd; return }` shape through
/// `lower_function`. The unit tests in `lowering_driver.rs` already cover
/// this; the integration-test version confirms the function is reachable
/// as a *public API* from outside the crate (no private-only path).
#[test]
fn scalar_iadd_lowers_via_public_driver_api() {
    let (func, _inst) = function_with_binary_op(Opcode::Iadd, types::I32);
    let lowered = lower_function(&func).expect("scalar iadd must lower");

    assert_eq!(lowered.blocks.len(), 1);
    assert_eq!(
        lowered.signature.params,
        vec![LoweredType::I32, LoweredType::I32],
    );
    assert_eq!(lowered.signature.returns, vec![LoweredType::I32]);
    let block = &lowered.blocks[0];
    assert_eq!(block.ops.len(), 2, "AddI + Return");
    assert!(matches!(block.ops[0], LoweredOp::AddI { .. }));
    assert!(matches!(block.ops[1], LoweredOp::Return { .. }));
    assert!(lowered.is_well_formed());
}

// ---- 3. fma shape --------------------------------------------------------

/// Drive a `fn(f32, f32, f32) -> f32 { fma; return }` shape through the
/// driver. Exercises the float family's ternary form (`Opcode::Fma`,
/// `InstructionData::Ternary`) and confirms the resulting `LoweredOp::Fma`
/// carries the three operand ids and a result id distinct from the inputs.
#[test]
fn scalar_fma_lowers_to_fma_and_return() {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::F32));
    sig.params.push(AbiParam::new(types::F32));
    sig.params.push(AbiParam::new(types::F32));
    sig.returns.push(AbiParam::new(types::F32));
    let mut func =
        Function::with_name_signature(UserFuncName::testcase("fma_kernel".as_bytes()), sig);
    let block = func.dfg.make_block();
    let pa = func.dfg.append_block_param(block, types::F32);
    let pb = func.dfg.append_block_param(block, types::F32);
    let pc = func.dfg.append_block_param(block, types::F32);
    func.layout.append_block(block);

    // Build the fma instruction directly via `make_inst` — the
    // `InstBuilder::Ternary` shim takes the same shape but going through
    // `InstructionData::Ternary` mirrors the form `lower_float`'s own
    // unit tests use, which is the form the lowering matches against.
    let fma_inst = func.dfg.make_inst(InstructionData::Ternary {
        opcode: Opcode::Fma,
        args: [pa, pb, pc],
    });
    func.dfg.make_inst_results(fma_inst, types::F32);
    func.layout.append_inst(fma_inst, block);
    let v_result = func.dfg.first_result(fma_inst);

    {
        let mut cursor = FuncCursor::new(&mut func).at_bottom(block);
        cursor.ins().return_(&[v_result]);
    }

    let lowered = lower_function(&func).expect("fma must lower");

    assert_eq!(
        lowered.signature.params,
        vec![LoweredType::F32, LoweredType::F32, LoweredType::F32],
    );
    assert_eq!(lowered.signature.returns, vec![LoweredType::F32]);
    assert_eq!(lowered.blocks.len(), 1);
    let block = &lowered.blocks[0];
    assert_eq!(block.ops.len(), 2, "Fma + Return");
    match &block.ops[0] {
        LoweredOp::Fma {
            ty,
            a,
            b,
            c,
            result,
        } => {
            assert_eq!(*ty, LoweredType::F32);
            // The three operand ids must be pairwise distinct and the
            // result must be a fresh id not equal to any of them.
            assert_ne!(a, b);
            assert_ne!(b, c);
            assert_ne!(a, c);
            assert_ne!(result, a);
            assert_ne!(result, b);
            assert_ne!(result, c);
        }
        other => panic!("expected LoweredOp::Fma, got {other:?}"),
    }
    match &block.ops[1] {
        LoweredOp::Return { values } => {
            // Return value must be the fma result.
            let fma_result = block.ops[0].result().expect("fma has a result");
            assert_eq!(values, &vec![fma_result]);
        }
        other => panic!("expected Return, got {other:?}"),
    }
    assert!(lowered.is_well_formed());
}

// ---- 4. Multi-block: brif diamond ----------------------------------------

/// Drive a three-block `brif` diamond through the driver. Shape:
///
/// ```text
/// fn(cond: i32, payload: i32) -> i32 {
///   block0(cond, payload):
///     brif cond, block1(payload), block2(payload)
///   block1(v: i32):
///     return v
///   block2(v: i32):
///     return v
/// }
/// ```
///
/// Asserts that the driver:
///
/// 1. Allocates three blocks with well-formed terminators.
/// 2. Lowers `brif` to `LoweredOp::CondBr` with the correct
///    `then_target`/`else_target` ids.
/// 3. Maps the entry block's param into both `then_args` and `else_args`.
/// 4. Surfaces each successor's `Return { values: [v] }` carrying that
///    block's param.
#[test]
fn multi_block_brif_diamond_lowers_three_blocks() {
    let mut sig = Signature::new(CallConv::SystemV);
    // (cond, payload) — cond is the brif boolean (i32 in Cranelift; the
    // bool type is also legal, but i32 is the form `lower_cf`'s own tests
    // exercise).
    sig.params.push(AbiParam::new(types::I32));
    sig.params.push(AbiParam::new(types::I32));
    sig.returns.push(AbiParam::new(types::I32));
    let mut func =
        Function::with_name_signature(UserFuncName::testcase("brif_diamond".as_bytes()), sig);

    let entry = func.dfg.make_block();
    let cond = func.dfg.append_block_param(entry, types::I32);
    let payload = func.dfg.append_block_param(entry, types::I32);

    let then_blk = func.dfg.make_block();
    let then_param = func.dfg.append_block_param(then_blk, types::I32);

    let else_blk = func.dfg.make_block();
    let else_param = func.dfg.append_block_param(else_blk, types::I32);

    func.layout.append_block(entry);
    func.layout.append_block(then_blk);
    func.layout.append_block(else_blk);

    // Build the brif terminator on the entry block. Both successors
    // receive `payload` as their single block arg.
    let then_call = BlockCall::new(then_blk, &[payload], &mut func.dfg.value_lists);
    let else_call = BlockCall::new(else_blk, &[payload], &mut func.dfg.value_lists);
    let brif = func.dfg.make_inst(InstructionData::Brif {
        opcode: Opcode::Brif,
        arg: cond,
        blocks: [then_call, else_call],
    });
    func.dfg.make_inst_results(brif, types::INVALID);
    func.layout.append_inst(brif, entry);

    // Then-block: return its param.
    {
        let mut cursor = FuncCursor::new(&mut func).at_bottom(then_blk);
        cursor.ins().return_(&[then_param]);
    }
    // Else-block: return its param.
    {
        let mut cursor = FuncCursor::new(&mut func).at_bottom(else_blk);
        cursor.ins().return_(&[else_param]);
    }

    let lowered = lower_function(&func).expect("multi-block diamond must lower");

    assert_eq!(lowered.blocks.len(), 3, "three blocks expected");
    assert!(lowered.is_well_formed());
    // Every block must end in a terminator.
    for lblock in &lowered.blocks {
        assert!(
            lblock.is_well_formed(),
            "block {} not well-formed: ops = {:?}",
            lblock.id,
            lblock.ops,
        );
    }
    // Entry id matches the first block's id (layout-order pre-allocation).
    let entry_lblock = &lowered.blocks[0];
    assert_eq!(lowered.entry, entry_lblock.id);

    // Entry block: single op `CondBr` pointing at the two successors.
    assert_eq!(entry_lblock.ops.len(), 1, "entry has only the CondBr");
    let (then_id, else_id) = (lowered.blocks[1].id, lowered.blocks[2].id);
    match &entry_lblock.ops[0] {
        LoweredOp::CondBr {
            cond: _,
            then_target,
            then_args,
            else_target,
            else_args,
        } => {
            assert_eq!(*then_target, then_id);
            assert_eq!(*else_target, else_id);
            assert_eq!(then_args.len(), 1, "then receives one arg (payload)");
            assert_eq!(else_args.len(), 1, "else receives one arg (payload)");
            assert_eq!(
                then_args[0], else_args[0],
                "both successors receive the same payload id",
            );
        }
        other => panic!("expected CondBr, got {other:?}"),
    }

    // Each successor returns its own (single) param.
    for succ in &lowered.blocks[1..] {
        assert_eq!(succ.params.len(), 1, "successor has one param");
        assert_eq!(succ.params[0].1, LoweredType::I32);
        assert_eq!(succ.ops.len(), 1, "successor has only the Return");
        match &succ.ops[0] {
            LoweredOp::Return { values } => {
                assert_eq!(values.len(), 1);
                assert_eq!(
                    values[0], succ.params[0].0,
                    "successor returns its block param",
                );
            }
            other => panic!("expected Return, got {other:?}"),
        }
    }
}

// ---- 5. Rejection / unsupported opcode -----------------------------------

/// An `atomic_load` instruction has no wave-1 lowering family and the
/// driver does not yet call the W2.1 reject-list pre-pass (see the
/// "Reject-list integration" block in `lowering_driver.rs`). The
/// instruction therefore surfaces through the "no family matched" path as
/// [`LoweringError::UnsupportedOpcode`] carrying the opcode mnemonic.
///
/// When the reject-list pre-pass is wired in (a follow-up to W2.4), this
/// test will need to switch to expecting `LoweringError::Rejected` — the
/// error variant change is the load-bearing signal of that wiring landing.
/// Until then, the `UnsupportedOpcode` outcome is the correct contract.
#[test]
fn atomic_load_surfaces_as_unsupported_opcode() {
    let mut sig = Signature::new(CallConv::SystemV);
    // The function takes a single i64 pointer-like param to feed the
    // atomic_load's address operand and returns an i32 (the loaded value
    // — we never actually return it; we return zero values to keep the
    // block well-formed without depending on iconst, which is itself
    // unsupported).
    sig.params.push(AbiParam::new(PTR_TY));
    let mut func = Function::with_name_signature(UserFuncName::testcase("atomic".as_bytes()), sig);
    let block = func.dfg.make_block();
    let ptr = func.dfg.append_block_param(block, PTR_TY);
    func.layout.append_block(block);

    // `atomic_load.i32 ptr` — the `LoadNoOffset` instruction-data form
    // the reject-list's own unit tests use.
    let atomic = func.dfg.make_inst(InstructionData::LoadNoOffset {
        opcode: Opcode::AtomicLoad,
        flags: MemFlags::new(),
        arg: ptr,
    });
    func.dfg.make_inst_results(atomic, types::I32);
    func.layout.append_inst(atomic, block);

    // Terminator so the block is well-formed before lowering hits the
    // atomic_load. We can't return the atomic_load's result (the driver
    // would have walked past it already by then) — instead we return
    // zero values and rely on the fact that lowering fails *before*
    // checking the return's value list against the signature.
    {
        let mut cursor = FuncCursor::new(&mut func).at_bottom(block);
        cursor.ins().return_(&[]);
    }

    let err = lower_function(&func).expect_err("atomic_load is unsupported");
    match err {
        LoweringError::UnsupportedOpcode { op, .. } => {
            assert!(
                op.contains("AtomicLoad"),
                "expected AtomicLoad in the error op string, got {op:?}",
            );
        }
        // If a future change wires the reject-list pre-pass into the
        // driver, this becomes the expected variant; until then it's a
        // signal that the driver's family dispatch order changed in a
        // way the test should be updated for.
        LoweringError::Rejected { reason, .. } => {
            assert!(
                reason.contains("atomic"),
                "expected 'atomic' in the rejection reason, got {reason:?}",
            );
        }
        other => panic!("expected UnsupportedOpcode or Rejected for atomic_load, got {other:?}",),
    }
}
