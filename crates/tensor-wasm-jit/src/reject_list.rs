// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Reject-list detector for the Pliron `dialect-mir` lowering pipeline.
//!
//! [`scan_function`] walks a [`cranelift_codegen::ir::Function`] and returns
//! every instruction whose opcode the wave-2 lowering cannot translate to
//! the [`crate::lowered_ir::LoweredOp`] interim IR. The categories mirror
//! the "Unsupported in v0.4" section of
//! [`crate::pliron_dialect`](crate::pliron_dialect) one-for-one:
//!
//! - **Atomics** ([`RejectReason::Atomic`]): `atomic_load`, `atomic_store`,
//!   `atomic_rmw`, `atomic_cas`. Deferred — Wasm threads + GPU atomics is a
//!   memory-model alignment problem larger than the wave-2 scope.
//! - **Strict-FP exception bits** ([`RejectReason::StrictFp`]):
//!   `fcvt_to_sint_sat`, `fcvt_to_uint_sat`. PTX default rounding diverges
//!   from Wasm-strict FP — these opcodes carry trap semantics the wave-2
//!   lowering cannot honour.
//! - **Host calls** ([`RejectReason::HostCall`]): `call`, `call_indirect`,
//!   `return_call`, `return_call_indirect`. PTX has no host-callback path;
//!   wave 3+ will distinguish device-side `func.call`s from host
//!   round-trips, but for wave 2 every call is rejected.
//!
//! # Categories documented but absent from this Cranelift version
//!
//! The canonical reject list in [`crate::pliron_dialect`] also names:
//!
//! - **Wasm table ops** (`table.get` / `table.set`)
//! - **Wasm GC / `ref.func`**
//! - **`memory.grow` / `memory.size`**
//! - **`memory.copy` / `memory.fill`**
//!
//! These are **not direct Cranelift opcodes in the pinned `0.111.9`
//! workspace dependency**: they are intercepted by the Wasm-to-Cranelift
//! translator (`cranelift-wasm` and Wasmtime), which lowers them into
//! sequences of `load`/`store`/`call`-to-libcall before they reach the
//! `ir::Function` this detector sees. The `call` reject above is therefore
//! load-bearing: any host-routed lowering surfaces here as a `Call` and
//! gets rejected on that ground. The dedicated [`RejectReason::TableOp`],
//! [`RejectReason::GcOp`], [`RejectReason::MemoryResize`] and
//! [`RejectReason::LargeMemcpy`] variants are pre-declared so the public
//! API is stable when a future Cranelift bump (or a direct Wasm front-end)
//! exposes those opcodes natively.
//!
//! # Why a separate pass
//!
//! The detector runs *before* the per-family `lower_*` modules. Returning
//! an early `Vec<Rejection>` (rather than letting the lowerings fail one
//! by one) is the contract that lets the auto-offload pipeline fall back
//! to the blueprint detector cleanly: an admissible function gets the
//! Pliron pipeline, an inadmissible one gets the legacy blueprint path —
//! never a half-lowered, half-rejected mess.

#![cfg(feature = "cuda-oxide-backend")]

use cranelift_codegen::ir::{Block, Function, Inst, Opcode};

/// A reason a Cranelift instruction cannot be lowered to a
/// [`crate::lowered_ir::LoweredOp`].
///
/// Each variant carries the offending opcode's mnemonic (`&'static str`)
/// so error messages and logs are grep-able against the [mapping
/// table](crate::pliron_dialect#mapping-table). The mnemonic is the same
/// string Cranelift's own [`Opcode`] `Display` impl prints (e.g.
/// `"atomic_rmw"`, not `"AtomicRmw"`), making it stable across the
/// `cranelift-codegen` version bumps that periodically rename enum
/// variants but keep the textual mnemonic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Atomic memory operation. Deferred — Wasm-threads-on-GPU is a
    /// memory-model problem out of scope for wave 2. Covers `atomic_load`,
    /// `atomic_store`, `atomic_rmw`, `atomic_cas`.
    Atomic(&'static str),

    /// Floating-point opcode with strict-FP exception semantics PTX
    /// default rounding cannot match. Covers `fcvt_to_sint_sat` and
    /// `fcvt_to_uint_sat` — both saturate-on-out-of-range, behaviour the
    /// PTX `cvt` does not provide by default. The full strict-FP scope
    /// (NaN propagation, denormals-as-zero) is broader; wave 2 limits
    /// itself to the trap-carrying conversions.
    StrictFp(&'static str),

    /// Wasm `table.get` / `table.set`. Tables live host-side; lowering
    /// them would require device-resident table mirrors. Hard-rejected.
    ///
    /// **Not currently wired in the detector**: `cranelift-codegen` 0.111
    /// has no direct `Opcode::TableGet` / `Opcode::TableSet` variant —
    /// `cranelift-wasm` lowers these to `load`/`store`+libcall sequences
    /// in the translator, so a Wasm module reaches us with the table
    /// access already expanded into a `call` to a runtime helper (and
    /// therefore is caught by [`RejectReason::HostCall`]). Pre-declared
    /// here so a future direct Wasm front-end or a Cranelift bump that
    /// introduces table opcodes natively does not have to reshuffle the
    /// public enum.
    TableOp(&'static str),

    /// Wasm GC / reference-type opcode (`ref.func`, `ref.null`,
    /// `ref.is_null` on a non-i31 ref). No device-side representation
    /// possible. Hard-rejected.
    ///
    /// **Not currently wired** for the same reason as
    /// [`RejectReason::TableOp`]: in `cranelift-codegen` 0.111 the GC
    /// proposal opcodes either do not exist as direct enum variants
    /// (`ref.func`) or are translated into other ops by `cranelift-wasm`
    /// before the detector sees them. Pre-declared for forward
    /// compatibility.
    GcOp(&'static str),

    /// `memory.grow` / `memory.size`. Linear-memory resizing requires a
    /// host round-trip; PTX kernels must run with a fixed memory
    /// snapshot.
    ///
    /// **Not currently wired**: `cranelift-codegen` 0.111 has no direct
    /// `Opcode::MemoryGrow` / `Opcode::MemorySize` — Wasm `memory.grow`
    /// reaches us as a libcall (a `Call` instruction targeting the
    /// runtime helper) and is therefore caught by
    /// [`RejectReason::HostCall`]. Pre-declared for the same forward-
    /// compatibility reason as the other absent categories.
    MemoryResize(&'static str),

    /// `memory.copy` / `memory.fill` above the inline-copy threshold.
    /// PTX has `cp.async.bulk` (sm_90+) but the wave-2 baseline is
    /// sm_80 — the cuda-oxide PTX target version pinned via
    /// [`crate::ptx_emit::DEFAULT_TARGET`](crate::ptx_emit::DEFAULT_TARGET).
    ///
    /// **Not currently wired**: same translator-expansion situation as
    /// the other absent categories — `memory.copy` / `memory.fill` reach
    /// us as `Call` instructions to the runtime, caught by
    /// [`RejectReason::HostCall`]. Pre-declared for forward compatibility.
    /// Wave 3+ will refine [`RejectReason::HostCall`] to distinguish
    /// device-internal calls from runtime-helper calls and may at that
    /// point route small `memory.copy` libcalls back to an inline
    /// `Load`/`Store` sequence instead of a hard reject.
    LargeMemcpy(&'static str),

    /// `call` / `call_indirect` / `return_call` / `return_call_indirect`.
    /// Host-callback prohibition: PTX has no path back into the Wasmtime
    /// runtime, so every call is rejected at the wave-2 detector. Wave
    /// 3+ will distinguish device-internal calls (legal: lowered to
    /// `func.call`) from host round-trips (illegal: rejected).
    HostCall(&'static str),
}

impl RejectReason {
    /// The Cranelift opcode mnemonic that triggered this rejection.
    ///
    /// Convenience accessor for log lines / error formatting that want
    /// the offending opcode name without `match`-ing on the variant.
    pub fn opcode_mnemonic(&self) -> &'static str {
        match self {
            Self::Atomic(m)
            | Self::StrictFp(m)
            | Self::TableOp(m)
            | Self::GcOp(m)
            | Self::MemoryResize(m)
            | Self::LargeMemcpy(m)
            | Self::HostCall(m) => m,
        }
    }
}

/// A single rejected instruction with its location in the function.
///
/// Returned by [`scan_function`]; the `(inst, block)` pair is enough for
/// the caller to render a diagnostic with `cranelift_codegen`'s own
/// pretty-printer (e.g. `func.dfg.display_inst(inst)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    /// Cranelift instruction that triggered the rejection.
    pub inst: Inst,
    /// Block containing the offending instruction.
    pub block: Block,
    /// Reason for rejection (also carries the opcode mnemonic).
    pub reason: RejectReason,
}

/// Walk `func` and return every rejection.
///
/// Empty `Vec` ⇔ the function is admissible to the Pliron pipeline modulo
/// per-opcode lowering errors caught later. The detector is intentionally
/// conservative: when in doubt, reject — it is cheaper to fall back to the
/// blueprint detector than to half-lower a function and then bail.
///
/// The scan is `O(n)` in the number of instructions and allocates only the
/// returned `Vec`. It does not run the Cranelift verifier and therefore
/// makes no well-formedness guarantees on `func` beyond what the layout
/// iterator exposes.
pub fn scan_function(func: &Function) -> Vec<Rejection> {
    let mut rejections = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if let Some(reason) = classify_opcode(func.dfg.insts[inst].opcode()) {
                rejections.push(Rejection {
                    inst,
                    block,
                    reason,
                });
            }
        }
    }
    rejections
}

/// Convenience wrapper: return the **first** rejection or `None`.
///
/// Useful for the call-site shortcut "is this function admissible?" —
/// avoids the `Vec` allocation when the caller does not need the full
/// list. Equivalent to `scan_function(func).into_iter().next()` but does
/// not walk the rest of the function once a rejection is found.
pub fn check_function(func: &Function) -> Option<Rejection> {
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if let Some(reason) = classify_opcode(func.dfg.insts[inst].opcode()) {
                return Some(Rejection {
                    inst,
                    block,
                    reason,
                });
            }
        }
    }
    None
}

/// Map a Cranelift [`Opcode`] to a [`RejectReason`] if it is on the
/// reject list, otherwise `None`.
///
/// Centralised match so the two scanners stay in lock-step. The mnemonic
/// strings are hard-coded `&'static str` literals (rather than
/// `opcode_name(op)` / `Display`) to keep [`RejectReason`] `Copy`-able
/// and avoid the runtime cost of stringification in the hot path; they
/// are kept identical to Cranelift's own mnemonics so the two are
/// interchangeable for log output.
fn classify_opcode(op: Opcode) -> Option<RejectReason> {
    match op {
        // Atomics — see RejectReason::Atomic.
        Opcode::AtomicLoad => Some(RejectReason::Atomic("atomic_load")),
        Opcode::AtomicStore => Some(RejectReason::Atomic("atomic_store")),
        Opcode::AtomicRmw => Some(RejectReason::Atomic("atomic_rmw")),
        Opcode::AtomicCas => Some(RejectReason::Atomic("atomic_cas")),

        // Strict-FP saturating conversions — see RejectReason::StrictFp.
        // PTX `cvt.f32.s32` / `cvt.f32.u32` do not saturate; the wave-2
        // lowering cannot honour the Wasm-strict saturation semantics
        // without an explicit min/max clamp the detector chooses not to
        // synthesize for v0.4.
        Opcode::FcvtToSintSat => Some(RejectReason::StrictFp("fcvt_to_sint_sat")),
        Opcode::FcvtToUintSat => Some(RejectReason::StrictFp("fcvt_to_uint_sat")),

        // Host-call prohibition — see RejectReason::HostCall. This also
        // catches the libcall-expanded forms of `memory.grow`,
        // `memory.size`, `memory.copy`, `memory.fill`, `table.get`, and
        // `table.set` that `cranelift-wasm` emits in pinned Cranelift
        // 0.111. Wave 3+ will refine this match to peek into `func_ref`
        // and distinguish device-internal calls from runtime helpers.
        Opcode::Call => Some(RejectReason::HostCall("call")),
        Opcode::CallIndirect => Some(RejectReason::HostCall("call_indirect")),
        Opcode::ReturnCall => Some(RejectReason::HostCall("return_call")),
        Opcode::ReturnCallIndirect => Some(RejectReason::HostCall("return_call_indirect")),

        // Everything else is provisionally admissible; the per-family
        // lowerings may still reject in their own pass.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::immediates::Offset32;
    use cranelift_codegen::ir::instructions::InstructionData;
    use cranelift_codegen::ir::{
        AtomicRmwOp, FuncRef, MemFlags, SigRef, Signature, UserFuncName, Value, ValueList,
    };
    use cranelift_codegen::isa::CallConv;

    /// Build a fresh empty function with a single empty entry block.
    /// The wave-2 detector never inspects values or types, only opcodes,
    /// so the dummy `Value(0)` / `FuncRef(0)` / `SigRef(0)` placeholders
    /// the tests use never need to be valid — `scan_function` does not
    /// dereference them.
    fn empty_func() -> (Function, Block) {
        let mut func = Function::with_name_signature(
            UserFuncName::user(0, 0),
            Signature::new(CallConv::SystemV),
        );
        let block = func.dfg.make_block();
        func.layout.append_block(block);
        (func, block)
    }

    /// Convenience: place `data` at the end of `block` and return the
    /// resulting `Inst`. Intentionally does NOT call `make_inst_results`
    /// — the detector ignores result values, and skipping result wiring
    /// keeps the test fixtures small and verifier-independent.
    fn append(func: &mut Function, block: Block, data: InstructionData) -> Inst {
        let inst = func.dfg.make_inst(data);
        func.layout.append_inst(inst, block);
        inst
    }

    /// A throwaway `Value` reference. Never dereferenced by the
    /// detector, so any well-formed `Value` ID is fine.
    fn dummy_val() -> Value {
        Value::from_u32(0)
    }

    // ---- Positive (admissible) case -----------------------------------

    /// A function with only admissible opcodes (`iadd` + `return`)
    /// produces no rejections.
    #[test]
    fn admissible_iadd_return_yields_empty() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::Binary {
                opcode: Opcode::Iadd,
                args: [dummy_val(), dummy_val()],
            },
        );
        append(
            &mut func,
            block,
            InstructionData::MultiAry {
                opcode: Opcode::Return,
                args: ValueList::new(),
            },
        );

        assert_eq!(scan_function(&func), vec![]);
        assert_eq!(check_function(&func), None);
    }

    // ---- Atomic family -------------------------------------------------

    /// `atomic_load` is rejected with the `Atomic` reason.
    #[test]
    fn atomic_load_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::LoadNoOffset {
                opcode: Opcode::AtomicLoad,
                flags: MemFlags::new(),
                arg: dummy_val(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, RejectReason::Atomic("atomic_load"));
        assert_eq!(rejections[0].block, block);
    }

    /// `atomic_store` is rejected with the `Atomic` reason.
    #[test]
    fn atomic_store_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::StoreNoOffset {
                opcode: Opcode::AtomicStore,
                flags: MemFlags::new(),
                args: [dummy_val(), dummy_val()],
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, RejectReason::Atomic("atomic_store"));
    }

    /// `atomic_rmw` is rejected with the `Atomic` reason.
    #[test]
    fn atomic_rmw_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::AtomicRmw {
                opcode: Opcode::AtomicRmw,
                flags: MemFlags::new(),
                op: AtomicRmwOp::Add,
                args: [dummy_val(), dummy_val()],
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, RejectReason::Atomic("atomic_rmw"));
    }

    /// `atomic_cas` is rejected with the `Atomic` reason.
    #[test]
    fn atomic_cas_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::AtomicCas {
                opcode: Opcode::AtomicCas,
                flags: MemFlags::new(),
                args: [dummy_val(), dummy_val(), dummy_val()],
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, RejectReason::Atomic("atomic_cas"));
    }

    // ---- Strict-FP family ----------------------------------------------

    /// `fcvt_to_sint_sat` is rejected with the `StrictFp` reason.
    #[test]
    fn fcvt_to_sint_sat_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::Unary {
                opcode: Opcode::FcvtToSintSat,
                arg: dummy_val(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(
            rejections[0].reason,
            RejectReason::StrictFp("fcvt_to_sint_sat")
        );
    }

    /// `fcvt_to_uint_sat` is also rejected (sibling saturating
    /// conversion). Covers the second variant of the `StrictFp` family
    /// the detector recognises.
    #[test]
    fn fcvt_to_uint_sat_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::Unary {
                opcode: Opcode::FcvtToUintSat,
                arg: dummy_val(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(
            rejections[0].reason,
            RejectReason::StrictFp("fcvt_to_uint_sat")
        );
    }

    // ---- Host-call family ----------------------------------------------

    /// `call` is rejected with the `HostCall` reason. The dummy
    /// `FuncRef(0)` is never resolved by the detector.
    #[test]
    fn call_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::Call {
                opcode: Opcode::Call,
                func_ref: FuncRef::from_u32(0),
                args: ValueList::new(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].reason, RejectReason::HostCall("call"));
    }

    /// `call_indirect` is rejected with the `HostCall` reason.
    #[test]
    fn call_indirect_is_rejected() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::CallIndirect {
                opcode: Opcode::CallIndirect,
                sig_ref: SigRef::from_u32(0),
                args: ValueList::new(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 1);
        assert_eq!(
            rejections[0].reason,
            RejectReason::HostCall("call_indirect")
        );
    }

    // ---- TableOp / GcOp / MemoryResize / LargeMemcpy -------------------
    //
    // These four categories are pre-declared in `RejectReason` for forward
    // compatibility but are not wired to a specific Cranelift opcode in
    // 0.111.9 (see module docs — the wasm-to-clif translator expands them
    // into `call` libcalls before the detector sees them, so they are
    // caught indirectly via `HostCall`). The unit tests below exercise
    // the *variant* surface — constructing and inspecting each reason —
    // so a future wiring patch only needs to add a `match` arm in
    // `classify_opcode`, not redesign the public enum.

    /// The `TableOp` variant round-trips through `opcode_mnemonic`.
    /// Pre-declared variant; no opcode wired in 0.111.
    #[test]
    fn table_op_variant_is_constructible() {
        let r = RejectReason::TableOp("table_get");
        assert_eq!(r.opcode_mnemonic(), "table_get");
    }

    /// The `GcOp` variant round-trips through `opcode_mnemonic`.
    /// Pre-declared variant; no opcode wired in 0.111.
    #[test]
    fn gc_op_variant_is_constructible() {
        let r = RejectReason::GcOp("ref_func");
        assert_eq!(r.opcode_mnemonic(), "ref_func");
    }

    /// The `MemoryResize` variant round-trips through `opcode_mnemonic`.
    /// Pre-declared variant; no opcode wired in 0.111.
    #[test]
    fn memory_resize_variant_is_constructible() {
        let r = RejectReason::MemoryResize("memory_grow");
        assert_eq!(r.opcode_mnemonic(), "memory_grow");
    }

    /// The `LargeMemcpy` variant round-trips through `opcode_mnemonic`.
    /// Pre-declared variant; no opcode wired in 0.111.
    #[test]
    fn large_memcpy_variant_is_constructible() {
        let r = RejectReason::LargeMemcpy("memory_copy");
        assert_eq!(r.opcode_mnemonic(), "memory_copy");
    }

    // ---- Multi-rejection / ordering ------------------------------------

    /// A function with two rejected instructions returns both, in
    /// layout order. Locks in the contract that the detector is a
    /// *list* not a "first failure" — wave 3+ diagnostics will rely on
    /// the full list to surface every issue at once.
    #[test]
    fn multiple_rejections_returned_in_layout_order() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::LoadNoOffset {
                opcode: Opcode::AtomicLoad,
                flags: MemFlags::new(),
                arg: dummy_val(),
            },
        );
        append(
            &mut func,
            block,
            InstructionData::Call {
                opcode: Opcode::Call,
                func_ref: FuncRef::from_u32(0),
                args: ValueList::new(),
            },
        );

        let rejections = scan_function(&func);
        assert_eq!(rejections.len(), 2);
        assert_eq!(rejections[0].reason, RejectReason::Atomic("atomic_load"));
        assert_eq!(rejections[1].reason, RejectReason::HostCall("call"));
    }

    /// `check_function` returns the first rejection encountered (early
    /// exit), not the full list.
    #[test]
    fn check_function_returns_first_rejection() {
        let (mut func, block) = empty_func();
        append(
            &mut func,
            block,
            InstructionData::Call {
                opcode: Opcode::Call,
                func_ref: FuncRef::from_u32(0),
                args: ValueList::new(),
            },
        );
        append(
            &mut func,
            block,
            InstructionData::LoadNoOffset {
                opcode: Opcode::AtomicLoad,
                flags: MemFlags::new(),
                arg: dummy_val(),
            },
        );

        let first = check_function(&func).expect("function has rejections");
        assert_eq!(first.reason, RejectReason::HostCall("call"));
    }

    /// `Offset32` is included in the `cranelift_codegen::ir::immediates`
    /// import set above so the test module compiles even when no test
    /// directly references it; this assertion keeps the import live and
    /// documents that the detector intentionally ignores instruction
    /// offsets (only opcode identity matters).
    #[test]
    fn offset_field_is_irrelevant_to_detection() {
        let _ = Offset32::new(0);
    }
}
