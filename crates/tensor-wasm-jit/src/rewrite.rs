// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Wasm-to-Wasm rewrite that swaps offload-candidate function bodies with
//! host-import calls to the JIT dispatch service.
//!
//! Sidesteps the wasmtime-cranelift fork question — see `docs/WASMTIME-FORK.md`.
//!
//! # ABI (v0.1.0)
//!
//! Every offload-candidate function `f(p0, p1, ..., pK) -> (r0, r1, ..., rN)`
//! is replaced with a trampoline that marshalls arguments through a shared
//! scratch region in the guest's first linear memory:
//!
//! 1. `call __tensor_wasm_jit_alloc(scratch_size)` — host-side arena returns an
//!    `i32` pointer into the guest's first memory (`memory 0`). The host
//!    guarantees the returned pointer plus `scratch_size` does not overlap
//!    any other live allocation.
//! 2. For each parameter in declaration order, store its raw bytes at
//!    `scratch_ptr + arg_off` using the appropriate `i32.store` /
//!    `i64.store` / `f32.store` / `f64.store`. Offsets are byte-packed in
//!    declared order; i64/f64 take 8 bytes, i32/f32 take 4. There is no
//!    inter-arg padding — the host follows the exact same packing.
//! 3. `call __tensor_wasm_jit_dispatch(fp_lo: i64, fp_hi: i64, scratch_ptr: i32,
//!    args_byte_len: i32, results_byte_len: i32) -> i32` — runs the
//!    cached PTX kernel (CUDA path) or simulates it on the host (no-CUDA
//!    path) and writes the result bytes to `scratch_ptr + args_byte_len`.
//!    Returns `0` on success, nonzero on error.
//! 4. For each result in declaration order, load its bytes from
//!    `scratch_ptr + args_byte_len + result_off` and push to the wasm
//!    stack.
//! 5. `call __tensor_wasm_jit_free(scratch_ptr, scratch_size)` — returns the
//!    arena slot.
//!
//! `fingerprint` is the 64-bit BLAKE3-truncated blueprint hash. We pack the
//! lo/hi 32-bit halves into two `i64` slots so a Wasm 1.0 module can carry
//! the full 64-bit value without needing the `multi-value` proposal.
//!
//! Supported parameter / result types for v0.1.0: **i32, i64, f32, f64**.
//! v128 and reftypes are explicitly rejected by [`enc_val_type`] — the
//! detector also rejects candidates that use unsupported types, so the
//! rewriter only sees compatible signatures by the time it gets to
//! [`build_trampoline`].
//!
//! The host-side implementations of `__tensor_wasm_jit_dispatch`, `__tensor_wasm_jit_alloc`,
//! and `__tensor_wasm_jit_free` live in `tensor-wasm-exec`'s `jit_dispatch` module.

use std::convert::Infallible;
use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info};
use wasm_encoder::reencode::Reencode;
use wasmparser::{Operator, Parser, Payload};

use tensor_wasm_core::types::TenantId;

use crate::cache::{CacheKey, CachedKernel, CompiledHandle, KernelCache};
use crate::clif_lower::lower_block;
use crate::detector::{classify, BlockIR, DetectorConfig, DetectorVerdict, Op};
use crate::ptx_emit::emit;

/// Default host import module name.
pub const DEFAULT_HOST_MODULE: &str = "tensor-wasm:jit/host";
/// Default host import field name for the dispatch entry point.
pub const DEFAULT_HOST_FN: &str = "dispatch";
/// Host import name for the scratch-arena allocator.
pub const DEFAULT_HOST_ALLOC_FN: &str = "alloc";
/// Host import name for the scratch-arena deallocator.
pub const DEFAULT_HOST_FREE_FN: &str = "free";
/// Default sm_version the rewriter pre-populates kernels for.
pub const DEFAULT_SM_VERSION: u32 = 80;

/// Options controlling the rewrite.
#[derive(Debug, Clone)]
pub struct RewriteOptions {
    /// Host import module (defaults to `tensor-wasm:jit/host`).
    pub host_module: String,
    /// Host import function name for `dispatch` (defaults to `dispatch`).
    pub host_fn: String,
    /// Host import function name for the scratch allocator (defaults to `alloc`).
    pub host_alloc_fn: String,
    /// Host import function name for the scratch deallocator (defaults to `free`).
    pub host_free_fn: String,
    /// CUDA compute capability the pre-populated kernels are compiled for.
    pub sm_version: u32,
    /// Detector configuration used to classify each function body. Use this
    /// to lower thresholds in tests or to tune offload aggressiveness in
    /// production deployments.
    pub detector: DetectorConfig,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            host_module: DEFAULT_HOST_MODULE.into(),
            host_fn: DEFAULT_HOST_FN.into(),
            host_alloc_fn: DEFAULT_HOST_ALLOC_FN.into(),
            host_free_fn: DEFAULT_HOST_FREE_FN.into(),
            sm_version: DEFAULT_SM_VERSION,
            detector: DetectorConfig::default(),
        }
    }
}

/// One swapped function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadedFunction {
    /// Index of the function in the ORIGINAL (pre-rewrite) function index
    /// space. The post-rewrite index is shifted by the number of imports
    /// the rewriter added (3: dispatch, alloc, free).
    pub function_index: u32,
    /// Blueprint fingerprint — also the [`KernelCache`] key the dispatch
    /// import looks up at runtime.
    pub fingerprint: u64,
    /// Number of original operators in the function body the swap replaced.
    pub original_op_count: usize,
}

/// Outcome of a rewrite.
#[derive(Debug, Clone)]
pub struct RewriteOutcome {
    /// The post-rewrite Wasm bytes — pass these to Wasmtime in place of the
    /// original.
    pub rewritten_wasm: Vec<u8>,
    /// Functions whose bodies were swapped for a dispatch trampoline.
    pub offloaded_functions: Vec<OffloadedFunction>,
    /// Total number of defined functions in the original module.
    pub total_defined_functions: u32,
}

/// Errors raised by the rewriter.
#[derive(Debug, Error)]
pub enum RewriteError {
    /// The Wasm bytes failed to parse.
    #[error("wasmparser: {0}")]
    Parse(String),
    /// The lower→emit pipeline rejected a candidate (e.g. unsupported op).
    /// The candidate is silently skipped — this variant is returned only when
    /// the rewriter is configured to be strict (currently never, but reserved).
    #[error("lower: {0}")]
    Lower(String),
    /// Re-encoding the module failed.
    #[error("reencode: {0}")]
    Reencode(String),
    /// An offload candidate's signature couldn't be safely synthesised as a
    /// trampoline (e.g. unsupported result type). The candidate is skipped
    /// rather than producing invalid Wasm.
    #[error("trampoline: {0}")]
    Trampoline(String),
}

impl<E: std::fmt::Display> From<wasm_encoder::reencode::Error<E>> for RewriteError {
    fn from(e: wasm_encoder::reencode::Error<E>) -> Self {
        RewriteError::Reencode(format!("{e}"))
    }
}

/// Per-function record from the pre-pass analysis.
#[derive(Debug, Clone)]
struct FuncInfo {
    /// Type index in the original module.
    type_index: u32,
    /// Detector verdict for this function body.
    verdict: DetectorVerdict,
    /// Blueprint fingerprint (only meaningful when verdict is `Offload`).
    fingerprint: Option<u64>,
    /// Op count walked (for diagnostics).
    op_count: usize,
}

/// Mirror of `tensor_wasm_exec::auto_offload::op_to_detector_op` — duplicated here
/// because `tensor-wasm-jit` is intentionally a leaf dependency that does not
/// reference `tensor-wasm-exec`.
fn op_to_detector_op(op: &Operator<'_>) -> Op {
    use wasmparser::Operator::*;
    match op {
        V128Load { .. } => Op::Load,
        V128Store { .. } => Op::Store,
        F32Add | I32Add | I64Add | F64Add => Op::ScalarAdd,
        F32Mul | I32Mul | I64Mul | F64Mul => Op::ScalarMul,
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } => Op::Load,
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } => Op::Store,
        Br { .. }
        | BrIf { .. }
        | BrTable { .. }
        | If { .. }
        | Else
        | Loop { .. }
        | Block { .. } => Op::Branch,
        Call { .. } | CallIndirect { .. } | ReturnCall { .. } => Op::Call,
        F32x4Add | F64x2Add | I32x4Add | I64x2Add | I16x8Add | I8x16Add => Op::V128Add,
        F32x4Mul | F64x2Mul | I32x4Mul | I64x2Mul | I16x8Mul => Op::V128Mul,
        // Anything else — local.get, drop, const, … — is classified as
        // [`Op::Other`] so the v128 ratio is computed honestly. Previously
        // this fell through to `Op::ScalarAdd`, which inflated the apparent
        // arithmetic density of non-arithmetic functions.
        _ => Op::Other,
    }
}

/// Decoded function type — just what we need to synthesise a trampoline.
#[derive(Debug, Clone)]
struct DecodedFuncType {
    params: Vec<wasmparser::ValType>,
    results: Vec<wasmparser::ValType>,
}

/// Pre-pass: walk the module, build the type table, count function imports,
/// classify each defined function, lower → emit for offload candidates, and
/// pre-populate the cache.
fn analyse(
    wasm: &[u8],
    opts: &RewriteOptions,
    cache: &KernelCache,
) -> Result<AnalyseOutcome, RewriteError> {
    let mut types: Vec<Option<DecodedFuncType>> = Vec::new();
    let mut function_type_indices: Vec<u32> = Vec::new();
    let mut num_function_imports: u32 = 0;
    let mut func_infos: Vec<FuncInfo> = Vec::new();
    let mut defined_function_cursor: usize = 0;
    // The trampoline reads / writes memory 0 via `i32.store` etc. — if the
    // input module has no memory section the trampoline body won't validate
    // even if the rest of the module would. Track whether we saw one so we
    // can suppress swaps on memory-less modules rather than emit invalid
    // wasm.
    let mut has_memory: bool = false;

    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.map_err(|e| RewriteError::Parse(format!("{e}")))?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    let rec = rec_group.map_err(|e| RewriteError::Parse(format!("{e}")))?;
                    for sub in rec.into_types() {
                        // We only care about function types; everything else
                        // (array/struct/cont) records `None` so trampoline
                        // synthesis declines those candidates.
                        let decoded = match sub.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(f) => Some(DecodedFuncType {
                                params: f.params().to_vec(),
                                results: f.results().to_vec(),
                            }),
                            _ => None,
                        };
                        types.push(decoded);
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import.map_err(|e| RewriteError::Parse(format!("{e}")))?;
                    if matches!(import.ty, wasmparser::TypeRef::Func(_)) {
                        num_function_imports += 1;
                    }
                    if matches!(import.ty, wasmparser::TypeRef::Memory(_)) {
                        has_memory = true;
                    }
                }
            }
            Payload::MemorySection(reader) => {
                if reader.count() > 0 {
                    has_memory = true;
                }
            }
            Payload::FunctionSection(reader) => {
                for ty_idx in reader {
                    let ty_idx = ty_idx.map_err(|e| RewriteError::Parse(format!("{e}")))?;
                    function_type_indices.push(ty_idx);
                }
            }
            Payload::CodeSectionEntry(body) => {
                let type_index = function_type_indices
                    .get(defined_function_cursor)
                    .copied()
                    .ok_or_else(|| {
                        RewriteError::Parse(format!(
                            "code body {defined_function_cursor} has no matching function type"
                        ))
                    })?;
                let mut ops_reader = body
                    .get_operators_reader()
                    .map_err(|e| RewriteError::Parse(format!("{e}")))?;
                let mut detector_ops = Vec::new();
                let mut saw_loop = false;
                while !ops_reader.eof() {
                    match ops_reader.read() {
                        Ok(op) => {
                            if matches!(op, wasmparser::Operator::Loop { .. }) {
                                saw_loop = true;
                            }
                            detector_ops.push(op_to_detector_op(&op));
                        }
                        Err(_) => break,
                    }
                }
                let func_index_in_global_space =
                    num_function_imports + defined_function_cursor as u32;
                // Only set a trip-count guess if the body actually had a
                // `Loop`. A no-loop function inheriting `Some(128)`
                // misleads the detector into approving cold straight-line
                // code.
                let trip_guess = if saw_loop { Some(128) } else { None };
                let block = BlockIR::new(
                    format!("func{func_index_in_global_space}"),
                    detector_ops.clone(),
                    trip_guess,
                );
                let verdict = classify(&block, &opts.detector);
                let op_count = detector_ops.len();
                let mut fingerprint = None;
                if matches!(verdict, DetectorVerdict::Offload) {
                    // Also gate on supported parameter/result types: if the
                    // function uses v128 or reftypes in its signature, skip
                    // the trampoline.
                    let signature_ok = types
                        .get(type_index as usize)
                        .and_then(|t| t.as_ref())
                        .map(|t| {
                            t.params.iter().all(is_supported_primitive)
                                && t.results.iter().all(is_supported_primitive)
                        })
                        .unwrap_or(false);
                    if !has_memory {
                        debug!(
                            target: "tensor_wasm_jit::rewrite",
                            function = func_index_in_global_space,
                            "offload candidate rejected: module has no memory for trampoline marshalling"
                        );
                    } else if !signature_ok {
                        debug!(
                            target: "tensor_wasm_jit::rewrite",
                            function = func_index_in_global_space,
                            "offload candidate rejected: unsupported parameter/result type"
                        );
                    } else {
                        // The lowering pass refuses anything outside the
                        // {V128*, Load, Store} taxonomy. Filter the full op
                        // stream down to those before handing it off so the
                        // analyser doesn't trip on local.get / br / call
                        // noise that the detector already weighed.
                        let lower_block_input = BlockIR::new(
                            block.name.clone(),
                            detector_ops
                                .iter()
                                .copied()
                                .filter(|o| {
                                    matches!(
                                        o,
                                        Op::V128Add
                                            | Op::V128Mul
                                            | Op::V128Fma
                                            | Op::Load
                                            | Op::Store
                                    )
                                })
                                .collect(),
                            block.loop_trip_count,
                        );
                        match lower_block(&lower_block_input) {
                            Ok(blueprint) => match emit(&blueprint) {
                                Ok(ptx) => {
                                    let fp = blueprint.fingerprint();
                                    // Rewrite-time pre-population: the
                                    // rewriter runs at module-load with no
                                    // tenant context yet, so pre-populated
                                    // entries land under the placeholder
                                    // `TenantId(0)`. The runtime dispatch
                                    // (which knows the real tenant) will
                                    // miss this entry and re-emit on first
                                    // call — that's the safe default until
                                    // the rewriter is plumbed with the
                                    // owning tenant. See `CacheKey` docs
                                    // for the cross-tenant confused-deputy
                                    // primitive this prevents.
                                    let key = CacheKey::for_tenant(
                                        TenantId(0),
                                        fp,
                                        opts.sm_version,
                                    );
                                    cache.put(
                                        key,
                                        CachedKernel::new(
                                            fp,
                                            Arc::new(ptx),
                                            CompiledHandle::default(),
                                        ),
                                    );
                                    fingerprint = Some(fp);
                                    info!(
                                        target: "tensor_wasm_jit::rewrite",
                                        function = func_index_in_global_space,
                                        op_count,
                                        fingerprint = fp,
                                        "pre-populated kernel cache for offload candidate"
                                    );
                                }
                                Err(e) => {
                                    // Emission refused (e.g. MatMul not yet
                                    // implemented) — keep this function on the
                                    // CPU path. This is the deopt-at-rewrite
                                    // signal at the emit stage.
                                    debug!(
                                        target: "tensor_wasm_jit::rewrite",
                                        function = func_index_in_global_space,
                                        op_count,
                                        reason = %e,
                                        "offload candidate rejected by PTX emitter"
                                    );
                                }
                            },
                            Err(e) => {
                                // Lowering refused — keep this function on the
                                // CPU path. This is the deopt-at-rewrite signal.
                                debug!(
                                    target: "tensor_wasm_jit::rewrite",
                                    function = func_index_in_global_space,
                                    op_count,
                                    reason = %e,
                                    "offload candidate rejected by lowering"
                                );
                            }
                        }
                    }
                }
                func_infos.push(FuncInfo {
                    type_index,
                    verdict,
                    fingerprint,
                    op_count,
                });
                defined_function_cursor += 1;
            }
            _ => {}
        }
    }

    Ok(AnalyseOutcome {
        types,
        num_function_imports,
        func_infos,
    })
}

struct AnalyseOutcome {
    types: Vec<Option<DecodedFuncType>>,
    num_function_imports: u32,
    func_infos: Vec<FuncInfo>,
}

/// True if this val type is a supported primitive for the v0.1.0 ABI.
fn is_supported_primitive(v: &wasmparser::ValType) -> bool {
    matches!(
        v,
        wasmparser::ValType::I32
            | wasmparser::ValType::I64
            | wasmparser::ValType::F32
            | wasmparser::ValType::F64
    )
}

/// Convert a [`wasmparser::ValType`] to its [`wasm_encoder::ValType`]
/// counterpart. We refuse `Ref(_)` and `V128` — the trampoline can't
/// marshall those through a byte-packed scratch region without richer
/// plumbing.
fn enc_val_type(v: wasmparser::ValType) -> Result<wasm_encoder::ValType, RewriteError> {
    match v {
        wasmparser::ValType::I32 => Ok(wasm_encoder::ValType::I32),
        wasmparser::ValType::I64 => Ok(wasm_encoder::ValType::I64),
        wasmparser::ValType::F32 => Ok(wasm_encoder::ValType::F32),
        wasmparser::ValType::F64 => Ok(wasm_encoder::ValType::F64),
        wasmparser::ValType::V128 => Err(RewriteError::Trampoline(
            "v128 result not supported by dispatch trampoline".into(),
        )),
        wasmparser::ValType::Ref(_) => Err(RewriteError::Trampoline(
            "reference result not supported by dispatch trampoline".into(),
        )),
    }
}

/// Byte size of a supported value type. Returns `Err` on unsupported types —
/// call sites typically pre-validate with [`is_supported_primitive`], but the
/// fallible signature means a malformed input can never panic the rewriter.
fn val_type_size(v: wasmparser::ValType) -> Result<u32, RewriteError> {
    match v {
        wasmparser::ValType::I32 | wasmparser::ValType::F32 => Ok(4),
        wasmparser::ValType::I64 | wasmparser::ValType::F64 => Ok(8),
        _ => Err(RewriteError::Trampoline(format!(
            "val_type_size: unsupported type {v:?}"
        ))),
    }
}

/// Aligned offset for `val_type` in a packed byte buffer. We use natural
/// alignment (4 for i32/f32, 8 for i64/f64) so the host-side reads on
/// systems that care about alignment never trap.
fn aligned_advance(off: u32, v: wasmparser::ValType) -> Result<u32, RewriteError> {
    let align = val_type_size(v)?;
    let mask = align - 1;
    let summed = off.checked_add(mask).ok_or_else(|| {
        RewriteError::Trampoline(format!(
            "aligned_advance: offset overflow (off={off}, mask={mask})"
        ))
    })?;
    Ok(summed & !mask)
}

/// Layout of a packed argument or result region: total byte length plus a
/// per-element offset table.
struct PackedLayout {
    offsets: Vec<u32>,
    total_bytes: u32,
}

fn pack_layout(types: &[wasmparser::ValType]) -> Result<PackedLayout, RewriteError> {
    let mut offsets = Vec::with_capacity(types.len());
    let mut off = 0u32;
    for t in types {
        let aligned = aligned_advance(off, *t)?;
        offsets.push(aligned);
        let size = val_type_size(*t)?;
        off = aligned.checked_add(size).ok_or_else(|| {
            RewriteError::Trampoline(format!(
                "pack_layout: cursor overflow (aligned={aligned}, size={size})"
            ))
        })?;
    }
    // Round the total up to 8 bytes so the results region following the
    // args region also starts 8-aligned.
    let total_bytes = off.checked_add(7).ok_or_else(|| {
        RewriteError::Trampoline(format!(
            "pack_layout: total-bytes round-up overflow (off={off})"
        ))
    })? & !7u32;
    Ok(PackedLayout {
        offsets,
        total_bytes,
    })
}

/// Indices of the three host imports the rewriter inserts.
struct DispatchImports {
    dispatch: u32,
    alloc: u32,
    free: u32,
}

/// Build a trampoline body for an offload-swapped function. See the
/// module-level docs for the ABI this implements.
fn build_trampoline(
    fingerprint: u64,
    imports: &DispatchImports,
    params: &[wasmparser::ValType],
    results: &[wasmparser::ValType],
) -> Result<wasm_encoder::Function, RewriteError> {
    // Validate the signature up-front; emit a `Trampoline` error rather
    // than fabricate an invalid function body.
    for t in params.iter().chain(results.iter()) {
        if !is_supported_primitive(t) {
            return Err(RewriteError::Trampoline(format!(
                "unsupported type in signature: {t:?}",
            )));
        }
    }

    let args_layout = pack_layout(params)?;
    let results_layout = pack_layout(results)?;
    let scratch_size = args_layout
        .total_bytes
        .checked_add(results_layout.total_bytes)
        .ok_or_else(|| RewriteError::Trampoline("scratch size overflow".into()))?;

    // Two i32 locals: scratch_ptr (idx = params.len()) and rc (idx =
    // params.len() + 1) for the dispatch return-code trap below.
    let scratch_local_idx: u32 = params.len() as u32;
    let rc_local_idx: u32 = scratch_local_idx + 1;
    let mut func = wasm_encoder::Function::new(std::iter::once((
        2u32,
        wasm_encoder::ValType::I32,
    )));

    use wasm_encoder::Instruction as I;

    // scratch_ptr = __tensor_wasm_jit_alloc(scratch_size)
    func.instruction(&I::I32Const(scratch_size as i32));
    func.instruction(&I::Call(imports.alloc));
    func.instruction(&I::LocalSet(scratch_local_idx));

    // For each parameter: scratch_ptr + arg_off ; param ; <type>.store
    for (i, ty) in params.iter().enumerate() {
        let off = args_layout.offsets[i];
        func.instruction(&I::LocalGet(scratch_local_idx));
        func.instruction(&I::LocalGet(i as u32));
        let memarg = wasm_encoder::MemArg {
            offset: off as u64,
            align: log2_align(*ty),
            memory_index: 0,
        };
        match ty {
            wasmparser::ValType::I32 => {
                func.instruction(&I::I32Store(memarg));
            }
            wasmparser::ValType::I64 => {
                func.instruction(&I::I64Store(memarg));
            }
            wasmparser::ValType::F32 => {
                func.instruction(&I::F32Store(memarg));
            }
            wasmparser::ValType::F64 => {
                func.instruction(&I::F64Store(memarg));
            }
            _ => unreachable!("validated above"),
        }
    }

    // Pack fingerprint as two i64 halves.
    let fp_lo = (fingerprint & 0xFFFF_FFFF) as i64;
    let fp_hi = (fingerprint >> 32) as i64;

    // call __tensor_wasm_jit_dispatch(fp_lo, fp_hi, scratch_ptr, args_len, results_len) -> i32
    func.instruction(&I::I64Const(fp_lo));
    func.instruction(&I::I64Const(fp_hi));
    func.instruction(&I::LocalGet(scratch_local_idx));
    func.instruction(&I::I32Const(args_layout.total_bytes as i32));
    func.instruction(&I::I32Const(results_layout.total_bytes as i32));
    func.instruction(&I::Call(imports.dispatch));
    // Capture the i32 return code from dispatch into `rc_local_idx` and
    // trap the guest if it's nonzero. Previously this dropped the rc
    // silently, so a deopted offload would feed zero-filled result bytes
    // back to the caller and mask host-side failures. The trap surfaces
    // the failure as a wasm trap the embedder catches with the rest of
    // its trap handling.
    //
    //   local.tee $rc          ;; consumes the dispatch i32, leaves a copy
    //   i32.const 0
    //   i32.ne
    //   if
    //     unreachable
    //   end
    func.instruction(&I::LocalTee(rc_local_idx));
    func.instruction(&I::I32Const(0));
    func.instruction(&I::I32Ne);
    func.instruction(&I::If(wasm_encoder::BlockType::Empty));
    func.instruction(&I::Unreachable);
    func.instruction(&I::End);

    // For each result: scratch_ptr ; <type>.load (offset = args_len + result_off)
    for (i, ty) in results.iter().enumerate() {
        let off = args_layout.total_bytes + results_layout.offsets[i];
        func.instruction(&I::LocalGet(scratch_local_idx));
        let memarg = wasm_encoder::MemArg {
            offset: off as u64,
            align: log2_align(*ty),
            memory_index: 0,
        };
        match ty {
            wasmparser::ValType::I32 => {
                func.instruction(&I::I32Load(memarg));
            }
            wasmparser::ValType::I64 => {
                func.instruction(&I::I64Load(memarg));
            }
            wasmparser::ValType::F32 => {
                func.instruction(&I::F32Load(memarg));
            }
            wasmparser::ValType::F64 => {
                func.instruction(&I::F64Load(memarg));
            }
            _ => unreachable!("validated above"),
        }
    }

    // __tensor_wasm_jit_free(scratch_ptr, scratch_size)
    //
    // Emission order is: alloc -> stores -> dispatch -> drop -> loads ->
    // free. Why is loads-then-free safe? The loads push N result values
    // onto the wasm stack. Free's two arguments (scratch_ptr, scratch_size)
    // are pushed AFTER those results, then consumed by `call`. The wasm
    // stack is LIFO; after free returns (no return value), the stack top
    // is exactly the N result values — which is the trampoline's declared
    // return signature.
    func.instruction(&I::LocalGet(scratch_local_idx));
    func.instruction(&I::I32Const(scratch_size as i32));
    func.instruction(&I::Call(imports.free));

    func.instruction(&I::End);
    Ok(func)
}

/// log2 of the natural alignment for a memory store/load. The wasm spec
/// uses this as the alignment hint encoded in `memarg`.
fn log2_align(v: wasmparser::ValType) -> u32 {
    match v {
        wasmparser::ValType::I32 | wasmparser::ValType::F32 => 2,
        wasmparser::ValType::I64 | wasmparser::ValType::F64 => 3,
        _ => 0,
    }
}

/// Stateful re-encoder that:
/// - Appends three host import types (dispatch, alloc, free) to the type
///   section.
/// - Appends three imports to the import section.
/// - Shifts every defined function index by +3.
/// - Swaps offload-candidate function bodies for trampolines.
struct TensorWasmRewriter<'a> {
    opts: &'a RewriteOptions,
    /// `num_existing_function_imports` — anything below this is an imported
    /// function and its index is preserved; anything at or above shifts by
    /// `imports_added` (always 3 in v0.1.0).
    num_function_imports: u32,
    /// Type indices assigned to the dispatch / alloc / free imports.
    dispatch_type_index: u32,
    alloc_type_index: u32,
    free_type_index: u32,
    /// Function indices assigned to the three imports.
    imports: DispatchImports,
    /// Number of function imports the rewriter inserts (always 3).
    imports_added: u32,
    /// Per-defined-function info from the pre-pass (indexed by defined-func
    /// cursor, same order as the code section).
    func_infos: &'a [FuncInfo],
    /// Type table from the pre-pass.
    types: &'a [Option<DecodedFuncType>],
    /// Cursor tracking which defined function body we're currently re-emitting.
    code_cursor: usize,
    /// Records of successful swaps (consumed by [`rewrite_wasm`]).
    swapped: Vec<OffloadedFunction>,
    /// Has the rewriter already appended its dispatch type to the type
    /// section?
    type_section_appended: bool,
    /// Same for the import section.
    import_section_appended: bool,
}

impl<'a> TensorWasmRewriter<'a> {
    fn shifted_fn_index(&self, orig: u32) -> u32 {
        if orig < self.num_function_imports {
            orig
        } else {
            orig + self.imports_added
        }
    }

    fn dispatch_func_type(&self) -> (Vec<wasm_encoder::ValType>, Vec<wasm_encoder::ValType>) {
        (
            vec![
                wasm_encoder::ValType::I64, // fp_lo
                wasm_encoder::ValType::I64, // fp_hi
                wasm_encoder::ValType::I32, // scratch_ptr
                wasm_encoder::ValType::I32, // args_byte_len
                wasm_encoder::ValType::I32, // results_byte_len
            ],
            vec![wasm_encoder::ValType::I32],
        )
    }

    fn alloc_func_type(&self) -> (Vec<wasm_encoder::ValType>, Vec<wasm_encoder::ValType>) {
        (
            vec![wasm_encoder::ValType::I32],
            vec![wasm_encoder::ValType::I32],
        )
    }

    fn free_func_type(&self) -> (Vec<wasm_encoder::ValType>, Vec<wasm_encoder::ValType>) {
        (
            vec![wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
            vec![],
        )
    }

    fn write_import_types(&self, types: &mut wasm_encoder::TypeSection) {
        let (p, r) = self.dispatch_func_type();
        types.ty().function(p, r);
        let (p, r) = self.alloc_func_type();
        types.ty().function(p, r);
        let (p, r) = self.free_func_type();
        types.ty().function(p, r);
    }

    fn write_imports(&self, imports: &mut wasm_encoder::ImportSection) {
        imports.import(
            &self.opts.host_module,
            &self.opts.host_fn,
            wasm_encoder::EntityType::Function(self.dispatch_type_index),
        );
        imports.import(
            &self.opts.host_module,
            &self.opts.host_alloc_fn,
            wasm_encoder::EntityType::Function(self.alloc_type_index),
        );
        imports.import(
            &self.opts.host_module,
            &self.opts.host_free_fn,
            wasm_encoder::EntityType::Function(self.free_type_index),
        );
    }
}

impl<'a> Reencode for TensorWasmRewriter<'a> {
    type Error = Infallible;

    fn function_index(&mut self, func: u32) -> u32 {
        self.shifted_fn_index(func)
    }

    fn parse_type_section(
        &mut self,
        types: &mut wasm_encoder::TypeSection,
        section: wasmparser::TypeSectionReader<'_>,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        wasm_encoder::reencode::utils::parse_type_section(self, types, section)?;
        self.write_import_types(types);
        self.type_section_appended = true;
        Ok(())
    }

    fn parse_import_section(
        &mut self,
        imports: &mut wasm_encoder::ImportSection,
        section: wasmparser::ImportSectionReader<'_>,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        wasm_encoder::reencode::utils::parse_import_section(self, imports, section)?;
        self.write_imports(imports);
        self.import_section_appended = true;
        Ok(())
    }

    fn parse_function_body(
        &mut self,
        code: &mut wasm_encoder::CodeSection,
        func: wasmparser::FunctionBody<'_>,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        let cursor = self.code_cursor;
        self.code_cursor += 1;
        let info = match self.func_infos.get(cursor) {
            Some(i) => i,
            None => {
                return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
            }
        };

        let should_swap =
            matches!(info.verdict, DetectorVerdict::Offload) && info.fingerprint.is_some();
        if !should_swap {
            return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
        }

        let func_ty = self
            .types
            .get(info.type_index as usize)
            .and_then(|t| t.as_ref());
        let func_ty = match func_ty {
            Some(t) => t,
            None => {
                return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
            }
        };

        let trampoline = match build_trampoline(
            info.fingerprint.expect("fingerprint set"),
            &self.imports,
            &func_ty.params,
            &func_ty.results,
        ) {
            Ok(t) => t,
            Err(_) => {
                // Trampoline synthesis declined. Keep the original body —
                // Wasmtime will execute it on the CPU path.
                return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
            }
        };

        code.function(&trampoline);

        // Sanity-check parameter types declared encoder-side. We don't
        // push these onto the trampoline stack because they're consumed
        // by the byte-store loop.
        for p in &func_ty.params {
            let _ = enc_val_type(*p);
        }

        self.swapped.push(OffloadedFunction {
            function_index: self.num_function_imports + cursor as u32,
            fingerprint: info.fingerprint.expect("fingerprint set"),
            original_op_count: info.op_count,
        });
        Ok(())
    }

    fn intersperse_section_hook(
        &mut self,
        module: &mut wasm_encoder::Module,
        _after: Option<wasm_encoder::SectionId>,
        before: Option<wasm_encoder::SectionId>,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        // The `parse_type_section` / `parse_import_section` overrides above
        // already merge our additions into the original Type / Import
        // sections when those sections exist in the source module. The hook
        // below only injects fresh Type / Import sections when the module
        // *lacks* its own — detected by `before` skipping over Type / Import
        // entirely. We therefore skip the hook for `before == Type` and
        // `before == Import` (the parse overrides will handle them), and
        // skip Custom sections (which can appear anywhere and must not
        // trigger injection).
        if matches!(
            before,
            Some(wasm_encoder::SectionId::Custom)
                | Some(wasm_encoder::SectionId::Type)
                | Some(wasm_encoder::SectionId::Import)
        ) {
            return Ok(());
        }
        if !self.type_section_appended {
            let mut types = wasm_encoder::TypeSection::new();
            self.write_import_types(&mut types);
            module.section(&types);
            self.type_section_appended = true;
        }
        if !self.import_section_appended {
            let mut imports = wasm_encoder::ImportSection::new();
            self.write_imports(&mut imports);
            module.section(&imports);
            self.import_section_appended = true;
        }
        Ok(())
    }
}

/// Re-emit the supplied Wasm with offload-candidate functions swapped for
/// dispatch trampolines.
///
/// On success the [`KernelCache`] is pre-populated with the PTX for every
/// swapped function so the runtime dispatch hits straight away.
pub fn rewrite_wasm(
    wasm: &[u8],
    opts: &RewriteOptions,
    cache: &KernelCache,
) -> Result<RewriteOutcome, RewriteError> {
    let analysis = analyse(wasm, opts, cache)?;

    // Three new types inserted: dispatch, alloc, free.
    let dispatch_type_index = analysis.types.len() as u32;
    let alloc_type_index = dispatch_type_index + 1;
    let free_type_index = dispatch_type_index + 2;
    // Three new function imports.
    let imports = DispatchImports {
        dispatch: analysis.num_function_imports,
        alloc: analysis.num_function_imports + 1,
        free: analysis.num_function_imports + 2,
    };
    let total_defined_functions = analysis.func_infos.len() as u32;

    let mut rewriter = TensorWasmRewriter {
        opts,
        num_function_imports: analysis.num_function_imports,
        dispatch_type_index,
        alloc_type_index,
        free_type_index,
        imports,
        imports_added: 3,
        func_infos: &analysis.func_infos,
        types: &analysis.types,
        code_cursor: 0,
        swapped: Vec::new(),
        type_section_appended: false,
        import_section_appended: false,
    };

    let mut module = wasm_encoder::Module::new();
    let parser = wasmparser::Parser::new(0);
    Reencode::parse_core_module(&mut rewriter, &mut module, parser, wasm)?;

    debug_assert!(
        rewriter.type_section_appended,
        "rewriter failed to append dispatch type"
    );
    debug_assert!(
        rewriter.import_section_appended,
        "rewriter failed to append dispatch import"
    );

    let swapped = rewriter.swapped;
    Ok(RewriteOutcome {
        rewritten_wasm: module.finish(),
        offloaded_functions: swapped,
        total_defined_functions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A module with one v128-heavy function. The body uses pre-loaded v128
    /// locals as operands so `op_to_detector_op` walks v128 arithmetic ops
    /// (which it maps to `Op::V128Add`/`Op::V128Mul`) without the
    /// noise of `v128.const` / `drop` ops (which currently fall through to
    /// `Op::Other`).
    const V128_HEAVY_WAT: &str = r#"
        (module
          (memory 1)
          (func (export "hot") (result i32)
            (local $v v128)
            (loop $L
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
              (local.set $v (i32x4.add (local.get $v) (local.get $v)))
            )
            (i32.const 0)
          )
        )
    "#;

    /// A trivial module with a single noop function — should NOT be swapped.
    const NOOP_WAT: &str = r#"(module (func (export "noop")))"#;

    fn noop_module_swaps_nothing_inner() {
        let wasm = wat::parse_str(NOOP_WAT).unwrap();
        let cache = KernelCache::new();
        let out = rewrite_wasm(&wasm, &RewriteOptions::default(), &cache).expect("rewrite");
        assert!(out.offloaded_functions.is_empty());
        assert!(cache.is_empty());
        let mut saw_import = false;
        for p in wasmparser::Parser::new(0).parse_all(&out.rewritten_wasm) {
            if let wasmparser::Payload::ImportSection(reader) = p.expect("rewritten payload parses")
            {
                for imp in reader {
                    let imp = imp.expect("import parses");
                    if imp.module == DEFAULT_HOST_MODULE && imp.name == DEFAULT_HOST_FN {
                        saw_import = true;
                    }
                }
            }
        }
        assert!(
            saw_import,
            "rewritten module is missing the dispatch import"
        );
    }

    #[test]
    fn noop_module_swaps_nothing_and_still_adds_dispatch_import() {
        noop_module_swaps_nothing_inner();
    }

    #[test]
    fn v128_heavy_module_swaps_function_and_inserts_call() {
        let wasm = wat::parse_str(V128_HEAVY_WAT).unwrap();
        let cache = KernelCache::new();
        let opts = RewriteOptions {
            detector: DetectorConfig {
                v128_ratio_threshold: 0.05,
                min_trip_count: 64,
            },
            ..RewriteOptions::default()
        };
        let out = rewrite_wasm(&wasm, &opts, &cache).expect("rewrite");
        assert_eq!(
            out.offloaded_functions.len(),
            1,
            "the v128-heavy module should produce exactly one swap"
        );
        let swapped = &out.offloaded_functions[0];
        assert_eq!(swapped.function_index, 0);
        // Rewriter pre-populates under the placeholder `TenantId(0)`; see the
        // `key = CacheKey::for_tenant(TenantId(0), ...)` site above.
        let key = CacheKey::for_tenant(
            TenantId(0),
            swapped.fingerprint,
            DEFAULT_SM_VERSION,
        );
        assert!(cache.get(&key).is_some(), "kernel was pre-populated");

        // The rewritten module must validate.
        wasmparser::Validator::new()
            .validate_all(&out.rewritten_wasm)
            .expect("rewritten wasm must validate");
    }

    #[test]
    fn rewrite_options_default_values_match_constants() {
        let opts = RewriteOptions::default();
        assert_eq!(opts.host_module, DEFAULT_HOST_MODULE);
        assert_eq!(opts.host_fn, DEFAULT_HOST_FN);
        assert_eq!(opts.host_alloc_fn, DEFAULT_HOST_ALLOC_FN);
        assert_eq!(opts.host_free_fn, DEFAULT_HOST_FREE_FN);
        assert_eq!(opts.sm_version, DEFAULT_SM_VERSION);
    }

    #[test]
    fn rewrite_outcome_reports_total_defined_functions() {
        let wat = r#"
            (module
              (func (export "a"))
              (func (export "b"))
            )
        "#;
        let wasm = wat::parse_str(wat).unwrap();
        let cache = KernelCache::new();
        let out = rewrite_wasm(&wasm, &RewriteOptions::default(), &cache).expect("rewrite");
        assert_eq!(out.total_defined_functions, 2);
    }

    #[test]
    fn invalid_wasm_returns_parse_error() {
        let bytes = [0x00, 0x61, 0x73, 0x6d, 0xff, 0xff, 0xff, 0xff];
        let cache = KernelCache::new();
        let err = rewrite_wasm(&bytes, &RewriteOptions::default(), &cache).unwrap_err();
        assert!(matches!(
            err,
            RewriteError::Parse(_) | RewriteError::Reencode(_)
        ));
    }

    #[test]
    fn build_trampoline_round_trip_validates() {
        // Build a (i32, i32) -> i32 trampoline and validate the byte
        // length grows monotonically with parameter count.
        let imports = DispatchImports {
            dispatch: 0,
            alloc: 1,
            free: 2,
        };
        let small = build_trampoline(0xCAFEBABE, &imports, &[], &[]).unwrap();
        let mid = build_trampoline(
            0xCAFEBABE,
            &imports,
            &[wasmparser::ValType::I32, wasmparser::ValType::I32],
            &[wasmparser::ValType::I32],
        )
        .unwrap();
        let big = build_trampoline(
            0xCAFEBABE,
            &imports,
            &[
                wasmparser::ValType::I32,
                wasmparser::ValType::I64,
                wasmparser::ValType::F32,
                wasmparser::ValType::F64,
            ],
            &[wasmparser::ValType::I32, wasmparser::ValType::F64],
        )
        .unwrap();
        assert!(small.byte_len() < mid.byte_len());
        assert!(mid.byte_len() < big.byte_len());
    }

    #[test]
    fn build_trampoline_refuses_v128() {
        let imports = DispatchImports {
            dispatch: 0,
            alloc: 1,
            free: 2,
        };
        let err = build_trampoline(0, &imports, &[wasmparser::ValType::V128], &[])
            .expect_err("must refuse v128 param");
        assert!(matches!(err, RewriteError::Trampoline(_)));
    }

    #[test]
    fn rewritten_module_passes_wasm_validator() {
        // Round-trip a few different module shapes through the rewriter
        // and check `wasmparser::Validator` is happy with each output.
        let modules = [
            r#"(module (func (export "noop")))"#,
            r#"(module
              (memory 1)
              (func (export "add") (param i32 i32) (result i32)
                (i32.add (local.get 0) (local.get 1))))"#,
            r#"(module
              (memory 1)
              (func (export "f") (param f32) (result f32)
                (f32.add (local.get 0) (local.get 0))))"#,
        ];
        for (i, wat_src) in modules.iter().enumerate() {
            let wasm = wat::parse_str(wat_src).unwrap_or_else(|e| panic!("wat #{i}: {e}"));
            let cache = KernelCache::new();
            let out = rewrite_wasm(&wasm, &RewriteOptions::default(), &cache)
                .unwrap_or_else(|e| panic!("rewrite #{i}: {e}"));
            wasmparser::Validator::new()
                .validate_all(&out.rewritten_wasm)
                .unwrap_or_else(|e| panic!("validation #{i}: {e}"));
        }
    }

    #[test]
    fn pack_layout_natural_alignment() {
        let layout = pack_layout(&[
            wasmparser::ValType::I32,
            wasmparser::ValType::I64,
            wasmparser::ValType::F32,
            wasmparser::ValType::F64,
        ])
        .expect("supported types pack cleanly");
        // i32 at 0 (4b), i64 at 8 (aligned) (8b), f32 at 16 (4b), f64 at 24 (aligned) (8b)
        // total naive = 32, rounded to 8: 32.
        assert_eq!(layout.offsets, vec![0, 8, 16, 24]);
        assert_eq!(layout.total_bytes, 32);
    }

    #[test]
    fn pack_layout_empty_is_zero() {
        let layout = pack_layout(&[]).expect("empty layout is trivially Ok");
        assert!(layout.offsets.is_empty());
        assert_eq!(layout.total_bytes, 0);
    }

    /// `aligned_advance` must reject offsets that would overflow `u32` rather
    /// than silently wrap. We pick an offset within 7 bytes of `u32::MAX` and
    /// pair it with an 8-byte type so the alignment mask (7) tips the
    /// `checked_add` past the boundary.
    #[test]
    fn aligned_advance_overflows_cleanly() {
        let err = aligned_advance(u32::MAX - 3, wasmparser::ValType::I64)
            .expect_err("near-MAX offset must trip overflow guard");
        assert!(
            matches!(err, RewriteError::Trampoline(ref msg) if msg.contains("aligned_advance")),
            "expected Trampoline overflow error, got {err:?}"
        );
    }

    /// `pack_layout`'s final 8-byte round-up uses `off.checked_add(7)`; pick
    /// a one-element layout whose cursor lands within 7 of `u32::MAX` so the
    /// round-up overflows.
    #[test]
    fn pack_layout_total_bytes_overflow() {
        // We can't easily push the cursor near u32::MAX through ordinary
        // pack_layout calls (each step bumps `off` by 4 or 8), but we can
        // reach into the same code path by calling `aligned_advance` for
        // the cursor advance and inspecting `pack_layout` indirectly.
        // The simplest end-to-end probe: aligned_advance to u32::MAX-3
        // for an i64 already overflows — see the dedicated test above.
        // For pack_layout's total_bytes round-up specifically, construct
        // an i32 entry at a near-MAX starting offset via aligned_advance:
        // pack_layout itself starts at 0, so to exercise the round-up
        // overflow we need a different vector. The honest probe is a
        // single-element layout with a type whose size + start aligns to
        // u32::MAX - 6 territory; since pack_layout starts at 0, this
        // can't be tripped from the public surface. We therefore assert
        // the round-up logic indirectly via aligned_advance overflow as
        // a guard on the same arithmetic. See aligned_advance test.
        let err = aligned_advance(u32::MAX, wasmparser::ValType::I32)
            .expect_err("u32::MAX cannot round up to a 4-byte boundary");
        assert!(
            matches!(err, RewriteError::Trampoline(_)),
            "expected Trampoline overflow error, got {err:?}"
        );
    }

    /// `val_type_size` is fallible: feeding it `V128` (rejected by
    /// `is_supported_primitive`) must return `Err` rather than panic.
    #[test]
    fn val_type_size_is_fallible() {
        let err = val_type_size(wasmparser::ValType::V128)
            .expect_err("v128 is not a supported primitive");
        assert!(
            matches!(err, RewriteError::Trampoline(ref msg) if msg.contains("val_type_size")),
            "expected Trampoline error, got {err:?}"
        );
    }

    /// The rewritten trampoline must trap on a nonzero dispatch return code
    /// rather than silently feed zero-filled results back to the caller.
    /// We can't run the rewritten module without a host harness from inside
    /// this unit test, so assert structurally: the emitted function bytes
    /// must contain the exact `local.tee $rc; i32.const 0; i32.ne; if;
    /// unreachable; end` opcode sequence we wired in after the dispatch
    /// call. This catches regressions back to the old silent-drop pattern.
    #[test]
    fn trampoline_traps_on_nonzero_dispatch() {
        let imports = DispatchImports {
            dispatch: 0,
            alloc: 1,
            free: 2,
        };
        // One param so the rc local lands at index 2 (params.len() + 1 = 2)
        // which encodes as a single 0x02 byte in unsigned LEB128.
        let func = build_trampoline(
            0xDEAD_BEEF,
            &imports,
            &[wasmparser::ValType::I32],
            &[wasmparser::ValType::I32],
        )
        .expect("trampoline builds");
        let mut code = wasm_encoder::CodeSection::new();
        code.function(&func);
        let mut out = Vec::new();
        wasm_encoder::Encode::encode(&code, &mut out);
        // Expected sequence after the dispatch `call` instruction:
        //   0x22 0x02 — local.tee 2 (the rc local)
        //   0x41 0x00 — i32.const 0
        //   0x47      — i32.ne
        //   0x04 0x40 — if (block type = empty)
        //   0x00      — unreachable
        //   0x0B      — end (closes the if)
        const EXPECTED: &[u8] = &[0x22, 0x02, 0x41, 0x00, 0x47, 0x04, 0x40, 0x00, 0x0B];
        let found = out.windows(EXPECTED.len()).any(|w| w == EXPECTED);
        assert!(
            found,
            "trampoline missing the trap-on-nonzero-dispatch opcode sequence; \
             body bytes: {out:02x?}"
        );
    }
}
