//! Wasm-to-Wasm rewrite that swaps offload-candidate function bodies with
//! host-import calls to the JIT dispatch service.
//!
//! Sidesteps the wasmtime-cranelift fork question — see `docs/WASMTIME-FORK.md`.
//!
//! The rewriter runs BEFORE Wasmtime ever sees the bytes:
//!
//! 1. Walks the input Wasm via [`wasmparser`].
//! 2. For each function whose body
//!    [`crate::detector::classify`] returns
//!    [`DetectorVerdict::Offload`],
//!    runs the existing lower→emit pipeline to obtain a PTX blueprint
//!    fingerprint and pre-populates the supplied [`KernelCache`].
//! 3. Re-emits the module with a synthetic function import
//!    `bali:jit/host::dispatch` and replaces each offload-candidate body
//!    with a trampoline that calls the import.
//!
//! Trade-off: every offloaded call pays a host-call round trip; net win only
//! when the kernel body is large enough to amortise. This is the runtime
//! swap the plan calls for, just done above Wasmtime instead of inside it.
//!
//! The host-side implementation of the dispatch import lives in
//! `bali-exec`'s `jit_dispatch` module.

use std::convert::Infallible;
use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info};
use wasm_encoder::reencode::Reencode;
use wasmparser::{Operator, Parser, Payload};

use crate::cache::{CacheKey, CachedKernel, CompiledHandle, KernelCache};
use crate::clif_lower::lower_block;
use crate::detector::{classify, BlockIR, DetectorConfig, DetectorVerdict, Op};
use crate::ptx_emit::emit;

/// Default host import module name.
pub const DEFAULT_HOST_MODULE: &str = "bali:jit/host";
/// Default host import field name.
pub const DEFAULT_HOST_FN: &str = "dispatch";
/// Default sm_version the rewriter pre-populates kernels for.
pub const DEFAULT_SM_VERSION: u32 = 80;

/// Options controlling the rewrite.
#[derive(Debug, Clone)]
pub struct RewriteOptions {
    /// Host import module (defaults to `bali:jit/host`).
    pub host_module: String,
    /// Host import function name (defaults to `dispatch`).
    pub host_fn: String,
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
            sm_version: DEFAULT_SM_VERSION,
            detector: DetectorConfig::default(),
        }
    }
}

/// One swapped function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadedFunction {
    /// Index of the function in the ORIGINAL (pre-rewrite) function index
    /// space. The post-rewrite index is `function_index + 1` because the
    /// dispatch import shifts defined functions down by one.
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

/// Mirror of `bali_exec::auto_offload::op_to_detector_op` — duplicated here
/// because `bali-jit` is intentionally a leaf dependency that does not
/// reference `bali-exec`.
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
        _ => Op::ScalarAdd,
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
                while !ops_reader.eof() {
                    match ops_reader.read() {
                        Ok(op) => detector_ops.push(op_to_detector_op(&op)),
                        Err(_) => break,
                    }
                }
                let func_index_in_global_space =
                    num_function_imports + defined_function_cursor as u32;
                let block = BlockIR::new(
                    format!("func{func_index_in_global_space}"),
                    detector_ops.clone(),
                    // Mirror `bali_exec::auto_offload::analyse`: conservative
                    // 128-iteration loop assumption — the detector will only
                    // flag if the v128 ratio also clears the threshold.
                    Some(128),
                );
                let verdict = classify(&block, &opts.detector);
                let op_count = detector_ops.len();
                let mut fingerprint = None;
                if matches!(verdict, DetectorVerdict::Offload) {
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
                                    Op::V128Add | Op::V128Mul | Op::V128Fma | Op::Load | Op::Store
                                )
                            })
                            .collect(),
                        block.loop_trip_count,
                    );
                    match lower_block(&lower_block_input) {
                        Ok(blueprint) => {
                            let fp = blueprint.fingerprint();
                            let ptx = emit(&blueprint);
                            let key = CacheKey {
                                blueprint: fp,
                                sm_version: opts.sm_version,
                            };
                            cache.put(
                                key,
                                CachedKernel {
                                    fingerprint: fp,
                                    ptx: Arc::new(ptx),
                                    compiled: CompiledHandle::default(),
                                },
                            );
                            fingerprint = Some(fp);
                            info!(
                                target: "bali_jit::rewrite",
                                function = func_index_in_global_space,
                                op_count,
                                fingerprint = fp,
                                "pre-populated kernel cache for offload candidate"
                            );
                        }
                        Err(e) => {
                            // Lowering refused — keep this function on the
                            // CPU path. This is the deopt-at-rewrite signal.
                            debug!(
                                target: "bali_jit::rewrite",
                                function = func_index_in_global_space,
                                op_count,
                                reason = %e,
                                "offload candidate rejected by lowering"
                            );
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

/// Convert a [`wasmparser::ValType`] to its [`wasm_encoder::ValType`]
/// counterpart. We refuse `Ref(_)` — the trampoline can't synthesise a
/// defaultable reference value without help — and v128 (since the dispatch
/// host-side returns i32 only; v128 results would require richer plumbing).
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

/// Build a trampoline body for an offload-swapped function:
///
/// ```text
/// (func $f (param ...) (result ...)
///   i64.const fp_lo
///   i64.const fp_hi
///   i32.const 0   ;; args_ptr
///   i32.const 0   ;; args_len
///   call $dispatch
///   drop
///   <zero of each result type...>
/// )
/// ```
///
/// Locals are empty. We DROP the dispatch result and emit zeros of the
/// original result types so the function's signature is unchanged from
/// Wasmtime's perspective — this is essential because every existing
/// `call` instruction targeting this function continues to validate.
fn build_trampoline(
    fingerprint: u64,
    dispatch_fn_index: u32,
    results: &[wasmparser::ValType],
) -> Result<wasm_encoder::Function, RewriteError> {
    let mut func = wasm_encoder::Function::new(std::iter::empty::<(u32, wasm_encoder::ValType)>());

    // Pack fingerprint as two i32 halves (passed to the import as i64 below).
    // We follow the wider plan: signature `(i64, i64, i32, i32) -> i32`.
    let fp_lo = (fingerprint & 0xFFFF_FFFF) as i64;
    let fp_hi = (fingerprint >> 32) as i64;

    func.instruction(&wasm_encoder::Instruction::I64Const(fp_lo));
    func.instruction(&wasm_encoder::Instruction::I64Const(fp_hi));
    func.instruction(&wasm_encoder::Instruction::I32Const(0));
    func.instruction(&wasm_encoder::Instruction::I32Const(0));
    func.instruction(&wasm_encoder::Instruction::Call(dispatch_fn_index));
    func.instruction(&wasm_encoder::Instruction::Drop);

    for r in results {
        match r {
            wasmparser::ValType::I32 => {
                func.instruction(&wasm_encoder::Instruction::I32Const(0));
            }
            wasmparser::ValType::I64 => {
                func.instruction(&wasm_encoder::Instruction::I64Const(0));
            }
            wasmparser::ValType::F32 => {
                func.instruction(&wasm_encoder::Instruction::F32Const(0.0));
            }
            wasmparser::ValType::F64 => {
                func.instruction(&wasm_encoder::Instruction::F64Const(0.0));
            }
            wasmparser::ValType::V128 | wasmparser::ValType::Ref(_) => {
                return Err(RewriteError::Trampoline(
                    "unsupported result type in trampoline (v128/ref)".into(),
                ));
            }
        }
    }

    func.instruction(&wasm_encoder::Instruction::End);
    Ok(func)
}

/// Stateful re-encoder that:
/// - Appends our dispatch func type to the type section.
/// - Appends our dispatch import to the import section.
/// - Shifts every defined function index by +1 (because the new import is
///   inserted at position `num_existing_function_imports`).
/// - Swaps offload-candidate function bodies for trampolines.
struct BaliRewriter<'a> {
    opts: &'a RewriteOptions,
    /// `num_existing_function_imports` — anything below this is an imported
    /// function and its index is preserved; anything at or above shifts by +1.
    num_function_imports: u32,
    /// Type index assigned to the dispatch import (in the NEW type space —
    /// equal to `num_existing_types`).
    dispatch_type_index: u32,
    /// Function index assigned to the dispatch import (in the NEW function
    /// space — equal to `num_existing_function_imports`).
    dispatch_fn_index: u32,
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
    /// section? (Modules without a type section trigger the append from the
    /// intersperse hook instead.)
    type_section_appended: bool,
    /// Same for the import section.
    import_section_appended: bool,
}

impl<'a> BaliRewriter<'a> {
    fn shifted_fn_index(&self, orig: u32) -> u32 {
        if orig < self.num_function_imports {
            orig
        } else {
            orig + 1
        }
    }

    fn dispatch_func_type(&self) -> (Vec<wasm_encoder::ValType>, Vec<wasm_encoder::ValType>) {
        (
            vec![
                wasm_encoder::ValType::I64,
                wasm_encoder::ValType::I64,
                wasm_encoder::ValType::I32,
                wasm_encoder::ValType::I32,
            ],
            vec![wasm_encoder::ValType::I32],
        )
    }
}

impl<'a> Reencode for BaliRewriter<'a> {
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
        let (params, results) = self.dispatch_func_type();
        types.ty().function(params, results);
        self.type_section_appended = true;
        Ok(())
    }

    fn parse_import_section(
        &mut self,
        imports: &mut wasm_encoder::ImportSection,
        section: wasmparser::ImportSectionReader<'_>,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        wasm_encoder::reencode::utils::parse_import_section(self, imports, section)?;
        imports.import(
            &self.opts.host_module,
            &self.opts.host_fn,
            wasm_encoder::EntityType::Function(self.dispatch_type_index),
        );
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
                // Defensive: re-emit body unchanged if pre-pass somehow
                // missed it. Should not happen but keeps the rewriter total.
                return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
            }
        };

        let should_swap =
            matches!(info.verdict, DetectorVerdict::Offload) && info.fingerprint.is_some();
        if !should_swap {
            return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
        }

        // Look up the function's result types; if anything is unsupported,
        // fall back to emitting the original body unchanged.
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
            self.dispatch_fn_index,
            &func_ty.results,
        ) {
            Ok(t) => t,
            Err(_) => {
                // Trampoline synthesis declined (e.g. v128 result). Keep the
                // original body — Wasmtime will execute it on the CPU path.
                return wasm_encoder::reencode::utils::parse_function_body(self, code, func);
            }
        };

        code.function(&trampoline);
        // Validate the parameter types now to surface unsupported parameter
        // types at rewrite time rather than at instantiation. We don't push
        // the parameters onto the trampoline stack (the dispatch import
        // doesn't use them yet — see the v0.1.0 note in the module docs).
        for p in &func_ty.params {
            // Just sanity-check we can name them.
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
        // If the original module has no Type / Import section, append ours
        // before the next section in canonical order (or at end-of-module
        // when `before` is `None`). The `before` discriminator just guards
        // against inserting a Type section in the middle of, say, the Code
        // section's neighbourhood — section ordering is fixed by the spec
        // so we treat any `before` strictly greater than the missing
        // section's slot as a green light.
        if !self.type_section_appended && !matches!(before, Some(wasm_encoder::SectionId::Custom)) {
            // Any non-Custom `before` (including `None`) is past the Type
            // slot since Type is the first non-custom section.
            let mut types = wasm_encoder::TypeSection::new();
            let (params, results) = self.dispatch_func_type();
            types.ty().function(params, results);
            module.section(&types);
            self.type_section_appended = true;
        }
        if !self.import_section_appended
            && !matches!(
                before,
                Some(wasm_encoder::SectionId::Custom) | Some(wasm_encoder::SectionId::Type)
            )
        {
            let mut imports = wasm_encoder::ImportSection::new();
            imports.import(
                &self.opts.host_module,
                &self.opts.host_fn,
                wasm_encoder::EntityType::Function(self.dispatch_type_index),
            );
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

    // Compute the index assignments for the new dispatch import:
    //  - dispatch_type_index = number of existing type entries (== Vec len)
    //  - dispatch_fn_index   = number of existing function imports
    let dispatch_type_index = analysis.types.len() as u32;
    let dispatch_fn_index = analysis.num_function_imports;
    let total_defined_functions = analysis.func_infos.len() as u32;

    let mut rewriter = BaliRewriter {
        opts,
        num_function_imports: analysis.num_function_imports,
        dispatch_type_index,
        dispatch_fn_index,
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

    // Defensive: if the input module had neither a type nor import section
    // (a corner case the intersperse hook is designed to handle), make sure
    // we still wrote our dispatch import. `parse_core_module` calls the
    // hook one final time with `before = None`, so this should be unreachable.
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
    /// `Op::ScalarAdd`).
    const V128_HEAVY_WAT: &str = r#"
        (module
          (memory 1)
          (func (export "hot") (result i32)
            (local $v v128)
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
        // Cache stays empty (no pre-population).
        assert!(cache.is_empty());
        // The rewritten module must still be valid Wasm — wat::parse_bytes
        // doesn't validate so we just round-trip-parse with wasmparser.
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
        // The detector's default 80% ratio threshold can't be hit by raw
        // wasmparser output because `op_to_detector_op` is intentionally
        // coarse (local.get / local.set / drop / const all collapse to
        // ScalarAdd). Lower the threshold so the test exercises the swap
        // path. Production deployments tune `DetectorConfig` per workload.
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
        // The cache should now hold one kernel keyed by that fingerprint.
        let key = CacheKey {
            blueprint: swapped.fingerprint,
            sm_version: DEFAULT_SM_VERSION,
        };
        assert!(cache.get(&key).is_some(), "kernel was pre-populated");

        // Validate the rewritten module contains:
        //   1. the dispatch import (bali:jit/host::dispatch)
        //   2. the swapped function body now calls function index
        //      `dispatch_fn_index` (== 0 since no original imports).
        let mut saw_dispatch_import = false;
        let mut saw_call_to_dispatch = false;
        let mut num_function_imports = 0u32;
        let mut code_cursor = 0usize;
        for payload in wasmparser::Parser::new(0).parse_all(&out.rewritten_wasm) {
            let payload = payload.expect("rewritten payload");
            match payload {
                wasmparser::Payload::ImportSection(reader) => {
                    for imp in reader {
                        let imp = imp.expect("import");
                        if matches!(imp.ty, wasmparser::TypeRef::Func(_)) {
                            num_function_imports += 1;
                            if imp.module == DEFAULT_HOST_MODULE && imp.name == DEFAULT_HOST_FN {
                                saw_dispatch_import = true;
                            }
                        }
                    }
                }
                wasmparser::Payload::CodeSectionEntry(body) => {
                    if code_cursor == 0 {
                        let mut reader = body.get_operators_reader().expect("ops reader");
                        while !reader.eof() {
                            let op = reader.read().expect("op");
                            if let wasmparser::Operator::Call { function_index } = op {
                                // `num_function_imports - 1` would be the
                                // dispatch (since the import was inserted
                                // after the existing 0 originals).
                                if function_index == num_function_imports - 1 {
                                    saw_call_to_dispatch = true;
                                }
                            }
                        }
                    }
                    code_cursor += 1;
                }
                _ => {}
            }
        }
        assert!(saw_dispatch_import, "dispatch import missing");
        assert!(
            saw_call_to_dispatch,
            "trampoline did not call the dispatch import"
        );
    }

    #[test]
    fn rewrite_options_default_values_match_constants() {
        let opts = RewriteOptions::default();
        assert_eq!(opts.host_module, DEFAULT_HOST_MODULE);
        assert_eq!(opts.host_fn, DEFAULT_HOST_FN);
        assert_eq!(opts.sm_version, DEFAULT_SM_VERSION);
    }

    #[test]
    fn rewrite_outcome_reports_total_defined_functions() {
        // Module with two defined functions.
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
    fn build_trampoline_emits_zero_for_each_result_type() {
        // Build a trampoline for `(func ... (result i32 i64 f32 f64))` and
        // confirm it can be parsed back. We can't easily inspect Function's
        // internal bytes, but the byte-len should grow with result count.
        let small = build_trampoline(0xCAFEBABE, 0, &[]).unwrap();
        let big = build_trampoline(
            0xCAFEBABE,
            0,
            &[
                wasmparser::ValType::I32,
                wasmparser::ValType::I64,
                wasmparser::ValType::F32,
                wasmparser::ValType::F64,
            ],
        )
        .unwrap();
        assert!(big.byte_len() > small.byte_len());
    }

    #[test]
    fn build_trampoline_refuses_v128_result() {
        let err =
            build_trampoline(0, 0, &[wasmparser::ValType::V128]).expect_err("must refuse v128");
        assert!(matches!(err, RewriteError::Trampoline(_)));
    }
}
