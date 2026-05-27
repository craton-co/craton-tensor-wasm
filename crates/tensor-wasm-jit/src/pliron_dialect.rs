// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Cranelift IR → Pliron `dialect-mir` lowering — **v0.3.1 scaffold only**.
//!
//! This module is the empty-stub landing site for the Pliron-based
//! auto-offload pipeline tracked in [RFC
//! 0001](../../../../rfcs/0001-cuda-oxide-integration.md) ("Pliron lever and
//! the auto-offload pipeline" / "Future possibilities"). It is gated behind
//! the opt-in `cuda-oxide-backend` feature because the parity port has not
//! landed yet; the workspace toolchain pin (`nightly-2026-04-03`) already
//! matches cuda-oxide's own pin, so enabling the feature on the default
//! workspace toolchain builds.
//!
//! # What this module IS
//!
//! - The final trait signature ([`WasmToPliron`]) that the v0.4+ port will
//!   implement. The string-to-string shape is intentional placeholder: the
//!   real implementation replaces the input/output types with Pliron's
//!   `Operation` / `Module` types once the crate dep lands. Call sites that
//!   want to write backend-agnostic lowering code today can target this
//!   trait and the v0.4 diff is a body + types swap, not a refactor.
//! - A [`StubLowerer`] that implements the trait, always returning
//!   [`PlironLoweringError::NotYetImplemented`] with an actionable message
//!   pointing to RFC 0001.
//! - A convenience entry point [`cranelift_to_dialect_mir`] that wraps the
//!   stub, mirroring the shape of the eventual public free function so the
//!   detector → lowering call site is stable across the port.
//! - The [Cranelift IR ⇄ dialect-mir mapping table](#mapping-table) below.
//!   This is the most load-bearing artifact in the file: the v0.4 author
//!   should be able to walk this table top-to-bottom and produce the real
//!   lowering without re-deriving which Cranelift opcodes need device-side
//!   semantics and which are PTX-illegal.
//!
//! # What this module is NOT (yet)
//!
//! - The real lowering implementation. That is the v0.4+ research
//!   investment per RFC 0001 step 4. The cuda-oxide compiler pipeline is
//!   `Rust → rustc_public (Stable MIR) → dialect-mir → mem2reg →
//!   dialect-llvm → LLVM IR → PTX`; this module would graft a fourth
//!   front-end (`Wasm → Cranelift IR → dialect-mir`) onto the same
//!   pipeline, so we get arbitrary pure-compute auto-offload coverage
//!   instead of the three hand-written blueprints today's
//!   [`crate::detector`] recognises (matmul, vector_add, conv2d 3x3).
//! - A binding to the Pliron crate. Pliron is pinned to a `git` rev in
//!   cuda-oxide's `Cargo.toml` and not yet on crates.io — see RFC 0001
//!   "Drawbacks". Adding a `git`-pinned dep today would import the
//!   reproducible-builds (W3.6) and `cargo-deny` allowlist burden into
//!   `tensor-wasm-jit` before any code uses it; deferring is the right
//!   call.
//! - A `dialect-llvm` or `mem2reg` pass binding. Those live further down
//!   the cuda-oxide pipeline; this module's contract ends at `dialect-mir`.
//!
//! # Why it exists today
//!
//! So the v0.4 author has (a) a concrete target signature to plug into,
//! (b) a documented mapping table that captures the load-bearing op-by-op
//! decisions, and (c) a unit-test harness asserting trait-object
//! Send+Sync — instead of staring at a blank file and re-deriving the
//! design from RFC 0001 + the cuda-oxide architecture overview.
//!
//! # References
//!
//! - [RFC 0001 — cuda-oxide as the v0.5 cust successor](../../../../rfcs/0001-cuda-oxide-integration.md)
//! - cuda-oxide architecture overview: <https://nvlabs.github.io/cuda-oxide/>
//! - Pliron upstream: <https://github.com/vaivaswatha/pliron>
//!
//! # Mapping table
//!
//! Cranelift opcodes the v0.4 author needs to handle, paired with their
//! Pliron `dialect-mir` equivalents. The table is **load-bearing**: this is
//! the design decision that should not be re-derived in the v0.4 PR.
//!
//! | Cranelift op           | Pliron `dialect-mir` op       | Notes |
//! |------------------------|-------------------------------|-------|
//! | `iadd`                 | `arith.addi`                  | Integer add. Width carried in the result type. |
//! | `isub`                 | `arith.subi`                  | Integer subtract. |
//! | `imul`                 | `arith.muli`                  | Integer multiply. Overflow semantics inherit from Cranelift (wrap on signed/unsigned). |
//! | `idiv` / `udiv`        | `arith.divsi` / `arith.divui` | Signed vs unsigned chosen by Cranelift opcode discriminant. PTX `div` is well-defined on Ampere+. |
//! | `fadd`                 | `arith.addf`                  | IEEE-754 float add. |
//! | `fsub`                 | `arith.subf`                  | Float subtract. |
//! | `fmul`                 | `arith.mulf`                  | Float multiply. |
//! | `fdiv`                 | `arith.divf`                  | Float divide. PTX `div.rn.f32` is the default rounding mode. |
//! | `fma`                  | `arith.fma`                   | Fused multiply-add. Maps directly to PTX `fma.rn`. Mandatory: emitting `mul`+`add` instead loses the FMA-rounding contract Cranelift guarantees. |
//! | `load`                 | `memref.load`                 | **Device-pointer translation** required: the Cranelift `load` operates on a guest linear-memory offset; the dialect-mir `memref.load` operates on a device pointer. The v0.4 port must thread the W1.1 kernel-args base-pointer through here. |
//! | `store`                | `memref.store`                | Same device-pointer translation as `load`. |
//! | `stack_load`           | `memref.load` + SSA local     | Cranelift stack slots become SSA-local `memref` allocas in dialect-mir. PTX has no stack — the lowering must promote to registers via the downstream `mem2reg` pass. |
//! | `stack_store`          | `memref.store` + SSA local    | Symmetric with `stack_load`. |
//! | `select`               | `arith.select`                | Three-operand select. Maps to PTX `selp`. |
//! | `br_table`             | `cf.switch`                   | Multi-way branch. PTX has no native switch — dialect-llvm lowers this to a sequence of `setp` + `bra`. |
//! | `jump`                 | `cf.br`                       | Unconditional branch. |
//! | `brif` / `brz` / `brnz`| `cf.cond_br`                  | Conditional branch with two successors. |
//! | `call`                 | `func.call`                   | **Device-function calls only.** Calls into other dialect-mir-lowered kernels are legal; calls into host functions (anything reaching back into the Wasmtime runtime) are **forbidden** in PTX — the detector must reject candidates that contain host calls before reaching this lowering. |
//! | `call_indirect`        | `func.call_indirect`          | Same host-call prohibition. PTX function pointers exist (sm_70+) but are expensive; the v0.4 port should consider devirtualising before lowering. |
//! | `bitcast`              | `arith.bitcast`               | Reinterpret bits. Free in PTX — produces no instruction. |
//! | `breduce`              | `arith.trunci`                | Width-reducing integer truncation. Maps to PTX `cvt.<small>.<large>`. |
//! | `bextend` / `sextend`  | `arith.extui` / `arith.extsi` | Width-extending. Sign-vs-zero extend chosen by Cranelift opcode. |
//! | `vmin`                 | `vector.minimum`              | SIMD per-lane min. Maps to per-warp-lane PTX (one lane per thread within a 32-thread warp). |
//! | `vmax`                 | `vector.maximum`              | SIMD per-lane max. Same warp-lane mapping. |
//! | `vsplat`               | `vector.splat`                | Broadcast scalar to all lanes. PTX has no native splat — the lowering writes the scalar to every lane via the warp-shuffle intrinsic. |
//! | `vselect`              | `vector.select`               | Per-lane mask select. Maps to PTX `selp.b32` per lane. |
//! | `vall_true` / `vany_true` | `vector.reduce_and/or`     | Warp-wide reduction. Maps to PTX `vote.all` / `vote.any`. |
//!
//! ## Unsupported in v0.4 (deferred or hard-rejected)
//!
//! These Cranelift opcodes have no dialect-mir lowering planned; the
//! detector must filter candidates containing them before invoking this
//! pipeline. Documented here so the v0.4 author does not waste cycles
//! mapping them:
//!
//! - **Atomics** (`atomic_load`, `atomic_store`, `atomic_rmw`,
//!   `atomic_cas`). PTX has atomics, but Wasm threads + GPU atomics is a
//!   memory-model alignment problem larger than the v0.4 scope. Deferred
//!   to a future RFC.
//! - **Floating-point exception bits** (`fcvt_to_sint_sat` traps,
//!   `f32const NaN` propagation guarantees beyond IEEE-754 default). PTX
//!   default rounding + flush-to-zero diverges from Wasm-strict FP. The
//!   v0.4 detector should reject candidates that depend on strict-FP.
//! - **`table.get` / `table.set`** (Wasm reference types). Tables live
//!   host-side; lowering them would require device-resident table mirrors,
//!   which is out of scope.
//! - **`ref.func` / GC ops** (Wasm GC proposal). No device-side
//!   representation; hard-rejected.
//! - **`memory.grow` / `memory.size`**. Linear-memory resizing requires a
//!   host round-trip; the kernel must run with a fixed memory snapshot.
//! - **`memory.copy` / `memory.fill` >4 KiB**. PTX has `cp.async.bulk`
//!   (sm_90+) but the v0.4 port targets sm_80 baseline (per
//!   [`ptx_emit::DEFAULT_TARGET`](crate::ptx_emit::DEFAULT_TARGET)). Small
//!   copies inline; large copies fall back to a host bounce.

#![cfg(feature = "cuda-oxide-backend")]

use thiserror::Error;

/// Errors produced by the (still-scaffolded) Cranelift → Pliron
/// `dialect-mir` lowering.
///
/// Every variant carries enough context for the v0.4 author to grep for
/// the call site and replace the stub with the real lowering. The
/// `NotYetImplemented` variant is the one every scaffold call site returns
/// today; `UnsupportedOp` and `DialectVersionMismatch` are pre-declared so
/// the public error surface is stable across the v0.4 port.
#[derive(Debug, Error)]
pub enum PlironLoweringError {
    /// The pipeline is scaffold-only. The `&'static str` payload names the
    /// specific lowering pass or entry point the caller hit, so the v0.4
    /// author can land each pass independently without juggling distinct
    /// error variants per stub.
    #[error("pliron_dialect: not yet implemented ({0}) -- see RFC 0001 v0.4 port")]
    NotYetImplemented(&'static str),

    /// The lowering encountered a Cranelift opcode it does not know how to
    /// translate to `dialect-mir`. Populated with the opcode mnemonic so
    /// the error is grep-able against the [mapping
    /// table](crate::pliron_dialect#mapping-table). Pre-declared today; no
    /// scaffold path returns it.
    #[error("pliron_dialect: unsupported Cranelift opcode `{op}` -- see mapping table")]
    UnsupportedOp {
        /// Mnemonic of the unsupported Cranelift opcode (e.g. `"vall_true"`).
        op: String,
    },

    /// The Pliron dialect version pinned by `tensor-wasm-jit` does not
    /// match the version actually loaded at runtime. Pre-declared today
    /// because Pliron is `git`-pinned (RFC 0001 "Drawbacks") and the v0.4
    /// port will need a version-check entry point as part of the dep
    /// landing. No scaffold path returns it.
    DialectVersionMismatch {
        /// Pinned dialect-mir version that `tensor-wasm-jit` was built
        /// against (e.g. `"dialect-mir@0.1.0"`).
        expected: &'static str,

        /// Dialect-mir version reported by the loaded Pliron runtime.
        found: String,
    },
}

// Manual Display for DialectVersionMismatch so the error message stays on
// one line in tracing output — `thiserror`'s `#[error("...")]` attribute
// supports field interpolation, but spelling the format out keeps the
// struct-field rustdoc clean.
impl PlironLoweringError {
    /// True if this error is a "wait for the v0.4 port" stub error rather
    /// than a real downstream failure. Convenience for callers that want
    /// to silently fall back to the blueprint detector while the Pliron
    /// path is still scaffolded.
    pub fn is_scaffold_stub(&self) -> bool {
        matches!(self, Self::NotYetImplemented(_))
    }
}

/// Lowering entry point trait.
///
/// Implementors translate a Cranelift IR module (today: serialised as a
/// `&str` placeholder; v0.4: replaced with the real Cranelift module
/// type and a Pliron `Module` return type) into Pliron `dialect-mir`
/// suitable for handing off to cuda-oxide's downstream `mem2reg` →
/// `dialect-llvm` → LLVM IR → PTX pipeline.
///
/// The string-to-string signature is intentional **placeholder** for the
/// v0.3.1 scaffold landing. The v0.4 port will replace the parameter
/// types with the real Pliron `Operation`/`Module` types once the crate
/// dep is added — see RFC 0001 step 4 and the module-level docs for why
/// the dep is deferred today.
///
/// # Send + Sync
///
/// Implementations must be `Send + Sync` so they can flow through the
/// Tokio-backed dispatch path
/// ([`crate::deopt::DeoptGuard`]-adjacent call sites and the eventual
/// `cuda-async`-based future-sync replacement RFC 0001 mentions under
/// "Future possibilities"). The trait itself does not enforce the bound
/// — Rust adds it implicitly for `dyn WasmToPliron + Send + Sync` — but
/// the unit tests in this module include a compile-time assertion that
/// [`StubLowerer`] satisfies it.
pub trait WasmToPliron {
    /// Lower a Cranelift IR module to a Pliron `dialect-mir` module.
    ///
    /// **v0.3.1 contract:** the parameter is a `&str` placeholder for the
    /// eventual Cranelift module type; the success return is a `String`
    /// placeholder for the eventual Pliron `Module`. Today every
    /// implementation returns
    /// [`PlironLoweringError::NotYetImplemented`].
    fn lower(&self, cranelift_module: &str) -> Result<String, PlironLoweringError>;
}

/// Zero-state stub implementor of [`WasmToPliron`].
///
/// **Scaffold only.** Always returns
/// [`PlironLoweringError::NotYetImplemented`] with the sentinel
/// `"StubLowerer::lower"` tag. The v0.4 port will either replace this
/// type with the real lowering or keep it for testing as a known-failing
/// implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubLowerer;

impl WasmToPliron for StubLowerer {
    fn lower(&self, _cranelift_module: &str) -> Result<String, PlironLoweringError> {
        // Intentionally swallow the module argument — the v0.4 port will
        // parse it. Keeping the parameter name in the trait so the
        // rustdoc surface is final.
        Err(PlironLoweringError::NotYetImplemented("StubLowerer::lower"))
    }
}

/// Convenience entry point: lower a Cranelift IR module to Pliron
/// `dialect-mir` via the default [`StubLowerer`].
///
/// **Scaffold only.** Always returns
/// `Err(PlironLoweringError::NotYetImplemented("cranelift_to_dialect_mir"))`.
/// The v0.4 port will rewrite the body to construct the real lowering
/// pipeline; the signature is intentionally already final so the
/// detector → lowering call site is stable across the port.
pub fn cranelift_to_dialect_mir(_module: &str) -> Result<String, PlironLoweringError> {
    // The dedicated `cranelift_to_dialect_mir` sentinel (rather than
    // delegating to StubLowerer) keeps the error tag grep-able to the
    // entry point — useful when triaging which surface the caller hit.
    Err(PlironLoweringError::NotYetImplemented(
        "cranelift_to_dialect_mir",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stub `lower` always returns the documented sentinel variant.
    /// The v0.4 port deletes this test (or rewrites it to assert a
    /// successful lowering against a fixture Cranelift module).
    #[test]
    fn stub_lowerer_returns_not_yet_implemented() {
        let lowerer = StubLowerer;
        let err = lowerer
            .lower("function %trivial() {}")
            .expect_err("scaffold must error");
        match err {
            PlironLoweringError::NotYetImplemented(tag) => {
                assert_eq!(tag, "StubLowerer::lower");
            }
            other => panic!(
                "expected PlironLoweringError::NotYetImplemented, got {other:?}"
            ),
        }
    }

    /// The convenience entry point produces its own grep-able sentinel
    /// distinct from the trait stub.
    #[test]
    fn cranelift_to_dialect_mir_returns_not_yet_implemented() {
        let err = cranelift_to_dialect_mir("function %trivial() {}")
            .expect_err("scaffold must error");
        match err {
            PlironLoweringError::NotYetImplemented(tag) => {
                assert_eq!(tag, "cranelift_to_dialect_mir");
            }
            other => panic!(
                "expected PlironLoweringError::NotYetImplemented, got {other:?}"
            ),
        }
    }

    /// `is_scaffold_stub` is true for the stub variant and false for the
    /// pre-declared (currently unreachable) variants. Locks in the
    /// classification helper before the v0.4 port adds the first real
    /// non-stub error path.
    #[test]
    fn is_scaffold_stub_classifies_variants() {
        assert!(PlironLoweringError::NotYetImplemented("anything").is_scaffold_stub());
        assert!(!PlironLoweringError::UnsupportedOp {
            op: "atomic_rmw".into(),
        }
        .is_scaffold_stub());
        assert!(!PlironLoweringError::DialectVersionMismatch {
            expected: "dialect-mir@0.1.0",
            found: "dialect-mir@0.2.0".into(),
        }
        .is_scaffold_stub());
    }

    /// `PlironLoweringError` must round-trip through `Debug` (used by
    /// every `tracing::error!` and `expect`/`unwrap` site). Locks in the
    /// derive today so the v0.4 port cannot accidentally drop it when
    /// extending the enum.
    #[test]
    fn error_round_trips_debug() {
        let err = PlironLoweringError::NotYetImplemented("trip");
        let dbg = format!("{err:?}");
        assert!(
            dbg.contains("NotYetImplemented"),
            "Debug output missing variant name: {dbg}"
        );
        assert!(
            dbg.contains("trip"),
            "Debug output missing payload tag: {dbg}"
        );

        let unsupported = PlironLoweringError::UnsupportedOp {
            op: "atomic_rmw".into(),
        };
        let dbg = format!("{unsupported:?}");
        assert!(dbg.contains("UnsupportedOp"));
        assert!(dbg.contains("atomic_rmw"));

        let mismatch = PlironLoweringError::DialectVersionMismatch {
            expected: "dialect-mir@0.1.0",
            found: "dialect-mir@0.2.0".into(),
        };
        let dbg = format!("{mismatch:?}");
        assert!(dbg.contains("DialectVersionMismatch"));
        assert!(dbg.contains("0.1.0"));
        assert!(dbg.contains("0.2.0"));
    }

    /// Compile-time assertion: the trait, the stub implementor, and the
    /// error type must all be `Send + Sync` so they can flow through the
    /// Tokio-backed dispatch path the v0.4 port will wire. A change that
    /// silently introduces a `Rc<...>` field will fail to compile this
    /// test.
    #[test]
    fn trait_and_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StubLowerer>();
        assert_send_sync::<PlironLoweringError>();
        // `dyn WasmToPliron + Send + Sync` is the shape the dispatch path
        // will hold; assert the unsized version is constructible.
        fn assert_dyn_compatible<T: ?Sized + Send + Sync>() {}
        assert_dyn_compatible::<dyn WasmToPliron + Send + Sync>();
    }
}
