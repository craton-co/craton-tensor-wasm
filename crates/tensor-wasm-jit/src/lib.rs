// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! JIT pipeline: Cranelift detector, IR normalisation, PTX codegen, kernel cache, deopt.
//!
//! # Unstable surface — `pliron_*` modules
//!
//! The `pliron_dialect`, `pliron_lowering`, and `pliron_ptx` modules (and
//! any other `pliron_*` module exposed by this crate) are an **in-progress
//! implementation surface** for the W3.x / W4.x cuda-oxide backend tracked
//! in [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md). They contain
//! several hundred lines of `NotYetWired` scaffolding stubs that exist only
//! so the v0.4 author has a target to fill in. They are deliberately
//! `#[doc(hidden)]` to keep them out of the rendered docs.rs surface, and
//! they are **NOT covered by semver compatibility until tensor-wasm-jit
//! reaches 1.0** — any name, signature, or behaviour inside a `pliron_*`
//! module is liable to change in a 0.x.y patch release without notice.
//! External consumers should not import from them.
#![deny(missing_docs)]

pub mod cache;
pub mod clif_lower;
pub mod deopt;
pub mod detector;
pub mod ir;
pub mod ptx_emit;
pub mod rewrite;

#[doc(hidden)]
#[cfg(feature = "cuda-oxide-backend")]
pub mod pliron_dialect;

// Wave 1 of the Pliron pipeline: pure-Rust interim IR + per-family
// Cranelift-IR lowering passes. See [`lowered_ir`] for the IR contract
// and [`pliron_dialect`] for the trait surface that ties them together.
#[cfg(feature = "cuda-oxide-backend")]
pub mod lowered_ir;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_arith;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_float;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_memory;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_cf;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_vector;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_conv;
/// Internal test fixtures for the lowering passes.
///
/// Gated behind the opt-in `test-utils` feature so the symbols don't leak
/// into the stable public API. The crate's own unit and integration tests
/// see the module via `cfg(test)` (unit tests) or via the `test-utils`
/// feature flag enabled on the test target (integration tests). External
/// consumers that genuinely need these fixtures must opt in by adding
/// `tensor-wasm-jit = { ..., features = ["test-utils"] }` to their
/// `[dev-dependencies]` — they are not part of the stable surface.
#[cfg(any(test, feature = "test-utils"))]
#[cfg(feature = "cuda-oxide-backend")]
pub mod lowering_test_support;
#[cfg(feature = "cuda-oxide-backend")]
pub mod reject_list;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lower_signature;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lowering_builder;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lowering_errors;
#[cfg(feature = "cuda-oxide-backend")]
pub mod blueprint_adapter;
#[cfg(feature = "cuda-oxide-backend")]
pub mod lowering_driver;
#[doc(hidden)]
#[cfg(feature = "cuda-oxide-backend")]
pub mod pliron_lowering;
#[doc(hidden)]
#[cfg(feature = "cuda-oxide-backend")]
pub mod pliron_ptx;
