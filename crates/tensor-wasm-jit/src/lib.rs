// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! JIT pipeline: Cranelift detector, IR normalisation, PTX codegen, kernel cache, deopt.
#![deny(missing_docs)]

pub mod cache;
pub mod clif_lower;
pub mod deopt;
pub mod detector;
pub mod ir;
pub mod ptx_emit;
pub mod rewrite;

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
#[cfg(feature = "cuda-oxide-backend")]
pub mod pliron_lowering;
