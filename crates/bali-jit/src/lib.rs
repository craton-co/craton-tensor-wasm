//! JIT pipeline: Cranelift detector, IR normalisation, PTX codegen, kernel cache, deopt.
#![warn(missing_docs)]

pub mod cache;
pub mod clif_lower;
pub mod deopt;
pub mod detector;
pub mod ir;
pub mod ptx_emit;
