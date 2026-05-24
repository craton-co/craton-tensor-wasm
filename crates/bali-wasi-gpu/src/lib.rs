//! `wasi-cuda` host bridge — the explicit GPU kernel-launch API exposed to
//! Wasm modules.
//!
//! Wasm code imports the functions defined in [`abi::MODULE`] and uses them
//! to upload PTX modules ([`abi::FN_LOAD_PTX`]) and launch kernels
//! ([`abi::FN_LAUNCH`]). On hosts without CUDA the host functions return
//! [`abi::AbiError::NotAvailable`]; on CUDA hosts they call into the
//! `cust` crate via the `cuda` feature.
#![deny(missing_docs)]

pub mod abi;
pub mod async_dispatch;
pub mod host;
pub mod registry;
