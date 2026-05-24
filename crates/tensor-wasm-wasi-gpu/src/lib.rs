// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `wasi-cuda` host bridge — the explicit GPU kernel-launch API exposed to
//! Wasm modules.
//!
//! Wasm code imports the functions defined in [`abi::MODULE`] and uses them
//! to upload PTX modules ([`abi::FN_LOAD_PTX`]) and launch kernels
//! ([`abi::FN_LAUNCH`]). On hosts without CUDA the host functions return
//! [`abi::AbiError::NotAvailable`]; on CUDA hosts they call into the
//! `cust` crate via the `cuda` feature.
//!
//! See `wit/wasi-cuda.wit` at the workspace root for the Component-Model
//! interface definition (`wasi:cuda/host@0.1.0`) — the WIT and the
//! constants in [`abi`] are kept in lockstep.
#![deny(missing_docs)]

pub mod abi;
pub mod async_dispatch;
pub mod host;
pub mod registry;
