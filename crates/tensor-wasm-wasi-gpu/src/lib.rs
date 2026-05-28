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
//! interface definition (`wasi:cuda/host@0.2.0`) — the WIT and the
//! constants in [`abi`] are kept in lockstep.
//!
//! The [`scheduler`] module exposes a separate `wasi:scheduler/host@0.1.0`
//! interface for cooperative deadlines (roadmap feature #4) — guests
//! offer suspend points via `yield()` and the host returns a non-zero
//! code when the per-instance deadline is approaching. See
//! `docs/COOPERATIVE-YIELD.md`.
#![deny(missing_docs)]

pub mod abi;
pub mod async_dispatch;
pub mod host;
pub mod kernel_args;
pub mod registry;
pub mod scheduler;
