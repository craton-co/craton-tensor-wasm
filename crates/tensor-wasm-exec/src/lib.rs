// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Wasmtime + Tokio async execution engine for TensorWasm instances.
//!
//! `tensor-wasm-exec` wraps [`wasmtime`] with a TensorWasm-specific [`engine::TensorWasmEngine`]
//! that wires in async execution, epoch-based interruption, and a custom
//! linear-memory creator backed by [`tensor_wasm_mem`]. The [`instance`] module
//! manages per-tenant instance lifecycles, and [`executor`] drives async
//! invocation of guest exports against a shared engine and store pool.
#![deny(missing_docs)]

pub mod auto_offload;
pub mod engine;
pub mod executor;
pub mod instance;
pub mod jit_dispatch;
