// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Placeholder for the W2.4 wave-2 task (module-level lowering driver).
//! Replaced by the W2.4 agent with `lower_function` that walks a
//! Cranelift `Function`, dispatches each instruction to the L1-L6
//! per-family lowerers, and assembles a [`crate::lowered_ir::LoweredFunction`].

#![cfg(feature = "cuda-oxide-backend")]
#![allow(dead_code)]

// Wave-2 placeholder. Replaced by W2.4 agent output.
