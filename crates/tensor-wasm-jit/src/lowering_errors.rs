// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Placeholder for the W2.10 wave-2 task (structured lowering errors).
//! Replaced by the W2.10 agent with a `LoweringError` enum covering
//! UnsupportedOpcode, UnsupportedType, UndefinedValue, BadBlockReference,
//! MalformedTerminator, etc. Bridges into
//! [`crate::pliron_dialect::PlironLoweringError`] via `#[from]`.

#![cfg(feature = "cuda-oxide-backend")]
#![allow(dead_code)]

// Wave-2 placeholder. Replaced by W2.10 agent output.
