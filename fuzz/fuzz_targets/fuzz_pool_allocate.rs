// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `tensor_wasm_mem::pool::UnifiedMemoryPool::allocate`.
//!
//! Drives `allocate(size, align)` with arbitrary `(u32, u32)` widened to
//! `usize`. Contract: must never panic. Every documented failure mode
//! (zero size, non-power-of-two alignment, alignment-too-large, slab
//! exhaustion, integer overflow inside the alignment math) returns an
//! `Err(UnifiedError::…)` — a panic in any of those paths would surface
//! as a guest-triggered host abort in production.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_mem::pool::UnifiedMemoryPool;

fuzz_target!(|input: (u32, u32)| {
    let (size, align) = (input.0 as usize, input.1 as usize);

    // 4 MiB slab — big enough that legitimate-shape requests succeed and
    // we exercise the bump-pointer fast path, small enough that the
    // exhaustion branch is reachable for moderate `size` values.
    //
    // The pool constructor itself may fail when CUDA is unavailable; that
    // is not the property under test, so silently skip on construction
    // error.
    let Ok(pool) = UnifiedMemoryPool::new(4 * 1024 * 1024) else {
        return;
    };

    // The property under test: `allocate` returns a `Result` for every
    // input the fuzzer can synthesise, never panics. We deliberately
    // discard the outcome — both `Ok(_)` and `Err(_)` are acceptable.
    let _ = pool.allocate(size, align);
});
