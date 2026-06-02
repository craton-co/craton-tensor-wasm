// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `tensor_wasm_jit::rewrite::rewrite_wasm`.
//!
//! Property: if the input bytes validate as Wasm, then either
//! `rewrite_wasm` returns an `Err` *or* the rewritten module *also*
//! validates as Wasm. The rewriter must never produce a structurally
//! invalid module from a structurally valid one — that would manifest
//! as a wasmtime trap at instantiation time on real user input.
//!
//! Inputs that fail `wasmparser::validate` up-front are skipped — those
//! exercise `wasmtime::Module::from_binary`'s parser, which already has its
//! own fuzz target (`fuzz_wasm_compile`).

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_jit::cache::KernelCache;
use tensor_wasm_jit::rewrite::{rewrite_wasm, RewriteOptions};

fuzz_target!(|data: &[u8]| {
    // Cap input size to keep iteration rate high. The rewriter walks every
    // function body twice (analysis + re-encode); much beyond 64 KiB the
    // throughput drops below the useful threshold for the corpus we
    // currently have.
    if data.len() > 64 * 1024 {
        return;
    }

    // Skip inputs that aren't valid Wasm to begin with — those don't
    // exercise the rewriter, they exercise `wasmparser::validate` itself
    // (which has its own upstream fuzz coverage).
    if wasmparser::validate(data).is_err() {
        return;
    }

    let opts = RewriteOptions::default();
    let cache = KernelCache::new();
    if let Ok(outcome) = rewrite_wasm(data, &opts, &cache) {
        // The core invariant: rewrites preserve validity. A rewritten
        // module that fails to re-validate would crash the runtime at
        // instantiation on real user code.
        assert!(
            wasmparser::validate(&outcome.rewritten_wasm).is_ok(),
            "rewrite produced invalid wasm",
        );
    }
});
