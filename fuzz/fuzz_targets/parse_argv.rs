// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Split-input fuzz target for `tensor_wasm_wasi_gpu::kernel_args::parse_argv`.
//!
//! Complementary to `fuzz_parse_argv.rs`, which fuzzes the argv buffer
//! against a fixed-size zeroed `mem`. This target instead splits each
//! input down the middle so the fuzzer can vary the argv bytes *and*
//! the `mem` bytes independently — useful for finding pointer-arg shapes
//! whose `[off, off+len)` window straddles a corner of the (now also
//! attacker-shaped) memory slice.
//!
//! Same contract as the other parse-argv target: any panic is a finding,
//! and only the three documented error variants
//! (`InvalidArgs`, `InvalidPointer`, `KernelArgsUnsupported`) are
//! acceptable on the `Err` arm.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_wasi_gpu::abi::AbiError;
use tensor_wasm_wasi_gpu::kernel_args::parse_argv;

fuzz_target!(|data: &[u8]| {
    // Need at least 64 bytes so each side of the split is plausible
    // (the smallest meaningful argv record is 5 bytes, and `mem` needs
    // a few bytes for pointer-arg bounds to be interesting). Inputs
    // smaller than 64 bytes are skipped so the mutator stays on the
    // productive range.
    if data.len() < 64 {
        return;
    }
    // Split data: first half is the argv buffer, second half is the
    // fake guest memory. A guest-shaped attack would normally control
    // both — the argv bytes encode the (offset, len) pair that names
    // the window into `mem`, and `mem` is the linear memory it's read
    // out of.
    let split = data.len() / 2;
    let argv = &data[..split];
    let mem = &data[split..];

    // Call parse_argv; any panic is a fuzzer-reportable failure. All
    // documented errors (`InvalidArgs`, `InvalidPointer`,
    // `KernelArgsUnsupported`) are acceptable Err returns — anything
    // else is a contract violation we want surfaced.
    match parse_argv(argv, mem) {
        Ok(_) => {}
        Err(AbiError::InvalidArgs)
        | Err(AbiError::InvalidPointer)
        | Err(AbiError::KernelArgsUnsupported) => {}
        Err(other) => panic!("parse_argv returned undocumented variant: {other:?}"),
    }
});
