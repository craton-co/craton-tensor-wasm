// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `tensor_wasm_wasi_gpu::kernel_args::parse_argv`.
//!
//! The wire-format argv parser is fed straight off the host-trust boundary —
//! guest Wasm produces the byte buffer and the host has to refuse every
//! malformed shape without panicking. Pure validation failures are the
//! happy path: we expect `Ok(_)` or one of the three documented
//! [`AbiError`] variants (`InvalidArgs`, `InvalidPointer`,
//! `KernelArgsUnsupported`). Any other variant — and especially any panic —
//! is a finding.
//!
//! The mock guest-memory slice mirrors the `fake_mem()` helper used by the
//! crate's own unit tests (`kernel_args.rs::tests::fake_mem`): a fixed
//! 4 KiB zeroed buffer that's large enough for any in-bounds pointer arg
//! the fuzzer might cook up while small enough that an attacker-crafted
//! offset still hits the bounds-check path most of the time.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_wasi_gpu::abi::AbiError;
use tensor_wasm_wasi_gpu::kernel_args::parse_argv;

fuzz_target!(|data: &[u8]| {
    // The parser's own cap is 4 KiB (`MAX_KERNEL_ARGS_BYTES`); anything
    // larger short-circuits to `KernelArgsUnsupported` without exercising
    // interesting code. Skip those inputs so the fuzzer's mutator stays in
    // the productive range.
    if data.len() > 4 * 1024 {
        return;
    }

    // Fixed 4 KiB guest-memory mock. Matches the `fake_mem()` helper in
    // `kernel_args.rs::tests` — large enough that pointer args can land
    // in-bounds, small enough that off-by-one and overflow paths still
    // get exercised by mutated offsets.
    let mem = [0u8; 4096];

    match parse_argv(data, &mem) {
        Ok(_) => {}
        Err(AbiError::InvalidArgs)
        | Err(AbiError::InvalidPointer)
        | Err(AbiError::KernelArgsUnsupported) => {}
        // Any other documented `AbiError` variant is a contract violation —
        // `parse_argv`'s rustdoc lists exactly these three. Surface it
        // immediately rather than silently accepting it.
        Err(other) => panic!("parse_argv returned undocumented variant: {other:?}"),
    }
});
