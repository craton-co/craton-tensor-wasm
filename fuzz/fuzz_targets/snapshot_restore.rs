// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `SnapshotReader::restore` on the v2 (unsigned) path.
//!
//! Drives the reader with arbitrary bytes and no HMAC key configured, so the
//! input is classified as v2 (or rejected outright as malformed). Contract:
//! `restore` must never panic — every adversarial input is expected to either
//! deserialise into a `Snapshot` or be surfaced as a structured
//! `TensorWasmError::Serialization`. A panic from this target is a hard failure.
//!
//! Pairs with `snapshot_restore_signed.rs`, which exercises the v3 / HMAC code
//! path. The two are split rather than `match`-ed inside one fuzz_target so
//! libFuzzer's per-corpus statistics stay readable.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_snapshot::reader::SnapshotReader;

fuzz_target!(|data: &[u8]| {
    // No HMAC key path — exercises the v2 reader. Cap the input length to
    // mirror what real call sites enforce, so the fuzzer doesn't waste cycles
    // on inputs `restore` would reject in its first cheap length check anyway.
    if data.len() > 64 * 1024 * 1024 {
        return;
    }
    let _ = SnapshotReader::default().restore(data);
});
