// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `SnapshotReader::restore` on the v3 (HMAC-SHA256) path.
//!
//! Splits the first 32 bytes of fuzzer input into a synthetic HMAC key and
//! hands the remaining bytes to `restore` through a reader configured with
//! that key. This exercises the "authenticate then parse" pre-decompression
//! HMAC verification — the overwhelmingly common outcome is `HMAC mismatch`
//! (which is fine and exactly the property under test: bad signatures must be
//! rejected without crashing or decoding the payload). Occasionally the
//! fuzzer will land on a key+blob pair that survives the trailer check and
//! drives the zstd/bincode path; either way, `restore` must never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_snapshot::reader::SnapshotReader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }
    // Cap payload length (after consuming the 32-byte key prefix) to the same
    // 64 MiB ceiling the v2 target uses; the reader's own `MAX_INPUT_BYTES`
    // check would reject anything bigger immediately.
    if data.len() - 32 > 64 * 1024 * 1024 {
        return;
    }
    let (key_bytes, payload) = data.split_at(32);
    let key: [u8; 32] = key_bytes
        .try_into()
        .expect("split_at(32) yields exactly 32 bytes");
    let _ = SnapshotReader::default()
        .with_hmac_sha256_key(key)
        .restore(payload);
});
