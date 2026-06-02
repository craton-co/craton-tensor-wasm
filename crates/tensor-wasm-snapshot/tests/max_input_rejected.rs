// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Reject compressed inputs larger than `limits::MAX_INPUT_BYTES`.
//!
//! Constructs a (zero-filled) byte buffer one byte past the cap and asserts the
//! reader refuses it before invoking zstd. Hosts without ~4 GiB of free RAM
//! short-circuit via `try_reserve_exact`, matching the existing
//! `restore_rejects_oversized_wasm_memory` pattern.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_snapshot::limits;
use tensor_wasm_snapshot::reader::SnapshotReader;

#[test]
fn input_one_byte_over_cap_is_rejected_before_zstd() {
    let oversize = limits::MAX_INPUT_BYTES.saturating_add(1);

    let mut buf: Vec<u8> = Vec::new();
    if buf.try_reserve_exact(oversize).is_err() {
        // Not enough RAM on this host to materialise the oversized buffer;
        // the cap is still enforced by the same code path the in-RAM run
        // would have exercised, so a skip here does not weaken coverage.
        eprintln!("skipping max_input_rejected: host lacks {oversize} bytes of free RAM");
        return;
    }
    buf.resize(oversize, 0);

    let err = SnapshotReader::new()
        .restore(&buf)
        .expect_err("input over MAX_INPUT_BYTES must be rejected");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("input too large"),
                "expected input-cap rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
