// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Bincode length-field abuse.
//!
//! Builds a tiny bincode blob whose `wasm_memory` field claims length
//! `u64::MAX`. Without `bincode::Options::with_limit`, the deserialiser would
//! try to pre-allocate ~16 EiB and OOM the host. With the limit in place, the
//! reader must reject the blob long before touching the allocator.

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{DEFAULT_ZSTD_LEVEL, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};

#[test]
fn bincode_vec_length_overflow_is_refused_without_allocating() {
    // Hand-roll the bincode prefix: little-endian, fixint, matching how
    // `bincode::serialize(&Snapshot { .. })` would lay out the first two `u32`
    // fields and the start of `wasm_memory`'s length prefix.
    let mut blob = Vec::with_capacity(32);
    blob.extend_from_slice(&SNAPSHOT_MAGIC.to_le_bytes());
    blob.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    // wasm_memory length prefix — `u64::MAX` is the worst-case allocation
    // request the deserialiser could honour without our cap in place.
    blob.extend_from_slice(&u64::MAX.to_le_bytes());
    // No body bytes follow; if the deserialiser ignored the limit and tried
    // to honour the length prefix, it would either OOM or read past EOF.

    let compressed = zstd::encode_all(blob.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd encode");

    let err = SnapshotReader::new()
        .restore(&compressed)
        .expect_err("u64::MAX length prefix must be rejected by the bincode size cap");

    match err {
        TensorWasmError::Serialization(msg) => {
            assert!(
                msg.contains("bincode") || msg.contains("limit"),
                "expected bincode-limit rejection, got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
