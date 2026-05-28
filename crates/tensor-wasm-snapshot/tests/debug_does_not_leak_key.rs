// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for the HMAC-key Debug-leak finding (snapshot 3.3).
//!
//! `SnapshotWriter` and `SnapshotReader` derived `Debug` over their
//! `Option<[u8; 32]>` HMAC key, so `tracing::debug!(?writer)` printed every
//! key byte. The fix replaces the derive with a manual impl that prints a
//! `<REDACTED>` placeholder. This test asserts no key bytes appear in the
//! Debug output and that the redaction marker is present.
//!
//! Also covers the `Snapshot` Debug impl which previously dumped multi-GiB
//! `Vec<u8>` fields — now it prints `*_len` placeholders.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_snapshot::writer::SnapshotWriter;
use tensor_wasm_snapshot::reader::SnapshotReader;

const SENTINEL_KEY: [u8; 32] = [0xAB; 32];

#[test]
fn writer_debug_redacts_hmac_key() {
    let writer = SnapshotWriter::new().with_hmac_sha256_key(SENTINEL_KEY);
    let dbg = format!("{writer:?}");
    assert!(
        !dbg.contains("171") && !dbg.to_lowercase().contains("ab, ab"),
        "writer Debug leaked key bytes: {dbg}"
    );
    assert!(
        dbg.to_lowercase().contains("redacted"),
        "writer Debug missing REDACTED marker: {dbg}"
    );
}

#[test]
fn reader_debug_redacts_hmac_key() {
    let reader = SnapshotReader::new().with_hmac_sha256_key(SENTINEL_KEY);
    let dbg = format!("{reader:?}");
    assert!(
        !dbg.contains("171") && !dbg.to_lowercase().contains("ab, ab"),
        "reader Debug leaked key bytes: {dbg}"
    );
    assert!(
        dbg.to_lowercase().contains("redacted"),
        "reader Debug missing REDACTED marker: {dbg}"
    );
}

#[test]
fn writer_debug_without_key_does_not_say_redacted() {
    let writer = SnapshotWriter::new();
    let dbg = format!("{writer:?}");
    // No key -> no REDACTED placeholder; field renders as None
    assert!(
        dbg.contains("None"),
        "writer with no key should render hmac_key: None, got: {dbg}"
    );
}
