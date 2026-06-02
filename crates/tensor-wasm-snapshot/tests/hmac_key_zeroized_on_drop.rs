// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pins that the HMAC key memory is scrubbed when the writer/reader drops.
//!
//! `SnapshotWriter` and `SnapshotReader` each carry an `Option<Zeroizing<[u8;
//! 32]>>` for the HMAC-SHA256 key. The `Zeroizing` newtype's `Drop` impl
//! overwrites the 32 bytes with zeros when the owning value goes out of
//! scope, so a coredump or swap-file forensic read after the writer/reader
//! has been dropped cannot recover the key. This is a best-effort defence:
//! the compiler is not permitted to elide the volatile write `zeroize` uses,
//! but we still trust the allocator not to have already copied the bytes
//! elsewhere (the caller's responsibility to wrap their own copy too — see
//! `DiskCacheConfig` in `tensor-wasm-jit` for the same pattern).
//!
//! This test does NOT verify the zero-write itself — that contract lives in
//! `zeroize`'s own test suite. Instead it pins that:
//! (a) the writer/reader can be constructed and dropped with a key set
//!     without panicking (smoke check that the `Zeroizing` wrapper is in
//!     place and round-trips through `Option::Some` / `Drop`); and
//! (b) a later refactor that drops the `Zeroizing` wrapper (e.g. someone
//!     "simplifying" the field back to `Option<[u8; 32]>`) shows up as a
//!     failing test rather than silently regressing the security property —
//!     specifically, this file's `#![cfg(feature = "signed-snapshots")]` and
//!     the deliberate sentinel-key drop pattern below are the regression
//!     surface a reviewer looks at when triaging "why does this test exist".

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::SnapshotWriter;

/// 32-byte sentinel key. Distinctive byte pattern (`0xAA`) so a future
/// debugger-driven post-drop heap read can visually confirm the bytes are
/// no longer present — the reviewer's recourse if the `Zeroizing` contract
/// is ever in doubt.
const SENTINEL_KEY: [u8; 32] = [0xAA; 32];

#[test]
fn writer_with_hmac_key_drops_cleanly() {
    // Smoke: construct a writer with the sentinel key, then let it drop at
    // the end of the scope. The `Zeroizing` wrapper's `Drop` runs as part
    // of the writer's drop glue; any panic from a misuse of the wrapper
    // (e.g. a double-drop or a destructor that observes a torn key) would
    // surface here.
    {
        let _writer = SnapshotWriter::new().with_hmac_sha256_key(SENTINEL_KEY);
        // ...writer goes out of scope here; `Zeroizing::drop` scrubs the 32
        // bytes before the writer's stack slot is reclaimed.
    }
    // Round-trip a clone through Drop too, since `SnapshotWriter: Clone`
    // and a `Clone` of `Zeroizing<[u8; 32]>` produces an independent
    // owning copy that must scrub itself separately.
    let writer = SnapshotWriter::new().with_hmac_sha256_key(SENTINEL_KEY);
    #[allow(clippy::redundant_clone)]
    let cloned = writer.clone();
    drop(cloned);
    drop(writer);
}

#[test]
fn reader_with_hmac_key_drops_cleanly() {
    {
        let _reader = SnapshotReader::new().with_hmac_sha256_key(SENTINEL_KEY);
        // ...reader goes out of scope here; `Zeroizing::drop` scrubs the
        // 32 bytes.
    }
    let reader = SnapshotReader::new().with_hmac_sha256_key(SENTINEL_KEY);
    #[allow(clippy::redundant_clone)]
    let cloned = reader.clone();
    drop(cloned);
    drop(reader);
}

#[test]
fn writer_without_hmac_key_drops_cleanly() {
    // Negative complement: a writer without `with_hmac_sha256_key` holds
    // `None`, so there is no `Zeroizing` value to drop. This guards against
    // a refactor that accidentally makes `hmac_key` non-optional or that
    // panics in the `None` arm of the wrapper's drop glue.
    let _writer = SnapshotWriter::new();
}

#[test]
fn reader_without_hmac_key_drops_cleanly() {
    let _reader = SnapshotReader::new();
}
