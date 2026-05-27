// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Wasm + GPU memory snapshot and restore for fast cold-start.
//!
//! Craton TensorWasm captures combined Wasm linear memory, GPU device-side memory,
//! and register-file state into a single zstd-compressed bincode blob that can
//! be persisted, shipped between nodes, and reloaded on demand. The [`writer`]
//! module produces blobs from a live [`writer::InstanceState`], while the
//! [`reader`] module restores them with strict magic + version checking so
//! malformed inputs are surfaced as errors rather than panics.
//!
//! See `FORMAT.md` in the crate root for the on-disk wire layout, version
//! history, and the size caps that bound restore-time memory pressure.
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// tensor-wasm-snapshot stores `usize` lengths in caps that exceed 2^31. Building on a
// 32-bit target would silently truncate them and defeat the size-bounding
// guarantees the reader makes. Fail the build instead.
const _: () = assert!(
    usize::BITS >= 64,
    "tensor-wasm-snapshot requires a 64-bit target",
);

pub mod format;
pub mod reader;
pub mod writer;

pub use crate::format::{
    SignatureKind, HMAC_SHA256_SIG_LEN, SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN,
    SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3,
};
pub use crate::writer::{limits, payload_crc32, Snapshot, SnapshotMetadata};
