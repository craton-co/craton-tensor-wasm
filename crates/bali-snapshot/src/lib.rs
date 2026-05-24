//! Wasm + GPU memory snapshot and restore for fast cold-start.
//!
//! Project Bali captures combined Wasm linear memory, GPU device-side memory,
//! and register-file state into a single zstd-compressed bincode blob that can
//! be persisted, shipped between nodes, and reloaded on demand. The [`writer`]
//! module produces blobs from a live [`writer::InstanceState`], while the
//! [`reader`] module restores them with strict magic + version checking so
//! malformed inputs are surfaced as errors rather than panics.
#![deny(missing_docs)]

pub mod reader;
pub mod writer;

pub use crate::writer::{limits, payload_crc32, Snapshot, SnapshotMetadata};
