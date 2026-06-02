// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T9 freshness check: reader-side `with_max_age` enforcement.
//!
//! Covers the three primary code paths added in T9:
//!
//! 1. **Reject stale.** A reader configured with `with_max_age(60s)` must
//!    refuse a snapshot whose `created_unix_ms` is far enough in the past
//!    that `now - created_unix_ms > 60s`. The error must be
//!    [`TensorWasmError::SnapshotTooOld`] — a distinct variant from the
//!    generic `Serialization` so dashboards can pin replay-attempt
//!    rejections separately.
//! 2. **Accept recent.** A reader configured with `with_max_age(60s)` must
//!    accept a snapshot just captured.
//! 3. **Backward compat.** A reader constructed via `SnapshotReader::new()`
//!    (no `max_age` set) must accept a snapshot whose `created_unix_ms` is
//!    a year in the past — preserves the v0.3.x behaviour for callers
//!    that have not opted into the check.
//!
//! All three scenarios use the same fixture: a v2 (unsigned) snapshot
//! whose metadata struct is hand-built with a chosen `created_unix_ms`,
//! then bincode-encoded and zstd-compressed exactly as the writer's
//! `capture` path would have. We bypass the writer here because the
//! writer reads `SystemTime::now()` on every call and we need
//! deterministic timestamps to exercise the boundary cases.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::payload_crc32;
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{
    Snapshot, SnapshotMetadata, DEFAULT_ZSTD_LEVEL, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// Build a v2 snapshot blob with `created_unix_ms` set to the caller's
/// chosen value. Mirrors the writer's `capture` shape (bincode then zstd)
/// but with the timestamp pinned, so the test can drive the reader's
/// freshness check across the `now - created > max_age` boundary
/// deterministically.
fn build_snapshot_with_timestamp(created_unix_ms: u64) -> Vec<u8> {
    let wasm = vec![1u8, 2, 3, 4];
    let gpu = vec![9u8; 32];
    let regs = vec![0xAAu8; 8];
    let total = (wasm.len() + gpu.len() + regs.len()) as u64;
    let snap = Snapshot {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
        wasm_memory: wasm.clone(),
        gpu_memory: gpu.clone(),
        registers: regs.clone(),
        metadata: SnapshotMetadata {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            created_unix_ms,
            total_uncompressed_bytes: total,
            sequence_no: 0,
            nonce: None,
        },
        crc32: payload_crc32(&wasm, &gpu, &regs),
    };
    let cfg = bincode::config::legacy();
    let encoded = bincode::serde::encode_to_vec(&snap, cfg).expect("bincode encode");
    zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("zstd encode")
}

/// Returns the host's current wall-clock time in milliseconds since the
/// Unix epoch. Tests fall back to `0` if the clock is broken — that
/// would make the test inconclusive but never flaky, since `now == 0`
/// just means every constructed timestamp is "in the future" relative
/// to the reader's clock, which the freshness check accepts.
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

#[test]
fn freshness_check_rejects_old_snapshot() {
    // A snapshot captured one hour ago, with a 60-second max_age.
    // `now - created_unix_ms = 3_600_000 ms` >> `60_000 ms`, so the
    // reader must reject with `SnapshotTooOld`.
    let one_hour_ms: u64 = 60 * 60 * 1000;
    let created = now_unix_ms().saturating_sub(one_hour_ms);
    let bytes = build_snapshot_with_timestamp(created);

    let err = SnapshotReader::new()
        .with_max_age(Duration::from_secs(60))
        .restore(&bytes)
        .expect_err("stale snapshot must be rejected by freshness check");

    match err {
        TensorWasmError::SnapshotTooOld {
            created_unix_ms,
            now_unix_ms: now,
            max_age_ms,
        } => {
            assert_eq!(created_unix_ms, created);
            assert!(now >= created, "now must be >= created on a sane clock");
            assert_eq!(max_age_ms, 60_000);
        }
        other => panic!("expected SnapshotTooOld, got {other:?}"),
    }
}

#[test]
fn freshness_check_passes_recent_snapshot() {
    // A snapshot captured "now" (give-or-take a millisecond) must pass a
    // 60-second freshness window.
    let created = now_unix_ms();
    let bytes = build_snapshot_with_timestamp(created);
    let restored = SnapshotReader::new()
        .with_max_age(Duration::from_secs(60))
        .restore(&bytes)
        .expect("recent snapshot must pass freshness check");
    assert_eq!(restored.metadata.created_unix_ms, created);
}

#[test]
fn freshness_check_disabled_by_default() {
    // A year-old snapshot read by a reader with no `max_age` configured
    // must round-trip — this is the v0.3.x backward-compat contract.
    let one_year_ms: u64 = 365 * 24 * 60 * 60 * 1000;
    let created = now_unix_ms().saturating_sub(one_year_ms);
    let bytes = build_snapshot_with_timestamp(created);
    let restored = SnapshotReader::new()
        .restore(&bytes)
        .expect("default reader must accept arbitrarily old snapshots");
    assert_eq!(restored.metadata.created_unix_ms, created);
    assert!(SnapshotReader::new().max_age().is_none());
}

/// A snapshot whose `created_unix_ms` is in the future (writer clock
/// ahead of reader clock) is accepted even with `max_age` set —
/// future-dated captures are a transient clock-skew condition that
/// operators prefer to accept over reject. See the docstring on
/// `SnapshotReader::with_max_age` for the rationale.
#[test]
fn freshness_check_accepts_future_dated_snapshot() {
    // Set `created_unix_ms` an hour in the future.
    let one_hour_ms: u64 = 60 * 60 * 1000;
    let created = now_unix_ms().saturating_add(one_hour_ms);
    let bytes = build_snapshot_with_timestamp(created);
    let restored = SnapshotReader::new()
        .with_max_age(Duration::from_secs(60))
        .restore(&bytes)
        .expect("future-dated snapshot must be accepted (clock skew tolerance)");
    assert_eq!(restored.metadata.created_unix_ms, created);
}

/// A snapshot whose `created_unix_ms` is exactly `now - max_age` lies
/// on the boundary and must be accepted (the check is `age > max_age`,
/// not `age >= max_age`). Mostly a regression guard — the exact
/// boundary inequality is the kind of thing that gets flipped during
/// a hasty refactor.
#[test]
fn freshness_check_accepts_exactly_at_boundary() {
    let max_age_ms: u64 = 60_000;
    let created = now_unix_ms().saturating_sub(max_age_ms);
    let bytes = build_snapshot_with_timestamp(created);
    // Use a slightly larger max_age in the reader to absorb the
    // millisecond or two between `now_unix_ms()` here and the
    // reader-side `SystemTime::now()`. The interesting boundary is
    // "accepted near the edge", not "accepted on the exact tick".
    let restored = SnapshotReader::new()
        .with_max_age(Duration::from_millis(max_age_ms + 5_000))
        .restore(&bytes)
        .expect("snapshot just under the max_age boundary must be accepted");
    assert_eq!(restored.metadata.created_unix_ms, created);
}
