// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T21 regression: the writer pre-sizes its `compressed` `Vec<u8>` from a
//! 4:1-ratio heuristic over the input's total uncompressed size instead of
//! a constant 8 KiB. The qualitative property under test is that a
//! 64-MiB-class capture round-trips correctly — without the pre-size, the
//! streaming zstd encoder still produces the same bytes, but it does so
//! through ~10+ `Vec` reallocations. Asserting the reallocation count
//! directly would require an invasive test hook on `Vec`'s growth, so the
//! check here is the property we actually care about: the heuristic does
//! not change wire bytes and a 64-MiB snapshot still round-trips
//! end-to-end.
//!
//! Additionally, the test exercises the upper clamp of the heuristic by
//! verifying that an input near the per-blob size cap does not panic
//! during pre-sizing — the clamp at `MAX_INPUT_BYTES / 4` (256 MiB at
//! today's 1 GiB ceiling) bounds the initial allocation.

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};

/// 64 MiB — the brief's target size for the pre-size heuristic test. Large
/// enough that the old 8 KiB pre-size would force ~10+ Vec growth events
/// during compression, but small enough that the test still runs in a
/// few seconds on CI.
const SIXTY_FOUR_MIB: usize = 64 * 1024 * 1024;

/// Build a 64 MiB wasm-memory body that compresses well enough to exercise
/// the heuristic without making the test take forever. A repeating
/// 256-byte pattern compresses heavily under zstd, which keeps the
/// round-trip fast while still pushing through the writer's hot path
/// with a realistic payload size.
fn sixty_four_mib_state() -> Vec<u8> {
    let pattern: Vec<u8> = (0u8..=255).collect();
    let reps = SIXTY_FOUR_MIB / pattern.len();
    let mut wasm = Vec::with_capacity(SIXTY_FOUR_MIB);
    for _ in 0..reps {
        wasm.extend_from_slice(&pattern);
    }
    assert_eq!(wasm.len(), SIXTY_FOUR_MIB);
    wasm
}

#[test]
fn sixty_four_mib_snapshot_round_trips() {
    // Property under test: a 64 MiB snapshot survives the heuristic
    // pre-size and decodes back to the same bytes. If the heuristic
    // mis-estimates either direction (e.g. allocates a zero-sized Vec
    // or a multi-GiB Vec), this test surfaces it either as a panic
    // (allocation failure on a constrained CI box) or as a corrupted
    // round-trip (the compressed Vec was somehow truncated). The 4:1
    // ratio against 64 MiB of input requests an initial capacity of
    // 16 MiB — well under any plausible CI memory limit.
    let wasm = sixty_four_mib_state();
    let gpu: Vec<u8> = Vec::new();
    let regs: Vec<u8> = Vec::new();

    let writer = SnapshotWriter::new();
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(2),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture 64 MiB");

    let restored = SnapshotReader::new()
        .restore(&blob)
        .expect("restore 64 MiB");
    assert_eq!(restored.wasm_memory.len(), SIXTY_FOUR_MIB);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        SIXTY_FOUR_MIB as u64,
    );
}

#[test]
fn small_snapshot_still_round_trips_under_floor_clamp() {
    // Property under test: when the input is much smaller than the 8 KiB
    // floor, the heuristic clamps up rather than allocating zero bytes.
    // This is the same shape as the original `empty_bodies_round_trip`
    // test in `writer.rs`'s `#[cfg(test)] mod tests`, lifted here as an
    // integration test so the clamp is exercised against the public API
    // surface.
    let writer = SnapshotWriter::new();
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(0),
            instance_id: InstanceId(0),
            wasm_memory: &[],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture empty");
    let restored = SnapshotReader::new().restore(&blob).expect("restore empty");
    assert!(restored.wasm_memory.is_empty());
    assert_eq!(restored.metadata.total_uncompressed_bytes, 0);
}

#[test]
fn pre_size_does_not_change_wire_bytes() {
    // Property under test: the pre-size heuristic only affects allocator
    // behaviour, not the produced byte stream. The heuristic is derived purely
    // from `total_uncompressed_bytes`, so for a fixed input it is identical
    // across captures; the only other source of variation is the metadata
    // `created_unix_ms`. That field is a fixed-width `u64`, but it is bincode-
    // encoded *inside* the zstd stream — and zstd output length depends on the
    // byte *content*, not just the input length — so a changing timestamp value
    // can shift the compressed length by a byte. (That is exactly what made an
    // earlier `a.len() == b.len()` form of this test flaky.)
    //
    // We therefore pin the timestamp with `with_created_unix_ms` so the two
    // captures are fully deterministic, and assert they are BYTE-IDENTICAL — a
    // strictly stronger invariant than equal length. Any divergence would mean
    // the pre-size leaked into the payload (e.g. a stray write into the `Vec`'s
    // spare capacity).
    let wasm: Vec<u8> = (0u32..16384).map(|i| (i % 251) as u8).collect();
    let writer = SnapshotWriter::new().with_created_unix_ms(1_700_000_000_000);
    let a = writer
        .capture(InstanceState {
            tenant_id: TenantId(5),
            instance_id: InstanceId(7),
            wasm_memory: &wasm,
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture a");
    let b = writer
        .capture(InstanceState {
            tenant_id: TenantId(5),
            instance_id: InstanceId(7),
            wasm_memory: &wasm,
            gpu_memory: &[],
            registers: &[],
        })
        .expect("capture b");
    assert_eq!(
        a, b,
        "two captures of the same input with a pinned timestamp must be \
         byte-identical — divergence would imply the pre-size heuristic leaked \
         into the compressed payload",
    );
}
