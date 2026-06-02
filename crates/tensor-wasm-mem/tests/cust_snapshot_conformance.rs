// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W4.2 cross-backend snapshot conformance: cust / `unified-memory` half.
//!
//! Compiled only when the `unified-memory` feature is on. Exercises the
//! same byte-source round-trip that
//! [`cudarc_snapshot_conformance.rs`][cudarc] and
//! [`cuda_oxide_snapshot_conformance.rs`][oxide] drive against their
//! respective backings, but routes the producer-side bytes through the
//! cust [`crate::unified::UnifiedBuffer`][unified] (which under
//! `--features unified-memory` resolves to a `cust::memory::UnifiedBuffer`
//! per the precedence table in `unified.rs`).
//!
//! The unignored portion runs on host-only CI: it builds a pair of
//! `Vec<u8>` payloads and snapshots them, proving that the snapshot wire
//! format itself stays stable when compiled with `--features
//! unified-memory`. The hardware-gated portion additionally allocates a
//! real `cuMemAllocManaged` region, populates it with a known pattern,
//! and round-trips through `SnapshotWriter::capture` →
//! `SnapshotReader::restore`. Marked `#[ignore = "requires CUDA
//! hardware"]` per the repo convention; the S22 CUDA runner picks it up
//! via `cargo test --features unified-memory -- --ignored`.
//!
//! [cudarc]: ./cudarc_snapshot_conformance.rs
//! [oxide]: ./cuda_oxide_snapshot_conformance.rs
//! [unified]: ../../src/unified.rs

#![cfg(feature = "unified-memory")]

use tensor_wasm_mem::unified::UnifiedBuffer;

mod common;

use common::{
    populate_payloads_into, snapshot_round_trip_with_source, CONFORMANCE_GPU_LEN,
    CONFORMANCE_REGISTERS_LEN, CONFORMANCE_WASM_LEN,
};

/// Pure-`Vec<u8>` round-trip compiled under `--features unified-memory`.
///
/// The cust crate is in the resolved dep graph (so the test binary
/// links against `libcuda`-bridging code), but no allocation hits the
/// driver. This is the host-only assertion: enabling
/// `unified-memory` does NOT perturb the snapshot wire format. If the
/// cust feature accidentally pulled in a `Cargo.toml` re-export or a
/// macro that re-defined `SnapshotWriter` (the historical RFC 0001 hazard
/// of "feature-gated production code can quietly change behaviour
/// downstream"), this test would catch it.
#[test]
fn cust_feature_does_not_perturb_snapshot_wire_format() {
    let mut wasm = vec![0u8; CONFORMANCE_WASM_LEN];
    let mut gpu = vec![0u8; CONFORMANCE_GPU_LEN];
    let mut regs = vec![0u8; CONFORMANCE_REGISTERS_LEN];
    populate_payloads_into(&mut wasm, &mut gpu, &mut regs);

    snapshot_round_trip_with_source(&wasm, &gpu, &regs);
}

/// Hardware-gated end-to-end: allocate a cust-backed managed buffer,
/// populate it via `UnifiedBuffer::as_mut_slice`, then drive the same
/// `SnapshotWriter::capture` → `SnapshotReader::restore` round-trip
/// the host-only test does. The decisive assertion is that bytes
/// scribbled into a `cudaMallocManaged` region land in the snapshot
/// payload byte-for-byte, with no aliasing between the `wasm_memory`,
/// `gpu_memory`, and `registers` slots even though all three are
/// backed by independent managed allocations.
///
/// Requires a CUDA driver and at least one visible GPU; the S22
/// runner unignores via `-- --ignored`.
#[test]
#[ignore = "requires CUDA hardware"]
fn cust_unified_buffer_snapshot_round_trip_on_device() {
    let mut wasm_buf =
        UnifiedBuffer::new(CONFORMANCE_WASM_LEN).expect("alloc wasm-memory unified buffer");
    let mut gpu_buf =
        UnifiedBuffer::new(CONFORMANCE_GPU_LEN).expect("alloc gpu-memory unified buffer");
    let mut regs_buf =
        UnifiedBuffer::new(CONFORMANCE_REGISTERS_LEN).expect("alloc registers unified buffer");

    populate_payloads_into(
        wasm_buf.as_mut_slice(),
        gpu_buf.as_mut_slice(),
        regs_buf.as_mut_slice(),
    );

    snapshot_round_trip_with_source(wasm_buf.as_slice(), gpu_buf.as_slice(), regs_buf.as_slice());
}
