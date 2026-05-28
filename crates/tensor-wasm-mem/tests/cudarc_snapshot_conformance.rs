// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W4.2 cross-backend snapshot conformance: cudarc half.
//!
//! Compiled only when the `cudarc-backend` feature is on. Mirrors
//! [`cust_snapshot_conformance.rs`][cust] and
//! [`cuda_oxide_snapshot_conformance.rs`][oxide] but routes the
//! producer-side bytes through
//! [`crate::cudarc_backend::CudarcUnifiedBuffer`][cudarc_buf], which
//! wraps `cuMemAllocManaged` via the cudarc driver crate.
//!
//! The unignored portion runs on host-only CI and verifies the snapshot
//! wire format stays stable when compiled with `--features
//! cudarc-backend`. The hardware-gated portion allocates a real cudarc
//! UVM buffer, populates it with a known pattern, and round-trips the
//! bytes through `SnapshotWriter::capture` → `SnapshotReader::restore`.
//! Marked `#[ignore = "requires CUDA hardware"]` per the repo
//! convention; the S22 CUDA runner picks it up via `cargo test
//! --features cudarc-backend -- --ignored`.
//!
//! [cust]: ./cust_snapshot_conformance.rs
//! [oxide]: ./cuda_oxide_snapshot_conformance.rs
//! [cudarc_buf]: ../../src/cudarc_backend.rs

#![cfg(feature = "cudarc-backend")]

use tensor_wasm_mem::cudarc_backend::CudarcUnifiedBuffer;

mod common;

use common::{
    populate_payloads_into, snapshot_round_trip_with_source, CONFORMANCE_GPU_LEN,
    CONFORMANCE_REGISTERS_LEN, CONFORMANCE_WASM_LEN,
};

/// Pure-`Vec<u8>` round-trip compiled under `--features cudarc-backend`.
///
/// The cudarc crate is in the resolved dep graph (the test binary links
/// against the cudarc dynamic-loader shim), but no allocation hits the
/// driver. This is the host-only assertion: enabling the cudarc backend
/// does NOT perturb the snapshot wire format.
#[test]
fn cudarc_feature_does_not_perturb_snapshot_wire_format() {
    let mut wasm = vec![0u8; CONFORMANCE_WASM_LEN];
    let mut gpu = vec![0u8; CONFORMANCE_GPU_LEN];
    let mut regs = vec![0u8; CONFORMANCE_REGISTERS_LEN];
    populate_payloads_into(&mut wasm, &mut gpu, &mut regs);

    snapshot_round_trip_with_source(&wasm, &gpu, &regs);
}

/// Hardware-gated end-to-end: allocate three cudarc-backed managed
/// buffers (one per snapshot field), populate them via
/// `CudarcUnifiedBuffer::as_mut_slice`, then drive the
/// `SnapshotWriter::capture` → `SnapshotReader::restore` round-trip.
/// The decisive assertion is that bytes scribbled into a cudarc-managed
/// region land in the snapshot payload byte-for-byte, with no aliasing
/// between the three independent allocations.
///
/// Requires a CUDA driver and at least one visible GPU; the S22 runner
/// unignores via `-- --ignored`.
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_unified_buffer_snapshot_round_trip_on_device() {
    let mut wasm_buf =
        CudarcUnifiedBuffer::new(CONFORMANCE_WASM_LEN).expect("alloc wasm-memory cudarc buffer");
    let mut gpu_buf =
        CudarcUnifiedBuffer::new(CONFORMANCE_GPU_LEN).expect("alloc gpu-memory cudarc buffer");
    let mut regs_buf =
        CudarcUnifiedBuffer::new(CONFORMANCE_REGISTERS_LEN).expect("alloc registers cudarc buffer");

    populate_payloads_into(
        wasm_buf.as_mut_slice(),
        gpu_buf.as_mut_slice(),
        regs_buf.as_mut_slice(),
    );

    snapshot_round_trip_with_source(wasm_buf.as_slice(), gpu_buf.as_slice(), regs_buf.as_slice());
}
