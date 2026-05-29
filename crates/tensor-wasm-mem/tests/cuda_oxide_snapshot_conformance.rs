// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W4.2 cross-backend snapshot conformance: cuda-oxide half.
//!
//! Compiled only when the dep-less `cuda-oxide-backend` scaffold feature
//! is on, and split internally so the hardware-touching path is
//! additionally gated on the strict-superset `cuda-oxide-host-backend`
//! feature (which pulls in the W4.1 `cuda-host` / `cuda-core` git deps —
//! the LIBCLANG + CUDA Toolkit prerequisites are documented in
//! `docs/CUDA-SETUP.md`).
//!
//! Mirrors [`cust_snapshot_conformance.rs`][cust] and
//! [`cudarc_snapshot_conformance.rs`][cudarc] so the diff between the
//! three backends is body-only.
//!
//! Three test bands:
//!
//! 1. **Host-only, scaffold OR host-backend.** A pure-`Vec<u8>`
//!    round-trip proving the snapshot wire format stays byte-stable
//!    when the cuda-oxide module is compiled into the test binary.
//!
//! 2. **Host-only, scaffold-only.** Witness that
//!    `CudaOxideUnifiedBuffer::allocate` returns the documented
//!    `NOT_YET_WIRED` sentinel — and that the snapshot path therefore
//!    is unreachable through this backend until the host-backend feature
//!    lands. Skipped under the host-backend build because that build
//!    either succeeds or returns a real driver error, never the
//!    sentinel string.
//!
//! 3. **Hardware-gated, host-backend only.** Allocate three real
//!    cuMemAllocManaged regions via the W4.1 host-backend impl,
//!    populate them, and run the same round-trip. Marked `#[ignore =
//!    "requires CUDA hardware"]`. The S22 runner unignores via `cargo
//!    test --features cuda-oxide-host-backend -- --ignored`.
//!
//! [cust]: ./cust_snapshot_conformance.rs
//! [cudarc]: ./cudarc_snapshot_conformance.rs

#![cfg(feature = "cuda-oxide-backend")]

use tensor_wasm_mem::cuda_oxide_backend::CudaOxideUnifiedBuffer;
// `UnifiedError` is only referenced by the scaffold-only negative-witness
// test below; under `cuda-oxide-host-backend` the import would be unused
// and would warn under `-D warnings`.
#[cfg(not(feature = "experimental-cuda-oxide-host-backend"))]
use tensor_wasm_mem::unified::UnifiedError;

mod common;

use common::{
    populate_payloads_into, snapshot_round_trip_with_source, CONFORMANCE_GPU_LEN,
    CONFORMANCE_REGISTERS_LEN, CONFORMANCE_WASM_LEN,
};

/// Pure-`Vec<u8>` round-trip compiled under `--features
/// cuda-oxide-backend`.
///
/// Active on both the scaffold and host-backend builds because the
/// snapshot path itself does not touch the cuda-oxide module. Catches
/// the regression where enabling the cuda-oxide feature accidentally
/// pulls a `serde` re-export or a tweaked `bincode` config into scope
/// and changes the on-disk format — RFC 0001's "feature-gated
/// production code must not change downstream behaviour" guard.
#[test]
fn cuda_oxide_feature_does_not_perturb_snapshot_wire_format() {
    let mut wasm = vec![0u8; CONFORMANCE_WASM_LEN];
    let mut gpu = vec![0u8; CONFORMANCE_GPU_LEN];
    let mut regs = vec![0u8; CONFORMANCE_REGISTERS_LEN];
    populate_payloads_into(&mut wasm, &mut gpu, &mut regs);

    snapshot_round_trip_with_source(&wasm, &gpu, &regs);
}

/// Under the dep-less scaffold build, `CudaOxideUnifiedBuffer::allocate`
/// returns the documented sentinel string and the snapshot producer
/// path is therefore unreachable through this backend. Witness that
/// behaviour explicitly so future scaffold tweaks cannot silently
/// regress the contract (e.g. someone making allocate succeed with a
/// fake host pointer that the snapshot writer would then happily
/// serialise — a v0.5 cust-successor compat hazard called out in RFC
/// 0001).
///
/// Skipped under `cuda-oxide-host-backend` because that build *does*
/// have a working `allocate` (which either succeeds or returns a real
/// driver error — never the sentinel).
#[cfg(not(feature = "experimental-cuda-oxide-host-backend"))]
#[test]
fn cuda_oxide_scaffold_allocate_blocks_snapshot_producer_path() {
    let err = CudaOxideUnifiedBuffer::allocate(CONFORMANCE_WASM_LEN)
        .expect_err("scaffold allocate must error until host-backend port");
    match err {
        UnifiedError::Cuda(msg) => {
            assert!(
                msg.contains("not yet wired"),
                "expected sentinel error string, got: {msg}",
            );
            assert!(
                msg.contains("RFC 0001"),
                "expected RFC reference in sentinel error, got: {msg}",
            );
        }
        other => panic!("expected UnifiedError::Cuda(NOT_YET_WIRED), got {other:?}",),
    }
}

/// Hardware-gated end-to-end: allocate three real `cuMemAllocManaged`
/// regions via the W4.1 cuda-oxide host backend, populate them via
/// `CudaOxideUnifiedBuffer::as_mut_slice`, and round-trip through
/// `SnapshotWriter::capture` → `SnapshotReader::restore`. The
/// decisive assertion is the cross-backend conformance guarantee: a
/// snapshot blob produced from cuda-oxide-allocated bytes is
/// bit-identical (in the restored-payload sense) to one produced from
/// cust- or cudarc-allocated bytes carrying the same logical content.
///
/// Compiled only under `cuda-oxide-host-backend` (the strict-superset
/// W4.1 feature). Marked `#[ignore]` per the repo convention; runs on
/// the S22 CUDA runner via `cargo test --features
/// cuda-oxide-host-backend -- --ignored`.
#[cfg(feature = "experimental-cuda-oxide-host-backend")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_oxide_host_backend_snapshot_round_trip_on_device() {
    let mut wasm_buf = CudaOxideUnifiedBuffer::allocate(CONFORMANCE_WASM_LEN)
        .expect("alloc wasm-memory cuda-oxide buffer");
    let mut gpu_buf = CudaOxideUnifiedBuffer::allocate(CONFORMANCE_GPU_LEN)
        .expect("alloc gpu-memory cuda-oxide buffer");
    let mut regs_buf = CudaOxideUnifiedBuffer::allocate(CONFORMANCE_REGISTERS_LEN)
        .expect("alloc registers cuda-oxide buffer");

    populate_payloads_into(
        wasm_buf.as_mut_slice(),
        gpu_buf.as_mut_slice(),
        regs_buf.as_mut_slice(),
    );

    snapshot_round_trip_with_source(wasm_buf.as_slice(), gpu_buf.as_slice(), regs_buf.as_slice());
}
