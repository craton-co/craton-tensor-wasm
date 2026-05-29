// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W4.2 cross-backend snapshot conformance: cuda-oxide half.
//!
//! Compiled only when the dep-less `cuda-oxide-backend` scaffold feature
//! is on.
//!
//! Mirrors [`cust_snapshot_conformance.rs`][cust] and
//! [`cudarc_snapshot_conformance.rs`][cudarc] so the diff between the
//! three backends is body-only.
//!
//! Two test bands:
//!
//! 1. A pure-`Vec<u8>` round-trip proving the snapshot wire format stays
//!    byte-stable when the cuda-oxide module is compiled into the test
//!    binary.
//!
//! 2. Witness that `CudaOxideUnifiedBuffer::allocate` returns the
//!    documented `NOT_YET_WIRED` sentinel — and that the snapshot path
//!    therefore is unreachable through this backend until the host port
//!    lands.
//!
//! NOTE: the hardware-gated band that exercised the real
//! `cuMemAllocManaged` host backend (formerly gated on the
//! `experimental-cuda-oxide-host-backend` feature) was removed for
//! crates.io publishability — it relied on git-pinned cuda-oxide crates.
//! Restore it with the host port (RFC 0001 / docs/CUDA-OXIDE-CUTOVER.md).
//!
//! [cust]: ./cust_snapshot_conformance.rs
//! [cudarc]: ./cudarc_snapshot_conformance.rs

#![cfg(feature = "cuda-oxide-backend")]

use tensor_wasm_mem::cuda_oxide_backend::CudaOxideUnifiedBuffer;
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

// NOTE: the hardware-gated end-to-end test that allocated three real
// `cuMemAllocManaged` regions via the cuda-oxide host backend was removed
// together with the `experimental-cuda-oxide-host-backend` feature and the
// git-pinned cuda-oxide crates it depended on, which blocked crates.io
// publishing. Restore it with the host port (RFC 0001 /
// docs/CUDA-OXIDE-CUTOVER.md).
