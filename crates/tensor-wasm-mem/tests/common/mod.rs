// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Shared helpers for the W4.2 cross-backend snapshot conformance suite.
//!
//! The W1.3 [cross-version snapshot compat suite][compat] in
//! `tensor-wasm-snapshot/tests/compat.rs` already proves the on-disk format
//! is **backend-independent** — it reads checked-in golden fixtures with
//! the current reader and asserts byte-level equivalence. What it does
//! *not* exercise is the **producer** side under each CUDA backing: when
//! the bytes destined for the snapshot blob come out of a real
//! `cuMemAllocManaged` region (cust, cudarc, or cuda-oxide host), do they
//! still serialise to the identical wire format that golden-fixture
//! consumers expect?
//!
//! That is the regression guard this module enables. It exposes a single
//! generic helper, [`snapshot_round_trip_with_source`], that drives the
//! `SnapshotWriter` → `SnapshotReader` round-trip against three byte
//! sources supplied by the caller. Three sibling test files
//! (`cust_snapshot_conformance.rs`, `cudarc_snapshot_conformance.rs`,
//! `cuda_oxide_snapshot_conformance.rs`) each compile under a different
//! backend feature, allocate buffers via that backend's primitive,
//! populate them with deterministic patterns, and call the helper. The
//! helper asserts the captured blob is decode-able and the restored
//! payload is byte-for-byte identical to the input.
//!
//! Where applicable, tests that touch a real CUDA driver are marked
//! `#[ignore = "requires CUDA hardware"]` per the repo convention so a
//! host-only contributor box runs the *unignored* portion (the
//! pure-`Vec<u8>` baseline that proves the snapshot format itself is
//! backend-stable) without dlopen-ing libcuda.
//!
//! [compat]: ../../tensor-wasm-snapshot/tests/compat.rs
//!
//! # Why this lives in `tensor-wasm-mem` rather than `tensor-wasm-snapshot`
//!
//! The backend allocators (`UnifiedBuffer`, `CudarcUnifiedBuffer`,
//! `CudaOxideUnifiedBuffer`) are defined here; the snapshot crate is
//! deliberately decoupled from any CUDA backing so it remains
//! `cargo build`-able on hosts without a toolkit. The conformance
//! direction we want to exercise is *backend → snapshot*, so the test
//! belongs in the crate that knows what the backends are.

// Some backend-gated test binaries do not pull in every helper. Suppressing
// the per-binary `dead_code` warning here keeps the module reusable across
// the three sibling test files without needing per-helper `#[allow]`
// attributes that would have to be kept in sync.
#![allow(dead_code)]

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{
    InstanceState, SnapshotWriter, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// Deterministic tenant ID embedded into every conformance snapshot.
/// Chosen to share no nibble pattern with `CONFORMANCE_INSTANCE_ID` so a
/// byte-swap regression in the metadata encoder is obvious from the
/// failure diff.
pub const CONFORMANCE_TENANT_ID: u64 = 0xB4CC_0AF2_DEAD_BEEF;

/// Deterministic instance ID embedded into every conformance snapshot.
pub const CONFORMANCE_INSTANCE_ID: u128 = 0x1234_5678_9ABC_DEF0_F00D_BABE_CAFE_5A5A;

/// Deterministic wasm-memory payload size in bytes used by the conformance
/// suite. Sized at 8 KiB so:
///   - it is large enough to exercise zstd's dictionary path (>4 KiB
///     compresses meaningfully);
///   - the pattern wraps the byte domain (256) several times to catch any
///     off-by-256 indexing error;
///   - the total round-trip stays well under the cust per-managed-buffer
///     test budget (16 MiB).
pub const CONFORMANCE_WASM_LEN: usize = 8 * 1024;

/// Deterministic gpu-memory payload size in bytes. Smaller than the wasm
/// payload so a swap-of-fields regression in the writer struct is caught
/// by a length mismatch in the assertion.
pub const CONFORMANCE_GPU_LEN: usize = 4 * 1024;

/// Deterministic register-blob size in bytes. Tiny by design — registers
/// are typically a few hundred bytes in practice and the value here
/// prevents the test from accidentally turning into a compression
/// benchmark.
pub const CONFORMANCE_REGISTERS_LEN: usize = 512;

/// Fill `dst` with the wasm-memory byte pattern used by the conformance
/// suite. The pattern is `(i % 251) as u8` — `251` is the largest prime
/// below 256, which guarantees every byte value in `0..251` appears
/// within any contiguous 251-byte window. Choosing a prime modulus also
/// rules out accidental alignment-coincidence bugs (a power-of-two
/// modulus would mask off-by-`page_size` errors).
///
/// Returns `dst` for the convenience of in-line use.
pub fn fill_wasm_pattern(dst: &mut [u8]) -> &mut [u8] {
    for (i, b) in dst.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    dst
}

/// Fill `dst` with the gpu-memory byte pattern used by the conformance
/// suite. Distinct from [`fill_wasm_pattern`] so a swap-of-fields bug in
/// the writer would surface as both `wasm_memory` and `gpu_memory`
/// assertions failing simultaneously rather than passing by coincidence.
pub fn fill_gpu_pattern(dst: &mut [u8]) -> &mut [u8] {
    for (i, b) in dst.iter_mut().enumerate() {
        *b = (((i as u32).wrapping_mul(17)) % 253) as u8;
    }
    dst
}

/// Fill `dst` with the register byte pattern used by the conformance
/// suite. XOR-stamped so it is visibly distinct from the other two
/// payloads when staring at a hex dump.
pub fn fill_registers_pattern(dst: &mut [u8]) -> &mut [u8] {
    for (i, b) in dst.iter_mut().enumerate() {
        *b = ((i ^ 0xA5) & 0xFF) as u8;
    }
    dst
}

/// Drive a full `SnapshotWriter::capture` → `SnapshotReader::restore`
/// round-trip against caller-supplied byte slices and assert the
/// recovered payload is byte-for-byte identical to the input.
///
/// The slices may be backed by *any* allocator — a heap `Vec<u8>`, a
/// cust `UnifiedBuffer`, a cudarc `CudarcUnifiedBuffer`, or a
/// `CudaOxideUnifiedBuffer`. The whole point of the helper is that the
/// caller's choice of backing must not influence the wire format
/// produced by the writer or the bytes recovered by the reader.
///
/// # Why no `&InstanceState` parameter
///
/// The InstanceState shape is fixed across backends; only the bytes
/// vary. The helper builds the InstanceState itself with deterministic
/// tenant/instance IDs ([`CONFORMANCE_TENANT_ID`] /
/// [`CONFORMANCE_INSTANCE_ID`]) so the assertions can pin those fields
/// without each call site having to repeat them.
pub fn snapshot_round_trip_with_source(
    wasm_bytes: &[u8],
    gpu_bytes: &[u8],
    registers_bytes: &[u8],
) {
    let writer = SnapshotWriter::new();
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(CONFORMANCE_TENANT_ID),
            instance_id: InstanceId(CONFORMANCE_INSTANCE_ID),
            wasm_memory: wasm_bytes,
            gpu_memory: gpu_bytes,
            registers: registers_bytes,
        })
        .expect("SnapshotWriter::capture must succeed for valid InstanceState");

    let restored = SnapshotReader::new()
        .restore(&blob)
        .expect("SnapshotReader::restore must accept a freshly captured blob");

    assert_eq!(
        restored.magic, SNAPSHOT_MAGIC,
        "magic must match the in-source constant for any backend",
    );
    assert_eq!(
        restored.version, SNAPSHOT_VERSION,
        "version must match the in-source constant for any backend",
    );
    assert_eq!(
        restored.metadata.tenant_id,
        TenantId(CONFORMANCE_TENANT_ID),
        "tenant_id must survive the round-trip exactly",
    );
    assert_eq!(
        restored.metadata.instance_id,
        InstanceId(CONFORMANCE_INSTANCE_ID),
        "instance_id must survive the round-trip exactly",
    );
    assert_eq!(
        restored.wasm_memory, wasm_bytes,
        "wasm_memory bytes must round-trip identically regardless of backend",
    );
    assert_eq!(
        restored.gpu_memory, gpu_bytes,
        "gpu_memory bytes must round-trip identically regardless of backend",
    );
    assert_eq!(
        restored.registers, registers_bytes,
        "registers bytes must round-trip identically regardless of backend",
    );
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm_bytes.len() + gpu_bytes.len() + registers_bytes.len()) as u64,
        "total_uncompressed_bytes must equal the sum of the three payload \
         lengths the writer was handed",
    );
}

/// Build the three deterministic byte payloads (wasm, gpu, registers)
/// used by the conformance suite into caller-supplied destination
/// slices.
///
/// The destinations are passed in rather than allocated here so each
/// backend's test can route the writes through its own allocator
/// (`UnifiedBuffer::as_mut_slice`, `CudarcUnifiedBuffer::as_mut_slice`,
/// `CudaOxideUnifiedBuffer::as_mut_slice`) and the helper stays
/// allocator-agnostic.
///
/// Panics if any destination length differs from the corresponding
/// `CONFORMANCE_*_LEN` constant — that mismatch indicates a test wrote
/// the wrong setup code, not a backend regression.
pub fn populate_payloads_into(wasm_dst: &mut [u8], gpu_dst: &mut [u8], registers_dst: &mut [u8]) {
    assert_eq!(
        wasm_dst.len(),
        CONFORMANCE_WASM_LEN,
        "wasm payload destination must be exactly {} bytes",
        CONFORMANCE_WASM_LEN,
    );
    assert_eq!(
        gpu_dst.len(),
        CONFORMANCE_GPU_LEN,
        "gpu payload destination must be exactly {} bytes",
        CONFORMANCE_GPU_LEN,
    );
    assert_eq!(
        registers_dst.len(),
        CONFORMANCE_REGISTERS_LEN,
        "registers payload destination must be exactly {} bytes",
        CONFORMANCE_REGISTERS_LEN,
    );
    fill_wasm_pattern(wasm_dst);
    fill_gpu_pattern(gpu_dst);
    fill_registers_pattern(registers_dst);
}
