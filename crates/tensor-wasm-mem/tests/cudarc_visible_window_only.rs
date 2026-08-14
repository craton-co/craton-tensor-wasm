// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Audit T14 regression: the cudarc `Backing::allocate` path zeroes only
//! the visible window, NOT the whole allocation.
//!
//! Before T14, the cudarc backing followed its `init_zero_bytes` fill with
//! an unconditional `ptr::write_bytes(ptr, 0, size)` over the entire
//! allocation. That defeated the `visible_bytes` optimisation that
//! motivates
//! [`tensor_wasm_mem::unified::UnifiedBuffer::new_with_visible_window_on`]:
//! a 256 MiB Wasm linear-memory spawn would pay a full 256 MiB memset
//! even when only one 64 KiB Wasm page was visible. The audit flagged
//! this as a correctness-vs-performance contradiction in the doc claim.
//!
//! T14 drops the redundant full-allocation memset. The audit H2 invariant
//! (no cross-tenant data leak) is preserved by the layer above:
//!
//! - The pool path
//!   ([`tensor_wasm_mem::pool::UnifiedMemoryPool::allocate`]) zeroes
//!   every carved `[offset, offset + size)` region before returning the
//!   `PoolAllocation` to the tenant. That defends slabs that are
//!   recycled across tenants (audit H1).
//! - The direct linear-memory path
//!   ([`tensor_wasm_mem::wasm_memory::TensorWasmLinearMemory`]) zeroes the
//!   visible window at construction time (via the cudarc backing's
//!   `init_zero_bytes` fill) and zeroes bytes freshly exposed by
//!   `memory.grow` in `grow_to` itself.
//!
//! This test pins the pool half of that contract specifically against the
//! cudarc backing: a tenant writes a sentinel near the end of a 2 MiB
//! pool allocation, releases it, the pool is reset, and a fresh
//! allocation lands on the same byte range. The recycled bytes MUST read
//! as zero. The companion in-crate test `recycled_allocation_reads_as_zero`
//! (in `src/pool.rs`) exercises the same invariant on whatever backing is
//! active in the no-feature default build; this test re-pins it under
//! `--features cudarc-backend` so a future regression that re-introduces a
//! cudarc-specific "all bytes start zero" assumption (and therefore drops
//! the pool's per-allocation zero-fill) cannot sneak past the cudarc-only
//! CI matrix entry.
//!
//! Run with hardware:
//!
//! ```ignore
//! cargo test -p tensor-wasm-mem --features cudarc-backend \
//!     --test cudarc_visible_window_only -- --ignored
//! ```

#![cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]

use tensor_wasm_mem::pool::UnifiedMemoryPool;

/// 2 MiB allocation, sentinel written at the last byte of the issued
/// region, slab recycled across a `reset()`, second allocation must read
/// the sentinel offset as zero. The 4 KiB / 2 MiB sizing in the test
/// name matches the audit-T14 description: the "visible window" the
/// optimisation cares about is typically one Wasm page (4-64 KiB) while
/// the full allocation is megabytes — large enough that any off-by-one
/// in the pool's per-allocation zero-fill would surface here.
///
/// Marked `#[ignore]` because allocating a 4 MiB cudarc-backed pool slab
/// hits `cuMemAllocManaged`, which requires a working CUDA driver AND the
/// `cudarc-backend` feature flag (this whole file is `#![cfg]`-gated on it).
///
/// HARDWARE NOTE (mem finding 7): this box now HAS a CUDA GPU (RTX 2060,
/// sm_75, CUDA 13.2), so this test is hardware-runnable today via
/// `cargo test -p tensor-wasm-mem --features cudarc-backend \
///   --test cudarc_visible_window_only -- --ignored`. It stays `#[ignore]`d
/// by default because it still needs the non-default feature flag and a
/// driver, so the plain host-only `cargo test` must not attempt it. The
/// backing-agnostic half of this contract (recycled bytes read zero, and the
/// fresh-vs-recycled memset split) is additionally pinned WITHOUT hardware by
/// `src/pool.rs`'s `recycled_allocation_reads_as_zero` and
/// `fresh_then_recycled_carve_reads_zero_across_high_water` unit tests.
#[test]
#[ignore = "requires CUDA hardware + --features cudarc-backend (GPU now present; run with -- --ignored)"]
fn cudarc_allocate_does_not_zero_beyond_visible_window() {
    const ALLOC_SIZE: usize = 2 * 1024 * 1024; // 2 MiB
    const SLAB_SIZE: usize = 4 * 1024 * 1024; // 4 MiB — fits two non-overlapping carves.

    // `reset` takes `&mut self` (audit T4); keep the pool mutable so we can
    // recycle it across the two carves.
    let mut pool = UnifiedMemoryPool::new(SLAB_SIZE).expect("alloc 4 MiB cudarc-backed pool slab");

    // Tenant A: write a sentinel at the last byte of the allocation. The
    // pre-T14 cudarc backing zeroed the entire slab at allocation time;
    // post-T14 it zeroes only the visible window (which equals `size` on
    // the slab-construction path because `UnifiedMemoryPool::new_on` calls
    // `UnifiedBuffer::new_on`, which passes `visible_bytes = size`). Either
    // way the per-PoolAllocation zero-fill — exercised here — is what
    // protects tenant B from observing tenant A's sentinel.
    let offset_a = {
        let mut a = pool.allocate(ALLOC_SIZE, 64).expect("tenant A allocate");
        let offset = a.offset();
        let slice = a.as_mut_slice();
        // Sentinel at the last byte; if the pool's zero-fill is one byte
        // short, this exact byte is where the leak would surface.
        slice[ALLOC_SIZE - 1] = 0xAB;
        // Also poison the middle so we catch any "zero-fill covered the
        // first and last cacheline but skipped the body" regression.
        slice[ALLOC_SIZE / 2] = 0xCD;
        assert_eq!(slice[ALLOC_SIZE - 1], 0xAB, "sentinel write must take");
        assert_eq!(slice[ALLOC_SIZE / 2], 0xCD, "mid poison must take");
        offset
    };

    // Release tenant A and rewind the bump pointer so the next allocate()
    // can land on the same VA range.
    pool.reset().expect("reset after tenant A drops");

    // Tenant B: same shape, same alignment. By the bump-allocator
    // discipline plus the post-reset bump=0 invariant, tenant B's region
    // overlaps tenant A's byte-for-byte.
    let b = pool.allocate(ALLOC_SIZE, 64).expect("tenant B allocate");
    assert_eq!(
        b.offset(),
        offset_a,
        "tenant B must land on tenant A's offset for the test to be meaningful",
    );

    let slice = b.as_slice();
    assert_eq!(
        slice[ALLOC_SIZE - 1],
        0,
        "tenant B observed tenant A's sentinel at the last byte — \
         the pool's per-allocation zero-fill regressed (or the cudarc backing's \
         T14 visible-window-only contract was tightened without restoring the \
         full-slab memset). See crates/tensor-wasm-mem/src/pool.rs::allocate \
         and crates/tensor-wasm-mem/src/unified.rs::backing::Backing::allocate.",
    );
    assert_eq!(
        slice[ALLOC_SIZE / 2],
        0,
        "tenant B observed tenant A's mid-region poison — same regression as the \
         last-byte assertion above; the zero-fill is missing the interior.",
    );
    // Spot-check a few more offsets to defend against a regression that
    // zeroes only the boundary cachelines.
    for &probe in &[
        0usize,
        1,
        4095,
        4096,
        65_536,
        ALLOC_SIZE / 4,
        ALLOC_SIZE - 64,
    ] {
        assert_eq!(
            slice[probe], 0,
            "tenant B observed non-zero byte at offset {probe} — pool zero-fill regression",
        );
    }
}
