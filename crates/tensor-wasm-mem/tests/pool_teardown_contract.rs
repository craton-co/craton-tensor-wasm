// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Contract test for the one-shot teardown semantics of pool-backed Wasm
//! linear memories.
//!
//! `TensorWasmMemoryCreator::new_memory` leaks the `PoolAllocation` drop guard
//! when carving a `PooledLinearMemory` (`std::mem::forget(alloc)`) so the
//! bump pointer stays monotonic for the pool's lifetime. The documented
//! consequence is that dropping the resulting linear memory does NOT
//! decrement `live_allocations()` — the counter is sticky once any pool-backed
//! Wasm memory has been issued, and `reset()` will permanently refuse to run
//! on that pool. Operators wanting per-tenant resets must use a per-tenant
//! `UnifiedMemoryPool` instance.
//!
//! This test pins that contract so a future change that "fixes" the leak by
//! decrementing the counter (and thereby silently invalidates pool-backed
//! `base_ptr`s by allowing reset+recarve from the front of the slab) fails CI.

use std::sync::Arc;

use tensor_wasm_mem::pool::UnifiedMemoryPool;
use tensor_wasm_mem::unified::DeviceId;
use tensor_wasm_mem::wasm_memory::TensorWasmMemoryCreator;
use wasmtime::{MemoryCreator, MemoryType};

#[test]
fn dropping_pooled_linear_memory_does_not_decrement_live_count() {
    // 8 MiB slab — large enough for one Wasm linear memory carve.
    let pool = Arc::new(UnifiedMemoryPool::new(8 * 1024 * 1024).expect("pool"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), Arc::clone(&pool));

    assert_eq!(
        pool.live_allocations(),
        0,
        "freshly created pool must have zero live allocations"
    );

    // Carve one pool-backed Wasm linear memory.
    let mt = MemoryType::new(1, Some(2)); // 1 page min, 2 pages max
    let mem = creator
        .new_memory(mt, 64 * 1024, Some(128 * 1024), None, 0)
        .expect("carve from pool");

    assert_eq!(
        pool.live_allocations(),
        1,
        "carving a pool-backed Wasm memory must increment live_allocations"
    );

    // Drop the linear memory. The documented one-shot teardown contract says
    // the live counter must STAY at 1 — the `PoolAllocation` drop guard was
    // intentionally leaked at carve time so the bump pointer remains
    // monotonic for the pool's lifetime.
    drop(mem);

    assert_eq!(
        pool.live_allocations(),
        1,
        "dropping a pool-backed Wasm memory must NOT decrement live_allocations \
         (one-shot teardown contract — `PoolAllocation` is intentionally leaked)"
    );

    // And as the corollary the documentation calls out: `reset` permanently
    // refuses to run on a pool that has ever served a `PooledLinearMemory`.
    assert!(
        pool.reset().is_err(),
        "reset must fail because the leaked PoolAllocation keeps live_allocations > 0; \
         operators wanting per-tenant resets must use a per-tenant pool instance"
    );
}
