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
    let mut pool = Arc::new(UnifiedMemoryPool::new(8 * 1024 * 1024).expect("pool"));
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
    //
    // After audit T4, `UnifiedMemoryPool::reset` takes `&mut self`, so reset
    // can only be attempted on an `Arc<UnifiedMemoryPool>` if every other
    // clone has been dropped (so `Arc::get_mut` returns `Some`). The
    // `creator` here still holds a cloned `Arc<UnifiedMemoryPool>` keepalive,
    // so `Arc::get_mut` MUST return `None` — the &mut-self gate already
    // prevents reset from running before we even check the live counter.
    // This is a stronger guarantee than the old `&self` reset (which would
    // still acquire the interior mutex and return an `Err` based on the live
    // counter); the type system now refuses to rewind a slab that other
    // Arc holders may still be reading.
    assert!(
        Arc::get_mut(&mut pool).is_none(),
        "Arc::get_mut must return None while the creator holds a cloned Arc; \
         this is the type-level refusal to reset that supersedes the old \
         live-allocations check"
    );

    // Drop the creator so we are the last Arc holder. The leaked
    // `PoolAllocation` still keeps `live_allocations() > 0`, so the runtime
    // check inside `reset` must now fire.
    drop(creator);
    let pool_mut = Arc::get_mut(&mut pool)
        .expect("creator dropped, this test holds the sole remaining Arc");
    assert!(
        pool_mut.reset().is_err(),
        "reset must fail because the leaked PoolAllocation keeps live_allocations > 0; \
         operators wanting per-tenant resets must use a per-tenant pool instance"
    );
}
