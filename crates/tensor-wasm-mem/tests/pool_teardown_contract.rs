// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Contract test for the teardown semantics of pool-backed Wasm linear
//! memories (audit T6).
//!
//! `TensorWasmMemoryCreator::new_memory` leaks the `PoolAllocation` drop guard
//! when carving a `PooledLinearMemory` (`std::mem::forget(alloc)`) so the
//! bump pointer stays monotonic for the pool's lifetime.
//!
//! Before T6, that leak also kept the pool's `live` counter elevated forever
//! once any pool-backed memory had been issued, permanently blocking
//! `UnifiedMemoryPool::reset()` even after every issued memory had been
//! dropped. That left operators with no path to recycle a slab across tenants
//! short of dropping the pool wholesale — a HIGH safety finding because it
//! traded correctness (counter-as-misuse-signal) for an unrecoverable
//! resource leak (slab pinned for the pool's lifetime).
//!
//! T6 fix: `PooledLinearMemory::Drop` now calls `pool.release(offset, size)`
//! from its destructor, mirroring the forgotten `PoolAllocation::Drop`. The
//! `live` counter walks back down to zero as memories are dropped, but the
//! bump pointer is NOT rewound (monotonic-bump semantics preserved). A
//! subsequent `reset()` therefore succeeds — both the `&mut self` gate and
//! the `live == 0` runtime check remain in place; the bug was that `live`
//! could never reach zero, not that the gates themselves were wrong.
//!
//! These tests pin both halves of the new contract: (a) live walks down on
//! drop, (b) reset succeeds only when every issued memory has been dropped
//! AND every `Arc<UnifiedMemoryPool>` clone has been dropped.

use std::sync::Arc;

use tensor_wasm_mem::pool::UnifiedMemoryPool;
use tensor_wasm_mem::unified::DeviceId;
use tensor_wasm_mem::wasm_memory::TensorWasmMemoryCreator;
use wasmtime::{MemoryCreator, MemoryType};

#[test]
fn dropping_pooled_linear_memory_decrements_live_count() {
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

    // Audit T6 contract: dropping the linear memory walks the live counter
    // back down. The `PoolAllocation` drop guard was leaked at carve time so
    // the bump pointer remains monotonic, but `PooledLinearMemory::Drop` now
    // calls `pool.release(offset, size)` to mirror the leaked drop's effect
    // on the counter.
    drop(mem);

    assert_eq!(
        pool.live_allocations(),
        0,
        "dropping a pool-backed Wasm memory must decrement live_allocations \
         (audit T6: `PooledLinearMemory::Drop` mirrors the leaked \
         `PoolAllocation::Drop` so a subsequent reset can run)"
    );
}

#[test]
fn reset_succeeds_after_all_pooled_memories_drop() {
    // The headline regression test for the audit T6 finding: a pool that has
    // served pool-backed Wasm linear memories MUST be resettable once every
    // issued memory has been dropped.
    let mut pool = Arc::new(UnifiedMemoryPool::new(8 * 1024 * 1024).expect("pool"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), Arc::clone(&pool));

    let mt = MemoryType::new(1, Some(2));
    let mem = creator
        .new_memory(mt, 64 * 1024, Some(128 * 1024), None, 0)
        .expect("carve from pool");
    assert_eq!(pool.live_allocations(), 1);

    drop(mem);
    assert_eq!(pool.live_allocations(), 0);

    // The creator still holds an `Arc<UnifiedMemoryPool>` clone, so the
    // `&mut self` gate on `reset()` refuses to engage. This is the type-level
    // half of the contract: even if `live == 0`, `reset` cannot run while any
    // other `Arc` clone exists, because `Arc::get_mut` returns `None`.
    assert!(
        Arc::get_mut(&mut pool).is_none(),
        "Arc::get_mut must return None while the creator holds a cloned Arc"
    );

    // Drop the creator; we are now the sole `Arc` holder.
    drop(creator);
    let pool_mut = Arc::get_mut(&mut pool)
        .expect("creator dropped, this test holds the sole remaining Arc");

    // Headline assertion: reset succeeds. Before T6, this would fail because
    // `live_allocations()` would still be 1 — permanently — even though we
    // dropped the only issued memory.
    pool_mut
        .reset()
        .expect("reset must succeed after all pool-backed memories are dropped (audit T6)");

    // Sanity: the bump pointer was rewound by reset, so the full slab is
    // available again. The bump pointer was NOT rewound by the earlier
    // `drop(mem)` — monotonic-bump semantics held until the explicit reset.
    assert_eq!(pool_mut.remaining(), pool_mut.capacity());
}

#[test]
fn reset_blocked_while_one_pooled_memory_outstanding() {
    // Two carves; drop only one; reset must still fail because `live == 1`.
    // Drop the second; reset must then succeed. This is the regression test
    // for the per-memory accounting: T6's fix must decrement `live` exactly
    // once per drop, not collapse the counter to zero on the first drop.
    let mut pool = Arc::new(UnifiedMemoryPool::new(8 * 1024 * 1024).expect("pool"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), Arc::clone(&pool));

    let mem_a = creator
        .new_memory(MemoryType::new(1, Some(2)), 64 * 1024, Some(128 * 1024), None, 0)
        .expect("carve A");
    let mem_b = creator
        .new_memory(MemoryType::new(1, Some(2)), 64 * 1024, Some(128 * 1024), None, 0)
        .expect("carve B");
    assert_eq!(pool.live_allocations(), 2);

    drop(mem_a);
    assert_eq!(
        pool.live_allocations(),
        1,
        "dropping one of two pool-backed memories must decrement live by exactly one"
    );

    // Drop the creator so it no longer holds a clone — the only remaining
    // pool-keepalives are this test's `Arc` and the one inside `mem_b`.
    drop(creator);

    // `Arc::get_mut` must still refuse because `mem_b`'s
    // `Arc<UnifiedMemoryPool>` keepalive is alive.
    assert!(
        Arc::get_mut(&mut pool).is_none(),
        "Arc::get_mut must return None while a PooledLinearMemory holds a clone"
    );

    // Drop the second memory; `live` returns to zero AND the only remaining
    // `Arc` clone is this test's.
    drop(mem_b);
    assert_eq!(pool.live_allocations(), 0);
    let pool_mut = Arc::get_mut(&mut pool)
        .expect("all PooledLinearMemory keepalives dropped; we hold the sole Arc");
    pool_mut
        .reset()
        .expect("reset must succeed once every issued memory is dropped (audit T6)");
}

#[test]
fn bump_pointer_not_rewound_by_drop() {
    // T6 must NOT rewind the bump pointer on drop — only `reset()` does
    // that. This is the monotonic-bump invariant: a second carve after a
    // drop must land at a fresh offset, not reuse the just-released region.
    // (Reusing the region would race any host-side pointer still living in
    // application code that hasn't yet flushed the drop through its own
    // drop chain.)
    let pool = Arc::new(UnifiedMemoryPool::new(8 * 1024 * 1024).expect("pool"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), Arc::clone(&pool));

    let remaining_before = pool.remaining();
    let mt = MemoryType::new(1, Some(2));
    let mem = creator
        .new_memory(mt, 64 * 1024, Some(128 * 1024), None, 0)
        .expect("carve");
    let remaining_after_carve = pool.remaining();
    assert!(
        remaining_after_carve < remaining_before,
        "carve must consume slab space"
    );

    // Drop the memory. `live` walks down, but `remaining` does NOT walk back
    // up — the bump pointer is intentionally not rewound.
    drop(mem);
    assert_eq!(pool.live_allocations(), 0);
    assert_eq!(
        pool.remaining(),
        remaining_after_carve,
        "PooledLinearMemory::Drop must NOT rewind the bump pointer \
         (monotonic-bump invariant; only `reset()` rewinds)"
    );
}
