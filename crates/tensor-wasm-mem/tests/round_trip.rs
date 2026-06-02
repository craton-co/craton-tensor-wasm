// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Cross-buffer round-trip integration test for `tensor-wasm-mem`.

use tensor_wasm_mem::pinned_host::GuardedHostBuffer;
use tensor_wasm_mem::pool::UnifiedMemoryPool;
use tensor_wasm_mem::unified::UnifiedBuffer;

#[test]
fn unified_buffer_round_trip() {
    let mut b = UnifiedBuffer::new(256).expect("alloc");
    for (i, byte) in b.as_mut_slice().iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }
    for (i, &byte) in b.as_slice().iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8);
    }
}

#[test]
fn guarded_host_buffer_round_trip() {
    let mut b = GuardedHostBuffer::new(256).expect("alloc");
    b.as_mut_slice().copy_from_slice(&[0xFFu8; 256]);
    assert!(b.as_slice().iter().all(|&v| v == 0xFF));
}

#[test]
fn pool_disjoint_allocations() {
    let pool = UnifiedMemoryPool::new(64 * 1024).expect("alloc");
    let mut handles = Vec::new();
    for i in 0..16 {
        let mut h = pool.allocate(1024, 64).expect("alloc from pool");
        h.as_mut_slice().fill(i as u8);
        handles.push(h);
    }
    for (i, h) in handles.iter().enumerate() {
        assert!(
            h.as_slice().iter().all(|&v| v == i as u8),
            "alloc {i} corrupted"
        );
    }
}

#[test]
fn pool_round_trip_after_reset() {
    // `reset` takes `&mut self` (audit T4) — the binding must be `mut` so the
    // test can rewind the bump pointer between the allocation and remaining-
    // capacity assertions. The allocations are dropped at the end of the
    // inner block, so no `&self` borrow is live at the `reset` call site.
    let mut pool = UnifiedMemoryPool::new(8 * 1024).unwrap();
    {
        let _a = pool.allocate(1024, 16).unwrap();
        let _b = pool.allocate(1024, 16).unwrap();
        assert_eq!(pool.live_allocations(), 2);
    }
    pool.reset().expect("reset after drop");
    assert_eq!(pool.remaining(), pool.capacity());
}
