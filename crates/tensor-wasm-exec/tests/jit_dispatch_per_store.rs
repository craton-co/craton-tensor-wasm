//! Regression test for exec S-6 (cross-tenant arena pollution).
//!
//! Before the fix, [`add_jit_dispatch_to_linker`] captured an
//! `Arc<Mutex<ArenaState>>` shared across every instance the linker
//! instantiated, so two tenants instantiated from the same `Linker`
//! observed each other's bump cursor and LIFO `live` stack. With the
//! arena moved into the per-store [`InstanceState`] payload, two stores
//! built from the same linker must each see an independent arena: the
//! first alloc on each store must produce the same pointer (top of
//! its own memory minus its own allocation), and the bump cursors
//! must be independent — alloc'ing on one must NOT shift the cursor
//! on the other.
//!
//! See `crates/tensor-wasm-exec/src/jit_dispatch.rs` and
//! `crates/tensor-wasm-exec/src/instance.rs` for the production
//! wiring; this test exercises both at once through a single shared
//! linker.

use std::sync::Arc;

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_exec::instance::InstanceState;
use tensor_wasm_exec::jit_dispatch::add_jit_dispatch_to_linker;
use tensor_wasm_jit::cache::KernelCache;
use wasmtime::{Config, Engine, Linker, Module, Store};

/// Minimal Wasm module that imports the JIT alloc trio and re-exports
/// `alloc(size) -> ptr`. Memory size is one page so the arena's lazy
/// grow path runs deterministically.
const DRIVER_WAT: &str = r#"
    (module
      (import "tensor-wasm:jit/host" "dispatch"
        (func $d (param i64 i64 i32 i32 i32) (result i32)))
      (import "tensor-wasm:jit/host" "alloc"
        (func $a (param i32) (result i32)))
      (import "tensor-wasm:jit/host" "free"
        (func $f (param i32 i32)))
      (memory (export "memory") 1)
      (func (export "drive_alloc") (param $sz i32) (result i32)
        (call $a (local.get $sz)))
    )
"#;

fn make_engine() -> Engine {
    let mut cfg = Config::new();
    cfg.wasm_simd(true);
    Engine::new(&cfg).expect("engine")
}

fn instance_state(tenant: u64, id: u64) -> InstanceState {
    InstanceState::new(TenantId(tenant), InstanceId(id as u128))
}

/// Two stores from the SAME linker must not share an arena.
///
/// We alloc on store A first, then on store B, and assert:
/// 1. Both allocations succeed (non-negative pointer).
/// 2. The two pointers are equal — each store starts from a fresh
///    cursor at the top of its own (identically-sized) memory, so the
///    first alloc of the same size lands at the same offset. If the
///    arena were shared, B's alloc would land BELOW A's pointer.
#[test]
fn two_stores_sharing_linker_have_independent_arenas() {
    let engine = make_engine();
    let cache = Arc::new(KernelCache::new());

    let mut linker: Linker<InstanceState> = Linker::new(&engine);
    add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");

    let wasm = wat::parse_str(DRIVER_WAT).expect("wat");
    let module = Module::new(&engine, &wasm).expect("module");

    // Tenant A.
    let mut store_a = Store::new(&engine, instance_state(1, 100));
    let inst_a = linker
        .instantiate(&mut store_a, &module)
        .expect("instantiate A");
    let alloc_a = inst_a
        .get_typed_func::<i32, i32>(&mut store_a, "drive_alloc")
        .expect("typed A");

    // Tenant B — instantiated from the SAME linker. This is the
    // configuration that triggered the pre-fix pollution bug.
    let mut store_b = Store::new(&engine, instance_state(2, 200));
    let inst_b = linker
        .instantiate(&mut store_b, &module)
        .expect("instantiate B");
    let alloc_b = inst_b
        .get_typed_func::<i32, i32>(&mut store_b, "drive_alloc")
        .expect("typed B");

    let ptr_a = alloc_a.call(&mut store_a, 64).expect("alloc A");
    let ptr_b = alloc_b.call(&mut store_b, 64).expect("alloc B");

    assert!(ptr_a > 0, "alloc on store A must succeed, got {ptr_a}");
    assert!(ptr_b > 0, "alloc on store B must succeed, got {ptr_b}");
    assert_eq!(
        ptr_a, ptr_b,
        "two stores with identical memory layouts must allocate at the same offset \
         from their independent bump cursors (pre-fix, B's pointer would be below A's)"
    );

    // Confirm the per-store arenas reflect each store's allocation
    // independently — neither store sees the other's cursor.
    let cursor_a = store_a.data().jit_arena().bump_cursor();
    let cursor_b = store_b.data().jit_arena().bump_cursor();
    assert_eq!(
        cursor_a, cursor_b,
        "identical alloc requests against identically-sized memories must \
         land the bump cursor at the same offset in each store"
    );
    assert_eq!(
        store_a.data().jit_arena().live_count(),
        1,
        "store A must have exactly one live slot after one alloc"
    );
    assert_eq!(
        store_b.data().jit_arena().live_count(),
        1,
        "store B must have exactly one live slot after one alloc"
    );
}

/// Alloc'ing twice on store A and once on store B must not affect B's
/// cursor. After the alloc pair on A, A's cursor has dropped by 2 *
/// aligned-size; B's cursor (allocated once with the same size) should
/// match what A's cursor was after its FIRST alloc — proving B's arena
/// did not absorb A's second drop.
#[test]
fn allocs_on_one_store_do_not_shift_other_store_cursor() {
    let engine = make_engine();
    let cache = Arc::new(KernelCache::new());

    let mut linker: Linker<InstanceState> = Linker::new(&engine);
    add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");

    let wasm = wat::parse_str(DRIVER_WAT).expect("wat");
    let module = Module::new(&engine, &wasm).expect("module");

    let mut store_a = Store::new(&engine, instance_state(1, 100));
    let inst_a = linker
        .instantiate(&mut store_a, &module)
        .expect("instantiate A");
    let alloc_a = inst_a
        .get_typed_func::<i32, i32>(&mut store_a, "drive_alloc")
        .expect("typed A");

    let mut store_b = Store::new(&engine, instance_state(2, 200));
    let inst_b = linker
        .instantiate(&mut store_b, &module)
        .expect("instantiate B");
    let alloc_b = inst_b
        .get_typed_func::<i32, i32>(&mut store_b, "drive_alloc")
        .expect("typed B");

    // Two allocs on A — drops the cursor twice.
    let a1 = alloc_a.call(&mut store_a, 64).expect("alloc A1");
    let a2 = alloc_a.call(&mut store_a, 64).expect("alloc A2");
    assert!(a1 > 0 && a2 > 0, "A allocs succeed");
    assert!(
        a2 < a1,
        "second alloc on A must land below first ({a2} < {a1})"
    );

    // One alloc on B — must match A's *first* pointer, not A's second.
    let b1 = alloc_b.call(&mut store_b, 64).expect("alloc B1");
    assert_eq!(
        b1, a1,
        "B's first alloc must land at the same offset A's first alloc did, \
         proving A's second alloc did not leak the cursor into B's arena"
    );

    assert_eq!(
        store_a.data().jit_arena().live_count(),
        2,
        "store A must record both allocations"
    );
    assert_eq!(
        store_b.data().jit_arena().live_count(),
        1,
        "store B must record only its own single allocation"
    );
}
