//! S6 integration test: per-instance isolation of `BaliLinearMemory`.
//!
//! Two properties are exercised:
//!
//! 1. Out-of-bounds writes from inside Wasm are caught by Wasmtime's bounds
//!    check and surface as a trap — they cannot escape into adjacent host
//!    allocations.
//! 2. Two independent `Store`s sharing the same `Engine` (and therefore the
//!    same `BaliMemoryCreator`) get fully independent backing buffers, so a
//!    write into instance A is invisible to instance B and vice versa.

use std::sync::Arc;

use bali_mem::wasm_memory::BaliMemoryCreator;
use wasmtime::{Config, Engine, Instance, Module, Store};

/// Single module shared by both tests: it exports `mem` (1 page, fixed), a
/// bounded `write(off, val)` helper, and an `oob_write(off)` helper that
/// hard-codes the sentinel pattern `0x41414141` so we can recognise it if it
/// ever leaks anywhere it should not.
const WAT: &str = r#"
(module
  (memory (export "mem") 1 1)
  (func (export "write") (param $off i32) (param $val i32)
    (i32.store offset=0 (local.get $off) (local.get $val)))
  (func (export "oob_write") (param $off i32)
    (i32.store offset=0 (local.get $off) (i32.const 0x41414141)))
)
"#;

/// Build an `Engine` wired to `BaliMemoryCreator`. The Config knobs match
/// `memory_roundtrip.rs`: no guard pages, no static/dynamic guard sizes, so
/// Wasmtime relies entirely on explicit bounds checks against
/// `BaliLinearMemory::byte_size`.
fn make_engine() -> Engine {
    let mut config = Config::new();
    let creator = Arc::new(BaliMemoryCreator::default());
    config.with_host_memory(creator);
    config.guard_before_linear_memory(false);
    config.static_memory_maximum_size(0);
    config.dynamic_memory_guard_size(0);
    Engine::new(&config).expect("engine")
}

#[test]
fn wasm_oob_write_traps_and_does_not_escape() {
    let engine = make_engine();
    let wasm = wat::parse_str(WAT).expect("wat -> wasm");
    let module = Module::new(&engine, &wasm).expect("module compile");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let memory = instance
        .get_memory(&mut store, "mem")
        .expect("memory export");

    // 1 page = 64 KiB; this offset is well past the end.
    const OOB_OFFSET: i32 = 1024 * 1024;

    // Sanity: the requested offset is genuinely past the visible memory.
    let visible = memory.data_size(&store);
    assert!(
        (OOB_OFFSET as usize) >= visible,
        "test setup wrong: OOB offset {OOB_OFFSET} must exceed visible size {visible}"
    );

    let oob = instance
        .get_typed_func::<i32, ()>(&mut store, "oob_write")
        .expect("function");

    let err = oob
        .call(&mut store, OOB_OFFSET)
        .expect_err("OOB write must trap");

    // The error should either present as a Wasmtime trap or contain the
    // familiar "out of bounds" phrasing somewhere in its chain. We check
    // both: downcast to `wasmtime::Trap` first, fall back to a string scan
    // through the source chain.
    let chain = format!("{err:#}").to_lowercase();
    let is_trap = err.downcast_ref::<wasmtime::Trap>().is_some();
    let mentions_oob = chain.contains("out of bounds")
        || chain.contains("out-of-bounds")
        || chain.contains("memory access");
    assert!(
        is_trap || mentions_oob,
        "expected Wasmtime trap or 'out of bounds' in error, got: {err:#}"
    );

    // The trap is observable; the host buffer must still be intact at the
    // sentinel offset 0 (no smashing happened before the bounds check).
    let view = memory.data(&store);
    assert_eq!(
        &view[0..4],
        &[0u8; 4],
        "OOB write must not have touched in-bounds memory"
    );
}

#[test]
fn cross_instance_memory_is_independent() {
    let engine = make_engine();
    let wasm = wat::parse_str(WAT).expect("wat -> wasm");
    let module = Module::new(&engine, &wasm).expect("module compile");

    // Two independent stores -> two independent `BaliLinearMemory`
    // allocations from the shared `BaliMemoryCreator`.
    let mut store_a = Store::new(&engine, ());
    let instance_a = Instance::new(&mut store_a, &module, &[]).expect("instantiate A");
    let mem_a = instance_a
        .get_memory(&mut store_a, "mem")
        .expect("memory A");
    let write_a = instance_a
        .get_typed_func::<(i32, i32), ()>(&mut store_a, "write")
        .expect("write A");

    let mut store_b = Store::new(&engine, ());
    let instance_b = Instance::new(&mut store_b, &module, &[]).expect("instantiate B");
    let mem_b = instance_b
        .get_memory(&mut store_b, "mem")
        .expect("memory B");
    let write_b = instance_b
        .get_typed_func::<(i32, i32), ()>(&mut store_b, "write")
        .expect("write B");

    // Each instance stamps its own distinctive pattern at offset 0.
    let pattern_a: u32 = 0xAAAA_AAAA;
    let pattern_b: u32 = 0xBBBB_BBBB;
    write_a
        .call(&mut store_a, (0, pattern_a as i32))
        .expect("write A");
    write_b
        .call(&mut store_b, (0, pattern_b as i32))
        .expect("write B");

    let view_a = &mem_a.data(&store_a)[0..4];
    let view_b = &mem_b.data(&store_b)[0..4];

    assert_eq!(
        view_a,
        &pattern_a.to_le_bytes(),
        "instance A must see its own pattern"
    );
    assert_eq!(
        view_b,
        &pattern_b.to_le_bytes(),
        "instance B must see its own pattern"
    );
    assert_ne!(
        view_a, view_b,
        "instances must not share backing memory at offset 0"
    );
}
