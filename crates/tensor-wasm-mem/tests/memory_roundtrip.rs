// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! S5 integration test: bytes written from inside Wasm appear in the host
//! buffer that backs [`TensorWasmLinearMemory`] — no copy, same allocation.

use std::sync::Arc;

use tensor_wasm_mem::wasm_memory::TensorWasmMemoryCreator;
use wasmtime::{Config, Engine, Instance, Module, Store};

const WAT: &str = r#"
(module
  (memory (export "mem") 1 4)
  (func (export "write_pattern") (param $off i32)
    (i32.store offset=0 (local.get $off) (i32.const 0xDEADBEEF))
    (i32.store offset=4 (local.get $off) (i32.const 0x12345678))
    (i32.store offset=8 (local.get $off) (i32.const 0xCAFEBABE))
  )
)
"#;

#[test]
fn wasm_writes_visible_in_host_buffer() {
    let mut config = Config::new();
    let creator = Arc::new(TensorWasmMemoryCreator::default());
    config.with_host_memory(creator);
    // Static memory size is at most one full Wasm page; growth happens via
    // TensorWasmLinearMemory::grow_to, not via guard-page magic.
    config.guard_before_linear_memory(false);
    config.memory_reservation(0);
    config.memory_guard_size(0);

    let engine = Engine::new(&config).expect("engine");
    let wasm = wat::parse_str(WAT).expect("wat → wasm");
    let module = Module::new(&engine, &wasm).expect("module compile");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let memory = instance
        .get_memory(&mut store, "mem")
        .expect("memory export");
    let write = instance
        .get_typed_func::<i32, ()>(&mut store, "write_pattern")
        .expect("function");

    write.call(&mut store, 256).expect("call write_pattern");

    // Read through Wasmtime's accessor — same backing buffer.
    let view = memory.data(&store);
    assert_eq!(&view[256..260], &0xDEADBEEFu32.to_le_bytes());
    assert_eq!(&view[260..264], &0x12345678u32.to_le_bytes());
    assert_eq!(&view[264..268], &0xCAFEBABEu32.to_le_bytes());
}

#[test]
fn memory_grow_through_wasm() {
    const GROW_WAT: &str = r#"
    (module
      (memory (export "mem") 1 4)
      (func (export "grow_one_page") (result i32)
        (memory.grow (i32.const 1))
      )
    )
    "#;

    let mut config = Config::new();
    let creator = Arc::new(TensorWasmMemoryCreator::default());
    config.with_host_memory(creator);
    config.guard_before_linear_memory(false);
    config.memory_reservation(0);
    config.memory_guard_size(0);

    let engine = Engine::new(&config).expect("engine");
    let wasm = wat::parse_str(GROW_WAT).expect("wat → wasm");
    let module = Module::new(&engine, &wasm).expect("module compile");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let grow = instance
        .get_typed_func::<(), i32>(&mut store, "grow_one_page")
        .expect("function");

    let prev_pages = grow.call(&mut store, ()).expect("grow call");
    // memory.grow returns the previous size in pages.
    assert!(prev_pages >= 1);

    let memory = instance
        .get_memory(&mut store, "mem")
        .expect("memory export");
    assert_eq!(memory.size(&store), prev_pages as u64 + 1);
}

#[test]
fn pre_grow_bytes_survive_grow() {
    const WAT: &str = r#"
    (module
      (memory (export "mem") 1 4)
      (func (export "write_pattern") (param $off i32)
        (i32.store offset=0 (local.get $off) (i32.const 0xDEADBEEF)))
      (func (export "grow_one_page") (result i32)
        (memory.grow (i32.const 1)))
    )
    "#;

    let mut config = Config::new();
    let creator = Arc::new(TensorWasmMemoryCreator::default());
    config.with_host_memory(creator);
    config.guard_before_linear_memory(false);
    config.memory_reservation(0);
    config.memory_guard_size(0);

    let engine = Engine::new(&config).expect("engine");
    let wasm = wat::parse_str(WAT).expect("wat → wasm");
    let module = Module::new(&engine, &wasm).expect("module compile");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let memory = instance
        .get_memory(&mut store, "mem")
        .expect("memory export");
    let write = instance
        .get_typed_func::<i32, ()>(&mut store, "write_pattern")
        .expect("function");
    let grow = instance
        .get_typed_func::<(), i32>(&mut store, "grow_one_page")
        .expect("function");

    // Write a known pattern at offset 1024.
    write.call(&mut store, 1024).expect("write before grow");
    let pre = memory.data(&store)[1024..1028].to_vec();
    assert_eq!(&pre[..], &0xDEADBEEFu32.to_le_bytes(), "pattern written");

    // Grow by one page.
    let prev_pages = grow.call(&mut store, ()).expect("grow");
    assert!(prev_pages >= 1);

    // Pre-grow bytes must still be there afterwards.
    let post = memory.data(&store)[1024..1028].to_vec();
    assert_eq!(
        &pre[..],
        &post[..],
        "memory.grow must not corrupt pre-existing bytes"
    );
}
