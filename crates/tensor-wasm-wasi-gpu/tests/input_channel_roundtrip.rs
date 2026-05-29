// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the `wasi:tensor/host` pull-model guest-input
//! channel (`input-len` / `read-input`).
//!
//! The pure-Rust unit tests for [`InputContext`] live next to the impl
//! in `src/streaming.rs`; this file exercises the end-to-end wiring
//! through a wasmtime `Linker` — an actual Wasm guest that calls
//! `input-len`, copies the staged bytes into its own linear memory via
//! `read-input`, and echoes them back out through `emit-chunk`. The
//! round-trip asserts the bytes the host staged are exactly the bytes the
//! guest emitted.
//!
//! See `wit/wasi-tensor.wit` for the WIT contract these tests pin.

use tensor_wasm_wasi_gpu::streaming::{
    add_input_to_linker, add_streaming_to_linker, HasInput, HasStreaming, InputContext,
    StreamingContext,
};
use tokio::sync::mpsc;

/// Per-store payload: an input context (source of the staged bytes) plus
/// a streaming context (sink the guest echoes them back into). The two
/// `add_*_to_linker` calls register both `wasi:tensor/host` host-fn
/// families against this payload.
struct TestStore {
    input: InputContext,
    streaming: StreamingContext,
}

impl HasInput for TestStore {
    fn input(&self) -> &InputContext {
        &self.input
    }
}

impl HasStreaming for TestStore {
    fn streaming(&self) -> &StreamingContext {
        &self.streaming
    }
}

fn make_engine_and_linker() -> (wasmtime::Engine, wasmtime::Linker<TestStore>) {
    let mut config = wasmtime::Config::new();
    config.async_support(true);
    let engine = wasmtime::Engine::new(&config).expect("engine");
    let mut linker: wasmtime::Linker<TestStore> = wasmtime::Linker::new(&engine);
    add_input_to_linker(&mut linker).expect("add_input_to_linker");
    add_streaming_to_linker(&mut linker).expect("add_streaming_to_linker");
    (engine, linker)
}

/// Guest that reads the entire staged input into linear memory at offset
/// 1024 and echoes it back via `emit-chunk`. It first calls `input-len`,
/// then `read-input(1024, len)`, then `emit-chunk(1024, written)`.
///
/// `read-input: func(ptr: u32, len: u32) -> s32` and
/// `emit-chunk: func(bytes: list<u8>) -> s32` both lower to a `(ptr, len)
/// -> i32` host ABI shape; `input-len: func() -> u32` lowers to `() ->
/// i32`.
const ECHO_INPUT_WAT: &str = r#"
(module
  (import "wasi:tensor/host" "input-len" (func $len (result i32)))
  (import "wasi:tensor/host" "read-input" (func $read (param i32 i32) (result i32)))
  (import "wasi:tensor/host" "emit-chunk" (func $emit (param i32 i32) (result i32)))
  (memory (export "memory") 1)

  ;; Read all staged input into [1024, 1024+n) and emit exactly the
  ;; number of bytes `read-input` reported writing. Returns the
  ;; `read-input` byte count so the host can assert on it directly.
  (func (export "echo_input") (result i32)
    (local $n i32)
    (local $written i32)
    (local.set $n (call $len))
    (local.set $written (call $read (i32.const 1024) (local.get $n)))
    (drop (call $emit (i32.const 1024) (local.get $written)))
    (local.get $written))
)
"#;

async fn run_echo(staged: &[u8]) -> (i32, Vec<Vec<u8>>) {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(ECHO_INPUT_WAT).expect("wat");
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");

    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            input: InputContext::new(staged.to_vec()),
            streaming: StreamingContext::with_channel(tx),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "echo_input")
        .expect("echo_input");
    let written = f.call_async(&mut store, ()).await.expect("call");

    // Drop the store (and the sender it owns) so the receiver closes and
    // we can drain every buffered chunk.
    drop(store);
    let mut chunks = Vec::new();
    while let Some(c) = rx.recv().await {
        chunks.push(c);
    }
    (written, chunks)
}

#[tokio::test]
async fn guest_reads_staged_input_and_echoes_it_back() {
    let staged = b"hello, guest input channel!";
    let (written, chunks) = run_echo(staged).await;

    assert_eq!(
        written as usize,
        staged.len(),
        "read-input must report writing every staged byte"
    );
    assert_eq!(chunks.len(), 1, "guest emits exactly one chunk");
    assert_eq!(
        chunks[0], staged,
        "round-trip: emitted bytes must equal the staged input verbatim"
    );
}

#[tokio::test]
async fn empty_input_reads_zero_bytes() {
    // No staged input → `input-len()` is 0, `read-input` writes nothing
    // and returns 0, and the guest emits an empty chunk.
    let (written, chunks) = run_echo(b"").await;
    assert_eq!(written, 0, "read-input on empty input returns 0");
    // emit-chunk(ptr, 0) still forwards an (empty) chunk.
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_empty());
}

#[tokio::test]
async fn read_input_clamps_to_available_bytes() {
    // The guest requests `input-len()` bytes (the natural sizing), so a
    // larger staged buffer round-trips fully without the guest needing to
    // know the size ahead of time.
    let staged = vec![0xABu8; 4096];
    let (written, chunks) = run_echo(&staged).await;
    assert_eq!(written as usize, staged.len());
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], staged);
}

/// Out-of-bounds destination must surface the `-2` InvalidPointer code
/// rather than writing past linear memory. Uses a dedicated guest that
/// asks `read-input` to write at a far-out-of-range pointer.
#[tokio::test]
async fn read_input_rejects_out_of_bounds_pointer() {
    const OOB_WAT: &str = r#"
    (module
      (import "wasi:tensor/host" "read-input" (func $read (param i32 i32) (result i32)))
      (memory (export "memory") 1)
      ;; One page = 65536 bytes. Asking to write 16 bytes at offset
      ;; 1_000_000 is well past the end of linear memory.
      (func (export "oob") (result i32)
        (call $read (i32.const 1000000) (i32.const 16)))
    )
    "#;
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(OOB_WAT).expect("wat");
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let (tx, _rx) = mpsc::channel::<Vec<u8>>(1);
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            input: InputContext::new(b"some staged bytes".to_vec()),
            streaming: StreamingContext::with_channel(tx),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "oob")
        .expect("oob");
    let rc = f.call_async(&mut store, ()).await.expect("call");
    // -2 == AbiError::InvalidPointer.
    assert_eq!(
        rc, -2,
        "out-of-bounds read-input must return the InvalidPointer (-2) code"
    );
}
