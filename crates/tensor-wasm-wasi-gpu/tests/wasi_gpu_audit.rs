// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Audit-wave coverage: tests added in the S17-S22 fix wave that exercise
//! the boundaries called out in the security review.
//!
//! These tests live alongside `wasi_gpu_smoke.rs` rather than replacing it
//! so the original smoke regression continues to run.

use std::sync::Arc;
use std::thread;

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::abi::{
    AbiError, FN_LAST_ERROR_COPY, FN_LAST_ERROR_LEN, FN_LAUNCH, FN_LOAD_PTX, FN_SYNC,
    MAX_KERNELS_PER_INSTANCE, MAX_PTX_BYTES, MODULE,
};
use tensor_wasm_wasi_gpu::host::{add_to_linker, HasWasiCuda, WasiCudaContext};
use tensor_wasm_wasi_gpu::registry::{KernelEntry, KernelRegistry};

struct TestStore {
    cuda: WasiCudaContext,
}

impl HasWasiCuda for TestStore {
    fn wasi_cuda(&self) -> &WasiCudaContext {
        &self.cuda
    }
}

fn make_entry(owner: InstanceId, entry: &str) -> KernelEntry {
    KernelEntry {
        owner,
        entry: entry.into(),
        ptx_bytes_len: 1024,
        #[cfg(feature = "cuda")]
        module: None,
    }
}

/// `MAX_KERNELS_PER_INSTANCE` is a hard cap; registering the (cap+1)-th
/// entry must return `QuotaExceeded`.
#[test]
fn register_at_kernel_cap_returns_quota_exceeded() {
    let reg = KernelRegistry::new();
    for i in 0..MAX_KERNELS_PER_INSTANCE {
        reg.register(make_entry(InstanceId(1), &format!("k{i}")))
            .expect("under per-instance cap");
    }
    assert_eq!(reg.len(), MAX_KERNELS_PER_INSTANCE);
    let err = reg
        .register(make_entry(InstanceId(1), "k_over"))
        .expect_err("cap+1 must be rejected");
    assert_eq!(err, AbiError::QuotaExceeded);
}

/// `KernelRegistry` survives concurrent register + remove with no
/// use-after-free or panic. This is the regression guard for the original
/// UAF in `host.rs`'s launch path.
#[test]
fn concurrent_register_remove_stress() {
    let reg = Arc::new(KernelRegistry::new());
    let workers = 8;
    let per_thread = 200;
    let mut handles = Vec::new();
    for tid in 0..workers {
        let reg = reg.clone();
        handles.push(thread::spawn(move || {
            for i in 0..per_thread {
                if let Ok(id) = reg.register(KernelEntry {
                    owner: InstanceId(tid as u128),
                    entry: format!("t{tid}_k{i}"),
                    ptx_bytes_len: 16,
                    #[cfg(feature = "cuda")]
                    module: None,
                }) {
                    // Look up our own handle, then remove it. The handle
                    // must remain readable after remove (the UAF
                    // guard).
                    let handle = reg
                        .lookup(id, InstanceId(tid as u128))
                        .expect("lookup own kernel");
                    let _ = reg.remove(id);
                    // Touching the handle's cloned fields here would
                    // have segfaulted under the old raw-pointer design.
                    assert!(handle.entry.starts_with("t"));
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panicked");
    }
    // No leak: removes ran at least as many times as registers were
    // accepted (registers may have been refused by the per-instance
    // cap), so the live len should equal the number of accepted
    // registrations that the workers didn't immediately remove —
    // which is zero in this test.
    assert_eq!(reg.len(), 0);
}

const AUDIT_WAT: &str = r#"
(module
  (import "wasi:cuda/host@0.2.0" "wasi_cuda_load_ptx"
      (func $load_ptx (param i32 i32 i32 i32) (result i64)))
  (import "wasi:cuda/host@0.2.0" "wasi_cuda_launch"
      (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi:cuda/host@0.2.0" "wasi_cuda_sync"
      (func $sync (result i32)))
  (import "wasi:cuda/host@0.2.0" "wasi_cuda_last_error_len"
      (func $last_err_len (result i32)))
  (import "wasi:cuda/host@0.2.0" "wasi_cuda_last_error_copy"
      (func $last_err_copy (param i32 i32) (result i32)))

  (memory (export "memory") 1 4)
  ;; 27 bytes of "not valid ptx" + 10-byte entry name + non-utf8 entry.
  (data (i32.const 0)  "this is not valid ptx bytes")
  (data (i32.const 64) "vector_add")
  ;; A non-UTF8 byte sequence (0xff, 0xfe) for the entry-name test.
  (data (i32.const 96) "\ff\fe\ff")

  ;; Trigger a load_ptx failure so last_error is populated, then attempt
  ;; to copy the error into a destination region far past memory end.
  ;; Expect last_error_copy to return -2 (InvalidPointer), not 0.
  (func (export "load_then_copy_err_past_end") (result i32)
    (call $load_ptx (i32.const 0) (i32.const 27)
                    (i32.const 64) (i32.const 10))
    drop
    ;; 1 page = 65536; dst_ptr way past memory end:
    (call $last_err_copy (i32.const 16777216) (i32.const 64)))

  ;; load_ptx with a non-UTF8 entry name (3 bytes at offset 96).
  (func (export "load_with_non_utf8_entry") (result i64)
    ;; ptx points to the invalid PTX text just to keep ptx_len > 0; even
    ;; if PTX validation runs first the test still demonstrates that a
    ;; bogus entry name doesn't crash the host. We expect InvalidArgs
    ;; (-9) because the entry-name UTF-8 check runs before the PTX
    ;; validity check.
    (call $load_ptx (i32.const 0) (i32.const 27)
                    (i32.const 96) (i32.const 3)))

  ;; Launch with grid_x = 1, block_x = 2048 (> 1024 threads/block):
  ;; expect InvalidDimensions (-8).
  (func (export "launch_block_too_large") (result i32)
    (call $launch (i64.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 2048) (i32.const 1) (i32.const 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))

  ;; Launch with threads-per-block product = 32 * 32 * 2 = 2048: expect
  ;; InvalidDimensions even though each axis is in range.
  (func (export "launch_block_product_too_large") (result i32)
    (call $launch (i64.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 32) (i32.const 32) (i32.const 2)
                  (i32.const 0) (i32.const 0) (i32.const 0)))

  ;; Launch with shared_mem = -1: InvalidDimensions.
  (func (export "launch_negative_shared_mem") (result i32)
    (call $launch (i64.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const -1) (i32.const 0) (i32.const 0)))
)
"#;

fn make_engine_and_linker() -> (wasmtime::Engine, wasmtime::Linker<TestStore>) {
    let mut config = wasmtime::Config::new();
    config.async_support(true);
    let engine = wasmtime::Engine::new(&config).expect("engine");
    let mut linker: wasmtime::Linker<TestStore> = wasmtime::Linker::new(&engine);
    add_to_linker(&mut linker).expect("add_to_linker");
    (engine, linker)
}

#[tokio::test]
async fn last_error_copy_returns_invalid_pointer_past_memory_end() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(AUDIT_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(31)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let f = instance
        .get_typed_func::<(), i32>(&mut store, "load_then_copy_err_past_end")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidPointer.code(),
        "last_error_copy with out-of-bounds dst must return InvalidPointer (-2), not 0"
    );
}

#[tokio::test]
async fn load_ptx_with_non_utf8_entry_returns_invalid_args() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(AUDIT_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(32)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let f = instance
        .get_typed_func::<(), i64>(&mut store, "load_with_non_utf8_entry")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidArgs.code() as i64,
        "non-UTF8 entry name must return InvalidArgs"
    );
}

#[tokio::test]
async fn launch_with_block_dim_over_cap_returns_invalid_dimensions() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(AUDIT_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(33)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_block_too_large")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidDimensions.code(),
        "block_x = 2048 must return InvalidDimensions"
    );
}

#[tokio::test]
async fn launch_with_block_product_over_cap_returns_invalid_dimensions() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(AUDIT_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(34)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_block_product_too_large")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidDimensions.code(),
        "32*32*2 = 2048 threads-per-block must return InvalidDimensions"
    );
}

#[tokio::test]
async fn launch_with_negative_shared_mem_returns_invalid_dimensions() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(AUDIT_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(35)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_negative_shared_mem")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidDimensions.code(),
        "shared_mem = -1 must return InvalidDimensions"
    );
}

/// `MAX_PTX_BYTES` constant guard — fail loudly if the boundary shifts.
#[test]
fn max_ptx_bytes_constant_pinned() {
    assert_eq!(MAX_PTX_BYTES, 8 * 1024 * 1024);
}

/// `MAX_PTX_BYTES` boundary: a `load_ptx` call with `ptx_len` exactly one
/// byte past the cap must return `QuotaExceeded`. The check runs before
/// any memory read, so the `ptx_ptr` can be anything addressable.
#[tokio::test]
async fn load_ptx_over_max_ptx_bytes_returns_quota_exceeded() {
    let (engine, linker) = make_engine_and_linker();
    // ptx_len = MAX_PTX_BYTES + 1 = 8 MiB + 1.
    let too_big = (MAX_PTX_BYTES + 1) as i32;
    let wat = format!(
        r#"
        (module
          (import "wasi:cuda/host@0.2.0" "wasi_cuda_load_ptx"
              (func $load_ptx (param i32 i32 i32 i32) (result i64)))
          (memory (export "memory") 1 4)
          (data (i32.const 64) "k")
          (func (export "load_over_cap") (result i64)
            (call $load_ptx (i32.const 0) (i32.const {too_big})
                            (i32.const 64) (i32.const 1))))
        "#,
    );
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(37)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i64>(&mut store, "load_over_cap")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::QuotaExceeded.code() as i64,
        "ptx_len > MAX_PTX_BYTES must return QuotaExceeded"
    );
}

/// Property: `read_bytes`-style bounds check rejects a region that
/// straddles or extends past memory end. We exercise the same code path
/// via the launch host function: pass `args_ptr = i32::MAX, args_len = 4`
/// and observe `InvalidPointer`. On 64-bit hosts this triggers the
/// "end > mem_len" branch; on a 32-bit host the same input would trigger
/// the `checked_add` overflow branch — either way the host must refuse to
/// dereference.
#[tokio::test]
async fn launch_args_region_past_end_returns_invalid_pointer() {
    let (engine, linker) = make_engine_and_linker();
    let wat = r#"
        (module
          (import "wasi:cuda/host@0.2.0" "wasi_cuda_launch"
              (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1 4)
          (func (export "overflow_launch") (result i32)
            (call $launch (i64.const 1)
                          (i32.const 1) (i32.const 1) (i32.const 1)
                          (i32.const 1) (i32.const 1) (i32.const 1)
                          (i32.const 0)
                          ;; args_ptr = i32::MAX = 2_147_483_647
                          (i32.const 2147483647)
                          ;; args_len = 4: ptr + len exceeds memory length
                          (i32.const 4)))
        )
    "#;
    let wasm = wat::parse_str(wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: WasiCudaContext::new(InstanceId(36)),
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "overflow_launch")
        .expect("function");
    let code = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidPointer.code(),
        "args region past memory end must return InvalidPointer"
    );
}

// Keep the imports used by the dead-name-symbol check (some are only
// referenced in commented-out diagnostics but the constants exist).
#[allow(dead_code)]
const _SYMBOL_REFS: (&str, &str, &str, &str, &str, &str) = (
    MODULE,
    FN_LOAD_PTX,
    FN_LAUNCH,
    FN_SYNC,
    FN_LAST_ERROR_LEN,
    FN_LAST_ERROR_COPY,
);
