// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for the wasi-cuda ABI: drives a wasmtime instance importing
//! the `wasi:cuda/host@0.1.0` functions and feeds the host fuzzer-derived
//! `(ptr, len)` pairs for `load_ptx`, `launch`, and `last_error_copy`.
//!
//! The goal is to crash the host on memory-safety bugs in the ABI
//! boundary (UAF in the registry, overflow in `read_bytes`, etc.). Pure
//! validation failures (`MalformedPtx`, `InvalidPointer`) are expected
//! and explicitly fine.
//!
//! NOTE for Batch M: this binary needs a `[[bin]]` entry in
//! `fuzz/Cargo.toml` and `tensor-wasm-wasi-gpu` added to `[dependencies]`. See
//! the suggested snippet at the end of this file.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::host::{add_to_linker, HasWasiCuda, WasiCudaContext};

#[derive(Debug, Arbitrary)]
enum FuzzOp {
    LoadPtx {
        ptx_ptr: i32,
        ptx_len: i32,
        entry_ptr: i32,
        entry_len: i32,
    },
    Launch {
        kernel_id: i64,
        grid_x: i32,
        grid_y: i32,
        grid_z: i32,
        block_x: i32,
        block_y: i32,
        block_z: i32,
        shared_mem: i32,
        args_ptr: i32,
        args_len: i32,
    },
    LastErrorCopy {
        dst_ptr: i32,
        dst_len: i32,
    },
    Sync,
}

struct Store {
    cuda: WasiCudaContext,
}

impl HasWasiCuda for Store {
    fn wasi_cuda(&self) -> &WasiCudaContext {
        &self.cuda
    }
}

const HARNESS_WAT: &str = r#"
(module
  (import "wasi:cuda/host@0.1.0" "wasi_cuda_load_ptx"
      (func $load_ptx (param i32 i32 i32 i32) (result i64)))
  (import "wasi:cuda/host@0.1.0" "wasi_cuda_launch"
      (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi:cuda/host@0.1.0" "wasi_cuda_sync"
      (func $sync (result i32)))
  (import "wasi:cuda/host@0.1.0" "wasi_cuda_last_error_len"
      (func $last_err_len (result i32)))
  (import "wasi:cuda/host@0.1.0" "wasi_cuda_last_error_copy"
      (func $last_err_copy (param i32 i32) (result i32)))

  ;; 1 page = 64 KiB so fuzzer-derived ptr/len can sweep around the boundary.
  (memory (export "memory") 1 4)

  (func (export "drive_load_ptx")
        (param $ptx_ptr i32) (param $ptx_len i32)
        (param $ent_ptr i32) (param $ent_len i32) (result i64)
    (call $load_ptx (local.get $ptx_ptr) (local.get $ptx_len)
                    (local.get $ent_ptr) (local.get $ent_len)))

  (func (export "drive_launch")
        (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
    (call $launch
        (local.get 0) (local.get 1) (local.get 2) (local.get 3)
        (local.get 4) (local.get 5) (local.get 6) (local.get 7)
        (local.get 8) (local.get 9)))

  (func (export "drive_last_error_copy") (param i32 i32) (result i32)
    (call $last_err_copy (local.get 0) (local.get 1)))

  (func (export "drive_sync") (result i32) (call $sync))
)
"#;

fuzz_target!(|data: &[u8]| {
    // Cap input size so each fuzz iteration stays fast.
    if data.len() > 4096 {
        return;
    }
    let mut u = Unstructured::new(data);
    let Ok(ops): Result<Vec<FuzzOp>, _> = u.arbitrary() else {
        return;
    };
    if ops.len() > 64 {
        return;
    }

    // Single-threaded async runtime: each fuzz iteration is a fresh
    // wasmtime store + context. We rebuild from scratch every iter so
    // any UAF in registry or launch surfaces immediately when libfuzzer
    // exercises the same input shape twice.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => return,
    };

    rt.block_on(async move {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = match wasmtime::Engine::new(&config) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut linker: wasmtime::Linker<Store> = wasmtime::Linker::new(&engine);
        if add_to_linker(&mut linker).is_err() {
            return;
        }
        let bytes = match wat::parse_str(HARNESS_WAT) {
            Ok(b) => b,
            Err(_) => return,
        };
        let module = match wasmtime::Module::new(&engine, &bytes) {
            Ok(m) => m,
            Err(_) => return,
        };
        let mut store = wasmtime::Store::new(
            &engine,
            Store {
                cuda: WasiCudaContext::new(InstanceId(0xf_u128)),
            },
        );
        let instance = match linker.instantiate_async(&mut store, &module).await {
            Ok(i) => i,
            Err(_) => return,
        };

        for op in ops {
            match op {
                FuzzOp::LoadPtx {
                    ptx_ptr,
                    ptx_len,
                    entry_ptr,
                    entry_len,
                } => {
                    if let Ok(f) = instance.get_typed_func::<(i32, i32, i32, i32), i64>(
                        &mut store,
                        "drive_load_ptx",
                    ) {
                        let _ = f
                            .call_async(&mut store, (ptx_ptr, ptx_len, entry_ptr, entry_len))
                            .await;
                    }
                }
                FuzzOp::Launch {
                    kernel_id,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    shared_mem,
                    args_ptr,
                    args_len,
                } => {
                    if let Ok(f) = instance.get_typed_func::<(
                        i64,
                        i32,
                        i32,
                        i32,
                        i32,
                        i32,
                        i32,
                        i32,
                        i32,
                        i32,
                    ), i32>(
                        &mut store, "drive_launch"
                    ) {
                        let _ = f
                            .call_async(
                                &mut store,
                                (
                                    kernel_id, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                                    shared_mem, args_ptr, args_len,
                                ),
                            )
                            .await;
                    }
                }
                FuzzOp::LastErrorCopy { dst_ptr, dst_len } => {
                    if let Ok(f) = instance.get_typed_func::<(i32, i32), i32>(
                        &mut store,
                        "drive_last_error_copy",
                    ) {
                        let _ = f.call_async(&mut store, (dst_ptr, dst_len)).await;
                    }
                }
                FuzzOp::Sync => {
                    if let Ok(f) =
                        instance.get_typed_func::<(), i32>(&mut store, "drive_sync")
                    {
                        let _ = f.call_async(&mut store, ()).await;
                    }
                }
            }
        }
    });
});

// =========================================================================
// NOTE for Batch M (`fuzz/Cargo.toml` owner):
// Add the following to `fuzz/Cargo.toml` to register this target:
//
//   [dependencies]
//   tensor-wasm-wasi-gpu = { path = "../crates/tensor-wasm-wasi-gpu" }
//   tokio = { version = "1.40", features = ["rt", "macros"] }
//   wat = "1"
//
//   [[bin]]
//   name = "fuzz_wasi_cuda_abi"
//   path = "fuzz_targets/fuzz_wasi_cuda_abi.rs"
//   test = false
//   doc = false
//   bench = false
// =========================================================================
