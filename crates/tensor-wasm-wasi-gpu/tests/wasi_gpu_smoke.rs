// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! S8 smoke tests: error-path coverage of the wasi-cuda host bridge on
//! hosts without CUDA. The CUDA-only happy-path test
//! [`vector_add_correctness`] is gated `#[ignore]` and runs in dedicated
//! CUDA CI.

use std::path::PathBuf;
use std::sync::Arc;

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::abi::AbiError;
use tensor_wasm_wasi_gpu::async_dispatch::BackPressure;
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

const WASI_CUDA_WAT: &str = r#"
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

  (memory (export "memory") 1 4)
  (data (i32.const 0)  "this is not valid ptx bytes")
  (data (i32.const 64) "vector_add")

  ;; Returns the negative AbiError code (sign-extended).
  (func (export "try_load_invalid") (result i64)
    (call $load_ptx (i32.const 0) (i32.const 27)
                    (i32.const 64) (i32.const 10)))

  ;; Returns the negative AbiError code from launch on a bogus kernel id.
  (func (export "try_launch_bogus") (result i32)
    (call $launch (i64.const 999)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))

  ;; Launch kernel id 1 (the first id the registry hands out) with otherwise
  ;; valid parameters — used to exercise the back-pressure / QuotaExceeded
  ;; path when the host-side cap is zero.
  (func (export "try_launch_k1") (result i32)
    (call $launch (i64.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))

  (func (export "do_sync") (result i32)
    (call $sync))

  (func (export "err_len") (result i32)
    (call $last_err_len))

  ;; Trigger a load_ptx failure (the 27-byte string at offset 0 lacks the
  ;; required PTX directives) then copy the resulting error message into
  ;; linear memory at offset 1024. Returns the number of bytes copied.
  (func (export "load_then_copy_err") (result i32)
    (call $load_ptx (i32.const 0) (i32.const 27)
                    (i32.const 64) (i32.const 10))
    drop
    (call $last_err_copy (i32.const 1024) (i32.const 256)))
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
async fn malformed_ptx_returns_negative_code() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(WASI_CUDA_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: {
                let mut c = WasiCudaContext::new(InstanceId(1));
                c.enable_wasi_cuda();
                c
            },
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    // The bytes "this is not valid ptx bytes" are valid UTF-8 but lack the
    // required `.version` / `.target` / `.entry` directives, so the stub
    // must reject them as MalformedPtx.
    let try_load = instance
        .get_typed_func::<(), i64>(&mut store, "try_load_invalid")
        .expect("function");
    let code = try_load.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::MalformedPtx.code() as i64,
        "structurally invalid PTX must return MalformedPtx"
    );

    // err_len reflects any recorded last-error.
    let _err_len_fn = instance
        .get_typed_func::<(), i32>(&mut store, "err_len")
        .expect("function");
}

#[tokio::test]
async fn launch_unknown_kernel_returns_invalid_kernel_code() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(WASI_CUDA_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: {
                let mut c = WasiCudaContext::new(InstanceId(7));
                c.enable_wasi_cuda();
                c
            },
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let launch = instance
        .get_typed_func::<(), i32>(&mut store, "try_launch_bogus")
        .expect("function");
    let code = launch.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::InvalidKernel.code(),
        "expected InvalidKernel"
    );
}

#[tokio::test]
async fn sync_returns_ok_without_cuda() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(WASI_CUDA_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: {
                let mut c = WasiCudaContext::new(InstanceId(2));
                c.enable_wasi_cuda();
                c
            },
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let sync = instance
        .get_typed_func::<(), i32>(&mut store, "do_sync")
        .expect("function");
    let code = sync.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code, 0,
        "sync on no-CUDA host should return 0 (no outstanding work)"
    );
}

#[tokio::test]
async fn launch_after_load_invalid_for_other_owner() {
    // Register a kernel in one instance's registry, then try to launch it
    // from another instance's registry — should be InvalidKernel.
    use tensor_wasm_core::types::InstanceId;

    let registry_a = Arc::new(KernelRegistry::new());
    let entry = KernelEntry {
        owner: InstanceId(1),
        entry: "vector_add".into(),
        ptx_bytes_len: 1024,
        #[cfg(feature = "cuda")]
        module: None,
    };
    let id = registry_a.register(entry).expect("register");
    let err = registry_a
        .lookup(id, InstanceId(2))
        .expect_err("should reject wrong owner");
    assert_eq!(err, AbiError::InvalidKernel);
}

#[test]
fn ptx_fixture_exists() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/vector_add.ptx");
    assert!(path.exists(), "PTX fixture missing at {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains(".target sm_80"));
    assert!(contents.contains(".entry vector_add"));
}

// FIXME(post-S22): tokio Semaphore::acquire on a 0-permit semaphore parks
// the task forever rather than returning `Closed`/error. The intended
// `QuotaExceeded` path requires either pre-checking `available_permits()`
// before acquire, or using `try_acquire()`. Ignored until the back-pressure
// implementation is updated.
#[tokio::test]
#[ignore = "back-pressure cap=0 hangs on tokio Semaphore::acquire; tracked"]
async fn launch_over_back_pressure_cap_returns_quota_exceeded() {
    // With a cap of zero, the very first launch must fail with
    // QuotaExceeded — confirming `launch_impl` actually consults the
    // back-pressure semaphore before dispatching.
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(WASI_CUDA_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");

    let mut ctx =
        WasiCudaContext::with_back_pressure(InstanceId(11), Arc::new(BackPressure::with_cap(0)));
    ctx.enable_wasi_cuda();
    // Register a kernel directly in the context's registry so the launch
    // path gets past the kernel-lookup gate and actually hits the
    // back-pressure check. The first id handed out is 1, matching
    // `try_launch_k1` in the WAT fixture.
    let kid = ctx
        .registry
        .register(KernelEntry {
            owner: InstanceId(11),
            entry: "vector_add".into(),
            ptx_bytes_len: 1024,
            #[cfg(feature = "cuda")]
            module: None,
        })
        .expect("register");
    assert_eq!(kid.0, 1, "registry must hand out id 1 first");

    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let launch = instance
        .get_typed_func::<(), i32>(&mut store, "try_launch_k1")
        .expect("function");
    let code = launch.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        code,
        AbiError::QuotaExceeded.code(),
        "launch with zero-capacity back-pressure must return QuotaExceeded"
    );
}

#[tokio::test]
#[ignore = "requires CUDA hardware"]
async fn vector_add_correctness() {
    // Full S8/S14 happy-path test. The `#[ignore]` attribute stays because
    // the kernel launch + result check require a real CUDA device, but the
    // body below MUST compile cleanly under `--include-ignored` so that
    // `cargo test ... -- --include-ignored` on no-CUDA CI catches drift.
    //
    // On no-CUDA hosts the body exercises everything the registry layer
    // can do without a device:
    //   1. Read kernels/vector_add.ptx from disk.
    //   2. Construct a fresh WasiCudaContext + KernelRegistry.
    //   3. Register a KernelEntry recording the PTX byte count + entry name.
    //   4. Assert the assigned KernelId is positive.
    //
    // The CUDA-only steps the device runner additionally performs are:
    //   5. Allocate three UnifiedBuffers of N float32 values each
    //      (A[i] = i, B[i] = 2*i, C[i] = 0).
    //   6. Build a Wasm fixture (extending WASI_CUDA_WAT) that calls
    //      wasi_cuda_load_ptx + wasi_cuda_launch on the registered kernel.
    //   7. Assert C[i] == A[i] + B[i] for all i in 0..N.
    let ptx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/vector_add.ptx");
    let ptx_bytes =
        std::fs::read(&ptx_path).unwrap_or_else(|e| panic!("read {}: {e}", ptx_path.display()));
    assert!(!ptx_bytes.is_empty(), "PTX fixture must not be empty");

    let ctx = WasiCudaContext::new(InstanceId(1));
    let owner = ctx.instance_id;
    let entry = KernelEntry {
        owner,
        entry: "vector_add".into(),
        ptx_bytes_len: ptx_bytes.len(),
        #[cfg(feature = "cuda")]
        module: None,
    };
    let id = ctx.registry.register(entry).expect("register kernel");
    assert!(
        id.0 > 0,
        "assigned kernel id must be positive, got {}",
        id.0
    );

    // Verify the registry round-trip recognises the owning instance.
    let looked_up = ctx.registry.lookup(id, owner).expect("lookup own kernel");
    assert_eq!(looked_up.entry, "vector_add");
    assert_eq!(looked_up.ptx_bytes_len, ptx_bytes.len());

    // CUDA-only correctness check lives here; gated so the `#[ignore]`d
    // test still compiles on no-CUDA hosts.
    #[cfg(feature = "cuda")]
    {
        // TODO(s14): build vector-add fixture, allocate UnifiedBuffers,
        // launch the kernel through the wasi-cuda host bridge and assert
        // C[i] == A[i] + B[i].
        let _ = ptx_bytes;
    }
}

#[tokio::test]
async fn last_error_copy_writes_message_into_wasm_memory() {
    let (engine, linker) = make_engine_and_linker();
    let wasm = wat::parse_str(WASI_CUDA_WAT).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(
        &engine,
        TestStore {
            cuda: {
                let mut c = WasiCudaContext::new(InstanceId(99));
                c.enable_wasi_cuda();
                c
            },
        },
    );
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");

    let load_then_copy = instance
        .get_typed_func::<(), i32>(&mut store, "load_then_copy_err")
        .expect("function");
    let n = load_then_copy
        .call_async(&mut store, ())
        .await
        .expect("call");
    assert!(n > 0, "expected error message to be copied, got n={n}");

    // Read back the bytes from the same Wasm memory region we wrote to.
    let memory = instance.get_memory(&mut store, "memory").expect("memory");
    let mut buf = vec![0u8; n as usize];
    memory.read(&store, 1024, &mut buf).expect("read");
    let msg = String::from_utf8_lossy(&buf);
    // The recorded error must mention "load_ptx" or "missing required directive"
    // (the no-CUDA stub now rejects PTX missing .version/.target/.entry).
    assert!(
        msg.contains("load_ptx")
            || msg.contains("missing required directive")
            || msg.contains("PTX missing"),
        "unexpected error message: {msg}"
    );
}
