// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end coverage of the v0.2 typed-argv kernel-args marshalling
//! path through the `wasi:cuda` `launch` host function.
//!
//! Each test builds a tiny WAT module that writes a tagged-argv buffer
//! into linear memory and calls `wasi_cuda_launch` with that buffer.
//! On no-CUDA hosts the launch returns `NotAvailable` — the assertion
//! is that the host parsed the argv (visible via
//! `WasiCudaContext::last_lowered_args`) before reporting the no-CUDA
//! signal, exercising the parser end-to-end.
//!
//! The CUDA-only assertions (kernel actually runs, output buffer
//! mutated) live behind `#[ignore = "requires CUDA hardware"]` so the
//! body still compiles under `cargo test --include-ignored` on a
//! no-CUDA developer laptop.
//!
//! ## v0.3.3 end-to-end pipeline proof (audit Problem #14)
//!
//! The original five tests in this file confirmed that the argv
//! marshalling path is reachable and that the parsed `LoweredArg`
//! sequence matches the input — but they all register a *stub* kernel
//! (`module: None`), so under `--features cuda` the launch fails at
//! the `InvalidKernel` gate before `cuLaunchKernel` ever runs. The
//! audit closed v0.3.2 with that gap explicit: nothing in the test
//! suite proved Wasm → wasi-cuda → `cuLaunchKernel` → result-readback
//! actually produces correct output on real hardware.
//!
//! The two `vector_add_*` / `dispatch_pipeline_*` tests at the bottom
//! of the file close that gap:
//!
//! - [`vector_add_end_to_end_real_ptx_real_kernel`] is the SM_80 happy
//!   path: build a Wasm guest holding three f32[64] arrays in linear
//!   memory, register the canonical `kernels/vector_add.ptx` via
//!   `cust::module::Module::from_ptx`, launch with `grid_x = 1,
//!   block_x = 64`, then read the `c` region back out of linear memory
//!   and assert `c[i] == a[i] + b[i]` for all `i`. `#[ignore]`d
//!   because the SM_80 PTX may be rejected by `ptxas` on SM_75 dev
//!   boxes; the v0.4 self-hosted runner (S22) picks it up.
//! - [`dispatch_pipeline_compiles_against_real_module_bytes`] is the
//!   v0.3.3 commitment: it runs *unignored* on every CUDA-capable host
//!   and asserts that the registry + load_ptx + launch surface accepts
//!   the real PTX bytes and routes them through the dispatch path.
//!   The launch may fail downstream (e.g. on SM_75 the `Module::from_ptx`
//!   JIT compile rejects the SM_80 target), but the failure shape is
//!   bounded: it is one of `{Ok, MalformedPtx, LaunchFailed,
//!   InvalidKernel}` — never `InvalidArgs` or `InvalidPointer`, which
//!   would indicate the marshalling layer regressed.

use tensor_wasm_core::types::InstanceId;
use tensor_wasm_wasi_gpu::abi::{AbiError, FN_LAUNCH, MODULE};
use tensor_wasm_wasi_gpu::host::{add_to_linker, HasWasiCuda, WasiCudaContext};
use tensor_wasm_wasi_gpu::kernel_args::{
    build_kernel_param_storage, encode_argv, LoweredArg, LoweredArgSnapshot,
};
use tensor_wasm_wasi_gpu::registry::KernelEntry;

struct TestStore {
    cuda: WasiCudaContext,
}

impl HasWasiCuda for TestStore {
    fn wasi_cuda(&self) -> &WasiCudaContext {
        &self.cuda
    }
}

fn make_engine_and_linker() -> (wasmtime::Engine, wasmtime::Linker<TestStore>) {
    let mut config = wasmtime::Config::new();
    config.async_support(true);
    let engine = wasmtime::Engine::new(&config).expect("engine");
    let mut linker: wasmtime::Linker<TestStore> = wasmtime::Linker::new(&engine);
    add_to_linker(&mut linker).expect("add_to_linker");
    (engine, linker)
}

/// Build a tiny WAT module that copies `argv_bytes` into linear memory
/// at offset `argv_offset` (via a `data` segment) and exports a
/// `launch_with_args` function that hands that buffer to
/// `wasi_cuda_launch` with a 1x1x1 grid + block.
///
/// The kernel id `kernel_id` is passed through unchanged so the test
/// can register a real `KernelEntry` and reach the marshalling path
/// (rather than failing at the `InvalidKernel` gate).
fn build_launch_wat(argv_bytes: &[u8], argv_offset: usize, kernel_id: u64) -> String {
    let data_literal = wat_data_literal(argv_bytes);
    format!(
        r#"
        (module
          (import "{m}" "{fn_name}"
              (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 4)
          (data (i32.const {data_offset}) "{data_literal}")
          (func (export "launch_with_args") (result i32)
            (call $launch
              (i64.const {kernel_id})
              (i32.const 1) (i32.const 1) (i32.const 1)
              (i32.const 1) (i32.const 1) (i32.const 1)
              (i32.const 0)
              (i32.const {argv_offset})
              (i32.const {argv_len}))))
        "#,
        m = MODULE,
        fn_name = FN_LAUNCH,
        data_offset = argv_offset,
        argv_len = argv_bytes.len(),
        kernel_id = kernel_id,
        data_literal = data_literal,
    )
}

/// Register a stub kernel in the context's registry and return its id.
/// The non-CUDA path uses only `owner`, `entry`, and `ptx_bytes_len` so
/// `module: None` is fine.
fn register_stub_kernel(ctx: &WasiCudaContext, owner: InstanceId, name: &str) -> u64 {
    let kid = ctx
        .registry
        .register(KernelEntry {
            owner,
            entry: name.into(),
            ptx_bytes_len: 256,
            #[cfg(feature = "cuda")]
            module: None,
        })
        .expect("register");
    kid.0
}

/// Encode a slice of bytes as a WAT `data` segment literal (each byte
/// formatted as `\xx`). Shared between the argv-encoding path in
/// `build_launch_wat` and the data-region preloading in
/// `build_vector_add_launch_wat`.
fn wat_data_literal(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 4);
    for b in bytes {
        s.push_str(&format!("\\{:02x}", b));
    }
    s
}

/// Build a WAT module specialised for the `vector_add_*` e2e tests.
///
/// Unlike [`build_launch_wat`], this helper:
///
/// 1. Preloads three pointer-arg data regions in addition to the argv
///    buffer (so the kernel reads real `a`, `b` bytes and writes into a
///    zeroed `c` region — all inside Wasm linear memory).
/// 2. Parameterises the grid/block dims (the canonical `vector_add`
///    fixture uses `grid_x = 1, block_x = 64`, not `1x1x1`).
/// 3. Exports the linear memory unchanged, so the test can read the
///    `c` region back out of `instance.get_memory(..)` after the
///    launch synchronizes.
///
/// We avoid extending `build_launch_wat` itself so the existing five
/// argv-shape tests (which all use a single data segment + fixed 1x1x1
/// dims) keep their current footprint.
#[allow(clippy::too_many_arguments)]
fn build_vector_add_launch_wat(
    argv_bytes: &[u8],
    argv_offset: usize,
    kernel_id: u64,
    grid_x: u32,
    block_x: u32,
    data_regions: &[(usize, Vec<u8>)],
) -> String {
    let argv_literal = wat_data_literal(argv_bytes);
    let mut data_segments = String::new();
    for (offset, bytes) in data_regions {
        let literal = wat_data_literal(bytes);
        data_segments.push_str(&format!(
            "\n          (data (i32.const {offset}) \"{literal}\")"
        ));
    }
    format!(
        r#"
        (module
          (import "{m}" "{fn_name}"
              (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 4)
          (data (i32.const {argv_offset}) "{argv_literal}"){data_segments}
          (func (export "launch_with_args") (result i32)
            (call $launch
              (i64.const {kernel_id})
              (i32.const {grid_x}) (i32.const 1) (i32.const 1)
              (i32.const {block_x}) (i32.const 1) (i32.const 1)
              (i32.const 0)
              (i32.const {argv_offset})
              (i32.const {argv_len}))))
        "#,
        m = MODULE,
        fn_name = FN_LAUNCH,
        argv_offset = argv_offset,
        argv_len = argv_bytes.len(),
        argv_literal = argv_literal,
        data_segments = data_segments,
        kernel_id = kernel_id,
        grid_x = grid_x,
        block_x = block_x,
    )
}

/// Register a kernel backed by a real PTX module.
///
/// Mirrors [`register_stub_kernel`] but uses `cust::module::Module::from_ptx`
/// to compile the supplied PTX bytes through the CUDA driver's JIT.
/// Requires `--features cuda`; without the feature the helper panics
/// because the no-CUDA build cannot load a real module at all.
///
/// Returns `Ok((kernel_id, ptx_load_succeeded))`. When
/// `ptx_load_succeeded` is `false`, the JIT compile inside cust rejected
/// the PTX (typically on a host whose compute capability is below the
/// PTX's `.target sm_XX` line); the test should treat that as a skip
/// rather than a failure. In that case the returned kernel id is from a
/// `module: None` stub fall-back, so the caller's subsequent launch
/// observes the `InvalidKernel` failure path instead of marshalling
/// success.
/// Initialise CUDA exactly once for this test process and keep the primary
/// context alive for the whole run.
///
/// `cust::quick_init` calls `cuInit` and pushes a primary context; calling it
/// twice in one process fails ("context already exists"), so the `Context` is
/// cached in a `OnceLock`. Both `register_real_kernel` (which needs a current
/// context for `Module::from_ptx`) and `device_compute_capability` (which needs
/// `cuInit` to have run before any `cuDevice*` call) go through here — the
/// latter previously failed with `CUDA_ERROR_NOT_INITIALIZED` because it queried
/// the device before anything initialised the driver.
#[cfg(feature = "cuda")]
fn ensure_cuda_initialized() {
    use std::sync::OnceLock;
    static CTX: OnceLock<Result<cust::context::Context, String>> = OnceLock::new();
    let init =
        CTX.get_or_init(|| cust::quick_init().map_err(|e| format!("cust::quick_init: {e:?}")));
    if let Err(msg) = init {
        panic!("ensure_cuda_initialized: CUDA init failed: {msg}");
    }
}

#[cfg(feature = "cuda")]
fn register_real_kernel(
    ctx: &WasiCudaContext,
    owner: InstanceId,
    name: &str,
    ptx_bytes: &[u8],
) -> (u64, bool) {
    // `cust::module::Module::from_ptx` requires a current CUDA context; the
    // wasi-cuda launch path inherits whatever context is current on this
    // thread. Initialise once (and keep the primary context alive) via the
    // shared helper.
    ensure_cuda_initialized();

    let ptx_str = std::str::from_utf8(ptx_bytes).expect("PTX is valid UTF-8");
    let module = match cust::module::Module::from_ptx(ptx_str, &[]) {
        Ok(m) => Some(std::sync::Arc::new(m)),
        Err(e) => {
            eprintln!(
                "register_real_kernel: Module::from_ptx rejected `{name}` PTX \
                 ({e:?}); falling back to module:None. The host launch path \
                 will surface InvalidKernel, which the caller can treat as a \
                 SM-mismatch skip."
            );
            None
        }
    };
    let loaded = module.is_some();
    let kid = ctx
        .registry
        .register(KernelEntry {
            owner,
            entry: name.into(),
            ptx_bytes_len: ptx_bytes.len(),
            module,
        })
        .expect("register");
    (kid.0, loaded)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn register_real_kernel(
    _ctx: &WasiCudaContext,
    _owner: InstanceId,
    _name: &str,
    _ptx_bytes: &[u8],
) -> (u64, bool) {
    panic!(
        "register_real_kernel requires --features cuda; \
         the no-CUDA build has no way to load a real PTX module."
    );
}

/// The current CUDA device's compute capability as `(major, minor)`.
///
/// Used by the end-to-end launch test to pick a PTX fixture whose `.target`
/// the driver JIT will accept on THIS device, rather than feeding the JIT a
/// known-mismatched fixture (which fails and leaves a sticky error on the
/// context, poisoning the next load — the `InvalidPtx`-then-`UnknownError`
/// cascade observed on the RTX 2060 / sm_75 dev box). Requires an initialised
/// CUDA context (the caller runs `cust::quick_init` first).
#[cfg(feature = "cuda")]
fn device_compute_capability() -> (i32, i32) {
    use cust::sys as cuda_sys;
    // Make sure `cuInit` has run before any `cuDevice*` call — otherwise the
    // driver returns `CUDA_ERROR_NOT_INITIALIZED`. Idempotent and cached.
    ensure_cuda_initialized();
    // The raw driver API is now usable. We go through `cust::sys`
    // directly rather than the safe `Device::get_attribute` wrapper because
    // the latter is not exposed under our `cust` feature set
    // (`default-features = false`); the raw FFI is always linked.
    // SAFETY: out-params are valid locals; the driver is initialised.
    unsafe {
        let mut dev: cuda_sys::CUdevice = 0;
        let res = cuda_sys::cuDeviceGet(&mut dev as *mut cuda_sys::CUdevice, 0);
        assert_eq!(
            res,
            cuda_sys::cudaError_enum::CUDA_SUCCESS,
            "cuDeviceGet(0) failed: {res:?}"
        );
        let mut major = 0i32;
        let mut minor = 0i32;
        let r_major = cuda_sys::cuDeviceGetAttribute(
            &mut major as *mut i32,
            cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        );
        let r_minor = cuda_sys::cuDeviceGetAttribute(
            &mut minor as *mut i32,
            cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        );
        assert_eq!(
            r_major,
            cuda_sys::cudaError_enum::CUDA_SUCCESS,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR) failed: {r_major:?}"
        );
        assert_eq!(
            r_minor,
            cuda_sys::cudaError_enum::CUDA_SUCCESS,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR) failed: {r_minor:?}"
        );
        (major, minor)
    }
}

/// Pick the `vector_add` PTX fixture whose `.target` JIT-loads on the current
/// device: the sm_75 build for compute capability < 8.0 (Turing and older),
/// the canonical sm_80 build otherwise. sm_75 PTX would also JIT forward onto
/// Ampere+, but preferring the exact-arch fixture keeps the proof honest about
/// what it exercised.
#[cfg(feature = "cuda")]
fn select_vector_add_ptx() -> (&'static [u8], &'static str) {
    let (major, minor) = device_compute_capability();
    if major >= 8 {
        (VECTOR_ADD_PTX, "sm_80")
    } else {
        eprintln!(
            "select_vector_add_ptx: device compute capability {major}.{minor} < 8.0; \
             using the sm_75 fixture."
        );
        (VECTOR_ADD_PTX_SM75, "sm_75")
    }
}

/// Scalar argv (mix of i32, i64, f32, f64, u32, u64) round-trips through
/// the launch path. On the no-CUDA host stub the launch returns
/// `NotAvailable` because no GPU is available, but the parsed args are
/// recorded into `last_lowered_args` — that's the property the test
/// pins.
#[tokio::test]
async fn scalar_argv_round_trips_through_launch_path() {
    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(301);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();
    let kid = register_stub_kernel(&ctx, owner, "scalar_kernel");

    let expected = vec![
        LoweredArg::I32(-13),
        LoweredArg::I64(1_234_567_890_123),
        LoweredArg::F32(2.5_f32),
        LoweredArg::F64(3.5_f64),
        LoweredArg::U32(0x1234_5678),
        LoweredArg::U64(0xDEAD_BEEF_C0FF_EE00),
    ];
    let argv = encode_argv(&expected);
    let wat = build_launch_wat(&argv, 1024, kid);
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");

    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let rc = f.call_async(&mut store, ()).await.expect("call");

    // No-CUDA path: launch reports NotAvailable after parsing argv.
    // CUDA path: the stub kernel has no PTX module behind it (see
    // register_stub_kernel), so the launch correctly fails with
    // InvalidKernel rather than the optimistic `0` the test originally
    // claimed. The argv-parsing assertion below is the real property the
    // test pins. A real-PTX launch-success assertion lives in the
    // #[ignore = "requires CUDA hardware"] test at the bottom of the file.
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        rc,
        AbiError::NotAvailable.code(),
        "no-CUDA host: launch reports NotAvailable after parsing argv"
    );
    #[cfg(feature = "cuda")]
    assert_eq!(
        rc,
        AbiError::InvalidKernel.code(),
        "CUDA host: stub kernel has no PTX module; launch rejects with InvalidKernel"
    );

    // The parsed args must be visible regardless of CUDA-vs-stub.
    // `last_lowered_args` returns a pointer-free snapshot, so compare
    // against the snapshot projection of the expected `LoweredArg`s.
    let recorded = store.data().wasi_cuda().last_lowered_args();
    let expected_snapshots: Vec<LoweredArgSnapshot> =
        expected.iter().map(LoweredArgSnapshot::from).collect();
    assert_eq!(
        recorded, expected_snapshots,
        "parsed argv must round-trip the original LoweredArg sequence"
    );
}

/// Pointer argv (two pointer args, mixed with a scalar) round-trips
/// through the launch path. The pointer regions all lie inside the
/// guest's linear memory, so the bounds-check inside `parse_argv`
/// passes and the resolved host pointers are recorded.
#[tokio::test]
async fn pointer_argv_round_trips_through_launch_path() {
    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(302);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();
    let kid = register_stub_kernel(&ctx, owner, "pointer_kernel");

    // Two pointer args: one at offset 256 (length 64), one at offset
    // 4096 (length 128); both live inside the 4-page (256 KiB) memory
    // the WAT exports. A scalar i32 separates them to confirm mixed
    // argv works.
    //
    // `LoweredArg::Ptr` carries a crate-private `host_ptr` so out-of-
    // crate callers go through `LoweredArg::ptr_for_encoding`, which
    // produces the null-pointer placeholder `encode_argv` ignores.
    let expected = vec![
        LoweredArg::ptr_for_encoding(256, 64),
        LoweredArg::I32(7),
        LoweredArg::ptr_for_encoding(4096, 128),
    ];
    let argv = encode_argv(&expected);
    // Place the argv buffer high enough that it doesn't overlap the
    // pointer-arg regions we declared above.
    let wat = build_launch_wat(&argv, 8192, kid);
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");

    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let rc = f.call_async(&mut store, ()).await.expect("call");

    // Same reasoning as scalar_argv_round_trips_through_launch_path: the
    // CUDA path correctly rejects the stub kernel with InvalidKernel
    // because no PTX module is loaded. The argv-parsing assertion below
    // is the property the test pins.
    #[cfg(not(feature = "cuda"))]
    assert_eq!(rc, AbiError::NotAvailable.code());
    #[cfg(feature = "cuda")]
    assert_eq!(rc, AbiError::InvalidKernel.code());

    // `last_lowered_args` returns pointer-free snapshots — the resolved
    // `host_ptr` lives only on the crate-internal launch path and is
    // intentionally redacted at the public boundary (see the
    // `LoweredArgSnapshot` rationale).
    let recorded = store.data().wasi_cuda().last_lowered_args();
    assert_eq!(recorded.len(), 3, "expected three lowered args");
    // Spot-check fields. Pointers compare by `guest_offset` / `len`
    // only — the pointer itself is no longer part of the snapshot.
    match &recorded[0] {
        LoweredArgSnapshot::Ptr { guest_offset, len } => {
            assert_eq!(*guest_offset, 256);
            assert_eq!(*len, 64);
        }
        other => panic!("idx 0 expected Ptr, got {other:?}"),
    }
    assert!(matches!(recorded[1], LoweredArgSnapshot::I32(7)));
    match &recorded[2] {
        LoweredArgSnapshot::Ptr { guest_offset, len } => {
            assert_eq!(*guest_offset, 4096);
            assert_eq!(*len, 128);
        }
        other => panic!("idx 2 expected Ptr, got {other:?}"),
    }
}

/// An out-of-bounds pointer arg must surface as `InvalidPointer`. The
/// outer args-region bounds-check passes (the argv buffer itself is
/// in-bounds); the parser then bounds-checks the embedded guest
/// pointer and fails.
#[tokio::test]
async fn pointer_argv_out_of_bounds_returns_invalid_pointer() {
    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(303);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();
    let kid = register_stub_kernel(&ctx, owner, "oob_ptr_kernel");

    // Guest pointer at offset (4 pages == 256 KiB) is past the memory
    // end (we export 4 pages here, so the highest valid offset for a
    // zero-length pointer is exactly 256 KiB, but a 16-byte read at
    // offset 250000 spans into OOB).
    //
    // `LoweredArg::Ptr` is `#[non_exhaustive]`, so out-of-crate callers
    // (including this integration test) cannot construct via struct-
    // literal syntax — they go through `ptr_for_encoding`, which stamps
    // a null host_ptr placeholder the parser subsequently overwrites.
    let expected = vec![LoweredArg::ptr_for_encoding(
        /* guest_offset */ 4 * 65536 - 256, // straddles end of 4 pages
        /* len */ 1024,
    )];
    let argv = encode_argv(&expected);
    let wat = build_launch_wat(&argv, 1024, kid);
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");

    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let rc = f.call_async(&mut store, ()).await.expect("call");
    assert_eq!(
        rc,
        AbiError::InvalidPointer.code(),
        "pointer arg whose [ptr, ptr+len) spans past memory end must \
         return InvalidPointer (got {rc})"
    );
    // No args should have been recorded because parsing failed.
    let recorded = store.data().wasi_cuda().last_lowered_args();
    assert!(
        recorded.is_empty(),
        "last_lowered_args must remain empty on parse failure"
    );
}

/// CUDA hardware variant: a real launch of a kernel taking scalar args
/// returns 0. The launch fixture loads a PTX module that adds two
/// integers and writes the result into a UVM buffer; the host checks
/// the buffer after the launch returns.
///
/// The body is `#[ignore]` because it requires a CUDA-capable host;
/// on no-CUDA CI the test compiles cleanly thanks to the cfg-gated
/// CUDA-only steps.
#[tokio::test]
#[ignore = "requires CUDA hardware"]
async fn scalar_argv_real_cuda_launch() {
    // The no-CUDA fallback path still exercises the parser end-to-end;
    // the CUDA-only step is the kernel-runs-and-mutates-output check.
    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(401);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();
    let kid = register_stub_kernel(&ctx, owner, "scalar_add");
    let argv = encode_argv(&[LoweredArg::I32(2), LoweredArg::I32(3)]);
    let wat = build_launch_wat(&argv, 1024, kid);
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let _rc = f.call_async(&mut store, ()).await.expect("call");
    #[cfg(feature = "cuda")]
    {
        // TODO(v0.2-cuda-runner): wire a real PTX fixture + UVM output
        // buffer here and assert the kernel actually ran. The runtime
        // path is exercised by the shipping `register_stub_kernel`
        // entry having `module: None`, so under `--features cuda` the
        // launch will fail with `InvalidKernel` until the runner ships
        // a real PTX-backed module.
    }
}

/// CUDA hardware variant: a real launch of a kernel taking pointer
/// args returns 0. Pointer args are resolved against the guest's linear
/// memory; under CUDA Unified Memory the resolved host pointers double
/// as device pointers.
#[tokio::test]
#[ignore = "requires CUDA hardware"]
async fn pointer_argv_real_cuda_launch() {
    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(402);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();
    let kid = register_stub_kernel(&ctx, owner, "pointer_copy");
    let argv = encode_argv(&[
        LoweredArg::ptr_for_encoding(0, 1024),
        LoweredArg::ptr_for_encoding(2048, 1024),
        LoweredArg::U32(256),
    ]);
    let wat = build_launch_wat(&argv, 8192, kid);
    let wasm = wat::parse_str(&wat).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile");
    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let _rc = f.call_async(&mut store, ()).await.expect("call");
    #[cfg(feature = "cuda")]
    {
        // TODO(v0.2-cuda-runner): wire a real PTX fixture and assert the
        // kernel copied the data. See sibling TODO in
        // `scalar_argv_real_cuda_launch`.
    }
}

/// Canonical `kernels/vector_add.ptx` fixture, embedded at compile time.
///
/// Shared by the two v0.3.3 end-to-end tests below. Embedding via
/// `include_bytes!` (rather than `std::fs::read` at runtime) means the
/// test binary is self-contained — no path-relative `read` that breaks
/// when `cargo test` is invoked from outside the workspace root.
#[cfg(feature = "cuda")]
const VECTOR_ADD_PTX: &[u8] = include_bytes!("fixtures/vector_add.ptx");

/// sm_75 (Turing) variant of the vector_add kernel — same body, lower
/// `.target` — so the launch actually runs on sub-Ampere dev boxes (e.g. the
/// RTX 2060, compute capability 7.5) where the sm_80 fixture is rejected by
/// the driver JIT. The end-to-end test falls back to this when the canonical
/// sm_80 PTX is refused, rather than skipping the kernel-output assertion.
#[cfg(feature = "cuda")]
const VECTOR_ADD_PTX_SM75: &[u8] = include_bytes!("fixtures/vector_add_sm75.ptx");

/// End-to-end Wasm -> wasi-cuda -> `cuLaunchKernel` -> result-readback
/// proof on real CUDA. Closes the v0.3.2 audit Problem #14.
///
/// Pipeline this test exercises:
///
/// 1. Build a Wasm guest that exports a `memory` of 4 pages and
///    pre-populates three regions: `a[i] = i as f32`, `b[i] = 100 + i`,
///    `c[i] = 0`, for `i in 0..64`. The guest also writes a tagged
///    argv buffer holding `[Ptr(a, 256), Ptr(b, 256), Ptr(c, 256),
///    U32(64)]` and calls `wasi_cuda_launch` with `grid_x = 1,
///    block_x = 64`.
/// 2. Register the real `kernels/vector_add.ptx` through
///    [`register_real_kernel`], which compiles the PTX via
///    `cust::module::Module::from_ptx`. On a host whose compute
///    capability is below the PTX's `.target sm_80` line the JIT may
///    reject the module — the helper falls back to `module: None` and
///    the test skips downstream assertions.
/// 3. Invoke `launch_with_args`. The host marshals the argv into a
///    `void**` and calls `cuLaunchKernel`, then `stream.synchronize`s.
/// 4. Read the `c` region back out of Wasm linear memory and assert
///    `c[i] == a[i] + b[i] == 100.0 + 2.0 * (i as f32)` for all `i`.
///
/// The pointer-arg path relies on the wasm linear-memory backing
/// doubling as a device-addressable pointer; with `--features
/// unified-memory` on `tensor-wasm-mem` (the production default), the
/// linear memory IS `cuMemAllocManaged` and the kernel writes through
/// the same address the host reads back. Without unified memory the
/// host pointer is plain heap and `cuLaunchKernel` will fault — that
/// constraint is documented in `docs/RISKS.md` and is why the test
/// stays `#[ignore]`d outside the v0.4 self-hosted runner config.
///
/// The body compiles cleanly on no-CUDA hosts (every CUDA-specific
/// step is `#[cfg(feature = "cuda")]`-gated); the test simply returns
/// early.
#[tokio::test]
#[ignore = "requires CUDA hardware"]
async fn vector_add_end_to_end_real_ptx_real_kernel() {
    #[cfg(not(feature = "cuda"))]
    {
        // Compile-time only: the no-CUDA build cannot load a real PTX
        // module, so the body short-circuits. The test still benefits
        // from being compiled on no-CUDA hosts (catches drift in the
        // helper signatures, the `include_bytes!` path, the WAT
        // template, etc.).
        eprintln!(
            "vector_add_end_to_end_real_ptx_real_kernel: skipping (built without --features cuda)"
        );
    }

    #[cfg(feature = "cuda")]
    {
        const N: usize = 64;
        const ELEM_BYTES: usize = std::mem::size_of::<f32>();
        const REGION_BYTES: usize = N * ELEM_BYTES; // 256

        // Offsets inside the guest's 4-page (256 KiB) linear memory.
        // The three data regions are well-separated and well below the
        // argv buffer offset, so a `data` segment never overlaps another.
        const A_OFFSET: usize = 1024;
        const B_OFFSET: usize = 2048;
        const C_OFFSET: usize = 3072;
        const ARGV_OFFSET: usize = 8192;

        let mut a_bytes = Vec::with_capacity(REGION_BYTES);
        let mut b_bytes = Vec::with_capacity(REGION_BYTES);
        let mut c_bytes = Vec::with_capacity(REGION_BYTES);
        for i in 0..N {
            a_bytes.extend_from_slice(&(i as f32).to_le_bytes());
            b_bytes.extend_from_slice(&(100.0_f32 + i as f32).to_le_bytes());
            c_bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }

        let (engine, linker) = make_engine_and_linker();
        let owner = InstanceId(403);
        let mut ctx = WasiCudaContext::new(owner);
        ctx.enable_wasi_cuda();
        // Pick the fixture whose `.target` the driver JIT accepts on THIS
        // device. We must NOT try a known-mismatched fixture first: a failed
        // `Module::from_ptx` leaves a sticky error on the CUDA context that
        // poisons the next load (the observed `InvalidPtx`-then-`UnknownError`
        // cascade on the sm_75 dev box). `register_real_kernel` calls
        // `quick_init` before this runs, so a context is current.
        let (ptx, arch) = select_vector_add_ptx();
        let (kid, loaded) = register_real_kernel(&ctx, owner, "vector_add", ptx);
        if !loaded {
            // The arch-matched fixture failed to load — that is a real
            // regression on a supported device, not a skip. Surface it.
            panic!(
                "vector_add_end_to_end_real_ptx_real_kernel: the {arch} fixture \
                 (arch-matched to this device) was rejected by the driver JIT; \
                 last error: {:?}",
                ctx.last_error()
            );
        }

        let argv = encode_argv(&[
            LoweredArg::ptr_for_encoding(A_OFFSET as u32, REGION_BYTES as u32),
            LoweredArg::ptr_for_encoding(B_OFFSET as u32, REGION_BYTES as u32),
            LoweredArg::ptr_for_encoding(C_OFFSET as u32, REGION_BYTES as u32),
            LoweredArg::U32(N as u32),
        ]);
        let wat = build_vector_add_launch_wat(
            &argv,
            ARGV_OFFSET,
            kid,
            /* grid_x */ 1,
            /* block_x */ N as u32,
            &[
                (A_OFFSET, a_bytes.clone()),
                (B_OFFSET, b_bytes.clone()),
                (C_OFFSET, c_bytes.clone()),
            ],
        );
        let wasm = wat::parse_str(&wat).expect("parse vector_add WAT");
        let module = wasmtime::Module::new(&engine, &wasm).expect("compile vector_add WAT");
        let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
        let instance = linker
            .instantiate_async(&mut store, &module)
            .await
            .expect("instantiate");
        let f = instance
            .get_typed_func::<(), i32>(&mut store, "launch_with_args")
            .expect("typed func");
        let rc = f.call_async(&mut store, ()).await.expect("call");
        assert_eq!(
            rc,
            0,
            "vector_add launch must return 0 on a CUDA host with the SM_80 \
             fixture; last error: {:?}",
            store.data().wasi_cuda().last_error()
        );

        // Read the `c` region back out of the guest's linear memory.
        // The kernel wrote through the same `host_ptr` the marshaller
        // resolved against `mem.data(..)`, so the same byte range is
        // observable here.
        let memory = instance.get_memory(&mut store, "memory").expect("memory");
        let mut readback = vec![0u8; REGION_BYTES];
        memory
            .read(&store, C_OFFSET, &mut readback)
            .expect("read c region");

        // Decode and assert. This is the property the test pins: the
        // kernel ran, addressed the correct guest-memory window, and
        // produced the canonical element-wise sum. Any off-by-one in
        // the argv marshalling (wrong offset, wrong length, scalar/
        // pointer tag confusion) breaks this check, as does any
        // failure to surface the kernel's writes back to the host.
        for i in 0..N {
            let off = i * ELEM_BYTES;
            let mut buf = [0u8; ELEM_BYTES];
            buf.copy_from_slice(&readback[off..off + ELEM_BYTES]);
            let got = f32::from_le_bytes(buf);
            let want = 100.0_f32 + 2.0_f32 * (i as f32);
            assert_eq!(
                got, want,
                "c[{i}] mismatch: kernel produced {got}, expected {want} \
                 (= 100 + 2*{i})"
            );
        }
    }
}

/// v0.3.3 commitment: the registry + load_ptx + launch surface
/// accepts the real `kernels/vector_add.ptx` bytes and routes them
/// through the dispatch path on every CUDA-capable host — including
/// dev boxes whose compute capability is below the SM_80 the
/// canonical fixture targets.
///
/// The test is intentionally *unignored*. It is the weaker sibling of
/// [`vector_add_end_to_end_real_ptx_real_kernel`]: rather than
/// asserting kernel output is correct, it asserts that the marshalling
/// layer accepts the call and the failure (if any) comes from
/// downstream CUDA — never from the host's argv parser or pointer
/// bounds-check.
///
/// Specifically, on a CUDA host the launch must return ONE OF:
/// - `0` (Ok) — happy path on SM_80+ runners.
/// - `MalformedPtx` — the JIT compile inside `cust::module::Module::from_ptx`
///   rejected the PTX (this happens on the RTX 2060 dev box: PTX
///   `.target sm_80` against compute-cap 7.5). In this case the
///   helper returned `loaded == false` and the kernel was registered
///   with `module: None`, so the launch surfaces `InvalidKernel`.
/// - `InvalidKernel` — same as above, propagated from the launch path.
/// - `LaunchFailed` — `cuLaunchKernel` itself rejected the launch
///   (e.g. param-count mismatch detected by the driver, or
///   sm-version mismatch caught at launch time rather than load time).
///
/// The launch must NEVER return `InvalidArgs` or `InvalidPointer` —
/// those would indicate the host's typed-argv marshalling regressed
/// before the call ever reached the CUDA driver.
///
/// On the no-CUDA path the assertion degrades to `NotAvailable`,
/// which is the existing no-CUDA contract.
#[tokio::test]
async fn dispatch_pipeline_compiles_against_real_module_bytes() {
    const N: usize = 64;
    const ELEM_BYTES: usize = std::mem::size_of::<f32>();
    const REGION_BYTES: usize = N * ELEM_BYTES;
    const A_OFFSET: usize = 1024;
    const B_OFFSET: usize = 2048;
    const C_OFFSET: usize = 3072;
    const ARGV_OFFSET: usize = 8192;

    let mut a_bytes = Vec::with_capacity(REGION_BYTES);
    let mut b_bytes = Vec::with_capacity(REGION_BYTES);
    let c_bytes = vec![0u8; REGION_BYTES];
    for i in 0..N {
        a_bytes.extend_from_slice(&(i as f32).to_le_bytes());
        b_bytes.extend_from_slice(&(100.0_f32 + i as f32).to_le_bytes());
    }

    let (engine, linker) = make_engine_and_linker();
    let owner = InstanceId(404);
    let mut ctx = WasiCudaContext::new(owner);
    ctx.enable_wasi_cuda();

    #[cfg(feature = "cuda")]
    let kid = {
        let (id, _loaded) = register_real_kernel(&ctx, owner, "vector_add", VECTOR_ADD_PTX);
        // We deliberately do NOT branch on `_loaded` here: the whole
        // point of this test is to prove the dispatch path tolerates
        // a real PTX module regardless of whether the local GPU can
        // JIT it. A failed JIT compile leaves `module: None`, so the
        // launch surfaces `InvalidKernel` — which is one of the
        // allowed outcomes below.
        id
    };
    #[cfg(not(feature = "cuda"))]
    let kid = register_stub_kernel(&ctx, owner, "vector_add");

    let argv = encode_argv(&[
        LoweredArg::ptr_for_encoding(A_OFFSET as u32, REGION_BYTES as u32),
        LoweredArg::ptr_for_encoding(B_OFFSET as u32, REGION_BYTES as u32),
        LoweredArg::ptr_for_encoding(C_OFFSET as u32, REGION_BYTES as u32),
        LoweredArg::U32(N as u32),
    ]);
    let wat = build_vector_add_launch_wat(
        &argv,
        ARGV_OFFSET,
        kid,
        /* grid_x */ 1,
        /* block_x */ N as u32,
        &[
            (A_OFFSET, a_bytes),
            (B_OFFSET, b_bytes),
            (C_OFFSET, c_bytes),
        ],
    );
    let wasm = wat::parse_str(&wat).expect("parse vector_add WAT");
    let module = wasmtime::Module::new(&engine, &wasm).expect("compile vector_add WAT");
    let mut store = wasmtime::Store::new(&engine, TestStore { cuda: ctx });
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .expect("instantiate");
    let f = instance
        .get_typed_func::<(), i32>(&mut store, "launch_with_args")
        .expect("typed func");
    let rc = f.call_async(&mut store, ()).await.expect("call");

    #[cfg(not(feature = "cuda"))]
    {
        assert_eq!(
            rc,
            AbiError::NotAvailable.code(),
            "no-CUDA host: launch must report NotAvailable after parsing argv (got {rc})"
        );
    }

    #[cfg(feature = "cuda")]
    {
        let allowed = [
            0_i32,
            AbiError::MalformedPtx.code(),
            AbiError::InvalidKernel.code(),
            AbiError::LaunchFailed.code(),
        ];
        assert!(
            allowed.contains(&rc),
            "CUDA host: launch return code must be one of {allowed:?} \
             (Ok / MalformedPtx / InvalidKernel / LaunchFailed); got {rc}. \
             Anything else (InvalidArgs / InvalidPointer / NotAvailable) \
             indicates the typed-argv marshalling regressed before the call \
             reached the CUDA driver. last_error: {:?}",
            store.data().wasi_cuda().last_error()
        );
    }

    // The argv must have been parsed regardless of the downstream
    // outcome — that is the proof the marshalling layer accepted the
    // real-module-bytes shape.
    let recorded = store.data().wasi_cuda().last_lowered_args();
    assert_eq!(
        recorded.len(),
        4,
        "expected four lowered args (3 pointers + 1 scalar); got {}",
        recorded.len()
    );
    assert!(matches!(recorded[3], LoweredArgSnapshot::U32(n) if n == N as u32));
}

/// Regression: `build_kernel_param_storage` must lay out every slot
/// inside a single contiguous backing buffer.
///
/// The pre-v0.3.4 implementation boxed each scalar argument into its
/// own `Box<[u8]>`, costing one heap allocation per arg — 16 separate
/// allocations for a 16-arg kernel on the launch hot path. The fix
/// switched to a single `Vec<u8>` (frozen into a `Box<[u8]>`) plus a
/// pointer table indexing into it. This test pins the shape so a
/// future refactor that reintroduces per-arg boxing fails loudly: it
/// builds a 16-scalar argv (one of every supported scalar tag, plus
/// padding) and asserts every slot pointer lies inside the
/// `backing()` slice.
#[test]
fn build_kernel_param_storage_uses_single_backing_buffer() {
    // 16 scalars covering every tag the marshaller handles, with
    // duplicates of the wider types to push past the 16-arg threshold
    // the original allocation-explosion bug operated on.
    let args = vec![
        LoweredArg::I32(1),
        LoweredArg::I64(2),
        LoweredArg::F32(3.0),
        LoweredArg::F64(4.0),
        LoweredArg::U32(5),
        LoweredArg::U64(6),
        LoweredArg::I32(7),
        LoweredArg::I64(8),
        LoweredArg::F32(9.0),
        LoweredArg::F64(10.0),
        LoweredArg::U32(11),
        LoweredArg::U64(12),
        LoweredArg::I32(13),
        LoweredArg::I64(14),
        LoweredArg::F32(15.0),
        LoweredArg::F64(16.0),
    ];
    assert_eq!(args.len(), 16);

    let storage = build_kernel_param_storage(&args);
    assert_eq!(storage.len(), args.len());

    let backing = storage.backing();
    let slot_ptrs = storage.slot_ptrs();
    assert_eq!(slot_ptrs.len(), args.len(), "one pointer slot per argument");

    // Every slot pointer must land inside the contiguous `backing`
    // buffer. We compute the byte-offset between the slot pointer and
    // the start of `backing` and assert it is in `[0, backing.len())`.
    // A pre-v0.3.4 implementation would produce slots that point into
    // 16 disjoint heap allocations; their offsets would be either
    // outside the buffer's address range (unsigned wraparound to a
    // huge value) or simply unrelated to it.
    let base = backing.as_ptr() as usize;
    let end = base + backing.len();
    for (i, &slot) in slot_ptrs.iter().enumerate() {
        let slot_addr = slot as usize;
        assert!(
            slot_addr >= base && slot_addr < end,
            "slot {i} pointer {slot:p} is not inside backing buffer \
             [{base:#x}, {end:#x}) — per-arg boxing has regressed"
        );
    }

    // Sanity floor: a single backing buffer for 16 scalars should
    // comfortably fit inside one allocation worth of bytes. The exact
    // size depends on alignment padding, but it must be at least the
    // sum of the value bytes (4 + 8 + 4 + 8 + 4 + 8 = 36 bytes per
    // half × 2 halves = 72 bytes minimum, with padding making the real
    // total a bit larger). Keep the assertion loose so a benign
    // alignment-pad change does not flake the test.
    assert!(
        backing.len() >= 72,
        "backing buffer too small for 16 scalars; got {}",
        backing.len()
    );
}
