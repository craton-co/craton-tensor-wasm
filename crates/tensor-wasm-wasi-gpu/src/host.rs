// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Host-side implementations of the `wasi:cuda` functions registered with
//! `wasmtime::Linker`.
//!
//! On non-CUDA hosts every function returns [`AbiError::NotAvailable`] via
//! its wire-format negative i32 code. The Wasm side cannot tell whether
//! that's because the host lacks CUDA or because the runtime explicitly
//! declined a call — both are appropriate "kernel did not run" signals.
//!
//! When the `cuda` feature is enabled (S16+ on real hardware) the bodies
//! switch to real CUDA dispatch via the `cust` crate.
//!
//! NOTE: Cuda-feature code paths in this file are compile-tested on CUDA
//! hosts only; on no-CUDA hosts only the `#[cfg(not(feature = "cuda"))]`
//! branches are exercised. The cuda branches must be kept consistent with
//! the cust 0.3.x API.
//!
//! ## Launch dimension caps
//!
//! Every launch is validated against CUDA hardware ceilings before any
//! driver call:
//! - block dimensions must each be in `[1, MAX_BLOCK_DIM]` and the product
//!   `block_x * block_y * block_z` must be at most [`MAX_THREADS_PER_BLOCK`]
//!   (1024 across all current GPUs).
//! - grid dimensions must each be in `[1, MAX_GRID_DIM]`
//!   (2^31 - 1, the CUDA driver maximum for `grid_x`).
//! - `shared_mem` is bounded only by the device's per-block shared-memory
//!   limit; we leave that to the driver.
//!
//! Violations return [`AbiError::InvalidDimensions`] without ever calling
//! into `cuLaunchKernel` — the failure is reported with a structured
//! `last_error` describing which axis tripped the cap.
//!
//! ## Kernel argument marshalling (v0.1.0)
//!
//! The launch host function takes `(args_ptr, args_len)` describing a
//! byte buffer in the guest's linear memory. v0.1.0 intentionally
//! short-circuits the dispatch path: the args region is bounds-checked
//! and read into a host-side `Vec<u8>` (so pointer bugs surface
//! immediately), but a non-empty buffer is rejected with
//! [`AbiError::KernelArgsUnsupported`] because the per-arg packing
//! format hasn't been frozen yet and `cust 0.3.x`'s `launch!` macro
//! takes statically-typed args (synthesizing a dynamic argv from a raw
//! byte buffer requires dropping to `cuLaunchKernel` directly, which is
//! the BAL-422 / v0.2 effort).
//!
//! The distinction between [`AbiError::InvalidArgs`] (your input is
//! malformed) and [`AbiError::KernelArgsUnsupported`] (your input is
//! fine, the host can't pass it to CUDA yet) matters for guest
//! debugging — see `docs/RISKS.md` for the v0.1.0 contract.
//!
//! Only the zero-arg launch shape reaches `cuLaunchKernel` in this
//! release. Richer marshalling (per-arg type tags, packed primitives,
//! GPU-side pointer translation) lands in `BAL-422`; guests that need
//! parameters today should track that ticket.
//!
//! On non-CUDA builds the args region is bounds-checked but never
//! dereferenced — the launch returns [`AbiError::NotAvailable`] regardless.

use std::sync::Arc;
use std::sync::Mutex;

use tensor_wasm_core::types::{InstanceId, KernelId};
use tracing::{info, info_span, warn};
use wasmtime::{Caller, Linker};

use crate::abi::{
    AbiError, FN_LAST_ERROR_COPY, FN_LAST_ERROR_LEN, FN_LAUNCH, FN_LOAD_PTX, FN_SYNC,
    MAX_BLOCK_DIM, MAX_GRID_DIM, MAX_PTX_BYTES, MAX_THREADS_PER_BLOCK, MODULE,
};
use crate::async_dispatch::BackPressure;
use crate::registry::{KernelEntry, KernelRegistry};

/// Per-instance host state passed to wasi-cuda calls.
///
/// `WasiCudaContext` is stored in the wasmtime `Store`'s data type (or in a
/// resource handle thereon). The executor (`tensor-wasm-exec`) creates one per
/// instance at spawn time.
pub struct WasiCudaContext {
    /// Owning instance.
    pub instance_id: InstanceId,
    /// Kernel registry for this instance.
    pub registry: Arc<KernelRegistry>,
    /// Last error message produced by a wasi-cuda call on this instance.
    pub last_error: Mutex<Option<String>>,
    /// Back-pressure cap shared across launches. Wrapped in `Arc` so an
    /// executor (S7-style) can construct one cap and hand a clone to each
    /// per-instance context — making the limit a process-wide ceiling rather
    /// than a per-instance one. With [`WasiCudaContext::new`] each context
    /// still gets its own cap.
    pub back_pressure: Arc<BackPressure>,
}

impl WasiCudaContext {
    /// Construct a fresh context for the given instance with a dedicated
    /// (un-shared) back-pressure cap.
    pub fn new(instance_id: InstanceId) -> Self {
        Self {
            instance_id,
            registry: Arc::new(KernelRegistry::new()),
            last_error: Mutex::new(None),
            back_pressure: Arc::new(BackPressure::new()),
        }
    }

    /// Construct a context that shares the given [`BackPressure`] cap with
    /// other contexts. Used by the executor to enforce one process-wide
    /// concurrency limit across all Wasm instances.
    pub fn with_back_pressure(instance_id: InstanceId, bp: Arc<BackPressure>) -> Self {
        Self {
            instance_id,
            registry: Arc::new(KernelRegistry::new()),
            last_error: Mutex::new(None),
            back_pressure: bp,
        }
    }

    /// Borrow the shared back-pressure handle for observability / sharing.
    pub fn back_pressure(&self) -> &Arc<BackPressure> {
        &self.back_pressure
    }

    fn record_error(&self, msg: impl Into<String>) {
        let msg = msg.into();
        warn!(target: "tensor_wasm_wasi_gpu::host", instance = %self.instance_id, %msg, "wasi-cuda error");
        *self.last_error.lock().expect("last_error poisoned") = Some(msg);
    }

    /// Borrow the most recent error message.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().expect("last_error poisoned").clone()
    }
}

/// Trait implemented by store data types that can hand out a [`WasiCudaContext`].
///
/// `tensor-wasm-exec`'s `InstanceState` will implement this in a follow-up wiring
/// session; defining the trait now keeps the linker registration generic.
pub trait HasWasiCuda {
    /// Borrow the wasi-cuda context.
    fn wasi_cuda(&self) -> &WasiCudaContext;
}

/// Register all wasi-cuda host functions on a wasmtime `Linker`.
///
/// `T` is the store data type and must implement [`HasWasiCuda`].
///
/// `FN_LAUNCH` is registered with `func_wrap_async` so that on the CUDA
/// path the host can `tokio::task::spawn_blocking(stream.synchronize())`
/// without blocking the wasmtime fiber. The no-CUDA branch wraps the
/// existing synchronous path in an immediately-ready future so a single
/// async wrapper covers both feature configurations.
pub fn add_to_linker<T: HasWasiCuda + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    linker.func_wrap(
        MODULE,
        FN_LOAD_PTX,
        |mut caller: Caller<'_, T>,
         ptx_ptr: i32,
         ptx_len: i32,
         entry_ptr: i32,
         entry_len: i32|
         -> i64 {
            match load_ptx_impl(&mut caller, ptx_ptr, ptx_len, entry_ptr, entry_len) {
                Ok(k) => k.0 as i64,
                Err(e) => e.code() as i64,
            }
        },
    )?;

    linker.func_wrap_async(
        MODULE,
        FN_LAUNCH,
        |mut caller: Caller<'_, T>,
         (
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
        ): (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32)|
         -> Box<dyn std::future::Future<Output = i32> + Send + '_> {
            Box::new(async move {
                launch_impl_async(
                    &mut caller,
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
                )
                .await
                .map_or_else(|e| e.code(), |_| 0)
            })
        },
    )?;

    linker.func_wrap(MODULE, FN_SYNC, |caller: Caller<'_, T>| -> i32 {
        sync_impl(&caller).map_or_else(|e| e.code(), |_| 0)
    })?;

    // Note: `FN_LAST_ERROR_PTR` is deliberately NOT registered. The original
    // "host hands the guest a pointer into a pre-allocated buffer" shape
    // required coordination with the Wasm module's allocator; the
    // `last_error_copy` design below is the working path — the guest calls
    // `last_error_len` to learn the size, allocates its own buffer, and
    // hands the host a `(dst_ptr, dst_len)` pair to write into. The
    // `FN_LAST_ERROR_PTR` constant is preserved in `abi.rs` for ABI
    // backwards-compat but is now an unimported name from the guest's POV.

    linker.func_wrap(MODULE, FN_LAST_ERROR_LEN, |caller: Caller<'_, T>| -> i32 {
        caller
            .data()
            .wasi_cuda()
            .last_error()
            .map(|s| s.len() as i32)
            .unwrap_or(0)
    })?;

    linker.func_wrap(
        MODULE,
        FN_LAST_ERROR_COPY,
        |mut caller: Caller<'_, T>, dst_ptr: i32, dst_len: i32| -> i32 {
            // Sentinel return values:
            //   `0`              — no error currently recorded.
            //   `-2` (`AbiError::InvalidPointer.code()`) — the guest's
            //   `(dst_ptr, dst_len)` is invalid or the write into linear
            //   memory failed.
            //   `n > 0`          — number of bytes copied.
            // Distinguishing "no error" from "invalid pointer" matters: an
            // earlier version returned `0` on the write-failure path, which
            // made buggy guests silently consume corrupted error info.
            if dst_ptr < 0 || dst_len < 0 {
                return AbiError::InvalidPointer.code();
            }
            if dst_len == 0 {
                // Zero-length destination: technically valid but copies
                // nothing. Return 0 (matches "no error") rather than
                // InvalidPointer; the guest knows it asked for 0 bytes.
                return 0;
            }
            let msg = match caller.data().wasi_cuda().last_error() {
                Some(s) => s,
                None => return 0,
            };
            let bytes = msg.as_bytes();
            let to_copy = std::cmp::min(bytes.len(), dst_len as usize);
            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return AbiError::InvalidPointer.code(),
            };
            // Pre-validate the destination region against the current
            // memory size so a failed write returns InvalidPointer rather
            // than the ambiguous 0.
            let mem_len = memory.data(&caller).len();
            let start = dst_ptr as usize;
            let end = match start.checked_add(to_copy) {
                Some(e) => e,
                None => return AbiError::InvalidPointer.code(),
            };
            if end > mem_len {
                return AbiError::InvalidPointer.code();
            }
            let buf = bytes[..to_copy].to_vec();
            if memory.write(&mut caller, dst_ptr as usize, &buf).is_err() {
                return AbiError::InvalidPointer.code();
            }
            to_copy as i32
        },
    )?;

    Ok(())
}

fn read_bytes<T>(caller: &mut Caller<'_, T>, ptr: i32, len: i32) -> Result<Vec<u8>, AbiError> {
    if len < 0 || ptr < 0 {
        return Err(AbiError::InvalidPointer);
    }
    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or(AbiError::InvalidPointer)?;
    let data = memory.data(&caller);
    let start = ptr as usize;
    // `checked_add` here catches `ptr + len > usize::MAX`; without it a
    // guest could ask for `(ptr = usize::MAX - 1, len = 4)` and wrap to a
    // small `end` that looks in-bounds.
    let end = start
        .checked_add(len as usize)
        .ok_or(AbiError::InvalidPointer)?;
    if end > data.len() {
        return Err(AbiError::InvalidPointer);
    }
    Ok(data[start..end].to_vec())
}

fn load_ptx_impl<T: HasWasiCuda>(
    caller: &mut Caller<'_, T>,
    ptx_ptr: i32,
    ptx_len: i32,
    entry_ptr: i32,
    entry_len: i32,
) -> Result<KernelId, AbiError> {
    let _span = info_span!(
        "wasi_cuda.load_ptx",
        instance = %caller.data().wasi_cuda().instance_id,
        ptx_bytes = ptx_len as u64,
        entry_bytes = entry_len as u64,
    )
    .entered();
    if (ptx_len as usize) > MAX_PTX_BYTES {
        caller.data().wasi_cuda().record_error(format!(
            "load_ptx: ptx_len {ptx_len} exceeds MAX_PTX_BYTES {MAX_PTX_BYTES}"
        ));
        return Err(AbiError::QuotaExceeded);
    }
    let ptx = read_bytes(caller, ptx_ptr, ptx_len)?;
    let entry_bytes = read_bytes(caller, entry_ptr, entry_len)?;
    let entry = String::from_utf8(entry_bytes).map_err(|_| {
        caller
            .data()
            .wasi_cuda()
            .record_error("load_ptx: entry name is not valid UTF-8");
        AbiError::InvalidArgs
    })?;

    #[cfg(not(feature = "cuda"))]
    {
        // Validate format minimally even on the non-CUDA path: empty or
        // non-UTF8 PTX is malformed.
        if ptx.is_empty() {
            caller
                .data()
                .wasi_cuda()
                .record_error("load_ptx: PTX bytes empty");
            return Err(AbiError::MalformedPtx);
        }
        let ptx_str = match std::str::from_utf8(&ptx) {
            Ok(s) => s,
            Err(_) => {
                caller
                    .data()
                    .wasi_cuda()
                    .record_error("load_ptx: PTX bytes are not valid UTF-8");
                return Err(AbiError::MalformedPtx);
            }
        };
        // Structural sanity check: every well-formed PTX file declares a
        // `.version`, a `.target` SM, and at least one `.entry` kernel.
        // Missing any of these means the blob is not a PTX module — reject
        // it as MalformedPtx so the stub matches the plan's S8 done-when.
        for directive in [".version", ".target", ".entry"] {
            if !ptx_str.contains(directive) {
                caller.data().wasi_cuda().record_error(format!(
                    "load_ptx: PTX missing required directive {directive}"
                ));
                return Err(AbiError::MalformedPtx);
            }
        }
        let owner = caller.data().wasi_cuda().instance_id;
        let entry_record = KernelEntry {
            owner,
            entry: entry.clone(),
            ptx_bytes_len: ptx.len(),
        };
        let registry = caller.data().wasi_cuda().registry.clone();
        let id = registry.register(entry_record)?;
        info!(target: "tensor_wasm_wasi_gpu::host", instance = %owner, kernel = %id, entry, "PTX registered (stub: cuda feature off)");
        Ok(id)
    }

    #[cfg(feature = "cuda")]
    {
        use cust::module::Module;
        // Real path: compile the PTX through cust::module::Module::from_ptx.
        // Module::from_ptx panics if `string` contains a nul byte, so we
        // explicitly reject nul bytes before handing the slice over.
        let ptx_str = std::str::from_utf8(&ptx).map_err(|_| {
            caller
                .data()
                .wasi_cuda()
                .record_error("load_ptx: PTX bytes are not valid UTF-8");
            AbiError::MalformedPtx
        })?;
        if ptx_str.as_bytes().contains(&0u8) {
            caller
                .data()
                .wasi_cuda()
                .record_error("load_ptx: PTX bytes contain an interior NUL");
            return Err(AbiError::MalformedPtx);
        }
        let module = Module::from_ptx(ptx_str, &[]).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("load_ptx: cust compile failed: {e:?}"));
            AbiError::MalformedPtx
        })?;
        let owner = caller.data().wasi_cuda().instance_id;
        let entry_record = KernelEntry {
            owner,
            entry: entry.clone(),
            ptx_bytes_len: ptx.len(),
            module: Some(Arc::new(module)),
        };
        let registry = caller.data().wasi_cuda().registry.clone();
        let id = registry.register(entry_record)?;
        info!(target: "tensor_wasm_wasi_gpu::host", instance = %owner, kernel = %id, entry, "PTX compiled and registered via cust");
        Ok(id)
    }
}

/// Common argument-region validation extracted from the launch path so the
/// sync and async wrappers share one implementation.
///
/// Validates:
/// 1. `args_ptr` / `args_len` are non-negative and the region fits in
///    linear memory.
/// 2. `kernel_id` is non-negative.
/// 3. Block dimensions fit `[1, MAX_BLOCK_DIM]` each and the thread-per-
///    block product is `<= MAX_THREADS_PER_BLOCK`.
/// 4. Grid dimensions fit `[1, MAX_GRID_DIM]` each.
/// 5. `shared_mem >= 0`.
///
/// Failures return [`AbiError::InvalidDimensions`] for dimension-cap
/// violations and [`AbiError::InvalidPointer`] for memory-region issues,
/// allowing the guest to distinguish a launch-shape bug from a memory bug.
#[allow(clippy::too_many_arguments)]
fn validate_launch_args<T: HasWasiCuda>(
    caller: &mut Caller<'_, T>,
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
) -> Result<KernelId, AbiError> {
    if args_len < 0 || args_ptr < 0 {
        caller.data().wasi_cuda().record_error(format!(
            "launch: negative args_ptr ({args_ptr}) or args_len ({args_len})"
        ));
        return Err(AbiError::InvalidPointer);
    }
    if args_len > 0 {
        let memory = caller
            .get_export("memory")
            .and_then(|e| e.into_memory())
            .ok_or_else(|| {
                caller
                    .data()
                    .wasi_cuda()
                    .record_error("launch: caller has no exported memory but args_len > 0");
                AbiError::InvalidPointer
            })?;
        let mem_len = memory.data(&caller).len();
        let start = args_ptr as usize;
        let end = start.checked_add(args_len as usize).ok_or_else(|| {
            caller.data().wasi_cuda().record_error(format!(
                "launch: args_ptr + args_len overflows usize ({args_ptr} + {args_len})"
            ));
            AbiError::InvalidPointer
        })?;
        if end > mem_len {
            caller.data().wasi_cuda().record_error(format!(
                "launch: args region [{start}, {end}) exceeds Wasm memory len {mem_len}"
            ));
            return Err(AbiError::InvalidPointer);
        }
    }
    if kernel_id < 0 {
        return Err(AbiError::InvalidKernel);
    }
    // Per-axis lower / upper bounds and the thread-per-block product cap.
    if block_x <= 0 || block_y <= 0 || block_z <= 0 {
        caller.data().wasi_cuda().record_error(format!(
            "launch: block dim must be >= 1 (got {block_x}, {block_y}, {block_z})"
        ));
        return Err(AbiError::InvalidDimensions);
    }
    if grid_x <= 0 || grid_y <= 0 || grid_z <= 0 {
        caller.data().wasi_cuda().record_error(format!(
            "launch: grid dim must be >= 1 (got {grid_x}, {grid_y}, {grid_z})"
        ));
        return Err(AbiError::InvalidDimensions);
    }
    if shared_mem < 0 {
        caller
            .data()
            .wasi_cuda()
            .record_error(format!("launch: shared_mem must be >= 0 (got {shared_mem})"));
        return Err(AbiError::InvalidDimensions);
    }
    // Per-axis block-dim ceilings (CUDA hardware: 1024 for x and y, 64 for z
    // on current SMs; we cap each at MAX_BLOCK_DIM and rely on the
    // threads-per-block product check below to catch the z-axis variant).
    if (block_x as u32) > MAX_BLOCK_DIM
        || (block_y as u32) > MAX_BLOCK_DIM
        || (block_z as u32) > MAX_BLOCK_DIM
    {
        caller.data().wasi_cuda().record_error(format!(
            "launch: block dim exceeds MAX_BLOCK_DIM={MAX_BLOCK_DIM} (got {block_x}, {block_y}, {block_z})"
        ));
        return Err(AbiError::InvalidDimensions);
    }
    let threads_per_block = (block_x as u64)
        .checked_mul(block_y as u64)
        .and_then(|v| v.checked_mul(block_z as u64))
        .ok_or_else(|| {
            caller
                .data()
                .wasi_cuda()
                .record_error("launch: block dim product overflows u64");
            AbiError::InvalidDimensions
        })?;
    if threads_per_block > MAX_THREADS_PER_BLOCK as u64 {
        caller.data().wasi_cuda().record_error(format!(
            "launch: threads-per-block {threads_per_block} exceeds MAX_THREADS_PER_BLOCK={MAX_THREADS_PER_BLOCK}"
        ));
        return Err(AbiError::InvalidDimensions);
    }
    // Grid-axis ceilings. `MAX_GRID_DIM` is 2^31 - 1 (CUDA driver max for
    // grid_x). i32 already enforces this implicitly (positive i32 maxes at
    // 2^31 - 1), but we keep the explicit cast-and-compare so a future
    // widening of the wire types doesn't silently raise the cap.
    if (grid_x as u32) > MAX_GRID_DIM
        || (grid_y as u32) > MAX_GRID_DIM
        || (grid_z as u32) > MAX_GRID_DIM
    {
        caller.data().wasi_cuda().record_error(format!(
            "launch: grid dim exceeds MAX_GRID_DIM={MAX_GRID_DIM} (got {grid_x}, {grid_y}, {grid_z})"
        ));
        return Err(AbiError::InvalidDimensions);
    }
    Ok(KernelId(kernel_id as u64))
}

/// Asynchronous wrapper around the launch implementation.
///
/// On the no-CUDA path the body is essentially identical to the old
/// synchronous `launch_impl`: validate, acquire a back-pressure permit,
/// then return `Err(NotAvailable)`. On the CUDA path the body builds and
/// dispatches a real kernel via `cust`, then awaits a `spawn_blocking`
/// `stream.synchronize()` so the wasmtime fiber may be suspended while
/// the GPU runs.
///
/// See the module-level docs for the kernel-args marshalling contract.
#[allow(clippy::too_many_arguments)]
async fn launch_impl_async<T: HasWasiCuda>(
    caller: &mut Caller<'_, T>,
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
) -> Result<(), AbiError> {
    // NOTE: we deliberately do *not* hold an `EnteredSpan` here. That guard
    // is `!Send`, which would poison the `Send` future required by
    // `func_wrap_async`. Span context for this call is recorded through
    // the synchronous helpers (validation, lookup) which capture it via
    // the surrounding `info!`/`warn!` events.
    let _ = info_span!(
        "wasi_cuda.launch",
        instance = %caller.data().wasi_cuda().instance_id,
        kernel = kernel_id,
        grid_x = grid_x, grid_y = grid_y, grid_z = grid_z,
        block_x = block_x, block_y = block_y, block_z = block_z,
        shared_mem = shared_mem,
    );

    let kid = validate_launch_args(
        caller, kernel_id, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, args_ptr,
        args_len,
    )?;
    let owner = caller.data().wasi_cuda().instance_id;
    let registry = caller.data().wasi_cuda().registry.clone();

    // Read the args region into a host-side buffer before the back-pressure
    // permit is acquired so a slow launch can't pin a memory borrow across
    // an await. The buffer is moved into the CUDA branch below or
    // discarded on the no-CUDA branch.
    //
    // v0.1.0 marshalling: we read and bounds-check the buffer here, but
    // the CUDA dispatch path below only supports the zero-arg launch
    // shape. Non-empty args returns `KernelArgsUnsupported` so the guest
    // can distinguish "your input is malformed" (`InvalidArgs`) from
    // "your input is fine, but the host can't pass it to CUDA in this
    // release" (tracked in BAL-422). The bounds-check still runs first
    // because pointer bugs must surface regardless of whether dispatch
    // goes through — a malicious guest cannot trade a `MemoryFault` for
    // a `KernelArgsUnsupported`.
    let args_buf: Vec<u8> = if args_len > 0 {
        read_bytes(caller, args_ptr, args_len)?
    } else {
        Vec::new()
    };
    if !args_buf.is_empty() {
        // TODO(v0.2): wire dynamic argv through cuLaunchKernel (BAL-422).
        // cust 0.3.x's `launch!` macro takes statically-typed args, so
        // we cannot synthesize a dynamic argv from a raw byte buffer
        // without dropping to the raw driver API.
        caller.data().wasi_cuda().record_error(format!(
            "launch: dynamic kernel args not implemented in v0.1.0; \
             received args_len={args_len} bytes. See docs/RISKS.md and \
             wit/wasi-cuda.wit (BAL-422)."
        ));
        return Err(AbiError::KernelArgsUnsupported);
    }

    // Eagerly take a strong, owned handle to the kernel (Arc-wrapped on
    // CUDA builds). This both validates `kid` and frees the registry's
    // dashmap entry before we cross any await boundary, eliminating the
    // UAF window that existed when we kept a raw pointer derived from a
    // transient `dashmap::Ref` alive across the launch.
    let handle = registry.lookup(kid, owner)?;

    // Back-pressure: on the async path we await rather than reject so the
    // Wasm fiber suspends when the cap is reached. The permit is held for
    // the lifetime of this future — on return (success OR error) it drops
    // and the live-counter decrements, enforcing the cap regardless of
    // outcome.
    let bp = caller.data().wasi_cuda().back_pressure.clone();
    let _permit = bp.acquire().await;

    #[cfg(feature = "cuda")]
    {
        use cust::event::{Event, EventFlags};
        use cust::stream::{Stream, StreamFlags};

        // The strong `Arc` we already hold keeps the module alive across
        // launch + synchronize without any raw-pointer gymnastics.
        let module = handle.module.clone().ok_or_else(|| {
            caller
                .data()
                .wasi_cuda()
                .record_error("launch: kernel entry has no compiled module");
            AbiError::InvalidKernel
        })?;

        let func = module.get_function(&handle.entry).map_err(|e| {
            caller.data().wasi_cuda().record_error(format!(
                "launch: get_function({}) failed: {e:?}",
                handle.entry
            ));
            AbiError::LaunchFailed
        })?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: Stream::new failed: {e:?}"));
            AbiError::LaunchFailed
        })?;

        // v0.1.0 args marshalling: non-empty args were already rejected
        // above with KernelArgsUnsupported. The launch macro therefore
        // receives an empty parameter list, matching the zero-arg
        // kernel calling convention. Richer marshalling (per-arg
        // tagged blob, GPU-side pointer translation) lands in BAL-422
        // — see the TODO(v0.2) above and `docs/RISKS.md`.
        let _ = args_buf;

        // SAFETY: launching a kernel is inherently unsafe (no host-side
        // proof that the kernel arguments match the kernel signature).
        // We pass the pre-validated dims and a strong module reference;
        // zero arg parameters means a guest that loaded a non-zero-arg
        // PTX kernel will see an undefined-behavior launch from the
        // driver — those guests are expected to use the BAL-422
        // marshalling path once it ships. The host-side rejection above
        // means well-behaved guests cannot reach this point with args.
        unsafe {
            cust::launch!(
                func<<<(grid_x as u32, grid_y as u32, grid_z as u32),
                       (block_x as u32, block_y as u32, block_z as u32),
                       shared_mem as u32, stream>>>()
            )
            .map_err(|e| {
                caller
                    .data()
                    .wasi_cuda()
                    .record_error(format!("launch: cuLaunchKernel failed: {e:?}"));
                AbiError::LaunchFailed
            })?;
        }

        // Record an event on the stream so the dispatch future can poll
        // completion without holding a stream synchronize call open.
        let event = Event::new(EventFlags::DEFAULT).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: Event::new failed: {e:?}"));
            AbiError::LaunchFailed
        })?;
        event.record(&stream).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: event.record failed: {e:?}"));
            AbiError::LaunchFailed
        })?;

        // Move the stream + event into the blocking task so synchronize
        // doesn't block the wasmtime fiber. Stream + Event are Send. Keep
        // `handle` (and therefore the Arc<Module>) alive until after the
        // synchronize completes — that's why `handle` is moved into the
        // closure rather than `drop`ped explicitly here.
        let handle_for_keepalive = handle.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _keep_event = event;
            let _keep_module = handle_for_keepalive;
            stream.synchronize()
        })
        .await
        .map_err(|_| {
            // JoinError — internal scheduler issue.
            AbiError::Internal
        })?;
        result.map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: stream synchronize failed: {e:?}"));
            AbiError::LaunchFailed
        })?;
        // `handle` is still in scope here; it (and the Arc<Module>) is
        // released by Drop now that synchronize has returned. The clone
        // moved into the blocking task may still hold the Arc briefly,
        // which is fine: the module stays alive until *all* clones drop.
        drop(handle);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA we can't actually run the kernel; surface
        // `NotAvailable` so the Wasm caller knows.
        let _ = (args_buf, handle); // suppress unused warnings on no-CUDA.
        caller
            .data()
            .wasi_cuda()
            .record_error("launch: CUDA not available on this host");
        Err(AbiError::NotAvailable)
    }
}

fn sync_impl<T: HasWasiCuda>(_caller: &Caller<'_, T>) -> Result<(), AbiError> {
    let _span = info_span!(
        "wasi_cuda.sync",
        instance = %_caller.data().wasi_cuda().instance_id,
    )
    .entered();
    #[cfg(feature = "cuda")]
    {
        // Block on the current context's outstanding work. This is a
        // synchronous wasmtime function, so we can't await here; cust's
        // `CurrentContext::synchronize` is a blocking call that returns
        // once all queued work on the current context has finished.
        use cust::context::CurrentContext;
        match CurrentContext::synchronize() {
            Ok(()) => Ok(()),
            Err(e) => {
                _caller
                    .data()
                    .wasi_cuda()
                    .record_error(format!("sync: CurrentContext::synchronize failed: {e:?}"));
                Err(AbiError::LaunchFailed)
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        // No outstanding GPU work on the no-CUDA path; sync is trivially complete.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::FN_LAST_ERROR_PTR;

    struct Dummy(WasiCudaContext);
    impl HasWasiCuda for Dummy {
        fn wasi_cuda(&self) -> &WasiCudaContext {
            &self.0
        }
    }

    #[test]
    fn record_and_read_error() {
        let ctx = WasiCudaContext::new(InstanceId(42));
        ctx.record_error("oh no");
        assert_eq!(ctx.last_error().as_deref(), Some("oh no"));
    }

    #[test]
    fn add_to_linker_compiles() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");
    }

    /// Confirms `add_to_linker` does NOT register `FN_LAST_ERROR_PTR` — that
    /// symbol is kept in `abi.rs` for ABI-compat but the host deliberately
    /// does not expose it. We verify by attempting `Linker::get` for the
    /// (module, function) pair; the host registers every other wasi-cuda
    /// function and skipping this one is intentional. We also confirm a
    /// guest that imports `FN_LAST_ERROR_PTR` fails to instantiate.
    #[tokio::test]
    async fn add_to_linker_does_not_register_last_error_ptr() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");
        // Build a tiny WAT importing the not-registered symbol and assert
        // instantiation fails because the linker has no matching export.
        let wat = format!(
            r#"
            (module
              (import "{MODULE}" "{fn_name}" (func (result i32)))
            )
            "#,
            fn_name = FN_LAST_ERROR_PTR
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        let mut store = wasmtime::Store::new(
            &engine,
            Dummy(WasiCudaContext::new(InstanceId(101))),
        );
        let result = linker.instantiate_async(&mut store, &module).await;
        assert!(
            result.is_err(),
            "instantiation must fail because FN_LAST_ERROR_PTR is not registered"
        );
    }

    #[test]
    fn shared_back_pressure_constructor() {
        let bp = Arc::new(crate::async_dispatch::BackPressure::with_cap(8));
        let a = WasiCudaContext::with_back_pressure(InstanceId(1), bp.clone());
        let b = WasiCudaContext::with_back_pressure(InstanceId(2), bp.clone());
        assert_eq!(a.back_pressure().max_concurrent(), 8);
        assert_eq!(b.back_pressure().max_concurrent(), 8);
        // Confirm both contexts really share the same Arc<BackPressure>:
        assert!(Arc::ptr_eq(a.back_pressure(), b.back_pressure()));
    }

    /// A well-formed, in-bounds `args` buffer must surface as
    /// `KernelArgsUnsupported` (NOT `InvalidArgs`, NOT `InvalidPointer`).
    /// This is the v0.1.0 contract — see module docs and `docs/RISKS.md`.
    #[tokio::test]
    async fn launch_with_inbounds_args_returns_kernel_args_unsupported() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        // Tiny module: one page of memory (64 KiB), and an exported
        // `try_launch` that hands the host (kernel_id=0, 1x1x1 grid,
        // 1x1x1 block, shared_mem=0, args_ptr=0, args_len=4). The 4-byte
        // region at offset 0 is in-bounds, so the bounds-check must
        // pass and we must land on the unsupported branch.
        let wat = format!(
            r#"
            (module
              (import "{m}" "{fn_name}"
                (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "try_launch") (result i32)
                (call $launch
                  (i64.const 0)   ;; kernel_id
                  (i32.const 1) (i32.const 1) (i32.const 1)   ;; grid
                  (i32.const 1) (i32.const 1) (i32.const 1)   ;; block
                  (i32.const 0)   ;; shared_mem
                  (i32.const 0)   ;; args_ptr — inside the one-page region
                  (i32.const 4)   ;; args_len — 4 bytes, in-bounds
                ))
            )
            "#,
            m = MODULE,
            fn_name = FN_LAUNCH,
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        let mut store = wasmtime::Store::new(
            &engine,
            Dummy(WasiCudaContext::new(InstanceId(202))),
        );
        let instance = linker
            .instantiate_async(&mut store, &module)
            .await
            .expect("instantiate");
        let try_launch = instance
            .get_typed_func::<(), i32>(&mut store, "try_launch")
            .expect("typed func");
        let rc = try_launch.call_async(&mut store, ()).await.expect("call");
        assert_eq!(
            rc,
            AbiError::KernelArgsUnsupported.code(),
            "in-bounds args_len > 0 must return KernelArgsUnsupported, \
             not InvalidArgs ({}) or InvalidPointer ({})",
            AbiError::InvalidArgs.code(),
            AbiError::InvalidPointer.code(),
        );
        // Confirm the recorded error message points to the v0.1.0 contract.
        let last = store.data().wasi_cuda().last_error().unwrap_or_default();
        assert!(
            last.contains("dynamic kernel args not implemented"),
            "expected v0.1.0-contract message, got: {last}"
        );
    }

    /// An out-of-bounds args region must still return `InvalidPointer`
    /// — the bounds-check runs BEFORE the unsupported-args branch so a
    /// malicious guest cannot trade a `MemoryFault` (InvalidPointer) for
    /// the friendlier `KernelArgsUnsupported`.
    #[tokio::test]
    async fn launch_with_oob_args_returns_invalid_pointer() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        // One page = 65536 bytes. args_ptr=70000 is well past the end,
        // so even `args_len=4` (4-byte read) is out of bounds.
        let wat = format!(
            r#"
            (module
              (import "{m}" "{fn_name}"
                (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "try_launch") (result i32)
                (call $launch
                  (i64.const 0)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0)
                  (i32.const 70000)   ;; args_ptr — past end of single page
                  (i32.const 4)       ;; args_len — would overshoot
                ))
            )
            "#,
            m = MODULE,
            fn_name = FN_LAUNCH,
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        let mut store = wasmtime::Store::new(
            &engine,
            Dummy(WasiCudaContext::new(InstanceId(203))),
        );
        let instance = linker
            .instantiate_async(&mut store, &module)
            .await
            .expect("instantiate");
        let try_launch = instance
            .get_typed_func::<(), i32>(&mut store, "try_launch")
            .expect("typed func");
        let rc = try_launch.call_async(&mut store, ()).await.expect("call");
        assert_eq!(
            rc,
            AbiError::InvalidPointer.code(),
            "OOB args region must return InvalidPointer (memory fault), \
             not KernelArgsUnsupported — the bounds-check must run first"
        );
    }

    /// `KernelRegistry::lookup` returns a `KernelHandle` whose lifetime is
    /// independent of the underlying dashmap entry — `remove` does not
    /// invalidate a previously returned handle. This is the property that
    /// fixes the original UAF.
    #[test]
    fn registry_lookup_handle_outlives_remove() {
        let reg = KernelRegistry::new();
        let id = reg
            .register(KernelEntry {
                owner: InstanceId(7),
                entry: "k".into(),
                ptx_bytes_len: 16,
                #[cfg(feature = "cuda")]
                module: None,
            })
            .unwrap();
        let handle = reg.lookup(id, InstanceId(7)).unwrap();
        assert!(reg.remove(id).is_some());
        // handle still readable; no UAF possible.
        assert_eq!(handle.entry, "k");
    }
}
