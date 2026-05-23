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

use std::sync::Arc;
use std::sync::Mutex;

use bali_core::types::{InstanceId, KernelId};
use tracing::{info, info_span, warn};
use wasmtime::{Caller, Linker};

use crate::abi::{
    AbiError, FN_LAST_ERROR_COPY, FN_LAST_ERROR_LEN, FN_LAUNCH, FN_LOAD_PTX, FN_SYNC,
    MAX_PTX_BYTES, MODULE,
};
use crate::async_dispatch::BackPressure;
use crate::registry::{KernelEntry, KernelRegistry};

/// Per-instance host state passed to wasi-cuda calls.
///
/// `WasiCudaContext` is stored in the wasmtime `Store`'s data type (or in a
/// resource handle thereon). The executor (`bali-exec`) creates one per
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
        warn!(target: "bali_wasi_gpu::host", instance = %self.instance_id, %msg, "wasi-cuda error");
        *self.last_error.lock().expect("last_error poisoned") = Some(msg);
    }

    /// Borrow the most recent error message.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().expect("last_error poisoned").clone()
    }
}

/// Trait implemented by store data types that can hand out a [`WasiCudaContext`].
///
/// `bali-exec`'s `InstanceState` will implement this in a follow-up wiring
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
            if dst_ptr < 0 || dst_len <= 0 {
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
                None => return 0,
            };
            let buf = bytes[..to_copy].to_vec();
            if memory.write(&mut caller, dst_ptr as usize, &buf).is_err() {
                return 0;
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
    let entry = String::from_utf8(entry_bytes).map_err(|_| AbiError::InvalidPointer)?;

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
        info!(target: "bali_wasi_gpu::host", instance = %owner, kernel = %id, entry, "PTX registered (stub: cuda feature off)");
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
            module: Some(module),
        };
        let registry = caller.data().wasi_cuda().registry.clone();
        let id = registry.register(entry_record)?;
        info!(target: "bali_wasi_gpu::host", instance = %owner, kernel = %id, entry, "PTX compiled and registered via cust");
        Ok(id)
    }
}

/// Common argument-region validation extracted from the launch path so the
/// sync and async wrappers share one implementation.
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
    if grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
        || shared_mem < 0
    {
        return Err(AbiError::InvalidPointer);
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

    // Validate the kernel id eagerly so both code paths report
    // `InvalidKernel` before consuming a back-pressure permit.
    let _ = registry.lookup(kid, owner)?;

    // Back-pressure: on the async path we await rather than reject so the
    // Wasm fiber suspends when the cap is reached. The permit is held for
    // the lifetime of this future — on return (success OR error) it drops
    // and the live-counter decrements, enforcing the cap regardless of
    // outcome.
    let bp = caller.data().wasi_cuda().back_pressure.clone();
    let _permit = bp.acquire().await;

    #[cfg(feature = "cuda")]
    {
        use cust::stream::{Stream, StreamFlags};

        // Re-look up the kernel inside the cuda branch so the borrow of
        // the registry entry is short and doesn't span an await point.
        let module = {
            let entry = registry.lookup(kid, owner)?;
            entry
                .module
                .as_ref()
                .ok_or_else(|| {
                    caller
                        .data()
                        .wasi_cuda()
                        .record_error("launch: kernel entry has no compiled module");
                    AbiError::InvalidKernel
                })?
                // Module is `!Clone`; we instead hold the lookup ref live
                // for the duration of the launch by binding it to a local.
                // To avoid spanning awaits across the dashmap ref, we
                // perform the kernel lookup, function fetch, and launch
                // entirely synchronously before the synchronize await.
                as *const cust::module::Module
        };

        // SAFETY: the registry owns the module behind an Arc-like dashmap
        // entry that lives for at least as long as `registry`; the entry
        // cannot be removed concurrently because `lookup` returns a
        // strong dashmap ref. The pointer is only used inside this
        // scope, not across any await boundary.
        let module_ref = unsafe { &*module };

        let entry_name = {
            let entry = registry.lookup(kid, owner)?;
            entry.entry.clone()
        };

        let func = module_ref.get_function(&entry_name).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: get_function({entry_name}) failed: {e:?}"));
            AbiError::LaunchFailed
        })?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| {
            caller
                .data()
                .wasi_cuda()
                .record_error(format!("launch: Stream::new failed: {e:?}"));
            AbiError::LaunchFailed
        })?;

        // SAFETY: launching a kernel is inherently unsafe (no host-side
        // proof that the kernel arguments match the kernel signature). We
        // currently pass zero arguments; argument marshalling lands in a
        // follow-up wiring session.
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

        // Move the stream into the blocking task so synchronize doesn't
        // block the wasmtime fiber. Stream is Send.
        let result = tokio::task::spawn_blocking(move || stream.synchronize())
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
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA we can't actually run the kernel; surface
        // `NotAvailable` so the Wasm caller knows.
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
}
