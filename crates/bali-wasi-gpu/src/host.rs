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
pub fn add_to_linker<T: HasWasiCuda + 'static>(linker: &mut Linker<T>) -> wasmtime::Result<()> {
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

    linker.func_wrap(
        MODULE,
        FN_LAUNCH,
        |mut caller: Caller<'_, T>,
         kernel_id: i64,
         grid_x: i32,
         grid_y: i32,
         grid_z: i32,
         block_x: i32,
         block_y: i32,
         block_z: i32,
         shared_mem: i32,
         args_ptr: i32,
         args_len: i32|
         -> i32 {
            launch_impl(
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
            .map_or_else(|e| e.code(), |_| 0)
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
        // Real path: compile the PTX through cust::module::Module::from_ptx.
        // Concrete implementation lands in the CUDA-enabled session.
        let _ = (ptx, entry);
        caller
            .data()
            .wasi_cuda()
            .record_error("load_ptx: cuda feature implementation pending");
        Err(AbiError::Internal)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_impl<T: HasWasiCuda>(
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
    let _span = info_span!(
        "wasi_cuda.launch",
        instance = %caller.data().wasi_cuda().instance_id,
        kernel = kernel_id,
        grid_x = grid_x, grid_y = grid_y, grid_z = grid_z,
        block_x = block_x, block_y = block_y, block_z = block_z,
        shared_mem = shared_mem,
    )
    .entered();
    // Validate args region bounds up-front. We don't consume the bytes yet
    // (kernel argument marshalling lands with the real CUDA session) but
    // the (ptr, len) pair MUST refer to a region inside the caller's Wasm
    // memory — otherwise we'd be propagating a bogus pointer downstream.
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
    let kid = KernelId(kernel_id as u64);
    let owner = caller.data().wasi_cuda().instance_id;
    let registry = caller.data().wasi_cuda().registry.clone();
    let _entry = registry.lookup(kid, owner)?;
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

    // Back-pressure: acquire one slot for this launch. The permit is held
    // for the lifetime of this call — on return (success OR error) it drops
    // and the live-counter decrements, so the cap is enforced regardless of
    // outcome. We use the non-awaiting `try_acquire` because this is the
    // synchronous launch stub; over the cap we surface `QuotaExceeded`
    // immediately so the Wasm caller can back off.
    //
    // Note: the plan calls for the CUDA-enabled session to swap the launch
    // host function over to `linker.func_wrap_async`, at which point the
    // Wasm fiber will await `BackPressure::acquire(&self)` and suspend
    // (rather than reject with QuotaExceeded) when the cap is reached.
    let _permit = caller
        .data()
        .wasi_cuda()
        .back_pressure
        .try_acquire()
        .ok_or_else(|| {
            caller
                .data()
                .wasi_cuda()
                .record_error("launch: max concurrent GPU ops reached");
            AbiError::QuotaExceeded
        })?;

    #[cfg(feature = "cuda")]
    {
        // Real CUDA launch lands here.
        caller
            .data()
            .wasi_cuda()
            .record_error("launch: cuda feature implementation pending");
        Err(AbiError::Internal)
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
        Err(AbiError::Internal)
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
