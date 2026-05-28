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
//! ## Kernel argument marshalling (v0.2)
//!
//! The launch host function takes `(args_ptr, args_len)` describing a
//! byte buffer in the guest's linear memory. The buffer carries a
//! sequence of tagged records — see [`crate::kernel_args`] for the wire
//! format. The launch path bounds-checks the buffer against the caller's
//! linear memory, then [`crate::kernel_args::parse_argv`] turns it into a
//! `Vec<LoweredArg>` where pointer arguments have been resolved into raw
//! host pointers (under CUDA Unified Memory those are also valid device
//! addresses).
//!
//! On CUDA builds the parsed args flow into
//! [`crate::kernel_args::build_kernel_param_storage`] and then into a
//! direct `cuLaunchKernel` call — bypassing `cust::launch!` (which would
//! force statically-typed parameters at the call site). On no-CUDA
//! builds the parsed args are recorded on the [`WasiCudaContext`] for
//! testing (see [`WasiCudaContext::last_lowered_args`]) and the launch
//! returns [`AbiError::NotAvailable`].
//!
//! [`AbiError::KernelArgsUnsupported`] is reserved for sanity-cap busts
//! on otherwise well-formed argv — buffers longer than
//! [`crate::kernel_args::MAX_KERNEL_ARGS_BYTES`] (4 KiB) or carrying
//! more than [`crate::kernel_args::MAX_KERNEL_ARGS`] (128) tagged
//! records. The W1.1 typed-argv lane (live since v0.2.0 of the WIT)
//! lowers any scalar + pointer argv below those caps into a
//! `cuLaunchKernel` parameter array. Malformed argv (unknown tag
//! bytes, truncated records) surfaces as [`AbiError::InvalidArgs`];
//! out-of-bounds pointer arguments surface as
//! [`AbiError::InvalidPointer`]. The distinction keeps the error
//! story crisp for guest debugging.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use tensor_wasm_core::types::{InstanceId, KernelId};
use tracing::{info, info_span, warn, Instrument};
use wasmtime::{Caller, Linker};

use crate::abi::{
    AbiError, FN_LAST_ERROR_COPY, FN_LAST_ERROR_LEN, FN_LAUNCH, FN_LOAD_PTX, FN_SYNC,
    MAX_BLOCK_DIM, MAX_GRID_DIM, MAX_PTX_BYTES, MAX_THREADS_PER_BLOCK, MODULE,
};
use crate::async_dispatch::BackPressure;
use crate::kernel_args::{parse_argv, LoweredArg, LoweredArgSnapshot};
use crate::registry::{KernelEntry, KernelRegistry};

/// Maximum byte length of a single recorded `last_error` message.
///
/// `record_error` truncates any message above this cap (preserving UTF-8
/// boundaries and appending an ellipsis) before storing it. The cap defends
/// against a guest looping `launch` with malformed input — each call would
/// otherwise force a large `format!` allocation that is immediately
/// discarded on the next call.
pub const MAX_RECORDED_ERROR_BYTES: usize = 512;

/// Maximum byte length of a kernel entry-name passed to `load_ptx`.
///
/// CUDA identifiers are far below this in practice (PTX entries are
/// C-style identifiers, typically under 64 bytes). The cap prevents a
/// guest from forcing a multi-MiB UTF-8 validation + `String::from`
/// allocation per `load_ptx` call. The PTX-bytes side is already
/// bounded by [`MAX_PTX_BYTES`]; this is the matching cap for the
/// entry-name side.
pub const MAX_ENTRY_NAME_BYTES: usize = 256;

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
    /// The most recent successfully-parsed kernel argv from a `launch`
    /// call. On the no-CUDA host-stub path this is the only place the
    /// lowered args land — integration tests inspect it to confirm the
    /// argv made it through bounds-checking and type-tag parsing
    /// without actually launching a kernel. On CUDA builds it is also
    /// populated (after the launch returns) so the same observability
    /// works under `--features cuda`.
    ///
    /// HAZARD: pointer args inside [`LoweredArg::Ptr`] carry raw host
    /// pointers into the guest's linear memory. Those pointers are
    /// invalidated on any subsequent `memory.grow` by the same guest.
    /// Treat this field as observation-only and snapshot it
    /// immediately after the launch returns; do NOT cache the
    /// pointers across guest-callable boundaries.
    pub last_lowered_args: Mutex<Vec<LoweredArg>>,
    /// Capability flag controlling whether the wasi-cuda host functions
    /// linked via [`add_to_linker`] are allowed to perform real work on
    /// this instance.
    ///
    /// Defaults to `false`. The embedder must call
    /// [`WasiCudaContext::enable_wasi_cuda`] (or pre-set the field
    /// directly) before the guest invokes any wasi-cuda host function.
    /// Every host function bodies wired by `add_to_linker` short-circuits
    /// with [`AbiError::NotAvailable`] when this is `false` — including
    /// `last_error_len` / `last_error_copy`, so a guest cannot fingerprint
    /// the host's wasi-cuda capability indirectly through the error
    /// surface.
    ///
    /// Rationale: linking the wasi-cuda host module historically gave
    /// every guest that imported it full driver access. Capability gating
    /// follows the broader WASI design ("imports without capability are
    /// inert") and lets the executor link wasi-cuda once at engine setup
    /// while still admitting per-instance policy decisions.
    ///
    /// Stored as an `AtomicBool` and gated behind `pub(crate)` (wasi-gpu 1.3)
    /// so that an embedder cannot bypass [`Self::enable_wasi_cuda`] by
    /// writing to the field directly, and so reads from any host-import
    /// closure observe a consistent value even if the embedder ever shared
    /// the context across threads. Use [`Self::wasi_cuda_enabled`] /
    /// [`Self::enable_wasi_cuda`] / [`Self::disable_wasi_cuda`].
    pub(crate) wasi_cuda_enabled: AtomicBool,
    /// Per-invocation absolute deadline (T36 — cooperative deadlines).
    ///
    /// When `Some`, the launch path constructs a deadline-aware
    /// [`BackPressure`] clone via
    /// [`BackPressure::with_deadline_hint`] so the acquire decision
    /// agrees with the cooperative-yield verdict the guest sees from
    /// `wasi:scheduler/host`. Lives behind a `Mutex` so the executor
    /// can re-arm it at the top of each `call_export` without holding
    /// an exclusive borrow on the context (host functions only
    /// observe it through a borrow of `&self`).
    ///
    /// `None` means "no deadline configured" — the launch path falls
    /// back to the historical `acquire_borrowed` behaviour and host
    /// functions never reject on deadline grounds.
    pub bp_deadline: Mutex<Option<Instant>>,
}

impl WasiCudaContext {
    /// Construct a fresh context for the given instance with a dedicated
    /// (un-shared) back-pressure cap.
    ///
    /// The wasi-cuda capability defaults to **disabled**; the embedder
    /// must call [`WasiCudaContext::enable_wasi_cuda`] before the guest
    /// can use any wasi-cuda host function.
    pub fn new(instance_id: InstanceId) -> Self {
        Self {
            instance_id,
            registry: Arc::new(KernelRegistry::new()),
            last_error: Mutex::new(None),
            back_pressure: Arc::new(BackPressure::new()),
            last_lowered_args: Mutex::new(Vec::new()),
            wasi_cuda_enabled: AtomicBool::new(false),
            bp_deadline: Mutex::new(None),
        }
    }

    /// Construct a context that shares the given [`BackPressure`] cap with
    /// other contexts. Used by the executor to enforce one process-wide
    /// concurrency limit across all Wasm instances.
    ///
    /// The wasi-cuda capability defaults to **disabled**, mirroring
    /// [`WasiCudaContext::new`].
    pub fn with_back_pressure(instance_id: InstanceId, bp: Arc<BackPressure>) -> Self {
        Self {
            instance_id,
            registry: Arc::new(KernelRegistry::new()),
            last_error: Mutex::new(None),
            back_pressure: bp,
            last_lowered_args: Mutex::new(Vec::new()),
            wasi_cuda_enabled: AtomicBool::new(false),
            bp_deadline: Mutex::new(None),
        }
    }

    /// Borrow the shared back-pressure handle for observability / sharing.
    pub fn back_pressure(&self) -> &Arc<BackPressure> {
        &self.back_pressure
    }

    /// Install a per-invocation absolute deadline that drives the
    /// back-pressure rejection path (T36). The same `Instant` SHOULD
    /// be installed on the matching
    /// [`crate::scheduler::SchedulerContext`] via
    /// [`crate::scheduler::SchedulerContext::set_bp_deadline_instant`]
    /// so the guest's cooperative-yield verdicts agree with the
    /// acquire-side decisions.
    ///
    /// Passing `None` clears the deadline; subsequent launches fall
    /// back to the historical `acquire_borrowed` behaviour.
    pub fn set_bp_deadline(&self, deadline: Option<Instant>) {
        // Recover from a poisoned mutex rather than panicking — a
        // previous panic during a launch should not brick the
        // deadline-update path.
        let mut guard = self
            .bp_deadline
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *guard = deadline;
    }

    /// Read the currently-installed back-pressure deadline. Returns
    /// `None` when no deadline is configured.
    pub fn bp_deadline(&self) -> Option<Instant> {
        *self
            .bp_deadline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Build a deadline-aware [`BackPressure`] clone suitable for the
    /// hot launch path. The returned value carries the per-instance
    /// deadline installed via [`Self::set_bp_deadline`] (if any) but
    /// shares the underlying semaphore Arc with every other clone
    /// pulling from the same pool — so concurrency caps remain
    /// process-wide while deadlines remain per-instance.
    pub fn deadline_aware_back_pressure(&self) -> BackPressure {
        let bp = (*self.back_pressure).clone();
        bp.with_deadline_hint(self.bp_deadline())
    }

    /// Grant this context the wasi-cuda capability. Without this call the
    /// linked host functions return [`AbiError::NotAvailable`] regardless
    /// of host CUDA support — see [`WasiCudaContext::wasi_cuda_enabled`].
    pub fn enable_wasi_cuda(&mut self) {
        self.wasi_cuda_enabled.store(true, Ordering::Release);
    }

    /// Revoke the wasi-cuda capability granted by
    /// [`WasiCudaContext::enable_wasi_cuda`]. Subsequent host calls degrade
    /// to [`AbiError::NotAvailable`]. Also clears any previously-recorded
    /// `last_error` so flipping the capability cannot let a guest read
    /// state recorded while the capability was disabled (wasi-gpu 1.5
    /// follow-up).
    pub fn disable_wasi_cuda(&mut self) {
        self.wasi_cuda_enabled.store(false, Ordering::Release);
        if let Ok(mut guard) = self.last_error.lock() {
            *guard = None;
        }
    }

    /// `true` when [`WasiCudaContext::enable_wasi_cuda`] has been called.
    pub fn wasi_cuda_enabled(&self) -> bool {
        self.wasi_cuda_enabled.load(Ordering::Acquire)
    }

    fn record_error(&self, msg: impl Into<String>) {
        let mut msg = msg.into();
        // Cap the recorded message so a guest looping `launch` with
        // malformed input cannot keep forcing large `format!` allocations
        // that are immediately discarded on the next call. We must
        // truncate on a UTF-8 boundary — `String::truncate` panics
        // otherwise — so walk back from the cap to the largest valid
        // boundary index. `is_char_boundary(0)` is always true, so the
        // `unwrap_or(0)` branch is unreachable in practice but keeps the
        // expression total.
        if msg.len() > MAX_RECORDED_ERROR_BYTES {
            let cutoff = (0..=MAX_RECORDED_ERROR_BYTES)
                .rev()
                .find(|i| msg.is_char_boundary(*i))
                .unwrap_or(0);
            msg.truncate(cutoff);
            msg.push('\u{2026}');
        }
        warn!(target: "tensor_wasm_wasi_gpu::host", instance = %self.instance_id, %msg, "wasi-cuda error");
        // A panicked `record_error` call earlier in the launch path would
        // have poisoned this mutex. The error payload is still valid and
        // we'd rather overwrite it with the current call's message than
        // brick the rest of the instance — recover the inner String slot.
        *self
            .last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(msg);
    }

    /// Test-only accessor for the truncating `record_error` path.
    ///
    /// Exposed so integration tests in `tests/` (which cannot reach the
    /// private `record_error`) can exercise the cap. Production code
    /// outside this crate has no reason to inject error messages and
    /// should not call this method.
    #[doc(hidden)]
    pub fn record_error_for_test(&self, msg: impl Into<String>) {
        self.record_error(msg);
    }

    /// Borrow the most recent error message.
    pub fn last_error(&self) -> Option<String> {
        // Mirror `record_error`: recover from poisoning so a single
        // panicked call doesn't make subsequent observability queries
        // panic too.
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Pointer-free snapshot suitable for observability and tests; the
    /// host pointer is intentionally redacted to defend against
    /// use-after-grow.
    ///
    /// The internal [`LoweredArg::Ptr`] variant carries a raw host
    /// pointer into the guest's linear memory that the launch path
    /// consumes synchronously. Any subsequent `memory.grow` by the same
    /// guest can dangle that pointer; surfacing it to an embedder would
    /// hand them a use-after-grow primitive whose lifetime no Rust
    /// borrow check can express. Returning [`LoweredArgSnapshot`]
    /// strips the raw pointer at the public boundary so embedders and
    /// tests can still inspect the parsed-arg shape (variant,
    /// guest-declared length, guest offset) without that hazard.
    ///
    /// Intended for tests and diagnostics; production code should not
    /// depend on this value remaining stable across launches on the
    /// same context.
    pub fn last_lowered_args(&self) -> Vec<LoweredArgSnapshot> {
        self.last_lowered_args
            .lock()
            .expect("last_lowered_args poisoned")
            .iter()
            .map(LoweredArgSnapshot::from)
            .collect()
    }

    /// Crate-internal variant of [`Self::last_lowered_args`] that keeps
    /// the raw [`LoweredArg`] payload (including the host pointer
    /// inside `Ptr` variants).
    ///
    /// This is the shape the launch path itself needs — the host
    /// pointer is what eventually feeds `cuLaunchKernel`. It is
    /// deliberately not part of the public API: see
    /// [`Self::last_lowered_args`] for the use-after-grow rationale
    /// behind the public redaction.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn last_lowered_args_internal(&self) -> Vec<LoweredArg> {
        self.last_lowered_args
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
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
    // Capability gating: every host fn below first checks
    // `wasi_cuda_enabled` on the per-instance context. Guests whose
    // executor has not granted the capability see [`AbiError::NotAvailable`]
    // — indistinguishable from "this host lacks a GPU", which is the
    // desired posture (the guest cannot fingerprint whether the host is
    // capability-gating or genuinely CUDA-less). The check happens before
    // any other validation so even malformed inputs cannot be used to
    // probe state through error-discrimination side channels.
    linker.func_wrap(
        MODULE,
        FN_LOAD_PTX,
        |mut caller: Caller<'_, T>,
         ptx_ptr: i32,
         ptx_len: i32,
         entry_ptr: i32,
         entry_len: i32|
         -> i64 {
            if !caller.data().wasi_cuda().wasi_cuda_enabled.load(Ordering::Acquire) {
                // wasi-gpu 1.5: do NOT record_error on the disabled-capability
                // path. Matching the FN_LAST_ERROR_* gate, a recorded
                // message would (a) burn allocations + mutex traffic for a
                // hostile guest that hammers disabled calls, and (b)
                // become readable if the embedder ever flips the capability
                // back on. The NotAvailable code is the signal callers get.
                return AbiError::NotAvailable.code() as i64;
            }
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
                if !caller.data().wasi_cuda().wasi_cuda_enabled.load(Ordering::Acquire) {
                    // wasi-gpu 1.5: see the load_ptx branch above for why
                    // we skip record_error here.
                    return AbiError::NotAvailable.code();
                }
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
        if !caller.data().wasi_cuda().wasi_cuda_enabled.load(Ordering::Acquire) {
            // wasi-gpu 1.5: see the load_ptx branch above for the rationale.
            return AbiError::NotAvailable.code();
        }
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
        if !caller.data().wasi_cuda().wasi_cuda_enabled.load(Ordering::Acquire) {
            // Note: we do NOT call `record_error` here — the guest could
            // read that recorded message back via this same surface,
            // turning the gate into a leak channel. Returning the
            // negative `NotAvailable` code is unambiguous: a positive
            // `n > 0` is a real length, `0` means "no error" on a
            // gate-passing context, and a negative value means "the
            // surface is unavailable on this instance."
            return AbiError::NotAvailable.code();
        }
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
            if !caller.data().wasi_cuda().wasi_cuda_enabled.load(Ordering::Acquire) {
                // See the matching note on FN_LAST_ERROR_LEN: keep the
                // failure shape distinct from "no error" without recording
                // anything the guest could subsequently observe.
                return AbiError::NotAvailable.code();
            }
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
    // Bound the entry-name length BEFORE `read_bytes` so a guest cannot
    // force a multi-MiB UTF-8 validation + `String::from` allocation per
    // call. `entry_len` is i32 from the wire; the negative-check inside
    // `read_bytes` would still catch a negative value later, but checking
    // the positive overflow here lets us reject without ever copying out
    // of linear memory. We surface `QuotaExceeded` to match the existing
    // PTX-bytes cap above — both are "input too large" failures from the
    // guest's POV.
    if entry_len < 0 || (entry_len as usize) > MAX_ENTRY_NAME_BYTES {
        caller.data().wasi_cuda().record_error(format!(
            "load_ptx: entry_len {entry_len} exceeds MAX_ENTRY_NAME_BYTES {MAX_ENTRY_NAME_BYTES}"
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
/// ## Pointer-aliasing safety across the back-pressure await
///
/// Wasmtime's `Memory::data` borrow may be invalidated by *any* await on
/// the same store — including `memory.grow` triggered by an embedder
/// host hook. To keep raw pointers resolved by [`parse_argv`] from
/// becoming dangling at the `cuLaunchKernel` call, we structure the
/// body as:
///
///  1. Synchronously validate dims + the outer args region.
///  2. `await bp.acquire_borrowed()` — back-pressure permit, the only
///     await on the path.
///  3. After the await resolves, synchronously snapshot the args buffer,
///     run `parse_argv` (which resolves guest offsets to host pointers),
///     stash the lowered args for observability, look up the kernel
///     handle, and call `cuLaunchKernel`.
///
/// Step 3 cannot cross another await (the host function holds the
/// wasmtime store for its full duration after the permit resolves), so
/// the resolved host pointers remain valid through the
/// `cuLaunchKernel` call. The subsequent `spawn_blocking`
/// `stream.synchronize()` is also safe: `cuLaunchKernel` has already
/// captured the pointers, and the guest cannot run (let alone grow
/// memory) until this async fn returns.
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
    // Build the launch span up front and instrument the inner future with
    // it. `info_span!` returns a `Span` whose `.enter()` guard is `!Send`
    // — entering it directly here would poison the `Send` bound the
    // `func_wrap_async` boxed future carries. Wrapping the inner future
    // via `tracing::Instrument` attaches the span across `await` points
    // instead, so each poll re-enters the span and our log lines stay
    // attributed to this launch.
    let launch_span = info_span!(
        "wasi_cuda.launch",
        instance = %caller.data().wasi_cuda().instance_id,
        kernel = kernel_id,
        grid_x = grid_x, grid_y = grid_y, grid_z = grid_z,
        block_x = block_x, block_y = block_y, block_z = block_z,
        shared_mem = shared_mem,
    );
    launch_impl_async_inner(
        caller, kernel_id, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, args_ptr,
        args_len,
    )
    .instrument(launch_span)
    .await
}

#[allow(clippy::too_many_arguments)]
async fn launch_impl_async_inner<T: HasWasiCuda>(
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
    let kid = validate_launch_args(
        caller, kernel_id, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, args_ptr,
        args_len,
    )?;
    let owner = caller.data().wasi_cuda().instance_id;
    let registry = caller.data().wasi_cuda().registry.clone();

    // Back-pressure: on the async path we await rather than reject so the
    // Wasm fiber suspends when the cap is reached. The permit is held for
    // the lifetime of this future — on return (success OR error) it drops
    // and the live-counter decrements, enforcing the cap regardless of
    // outcome.
    //
    // CRITICAL ORDERING: the permit is acquired *before* we resolve any
    // pointers into guest linear memory. `parse_argv` calls
    // `mem.as_ptr().add(start)` on a `Memory::data(&caller)` borrow whose
    // pointers wasmtime is allowed to invalidate across any await on this
    // store. By awaiting the permit first and only then snapshotting +
    // parsing + dispatching to `cuLaunchKernel`, we never let a resolved
    // host pointer outlive the synchronous critical section that consumes
    // it. The remaining `spawn_blocking(stream.synchronize())` is safe
    // because `cuLaunchKernel` has already captured the pointers and the
    // guest cannot run until this fn returns.
    //
    // We use `acquire_borrowed` (not `acquire`) because this host function
    // never moves the permit across a `tokio::spawn` boundary: the
    // `spawn_blocking` call below moves only the CUDA stream/event/module
    // handle, not the permit. Borrowing skips the `Arc<Semaphore>` clone
    // the owned variant pays on every dispatch — a measurable saving on
    // the hot path. `&BackPressure` outlives the borrow because the
    // `WasiCudaContext` (which owns the Arc<BackPressure>) is held by the
    // wasmtime `Caller` for the duration of this async fn.
    // `acquire_borrowed` returns `Err(QuotaExceeded)` synchronously when
    // the cap is the cap-0 sentinel (no permits will ever be issued), so a
    // guest authored against a back-pressure-disabled embedder surfaces
    // the saturation error rather than hanging the wasm fiber forever.
    // Any other cap behaves as before: the await suspends until a permit
    // is released by a finishing dispatch.
    // T36: build a deadline-aware BackPressure clone so the acquire
    // path can refuse new permits when the per-invocation deadline is
    // near or elapsed. The underlying semaphore Arc is shared across
    // every per-instance clone, so cap enforcement remains
    // process-wide; only the deadline is per-instance. Without an
    // installed deadline this collapses to the pre-T36 behaviour.
    let bp = caller.data().wasi_cuda().deadline_aware_back_pressure();
    let _permit = bp.acquire_borrowed().await?;

    // Resolve argv now, after the permit has been acquired. Pointer args
    // are resolved against the caller's current linear-memory snapshot;
    // the resolution and the consuming `cuLaunchKernel` call live in this
    // same synchronous critical section, so the wasmtime guest cannot
    // run between them and the resolved pointers are guaranteed valid at
    // launch time. Once `cuLaunchKernel` returns, CUDA has its own copy
    // of the parameter slot bytes and we never re-read the resolved
    // pointers from this side.
    //
    // `KernelArgsUnsupported` is preserved as a fallback for buffers that
    // exceed the kernel-args sanity caps — see
    // [`crate::kernel_args::MAX_KERNEL_ARGS_BYTES`] /
    // [`crate::kernel_args::MAX_KERNEL_ARGS`]. Genuinely-malformed argv
    // (unknown tag, truncated record) surfaces as `InvalidArgs`; OOB
    // pointer arg returns `InvalidPointer`. The bounds-check on the
    // outer buffer still runs first inside `validate_launch_args`, so a
    // malicious guest cannot trade a `MemoryFault` for the friendlier
    // tag-byte error.
    let lowered_args: Vec<LoweredArg> = if args_len > 0 {
        // PERF (T23): skip the `read_bytes` Vec copy of the argv buffer.
        // `parse_argv` already takes both inputs as `&[u8]`, so we can
        // pass it slices directly into the caller's linear memory.
        // `validate_launch_args` above has already verified that
        // `args_ptr >= 0`, `args_len >= 0`, and `[args_ptr, args_ptr +
        // args_len) ⊆ memory`, so the slicing below cannot panic. A
        // single `mem.data(&caller)` borrow covers both the args region
        // and the whole-memory bounds-check that pointer args inside
        // `parse_argv` need.
        let mem = caller
            .get_export("memory")
            .and_then(|e| e.into_memory())
            .ok_or(AbiError::InvalidPointer)?;
        let mem_data = mem.data(&caller);
        let start = args_ptr as usize;
        let end = start + args_len as usize;
        let argv_slice = &mem_data[start..end];
        match parse_argv(argv_slice, mem_data) {
            Ok(v) => v,
            Err(e) => {
                caller.data().wasi_cuda().record_error(format!(
                    "launch: kernel argv parse failed ({}); args_len={args_len}",
                    e.name()
                ));
                return Err(e);
            }
        }
    } else {
        Vec::new()
    };

    // Stash the parsed argv for observability BEFORE the kernel-handle
    // lookup. Tests inspect `last_lowered_args` to confirm the
    // marshalling round-trip held; surfacing it only after the launch
    // synchronizes (or after the CUDA branch's many error paths) means
    // a missing-PTX or stream-failure case loses the parse signal,
    // which is the more valuable data point for diagnostics. The CUDA
    // and no-CUDA branches further down both overwrite this slot on
    // their own happy path, so the duplication is intentional.
    *caller
        .data()
        .wasi_cuda()
        .last_lowered_args
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = lowered_args.clone();

    // Eagerly take a strong, owned handle to the kernel (Arc-wrapped on
    // CUDA builds). This both validates `kid` and frees the registry's
    // dashmap entry before any further work, eliminating the UAF window
    // that existed when we kept a raw pointer derived from a transient
    // `dashmap::Ref` alive across the launch.
    let handle = registry.lookup(kid, owner)?;

    #[cfg(feature = "cuda")]
    {
        use cust::event::{Event, EventFlags};
        use cust::stream::{Stream, StreamFlags};

        use crate::kernel_args::build_kernel_param_storage;

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

        // Build the `void**` parameter storage from the parsed argv. The
        // storage owns the per-arg value bytes (scalars) and the
        // pointer-of-pointer slots that `cuLaunchKernel` consumes; we
        // keep it alive across the launch call below.
        //
        // For zero-arg launches `storage.as_ptr()` is still a valid
        // pointer to an empty slot vec — `cuLaunchKernel` interprets a
        // zero parameter count as "ignore the argv pointer."
        let mut storage = build_kernel_param_storage(&lowered_args);
        let param_count = storage.len();
        // CUDA reads `kernelParams` only when the kernel's PTX `.param`
        // block is non-empty; for zero-arg kernels we pass NULL rather
        // than the (possibly dangling) `Vec::as_mut_ptr()` of an empty
        // slot vec.
        let kernel_params_ptr: *mut *mut std::ffi::c_void = if param_count == 0 {
            std::ptr::null_mut()
        } else {
            storage.as_ptr()
        };

        // Drop down to `cust::sys::cuLaunchKernel` because `cust::launch!`
        // forces statically-typed args at the call site. The raw call
        // takes a `*mut *mut c_void` of length `param_count` — exactly
        // what `storage.as_ptr()` provides. We pass a null `extra`
        // pointer because CUDA accepts either form (params XOR extra).
        //
        // NOTE: `cust 0.3`'s `Function` and `Stream` raw-handle
        // accessors (`as_raw` / `as_inner`) are stable across the 0.3.x
        // line; if a future cust bump renames them this is the only
        // call site that needs to follow.
        //
        // SAFETY: launching a kernel is inherently unsafe — the host has
        // no proof the kernel signature matches the parsed argv. The
        // caller (the Wasm guest) is responsible for that match; we
        // guarantee only that (a) every pointer arg points into the
        // guest's own linear memory, (b) the dims fit the CUDA caps,
        // and (c) the stream/function/module references are live for
        // the duration of the call (the `Arc<Module>` clone held in
        // `handle` keeps the module alive across the launch).
        use cust::sys as cuda_sys;
        let launch_status = unsafe {
            cuda_sys::cuLaunchKernel(
                // cust 0.3.2 exposes the raw CUfunction handle via `to_raw()`
                // (NOT `as_raw()`); the spike originally guessed wrong.
                func.to_raw(),
                grid_x as u32,
                grid_y as u32,
                grid_z as u32,
                block_x as u32,
                block_y as u32,
                block_z as u32,
                shared_mem as u32,
                stream.as_inner(),
                kernel_params_ptr,
                std::ptr::null_mut(),
            )
        };
        if launch_status != cuda_sys::CUresult::CUDA_SUCCESS {
            caller.data().wasi_cuda().record_error(format!(
                "launch: cuLaunchKernel failed with status {launch_status:?}; \
                 param_count={param_count}"
            ));
            return Err(AbiError::LaunchFailed);
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

        // Move the stream + event + arg storage into the blocking task so
        // synchronize doesn't block the wasmtime fiber. Stream + Event
        // are Send; `KernelParamStorage` has a Send impl that asserts
        // its raw pointers are not concurrently shared. Keep `handle`
        // (and therefore the Arc<Module>) alive until after synchronize
        // completes — the storage may carry raw host pointers into the
        // guest's linear memory which `cuLaunchKernel` has already
        // captured, but the closure keeps the backing alive in case
        // CUDA dereferences it again during sync.
        let handle_for_keepalive = handle.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _keep_event = event;
            let _keep_module = handle_for_keepalive;
            let _keep_storage = storage;
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
        // Stash the parsed args for observability before releasing the
        // handle. Tests inspect `last_lowered_args` to confirm the
        // marshalling round-trip held.
        *caller
            .data()
            .wasi_cuda()
            .last_lowered_args
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = lowered_args;
        // `handle` is still in scope here; it (and the Arc<Module>) is
        // released by Drop now that synchronize has returned. The clone
        // moved into the blocking task may still hold the Arc briefly,
        // which is fine: the module stays alive until *all* clones drop.
        drop(handle);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA we can't actually run the kernel; record the
        // parsed args (so tests can confirm the lowering held) and
        // surface `NotAvailable` so the Wasm caller knows the launch
        // did not run.
        let _ = handle; // suppress unused warning on no-CUDA.
        let parsed_count = lowered_args.len();
        *caller
            .data()
            .wasi_cuda()
            .last_lowered_args
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = lowered_args;
        caller.data().wasi_cuda().record_error(format!(
            "launch: CUDA not available on this host (argv parsed: {parsed_count} args)"
        ));
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

    /// A well-formed, in-bounds `args` buffer whose first byte is an
    /// unknown tag must surface as `InvalidArgs` — distinct from
    /// `KernelArgsUnsupported` (reserved for size-cap fallbacks) and
    /// from `InvalidPointer` (reserved for OOB pointers). The 4
    /// zero-bytes the WAT writes parse as a leading 0x00 tag, which is
    /// not assigned. This is the v0.2 contract — see module docs and
    /// `docs/RISKS.md`.
    #[tokio::test]
    async fn launch_with_inbounds_unknown_tag_returns_invalid_args() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        // Tiny module: one page of memory (64 KiB), and an exported
        // `try_launch` that hands the host (kernel_id=0, 1x1x1 grid,
        // 1x1x1 block, shared_mem=0, args_ptr=0, args_len=4). The 4-byte
        // region at offset 0 is in-bounds (all zero bytes), so the
        // bounds-check passes; the parser then sees a leading 0x00 tag
        // and rejects it as `InvalidArgs`.
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
        let mut ctx = WasiCudaContext::new(InstanceId(202));
        ctx.enable_wasi_cuda();
        let mut store = wasmtime::Store::new(&engine, Dummy(ctx));
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
            AbiError::InvalidArgs.code(),
            "unknown leading tag byte must return InvalidArgs ({}), \
             not KernelArgsUnsupported ({}) or InvalidPointer ({})",
            AbiError::InvalidArgs.code(),
            AbiError::KernelArgsUnsupported.code(),
            AbiError::InvalidPointer.code(),
        );
        // Confirm the recorded error message reports the parse failure.
        let last = store.data().wasi_cuda().last_error().unwrap_or_default();
        assert!(
            last.contains("kernel argv parse failed"),
            "expected argv-parse error, got: {last}"
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
        let mut ctx = WasiCudaContext::new(InstanceId(203));
        ctx.enable_wasi_cuda();
        let mut store = wasmtime::Store::new(&engine, Dummy(ctx));
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

    /// A buffer larger than the kernel-args sanity cap surfaces as
    /// `KernelArgsUnsupported`. The cap is enforced by `parse_argv`
    /// before any per-record work — the host stub records the parse
    /// error in `last_error` and returns the negative code.
    ///
    /// We trigger this via a launch with `args_len` greater than
    /// `kernel_args::MAX_KERNEL_ARGS_BYTES` (the buffer itself is
    /// in-bounds because the WAT exports 16 pages = 1 MiB of linear
    /// memory).
    #[tokio::test]
    async fn launch_with_oversized_argv_returns_kernel_args_unsupported() {
        use crate::kernel_args::MAX_KERNEL_ARGS_BYTES;

        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        let oversized = (MAX_KERNEL_ARGS_BYTES + 1) as i32;
        // 16 pages = 1 MiB, comfortably larger than the cap so the
        // outer bounds-check passes and the size-cap is what triggers.
        let wat = format!(
            r#"
            (module
              (import "{m}" "{fn_name}"
                (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 16)
              (func (export "try_launch") (result i32)
                (call $launch
                  (i64.const 0)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0)
                  (i32.const 0)
                  (i32.const {oversized})))
            )
            "#,
            m = MODULE,
            fn_name = FN_LAUNCH,
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        let mut ctx = WasiCudaContext::new(InstanceId(220));
        ctx.enable_wasi_cuda();
        let mut store = wasmtime::Store::new(&engine, Dummy(ctx));
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
            "argv buffer past sanity cap must return KernelArgsUnsupported"
        );
    }

    /// Capability gating: when `wasi_cuda_enabled` is `false` (the default
    /// for a brand-new context), every host function linked by
    /// [`add_to_linker`] must short-circuit with `AbiError::NotAvailable`
    /// — even otherwise-valid calls. This prevents a guest from gaining
    /// CUDA access just by importing the module.
    ///
    /// Post-e4e30b6 the disabled-capability paths deliberately do NOT
    /// `record_error`: a recorded message would (a) be readable by the
    /// guest via `last_error_*` if the embedder ever flipped the
    /// capability back on, turning the gate into a leak channel, and
    /// (b) burn allocations + mutex traffic for a hostile guest that
    /// hammers disabled calls. The `NotAvailable` return code is the
    /// only signal; this test asserts that contract.
    #[tokio::test]
    async fn launch_without_capability_returns_not_available() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        // A trivially-valid launch: 1x1x1 grid + block, zero args. With the
        // capability enabled this would either succeed (CUDA) or return
        // `NotAvailable` from the no-CUDA branch *after* full validation;
        // with the capability disabled we expect `NotAvailable` directly
        // from the linker wrapper, *before* any validation work.
        //
        // We also import `last_error_len` and call it after the rejected
        // launch: on a disabled-capability context that surface returns
        // `AbiError::NotAvailable.code()` (the "surface unavailable on
        // this instance" sentinel) — NOT a positive length, because no
        // error must have been recorded.
        let wat = format!(
            r#"
            (module
              (import "{m}" "{fn_launch}"
                (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
              (import "{m}" "{fn_last_err_len}"
                (func $last_error_len (result i32)))
              (memory (export "memory") 1)
              (func (export "try_launch") (result i32)
                (call $launch
                  (i64.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))
              (func (export "probe_last_error_len") (result i32)
                (call $last_error_len))
            )
            "#,
            m = MODULE,
            fn_launch = FN_LAUNCH,
            fn_last_err_len = FN_LAST_ERROR_LEN,
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        // Note: we deliberately do NOT call `enable_wasi_cuda()`.
        let ctx = WasiCudaContext::new(InstanceId(901));
        assert!(
            !ctx.wasi_cuda_enabled(),
            "freshly-constructed context must default to disabled"
        );
        let mut store = wasmtime::Store::new(&engine, Dummy(ctx));
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
            AbiError::NotAvailable.code(),
            "ungranted wasi-cuda capability must surface as NotAvailable, \
             got {rc}"
        );
        // No error message must have been recorded — recording one would
        // leak through last_error_* if the embedder ever flipped the
        // capability back on (see the doc comment above and the rationale
        // in host.rs lines 360-368 / 393-398 / 419-422 / 436-444 / 458-462).
        assert!(
            store.data().wasi_cuda().last_error().is_none(),
            "disabled-capability path must NOT record_error, but found: {:?}",
            store.data().wasi_cuda().last_error()
        );
        // Calling last_error_len through the real ABI surface on a
        // disabled context returns `NotAvailable.code()` (i.e. -1) — the
        // documented "this surface is unavailable on this instance"
        // sentinel. Crucially this is NOT `0` (which would mean "no error
        // on a gate-passing context") and NOT a positive length (which
        // would mean an error was recorded, leaking the gate state).
        let probe = instance
            .get_typed_func::<(), i32>(&mut store, "probe_last_error_len")
            .expect("typed func");
        let len_rc = probe.call_async(&mut store, ()).await.expect("call");
        assert_eq!(
            len_rc,
            AbiError::NotAvailable.code(),
            "last_error_len on a disabled-capability context must return \
             the NotAvailable sentinel ({}), got {len_rc}",
            AbiError::NotAvailable.code()
        );
    }

    /// Once the capability is granted on the same context the launch
    /// proceeds through validation and reaches the launch dispatch path
    /// (which on no-CUDA hosts ultimately also returns `NotAvailable`,
    /// but only *after* validation runs — confirming the gate flipped).
    #[tokio::test]
    async fn launch_with_capability_passes_gate() {
        let mut config = wasmtime::Config::new();
        config.async_support(true);
        let engine = wasmtime::Engine::new(&config).unwrap();
        let mut linker: Linker<Dummy> = Linker::new(&engine);
        add_to_linker(&mut linker).expect("add_to_linker");

        // Use a bogus kernel id (999): with the capability enabled
        // validation passes the dimension caps and reaches the kernel
        // lookup, which returns `InvalidKernel`. Without the capability
        // we'd see `NotAvailable` instead — the cross-check that
        // confirms `enable_wasi_cuda` actually flips behaviour.
        let wat = format!(
            r#"
            (module
              (import "{m}" "{fn_name}"
                (func $launch (param i64 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "try_launch") (result i32)
                (call $launch
                  (i64.const 999)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 1) (i32.const 1) (i32.const 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))
            )
            "#,
            m = MODULE,
            fn_name = FN_LAUNCH,
        );
        let bytes = wat::parse_str(&wat).unwrap();
        let module = wasmtime::Module::new(&engine, &bytes).expect("compile");
        let mut ctx = WasiCudaContext::new(InstanceId(902));
        ctx.enable_wasi_cuda();
        assert!(
            ctx.wasi_cuda_enabled(),
            "enable_wasi_cuda() must flip the flag"
        );
        let mut store = wasmtime::Store::new(&engine, Dummy(ctx));
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
            AbiError::InvalidKernel.code(),
            "with capability granted, an unknown kernel id must reach the \
             registry lookup and return InvalidKernel; got {rc}"
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
