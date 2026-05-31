// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Process-wide CUDA primary-context management for the `wasi-cuda` launch
//! path (roadmap fix #6).
//!
//! ## The thread-bound-context bug this closes
//!
//! `cust` 0.3 uses CUDA's *primary context* model: [`cust::quick_init`] runs
//! `cuInit` + `cuDevicePrimaryCtxRetain` and makes the retained context
//! current via `cuCtxSetCurrent` — but `cuCtxSetCurrent` binds the context to
//! the **calling thread only**. The CUDA driver keeps the "current context" in
//! thread-local state.
//!
//! The launch path is multi-threaded:
//!
//! * `load_ptx` / `launch` run on whatever thread the wasmtime async fiber is
//!   polled on (a tokio worker), which may never have touched CUDA;
//! * `launch` then moves `stream.synchronize()` into
//!   [`tokio::task::spawn_blocking`], which runs on a *different* thread from
//!   the blocking pool;
//! * libtest runs each `#[test]` / `#[tokio::test]` on its own thread.
//!
//! Before this module, only the first thread that happened to call
//! `quick_init` (typically a `tensor-wasm-mem` allocation, or the e2e test's
//! one-shot init) had a current context. Every other thread that reached a
//! `cuLaunchKernel` / `cuStreamSynchronize` saw `CUDA_ERROR_INVALID_CONTEXT`.
//! That surfaced as spurious `InvalidContext` launch failures under libtest
//! and — more importantly — in the production `spawn_blocking` synchronize
//! path.
//!
//! ## The fix
//!
//! Retain the device's primary context **once** per process (cached in a
//! `OnceLock`) and re-bind it to the **calling thread** on every entry into a
//! CUDA operation via [`ensure_current_context`]. Because primary contexts are
//! reference-counted and unique per device, retaining the same device's
//! primary context from several call sites (here, plus
//! `tensor-wasm-mem`'s `unified::ensure_cuda_init`) all resolve to the *same*
//! underlying `CUcontext`; `cuCtxSetCurrent` on any handle binds that one
//! context. So binding is always consistent regardless of which subsystem
//! initialised CUDA first.
//!
//! `cuCtxSetCurrent` is idempotent and cheap on a thread already bound to the
//! same context (the driver compares pointer identity — a single TLS write),
//! so callers invoke [`ensure_current_context`] unconditionally at the top of
//! every CUDA entry point, including inside the `spawn_blocking` closure.
//!
//! This mirrors the `mem M4` `ensure_context_bound` discipline the cudarc
//! backend already applies (`crates/tensor-wasm-mem/src/cudarc_backend.rs`);
//! the cust launch path needed the same treatment.

#![cfg(feature = "cuda")]

use std::sync::OnceLock;

use cust::sys as cuda_sys;

/// The process-wide device-0 primary-context handle, retained once.
///
/// We deliberately do NOT use `cust::quick_init` here. `quick_init` (and
/// `cust::context::Context::new`) is a *one-per-process* initialiser: calling it
/// after another subsystem has already initialised CUDA fails. In particular
/// `tensor-wasm-mem`'s `unified-memory` (cust) path runs its own init when the
/// `TensorWasmMemoryCreator` allocates a guest linear memory, so a `quick_init`
/// here would race it and one of the two would cache an `Err` forever — which
/// surfaced as spurious `CUDA_ERROR_INVALID_CONTEXT` at `Module::from_ptx` /
/// `cuLaunchKernel` once both code paths ran in the same process.
///
/// Instead we retain the device's PRIMARY context directly via
/// `cuDevicePrimaryCtxRetain`. The primary context is reference-counted and
/// unique per device, so every retainer in the process (here, plus
/// tensor-wasm-mem, plus cust's own `quick_init` if anything calls it) resolves
/// to the SAME `CUcontext`. Binding it current on a thread is then just a
/// thread-local `cuCtxSetCurrent`. We keep the handle for the process lifetime
/// (never `cuDevicePrimaryCtxRelease`) so the refcount never hits zero and
/// invalidates outstanding allocations.
///
/// Stored as `usize` (the raw `CUcontext` pointer value) so the `OnceLock` is
/// `Send + Sync`; the driver handle is just an opaque pointer.
static PRIMARY_CTX: OnceLock<Result<usize, String>> = OnceLock::new();

fn retain_primary_ctx() -> Result<usize, String> {
    // SAFETY: all calls are the documented driver C ABI; out-params are valid
    // locals. `cuInit` is idempotent; `cuDevicePrimaryCtxRetain` is
    // reference-counted and returns the per-device singleton context.
    unsafe {
        let r = cuda_sys::cuInit(0);
        if r != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuInit -> {r:?}"));
        }
        let mut dev: cuda_sys::CUdevice = 0;
        let r = cuda_sys::cuDeviceGet(&mut dev as *mut _, 0);
        if r != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuDeviceGet(0) -> {r:?}"));
        }
        let mut ctx: cuda_sys::CUcontext = std::ptr::null_mut();
        let r = cuda_sys::cuDevicePrimaryCtxRetain(&mut ctx as *mut _, dev);
        if r != cuda_sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuDevicePrimaryCtxRetain -> {r:?}"));
        }
        Ok(ctx as usize)
    }
}

/// Ensure device 0's primary CUDA context exists **and** is current on the
/// calling thread.
///
/// Idempotent and safe to call from any thread — the tokio fiber thread, a
/// `spawn_blocking` pool thread, or a libtest harness thread. The first call in
/// the process retains the device-0 primary context; every call (including the
/// first) then `cuCtxSetCurrent`s it onto the current thread.
///
/// Returns the underlying CUDA error as a `String` on failure so callers can
/// fold it into their own `record_error` + `AbiError` mapping.
pub fn ensure_current_context() -> Result<(), String> {
    let ctx = PRIMARY_CTX.get_or_init(retain_primary_ctx);
    match ctx {
        Ok(raw) => {
            // SAFETY: `raw` is the live primary-context handle retained above
            // (held for the process lifetime, so still valid). `cuCtxSetCurrent`
            // binds it to this thread; idempotent and cheap when already bound.
            let res = unsafe { cuda_sys::cuCtxSetCurrent(*raw as cuda_sys::CUcontext) };
            if res == cuda_sys::cudaError_enum::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!("cuCtxSetCurrent -> {res:?}"))
            }
        }
        Err(msg) => Err(msg.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This module is `#![cfg(feature = "cuda")]`, so the test only exists in
    /// `--features cuda` builds. On such a build WITHOUT a live driver/GPU
    /// (e.g. a cuda-feature compile-check box) `quick_init` fails and the error
    /// is cached and returned verbatim — the call must not panic. On a real
    /// CUDA host this returns `Ok(())`; the end-to-end hardware behaviour
    /// (context current on the `spawn_blocking` synchronize thread) is covered
    /// by the `#[ignore]`d e2e launch test.
    #[test]
    fn ensure_current_context_does_not_panic() {
        // We only assert it returns (Ok on hardware, Err without a driver)
        // without panicking. The result is intentionally not unwrapped so a
        // driver-less cuda-feature build stays green.
        let _ = ensure_current_context();
    }
}
