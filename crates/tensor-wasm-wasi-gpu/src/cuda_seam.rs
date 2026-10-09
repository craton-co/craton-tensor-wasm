// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Mockable CUDA driver seam (FEATURE wasi-gpu 6).
//!
//! The explicit device-memory host functions (`alloc` / `free` /
//! `memcpy-h2d` / `memcpy-d2h`) and the launch pointer-residency check all
//! bottom out in a handful of raw `cust::sys` driver calls. Those calls can
//! only run on a CUDA host with a live GPU, which made the host-side
//! *logic* around them (size caps, owner checks, bounds checks, rollback on
//! cap-trip) untestable without hardware.
//!
//! This module introduces a thin trait — [`CudaDriver`] — over exactly the
//! driver primitives those paths use. Two implementations exist:
//!
//! * [`RealCudaDriver`] (`#[cfg(feature = "cuda")]`) forwards each call to
//!   the real `cust::sys` symbol. It is a pure pass-through: the verified
//!   hardware behaviour is unchanged.
//! * [`MockCudaDriver`] (always compiled) records every call and returns
//!   caller-configured verdicts. It lets the default (no-CUDA) test build
//!   exercise the host-side control flow — including the driver-error and
//!   cap-rollback branches — with zero hardware.
//!
//! The seam is deliberately small: it models a *device pointer* as an
//! opaque `u64` and a *driver status* as [`SeamStatus`] (success / failure)
//! rather than re-exporting `cust` types, so the trait and its mock compile
//! on a no-CUDA host.

use std::sync::Mutex;

/// Opaque device-pointer value handed across the seam.
///
/// On CUDA builds this is the numeric value of a `cust::sys::CUdeviceptr`;
/// on no-CUDA builds it is an arbitrary token the mock hands out. Callers
/// treat it as opaque.
pub type SeamDevicePtr = u64;

/// Result of a seam driver call: success, or a failure carrying a short
/// diagnostic the caller folds into its `record_error` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamStatus {
    /// The driver call succeeded.
    Success,
    /// The driver call failed; the `String` is a short reason for logs.
    Failure(String),
}

impl SeamStatus {
    /// `true` when the call succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, SeamStatus::Success)
    }
}

/// The set of CUDA driver primitives the device-memory + launch paths use.
///
/// Every method is infallible at the Rust level (returns a [`SeamStatus`]
/// or a typed result) so the host code can map a driver failure onto the
/// appropriate [`crate::abi::AbiError`] without panicking.
pub trait CudaDriver: Send + Sync {
    /// Allocate `size` device bytes. Returns the device pointer on success.
    fn mem_alloc(&self, size: u64) -> Result<SeamDevicePtr, SeamStatus>;

    /// Free a previously-allocated device pointer.
    fn mem_free(&self, ptr: SeamDevicePtr) -> SeamStatus;

    /// Copy `len` host bytes from `src` into device pointer `dst`.
    ///
    /// # Safety
    ///
    /// `src` must point to at least `len` readable bytes and `dst` must name
    /// at least `len` bytes of live device memory. The host call sites
    /// bounds-check both before invoking this.
    unsafe fn memcpy_h2d(&self, dst: SeamDevicePtr, src: *const u8, len: usize) -> SeamStatus;

    /// Copy `len` device bytes from `src` into host pointer `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least `len` writable bytes and `src` must name
    /// at least `len` bytes of live device memory. The host call sites
    /// bounds-check both before invoking this.
    unsafe fn memcpy_d2h(&self, dst: *mut u8, src: SeamDevicePtr, len: usize) -> SeamStatus;

    /// Query whether the address `ptr` is CUDA managed (unified) memory —
    /// i.e. safe to hand to `cuLaunchKernel` as a device pointer.
    ///
    /// Returns `Ok(true)` for managed memory, `Ok(false)` for a
    /// driver-known-but-unmanaged address, and `Err` when the driver does
    /// not recognise the address at all (a plain host-heap pointer).
    fn pointer_is_managed(&self, ptr: u64) -> Result<bool, SeamStatus>;
}

/// Pass-through [`CudaDriver`] backed by the real `cust::sys` symbols.
///
/// Only compiled under `--features cuda`. Each method forwards verbatim to
/// the driver call the host functions previously made inline, so the
/// verified hardware behaviour is preserved.
#[cfg(feature = "cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub struct RealCudaDriver;

#[cfg(feature = "cuda")]
impl CudaDriver for RealCudaDriver {
    fn mem_alloc(&self, size: u64) -> Result<SeamDevicePtr, SeamStatus> {
        use cust::sys as cuda_sys;
        // `CUdeviceptr` is a `u64` alias, so the seam's `SeamDevicePtr`
        // round-trips it without a cast.
        let mut device_ptr: cuda_sys::CUdeviceptr = 0;
        // SAFETY: `cuMemAlloc` writes a fresh device pointer into
        // `device_ptr`; `size` is bounded by the caller (MAX_DEVICE_ALLOC_BYTES).
        let status = unsafe { cuda_sys::cuMemAlloc_v2(&mut device_ptr, size as usize) };
        if status != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(SeamStatus::Failure(format!("cuMemAlloc -> {status:?}")));
        }
        Ok(device_ptr)
    }

    fn mem_free(&self, ptr: SeamDevicePtr) -> SeamStatus {
        use cust::sys as cuda_sys;
        // SAFETY: `ptr` was produced by `mem_alloc` (a `cuMemAlloc` pointer)
        // and is freed exactly once (the registry guarantees single ownership).
        let status = unsafe { cuda_sys::cuMemFree_v2(ptr) };
        if status != cuda_sys::CUresult::CUDA_SUCCESS {
            return SeamStatus::Failure(format!("cuMemFree -> {status:?}"));
        }
        SeamStatus::Success
    }

    unsafe fn memcpy_h2d(&self, dst: SeamDevicePtr, src: *const u8, len: usize) -> SeamStatus {
        use cust::sys as cuda_sys;
        let status = cuda_sys::cuMemcpyHtoD_v2(dst, src as *const std::ffi::c_void, len);
        if status != cuda_sys::CUresult::CUDA_SUCCESS {
            return SeamStatus::Failure(format!("cuMemcpyHtoD -> {status:?}"));
        }
        SeamStatus::Success
    }

    unsafe fn memcpy_d2h(&self, dst: *mut u8, src: SeamDevicePtr, len: usize) -> SeamStatus {
        use cust::sys as cuda_sys;
        let status = cuda_sys::cuMemcpyDtoH_v2(dst as *mut std::ffi::c_void, src, len);
        if status != cuda_sys::CUresult::CUDA_SUCCESS {
            return SeamStatus::Failure(format!("cuMemcpyDtoH -> {status:?}"));
        }
        SeamStatus::Success
    }

    fn pointer_is_managed(&self, ptr: u64) -> Result<bool, SeamStatus> {
        use cust::sys as cuda_sys;
        let mut is_managed: std::os::raw::c_int = 0;
        // SAFETY: `is_managed` is a valid out-param for the IS_MANAGED
        // attribute; `cuPointerGetAttribute` only reads driver bookkeeping
        // for the address — it never dereferences it.
        let res = unsafe {
            cuda_sys::cuPointerGetAttribute(
                &mut is_managed as *mut std::os::raw::c_int as *mut std::ffi::c_void,
                cuda_sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_MANAGED,
                ptr,
            )
        };
        if res != cuda_sys::CUresult::CUDA_SUCCESS {
            // Host-heap pointers are unknown to the driver and return
            // `CUDA_ERROR_INVALID_VALUE`. Surface as `Err` so the caller can
            // distinguish "driver does not know this address" from
            // "known-but-unmanaged."
            return Err(SeamStatus::Failure(format!(
                "cuPointerGetAttribute IS_MANAGED -> {res:?}"
            )));
        }
        Ok(is_managed != 0)
    }
}

/// One recorded call against a [`MockCudaDriver`], in call order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockCall {
    /// `mem_alloc(size)`.
    Alloc(u64),
    /// `mem_free(device_ptr)`.
    Free(SeamDevicePtr),
    /// `memcpy_h2d(dst, len)` — the raw `src` pointer is not recorded.
    H2d(SeamDevicePtr, usize),
    /// `memcpy_d2h(src, len)` — the raw `dst` pointer is not recorded.
    D2h(SeamDevicePtr, usize),
    /// `pointer_is_managed(ptr)`.
    IsManaged(u64),
}

/// In-memory [`CudaDriver`] for unit tests — no hardware required.
///
/// Hands out monotonically-increasing synthetic device pointers, records
/// every call, and lets a test force failures (`fail_alloc` / `fail_copy`)
/// or control the residency verdict (`managed_addrs`). Lets the default
/// (no-CUDA) test build exercise the host-side alloc/free/memcpy/launch
/// control flow that the real driver otherwise gates behind hardware.
pub struct MockCudaDriver {
    inner: Mutex<MockState>,
}

struct MockState {
    next_ptr: SeamDevicePtr,
    calls: Vec<MockCall>,
    fail_alloc: bool,
    fail_copy: bool,
    /// Addresses the mock reports as managed; any other address is reported
    /// as unknown-to-the-driver (`Err`).
    managed_addrs: Vec<u64>,
    /// When `true`, every `pointer_is_managed` returns `Ok(true)`.
    all_managed: bool,
}

impl Default for MockCudaDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl MockCudaDriver {
    /// Construct a mock that succeeds on alloc/free/copy and reports every
    /// pointer as managed (the "happy path" default).
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MockState {
                // Start well above 0 so a synthetic pointer is never confused
                // with a null / sentinel value.
                next_ptr: 0x1_0000,
                calls: Vec::new(),
                fail_alloc: false,
                fail_copy: false,
                managed_addrs: Vec::new(),
                all_managed: true,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MockState> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Force every subsequent `mem_alloc` to fail.
    pub fn set_fail_alloc(&self, fail: bool) {
        self.lock().fail_alloc = fail;
    }

    /// Force every subsequent `memcpy_*` to fail.
    pub fn set_fail_copy(&self, fail: bool) {
        self.lock().fail_copy = fail;
    }

    /// Stop reporting every pointer as managed; only `addrs` are managed and
    /// any other address is reported as unknown-to-the-driver.
    pub fn set_managed_addrs(&self, addrs: Vec<u64>) {
        let mut g = self.lock();
        g.all_managed = false;
        g.managed_addrs = addrs;
    }

    /// Snapshot the recorded call log.
    pub fn calls(&self) -> Vec<MockCall> {
        self.lock().calls.clone()
    }
}

impl CudaDriver for MockCudaDriver {
    fn mem_alloc(&self, size: u64) -> Result<SeamDevicePtr, SeamStatus> {
        let mut g = self.lock();
        g.calls.push(MockCall::Alloc(size));
        if g.fail_alloc {
            return Err(SeamStatus::Failure("mock alloc forced failure".into()));
        }
        let ptr = g.next_ptr;
        g.next_ptr = g.next_ptr.wrapping_add(size.max(1));
        Ok(ptr)
    }

    fn mem_free(&self, ptr: SeamDevicePtr) -> SeamStatus {
        self.lock().calls.push(MockCall::Free(ptr));
        SeamStatus::Success
    }

    unsafe fn memcpy_h2d(&self, dst: SeamDevicePtr, _src: *const u8, len: usize) -> SeamStatus {
        let mut g = self.lock();
        g.calls.push(MockCall::H2d(dst, len));
        if g.fail_copy {
            return SeamStatus::Failure("mock h2d forced failure".into());
        }
        SeamStatus::Success
    }

    unsafe fn memcpy_d2h(&self, _dst: *mut u8, src: SeamDevicePtr, len: usize) -> SeamStatus {
        let mut g = self.lock();
        g.calls.push(MockCall::D2h(src, len));
        if g.fail_copy {
            return SeamStatus::Failure("mock d2h forced failure".into());
        }
        SeamStatus::Success
    }

    fn pointer_is_managed(&self, ptr: u64) -> Result<bool, SeamStatus> {
        let mut g = self.lock();
        g.calls.push(MockCall::IsManaged(ptr));
        if g.all_managed {
            return Ok(true);
        }
        if g.managed_addrs.contains(&ptr) {
            Ok(true)
        } else {
            Err(SeamStatus::Failure(
                "mock: address unknown to driver".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_alloc_hands_out_unique_pointers_and_records_calls() {
        let drv = MockCudaDriver::new();
        let a = drv.mem_alloc(4096).expect("alloc a");
        let b = drv.mem_alloc(4096).expect("alloc b");
        assert_ne!(a, b, "mock must hand out distinct pointers");
        assert_eq!(drv.calls(), vec![MockCall::Alloc(4096), MockCall::Alloc(4096)]);
    }

    #[test]
    fn mock_alloc_failure_is_observable() {
        let drv = MockCudaDriver::new();
        drv.set_fail_alloc(true);
        assert!(drv.mem_alloc(8).is_err());
    }

    #[test]
    fn mock_copy_records_and_can_fail() {
        let drv = MockCudaDriver::new();
        let dst = drv.mem_alloc(16).unwrap();
        // SAFETY: mock never dereferences the host pointer.
        let ok = unsafe { drv.memcpy_h2d(dst, std::ptr::null(), 16) };
        assert!(ok.is_success());
        drv.set_fail_copy(true);
        // SAFETY: mock never dereferences the host pointer.
        let bad = unsafe { drv.memcpy_d2h(std::ptr::null_mut(), dst, 16) };
        assert!(!bad.is_success());
        assert_eq!(
            drv.calls(),
            vec![MockCall::Alloc(16), MockCall::H2d(dst, 16), MockCall::D2h(dst, 16)]
        );
    }

    #[test]
    fn mock_residency_verdict_is_controllable() {
        let drv = MockCudaDriver::new();
        // Default: everything managed.
        assert_eq!(drv.pointer_is_managed(0xdead_beef), Ok(true));
        // Restrict to a single managed address.
        drv.set_managed_addrs(vec![0x1234]);
        assert_eq!(drv.pointer_is_managed(0x1234), Ok(true));
        assert!(drv.pointer_is_managed(0x5678).is_err());
    }

    #[test]
    fn seam_status_is_success_helper() {
        assert!(SeamStatus::Success.is_success());
        assert!(!SeamStatus::Failure("x".into()).is_success());
    }

    /// Exercise the alloc → h2d → d2h → free sequence the device-memory host
    /// functions drive, entirely against the mock — the "unit-testable
    /// without hardware" property the seam exists for (feature #6). The call
    /// log proves the order and arguments the host logic would have issued.
    #[test]
    fn host_device_mem_sequence_is_mockable_without_hardware() {
        let drv = MockCudaDriver::new();
        let dev = drv.mem_alloc(256).expect("alloc");
        // SAFETY: mock never dereferences the host pointers.
        let h2d = unsafe { drv.memcpy_h2d(dev, std::ptr::null(), 128) };
        let d2h = unsafe { drv.memcpy_d2h(std::ptr::null_mut(), dev, 64) };
        let free = drv.mem_free(dev);
        assert!(h2d.is_success() && d2h.is_success() && free.is_success());
        assert_eq!(
            drv.calls(),
            vec![
                MockCall::Alloc(256),
                MockCall::H2d(dev, 128),
                MockCall::D2h(dev, 64),
                MockCall::Free(dev),
            ]
        );
    }

    /// Mirror the launch path's alloc-then-cap-rollback shape: when a
    /// downstream cap trips after a successful driver alloc, the host frees
    /// the device pointer it just grabbed. The mock records both calls.
    #[test]
    fn alloc_then_rollback_free_is_mockable() {
        let drv = MockCudaDriver::new();
        let dev = drv.mem_alloc(1024).expect("alloc");
        // Simulate the host's "registry cap tripped → free what we grabbed".
        let _ = drv.mem_free(dev);
        assert_eq!(drv.calls(), vec![MockCall::Alloc(1024), MockCall::Free(dev)]);
    }
}
