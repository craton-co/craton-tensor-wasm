// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Wasmtime [`MemoryCreator`] backed by [`UnifiedBuffer`].
//!
//! When a Wasm module declares a `(memory 1)` (one page), Wasmtime asks the
//! [`MemoryCreator`] to materialise that memory. We pre-allocate the requested
//! `maximum` (or a default cap), so growth is just a size check — no realloc
//! and no host-to-device copy. The `as_ptr` returned points directly into
//! Unified Memory on CUDA hosts, meaning CUDA kernels can read/write the
//! buffer the Wasm guest sees, with no explicit DMA.
//!
//! # `memory_protection_keys` — known gap vs the S5 plan
//!
//! The S5 plan calls for enabling Wasmtime's `memory_protection_keys` for
//! intra-process Wasm isolation "where available". In wasmtime 25 this knob
//! lives on `PoolingAllocationConfig::memory_protection_keys`, NOT on
//! `Config`, and it is mutually exclusive with `Config::with_host_memory`:
//! pooling MPK colours wasmtime-owned slabs, whereas this crate's whole
//! purpose is to hand Wasmtime host-owned `UnifiedBuffer` memories. Inter-
//! tenant isolation therefore relies on (a) Wasmtime's Cranelift-emitted
//! bounds checks on every Wasm load/store, (b) each tenant owning a
//! distinct `UnifiedBuffer` with no shared backing store, and (c) the
//! per-instance stream and per-tenant context machinery wired up in S7
//! (`tensor-wasm-exec`) and S16 (`tensor-wasm-tenant`) — not on CPU protection keys
//! and not on OS-level page guards (managed memory is migrated by the
//! CUDA driver, which is incompatible with host `mprotect`). See
//! `SECURITY.md` at the repo root for the full threat model.
//!
//! MPK is now available as an *alternate* engine mode via
//! `tensor_wasm_exec::engine::MemoryBackend::PoolingMpk`, at the explicit cost of
//! dropping `TensorWasmMemoryCreator` / `UnifiedBuffer` integration (and therefore
//! the GPU integration path). Operators can pick the mode that fits their
//! workload: `UnifiedBuffer` for the GPU integration path described above,
//! or `PoolingMpk` for CPU-only / batch-GPU workloads that want intra-process
//! Wasm isolation via CPU PKU. The architectural exclusivity between the
//! two modes is real and enforced by Wasmtime.

use std::ops::Range;
use std::sync::Arc;

use wasmtime::{LinearMemory, MemoryCreator, MemoryType};

use crate::pool::UnifiedMemoryPool;
use crate::unified::{DeviceId, UnifiedBuffer, UnifiedError};

/// Default maximum capacity, in bytes, when a Wasm module declares no upper
/// bound on its linear memory. 256 MiB matches the plan's working-set cap.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// A Wasm linear memory backed by [`UnifiedBuffer`].
///
/// The buffer is allocated at construction time with the requested *maximum*
/// (or [`DEFAULT_MAX_BYTES`]) so `grow_to` becomes a size check. This avoids
/// the cost of `cudaMemcpy` on every `memory.grow` and keeps the kernel-side
/// pointer stable across growth events.
#[derive(Debug)]
pub struct TensorWasmLinearMemory {
    buffer: UnifiedBuffer,
    /// Currently-visible (logical) size. `<= buffer.len()`. Mutation goes
    /// through `&mut self` (see `LinearMemory::grow_to`), so no interior
    /// lock is required; `Send + Sync` are auto-derived from the fields.
    current_size: usize,
    /// Hard cap. Always `<= buffer.len()`.
    maximum_size: usize,
}

impl TensorWasmLinearMemory {
    /// Create a new linear memory.
    ///
    /// - `minimum_bytes`: initial visible size (Wasm pages × 65 536).
    /// - `maximum_bytes`: cap on growth. If `None`, [`DEFAULT_MAX_BYTES`] is used.
    pub fn new(minimum_bytes: usize, maximum_bytes: Option<usize>) -> Result<Self, UnifiedError> {
        Self::new_on(minimum_bytes, maximum_bytes, DeviceId::default())
    }

    /// Same as [`new`](Self::new) but on a specific device.
    pub fn new_on(
        minimum_bytes: usize,
        maximum_bytes: Option<usize>,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        let max = maximum_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        if minimum_bytes > max {
            return Err(UnifiedError::Allocation(format!(
                "minimum {minimum_bytes} > maximum {max}"
            )));
        }
        // Allocate at least 1 byte so the underlying allocator never sees zero.
        let cap = max.max(1);
        // Only zero the visible window. Wasm semantics require the initial
        // `minimum_bytes` to read as zero; any bytes later exposed by
        // `memory.grow` are zero-filled by Wasmtime itself. On the cust path
        // this avoids paying a `cap`-sized memset on every Wasm spawn — a
        // 256 MiB cost under the default `DEFAULT_MAX_BYTES` cap.
        let buffer = UnifiedBuffer::new_with_visible_window_on(cap, minimum_bytes, device_id)?;
        Ok(Self {
            buffer,
            current_size: minimum_bytes,
            maximum_size: max,
        })
    }

    /// Current logical size in bytes.
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Pre-allocated capacity (the hard cap).
    pub fn capacity(&self) -> usize {
        self.maximum_size
    }

    /// Whether the underlying linear-memory backing is CUDA Unified Memory.
    ///
    /// Returns `true` when the crate was compiled with EITHER
    /// `--features unified-memory` (cust path) OR `--features cudarc-backend`
    /// (the W1.2 spike, used as the `Backing::Cudarc` variant when
    /// `unified-memory` is off — see the precedence table in
    /// [`crate::unified`]). This is the compile-time probe that closes the
    /// v0.3.2 audit's "wasm linear memory not UVM-backed" gap: a guest
    /// pointer resolved through the W1.1 wasi-cuda kernel-args pipeline
    /// doubles as a device pointer iff this returns `true`. The pool-backed
    /// [`PooledLinearMemory`] path also goes through [`UnifiedBuffer`] under
    /// the hood, so it shares this property regardless of which CUDA
    /// backing feature is active.
    ///
    /// # Memory-growth semantics
    ///
    /// `cuMemAllocManaged` returns a fixed-size allocation that cannot be
    /// resized in place. We therefore pre-allocate the requested
    /// `maximum_bytes` (or [`DEFAULT_MAX_BYTES`]) at construction time and
    /// treat [`LinearMemory::grow_to`] as a logical-size bump up to that
    /// cap (option (a) from the v0.3.3 design tracker). This matches
    /// Wasmtime's `static` memory model, keeps the kernel-side pointer
    /// stable across growth events, and keeps the hot path zero-copy at
    /// the cost of reserving the worst-case footprint up front. Growing
    /// the *physical* allocation (option (b): allocate-copy-free) is a
    /// v0.4 follow-up tracked in `docs/RISKS.md`.
    pub fn is_uvm_backed(&self) -> bool {
        self.buffer.is_uvm_backed()
    }

    /// Borrow the buffer as a shared byte slice over the *current* size.
    ///
    /// # Safety contract
    ///
    /// This method is `pub(crate)` because the returned `&[u8]` aliases the
    /// same bytes that Wasmtime mutates through [`LinearMemory::as_ptr`]
    /// while the owning `Store` is executing guest code. The caller MUST NOT
    /// hold the returned slice across any operation that may transfer control
    /// to the guest (e.g. `TypedFunc::call`) — doing so would race a `&[u8]`
    /// against the guest's mutable view of its linear memory, which is UB.
    /// Intended uses: host-side inspection in tests and the trusted host
    /// helpers within this crate, between guest invocations.
    pub(crate) fn as_slice(&self) -> &[u8] {
        let size = self.current_size;
        // SAFETY: the underlying buffer covers `maximum_size >= size` bytes.
        // Aliasing with the guest is the caller's responsibility per the
        // doc-comment safety contract above.
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), size) }
    }
}

unsafe impl LinearMemory for TensorWasmLinearMemory {
    fn byte_size(&self) -> usize {
        self.current_size()
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        Some(self.maximum_size)
    }

    fn grow_to(&mut self, new_size: usize) -> anyhow::Result<()> {
        if new_size > self.maximum_size {
            return Err(anyhow::anyhow!(
                "memory.grow requested {new_size} > maximum {}",
                self.maximum_size
            ));
        }
        if new_size < self.current_size {
            return Err(anyhow::anyhow!(
                "memory.grow cannot shrink ({new_size} < current {})",
                self.current_size
            ));
        }
        self.current_size = new_size;
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        // SAFETY: returning the raw underlying pointer is the contract of
        // LinearMemory. Wasmtime guarantees synchronised access via its own
        // borrow tracking.
        self.buffer.as_ptr() as *mut u8
    }

    fn wasm_accessible(&self) -> Range<usize> {
        let base = self.buffer.as_ptr() as usize;
        base..(base + self.maximum_size)
    }
}

/// Wasm linear memory carved out of an [`UnifiedMemoryPool`] slab.
///
/// Holds an `Arc<UnifiedMemoryPool>` keepalive so the slab outlives the Wasm
/// instance, plus a raw pointer + length into the slab's bytes. The
/// [`crate::pool::PoolAllocation`] drop guard is intentionally leaked at
/// construction time (see `TensorWasmMemoryCreator::new_memory`); the live-allocation
/// counter therefore stays elevated for the lifetime of this memory and serves
/// as a misuse signal if the caller tries to [`UnifiedMemoryPool::reset`] while
/// any pool-backed memory still exists.
struct PooledLinearMemory {
    /// Keeps the slab alive while this memory is in use. Never read
    /// directly — its Drop is the whole point.
    #[allow(dead_code)]
    pool_keepalive: Arc<UnifiedMemoryPool>,
    /// Pointer to the first byte of the carved region.
    base_ptr: *mut u8,
    /// Bytes carved (the hard cap).
    size: usize,
    /// Currently-visible (logical) size. `<= size`.
    current_size: usize,
    /// Hard cap exposed to Wasm. Equals `size`.
    max_size: usize,
}

impl std::fmt::Debug for PooledLinearMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledLinearMemory")
            .field("base_ptr", &self.base_ptr)
            .field("size", &self.size)
            .field("current_size", &self.current_size)
            .field("max_size", &self.max_size)
            .finish()
    }
}

// SAFETY: `base_ptr` points into the slab owned by `pool_keepalive`, which is
// an `Arc<UnifiedMemoryPool>`. `UnifiedMemoryPool` (via `UnifiedBuffer`) is
// `Send + Sync` and the carved region is disjoint from every other pool
// allocation by the bump allocator's invariant. Concurrent mutation through
// the raw pointer requires external synchronisation — same contract as
// `TensorWasmLinearMemory` and `Vec<u8>` once a `&mut [u8]` exists.
unsafe impl Send for PooledLinearMemory {}
unsafe impl Sync for PooledLinearMemory {}

unsafe impl LinearMemory for PooledLinearMemory {
    fn byte_size(&self) -> usize {
        self.current_size
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        Some(self.max_size)
    }

    fn grow_to(&mut self, new_size: usize) -> anyhow::Result<()> {
        if new_size > self.max_size {
            return Err(anyhow::anyhow!(
                "memory.grow requested {new_size} > maximum {}",
                self.max_size
            ));
        }
        if new_size < self.current_size {
            return Err(anyhow::anyhow!(
                "memory.grow cannot shrink ({new_size} < current {})",
                self.current_size
            ));
        }
        self.current_size = new_size;
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        self.base_ptr
    }

    fn wasm_accessible(&self) -> Range<usize> {
        let base = self.base_ptr as usize;
        base..(base + self.max_size)
    }
}

/// A [`MemoryCreator`] that hands out [`TensorWasmLinearMemory`] instances.
///
/// Wrap in [`Arc`] and pass to `wasmtime::Config::with_host_memory`.
#[derive(Clone)]
pub struct TensorWasmMemoryCreator {
    inner: Arc<MemoryCreatorState>,
}

struct MemoryCreatorState {
    device_id: DeviceId,
    /// Optional pre-allocated slab. When set, `new_memory` carves Wasm linear
    /// memories from this pool via [`UnifiedMemoryPool::allocate`]; if the
    /// slab is exhausted it falls back to a fresh [`UnifiedBuffer`].
    pool: Option<Arc<UnifiedMemoryPool>>,
}

impl Default for TensorWasmMemoryCreator {
    fn default() -> Self {
        Self::new(DeviceId::default())
    }
}

impl TensorWasmMemoryCreator {
    /// Construct without an underlying pool. New memories allocate fresh
    /// [`UnifiedBuffer`]s on every `new_memory` call.
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: None,
            }),
        }
    }

    /// Construct with a pre-allocated pool. `new_memory` carves out of this
    /// slab; on exhaustion it falls back to a fresh [`UnifiedBuffer`] so a
    /// short slab does not turn into a fatal error.
    ///
    /// **Lifetime contract.** Pool-backed linear memories form an
    /// all-or-nothing batch: the caller must drop every Wasm instance handed
    /// out by this creator *before* dropping the last `Arc<UnifiedMemoryPool>`
    /// reference and calling [`UnifiedMemoryPool::reset`]. See `new_memory`
    /// for the `unsafe` rationale.
    pub fn with_pool(device_id: DeviceId, pool: Arc<UnifiedMemoryPool>) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: Some(pool),
            }),
        }
    }

    /// The device this creator targets.
    pub fn device_id(&self) -> DeviceId {
        self.inner.device_id
    }

    /// The underlying pool, if one was provided at construction.
    pub fn pool(&self) -> Option<&Arc<UnifiedMemoryPool>> {
        self.inner.pool.as_ref()
    }
}

unsafe impl MemoryCreator for TensorWasmMemoryCreator {
    fn new_memory(
        &self,
        _ty: MemoryType,
        minimum: usize,
        maximum: Option<usize>,
        reserved_size_in_bytes: Option<usize>,
        guard_size_in_bytes: usize,
    ) -> Result<Box<dyn LinearMemory>, String> {
        let max = maximum.unwrap_or(DEFAULT_MAX_BYTES);
        if minimum > max {
            return Err(format!("minimum {minimum} > maximum {max}"));
        }

        // We do not allocate OS guard pages for `TensorWasmLinearMemory`: it is
        // backed by `UnifiedBuffer` (managed memory on CUDA hosts), and the
        // CUDA driver migrates managed pages between host and device — host
        // `mprotect`/`VirtualProtect` would race the migration machinery. We
        // therefore cannot honour a non-zero `guard_size_in_bytes` and must
        // reject the configuration outright rather than silently degrade
        // Wasmtime's expectations of OOB-trapping behaviour. Callers wanting
        // page-level guards should disable the pooling/guard knobs in
        // `wasmtime::Config` (see `crates/tensor-wasm-mem/tests/isolation.rs`) or
        // switch the engine to the `PoolingMpk` backend.
        if guard_size_in_bytes > 0 {
            return Err(format!(
                "TensorWasmMemoryCreator cannot honour guard_size_in_bytes = {guard_size_in_bytes}: \
                 managed-memory backings are incompatible with host page guards. \
                 Set `Config::dynamic_memory_guard_size(0)` and \
                 `Config::guard_before_linear_memory(false)`, or use the PoolingMpk backend."
            ));
        }
        // A `reserved_size_in_bytes` request larger than what we are about to
        // allocate cannot be satisfied — we always allocate exactly the cap.
        if let Some(reserved) = reserved_size_in_bytes {
            if reserved > max {
                return Err(format!(
                    "TensorWasmMemoryCreator cannot reserve {reserved} bytes: backing capacity is {max}"
                ));
            }
        }

        if let Some(pool) = self.inner.pool.as_ref() {
            // Wasm linear memory is page-aligned by spec (64 KiB pages).
            const WASM_PAGE: usize = 65_536;
            let carve_size = max.max(1);
            match pool.allocate(carve_size, WASM_PAGE) {
                Ok(mut alloc) => {
                    let slice = alloc.as_mut_slice();
                    let base_ptr = slice.as_mut_ptr();
                    let size = slice.len();
                    // SAFETY: We leak the PoolAllocation drop guard intentionally.
                    // The pool uses "batch reclaim" semantics (reset() succeeds
                    // only when live count is zero) and pool-backed linear
                    // memories are torn down together when the parent
                    // TensorWasmMemoryCreator's Arc<UnifiedMemoryPool> drops. The
                    // PoolAllocation's Drop would decrement the live counter;
                    // we keep that counter as a leak-detection signal for misuse.
                    //
                    // Trade-off: this prevents the pool from being reset while
                    // ANY pooled memory exists. Caller's responsibility to drop
                    // the creator (and thus the pool Arc) before reset.
                    //
                    // The raw `base_ptr` remains valid because `pool_keepalive`
                    // holds the slab alive for the lifetime of the returned
                    // `PooledLinearMemory`, and the bump allocator never hands
                    // the same byte range to another allocation.
                    std::mem::forget(alloc);
                    return Ok(Box::new(PooledLinearMemory {
                        pool_keepalive: Arc::clone(pool),
                        base_ptr,
                        size,
                        current_size: minimum,
                        max_size: size,
                    }));
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        requested = carve_size,
                        remaining = pool.remaining(),
                        "pool exhausted; falling back to fresh UnifiedBuffer"
                    );
                    // Pool/creator device-id mismatch is a configuration smell:
                    // the fallback `UnifiedBuffer` will be anchored to the
                    // creator's device, not the pool's, so memory locality
                    // assumptions baked into the pool layout no longer hold.
                    // Surface as a warning rather than an error so the
                    // fallback still serves the allocation.
                    let pool_device = pool.device_id();
                    if pool_device != self.inner.device_id {
                        tracing::warn!(
                            pool_device_id = %pool_device,
                            creator_device_id = %self.inner.device_id,
                            "fallback UnifiedBuffer will use creator's device, \
                             which differs from the exhausted pool's device"
                        );
                    }
                }
            }
        }

        let mem = TensorWasmLinearMemory::new_on(minimum, maximum, self.inner.device_id)
            .map_err(|e| e.to_string())?;
        Ok(Box::new(mem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_query_size() {
        let mem = TensorWasmLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        assert_eq!(mem.byte_size(), 64 * 1024);
        assert_eq!(mem.maximum_byte_size(), Some(1024 * 1024));
        assert_eq!(mem.capacity(), 1024 * 1024);
    }

    #[test]
    fn grow_increases_visible_size() {
        let mut mem = TensorWasmLinearMemory::new(64 * 1024, Some(256 * 1024)).unwrap();
        mem.grow_to(128 * 1024).expect("grow");
        assert_eq!(mem.byte_size(), 128 * 1024);
    }

    #[test]
    fn grow_beyond_maximum_rejected() {
        let mut mem = TensorWasmLinearMemory::new(64 * 1024, Some(128 * 1024)).unwrap();
        let err = mem.grow_to(256 * 1024).unwrap_err();
        assert!(err.to_string().contains("maximum"));
    }

    #[test]
    fn shrink_rejected() {
        let mut mem = TensorWasmLinearMemory::new(128 * 1024, Some(1024 * 1024)).unwrap();
        let err = mem.grow_to(64 * 1024).unwrap_err();
        assert!(err.to_string().contains("shrink"));
    }

    #[test]
    fn ptr_stable_across_grow() {
        let mut mem = TensorWasmLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        let p1 = mem.as_ptr();
        mem.grow_to(256 * 1024).unwrap();
        let p2 = mem.as_ptr();
        assert_eq!(p1, p2, "as_ptr must be stable across grow_to");
    }

    #[test]
    fn wasm_accessible_covers_capacity() {
        let mem = TensorWasmLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        let r = mem.wasm_accessible();
        assert_eq!(r.end - r.start, 1024 * 1024);
    }

    #[test]
    fn minimum_exceeds_maximum_rejected() {
        let err = TensorWasmLinearMemory::new(1024 * 1024, Some(512 * 1024)).unwrap_err();
        assert!(matches!(err, UnifiedError::Allocation(_)));
    }

    #[test]
    fn creator_default_device_zero() {
        let c = TensorWasmMemoryCreator::default();
        assert_eq!(c.device_id(), DeviceId(0));
    }

    #[test]
    fn creator_default_pool_is_none() {
        let c = TensorWasmMemoryCreator::default();
        assert!(c.pool().is_none());
    }

    #[test]
    fn creator_with_pool_round_trip() {
        let pool = std::sync::Arc::new(crate::pool::UnifiedMemoryPool::new(64 * 1024).unwrap());
        let creator = TensorWasmMemoryCreator::with_pool(DeviceId(2), pool.clone());
        assert_eq!(creator.device_id(), DeviceId(2));
        assert!(creator.pool().is_some());
        assert!(std::sync::Arc::ptr_eq(creator.pool().unwrap(), &pool));
    }

    #[test]
    fn as_slice_reflects_written_bytes_and_current_size() {
        // Grow into the pre-allocated cap, scribble through the LinearMemory
        // `as_ptr()` contract that Wasmtime would use, then read back via the
        // crate-private `as_slice()` host-inspection path. This covers the
        // intended use: post-execution snapshot/diagnostic reads between guest
        // invocations (see the safety contract on `as_slice`).
        let mut mem = TensorWasmLinearMemory::new(64 * 1024, Some(256 * 1024)).unwrap();
        mem.grow_to(128 * 1024).expect("grow");
        // SAFETY: no guest is executing; we hold `&mut mem` exclusively, so
        // writing through `as_ptr()` cannot race the Wasmtime borrow tracker.
        unsafe {
            let p = mem.as_ptr();
            *p.add(0) = 0xDE;
            *p.add(1) = 0xAD;
            *p.add(64 * 1024) = 0xBE;
            *p.add(128 * 1024 - 1) = 0xEF;
        }
        let s = mem.as_slice();
        assert_eq!(s.len(), 128 * 1024, "slice tracks current_size, not capacity");
        assert_eq!(s[0], 0xDE);
        assert_eq!(s[1], 0xAD);
        assert_eq!(s[64 * 1024], 0xBE);
        assert_eq!(s[128 * 1024 - 1], 0xEF);
    }

    #[test]
    fn is_uvm_backed_matches_feature_flag() {
        // Closes the v0.3.2 audit gap (Problem #5). With EITHER `--features
        // unified-memory` (cust path) OR `--features cudarc-backend` (the
        // W1.2 spike now wired in as a third `UnifiedBuffer` Backing
        // variant, see `crate::unified` precedence table), the wasm linear
        // memory's backing IS `cuMemAllocManaged` and the host pointer
        // doubles as a device pointer for kernel args. Without either
        // feature, the backing is a heap `Box<[u8]>`. Either way the probe
        // must reflect build configuration — never lie.
        let mem = TensorWasmLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        #[cfg(feature = "unified-memory")]
        assert!(
            mem.is_uvm_backed(),
            "with --features unified-memory the wasm linear memory MUST be UVM-backed"
        );
        #[cfg(all(not(feature = "unified-memory"), feature = "cudarc-backend"))]
        assert!(
            mem.is_uvm_backed(),
            "with --features cudarc-backend the wasm linear memory MUST be UVM-backed"
        );
        #[cfg(all(not(feature = "unified-memory"), not(feature = "cudarc-backend")))]
        assert!(
            !mem.is_uvm_backed(),
            "without any CUDA backing feature the linear memory must be the heap Box<[u8]>"
        );
    }

    #[test]
    fn as_ptr_returns_non_null_inside_backing_region() {
        // The kernel-args pipeline (W1.1, see
        // `crates/tensor-wasm-wasi-gpu/src/kernel_args.rs`) translates a
        // guest pointer to a host pointer via `as_ptr() + guest_offset`.
        // Under --features unified-memory that host pointer doubles as a
        // device pointer, so two properties must hold for every backing:
        // (1) `as_ptr()` is non-null; (2) the returned pointer lies inside
        // the buffer's `wasm_accessible()` region, i.e. the byte at
        // offset 0 is reachable from a kernel via the same address.
        let mem = TensorWasmLinearMemory::new(64 * 1024, Some(1024 * 1024)).unwrap();
        let p = LinearMemory::as_ptr(&mem);
        assert!(!p.is_null(), "as_ptr must be non-null");
        let r = mem.wasm_accessible();
        let p_addr = p as usize;
        assert!(
            r.contains(&p_addr),
            "as_ptr ({p_addr:#x}) must land inside wasm_accessible ({:#x}..{:#x})",
            r.start,
            r.end,
        );
    }

    #[test]
    fn grow_up_to_preallocated_cap_succeeds_beyond_fails() {
        // Memory-growth semantics for the UVM path: pre-allocate at the
        // requested maximum and treat grow_to as a logical-size bump up
        // to that cap (option (a) in the task brief). This test pins
        // both halves of that contract: grow up to the cap succeeds in
        // a single step; one byte beyond the cap fails. It runs on both
        // the heap and the UVM backing because the contract is the same
        // for both; only the underlying allocator differs.
        const MAX: usize = 256 * 1024;
        let mut mem = TensorWasmLinearMemory::new(64 * 1024, Some(MAX)).unwrap();
        // Step 1: grow exactly to the cap — must succeed.
        mem.grow_to(MAX).expect("grow up to cap must succeed");
        assert_eq!(mem.byte_size(), MAX);
        // Step 2: anything past the cap is rejected; the pre-allocated
        // region cannot be resized in place under `cuMemAllocManaged`.
        let err = mem
            .grow_to(MAX + 1)
            .expect_err("grow past cap must fail");
        assert!(err.to_string().contains("maximum"));
    }

    #[test]
    fn creator_with_pool_carves_from_slab() {
        use std::sync::Arc;
        let pool = Arc::new(crate::pool::UnifiedMemoryPool::new(8 * 1024 * 1024).unwrap());
        let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), pool.clone());
        let mt = wasmtime::MemoryType::new(1, Some(2));
        use wasmtime::MemoryCreator;
        let mem = creator
            .new_memory(mt, 64 * 1024, Some(128 * 1024), None, 0)
            .expect("new_memory");
        assert!(mem.byte_size() == 64 * 1024);
        assert!(mem.maximum_byte_size() == Some(128 * 1024));
        // Verify the pool's live count incremented (proving the carving path ran)
        assert_eq!(pool.live_allocations(), 1);
        // Note: pool.reset() will fail until the leaked PoolAllocation count is
        // cleared — by design (see SAFETY comment in new_memory).
    }
}
