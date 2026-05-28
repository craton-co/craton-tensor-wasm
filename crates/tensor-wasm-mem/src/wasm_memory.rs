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

/// Absolute upper bound on linear-memory capacity per Wasm instance.
///
/// Wasmtime's [`wasmtime::ResourceLimiter::memory_growing`] only fires on
/// `memory.grow` — NOT on the initial allocation a module declares with
/// `(memory N M)`. A malicious or buggy guest can therefore force the host
/// to allocate up to 4 GiB (the spec maximum for a 32-bit Wasm memory:
/// 65 536 pages × 64 KiB) at instantiation time without ever calling
/// `memory.grow`. This constant is the hard ceiling enforced by
/// [`TensorWasmLinearMemory::new_on`] and
/// [`TensorWasmMemoryCreator::new_memory`]; oversize requests are rejected
/// with [`UnifiedError::TooLarge`] before any backing allocation happens.
///
/// 4 GiB matches the wasm32 spec maximum so a compliant module that
/// reserves the full address space is still admitted (subject to the
/// per-engine `EngineConfig::max_memory_bytes`); anything larger than the
/// spec maximum is fail-fast on the host side.
pub const HARD_MAX_LINEAR_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

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
    ///
    /// Fails with [`UnifiedError::TooLarge`] when `maximum_bytes` (or the
    /// `minimum_bytes` floor) would exceed [`HARD_MAX_LINEAR_MEMORY_BYTES`].
    /// This closes mem-H5 / exec-S-2 / exec-S-10: Wasmtime's
    /// [`wasmtime::ResourceLimiter::memory_growing`] only fires on
    /// `memory.grow`, so a guest declaring `(memory 1 65536)` would
    /// otherwise force a 4 GiB allocation at instantiation. The hard cap
    /// is enforced *before* the backing allocator is invoked.
    pub fn new_on(
        minimum_bytes: usize,
        maximum_bytes: Option<usize>,
        device_id: DeviceId,
    ) -> Result<Self, UnifiedError> {
        let max = maximum_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        if max > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(UnifiedError::TooLarge {
                requested: max as u64,
                limit: HARD_MAX_LINEAR_MEMORY_BYTES as u64,
            });
        }
        if minimum_bytes > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(UnifiedError::TooLarge {
                requested: minimum_bytes as u64,
                limit: HARD_MAX_LINEAR_MEMORY_BYTES as u64,
            });
        }
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

    /// Tenant-aware variant of [`Self::new_on`].
    ///
    /// Routes the underlying [`UnifiedBuffer`] allocation through
    /// [`UnifiedBuffer::new_with_visible_window_on_with_tenant_context`]
    /// so the tenant's GPU memory cap is consulted before allocation
    /// and `release_gpu_bytes(cap)` is called on Drop. The same
    /// `HARD_MAX_LINEAR_MEMORY_BYTES` ceiling and `min > max` checks
    /// run first so a cap violation never races against a guest-bug
    /// rejection.
    ///
    /// Roadmap feature #8 path: invoked by
    /// `TensorWasmMemoryCreator::with_tenant_context` /
    /// `TensorWasmMemoryCreator::with_pool_and_tenant_context` whenever
    /// they fall through to the no-pool fresh-allocation branch.
    pub fn new_on_with_tenant_context(
        minimum_bytes: usize,
        maximum_bytes: Option<usize>,
        device_id: DeviceId,
        tenant_ctx: Arc<tensor_wasm_tenant::TenantContext>,
    ) -> Result<Self, tensor_wasm_core::error::TensorWasmError> {
        let max = maximum_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        if max > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(UnifiedError::TooLarge {
                requested: max as u64,
                limit: HARD_MAX_LINEAR_MEMORY_BYTES as u64,
            }
            .into());
        }
        if minimum_bytes > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(UnifiedError::TooLarge {
                requested: minimum_bytes as u64,
                limit: HARD_MAX_LINEAR_MEMORY_BYTES as u64,
            }
            .into());
        }
        if minimum_bytes > max {
            return Err(UnifiedError::Allocation(format!(
                "minimum {minimum_bytes} > maximum {max}"
            ))
            .into());
        }
        let cap = max.max(1);
        let buffer = UnifiedBuffer::new_with_visible_window_on_with_tenant_context(
            cap,
            minimum_bytes,
            device_id,
            tenant_ctx,
        )?;
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
        // Cross-tenant data-leak mitigation (audit H2):
        // -------------------------------------------------------------------
        // The WebAssembly spec requires bytes newly exposed by `memory.grow`
        // to read as zero. Because we pre-allocate the entire `maximum_size`
        // up front and only bump `current_size`, the host backing already
        // contains *whatever the allocator left there* in the
        // `[current_size, maximum_size)` window. Wasmtime does NOT zero
        // host-supplied memory it didn't allocate, so without an explicit
        // zero-fill here a guest could observe (a) the previous tenant's
        // data if this buffer came out of a recycled `UnifiedMemoryPool`
        // slab, or (b) uninitialised driver memory from `cuMemAllocManaged`.
        //
        // Zero the freshly-exposed range BEFORE the visible-size bump so a
        // concurrent reader (e.g. host-side telemetry via `as_slice()`) that
        // observes the new size also observes the zero bytes — never an
        // intermediate state where `current_size` has grown but the bytes
        // are still stale.
        let old_size = self.current_size;
        if new_size > old_size {
            // `buffer.as_mut_slice()` covers `maximum_size >= new_size` bytes,
            // so this range is in-bounds. The `&mut self` borrow on `grow_to`
            // proves no concurrent reader can hold a `&[u8]` alias.
            self.buffer.as_mut_slice()[old_size..new_size].fill(0);
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
        // Cross-tenant data-leak mitigation (audit H2): mirror the zero-fill
        // discipline of `TensorWasmLinearMemory::grow_to`. The carved slab
        // region was zeroed at `UnifiedMemoryPool::allocate` time, but
        // anything the *guest itself* scribbled into `[current_size,
        // new_size)` (which it could legally do via OOB writes that Wasmtime
        // bounds-checked against the OLD `current_size` only — those checks
        // pass for offsets up to old `current_size`, not `max_size`, but a
        // future code path or a host-side helper writing into the
        // pre-allocated tail would still be observable on grow). Zero the
        // newly-exposed window before bumping the visible size so the spec's
        // "newly accessible bytes read as zero" guarantee holds end-to-end.
        let old_size = self.current_size;
        if new_size > old_size {
            // SAFETY: `base_ptr` points to `size >= new_size` valid bytes
            // (size == max_size; new_size <= max_size checked above); the
            // bump allocator guarantees the carved region is disjoint from
            // every other `PoolAllocation`, and `&mut self` proves no
            // concurrent reader observes this `PooledLinearMemory`.
            unsafe {
                std::ptr::write_bytes(self.base_ptr.add(old_size), 0u8, new_size - old_size);
            }
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
    /// Tenant context for GPU memory accounting (roadmap feature #8).
    /// Set via [`TensorWasmMemoryCreator::with_tenant_context`]; when
    /// present, every fresh [`UnifiedBuffer`] allocation (the fallback
    /// path when the pool is exhausted, or the no-pool path) routes
    /// through
    /// [`UnifiedBuffer::new_on_with_tenant_context`], which calls
    /// `consume_gpu_bytes` before allocating and `release_gpu_bytes`
    /// on Drop.
    ///
    /// **Pool path is intentionally unmetered.** Pool-carved memories
    /// share one slab allocation that was already paid for at pool
    /// construction; counting each carve against the cap would
    /// double-count the slab. The pool teardown documented on
    /// `UnifiedMemoryPool` already enforces the all-or-nothing
    /// lifecycle. A future GPU-quota refinement may apportion the slab
    /// across tenants at pool-construction time; that is tracked in
    /// `docs/GPU-QUOTAS.md` "v0.4 follow-up".
    tenant_ctx: Option<Arc<tensor_wasm_tenant::TenantContext>>,
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
                tenant_ctx: None,
            }),
        }
    }

    /// Construct a tenant-aware creator that accounts every fresh
    /// [`UnifiedBuffer`] against the tenant's GPU memory cap.
    ///
    /// Roadmap feature #8 builder: holds an `Arc<TenantContext>` and
    /// routes the no-pool / pool-fallback allocation paths through
    /// [`UnifiedBuffer::new_on_with_tenant_context`]. On a cap
    /// violation the underlying error is
    /// [`tensor_wasm_core::error::TensorWasmError::GpuMemoryExhausted`];
    /// it surfaces here as a `String` because Wasmtime's
    /// `MemoryCreator::new_memory` returns `Result<_, String>` (the
    /// outer API is stuck on that signature). The structured error is
    /// still available to non-Wasmtime callers that allocate
    /// `UnifiedBuffer` directly.
    ///
    /// **Pool-carved memories are intentionally unmetered** — see the
    /// note on [`MemoryCreatorState::tenant_ctx`] for the
    /// rationale and the v0.4 follow-up.
    pub fn with_tenant_context(
        device_id: DeviceId,
        tenant_ctx: Arc<tensor_wasm_tenant::TenantContext>,
    ) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: None,
                tenant_ctx: Some(tenant_ctx),
            }),
        }
    }

    /// Tenant-aware variant of [`Self::with_pool`].
    ///
    /// Behaves like [`Self::with_pool`] for the pool-carve hot path
    /// (no extra counter traffic) and like
    /// [`Self::with_tenant_context`] for the fallback fresh-allocation
    /// path. Intended for deployments that want both the pool's
    /// amortised allocation cost and a per-tenant cap on the
    /// non-pooled overflow.
    pub fn with_pool_and_tenant_context(
        device_id: DeviceId,
        pool: Arc<UnifiedMemoryPool>,
        tenant_ctx: Arc<tensor_wasm_tenant::TenantContext>,
    ) -> Self {
        Self {
            inner: Arc::new(MemoryCreatorState {
                device_id,
                pool: Some(pool),
                tenant_ctx: Some(tenant_ctx),
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
                tenant_ctx: None,
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
        // Enforce HARD_MAX_LINEAR_MEMORY_BYTES on the cap declared by the
        // module's MemoryType BEFORE any backing allocation runs. Wasmtime's
        // ResourceLimiter only fires on `memory.grow`, not on initial
        // allocation, so without this guard a guest declaring
        // `(memory 1 65536)` would force a 4 GiB allocation here. See
        // [`HARD_MAX_LINEAR_MEMORY_BYTES`] for the closed mem-H5 /
        // exec-S-2 / exec-S-10 audit gap.
        if max > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(format!(
                "module-declared memory maximum {max} bytes exceeds hard cap {HARD_MAX_LINEAR_MEMORY_BYTES} bytes",
            ));
        }
        if minimum > HARD_MAX_LINEAR_MEMORY_BYTES {
            return Err(format!(
                "module-declared memory minimum {minimum} bytes exceeds hard cap {HARD_MAX_LINEAR_MEMORY_BYTES} bytes",
            ));
        }
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

        // Tenant-aware fresh allocation when a `TenantContext` was wired
        // in (roadmap feature #8). The Wasmtime `MemoryCreator` API
        // returns `Result<_, String>` so the structured
        // `GpuMemoryExhausted { requested, limit, current }` collapses
        // to a string here — non-Wasmtime callers that want the
        // structured form should reach for
        // `TensorWasmLinearMemory::new_on_with_tenant_context` directly.
        let mem = if let Some(tenant_ctx) = self.inner.tenant_ctx.as_ref() {
            TensorWasmLinearMemory::new_on_with_tenant_context(
                minimum,
                maximum,
                self.inner.device_id,
                Arc::clone(tenant_ctx),
            )
            .map_err(|e| e.to_string())?
        } else {
            TensorWasmLinearMemory::new_on(minimum, maximum, self.inner.device_id)
                .map_err(|e| e.to_string())?
        };
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
    fn maximum_above_hard_cap_rejected_without_allocating() {
        // 5 GiB > HARD_MAX_LINEAR_MEMORY_BYTES (4 GiB). Must fast-fail with
        // TooLarge BEFORE any backing allocation is attempted. The point is
        // to make sure a guest can't push us into a multi-GiB cudaMalloc
        // simply by declaring a giant maximum on its `(memory ...)`.
        let big = HARD_MAX_LINEAR_MEMORY_BYTES + 1;
        let err = TensorWasmLinearMemory::new(64 * 1024, Some(big)).unwrap_err();
        match err {
            UnifiedError::TooLarge { requested, limit } => {
                assert_eq!(requested, big as u64);
                assert_eq!(limit, HARD_MAX_LINEAR_MEMORY_BYTES as u64);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn minimum_above_hard_cap_rejected() {
        // A pathological module declaring an initial size above the hard
        // cap is rejected at the same gate, even if the maximum is None
        // (which would otherwise default to DEFAULT_MAX_BYTES, < cap).
        let big = HARD_MAX_LINEAR_MEMORY_BYTES + 1;
        let err = TensorWasmLinearMemory::new(big, Some(big + 1)).unwrap_err();
        assert!(matches!(err, UnifiedError::TooLarge { .. }));
    }

    #[test]
    fn maximum_exactly_at_hard_cap_admitted_but_not_allocated_in_test() {
        // We can't actually attempt the 4 GiB allocation here (CI hosts
        // don't have that much UVM / heap), so we only assert that the
        // CHECK at the cap boundary lets the request through and produces
        // an `Allocation` failure mode from the underlying allocator
        // rather than the early `TooLarge` rejection. The downstream
        // allocator may succeed on a beefy host; either outcome is fine
        // for this contract test.
        let at_cap = HARD_MAX_LINEAR_MEMORY_BYTES;
        // Use a tiny minimum so we exercise the cap check on `max` only.
        let result = TensorWasmLinearMemory::new(0, Some(at_cap));
        match result {
            Ok(_) => { /* well-resourced host */ }
            Err(UnifiedError::Allocation(_)) | Err(UnifiedError::Cuda(_)) => {
                // Allocator refused the 4 GiB request — fine.
            }
            Err(other) => panic!(
                "request at exactly the hard cap must not be rejected as TooLarge: got {other:?}"
            ),
        }
    }

    #[test]
    fn creator_rejects_module_max_above_hard_cap() {
        use wasmtime::MemoryCreator;
        let creator = TensorWasmMemoryCreator::default();
        let mt = wasmtime::MemoryType::new(1, None);
        let big = HARD_MAX_LINEAR_MEMORY_BYTES + 1;
        let err = creator
            .new_memory(mt, 64 * 1024, Some(big), None, 0)
            .expect_err("oversized module max must be refused");
        assert!(
            err.contains("hard cap"),
            "error must mention the hard cap; got: {err}"
        );
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
    fn grow_to_zero_fills_newly_exposed_bytes() {
        // Cross-tenant data-leak regression test (audit H2):
        // -------------------------------------------------------------------
        // The wasm spec requires bytes newly exposed by `memory.grow` to read
        // as zero. Because `TensorWasmLinearMemory` pre-allocates the entire
        // `maximum_size` up front and only bumps a `current_size`, the host
        // backing already contains *whatever was previously written* in the
        // `[current_size, max)` window. This test scribbles 0xCD into a band
        // that sits in the pre-allocated tail (past `current_size`), then
        // grows past that band and asserts every newly-exposed byte reads
        // zero — including the previously-poisoned range.
        const MIN: usize = 64 * 1024;
        const MAX: usize = 4 * 64 * 1024; // four wasm pages
        const POISON_START: usize = MIN + 4096;
        const POISON_END: usize = MIN + 8192;
        const GROW_TO: usize = MAX;

        let mut mem = TensorWasmLinearMemory::new(MIN, Some(MAX)).unwrap();

        // Poison a band sitting in the pre-allocated tail. This simulates a
        // stale write left over from a previous tenant or a host helper.
        // SAFETY: we hold `&mut mem` exclusively and no guest is executing.
        // The buffer's physical extent covers `MAX` bytes even though
        // `current_size == MIN`, so writing into the tail is in-bounds.
        unsafe {
            let p = LinearMemory::as_ptr(&mem);
            for off in POISON_START..POISON_END {
                *p.add(off) = 0xCD;
            }
        }

        // Grow past the poisoned region.
        mem.grow_to(GROW_TO).expect("grow must succeed");
        assert_eq!(mem.byte_size(), GROW_TO);

        // Every byte in `[MIN, GROW_TO)` — the range newly exposed to the
        // guest by `grow_to` — must read as zero. If the zero-fill is
        // missing, the 0xCD band leaks straight through.
        let s = mem.as_slice();
        assert_eq!(s.len(), GROW_TO);
        let leaked: Vec<usize> = s[MIN..GROW_TO]
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(MIN + i) } else { None })
            .collect();
        assert!(
            leaked.is_empty(),
            "grow_to leaked {} non-zero bytes in newly-exposed range \
             [{MIN}, {GROW_TO}); first leak at offset {:?} (value would be 0xCD if poison)",
            leaked.len(),
            leaked.first(),
        );
    }

    #[test]
    fn pooled_grow_to_zero_fills_newly_exposed_bytes() {
        // Mirror of `grow_to_zero_fills_newly_exposed_bytes` for the
        // pool-backed linear memory path (`PooledLinearMemory`). The
        // exposure surface is the same — `grow_to` only bumps a logical
        // size against a larger physical reservation — so the zero-fill
        // discipline must match.
        use std::sync::Arc;
        const MIN: usize = 64 * 1024;
        const MAX: usize = 4 * 64 * 1024;
        const POISON_START: usize = MIN + 4096;
        const POISON_END: usize = MIN + 8192;
        const GROW_TO: usize = MAX;

        let pool = Arc::new(crate::pool::UnifiedMemoryPool::new(8 * 1024 * 1024).unwrap());
        let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), pool.clone());
        let mt = wasmtime::MemoryType::new(1, Some(4));
        use wasmtime::MemoryCreator;
        let mut mem = creator
            .new_memory(mt, MIN, Some(MAX), None, 0)
            .expect("new_memory");

        // Poison the pre-allocated tail.
        // SAFETY: same rationale as the non-pooled test above; we hold
        // exclusive `&mut mem` and the physical region is `MAX` bytes.
        unsafe {
            let p = mem.as_ptr();
            for off in POISON_START..POISON_END {
                *p.add(off) = 0xCD;
            }
        }

        mem.grow_to(GROW_TO).expect("grow must succeed");
        assert_eq!(mem.byte_size(), GROW_TO);

        // Read back the entire region via `as_ptr` + raw slice (the pooled
        // path does not expose a private `as_slice`; that's only on
        // `TensorWasmLinearMemory`).
        // SAFETY: no guest is executing; `as_ptr()` points to `GROW_TO`
        // valid bytes inside the carved slab.
        let s = unsafe { std::slice::from_raw_parts(mem.as_ptr() as *const u8, GROW_TO) };
        let leaked: Vec<usize> = s[MIN..GROW_TO]
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(MIN + i) } else { None })
            .collect();
        assert!(
            leaked.is_empty(),
            "pooled grow_to leaked {} non-zero bytes in [{MIN}, {GROW_TO}); first at {:?}",
            leaked.len(),
            leaked.first(),
        );
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
