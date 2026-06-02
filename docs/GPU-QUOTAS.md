# GPU memory quotas

Per-tenant GPU memory caps for Craton TensorWasm. Roadmap feature #8.

**Per-tenant pool-cap enforcement: LANDED (T39), requires
`--features gpu-mem-pool`.** Enforcement is **host-side**, inside
`TenantMemPool::allocate`, NOT driver-level — see the "Correction"
box below. The in-process counter remains the primary accounting
source; the pool cap is the bypass-resistant second line of defence
(see "Security note" at the bottom of this doc). On builds without
`gpu-mem-pool`, the cap is **recorded and enforced in-process only** —
the v0.3.7 behaviour described in the "v0.3.7 — record-only
behaviour" section continues to apply.

> **Correction (2026-05-30, `docs/GPU-VALIDATION-2026-05-30.md` BUG-1).**
> An earlier revision of this doc claimed the cap was enforced by the
> CUDA driver via `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
> cap)`. That is **wrong**: `RELEASE_THRESHOLD` is a memory-*retention*
> hint (how much freed memory the pool caches before returning it to the
> OS), not an allocation ceiling — and CUDA memory pools expose no hard
> max-size attribute. A hardware run confirmed a 128 MiB allocation
> against a 64 MiB-"capped" pool **succeeded**. The cap is now enforced
> host-side in `TenantMemPool::allocate` (a CAS counter over `live_bytes`
> vs `cap_bytes`, refused before `cuMemAllocFromPoolAsync`). The
> threshold is still set, but only for its real purpose (retention).

## Config knobs

The cap lives on `TenantContext` and is set at build time via
`TenantContextBuilder::with_gpu_memory_bytes_cap(bytes: u64)`.

| Setter | Type | Effect |
|---|---|---|
| `with_gpu_memory_bytes_cap(bytes)` | `u64` | Per-tenant GPU memory cap. Allocations that would push `gpu_bytes_in_use` above this value are refused with `TensorWasmError::GpuMemoryExhausted`. |
| _(default — no call)_ | — | `gpu_memory_bytes_cap == None`. The tenant's `gpu_bytes_in_use` counter is still maintained (so dashboards show real utilisation) but the allocator never refuses an allocation. This is the "operator-trust" mode appropriate for single-tenant deployments. |

Inspect at runtime via `TenantContext::gpu_memory_bytes_cap()` and
`TenantContext::gpu_bytes_in_use()`.

The allocator path that consults the cap is
`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context` (or
its pool-aware sibling `with_pool_and_tenant_context`). Wiring a
tenant context into the memory creator is what enables the cap —
constructions through the default `TensorWasmMemoryCreator::new` /
`with_pool` paths remain unmetered.

## v0.3.7 — record-only behaviour

`UnifiedBuffer::new_on_with_tenant_context`:

1. Calls `TenantContext::consume_gpu_bytes(size)`. On
   `Err(GpuMemoryExhausted)` returns the structured error untouched
   — no CUDA driver call happens on the rejection path.
2. On allocator success, stashes an `Arc<TenantContext>` on the
   resulting buffer.
3. The buffer's `Drop` calls `TenantContext::release_gpu_bytes(size)`.

The counter is a single `AtomicU64` mutated with the same CAS-loop
discipline as the CPU `bytes_in_use` counter — lock-free,
`checked_add` against overflow, `saturating_sub` on underflow. The
per-tenant series of
`tensor_wasm_core::metrics::TensorWasmMetrics::gpu_memory_bytes_per_tenant`
is updated on every successful transition when a metrics handle was
wired into the context via `TenantContextBuilder::with_metrics`.

**Pool-carved memories are intentionally unmetered.** Pool-backed
linear memories share one large slab allocation that was already paid
for at pool construction; double-counting each carve against the cap
would over-report utilisation. The pool's all-or-nothing teardown
contract (see `UnifiedMemoryPool::reset`) already serves as the
slab's lifecycle gate.

## `cuMemPool` per-tenant pool + host-side cap enforcement (T39, LANDED)

CUDA 11.2+ exposes `cuMemPool` APIs that give each tenant its own
allocation pool. T39 wires this through against cudarc 0.13 and enforces
the per-tenant cap **host-side** in `TenantMemPool::allocate`.

> **Why not the driver?** CUDA memory pools have no hard max-size
> attribute. `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` only controls how much
> freed memory the pool retains before returning it to the OS — it does
> not bound how much can be allocated. Relying on it as a cap let a
> 128 MiB allocation through a 64 MiB "cap" on real hardware (BUG-1). So
> the cap is a host-side admission check; the threshold is still set, but
> only for its actual retention purpose.

### What landed

* `tensor-wasm-mem::cuda_mem_pool::TenantMemPool::new(device_ordinal,
  cap_bytes)` calls `cuMemPoolCreate` followed by
  `cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
  &cap_bytes)` (retention hint), and records `cap_bytes` for the
  host-side check.
* `TenantMemPool::allocate(size)` reserves against the cap BEFORE the
  driver call: a CAS loop bumps `live_bytes` and refuses the allocation
  with a `CUDA_ERROR_OUT_OF_MEMORY`-shaped `UnifiedError::Cuda` if
  `live_bytes + size > cap_bytes`. `TenantPoolBacking::drop` calls
  `TenantMemPool::release_bytes(size)` to return the reservation. The
  arithmetic is unit-tested driver-free (`reserve_step_*` in
  `cuda_mem_pool.rs`); the end-to-end driver-reject test is in
  `tests/cuda_mem_pool_driver_pin.rs` (`#[ignore]`, hardware).
* The constructor uses the T26 per-ordinal device cache
  (`tensor-wasm-mem::cudarc_backend::DEVICE_CACHE`) to retain the
  primary context for the lifetime of the pool. Dropping the
  `TenantMemPool` calls `cuMemPoolDestroy`; the held
  `Arc<CudaDevice>` drops after the destroy so the primary context
  outlives the destroy call.
* `tensor-wasm-mem::unified::UnifiedBuffer::new_in_tenant_pool(pool,
  size, device_id)` routes through `cuMemAllocFromPoolAsync` on the
  null stream (after the host-side reservation passes). The freed
  allocation goes through `cuMemFreeAsync` on drop.
* `TenantContext.driver_mem_pool: Option<Arc<dyn DriverMemPool>>`
  carries the pool handle through the tenant lifecycle. See
  `TenantContextBuilder::with_driver_enforced_gpu_cap` for the
  builder entry point.

### What the in-process counter still does

The in-process `consume_gpu_bytes` / `release_gpu_bytes` pair remains
the always-correct accounting source of truth:

* It bumps the per-tenant Prometheus gauge
  (`tensor_wasm_gpu_memory_bytes_per_tenant`) on every transition.
* It surfaces the structured `TensorWasmError::GpuMemoryExhausted
  { requested, limit, current }` triple that the API layer maps to a
  4xx response body without scraping a driver error string.

The pool cap is the bypass-resistant *additional* gate: a workload that
allocates through the tenant pool (e.g. `UnifiedBuffer::new_in_tenant_pool`)
without going through `consume_gpu_bytes` still hits the host-side
reservation in `TenantMemPool::allocate` and cannot exceed `cap_bytes`.

> **Residual bypass (honest scope).** Because the cap is enforced
> host-side in `TenantMemPool::allocate`, a workload that obtained a raw
> CUDA driver handle and called `cuMemAlloc` / `cuMemAllocFromPoolAsync`
> *directly* — bypassing `TenantMemPool` entirely — is NOT capped. There
> is no driver-level pool ceiling to fall back on (see the correction
> box). This is acceptable under the current threat model: `wasi:cuda`
> guests cannot obtain raw driver handles. A future trusted-tenant
> deployment that hands out raw handles would need a different mechanism
> — a fixed-size VMM reservation pool (`cuMemCreate` + `cuMemAddressReserve`
> + `cuMemMap` sized to the cap), tracked as a v0.5 follow-up.

### Operator alignment requirement

The pool cap must STRICTLY MATCH the in-process cap value. Pass the same
`bytes` to BOTH `TenantContextBuilder::with_gpu_memory_bytes_cap(bytes)`
AND the `TenantMemPool` wired via
`TenantContextBuilder::with_driver_enforced_gpu_cap(pool)`. An alignment
failure between the two is the operator's bug, not ours; the builder does
not auto-derive one from the other so the distinction stays explicit and
auditable.

### Gating

`--features gpu-mem-pool` on `tensor-wasm-mem`. Strict-superset
alias for `--features cudarc-backend` (cust 0.3.x has no
`cuMemPool*` binding); the feature alias guarantees the resolver
picks up `cuda_mem_pool` and `UnifiedBuffer::new_in_tenant_pool`
together.

The metric series naming will also be revisited in v0.4: today the
CPU `consume_bytes` / `release_bytes` pair and the GPU
`consume_gpu_bytes` / `release_gpu_bytes` pair both write to
`gpu_memory_bytes_per_tenant` (last-write-wins). Splitting into
`gpu_memory_bytes_per_tenant` (GPU counter) and
`cpu_memory_bytes_per_tenant` (CPU counter) is a v0.4 dashboard /
alert-rule churn item.

## Security note

A tenant who somehow obtained direct access to the CUDA driver could
bypass the v0.3.7 in-process cap by calling `cuMemAlloc` /
`cuMemAllocManaged` directly — the counter is only updated by code
paths that go through `consume_gpu_bytes`. This is **not** a concern
for the `wasi:cuda` surface, where the host-side bridge is the only
way a guest can talk to the driver. It IS a concern for any future
"trusted-tenant" deployment that gives a tenant raw driver handles;
that deployment model is explicitly out of scope today. The
`gpu-mem-pool` enforcement narrows this gap for allocations routed
through `TenantMemPool` (the host-side reservation caps them
regardless of which code path called `allocate`), but does NOT close
it for a tenant calling the CUDA driver *directly* — see the
"Residual bypass" box above. Fully closing it requires a fixed-size
VMM reservation pool, a v0.5 follow-up.

## Cross-references

* `tensor-wasm-tenant`: `TenantContext::consume_gpu_bytes`,
  `TenantContext::release_gpu_bytes`,
  `TenantContextBuilder::with_gpu_memory_bytes_cap`.
* `tensor-wasm-mem`: `UnifiedBuffer::new_on_with_tenant_context`,
  `TensorWasmMemoryCreator::with_tenant_context`,
  `TensorWasmMemoryCreator::with_pool_and_tenant_context`,
  `TensorWasmLinearMemory::new_on_with_tenant_context`.
* `tensor-wasm-core`: `TensorWasmError::GpuMemoryExhausted`.
* Roadmap: [`PATH-TO-V1.md`](./PATH-TO-V1.md) — strategic features.
* Hardware validation: [`GPU-VALIDATION-2026-05-30.md`](./GPU-VALIDATION-2026-05-30.md)
  — BUG-1, the run that disproved the driver-level-cap claim.
* RFC: [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md)
  — the cust-successor migration.
