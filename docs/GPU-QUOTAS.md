# GPU memory quotas

Per-tenant GPU memory caps for Craton TensorWasm. Roadmap feature #8.
Today (v0.3.7) the cap is **recorded and enforced in-process only**;
the CUDA driver itself sees no cap until v0.4 wires `cuMemPool`.

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

## v0.4 — `cuMemPool` enforcement plan

CUDA 11.2+ exposes `cuMemPool` APIs that let the driver enforce a
hard per-pool memory cap. The plan:

* At `TenantContext` construction (when `gpu_memory_bytes_cap` is
  `Some`), call
  `cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &bytes)`
  on a tenant-owned pool. The release threshold backs the driver-side
  cap on outstanding allocations.
* Migrate the in-process `consume_gpu_bytes` / `release_gpu_bytes`
  pair into a driver-call thin wrapper, surfacing the
  `CUDA_ERROR_OUT_OF_MEMORY` driver return as the existing
  `TensorWasmError::GpuMemoryExhausted`.
* Keep the in-process counter as the always-correct accounting source
  of truth; the driver cap is the bypass-resistant *additional* gate.

This is gated on the cust → cudarc / cuda-oxide migration documented
in [RFC 0001](../rfcs/0001-cuda-oxide-integration.md) — `cust 0.3.x`
does not surface the `cuMemPool*` API.

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
that deployment model is explicitly out of scope today. The v0.4
`cuMemPool` enforcement closes this gap at the driver level: the
release-threshold attribute is bound to the *pool*, so any
allocation against the pool (regardless of how the call reached the
driver) is capped.

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
* RFC: [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md)
  — the cust-successor migration that unblocks the v0.4
  `cuMemPoolSetAttribute` wire-up.
