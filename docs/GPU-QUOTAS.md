# GPU memory quotas

Per-tenant GPU memory caps for Craton TensorWasm. Roadmap feature #8.

**Driver-level enforcement: LANDED in v0.3.7 (T39), requires
`--features gpu-mem-pool`.** The in-process counter remains the
primary accounting source; the driver pin is the bypass-resistant
second line of defence (see "Security note" at the bottom of this
doc). On builds without `gpu-mem-pool`, the cap is **recorded and
enforced in-process only** — the v0.3.7 behaviour described in the
"v0.3.7 — record-only behaviour" section continues to apply.

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

## v0.4 — `cuMemPool` driver-level enforcement (T39, LANDED)

CUDA 11.2+ exposes `cuMemPool` APIs that let the driver enforce a
hard per-pool memory cap. T39 wires this through against cudarc 0.13.

### What landed

* `tensor-wasm-mem::cuda_mem_pool::TenantMemPool::new(device_ordinal,
  cap_bytes)` now calls
  `cuMemPoolCreate` followed by
  `cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
  &cap_bytes)`. The release threshold is the driver-side cap on
  outstanding allocations from this pool.
* The constructor uses the T26 per-ordinal device cache
  (`tensor-wasm-mem::cudarc_backend::DEVICE_CACHE`) to retain the
  primary context for the lifetime of the pool. Dropping the
  `TenantMemPool` calls `cuMemPoolDestroy`; the held
  `Arc<CudaDevice>` drops after the destroy so the primary context
  outlives the destroy call.
* `tensor-wasm-mem::unified::UnifiedBuffer::new_in_tenant_pool(pool,
  size, device_id)` routes through `cuMemAllocFromPoolAsync` on the
  null stream. The freed allocation goes through `cuMemFreeAsync` on
  drop. The driver enforces the cap — over-cap allocations fail with
  `CUDA_ERROR_OUT_OF_MEMORY`, surfaced as `UnifiedError::Cuda`.
* `TenantContext.mem_pool: Option<Arc<TenantMemPool>>` carries the
  pool handle through the tenant lifecycle. See
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

The driver cap is the bypass-resistant *additional* gate: a workload
that obtained a raw CUDA driver handle (out of scope for `wasi:cuda`,
in scope for any future trusted-tenant deployment) cannot allocate
past the pool cap even though it did not go through `consume_gpu_bytes`.

### TOCTOU caveat (driver-API limitation)

The threshold is set in a separate FFI call after the pool's
creation. A racing observer can see the unprotected pool for ~µs
between `cuMemPoolCreate` and `cuMemPoolSetAttribute`. Acceptable
because (a) the only consumer of the pool handle in this codebase is
the just-finished constructor, and (b) the in-process counter still
applies as a second line of defence. cudarc 0.13's `CUmemPoolProps`
struct doesn't carry the threshold inline, so this race is
unavoidable in CUDA Driver API.

### Operator alignment requirement

The DRIVER-level pin must STRICTLY MATCH the in-process cap value.
Pass the same `bytes` to BOTH
`TenantContextBuilder::with_gpu_memory_bytes_cap(bytes)` AND
`TenantContextBuilder::with_driver_enforced_gpu_cap(bytes)`. An
alignment failure between the two is the operator's bug, not ours;
the v0.4 builder does not auto-derive one from the other so the
distinction stays explicit and auditable.

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
