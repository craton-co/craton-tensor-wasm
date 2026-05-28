# tensor-wasm-tenant

Multi-tenant CUDA context management for Craton TensorWasm. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `mps` | no | Prefer NVIDIA MPS-backed shared contexts when `/tmp/nvidia-mps` is present. |
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Per-tenant quotas

`TenantContext` carries two independent quotas, each set through a
chained `TenantContextBuilder` method and each enforced by a
lock-free CAS-loop counter on the context. CPU and GPU caps are
separate so a host-RAM exhaustion (e.g. a runaway Wasm instance
allocating its linear memory) cannot share a budget with a GPU
exhaustion (e.g. an over-eager `wasi:cuda` allocation).

| Setter | Counter | Refusal error | Default | Enforcement |
|---|---|---|---|---|
| `with_memory_quota_bytes(bytes)` | `bytes_in_use` (CPU / host-side) | `TensorWasmError::MemoryExhausted` | 8 GiB (`DEFAULT_QUOTA_BYTES`) | In-process. Every `consume_bytes_with_capability` checked against this cap; rejection leaves the counter unchanged. |
| `with_gpu_memory_bytes_cap(bytes)` | `gpu_bytes_in_use` (GPU-side) | `TensorWasmError::GpuMemoryExhausted` | `None` (operator trust) | In-process today (v0.3.7); v0.4 also pins a driver-level cap via `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`. See [`docs/GPU-QUOTAS.md`](../../docs/GPU-QUOTAS.md). |
| `with_recorded_cuda_mem_pool_quota(bytes)` | _(none)_ | _(none — value is informational)_ | `None` | Recorded only. The cust 0.3.x crate does not surface the `cuMemPool*` API; the driver never sees this number. Forward-compat hook for the v0.4 cust-successor migration. |

The GPU side of the quota is consumed and released through the
allocator path in `tensor-wasm-mem` —
`TensorWasmMemoryCreator::with_tenant_context` /
`with_pool_and_tenant_context` route every fresh `UnifiedBuffer`
through `consume_gpu_bytes` (on allocation) and `release_gpu_bytes`
(on `Drop`). Pool-carved memories share one slab allocation and are
intentionally unmetered; see `docs/GPU-QUOTAS.md` for the rationale.

## Dependencies

Internal workspace crates:
- `tensor-wasm-core` — shared `TenantId` / `InstanceId` newtypes and the `TensorWasmError` enum returned by quota enforcement.

External crates (pinned at workspace root):
- `tokio` — async runtime for context lifecycle tasks.
- `thiserror` — derive macro for tenant-level errors.
- `tracing` — structured spans/events for context create/destroy and the underflow / pop-failure warnings.
- `dashmap` — concurrent map of `TenantId` to `Arc<TenantContext>`; combined with the `AtomicU64` counters on `TenantContext`, this is the entirety of the registry's concurrency story — there are no mutexes on the hot path.
- `cust` (optional, behind the `cuda` feature) — CUDA driver-API bindings; provides the primary-context API used by `ContextIsolated` tenants.
