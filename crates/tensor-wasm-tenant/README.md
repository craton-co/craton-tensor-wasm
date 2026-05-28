# tensor-wasm-tenant

Multi-tenant CUDA context management for Craton TensorWasm. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `mps` | no | Prefer NVIDIA MPS-backed shared contexts when `/tmp/nvidia-mps` is present. |
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |
| `loom` | no | Swap `std::sync::atomic::AtomicU64` for `loom::sync::atomic::AtomicU64` on the `consume_bytes` / `release_bytes` CAS hot path so `tests/loom_consume_release.rs` can exhaustively explore the two-thread interleavings. Pure model-checking — do not enable in production builds. |
| `strict-cap-binding` | no | Bind `RegistryAdminCapability` and `TenantCapability` to the `TenantRegistry` that minted them. With the flag off, caps are an opaque "you-hold-*some*-cap" token and a cap minted by registry A is accepted by registry B (the v0.3 default). With the flag on, foreign caps are rejected at the cap-check site. See [Cap binding](#cap-binding) below. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Cap binding

`TenantRegistry::new()` mints a `RegistryAdminCapability`; every successful `register_with_capability` call mints a per-tenant `TenantCapability`. Both gate the hot-path mutation methods (admin enumeration / eviction for the former, per-tenant `consume_bytes_with_capability` / `release_bytes_with_capability` for the latter). The default mode and the strict mode differ only in how those caps relate to the *specific* registry instance they came from.

**Default mode (no `strict-cap-binding`):** capabilities are opaque tokens. A cap minted by registry A is accepted by registry B. Sufficient to prevent unauthenticated callers from enumerating tenants, but **not** sufficient to separate two independent registries running in the same process. Embedders that host more than one `TenantRegistry` are responsible for keeping the registry handles per-trust-domain.

**Strict mode (`--features strict-cap-binding`):** every cap carries an `Arc<()>` token cloned from the minting registry's per-instance allocation. The admin / quota check compares with `Arc::ptr_eq`. Foreign-cap admin calls surface as [`RegistryError::CapabilityFromForeignRegistry`] from the `*_strict` admin variants. Foreign-cap quota calls surface as [`TensorWasmError::TenantIsolationViolation`]. Recommended for multi-tenant deployments; v0.4 will flip this on by default.

## Per-tenant quotas

`TenantContext` carries two independent quotas, each set through a chained `TenantContextBuilder` method and each enforced by a lock-free CAS-loop counter on the context. CPU and GPU caps are separate so a host-RAM exhaustion cannot share a budget with a GPU exhaustion.

| Setter | Counter | Refusal error | Default | Enforcement |
|---|---|---|---|---|
| `with_memory_quota_bytes(bytes)` | `bytes_in_use` (CPU / host-side) | `TensorWasmError::MemoryExhausted` | 8 GiB (`DEFAULT_QUOTA_BYTES`) | In-process. |
| `with_gpu_memory_bytes_cap(bytes)` | `gpu_bytes_in_use` (GPU-side) | `TensorWasmError::GpuMemoryExhausted` | `None` (operator trust) | In-process today (v0.3.7); v0.4 also pins a driver-level cap via `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`. See [`docs/GPU-QUOTAS.md`](../../docs/GPU-QUOTAS.md). |
| `with_recorded_cuda_mem_pool_quota(bytes)` | _(none)_ | _(none — informational)_ | `None` | Recorded only; forward-compat hook for the v0.4 cust-successor migration. |

The GPU side of the quota is consumed and released through the allocator path in `tensor-wasm-mem` — `TensorWasmMemoryCreator::with_tenant_context` / `with_pool_and_tenant_context` route every fresh `UnifiedBuffer` through `consume_gpu_bytes` (on allocation) and `release_gpu_bytes` (on `Drop`). Pool-carved memories share one slab allocation and are intentionally unmetered; see `docs/GPU-QUOTAS.md` for the rationale.

## Dependencies

Internal workspace crates:
- `tensor-wasm-core` — shared `TenantId` / `InstanceId` newtypes and the `TensorWasmError` enum returned by quota enforcement.

External crates (pinned at workspace root):
- `tokio` — async runtime for context lifecycle tasks.
- `thiserror` — derive macro for tenant-level errors.
- `tracing` — structured spans/events for context create/destroy and the underflow / pop-failure warnings.
- `dashmap` — concurrent map of `TenantId` to `Arc<TenantContext>`; combined with the `AtomicU64` counters on `TenantContext`, this is the entirety of the registry's concurrency story — there are no mutexes on the hot path.
- `cust` (optional, behind the `cuda` feature) — CUDA driver-API bindings; provides the primary-context API used by `ContextIsolated` tenants.
