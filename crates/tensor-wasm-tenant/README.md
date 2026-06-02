# tensor-wasm-tenant

Multi-tenant CUDA context management for Craton TensorWasm. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |
| `loom` | no | Swap `std::sync::atomic::AtomicU64` for `loom::sync::atomic::AtomicU64` on the `consume_bytes` / `release_bytes` CAS hot path so `tests/loom_consume_release.rs` can exhaustively explore the two-thread interleavings. Pure model-checking — do not enable in production builds. |
| `strict-cap-binding` | **yes** | Gates the typed `*_strict` admin error variants (e.g. `RegistryError::CapabilityFromForeignRegistry`) and the `cap_binding_strict` integration test. **It no longer toggles any security behaviour** — capability-to-registry binding is now unconditional (see [Cap binding](#cap-binding)). Kept in `default` so the `*_strict` surface is available out of the box. |

NVIDIA MPS detection is unconditional — there is no feature flag for it. At runtime `TenantRegistry::mps_or_fallback` probes for the MPS control daemon (honouring `$CUDA_MPS_PIPE_DIRECTORY`, else `/tmp/nvidia-mps`) and returns `MpsDecision::Mps` when present, falling back to per-tenant contexts otherwise.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Cap binding

`TenantRegistry::new()` mints a `RegistryAdminCapability`; every successful `register_with_capability` call mints a per-tenant `TenantCapability`. Both gate the hot-path mutation methods (admin enumeration / eviction for the former, per-tenant `consume_bytes_with_capability` / `release_bytes_with_capability` for the latter).

**Binding is unconditional — capabilities are always registry-scoped.** Every cap carries an `Arc<()>` token cloned from the minting registry's per-instance allocation, and every admin / quota check compares with `Arc::ptr_eq`. A cap minted by registry A is therefore rejected by registry B even when both name the same `TenantId`, and forged/foreign admin caps are refused — whether or not the `strict-cap-binding` feature is enabled, including in a release build with `default-features = false`. Two independent registries running in the same process are cleanly separated by construction; embedders no longer have to keep registry handles per-trust-domain to get isolation. Foreign-cap quota calls surface as [`TensorWasmError::TenantIsolationViolation`].

The `strict-cap-binding` feature (on by default) no longer toggles this security behaviour. It only gates the typed `*_strict` admin variants, which surface a foreign cap as [`RegistryError::CapabilityFromForeignRegistry`] rather than the generic refusal, plus the `cap_binding_strict` integration test.

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
