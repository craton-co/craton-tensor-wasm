# tensor-wasm-tenant

Multi-tenant CUDA context management for Craton TensorWasm. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `mps` | no | Prefer NVIDIA MPS-backed shared contexts when `/tmp/nvidia-mps` is present. |
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

Internal workspace crates:
- `tensor-wasm-core` — shared `TenantId` / `InstanceId` newtypes and the `TensorWasmError` enum returned by quota enforcement.

External crates (pinned at workspace root):
- `tokio` — async runtime for context lifecycle tasks.
- `thiserror` — derive macro for tenant-level errors.
- `tracing` — structured spans/events for context create/destroy and the underflow / pop-failure warnings.
- `parking_lot` — fast mutexes guarding registry state.
- `dashmap` — concurrent map of `TenantId` to `Arc<TenantContext>`.
- `cust` (optional, behind the `cuda` feature) — CUDA driver-API bindings; provides the primary-context API used by `ContextIsolated` tenants.
