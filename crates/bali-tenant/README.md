# bali-tenant

Multi-tenant CUDA context management for Project Bali. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `mps` | no | Prefer NVIDIA MPS-backed shared contexts when `/tmp/nvidia-mps` is present. |
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime for context lifecycle tasks.
- `thiserror` — derive macro for tenant-level errors.
- `tracing` — structured spans/events for context create/destroy.
- `parking_lot` — fast mutexes guarding registry state.
- `dashmap` — concurrent map of `TenantId` to `TenantContext`.
- `cust` (optional) — CUDA driver-API bindings; only linked under `cuda`.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
