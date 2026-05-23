# bali-mem

CUDA Unified Memory allocator for Project Bali, plus integration glue that lets Wasmtime back its linear memories with unified memory pages directly visible to GPU kernels. Wraps `cudaMallocManaged` in a safe `UnifiedBuffer`, provides a bump-allocator pool, exposes `cudaMemAdvise` hint helpers (ReadMostly, PreferredLocation, AccessedBy), and implements Wasmtime's `MemoryCreator` / `LinearMemory` traits so guest memory is zero-copy reachable from the GPU.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `unified-memory` | yes | Link `cust` and provide `UnifiedBuffer` backed by `cudaMallocManaged`. Disable for non-CUDA hosts. |
| `pinned-host-memory` | no | Use page-locked host memory instead of unified memory. Mutually exclusive with `unified-memory`. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `thiserror` — derive macro for the crate's error enum.
- `tracing` — structured spans/events for allocator activity.
- `parking_lot` — fast mutexes guarding the bump pool and free lists.
- `cust` (optional) — CUDA driver-API bindings; only linked under `unified-memory`.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
