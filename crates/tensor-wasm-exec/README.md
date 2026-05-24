# tensor-wasm-exec

Async execution engine for Craton TensorWasm, built on Wasmtime and Tokio. Provides the `TensorWasmEngine` wrapper around `wasmtime::Engine` with a TensorWasm-specific `Config`, the per-instance `TensorWasmInstance` state machine, and the `TensorWasmExecutor` responsible for spawning, invoking, and terminating Wasm instances on demand. This is the runtime hot path that ties together memory, GPU bridges, and the JIT pipeline.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `async-execution` | yes | Enable Wasmtime async + epoch-based interrupt. Disabling produces a sync executor (development/debug only). |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime hosting the executor's task pool.
- `async-trait` — async methods on the public executor traits.
- `futures` — combinator primitives used in the dispatch layer.
- `wasmtime` — embedded Wasm engine driving instance execution.
- `wasmtime-wasi` — WASI preview implementations exposed to guests.
- `wasmparser` — pre-instantiation operator walk for the auto-offload analysis.
- `thiserror` — derive macro for executor-level errors.
- `tracing` — structured spans/events for instance lifecycle.
- `parking_lot` — fast synchronisation primitives for shared state.
- `dashmap` — concurrent map keyed by `InstanceId` (also backs the module cache).

Internal TensorWasm crate dependencies:
- `tensor-wasm-core` — shared `TensorWasmError`, `TensorWasmMetrics`, and identifier types (`TenantId`, `InstanceId`).
- `tensor-wasm-jit` — kernel cache + detector consumed by `auto_offload` and `jit_dispatch`.
- `tensor-wasm-mem` — `TensorWasmMemoryCreator` so linear memory is carved from CUDA Unified Memory on supported hosts.
