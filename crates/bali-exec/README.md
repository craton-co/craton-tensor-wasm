# bali-exec

Async execution engine for Project Bali, built on Wasmtime and Tokio. Provides the `BaliEngine` wrapper around `wasmtime::Engine` with a Bali-specific `Config`, the per-instance `BaliInstance` state machine, and the `BaliExecutor` responsible for spawning, invoking, and terminating Wasm instances on demand. This is the runtime hot path that ties together memory, GPU bridges, and the JIT pipeline.

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
- `thiserror` — derive macro for executor-level errors.
- `tracing` — structured spans/events for instance lifecycle.
- `parking_lot` — fast synchronisation primitives for shared state.
- `dashmap` — concurrent map keyed by `InstanceId`.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
