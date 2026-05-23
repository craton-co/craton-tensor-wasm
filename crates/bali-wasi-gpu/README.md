# bali-wasi-gpu

Host bridge implementing the `wasi-cuda` interface, giving Wasm guests an explicit GPU kernel launch API. Defines the `wasi_cuda_*` ABI, host-side implementations of every call, a `KernelRegistry` that caches compiled PTX modules keyed by `KernelId`, and an async dispatch layer with back-pressure to keep the GPU saturated without overwhelming it. This crate is the explicit-offload counterpart to bali-jit's implicit offload pipeline.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `cuda` | no | Link `cust` and provide real `wasi_cuda_*` host functions. Without this the host functions return `CudaError::NotAvailable`. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime hosting kernel dispatch tasks.
- `async-trait` — async methods on the host-function traits.
- `wasmtime` — exposes the host-function registration surface.
- `thiserror` — derive macro for the crate's error enum.
- `tracing` — structured spans/events for launch and back-pressure.
- `dashmap` — concurrent kernel cache keyed by `KernelId`.
- `cust` (optional) — CUDA driver-API bindings; only linked under `cuda`.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
