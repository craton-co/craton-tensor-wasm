# tensor-wasm-wasi-gpu

Host bridge implementing the `wasi-cuda` interface, giving Wasm guests an explicit GPU kernel launch API. Defines the `wasi_cuda_*` ABI, host-side implementations of every call, a `KernelRegistry` that caches compiled PTX modules keyed by `KernelId`, and an async dispatch layer with back-pressure to keep the GPU saturated without overwhelming it. This crate is the explicit-offload counterpart to tensor-wasm-jit's implicit offload pipeline.

The Component-Model interface (`wasi:cuda/host@0.2.0`) is defined in [`wit/wasi-cuda.wit`](./wit/wasi-cuda.wit) inside the crate — this in-crate copy is the authoritative source for `src/abi.rs` (`include_str!("../wit/wasi-cuda.wit")`) and is the version that ships in the published tarball. A mirror at the workspace-root [`wit/wasi-cuda.wit`](../../wit/wasi-cuda.wit) exists for cross-crate tooling; keep both in lockstep with `src/abi.rs`.

## Layout

| Path | Contents |
|---|---|
| `src/` | Host bridge implementation (`abi.rs`, `host.rs`, `registry.rs`, `async_dispatch.rs`). |
| `wit/wasi-cuda.wit` | Component-Model interface definition embedded by `abi.rs` via `include_str!`. |
| `tests/` | Integration tests: `wasi_gpu_smoke.rs`, `kernel_args_e2e.rs`. |
| `tests/fixtures/vector_add.ptx` | PTX kernel fixture embedded by the e2e tests via `include_bytes!`. Mirrors the workspace-root `kernels/vector_add.ptx`. |

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

Internal crate dependencies:
- `tensor-wasm-core` — shared core types/traits.
- `tensor-wasm-mem` — memory management primitives.
