# tensor-wasm-mem

CUDA Unified Memory allocator for Craton TensorWasm, plus integration glue that lets Wasmtime back its linear memories with unified-memory pages directly visible to GPU kernels. Wraps `cudaMallocManaged` in a safe `UnifiedBuffer`, provides a bump-allocator pool, exposes `cudaMemAdvise` hint helpers (ReadMostly, PreferredLocation, AccessedBy), and implements Wasmtime's `MemoryCreator` / `LinearMemory` traits so guest memory is zero-copy reachable from the GPU.

## Feature flags

The default build is empty — `unified-memory` is opt-in so `cargo build` succeeds on hosts without a CUDA toolkit (the `cust` build script panics if CUDA libraries are missing). Enable it on CUDA hosts via `--features unified-memory`.

| Flag | Default | What it enables |
|---|---|---|
| `unified-memory` | no | Links `cust` and switches `UnifiedBuffer`'s backing from `Box<[u8]>` to `cudaMallocManaged`. Required for the `cudaMemAdvise`/prefetch paths to actually call into the driver instead of being no-ops. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

| Crate | Purpose |
|---|---|
| `anyhow` | Error return type for the `wasmtime::LinearMemory` impl, which expects `anyhow::Result`. |
| `tensor-wasm-core` | Workspace error type — `tensor_wasm_core::error::TensorWasmError` — that `UnifiedError` converts into via `From`. |
| `parking_lot` | Fast `Mutex` guarding the bump-pool state. |
| `region` | Cross-platform anonymous mapping + `PROT_NONE` guard pages for `GuardedHostBuffer`. |
| `thiserror` | Derive macro for the `UnifiedError` enum. |
| `tracing` | Structured warnings/debug events for pool exhaustion and device-id mismatches. |
| `wasmtime` | `MemoryCreator` / `LinearMemory` traits implemented by `TensorWasmMemoryCreator` and `TensorWasmLinearMemory`. |
| `cust` (optional) | CUDA driver-API bindings; only linked under `unified-memory`. |
