# tensor-wasm-mem

CUDA Unified Memory allocator for Craton TensorWasm, plus integration glue that lets Wasmtime back its linear memories with unified-memory pages directly visible to GPU kernels. Wraps `cudaMallocManaged` in a safe `UnifiedBuffer`, provides a bump-allocator pool, exposes `cudaMemAdvise` hint helpers (ReadMostly, PreferredLocation, AccessedBy), and implements Wasmtime's `MemoryCreator` / `LinearMemory` traits so guest memory is zero-copy reachable from the GPU.

## Feature flags

The default build is empty — `unified-memory` is opt-in so `cargo build` succeeds on hosts without a CUDA toolkit (the `cust` build script panics if CUDA libraries are missing). Enable it on CUDA hosts via `--features unified-memory`.

| Flag | Default | What it enables |
|---|---|---|
| `unified-memory` | no | Links `cust` and switches `UnifiedBuffer`'s backing from `Box<[u8]>` to `cudaMallocManaged`. This is also what makes `TensorWasmLinearMemory` (the wasm linear memory) UVM-backed — see "Zero-copy wasm linear memory" below. Required for the `cudaMemAdvise`/prefetch paths to actually call into the driver instead of being no-ops. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Zero-copy wasm linear memory (the UVM wiring)

Under `--features unified-memory`, the wasm linear memory itself is
allocated in CUDA Unified Memory via `cuMemAllocManaged` (through the
`cust` crate). `TensorWasmLinearMemory::new` constructs a
[`UnifiedBuffer`](src/unified.rs) whose feature-gated `Backing` enum
selects `Backing::Cuda(cust::memory::UnifiedBuffer<u8>)` on CUDA hosts
and `Backing::Host(Box<[u8]>)` everywhere else. The wasmtime
`LinearMemory::as_ptr` accessor returns the raw UVM pointer directly —
no shim, no host-staging buffer — so a guest pointer produced by the
W1.1 wasi-cuda kernel-args pipeline (see
[`crates/tensor-wasm-wasi-gpu/src/kernel_args.rs`](../tensor-wasm-wasi-gpu/src/kernel_args.rs))
resolves to a host pointer that **doubles as a device pointer**. That
is the zero-copy property the v0.3.2 audit demanded and that
[`docs/RISKS.md`](../../docs/RISKS.md) advertises: kernels can read and
write the same bytes the guest sees, without `cudaMemcpy`. Callers can
probe the property at runtime via `TensorWasmLinearMemory::is_uvm_backed()`
(`true` only when built with `--features unified-memory`).

**Memory growth.** `cuMemAllocManaged` allocations are fixed-size, so
`TensorWasmLinearMemory` pre-allocates the declared `maximum_bytes` (or
`DEFAULT_MAX_BYTES`, 256 MiB) at construction time and treats
`LinearMemory::grow_to` as a logical-size bump up to that cap. This
matches Wasmtime's `static` memory model, keeps the kernel-side pointer
stable across growth events, and keeps the hot path zero-copy at the
cost of reserving the worst-case footprint up front. An in-place grow
that actually re-allocates and copies bytes is a v0.4 follow-up.

**Pool-backed memories.** `TensorWasmMemoryCreator::with_pool` still
carves linear memories from a `UnifiedMemoryPool` slab. The slab itself
is a `UnifiedBuffer`, so pool-backed memories share the same UVM
guarantee under `--features unified-memory`; the carving path just adds
amortised allocation. The pool API does not currently compose with a
parallel "UVM grow" path — pool slabs are pre-sized, full stop.

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
