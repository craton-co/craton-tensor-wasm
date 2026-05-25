# tensor-wasm-mem

CUDA Unified Memory allocator for Craton TensorWasm, plus integration glue that lets Wasmtime back its linear memories with unified-memory pages directly visible to GPU kernels. Wraps `cudaMallocManaged` in a safe `UnifiedBuffer`, provides a bump-allocator pool, exposes `cudaMemAdvise` hint helpers (ReadMostly, PreferredLocation, AccessedBy), and implements Wasmtime's `MemoryCreator` / `LinearMemory` traits so guest memory is zero-copy reachable from the GPU.

## Feature flags

The default build is empty — `unified-memory` is opt-in so `cargo build` succeeds on hosts without a CUDA toolkit (the `cust` build script panics if CUDA libraries are missing). Enable it on CUDA hosts via `--features unified-memory`.

| Flag | Default | What it enables |
|---|---|---|
| `unified-memory` | no | Links `cust` and switches `UnifiedBuffer`'s backing from `Box<[u8]>` to `cudaMallocManaged`. This is also what makes `TensorWasmLinearMemory` (the wasm linear memory) UVM-backed — see "Zero-copy wasm linear memory" below. Required for the `cudaMemAdvise`/prefetch paths to actually call into the driver instead of being no-ops. |
| `cudarc-backend` | no | Links `cudarc` and switches `UnifiedBuffer`'s backing to the W1.2 `CudarcUnifiedBuffer` spike, which wraps `cuMemAllocManaged` via `cudarc::driver::sys::lib()`. The cust path wins if `unified-memory` is also enabled (see precedence table in `src/unified.rs`); enable this alone (or with `--no-default-features`) to exercise the cudarc backing. This is the v0.5 cust-successor fallback per [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md). |

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
(`true` whenever the wasm linear memory is UVM-backed, which now covers
both the cust and cudarc routings — see below).

**`--features cudarc-backend` (the v0.5 cust-successor path).** Building
with `--features cudarc-backend` (and `unified-memory` OFF) ALSO routes
the wasm linear memory through CUDA Unified Memory, via the W1.2
[`CudarcUnifiedBuffer`](src/cudarc_backend.rs) spike. That spike wraps
`cuMemAllocManaged` (and `cuMemFree_v2`, `cuMemAdvise`,
`cuMemPrefetchAsync`) through `cudarc::driver::sys::lib()` because
cudarc 0.13's safe surface does not yet wrap managed memory directly.
`UnifiedBuffer`'s `Backing` enum picks the third variant
`Backing::Cudarc(CudarcUnifiedBuffer)` for this combination, and
`is_uvm_backed()` returns `true` — so the same kernel-args zero-copy
property holds whether the build links cust or cudarc. The cust path
wins if both features are enabled simultaneously (preserving the v0.3
default for any host that already opted in); see the precedence table
in [`src/unified.rs`](src/unified.rs) for the full feature-combination
matrix. The W1.2 spike now graduates from "smoke-tested allocator" to
"real backing for the wasm linear memory" and is the fallback if
cuda-oxide v0.2 does not ship its managed-memory wrapper in time for
the v0.5 default-flip — see [RFC 0001](../../rfcs/0001-cuda-oxide-integration.md)
for the three-backend coexistence plan.

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
| `cudarc` (optional) | CUDA driver-API bindings used by the W1.2 `CudarcUnifiedBuffer` backing; only linked under `cudarc-backend`. |
