# bali-jit

Just-in-time compilation pipeline that opportunistically offloads hot Wasm code to the GPU. Uses a Cranelift CLIF block analyser to detect offload candidates, lowers them through `BaliIR` (a normalised intermediate representation), emits PTX text, and caches compiled modules in an LRU. A `DeoptGuard` provides a safe CPU fallback path whenever GPU offload fails or guards trip at runtime.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `auto-offload` | no | Enable the Cranelift to BaliIR to PTX JIT pipeline. Off by default; enable when targeting CUDA hosts. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `thiserror` — derive macro for the JIT pipeline's error enum.
- `tracing` — structured spans/events for compilation and deopt.
- `dashmap` — concurrent map for offload-candidate metadata.
- `lru` — bounded cache of compiled PTX modules.
- `cranelift-codegen` — CLIF block analysis driving offload detection.
- `ptx-builder` (optional) — PTX emission backend; only linked under `auto-offload`.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
