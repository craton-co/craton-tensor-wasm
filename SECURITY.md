# Project Bali — Security Model

_Status: living document. First written for S6 of the implementation plan.
This is the threat model and isolation strategy summary; the full security
audit lands in S21 (`docs/SECURITY-AUDIT.md`)._

## Threat model

Bali runs untrusted Wasm modules that issue explicit (and, post-S14, implicit)
GPU kernel launches. The adversary controls the Wasm bytecode and the kernel
PTX. We assume the host kernel, CUDA driver, and Wasmtime runtime are trusted.

### Assets to protect

1. **Tenant memory confidentiality** — Tenant A's linear memory must not be
   readable by Tenant B's Wasm or any kernel Tenant B launches.
2. **Tenant memory integrity** — Same as above for writes.
3. **Host process integrity** — A malicious Wasm or PTX must not crash, hang,
   or corrupt the host runtime.
4. **Host kernel/driver stability** — Malformed PTX or an out-of-bounds CUDA
   operation must be rejected by the driver, not crash it.
5. **Availability of co-located tenants** — One tenant's runaway workload
   should not starve another.

### Adversary capabilities

- Arbitrary Wasm bytecode (validated by Wasmtime).
- Arbitrary PTX uploaded via `wasi_cuda_load_ptx` (validated by `ptxas`).
- Crafted kernel launch parameters (grid, block, args).
- Crafted snapshot files (S15) submitted to `nova restore`.
- Crafted HTTP requests at the API gateway (S17).

### Out of scope (for v0.1.0)

- Side-channel attacks via the GPU L2 cache (see "Known gaps" below).
- Hardware faults injected by the adversary.
- Compromise of the underlying NVIDIA driver or kernel module.

## Defences

### Memory isolation

The Wasm linear memory is the only memory addressable by Wasm code, and it is
backed by a per-instance `UnifiedBuffer` allocated by `BaliMemoryCreator`
(`bali-mem/src/wasm_memory.rs`). Cross-instance memory access is prevented by:

1. **Wasmtime bounds checks.** Every Wasm load/store is bounds-checked at
   compile time by Cranelift against the declared linear-memory size. A
   guest cannot synthesise an address outside its own `BaliLinearMemory`
   without the access being trapped by Wasmtime before it reaches CUDA.
2. **Distinct allocations.** Every `BaliLinearMemory` owns a distinct
   `UnifiedBuffer` returned by a separate `cudaMallocManaged` call. There
   is no shared backing store between instances, so even a confused-deputy
   bug in the host couldn't cause one tenant's pointer to alias another's.
3. **Per-tenant CUDA streams and contexts** (`bali-exec` streams in S7,
   `bali-tenant` contexts in S16). Kernels submitted by different tenants
   execute on different streams (and, in `ContextIsolated` mode, different
   contexts), so an in-flight kernel cannot observe or corrupt a sibling
   tenant's launches.

We deliberately do **not** apply OS-level `mprotect`/`PROT_NONE` guard pages
around the `UnifiedBuffer`. `cudaMallocManaged` memory is page-migrated
between host and device by the CUDA driver, and applying host page
protection to those pages is unsupported and would race with the driver's
migration machinery. Hardware-level page protection across tenants is the
responsibility of NVIDIA MIG; see the "GPU L2 cache timing side channel"
gap below for the MIG/MPS deployment story.

### Kernel launch hardening

`wasi_cuda_load_ptx` validates PTX with `ptxas --gpu-name <arch>` before
caching the compiled module. `wasi_cuda_launch` clamps grid/block sizes to
device-reported maxima and validates argument pointers fall within the
caller's `UnifiedBuffer` range. Kernel timeouts (`KernelTimeout`, see
`bali-core/src/error.rs`) are enforced via CUDA events plus an epoch timer.

### CPU/IO time

Wasmtime epoch-based interruption (`bali-exec`, S7) terminates instances that
exceed their per-invocation deadline. The HTTP API gateway (S17) enforces
per-tenant request rate limiting via `tower_governor`.

### Error containment

`BaliError::TenantIsolationViolation` is raised when any of the above checks
fail. The instance is terminated, its `UnifiedBuffer` freed, and the event is
emitted as a tracing span (`bali_core::telemetry`) plus a metric increment
(`bali_offload_fallback_total` is *not* the right one — S21 adds a dedicated
`bali_isolation_violations_total` counter).

## IsolationLevel taxonomy

The `bali_mem::isolation::IsolationLevel` enum (added in this session) makes
the operator's choice explicit:

| Level | Streams | Contexts | Use case |
|---|---|---|---|
| `Shared` | shared | shared | Single-tenant, fully trusted (development) |
| `StreamIsolated` (default) | per-instance | shared | Multi-tenant, latency-sensitive |
| `ContextIsolated` | per-instance | per-tenant (MPS/MIG) | Multi-tenant, strong isolation |

## Known gaps

These are documented limitations of v0.1.0; mitigations are tracked.

### GPU L2 cache timing side channel

NVIDIA's L2 cache is shared between all SMs and is not partitioned by stream
or context. A malicious Wasm module that issues a kernel can in principle
measure cache hit/miss timing patterns that leak information about a
co-located tenant's workload. We do not currently mitigate this.

**Long-term mitigation:** deploy on NVIDIA Multi-Instance GPU (MIG) where
hardware partitions the L2; or use Hopper-class MPS isolation extensions.
See `docs/MPS-SETUP.md` (S16).

### Driver instability under adversarial PTX

`ptxas` validation reduces but does not eliminate the risk of a PTX module
hitting an NVIDIA driver bug. We mitigate by:

- Running `ptxas --verbose` and rejecting modules with warnings.
- Capping per-kernel resource usage (`__launch_bounds__`).
- Restarting the host process on `CUDA_ERROR_OS_INVALID` and similar.

A confirmed driver-crash trace from a PTX module is treated as a Critical
incident (see `docs/SECURITY-AUDIT.md`).

### Wasmtime compile-time fuzzing surface

S21 will fuzz the Wasm → Cranelift → host transition with `cargo-fuzz`. v0.1.0
ships with Wasmtime's upstream fuzz corpus only.

## Reporting vulnerabilities

Email `security@<project-host>` with a reproducer. We aim to triage within
2 business days. Coordinated disclosure preferred.
