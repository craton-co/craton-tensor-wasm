# Craton TensorWasm — Security Model

_Status: living document. First written for S6 of the implementation plan.
This is the threat model and isolation strategy summary; the full security
audit lands in S21 (`docs/SECURITY-AUDIT.md`)._

## Threat model

Craton TensorWasm runs untrusted Wasm modules that issue explicit (and, post-S14, implicit)
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
- Crafted snapshot files (S15) submitted to `tensor-wasm restore`.
- Crafted HTTP requests at the API gateway (S17).

### Out of scope (for v0.1.0)

- Side-channel attacks via the GPU L2 cache (see "Known gaps" below).
- Hardware faults injected by the adversary.
- Compromise of the underlying NVIDIA driver or kernel module.

## Defences

### Memory isolation

The Wasm linear memory is the only memory addressable by Wasm code, and it is
backed by a per-instance `UnifiedBuffer` allocated by `TensorWasmMemoryCreator`
(`tensor-wasm-mem/src/wasm_memory.rs`). Cross-instance memory access is prevented by:

1. **Wasmtime bounds checks.** Every Wasm load/store is bounds-checked at
   compile time by Cranelift against the declared linear-memory size. A
   guest cannot synthesise an address outside its own `TensorWasmLinearMemory`
   without the access being trapped by Wasmtime before it reaches CUDA.
2. **Distinct allocations.** Every `TensorWasmLinearMemory` owns a distinct
   `UnifiedBuffer` returned by a separate `cudaMallocManaged` call. There
   is no shared backing store between instances, so even a confused-deputy
   bug in the host couldn't cause one tenant's pointer to alias another's.
3. **Per-tenant CUDA streams and contexts** (`tensor-wasm-exec` streams in S7,
   `tensor-wasm-tenant` contexts in S16). Kernels submitted by different tenants
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

The non-managed `PinnedHostBuffer` (`tensor-wasm-mem/src/pinned_host.rs`), used on
the `--no-default-features` host fallback path, **is** bracketed by OS-level
guard pages. Each allocation reserves `[PROT_NONE | usable | PROT_NONE]`
via the cross-platform `region` crate (`mprotect` on Linux/macOS,
`VirtualProtect(PAGE_NOACCESS)` on Windows). Out-of-bounds reads or writes
that miss Wasmtime's bounds checks raise a hardware fault before they can
corrupt adjacent allocations. This applies only to the host-backed buffer;
managed memory remains unguarded for the reason above.

### Kernel launch hardening

`wasi_cuda_load_ptx` validates PTX with `ptxas --gpu-name <arch>` before
caching the compiled module. `wasi_cuda_launch` clamps grid/block sizes to
device-reported maxima and validates argument pointers fall within the
caller's `UnifiedBuffer` range. Kernel timeouts (`KernelTimeout`, see
`tensor-wasm-core/src/error.rs`) are enforced via CUDA events plus an epoch timer.

### CPU/IO time

Wasmtime epoch-based interruption (`tensor-wasm-exec`, S7) terminates instances that
exceed their per-invocation deadline. The HTTP API gateway (S17) enforces
per-tenant request rate limiting via `tower_governor`.

### Error containment

`TensorWasmError::TenantIsolationViolation` is raised when any of the above checks
fail. The instance is terminated, its `UnifiedBuffer` freed, and the event is
emitted as a tracing span (`tensor_wasm_core::telemetry`) plus a metric increment
(`tensor_wasm_offload_fallback_total` is *not* the right one — S21 adds a dedicated
`tensor_wasm_isolation_violations_total` counter).

## IsolationLevel taxonomy

The `tensor_wasm_mem::isolation::IsolationLevel` enum (added in this session) makes
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

## Supported versions

Only the `0.1.x` line receives security fixes during the preview window.
Older pre-release tags are not supported. When `0.2.0` ships, this matrix
will be revised; until then the table below applies.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Authentication

`tensor-wasm-api` supports bearer-token authentication via the `TENSOR_WASM_API_TOKENS`
environment variable (a comma-separated list of accepted tokens). When set,
all `/functions/*` and `/jobs/*` routes require `Authorization: Bearer <token>`;
`/healthz` and `/metrics` remain unauthenticated. See
[`crates/tensor-wasm-api/API.md`](crates/tensor-wasm-api/API.md) for the wire details.

## Reporting vulnerabilities

Email `security@craton.com.ar` with a reproducer. We aim to triage within
2 business days. Coordinated disclosure preferred.

## Backport policy

This section documents how Craton TensorWasm maintains released major
versions after the next major has shipped. It is the operational
companion to the v1.0 exit criterion in
[`docs/PATH-TO-V1.md`](docs/PATH-TO-V1.md) ("Backport policy") and to
the Security committee role defined in [`GOVERNANCE.md`](GOVERNANCE.md).

### Window

Every released major version (v1.x, v2.x, ...) receives security
patches and severity-1 fixes for **12 months** after the next major's
GA. The v1.0 backport window therefore opens at v1.0 GA and runs
until 12 months after v2.0 GA. The same 12-month rule rolls forward
for every subsequent major: when v3.0 ships, v2.x enters a 12-month
sunset window, and so on.

### What backports cover

- Security patches (CVE-fixed or embargoed-fix releases handled by
  the Security committee under the disclosure process above).
- Severity-1 fixes, where "severity-1" means one of:
  - Data loss (snapshot corruption, tenant memory overwrite,
    silently dropped writes).
  - A security regression that re-opens a previously closed CVE or
    weakens a documented isolation boundary.
  - Complete service outage in a documented supported configuration
    (the runtime fails to start, panics on first request, or hangs
    indefinitely under expected load).

### What backports do NOT cover

- Feature requests, including ergonomic improvements and new CLI
  flags. New features land on the latest minor only.
- Performance regressions of less than 2x baseline. Larger
  regressions are evaluated case-by-case; the burden of proof is on
  the requester to demonstrate the regression on the maintenance
  branch with a reproducer.
- Cosmetic, docs-only, or refactor changes. Those go in the latest
  minor; the maintenance branch is for fixes that change behaviour
  the user is depending on, not for tidying.

### Branch model

A `release-v1.x` maintenance branch tracks the latest patch of the
v1 line; `release-v2.x` will track v2, and so on. Patches land on
`main` first and are then cherry-picked to the maintenance branch
by a maintainer in the relevant crate area. Direct commits to a
maintenance branch are reserved for the rare case where the fix
shape on the older branch genuinely differs from `main` (for
example, because an API was renamed in a later major); in those
cases, the branch-specific patch must reference the `main` PR that
fixed the same issue in the trunk.

### How to request a backport

- **For security issues:** use the disclosure process under
  [Reporting vulnerabilities](#reporting-vulnerabilities) above. The
  Security committee will explicitly mark backport-eligible fixes
  during triage and coordinate the cherry-pick alongside the public
  release on disclosure day, per the embargoed-handling rules in
  [`GOVERNANCE.md`](GOVERNANCE.md).
- **For severity-1 fixes:** open a public issue tagged
  `backport-request` that includes the affected version, the crash
  signature or symptom, and a minimal reproducer. The maintainers
  triage backport requests during normal review; if accepted, the
  fix is cherry-picked from `main` once it has landed there.

### Release cadence on maintenance branches

Patch releases on maintenance branches are cut as needed, with no
fixed schedule. The maintainers commit to a release within **14
days** of a confirmed backport-eligible merge to `main`; security
releases follow the disclosure-day timing set by the Security
committee rather than the 14-day window. Multiple eligible fixes
within the same 14-day window are bundled into one patch release.

### Communicating EOL

90 days before a maintenance branch reaches end-of-life, an EOL
notice is posted in [`CHANGELOG.md`](CHANGELOG.md) under the next
patch release's entry and is repeated in the release notes for that
patch. The notice names the EOL date, the recommended target major
to upgrade to, and a pointer to the relevant migration guide
(`docs/MIGRATION-vN-to-vM.md`). On the EOL date itself, a final
patch release may be cut to ship any in-flight backports; after
that, the branch is archived and receives no further commits.

### LTS / extended window

v1.x is **not** marked LTS. The 12-month window above is the full
commitment. At v2.0, the project will revisit whether to offer an
extended LTS window (24 months has been discussed) based on
design-partner feedback gathered during the v0.5-beta and v1.x
cycles. This is open decision 7 in
[`docs/PATH-TO-V1.md`](docs/PATH-TO-V1.md); the decision will be
recorded as an RFC under [`rfcs/`](rfcs/) before v2.0 GA and, if
accepted, reflected by an amendment to this section.

### CVEs that affect only a maintenance branch

A vulnerability that affects only `release-v1.x` (for example, a
fix that landed on `main` happens to also close a CVE that never
existed in the older branch's code, or a CVE specific to a
deprecated code path retained on the maintenance branch) is
reported through the same disclosure address as any other CVE. The
Security committee replies within 72 hours and treats branch-only
CVEs identically to main-line ones: same triage SLO, same 90-day
fix or workaround commitment, same coordinated-disclosure
preference. The advisory explicitly names which branches are
affected so users on unaffected lines are not asked to upgrade
unnecessarily.
