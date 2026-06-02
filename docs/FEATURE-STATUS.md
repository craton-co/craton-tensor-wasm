<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Feature-Status Matrix — Canonical Source of Truth

This file is the **single canonical status source** for every major
TensorWasm feature: whether it is wired, scaffold-only, hardware-gated,
or planned, and behind which Cargo feature flag (if any). It exists to
end the scaffold-vs-wired drift between `README.md`, `CHANGELOG.md`, and
`docs/OPENAI-COMPAT.md`.

**Workspace version: 0.3.7** (root `Cargo.toml`).

When this table and any prose elsewhere disagree, **this table wins**.
`README.md`, `CHANGELOG.md`, and the per-feature spec docs defer to it
for status; they may add narrative, but the Status column here is
authoritative.

## Status vocabulary

| Term | Meaning |
|---|---|
| **Wired** | Reachable and functional on the default build or a documented feature flag — works end-to-end through the invoke / HTTP / store path it claims. |
| **Landed** | Implementation present and exercised, but the *production-relevant* verification (typically the GPU path) is `#[ignore]`d pending hardware; host-side behaviour runs today. |
| **Scaffold** | Compiles, surface-area-stable types + tests exist, but the call site returns a documented not-yet-wired sentinel (e.g. `NotYetWired` / `NotYetImplemented` / `FEATURE_NOT_EXPOSED`). |
| **Hardware-gated** | Code is written and reviewed but has never executed against real NVIDIA silicon; CI links the CUDA stubs and the proving test is `#[ignore = "requires CUDA hardware"]`. See [`HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md). |
| **Proven on hardware** | The path has actually executed against real NVIDIA silicon and produced verified-correct output. The proving test is still `#[ignore = "requires CUDA hardware"]` (so it is skipped in GPU-less CI and needs a GPU runner / local `--features cuda --include-ignored` run) — `#[ignore]` here means "CI has no GPU", **not** "unproven". |
| **Planned-v0.4** | Not yet implemented; tracked for v0.4+. |

Scaffold is deliberately *not* called "Wired": a green default-build test
does **not** prove the driver call works. A path is only marked **Proven on
hardware** once it has actually run on a GPU; an unrun CUDA path stays
**Hardware-gated**. The end-to-end GPU launch path (`kernel_args_e2e` /
`vector_add_end_to_end_real_ptx_real_kernel`) is **Proven on hardware** —
this is the same claim `presentation.md` makes ("Proven on real silicon"),
and this table now agrees with it.

## Matrix

| Feature | Crate(s) | Status | Cargo feature flag | Notes / Tracking-ID |
|---|---|---|---|---|
| Typed multi-value guest args | `tensor-wasm-exec`, `tensor-wasm-cli`, `tensor-wasm-api` | Wired | none (default) | T33. `WasmArg` enum + JSON↔`Val` codec; `--args <JSON>` plumbed end-to-end through CLI / HTTP `/invoke{,-async,-stream}` / `SpawnConfig::with_args`. `call_export_with_args` supersedes deprecated `call_export`. |
| `/invoke-stream` SSE streaming | `tensor-wasm-wasi-gpu`, `tensor-wasm-api` | Wired | none (default) | T34. Guest `wasi:tensor/host.emit-chunk` calls surface as SSE `event: chunk` frames via `StreamingContext`. Honors T36 cooperative deadlines (`DEADLINE-ELAPSED` → terminal `event: error`). Replaces the 0.3.7 single not_yet_wired frame. Spec: [`STREAMING.md`](STREAMING.md). |
| Signed kernel registry / `DiskRegistry` | `tensor-wasm-jit` (+ `tensor-wasm-api` for HTTP) | Wired | `kernel-registry` (jit); `kernel-registry-api` (api) | T35. Disk-persisted `DiskRegistry` over `tensor-wasm-artifacts::DiskArtifactStore`, restart-safe, paginated (`list_paginated`, cap 1000), optional publisher allowlist. HTTP backend selected by `TENSOR_WASM_API_KERNEL_REGISTRY_DIR`. CLI `kernel publish\|list\|verify` is **wired** (B6.4): `publish` BLAKE3-hashes + signs a `KernelManifest` and POSTs it to `/kernels`, `list` GETs + renders the manifest table, `verify` re-computes the HMAC locally (constant-time) against a manifest blob on disk. Replaces the prior v0.3.6 scaffold that exited `FEATURE_NOT_EXPOSED` (3). Servers built without `--features kernel-registry-api` (the default) return `503 kernel_registry_not_configured`, which the CLI surfaces as a clear error. Source: `crates/tensor-wasm-cli/src/cmd/kernel.rs`. Signing envelope is v2 (T12). Spec: [`KERNEL-REGISTRY.md`](KERNEL-REGISTRY.md). |
| Cooperative epoch yield / deadlines | `tensor-wasm-wasi-gpu`, `tensor-wasm-exec` | Wired | none (default) | T36 + this session's executor change. `wasi:scheduler/host@0.1.0` `SchedulerContext`; the executor's per-invocation `Instant` deadline drives both scheduler verdicts (`CONTINUE` / `DEADLINE-NEAR` / `DEADLINE-ELAPSED`) and `BackPressure` acquire rejection (`DEADLINE_NEAR_WINDOW = 50ms`). Spec: [`COOPERATIVE-YIELD.md`](COOPERATIVE-YIELD.md). |
| Pre-instantiated instance pool | `tensor-wasm-exec` | Wired | none (default) | T37. `InstancePool` + `InstancePoolConfig` wired through the invoke path; per-`(tenant, module-hash)` channel with pre-spawn and reset-on-return. Spec: [`INSTANCE-POOL.md`](INSTANCE-POOL.md). |
| Differential correctness oracle | `tensor-wasm-jit` | Landed (host); GPU path hardware-gated | `differential-oracle` | T38. Proptest harness driving `DifferentialOracle` over matmul / vector_add / conv2d blueprints + per-kernel tolerance table. **Host (Wasmtime CPU) verdicts run end-to-end today; CUDA GPU verdicts are `#[ignore]` pending the S22 self-hosted runner.** Spec: [`DIFFERENTIAL-ORACLE.md`](DIFFERENTIAL-ORACLE.md). |
| Per-tenant GPU memory quotas (in-process) | `tensor-wasm-tenant` | Wired | none (default) | T39. `TenantContextBuilder::with_gpu_memory_bytes_cap` + `consume_gpu_bytes` / `release_gpu_bytes`. In-process counter is the **primary** accounting source. Spec: [`GPU-QUOTAS.md`](GPU-QUOTAS.md). |
| GPU memory quotas (`cuMemPool` + host-side cap) | `tensor-wasm-mem` | Hardware-gated | `gpu-mem-pool` (strict-superset of `cudarc-backend`) | T39. `TenantMemPool` gives each tenant a `cuMemPool` and enforces the cap **host-side** in `allocate` (CAS over `live_bytes` vs `cap_bytes`), routing allocations through `cuMemAllocFromPoolAsync` (`UnifiedBuffer::new_in_tenant_pool`). NB: `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` is a retention hint, NOT a driver-level cap — a hardware run (GPU-VALIDATION-2026-05-30 BUG-1) disproved the earlier driver-pin claim. Requires CUDA 11.2+; the driver-reject test is `#[ignore]`. Bypass-resistant for pool-routed allocations; a raw-driver-handle tenant is out of scope (see GPU-QUOTAS.md). |
| Artifact-backed snapshots | `tensor-wasm-snapshot`, `tensor-wasm-artifacts` | Wired (default; BREAKING for default writer) | `artifact-backing` (on by default) | T40. Default snapshot envelope flipped from legacy inline v3 to `DiskArtifactStore`-backed v4. Reads still accept legacy v2/v3 indefinitely. Opt out per-call via `capture_legacy()` or build `--no-default-features --features signed-snapshots`. |
| Unified content-addressed artifact store | `tensor-wasm-artifacts` | Wired | none (always-on crate; no `[features]`) | Roadmap #9. `ArtifactStore` trait + `InMemoryArtifactStore` + fully-implemented `DiskArtifactStore` (BLAKE3 content-hash + zstd + HMAC-SHA256 + 16-byte magic). Backs snapshots (T40) and the JIT disk cache (T30). Spec: [`ARTIFACT-STORE.md`](ARTIFACT-STORE.md). |
| OpenAI-compat gateway | `tensor-wasm-api` | Wired | none (always-on routes) | T41. `/v1/completions` + `/v1/chat/completions` translate to internal invoke via `TENSOR_WASM_API_OPENAI_MODEL_MAP` (`model:uuid,…`); buffered or SSE. Closes the 0.3.7 `501 openai_not_yet_wired` scaffold. Caveats: argv marshalling calls `_start() -> ()` (no typed prompt arg yet); `usage` token counts are zeros (no tokenizer); multimodal content dropped — all deferred to v0.5. Spec: [`OPENAI-COMPAT.md`](OPENAI-COMPAT.md). |
| CUDA backend — cust (UVM) | `tensor-wasm-mem` | Hardware-gated | `unified-memory` (pulls `cust`) | Default cust 0.3 backing. `cuMemAllocManaged` + `cuMemAdvise` + `cuMemPrefetchAsync`. Round-trip tests `#[ignore]`. The historical "default" GPU backing. |
| CUDA backend — cudarc | `tensor-wasm-mem` | Hardware-gated | `cudarc-backend` (pulls `cudarc`) | cust → cudarc migration spike (W1.2 / [`CUDARC-SPIKE.md`](CUDARC-SPIKE.md)). Parallel `UnifiedBuffer` impl. Allocation/prefetch tests `#[ignore]`. |
| CUDA backend — cuda-oxide | `tensor-wasm-mem`, `tensor-wasm-jit` | Scaffold | `cuda-oxide-backend` | RFC 0001 v0.5 cust-successor. **Dep-less** (no cuda-oxide crate pulled in); on `tensor-wasm-mem` exposes `CudaOxideUnifiedBuffer` returning the `NOT_YET_WIRED` sentinel; on `tensor-wasm-jit` pulls `pliron` 0.15 (crates.io) for the `pliron_dialect` scaffold. **The git-pinned `experimental-cuda-oxide-host-backend` feature and its cuda-host/cuda-core/cuda-async git deps were removed this cycle pending crates.io publish** (re-add per [`CUDA-OXIDE-CUTOVER.md`](CUDA-OXIDE-CUTOVER.md)). |
| Real CUDA host functions / kernel launch | `tensor-wasm-exec`, `tensor-wasm-wasi-gpu`, `tensor-wasm-snapshot`, `tensor-wasm-tenant` | **End-to-end launch PROVEN ON HARDWARE**; lower-level host fns still partly hardware-gated | `cuda` | Real `wasi:cuda` host fns, contexts, `cuLaunchKernel`, GPU snapshot restore. Typed argv lowering for scalar + pointer kernel args (W1.1). **The full launch path — Wasm guest → `wasi:cuda` → `cuLaunchKernel` → results read back and asserted (`c[i]==a[i]+b[i]` from managed memory) — is VERIFIED ON REAL SILICON: `kernel_args_e2e` (incl. `vector_add_end_to_end_real_ptx_real_kernel`) passes 8/8 on an RTX 2060 (cc 7.5, CUDA 13.2), re-confirmed 2026-06-01.** These tests remain `#[ignore = "requires CUDA hardware"]` so they are skipped in (GPU-less) CI — they require a GPU runner or a local `--features cuda --include-ignored` run, not because the path is unproven. Individual lower-level cuda host fns still flagged UNVERIFIED-PENDING-HARDWARE in [`HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md) remain as-is unless covered by this e2e proof. See [`GPU-VALIDATION-2026-05-30.md`](GPU-VALIDATION-2026-05-30.md). |
| JIT auto-offload pipeline | `tensor-wasm-jit` | Wired (pipeline always compiled); CUDA wiring hardware-gated | `auto-offload` gates extra CUDA-side wiring | The Cranelift-free detector → `BlockIR` → PTX-text pipeline is always compiled and runs on host; the feature only gates CUDA-side wiring tested under `--features cuda`. Spec: [`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md). |
| JIT MatMul / wmma PTX emission | `tensor-wasm-jit` | Scaffold (refused by default); wmma hardware-unverified | runtime `EmitConfig::enable_experimental_matmul` (no Cargo flag) | `MatMul` returns `EmitError::NotYetImplemented` by default. The `wmma.mma.sync` sm_80+ lowering fires only when `enable_experimental_matmul = true`, and even then the emitted PTX is **hardware-unverified** (the dev RTX 2060 is SM_75 and cannot run it; needs the SM_89 runner). See [`HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md) item 7. |
| pliron PTX pipeline (stages) | `tensor-wasm-jit` | Scaffold | `cuda-oxide-backend`; stage-2 `twasm.*`→`llvm.*` under `pliron-llvm-backend` | `pliron_dialect` / `pliron_lowering` stages return `NotYetImplemented` / `NotYetWired` sentinels. `pliron-llvm-backend` is a strict superset of `cuda-oxide-backend` and carries a hard `llvm-sys = "221"` dep (needs system LLVM 221). Spec: [`PLIRON-PIPELINE.md`](PLIRON-PIPELINE.md). |

## Non-feature-flag capabilities (always on)

These ship by default with no Cargo feature gate; configured at runtime
(env/config), included here so the matrix doubles as a complete status
reference:

| Capability | Status | Configuration |
|---|---|---|
| Signed snapshots (HMAC-SHA256, v3) | Wired | `signed-snapshots` feature (on by default) |
| Capability-to-registry binding | Wired (unconditional; enforced even with `default-features = false`) | `strict-cap-binding` gates only the typed `*_strict` admin APIs |
| Scoped bearer tokens / audit log / per-token rate limit / HTTP metrics | Wired | env vars (`TENSOR_WASM_API_*`) |
| Async execution + epoch interruption | Wired | always-on behaviour, not a flag |
| NVIDIA MPS shared contexts | Runtime-selected | env/config, not a Cargo flag — see [`MPS-SETUP.md`](MPS-SETUP.md) |
| OTLP exporter | Opt-in | `otlp` feature (`tensor-wasm-core`) |

## Sources

- [`../CHANGELOG.md`](../CHANGELOG.md) `[0.3.7]` section and the T-task wire-up notes (T30, T33–T41, T8/T9/T12).
- Per-crate `[features]` tables in each `crates/*/Cargo.toml`.
- [`OPENAI-COMPAT.md`](OPENAI-COMPAT.md) — T41 wiring + v0.5 caveats.
- [`HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md) — authoritative inventory of unverified-on-silicon paths.
- The per-feature spec docs linked in the Notes column.
