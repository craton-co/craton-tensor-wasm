# Changelog

All notable changes to Craton TensorWasm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-25

This release consolidates the v0.2, v0.3 and v0.4 PATH-TO-V1 milestones'
implementable items into a single shipped version. CUDA-hardware-bound
items (S22 runner, measured GPU dispatch numbers, MPS production
validation) and external-party items (pen-test, design-partner
deployments) remain deferred — see `docs/PATH-TO-V1.md` for the
ongoing roadmap.

### Upgrade notes

- **Bare-token entries in `TENSOR_WASM_API_TOKENS` are deprecated.**
  Rewrite as `token:tenant=*` (preserves current behavior) or
  `token:tenant=1,2,3` (per-tenant scope). Removal target: 1.0.
- **Per-tenant scope enforcement on `/invoke`.** Calls that worked
  under 0.1 with a bare token now require `:tenant=*` or an explicit
  scope match; out-of-scope calls return `403 tenant_scope_denied`.
- **Audit log emission on state-mutating routes.** Default sink is
  stdout; disable with `TENSOR_WASM_API_AUDIT_LOG=none` if undesired.
- **Per-token rate limiting** subject to
  `TENSOR_WASM_API_RATE_LIMIT_QPS` and `_BURST`. Disabled when either
  is unset/zero (matches pre-0.3 behavior).
- **HTTP request metrics** now emitted under
  `tensor_wasm_http_requests_total`, `*_duration_seconds`, and
  `*_in_flight`. Dashboards that previously had TODO markers for these
  series now have real data.
- **Snapshot reader** still enforces the 256 MiB decompression-bomb cap
  by default; trusted-snapshot callers should opt in via
  `SnapshotReader::with_max_decompressed`.

See `docs/MIGRATION-v0-to-v1.md` for the full deprecation /
behavioral-change table.

## [0.3.1] - 2026-05-25

Additive cuda-oxide scaffolding release. No breaking changes; default
workspace build untouched. Three new opt-in landings under the
new `cuda-oxide-backend` feature flag.

### Added
- `rfcs/0001-cuda-oxide-integration.md` — first real RFC under the
  W1.7 process. Proposes Option C (three backends side-by-side from
  v0.3.1 through v0.4; default flips to cuda-oxide at v0.5 contingent
  on cuda-oxide reaching v0.2.0; cudarc-backend is the documented
  fallback if it doesn't). Advances PATH-TO-V1 Open Decision #1 (O1).
- `tensor-wasm-mem`: new `cuda-oxide-backend` feature flag and
  `cuda_oxide_backend` scaffold module (`CudaOxideUnifiedBuffer`,
  stubbed `allocate` / `apply_advice`, `Drop` with tracing). Workspace
  deps for `cuda-host`, `cuda-core`, `cuda-async` pinned via git rev
  pointing at github.com/NVlabs/cuda-oxide (NOT yet on crates.io as of
  2026-05-25; the crates.io `cuda-oxide` name is a different,
  unrelated 2018-era project). Coexists with `unified-memory` (cust)
  and `cudarc-backend` (W1.2); the three are independent (O2).
- `tensor-wasm-jit`: new `cuda-oxide-backend` feature flag and
  `pliron_dialect` scaffold module (the future `cranelift_to_dialect_mir`
  lowering pass that targets cuda-oxide's Pliron-based IR). Includes a
  23-row mapping table from Cranelift IR ops to Pliron `dialect-mir`
  ops so the v0.4 author has a concrete target (O3).
- `docs/CUDA-KERNELS.md`: new Section 5 — "Path C: Rust kernels via
  cuda-oxide". Decision table for picking among Path A (hand-PTX),
  Path B (.cu via nvcc), and Path C (Rust via cuda-oxide) (O4).

### Changed
- `docs/RISKS.md`: cust EOL row cross-references RFC 0001 + cuda-oxide
  as the third option (O5).
- `docs/CUDARC-SPIKE.md`: Version-chosen table gains a `cuda-host`
  row; cudarc recommendation still valid as v0.3.x default + v0.5
  fallback (O5).
- `docs/PATH-TO-V1.md`: Open Decision #1 gains an inline 2026-05-25
  update line cross-linking RFC 0001 (O5).

### Toolchain note (NOT changed)
- Workspace pin remains `nightly-2026-03-15`. Enabling the new
  `cuda-oxide-backend` feature requires a local override to
  `nightly-2026-04-03+` (cuda-oxide's pin). The workspace default
  bump is scheduled for v0.4 per RFC 0001 "Toolchain plan" step 3
  and PATH-TO-V1 Open Decision #8 (quarterly cadence).

## [0.3.2] - 2026-05-25

cuda-oxide v0.4-prep wave. Four follow-ups to v0.3.1's scaffolding.
Builds clean on the bumped nightly with default features; CUDA tests
(W1.1 kernel_args_e2e, W5.9 cudarc_smoke) still pass on real RTX 2060.

### Added
- `deploy/helm/tensor-wasm/values.yaml` + Nomad job specs gain
  `image.backend` (cust | cudarc | cuda-oxide | "") toggle. Build-time
  selection via image-tag suffix. Same shape mirrored as a manual-swap
  comment in `deploy/k8s/20-deployment.yaml`. Resolves RFC 0001
  Unresolved question #4 (F1).
- `deny.toml` at workspace root. Posture: deny all unknown git +
  registry sources. Two `allow-git` allowlist entries — NVlabs/cuda-oxide
  and vaivaswatha/pliron — each with rationale and cross-ref to
  RFC 0001 (F2).
- `docs/REPRODUCIBLE-BUILDS.md` "Git-pinned sources" subsection +
  audit-trail table (Crate / Repository / Pinned rev / Rationale) (F2).
- `crates/tensor-wasm-bench/benches/dispatch_future_backends.rs` —
  new tail-latency bench comparing the busy-poll DispatchFuture
  against a stubbed cuda-async backend. First baseline numbers
  committed to `bench-results/dispatch-future-backends.json`
  (busy-poll P50 400 ns, P99.9 1.8 µs on RTX 2060 / WDDM).
  cuda-async slot returns the documented "not yet wired" sentinel
  until the v0.4 port. Resolves RFC 0001 Unresolved question #3 (F3).

### Changed
- `Cargo.toml`: cuda-host / cuda-core / cuda-async git deps switch
  from `branch = "main"` to the v0.1.0 tag SHA
  `4a56e4220aab8ce5d085a411e7f806cebb647d14`. Reproducible +
  dependabot-blind + auditable (F2).
- `rust-toolchain.toml`: bumped from `nightly-2026-03-15` to
  `nightly-2026-04-03` to match cuda-oxide's pin. Per RFC 0001
  Toolchain plan step 3 / PATH-TO-V1 Open Decision #8 (quarterly
  cadence). Workspace + benches + tests compile clean on the new
  pin. Rollback is a one-character revert if anything downstream
  breaks (F4).

## [Unreleased]
_No entries yet — open the next PR adding one._



### Security
- Repository ownership transferred to Craton Software Company
- Added LICENSE / NOTICE files (Apache-2.0)
- `tensor-wasm-api` now supports per-tenant scoped bearer tokens via the
  `token:tenant=N,M` syntax in `TENSOR_WASM_API_TOKENS`; cross-tenant access
  is refused with a `tenant_scope_denied` 403 (W2.1, advances v0.3).
- Structured audit log for state-mutating routes, opt-in via
  `TENSOR_WASM_API_AUDIT_LOG`; format and rotation guidance in
  `docs/AUDIT-LOG.md` (W2.2, advances v0.3).
- `docs/DEPLOYMENT.md` gains an mTLS section covering both self-terminated
  TLS and reverse-proxy fronting (W2.8, advances v0.3).
### Added
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, MAINTAINERS.md, docs/RISKS.md
- GitHub issue/PR templates and dependabot configuration
- `tensor-wasm-wasi-gpu` — typed argv lowering for scalar and pointer kernel
  arguments; `KernelArgsUnsupported` is now reserved for sanity-cap
  rejections rather than the missing-marshaller case (W1.1, advances v0.2).
- `tensor-wasm-mem` — cudarc backend spike behind the new
  `--features cudarc-backend` flag; `cust` remains the default backend
  (W1.2, advances v0.2).
- `tensor-wasm-snapshot` — cross-version compatibility test framework with
  golden fixtures under `crates/tensor-wasm-snapshot/tests/fixtures/`
  (W1.3, advances v0.2).
- `tensor-wasm-api` — per-token QPS rate limiting middleware configured via
  `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST`;
  retires the global `ConcurrencyLimitLayer(64)` workaround called out in
  the 0.1.0 known limitations (W1.4, advances v0.2).
- `tensor-wasm-api` — HTTP request metrics middleware exporting
  `tensor_wasm_http_requests_total`, a request-duration histogram, and an
  `in_flight` gauge (W2.3, advances v0.3).
- `tensor-wasm-cli` — `tensor-wasm observe` subcommand for live metrics
  tailing (W1.5, advances v0.2).
- `tensor-wasm-cli` — generated shell completions and man pages, plus a
  `tensor-wasm man` subcommand that prints them at runtime (W2.4, advances
  v0.3).
- `deploy/` — Kubernetes manifests and a Helm chart for the API gateway
  (W2.7, advances v0.3).
- `docs/CUDA-SETUP.md` rewritten end-to-end with no hedging language
  (W1.6, advances v0.2).
- `rfcs/` — lightweight RFC process with template and index
  (W1.7, advances v0.2).
- `GOVERNANCE.md` — project governance scaffold (W1.8, advances v0.2).
- `docs/SLO.md` — SLO definitions and burn-rate alert recipes
  (W1.9, advances v0.2).
- `docs/dashboards/` — reference Grafana dashboard JSON
  (W2.5, advances v0.3).
- `docs/runbooks/` — per-alert operator runbooks keyed to the SLO alerts
  (W2.6, advances v0.3).
- `docs/WASMTIME-FORK.md` — wasmtime upgrade cadence policy
  (W2.9, advances v0.4).
### Changed
- Workspace authors and repository metadata updated to craton-co/craton-tensor-wasm
- Bumped `prometheus-client` from `0.22` to `0.24` so `tensor_wasm_gpu_memory_used_bytes`
  can use the native `Gauge<u64, AtomicU64>` (added upstream in 0.23.0 via
  prometheus/client_rust#226) instead of the previous signed-int workaround.
### Deprecated
- Bare-token entries in `TENSOR_WASM_API_TOKENS` (tokens without a
  `:tenant=...` scope clause) are deprecated in favour of the scoped form
  introduced in W2.1. Unscoped tokens still authenticate but are logged
  with a deprecation warning; targeted for removal in v1.0.

## [0.1.0] — 2026-05-23

The first scaffold release of TensorWasm — a GPU-accelerated serverless WebAssembly
runtime. Every subsystem in the architecture is present and tested; CUDA-
bound paths are feature-gated and exercised by a self-hosted CUDA CI runner
(deferred until the appropriate hardware lands).

### Added

#### Crates
- `tensor-wasm-core` — `TensorWasmError` enum, `TenantId`/`InstanceId`/`KernelId`
  newtypes, Prometheus metrics registry, `tracing-subscriber` init with an
  optional OTLP exporter (`otlp` feature).
- `tensor-wasm-mem` — `UnifiedBuffer` with feature-gated CUDA backing,
  `UnifiedMemoryPool` bump allocator, `cudaMemAdvise` hint helpers,
  `PinnedHostBuffer` fallback, Wasmtime `MemoryCreator` integration,
  `IsolationLevel` taxonomy.
- `tensor-wasm-exec` — `TensorWasmEngine` wrapping `wasmtime::Engine` with epoch-based
  interruption; `TensorWasmInstance` + `InstanceState`; `TensorWasmExecutor` with
  async spawn / call / terminate. 100-concurrent integration test plus
  epoch-timeout regression test.
- `tensor-wasm-wasi-gpu` — `wasi-cuda` host bridge (`wasi:cuda/host@0.1.0` ABI:
  `wasi_cuda_load_ptx`, `wasi_cuda_launch`, `wasi_cuda_sync`,
  `wasi_cuda_last_error_*`). Instance-scoped `KernelRegistry` with
  per-owner authorisation. Stub on non-CUDA hosts; real dispatch behind
  the `cuda` feature. Async dispatch + back-pressure semaphore.
- `tensor-wasm-jit` — Cranelift-free detector over a simplified `BlockIR`,
  `TensorWasmKernelBlueprint` IR, CLIF→IR lowering, PTX text emitter for
  sm_80 (including `wmma` for MatMul), LRU `KernelCache`, `DeoptGuard`.
- `tensor-wasm-snapshot` — `SnapshotWriter::capture` and `SnapshotReader::restore`
  with CRC32 integrity check, per-field size limits, zstd compression,
  format version 2.
- `tensor-wasm-tenant` — `TenantContext` and `TenantRegistry`; MPS-or-fallback
  decision based on `/tmp/nvidia-mps` existence.
- `tensor-wasm-api` — Axum 0.7 HTTP gateway: `GET /healthz`, `GET /metrics`,
  `POST /functions`, `DELETE /functions/{id}`, `POST /functions/{id}/invoke`,
  `POST /functions/{id}/invoke-async`, `GET /jobs/{id}`. Structured JSON
  error envelope; tower-http timeout, trace, and concurrency-limit middleware;
  W3C `traceparent` propagation.
- `tensor-wasm-cli` — `tensor-wasm` binary: `run`, `deploy`, `invoke`, `bench`,
  `snapshot save/restore`, `metrics`, `completions`.
- `tensor-wasm-bench` — Criterion bench harness with 5 bench targets:
  `kernel_dispatch`, `cold_start`, `memory_bandwidth`, `jit_compile`,
  `e2e_inference`.

#### Documentation
- `README.md`, `ARCHITECTURE.md`, `SECURITY.md`.
- `docs/`: `BUILD.md`, `CUDA-SETUP.md`, `MPS-SETUP.md`, `AUTO-OFFLOAD.md`,
  `COLD-START.md`, `CLI.md`, `PERFORMANCE.md`, `OBSERVABILITY.md`,
  `SECURITY-AUDIT.md`, `WASMTIME-FORK.md`, `GETTING-STARTED.md`,
  `WASM-DEVELOPER-GUIDE.md`, `DEPLOYMENT.md`.
- `crates/tensor-wasm-api/API.md` — REST API reference.
- `kernels/vector_add.ptx` — sm_80 PTX fixture.
- `tests/wasm-fixtures/matrix_multiply.wat`.

#### Infrastructure
- `Cargo.toml` workspace with 10 member crates plus excluded `fuzz/` subdir.
- `rust-toolchain.toml` pinned to `nightly-2026-03-15`.
- `Makefile` with `build`, `test`, `bench`, `fmt`, `fmt-check`, `lint`,
  `check`, `doc`, `clean`, `ci`, `ci-bench`, `help` targets.
- `.github/workflows/ci.yml` — fmt + clippy + test + actionlint on
  ubuntu-latest with CUDA stub libs.
- `.github/workflows/bench.yml` — Criterion run with 10% regression check
  (regression diff stub; concrete diff lands in v0.2).
- `.github/workflows/release.yml` — `cargo publish --dry-run`, x86_64-linux
  binary build, GitHub Release upload on tag push.
- `docker-compose.yml` + `docker/` — observability stack (tensor-wasm-api +
  Prometheus + Grafana + Jaeger).
- `fuzz/` — cargo-fuzz package with three targets (`fuzz_wasm_compile`,
  `fuzz_ptx_emit`, `fuzz_snapshot_restore`).

### Known limitations
- CUDA paths (`unified-memory`, `cuda`, `auto-offload`, `mps`) require
  hardware that the public CI doesn't yet have. Local tests on a no-CUDA
  Windows host pass; CUDA-only tests are marked `#[ignore]`.
- Per-tenant rate limiting in `tensor-wasm-api` is currently global
  (`ConcurrencyLimitLayer(64)`). Per-tenant quotas track for v0.2 (BA-005).
- `cargo audit` is not yet in CI (BA-008).
- The `epoch_timeout` test is gated `#[ignore]` on Windows because of a
  Wasmtime fiber unwinding panic on epoch interrupt; the test runs on
  Linux/macOS CI.
- Auto-offload pipeline works against a simplified `BlockIR`; full
  Cranelift integration is deferred (see `docs/WASMTIME-FORK.md`).

[0.1.0]: https://github.com/craton-co/craton-tensor-wasm/releases/tag/v0.1.0
