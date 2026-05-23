# Changelog

All notable changes to Project Bali will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-23

The first scaffold release of Bali — a GPU-accelerated serverless WebAssembly
runtime. Every subsystem in the architecture is present and tested; CUDA-
bound paths are feature-gated and exercised by a self-hosted CUDA CI runner
(deferred until the appropriate hardware lands).

### Added

#### Crates
- `bali-core` — `BaliError` enum, `TenantId`/`InstanceId`/`KernelId`
  newtypes, Prometheus metrics registry, `tracing-subscriber` init with an
  optional OTLP exporter (`otlp` feature).
- `bali-mem` — `UnifiedBuffer` with feature-gated CUDA backing,
  `UnifiedMemoryPool` bump allocator, `cudaMemAdvise` hint helpers,
  `PinnedHostBuffer` fallback, Wasmtime `MemoryCreator` integration,
  `IsolationLevel` taxonomy.
- `bali-exec` — `BaliEngine` wrapping `wasmtime::Engine` with epoch-based
  interruption; `BaliInstance` + `InstanceState`; `BaliExecutor` with
  async spawn / call / terminate. 100-concurrent integration test plus
  epoch-timeout regression test.
- `bali-wasi-gpu` — `wasi-cuda` host bridge (`wasi:cuda/host@0.1.0` ABI:
  `wasi_cuda_load_ptx`, `wasi_cuda_launch`, `wasi_cuda_sync`,
  `wasi_cuda_last_error_*`). Instance-scoped `KernelRegistry` with
  per-owner authorisation. Stub on non-CUDA hosts; real dispatch behind
  the `cuda` feature. Async dispatch + back-pressure semaphore.
- `bali-jit` — Cranelift-free detector over a simplified `BlockIR`,
  `BaliKernelBlueprint` IR, CLIF→IR lowering, PTX text emitter for
  sm_80 (including `wmma` for MatMul), LRU `KernelCache`, `DeoptGuard`.
- `bali-snapshot` — `SnapshotWriter::capture` and `SnapshotReader::restore`
  with CRC32 integrity check, per-field size limits, zstd compression,
  format version 2.
- `bali-tenant` — `TenantContext` and `TenantRegistry`; MPS-or-fallback
  decision based on `/tmp/nvidia-mps` existence.
- `bali-api` — Axum 0.7 HTTP gateway: `GET /healthz`, `GET /metrics`,
  `POST /functions`, `DELETE /functions/{id}`, `POST /functions/{id}/invoke`,
  `POST /functions/{id}/invoke-async`, `GET /jobs/{id}`. Structured JSON
  error envelope; tower-http timeout, trace, and concurrency-limit middleware;
  W3C `traceparent` propagation.
- `bali-cli` — `bali` binary: `run`, `deploy`, `invoke`, `bench`,
  `snapshot save/restore`, `metrics`, `completions`.
- `bali-bench` — Criterion bench harness with 5 bench targets:
  `kernel_dispatch`, `cold_start`, `memory_bandwidth`, `jit_compile`,
  `e2e_inference`.

#### Documentation
- `README.md`, `ARCHITECTURE.md`, `SECURITY.md`.
- `docs/`: `BUILD.md`, `CUDA-SETUP.md`, `MPS-SETUP.md`, `AUTO-OFFLOAD.md`,
  `COLD-START.md`, `CLI.md`, `PERFORMANCE.md`, `OBSERVABILITY.md`,
  `SECURITY-AUDIT.md`, `WASMTIME-FORK.md`, `GETTING-STARTED.md`,
  `WASM-DEVELOPER-GUIDE.md`, `DEPLOYMENT.md`.
- `crates/bali-api/API.md` — REST API reference.
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
- `docker-compose.yml` + `docker/` — observability stack (bali-api +
  Prometheus + Grafana + Jaeger).
- `fuzz/` — cargo-fuzz package with three targets (`fuzz_wasm_compile`,
  `fuzz_ptx_emit`, `fuzz_snapshot_restore`).

### Known limitations
- CUDA paths (`unified-memory`, `cuda`, `auto-offload`, `mps`) require
  hardware that the public CI doesn't yet have. Local tests on a no-CUDA
  Windows host pass; CUDA-only tests are marked `#[ignore]`.
- Per-tenant rate limiting in `bali-api` is currently global
  (`ConcurrencyLimitLayer(64)`). Per-tenant quotas track for v0.2 (BA-005).
- `cargo audit` is not yet in CI (BA-008).
- The `epoch_timeout` test is gated `#[ignore]` on Windows because of a
  Wasmtime fiber unwinding panic on epoch interrupt; the test runs on
  Linux/macOS CI.
- Auto-offload pipeline works against a simplified `BlockIR`; full
  Cranelift integration is deferred (see `docs/WASMTIME-FORK.md`).

[0.1.0]: https://github.com/project-bali/bali/releases/tag/v0.1.0
