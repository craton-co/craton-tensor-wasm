# Craton TensorWasm

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust: nightly](https://img.shields.io/badge/Rust-nightly--2026--04--03-orange.svg)](rust-toolchain.toml)
[![Build](https://img.shields.io/badge/Build-GitHub_Actions-lightgrey.svg)](https://github.com/craton-co/craton-tensor-wasm/actions)
[![SBOM](https://img.shields.io/github/actions/workflow/status/craton-co/craton-tensor-wasm/sbom.yml?branch=main&label=SBOM)](.github/workflows/sbom.yml)
[![API reference](https://img.shields.io/github/actions/workflow/status/craton-co/craton-tensor-wasm/api-reference.yml?branch=main&label=api-reference)](.github/workflows/api-reference.yml)

> A GPU-accelerated serverless WebAssembly runtime.

**Status: v0.3.6 — released 2026-05-27.** Consolidates the implementable
items from the v0.2 / v0.3 / v0.4 PATH-TO-V1 milestones into a single
shipped version.

Craton TensorWasm runs untrusted Wasm modules with explicit (and, opt-in via the
`auto-offload` feature, implicit) GPU kernel dispatch on CUDA. It's built
on Wasmtime + Tokio + cust (with a cudarc backend spike opt-in), exposes
an HTTP API, and ships with a developer CLI, a snapshot subsystem for fast
cold-starts, and OpenTelemetry tracing wired end-to-end.

## Status

**v0.3.6** — auth, observability, ops, and supply-chain hardening shipped.
The host-only execution path is solid (≈ 150 tests across 10 crates, all
green on a CUDA-free developer laptop). CUDA-bound paths (real
`cudaMallocManaged`, real PTX `ptxas` validation, real kernel launches)
are gated behind `--features unified-memory` and exercised by the CUDA
self-hosted runner once it lands (see `docs/CUDARC-SPIKE.md` for the
concrete compile-friction status as of the 0.3.6 cut). See
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) and
[`bench-results/tail-latency.json`](bench-results/tail-latency.json) for
the honest measured numbers.

What 0.3.6 ships on top of 0.1.0:

- **Auth & multi-tenancy.** Per-tenant scoped bearer tokens (W2.1) with
  `tenant_scope_denied` 403s; structured audit log opt-in via
  `TENSOR_WASM_API_AUDIT_LOG` (W2.2); mTLS deployment guide (W2.8).
- **Rate limiting & metrics.** Per-token QPS + burst (W1.4) replaces the
  global `ConcurrencyLimitLayer(64)` workaround; HTTP request metrics
  middleware (W2.3); `tensor_wasm_build_info` metric for version/commit/feature
  labels (W4.9); distributed tracing wired end-to-end (W4.1).
- **CLI ergonomics.** `tensor-wasm observe` live metrics tail (W1.5);
  generated shell completions + man pages with a runtime `tensor-wasm man`
  subcommand (W2.4).
- **Ops.** Reference Grafana dashboard (W2.5), one runbook per SLO alert
  (W2.6), Kubernetes manifests and a Helm chart for the API gateway (W2.7).
- **Supply chain.** CycloneDX SBOM attached to every release (W4.3);
  reproducible-build recipe documented; API reference auto-bundled per
  release (W4.8).
- **Docs.** New end-to-end guides: CUDA kernel authoring (W4.5), capacity
  planning, backup/restore, production-deployment tutorial, migrating
  from Wasmtime/Wasmer.

## 5-minute quickstart

### Prerequisites
- Rust toolchain via `rustup` (the repo pins `nightly-2026-04-03`).
- (Optional) CUDA 12.0+ for the GPU-accelerated path — see [`docs/CUDA-SETUP.md`](docs/CUDA-SETUP.md).

### Build & test

```sh
git clone https://github.com/craton-co/craton-tensor-wasm
cd tensor-wasm
cargo build --workspace
cargo test --workspace
```

This runs ~150 unit + integration tests across all 10 crates, all green
on a no-CUDA developer laptop.

### Run a sample Wasm function

```sh
# Build a minimal Wasm fixture (or use one from tests/wasm-fixtures/)
cargo run -p tensor-wasm-cli -- run path/to/vector_add.wasm --export add
```

### Spin up the HTTP API

```sh
cargo run --release --bin tensor-wasm -- serve --addr 0.0.0.0:8080
```

Then upload + invoke a function:

```sh
curl -X POST http://localhost:8080/functions \
  -H 'content-type: application/json' \
  -d '{"name":"add","wasm_b64":"<base64-of-wasm>"}'
# → {"id":"<uuid>"}

curl -X POST http://localhost:8080/functions/<uuid>/invoke \
  -H 'content-type: application/json' -d '{}'
```

See [`crates/tensor-wasm-api/API.md`](crates/tensor-wasm-api/API.md) for the full REST
reference.

## Feature matrix

| Feature | Default | Where | Purpose |
|---|---|---|---|
| `unified-memory` | off (CUDA-host only) | tensor-wasm-mem | Links `cust`; uses `cudaMallocManaged`. |
| `pinned-host-memory` | off | tensor-wasm-mem | Page-locked fallback (rarely needed; default off path is plain heap). |
| `cudarc-backend` | off | tensor-wasm-mem | Opt-in `cudarc` backend spike (W1.2). `cust` remains the default; see [`docs/CUDARC-SPIKE.md`](docs/CUDARC-SPIKE.md). |
| `async-execution` | on | tensor-wasm-exec | Wasmtime async + epoch interruption. |
| `cuda` | off | tensor-wasm-wasi-gpu, tensor-wasm-tenant | Real CUDA host functions / contexts. Typed argv lowering for scalar + pointer kernel args (W1.1). |
| `auto-offload` | off | tensor-wasm-jit | Cranelift→PTX JIT pipeline. |
| `mps` | off | tensor-wasm-tenant | Prefer NVIDIA MPS-backed shared contexts. |
| `otlp` | off | tensor-wasm-core | OpenTelemetry OTLP exporter; trace IDs propagate end-to-end through HTTP → tenant lookup → snapshot restore → dispatch (W4.1). |

Operational capabilities that ship on by default (no feature flag):

| Capability | Configuration | Notes |
|---|---|---|
| Scoped bearer tokens | `TENSOR_WASM_API_TOKENS=token:tenant=N,M` | W2.1. Bare tokens still authenticate but are deprecated. |
| Audit log | `TENSOR_WASM_API_AUDIT_LOG=<path>` | W2.2. Schema + rotation in [`docs/AUDIT-LOG.md`](docs/AUDIT-LOG.md). |
| Per-token rate limit | `TENSOR_WASM_API_RATE_LIMIT_QPS`, `_BURST` | W1.4. Retires the global concurrency cap. |
| HTTP request metrics | always on | W2.3. `tensor_wasm_http_requests_total`, duration histogram, in-flight gauge. |
| `build_info` metric | always on | W4.9. Version, git commit, enabled features as labels. |
| OpenAI-compatible inference gateway shim (`/v1/completions`, `/v1/chat/completions`) | always on | B4.9 scaffold returning 501; v0.4 wires translation. See [`docs/OPENAI-COMPAT.md`](docs/OPENAI-COMPAT.md). |

Full taxonomy: [`docs/BUILD.md`](docs/BUILD.md).

## GPU requirements

- Driver: CUDA 12.0+ (Linux ≥ 525.60.13 / Windows ≥ 527.41).
- Architecture: sm_80+ (Ampere) for PTX wmma kernels emitted by S12.
- Optional: NVIDIA MPS for multi-tenant context isolation — see
  [`docs/MPS-SETUP.md`](docs/MPS-SETUP.md).

## Documentation

For the complete sitemap (every Markdown doc in this repo, grouped by purpose with one-sentence summaries and wave tags) see [`docs/INDEX.md`](docs/INDEX.md).

### Architecture & reference
| Doc | Subject |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate dependency graph |
| [SECURITY.md](SECURITY.md) | Threat model + backport policy |
| [docs/API-REFERENCE.md](docs/API-REFERENCE.md) | Per-release rustdoc + OpenAPI bundle policy |
| [docs/BUILD.md](docs/BUILD.md) | Build matrix + feature flags |
| [crates/tensor-wasm-api/API.md](crates/tensor-wasm-api/API.md) | REST API reference |
| [docs/CLI.md](docs/CLI.md) | CLI reference |
| [docs/WASMTIME-FORK.md](docs/WASMTIME-FORK.md) | Why we don't fork Wasmtime |
| [docs/WASMTIME-UPGRADE.md](docs/WASMTIME-UPGRADE.md) | Wasmtime upgrade cadence policy (W2.9) |

### Getting started & guides
| Doc | Subject |
|---|---|
| [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) | Onboarding tutorial |
| [docs/WASM-DEVELOPER-GUIDE.md](docs/WASM-DEVELOPER-GUIDE.md) | Writing Wasm for TensorWasm |
| [docs/CUDA-KERNELS.md](docs/CUDA-KERNELS.md) | Writing CUDA kernels for TensorWasm (W4.5) |
| [docs/MIGRATING-FROM-WASMTIME-WASMER.md](docs/MIGRATING-FROM-WASMTIME-WASMER.md) | Migrating from Wasmtime/Wasmer (W3.9) |
| [docs/tutorials/production-deployment.md](docs/tutorials/production-deployment.md) | End-to-end production deployment (W3.8) |

### CUDA & performance
| Doc | Subject |
|---|---|
| [docs/AUTO-OFFLOAD.md](docs/AUTO-OFFLOAD.md) | Supported JIT patterns |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Comparing TensorWasm against other runtimes |
| [docs/COLD-START.md](docs/COLD-START.md) | Snapshot/restore latency |
| [docs/CUDA-SETUP.md](docs/CUDA-SETUP.md) | CUDA toolkit install + env (W1.6 rewrite) |
| [docs/CUDARC-SPIKE.md](docs/CUDARC-SPIKE.md) | `cudarc` backend spike (W1.2) |
| [docs/MPS-SETUP.md](docs/MPS-SETUP.md) | Multi-tenant GPU contexts |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Bench results |
| [docs/SNAPSHOT-COMPATIBILITY.md](docs/SNAPSHOT-COMPATIBILITY.md) | Cross-version snapshot policy (W1.3) |

### Operations
| Doc | Subject |
|---|---|
| [docs/AUDIT-LOG.md](docs/AUDIT-LOG.md) | Audit log schema + rotation (W2.2) |
| [docs/BACKUP-RESTORE.md](docs/BACKUP-RESTORE.md) | Backup and restore procedures (W3.7) |
| [docs/CAPACITY-PLANNING.md](docs/CAPACITY-PLANNING.md) | Tenants-per-host curves at fixed SLA (W4.4) |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment |
| [docs/deployment/mtls.md](docs/deployment/mtls.md) | mTLS deployment recipe (W2.8) |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Tracing & metrics |
| [docs/SLO.md](docs/SLO.md) | SLO definitions + burn-rate alerts (W1.9) |
| [docs/UPGRADE.md](docs/UPGRADE.md) | Rolling a fleet between versions (W3.3) |
| [docs/dashboards/README.md](docs/dashboards/README.md) | Reference Grafana dashboards (W2.5) |
| [docs/runbooks/README.md](docs/runbooks/README.md) | Operator runbooks index (W2.6) |
| [deploy/k8s/README.md](deploy/k8s/README.md) | Kubernetes manifests (W2.7) |
| [deploy/helm/tensor-wasm/README.md](deploy/helm/tensor-wasm/README.md) | Helm chart (W2.7) |

### Governance & supply chain
| Doc | Subject |
|---|---|
| [GOVERNANCE.md](GOVERNANCE.md) | Project governance (W1.8) |
| [MAINTAINERS.md](MAINTAINERS.md) | Maintainer list |
| [docs/MIGRATION-v0-to-v1.md](docs/MIGRATION-v0-to-v1.md) | v0.x → v1.0 migration plan (W3.2) |
| [docs/REPRODUCIBLE-BUILDS.md](docs/REPRODUCIBLE-BUILDS.md) | Reproducible-build recipe (W3.6) |
| [docs/SBOM.md](docs/SBOM.md) | CycloneDX SBOM generation + publishing (W4.3) |
| [docs/SECURITY-AUDIT.md](docs/SECURITY-AUDIT.md) | Audit findings |
| [docs/TRADEMARK.md](docs/TRADEMARK.md) | Trademark policy (W3.4) |
| [docs/PATH-TO-V1.md](docs/PATH-TO-V1.md) | Proposed roadmap from v0.1 preview to v1.0 |
| [rfcs/README.md](rfcs/README.md) | Lightweight RFC process (W1.7) |

## Architecture at a glance

```
                       ┌──────────────┐
                       │  tensor-wasm-cli    │ (developer CLI)
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  tensor-wasm-api    │ (HTTP gateway)
                       └──┬───────┬───┘
                          │       │
                ┌─────────▼───┐ ┌─▼─────────────┐
                │ tensor-wasm-snapshot │ tensor-wasm-tenant   │ (multi-tenant GPU)
                └─────────┬───┘ └─┬─────────────┘
                          │       │
                     ┌────▼───────▼────┐
                     │   tensor-wasm-exec     │ (Wasmtime + Tokio)
                     └────┬────────────┘
                          │
                     ┌────▼─────────────┐
                     │ tensor-wasm-wasi-gpu    │ ◀── tensor-wasm-jit (JIT pipeline)
                     └────┬─────────────┘
                          │
                     ┌────▼─────────────┐
                     │   tensor-wasm-mem       │ (CUDA Unified Memory)
                     └────┬─────────────┘
                          │
                     ┌────▼─────────────┐
                     │   tensor-wasm-core      │ (errors, metrics, telemetry)
                     └──────────────────┘
```

Full diagram in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency graph constraints.
- Read [GOVERNANCE.md](GOVERNANCE.md) for the maintainer decision process.
- Run `make ci` locally before pushing — it mirrors the GitHub Actions workflow.
- Fuzz harness lives in `fuzz/`; CUDA-only tests are marked `#[ignore = "requires CUDA hardware"]`.
- RFCs: substantive design changes go through the lightweight process in [`rfcs/README.md`](rfcs/README.md) before the implementation PR. Use [`rfcs/TEMPLATE.md`](rfcs/TEMPLATE.md) as a starting point.
- Security disclosures: see [SECURITY.md](SECURITY.md).

## Security

Vulnerability reports and disclosures are covered in [SECURITY.md](SECURITY.md).
Reach the maintainers at `security@craton.com.ar`. Coordinated disclosure is
preferred; please do not file security issues on the public tracker.

## Roadmap

See [docs/PATH-TO-V1.md](docs/PATH-TO-V1.md#post-v036-strategic-features) for the post-v0.3.6 strategic feature backlog (13 items grouped by tier).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) (committed
alongside the repo).
