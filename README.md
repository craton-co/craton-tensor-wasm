# Craton TensorWasm

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust: nightly](https://img.shields.io/badge/Rust-nightly--2026--03--15-orange.svg)](rust-toolchain.toml)
[![Build](https://img.shields.io/badge/Build-GitHub_Actions-lightgrey.svg)](https://github.com/craton-co/craton-tensor-wasm/actions)

> A GPU-accelerated serverless WebAssembly runtime.

**Status: v0.1.0 — preview release.**

Craton TensorWasm runs untrusted Wasm modules with explicit (and, opt-in via the
`auto-offload` feature, implicit) GPU kernel dispatch on CUDA. It's built
on Wasmtime + Tokio + cust, exposes an HTTP API, and ships with a developer
CLI, a snapshot subsystem for fast cold-starts, and OpenTelemetry tracing
out of the box.

## Status

**v0.1.0** — the scaffold release. Every subsystem in the architecture is
present and tested on a CUDA-free host; the CUDA-bound paths (real
`cudaMallocManaged`, real PTX `ptxas` validation, real kernel launches) are
gated behind `--features unified-memory` and exercised by the CUDA self-
hosted CI runner. See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for
honest numbers.

## 5-minute quickstart

### Prerequisites
- Rust toolchain via `rustup` (the repo pins `nightly-2026-03-15`).
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
| `async-execution` | on | tensor-wasm-exec | Wasmtime async + epoch interruption. |
| `cuda` | off | tensor-wasm-wasi-gpu, tensor-wasm-tenant | Real CUDA host functions / contexts. |
| `auto-offload` | off | tensor-wasm-jit | Cranelift→PTX JIT pipeline. |
| `mps` | off | tensor-wasm-tenant | Prefer NVIDIA MPS-backed shared contexts. |
| `otlp` | off | tensor-wasm-core | OpenTelemetry OTLP exporter. |

Full taxonomy: [`docs/BUILD.md`](docs/BUILD.md).

## GPU requirements

- Driver: CUDA 12.0+ (Linux ≥ 525.60.13 / Windows ≥ 527.41).
- Architecture: sm_80+ (Ampere) for PTX wmma kernels emitted by S12.
- Optional: NVIDIA MPS for multi-tenant context isolation — see
  [`docs/MPS-SETUP.md`](docs/MPS-SETUP.md).

## Documentation

| Doc | Subject |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate dependency graph |
| [SECURITY.md](SECURITY.md) | Threat model |
| [docs/BUILD.md](docs/BUILD.md) | Build matrix + feature flags |
| [docs/CUDA-SETUP.md](docs/CUDA-SETUP.md) | CUDA toolkit install + env |
| [docs/MPS-SETUP.md](docs/MPS-SETUP.md) | Multi-tenant GPU contexts |
| [docs/AUTO-OFFLOAD.md](docs/AUTO-OFFLOAD.md) | Supported JIT patterns |
| [docs/COLD-START.md](docs/COLD-START.md) | Snapshot/restore latency |
| [docs/CLI.md](docs/CLI.md) | CLI reference |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Bench results |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Comparing TensorWasm against other runtimes |
| [docs/PATH-TO-V1.md](docs/PATH-TO-V1.md) | Proposed roadmap from v0.1 preview to v1.0 |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Tracing & metrics |
| [docs/SECURITY-AUDIT.md](docs/SECURITY-AUDIT.md) | Audit findings |
| [docs/WASMTIME-FORK.md](docs/WASMTIME-FORK.md) | Why we don't fork Wasmtime |
| [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) | Onboarding tutorial |
| [docs/WASM-DEVELOPER-GUIDE.md](docs/WASM-DEVELOPER-GUIDE.md) | Writing Wasm for TensorWasm |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment |
| [crates/tensor-wasm-api/API.md](crates/tensor-wasm-api/API.md) | REST API reference |

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
- Run `make ci` locally before pushing — it mirrors the GitHub Actions workflow.
- Fuzz harness lives in `fuzz/`; CUDA-only tests are marked `#[ignore = "requires CUDA hardware"]`.
- RFCs: substantive design changes go through the lightweight process in [`rfcs/README.md`](rfcs/README.md) before the implementation PR.
- Security disclosures: see [SECURITY.md](SECURITY.md).

## Security

Vulnerability reports and disclosures are covered in [SECURITY.md](SECURITY.md).
Reach the maintainers at `security@craton.com.ar`. Coordinated disclosure is
preferred; please do not file security issues on the public tracker.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) (committed
alongside the repo).
