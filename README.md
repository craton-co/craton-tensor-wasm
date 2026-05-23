# Bali

> A GPU-accelerated serverless WebAssembly runtime.

Bali runs untrusted Wasm modules with explicit (and, opt-in via the
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
git clone https://github.com/project-bali/bali
cd bali
cargo build --workspace
cargo test --workspace
```

This runs ~150 unit + integration tests across all 10 crates, all green
on a no-CUDA developer laptop.

### Run a sample Wasm function

```sh
# Build a minimal Wasm fixture (or use one from tests/wasm-fixtures/)
cargo run -p bali-cli -- run path/to/vector_add.wasm --export add
```

### Spin up the HTTP API

```sh
cargo run --release --bin bali -- serve --addr 0.0.0.0:8080
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

See [`crates/bali-api/API.md`](crates/bali-api/API.md) for the full REST
reference.

## Feature matrix

| Feature | Default | Where | Purpose |
|---|---|---|---|
| `unified-memory` | off (CUDA-host only) | bali-mem | Links `cust`; uses `cudaMallocManaged`. |
| `pinned-host-memory` | off | bali-mem | Page-locked fallback (rarely needed; default off path is plain heap). |
| `async-execution` | on | bali-exec | Wasmtime async + epoch interruption. |
| `cuda` | off | bali-wasi-gpu, bali-tenant | Real CUDA host functions / contexts. |
| `auto-offload` | off | bali-jit | Cranelift→PTX JIT pipeline. |
| `mps` | off | bali-tenant | Prefer NVIDIA MPS-backed shared contexts. |
| `otlp` | off | bali-core | OpenTelemetry OTLP exporter. |

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
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Tracing & metrics |
| [docs/SECURITY-AUDIT.md](docs/SECURITY-AUDIT.md) | Audit findings |
| [docs/WASMTIME-FORK.md](docs/WASMTIME-FORK.md) | Why we don't fork Wasmtime |
| [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) | Onboarding tutorial |
| [docs/WASM-DEVELOPER-GUIDE.md](docs/WASM-DEVELOPER-GUIDE.md) | Writing Wasm for Bali |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment |
| [crates/bali-api/API.md](crates/bali-api/API.md) | REST API reference |

## Architecture at a glance

```
                       ┌──────────────┐
                       │  bali-cli    │ (developer CLI)
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  bali-api    │ (HTTP gateway)
                       └──┬───────┬───┘
                          │       │
                ┌─────────▼───┐ ┌─▼─────────────┐
                │ bali-snapshot │ bali-tenant   │ (multi-tenant GPU)
                └─────────┬───┘ └─┬─────────────┘
                          │       │
                     ┌────▼───────▼────┐
                     │   bali-exec     │ (Wasmtime + Tokio)
                     └────┬────────────┘
                          │
                     ┌────▼─────────────┐
                     │ bali-wasi-gpu    │ ◀── bali-jit (JIT pipeline)
                     └────┬─────────────┘
                          │
                     ┌────▼─────────────┐
                     │   bali-mem       │ (CUDA Unified Memory)
                     └────┬─────────────┘
                          │
                     ┌────▼─────────────┐
                     │   bali-core      │ (errors, metrics, telemetry)
                     └──────────────────┘
```

Full diagram in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency graph constraints.
- Run `make ci` locally before pushing — it mirrors the GitHub Actions workflow.
- Fuzz harness lives in `fuzz/`; CUDA-only tests are marked `#[ignore = "requires CUDA hardware"]`.
- Security disclosures: see [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See `LICENSE` (committed alongside the repo).
