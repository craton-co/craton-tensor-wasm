# Building Craton TensorWasm

Craton TensorWasm is a Cargo workspace of 10 crates implementing a GPU-accelerated serverless Wasm runtime. It supports three build matrices: a full CUDA host (real hardware), a CUDA stub (for CI), and a no-CUDA configuration (quick local checks). This document walks through each, plus feature flags, tests, benchmarks, docs, and CI parity.

## Prerequisites

- Rust toolchain: pinned in `rust-toolchain.toml` (currently `nightly-2026-03-15`). Rustup picks it up automatically the first time you run `cargo` in the workspace.
- For CUDA builds: see [CUDA-SETUP.md](./CUDA-SETUP.md).
- For no-CUDA host: nothing extra needed beyond Rust.

## Build matrix

| Config | Command | Active tensor-wasm-mem feature | Use case |
|---|---|---|---|
| No-CUDA (default) | `cargo build --workspace` | none (pure-Rust path) | Quick local check — no CUDA linkage at all |
| CUDA host | `cargo build --workspace --features tensor-wasm-mem/unified-memory` | `unified-memory` | Real hardware — `cudaMallocManaged` |
| CUDA stub (CI) | `cargo build --workspace --features tensor-wasm-mem/unified-memory` (stub `libcuda.so` on `LD_LIBRARY_PATH`) | `unified-memory` | CI build/test — links against stub libs |
| No-CUDA pinned host | `cargo build --workspace --features tensor-wasm-mem/pinned-host-memory` | `pinned-host-memory` | Page-locked host buffers without CUDA linkage |

Note: the workspace has **no default features** enabled. `tensor-wasm-mem` ships two opt-in features for memory backing — `unified-memory` (links `cust` and uses `cudaMallocManaged`, requires `libcuda.so` to be linkable) and `pinned-host-memory` (pure-Rust page-locked host buffer). Plain `cargo build --workspace` is the no-CUDA, no-linkage path and is the recommended quick check. Opt into one of the two memory features for production builds.

## Feature flag reference

Cross-crate feature taxonomy:

| Crate | Flag | Default | Effect |
|---|---|---|---|
| tensor-wasm-mem | `unified-memory` | no | Links cust; uses `cudaMallocManaged`. |
| tensor-wasm-mem | `pinned-host-memory` | no | Pure-Rust pinned host buffers (fallback). |
| tensor-wasm-exec | `async-execution` | no | Enables Wasmtime async; epoch-based interrupt. |
| tensor-wasm-wasi-gpu | `cuda` | no | Links cust for `wasi_cuda_*` host functions. |
| tensor-wasm-jit | `auto-offload` | no | Enables Cranelift→PTX JIT pipeline. |
| tensor-wasm-tenant | `mps` | no | Use NVIDIA MPS-shared context. |
| tensor-wasm-tenant | `cuda` | no | Use real CUDA contexts (vs in-process stub). |

## Per-crate quick builds

For per-crate work (faster iteration):

```sh
cargo build -p tensor-wasm-core
cargo build -p tensor-wasm-mem
cargo build -p tensor-wasm-mem --features unified-memory
cargo build -p tensor-wasm-jit --features auto-offload
cargo build -p tensor-wasm-api
```

## Tests

Three tiers:

1. **Unit tests** (no hardware): `cargo test --workspace --no-default-features`
2. **Stub-integration tests** (no hardware): `cargo test --workspace` — uses mock CUDA layer
3. **Hardware integration tests** (CUDA required): `cargo test --workspace --features unified-memory -- --include-ignored`

Hardware-only tests are marked `#[ignore = "requires CUDA"]` and skipped by default.

## Benchmarks

```sh
cargo bench --workspace
```

Criterion benchmarks land in S9 (kernel dispatch) and S19 (full suite).

## Documentation builds

```sh
cargo doc --workspace --no-deps --open
```

All public items are required to have docs (`#![warn(missing_docs)]` enforced per-crate; `#![deny(missing_docs)]` in S22).

## Make targets

The repo provides a `Makefile` for common workflows:

| Target | Description |
|---|---|
| `make build` | Build all crates (default features) |
| `make test` | Run all tests |
| `make bench` | Run all benchmarks |
| `make fmt` | Format all code |
| `make fmt-check` | Verify formatting (CI gate) |
| `make lint` | Clippy with `-D warnings` |
| `make check` | `cargo check --all-targets` |
| `make doc` | Build rustdoc |
| `make ci` | Full local CI emulation (fmt-check + lint + check + test) |
| `make clean` | `cargo clean` |

## Troubleshooting

Common build issues with copy-paste fixes:

- **`failed to run custom build command for cust`** — toolkit not installed or CUDA_ROOT not exported. See [CUDA-SETUP.md](./CUDA-SETUP.md).
- **`error: linker not found`** — install MSVC build tools (Windows) or `build-essential` (Linux).
- **``could not find `Cargo.toml`​``** — run cargo commands from the workspace root (`C:/Projects/bali/` or wherever you cloned it), not a subdirectory.
- **`error: package collision`** — `cargo clean` and rebuild; usually after a `rust-toolchain.toml` channel bump.

## CI parity

The `.github/workflows/ci.yml` workflow runs four jobs: `fmt` (cargo fmt --check), `clippy` (with CUDA stubs on `LD_LIBRARY_PATH`, default features), `test` (CUDA stubs, runs both `cargo build --workspace`, `cargo test --workspace --no-default-features`, and `cargo test --workspace`), and `actionlint`. To approximately mirror CI locally:

```sh
make ci
```

---
_Updated for tensor-wasm v0.1.0 (S2 of plan). See [ARCHITECTURE.md](../ARCHITECTURE.md) for the crate dependency graph._
