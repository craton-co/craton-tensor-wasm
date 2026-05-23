# Building Project Bali

Bali is a Cargo workspace of 10 crates implementing a GPU-accelerated serverless Wasm runtime. It supports three build matrices: a full CUDA host (real hardware), a CUDA stub (for CI), and a no-CUDA configuration (quick local checks). This document walks through each, plus feature flags, tests, benchmarks, docs, and CI parity.

## Prerequisites

- Rust toolchain: pinned in `rust-toolchain.toml` (currently `nightly-2026-03-15`). Rustup picks it up automatically the first time you run `cargo` in the workspace.
- For CUDA builds: see [CUDA-SETUP.md](./CUDA-SETUP.md).
- For no-CUDA host: nothing extra needed beyond Rust.

## Build matrix

| Config | Command | Active bali-mem feature | Use case |
|---|---|---|---|
| CUDA host | `cargo build --workspace` | `unified-memory` (default) | Real hardware — `cudaMallocManaged` |
| CUDA stub (CI) | `cargo build --workspace` (stub `libcuda.so` on `LD_LIBRARY_PATH`) | `unified-memory` (default) | CI build/test — links against stub libs |
| No-CUDA | `cargo build --workspace --no-default-features --features bali-mem/pinned-host-memory` | `pinned-host-memory` | Quick local check — no CUDA linkage at all |

Note: the `unified-memory` feature on `bali-mem` is on by default. It activates the `cust`-backed `UnifiedBuffer` and requires `libcuda.so` to be linkable — either real CUDA, or the empty stub libraries set up by the CI workflow. To build without any CUDA linkage, drop default features and opt into the `pinned-host-memory` fallback (a regular `Vec<u8>` aligned for page-locked access).

## Feature flag reference

Cross-crate feature taxonomy:

| Crate | Flag | Default | Effect |
|---|---|---|---|
| bali-mem | `unified-memory` | yes | Links cust; uses `cudaMallocManaged`. |
| bali-mem | `pinned-host-memory` | no | Pure-Rust pinned host buffers (fallback). |
| bali-exec | `async-execution` | yes | Enables Wasmtime async; epoch-based interrupt. |
| bali-wasi-gpu | `cuda` | no | Links cust for `wasi_cuda_*` host functions. |
| bali-jit | `auto-offload` | no | Enables Cranelift→PTX JIT pipeline. |
| bali-tenant | `mps` | no | Use NVIDIA MPS-shared context. |
| bali-tenant | `cuda` | no | Use real CUDA contexts (vs in-process stub). |

## Per-crate quick builds

For per-crate work (faster iteration):

```sh
cargo build -p bali-core
cargo build -p bali-mem
cargo build -p bali-mem --features unified-memory
cargo build -p bali-jit --features auto-offload
cargo build -p bali-api
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
_Updated for bali v0.1.0 (S2 of plan). See [ARCHITECTURE.md](../ARCHITECTURE.md) for the crate dependency graph._
