# Building Craton TensorWasm

Craton TensorWasm is a Cargo workspace of 11 crates implementing a GPU-accelerated serverless Wasm runtime. It supports three build matrices: a full CUDA host (real hardware), a CUDA stub (for CI), and a no-CUDA configuration (quick local checks). This document walks through each, plus feature flags, tests, benchmarks, docs, and CI parity.

## Prerequisites

- Rust toolchain: pinned in `rust-toolchain.toml` (currently `nightly-2026-04-03`). Rustup picks it up automatically the first time you run `cargo` in the workspace. The nightly pin exists to **align the in-repo dev/CI toolchain with cuda-oxide's own pin** (a prerequisite for the `cuda-oxide-backend` scaffold), not because the runtime code needs nightly: the only nightly feature used is `doc_cfg`, and it is gated behind `cfg(docsrs)` so it activates **only** on the docs.rs builder. Downstream consumers who depend on the published crates from crates.io build on **stable Rust ≥ 1.78** — the `rust-version` MSRV declared in every crate's `Cargo.toml`. The `rust-toolchain.toml` pin applies only to builds run from inside this checkout.
- For CUDA builds: see [CUDA-SETUP.md](./CUDA-SETUP.md).
- For no-CUDA host: nothing extra needed beyond Rust.

## Build matrix

| Config | Command | Active tensor-wasm-mem feature | Use case |
|---|---|---|---|
| No-CUDA (default) | `cargo build --workspace` | none (pure-Rust path) | Quick local check — no CUDA linkage at all |
| CUDA host | `cargo build --workspace --features tensor-wasm-mem/unified-memory` | `unified-memory` | Real hardware — `cudaMallocManaged` |
| CUDA stub (CI) | `cargo build --workspace --features tensor-wasm-mem/unified-memory` (stub `libcuda.so` on `LD_LIBRARY_PATH`) | `unified-memory` | CI build/test — links against stub libs |

Note: the **workspace root enables no default features** for the CUDA/GPU stack; some library crates have safe default-on features (listed below — `tensor-wasm-snapshot` ships `signed-snapshots` + `artifact-backing` on, and `tensor-wasm-tenant` ships `strict-cap-binding` on). `tensor-wasm-mem` ships opt-in features for memory backing — chiefly `unified-memory` (links `cust` and uses `cudaMallocManaged`, requires `libcuda.so` to be linkable). Plain `cargo build --workspace` is the no-CUDA, no-linkage path and is the recommended quick check. Opt into a memory feature for production CUDA builds.

## Feature flag reference

Cross-crate feature taxonomy (kept consistent with the feature matrix in [`../README.md`](../README.md)):

| Crate | Flag | Default | Effect |
|---|---|---|---|
| tensor-wasm-core | `otlp` | no | OpenTelemetry OTLP exporter; trace IDs propagate end-to-end. |
| tensor-wasm-mem | `unified-memory` | no | Links `cust`; uses `cudaMallocManaged`. |
| tensor-wasm-mem | `mock-cuda` | no | Hardware-free CUDA test doubles (CI-runnable rollback/drop paths). |
| tensor-wasm-mem | `cudarc-backend` | no | Opt-in `cudarc` backend spike (parallel `UnifiedBuffer`). |
| tensor-wasm-mem | `gpu-mem-pool` | no | Driver-level per-tenant GPU memory cap via `cuMemPool*`; strict-superset alias of `cudarc-backend`. |
| tensor-wasm-mem | `cuda-oxide-backend` | no | Dep-less v0.5 cust-successor scaffold module (RFC 0001). |
| tensor-wasm-exec | `cuda` | no | Real GPU kernel-launch path in `jit_dispatch`; pulls the CUDA host bridge. |
| tensor-wasm-wasi-gpu | `cuda` | no | Links `cust` for `wasi_cuda_*` host functions. |
| tensor-wasm-jit | `auto-offload` | no | Gates extra CUDA-side wiring; the Cranelift→PTX pipeline itself is always compiled. |
| tensor-wasm-jit | `cuda-oxide-backend` | no | `pliron_dialect` scaffold (pulls `pliron` from crates.io). |
| tensor-wasm-jit | `pliron-llvm-backend` | no | Stage-2 `twasm.*`→`llvm.*` rewrite; strict superset of `cuda-oxide-backend` (needs system LLVM). |
| tensor-wasm-jit | `kernel-registry` | no | Manifest verification + registry impls for the signed kernel registry. |
| tensor-wasm-jit | `differential-oracle` | no | Differential JIT correctness oracle (proptest harness). |
| tensor-wasm-snapshot | `signed-snapshots` | **yes** | HMAC-SHA256 snapshot signing/verification (v3 wire format). |
| tensor-wasm-snapshot | `artifact-backing` | **yes** | Routes snapshot writes through the unified `DiskArtifactStore` envelope (T40). |
| tensor-wasm-snapshot | `cuda` | no | GPU-side restore path into a CUDA `UnifiedBuffer`; links `cust`. |
| tensor-wasm-snapshot | `mmap` | no | `memmap2`-backed snapshot reads. |
| tensor-wasm-tenant | `strict-cap-binding` | **yes** | Gates the typed `*_strict` admin APIs (cap-to-registry binding is always enforced). |
| tensor-wasm-tenant | `cuda` | no | Use real CUDA contexts (vs in-process stub). |
| tensor-wasm-tenant | `loom` | no | Loom concurrency-model test harness. |
| tensor-wasm-api | `kernel-registry-api` | no | Compiles the `kernels` module and wires `POST/GET /kernels` into the router (B6.4). |

Note: `async-execution` is **not** a build flag — Wasmtime async + epoch interruption is always-on behaviour. NVIDIA MPS-backed shared contexts are likewise selected at runtime (env/config), not via a cargo feature; see [CUDA-SETUP.md](./CUDA-SETUP.md) and [the MPS setup guide](../docs/MPS-SETUP.md).

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
- **``could not find `Cargo.toml`​``** — run cargo commands from the workspace root (`C:/craton/tensor-wasm/` or wherever you cloned it), not a subdirectory.
- **`error: package collision`** — `cargo clean` and rebuild; usually after a `rust-toolchain.toml` channel bump.

## Platform support tiers

TensorWasm classifies host platforms into tiers based on what CI exercises and what the maintainers commit to keeping green. Lower-tier platforms may work but receive less coverage; bug reports against them are accepted, but fixes are best-effort and may depend on community patches.

| Tier | Platform | What CI runs | Notes |
|---|---|---|---|
| **Tier 1** | Linux x86_64 | Full feature matrix incl. CUDA on the S22 self-hosted runner; fmt, clippy, doc, tests with and without default features | Primary development and reference deployment target. All features supported. |
| **Tier 2** | Windows x86_64 MSVC | Default-feature build + tests in CI; no CUDA in CI | Tested but CUDA path is not exercised on Windows runners; users with a local CUDA toolkit can opt in via `tensor-wasm-mem/unified-memory`. |
| **Tier 3** | macOS (x86_64 / aarch64) | `cargo build --workspace --release` only (no tests, no CUDA features) | Compile-tested only. No CUDA backend (cust is Linux/Windows), no MPS, no GPU offload — pure-CPU paths only. Tests are not run because GitHub `macos-latest` runners are slow; the gate exists to catch portability breakage in the default workspace build. |
| **Best-effort** | aarch64-linux, riscv64, FreeBSD/OpenBSD/NetBSD | Not in CI | Community-tested. Patches accepted; regressions on these targets do not block releases. |

A Tier 1 break fails the build and blocks merging. A Tier 2 break fails the build. A Tier 3 break fails the build only at the compile level (test failures cannot fail because tests do not run there). Best-effort breaks are tracked in issues but do not block.

## CI parity

The `.github/workflows/ci.yml` workflow runs the following jobs: `fmt` (cargo fmt --check), `clippy` (with CUDA stubs on `LD_LIBRARY_PATH`, default features), `test` (CUDA stubs, runs both `cargo build --workspace`, `cargo test --workspace --no-default-features`, and `cargo test --workspace`), `macos-build` (compile-test on `macos-latest`, release profile, no CUDA features), `doc`, `openapi`, and `actionlint`. To approximately mirror CI locally:

```sh
make ci
```

---
_Updated for tensor-wasm v0.3.7. See [ARCHITECTURE.md](../ARCHITECTURE.md) for the crate dependency graph._
