<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# `cuda-oxide-backend` — v0.3.1 scaffold

This crate's `cuda-oxide-backend` Cargo feature is the **opt-in scaffold** for
the v0.5 `cust` successor adopted in [RFC
0001](../../rfcs/0001-cuda-oxide-integration.md). It lands the feature flag,
the module skeleton, and the smoke-test wiring so the v0.4 unified-memory
port (per RFC 0001 "Rollout — v0.4 (parity)") has a clean target to fill in.

## What this feature does today

- Pulls in three crates from NVIDIA Labs' [cuda-oxide
  v0.1.0](https://github.com/NVlabs/cuda-oxide) release (published
  2026-05-09): `cuda-host`, `cuda-core`, `cuda-async`.
- Exposes the public type `tensor_wasm_mem::cuda_oxide_backend::CudaOxideUnifiedBuffer`
  and the free function `apply_advice` so downstream call sites can write
  backend-agnostic code today.
- Returns a documented sentinel error (`"cuda-oxide-backend: allocate not yet
  wired -- see RFC 0001 v0.4 port"`) from every `allocate`/`apply_advice`
  call. The stub is observable from tests and production telemetry rather
  than silently panicking.
- Co-exists with `unified-memory` (cust) and `cudarc-backend`. All three
  features can be enabled at once; they live in independent modules and do
  not imply each other.

## What this feature does NOT do (yet)

- Actually allocate CUDA Unified Memory. Every entry point returns the
  sentinel error.
- Wire a `cuda_host::Stream` / `cuda_host::Event` through the surface.
- Cover the kernel-authoring side of cuda-oxide (`cuda-device`,
  `cuda-macros`). Those land on the W4.5 "Path C: Rust kernels via
  cuda-oxide" track, not this one.
- Run in CI on the workspace default toolchain. See the toolchain section
  below.

## Toolchain alignment

cuda-oxide v0.1.0 pins `nightly-2026-04-03` in its own `rust-toolchain.toml`.
The workspace `rust-toolchain.toml` now pins the same channel
(`nightly-2026-04-03`, bumped 2026-05-25 from `nightly-2026-03-15` per
RFC 0001 "Toolchain plan" step 3), so enabling
`--features cuda-oxide-backend` on the workspace default toolchain just
works — no `RUSTUP_TOOLCHAIN` override is needed.

```bash
cargo build -p tensor-wasm-mem --features cuda-oxide-backend
```

The earlier per-invocation override dance (`RUSTUP_TOOLCHAIN=nightly-2026-04-03 …`
or `cargo +nightly-2026-04-03 …`) is no longer required and was removed when
the workspace toolchain caught up at v0.3.x.

## Running the smoke tests

The integration tests in `tests/cuda_oxide_smoke.rs` mirror the shape of
`tests/cudarc_smoke.rs`. Three unignored tests assert the scaffold is
wired; one `#[ignore]`d test is the v0.4 hardware round-trip target.

```bash
cargo test \
    -p tensor-wasm-mem --features cuda-oxide-backend \
    --test cuda_oxide_smoke
```

## Feature interactions

| Feature combo | Behaviour |
|---|---|
| (none) | `Box<[u8]>` host backing; cuda-oxide module absent. |
| `unified-memory` | cust-backed `UnifiedBuffer` (today's default). |
| `cudarc-backend` | cudarc spike under `crate::cudarc_backend`. |
| `cuda-oxide-backend` | cuda-oxide scaffold under `crate::cuda_oxide_backend`. |
| All three at once | Valid. Each lives in its own module; the choice is per call site. |

## Where the real work lands

- **v0.4 port:** wire `cuda_host::DeviceBuffer<u8>` (or whatever the
  published v0.2 type name is) into `CudaOxideUnifiedBuffer`. See the
  `TODO(v0.4 port)` markers in
  [`src/cuda_oxide_backend.rs`](src/cuda_oxide_backend.rs).
- **v0.5 default flip:** promote `cuda-oxide-backend` to default and drop
  `cust` per RFC 0001 "Rollout — v0.5 (default flip)", contingent on
  cuda-oxide ≥ 0.2.0 with a stable host API.

## Related docs

- [RFC 0001 — cuda-oxide as the v0.5 cust successor](../../rfcs/0001-cuda-oxide-integration.md)
- [`docs/CUDARC-SPIKE.md`](../../docs/CUDARC-SPIKE.md) — the sibling backend's spike notes
- [`docs/CUDA-SETUP.md`](../../docs/CUDA-SETUP.md) — toolkit and driver setup (gains a cuda-oxide section in a follow-up PR)
