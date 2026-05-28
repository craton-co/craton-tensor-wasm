<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# GPU memory quotas

TensorWasm enforces a per-tenant GPU memory cap at two layers:

1. **In-process counter (B6.5, default-on).** Every allocation that
   routes through `TenantContext::consume_bytes` / `release_bytes`
   participates in a lock-free CAS counter (`bytes_in_use`) compared
   against `memory_quota_bytes` (default 8 GiB). A consume that would
   push the counter over the cap returns
   `TensorWasmError::MemoryExhausted { requested, limit }` and leaves
   the counter unmoved. See
   [`tensor-wasm-tenant/src/context.rs`](../crates/tensor-wasm-tenant/src/context.rs).

2. **Driver-level `cuMemPool` (v0.3.8 scaffold; v0.4 wired).** A
   tenant-scoped `cuMemPoolHandle_t` configured with
   `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` matching the tenant's cap, so
   allocations past the cap fail at the driver level with
   `CUDA_ERROR_OUT_OF_MEMORY`. This is the belt-and-suspenders layer
   defending against a tenant that somehow obtained a raw CUDA driver
   handle and bypassed layer 1.

## v0.4 follow-up

| Item | Status |
|------|--------|
| `cuMemPool`-based driver enforcement | LANDED scaffold in v0.3.8; tests gated on CUDA hardware. Pool create / drop + `with_driver_enforced_gpu_cap` builder method are wired. The allocator path (`UnifiedBuffer::new_in_tenant_pool` against `cuMemAllocFromPoolAsync`) is scaffolded and returns the documented `NotSupported { feature: "pool-allocate", backing: "cudarc-backend-v0.3.8" }` sentinel. v0.4 splits the `Backing` enum so pool allocations route correctly. |
| `cuMemPoolGetAttribute` round-trip getter for the effective cap | v0.4. The driver may round `cap_bytes` internally; the v0.3.8 `cap_bytes()` getter returns the *requested* value. |
| Per-pool destroy counter | v0.4. The scaffold's drop path logs failures via `tracing::error!` but does not increment a per-tenant counter the way `CudarcUnifiedBuffer` does for `cuMemFree_v2`. |
| Unified-memory vs pooled-device-memory split | v0.4. Today every UVM allocation still routes through `cuMemAllocManaged`; v0.4 lets operators pick pool-enforcement per-tenant. |

## Feature flags

| Crate | Feature | Pulls in | Purpose |
|-------|---------|----------|---------|
| `tensor-wasm-mem` | `cudarc-backend` | `cudarc` 0.13 | The cudarc FFI surface exposing `cuMemPool*`. The `cuda_mem_pool` module is `#[cfg(feature = "cudarc-backend")]` and does not exist without this feature. |
| `tensor-wasm-tenant` | `gpu-mem-pool` | `tensor-wasm-mem` with `cudarc-backend` | Opt-in to the driver-level enforcement layer. Adds `TenantContextBuilder::with_driver_enforced_gpu_cap(cap)`. Without this feature, the `mem_pool` field collapses out of `TenantContext` entirely. |

If a downstream caller enables `tensor-wasm-tenant/gpu-mem-pool` but
somehow blocks `tensor-wasm-mem/cudarc-backend` from resolving, the
cargo feature resolver errors out at the `Cargo.toml` line:

```toml
gpu-mem-pool = ["dep:tensor-wasm-mem", "tensor-wasm-mem/cudarc-backend"]
```

— there is no way to enable the tenant-side feature without the
upstream backend that actually defines `TenantMemPool`.

## Why two layers

The in-process counter is the primary enforcement and catches the
well-behaved-allocator path: every `UnifiedBuffer::new` consults
`consume_bytes` before allocating. But the API surface accepts
arbitrary `usize` byte counts and a buggy or hostile caller could in
principle hand the CUDA driver a value the in-process counter never
saw. `cuMemPool` provides the irrevocable upper bound: the driver
itself refuses to over-allocate against the pool, regardless of what
Rust-side accounting did or didn't see.

## Hardware-gated tests

The scaffold tests in
[`crates/tensor-wasm-mem/tests/cuda_mem_pool_scaffold.rs`](../crates/tensor-wasm-mem/tests/cuda_mem_pool_scaffold.rs)
are `#[ignore]`'d because they require a working CUDA driver. Run on a
hardware-equipped host with:

```text
cargo test --features cudarc-backend -- --ignored cuda_mem_pool
```

The unit tests inside
[`src/cuda_mem_pool.rs`](../crates/tensor-wasm-mem/src/cuda_mem_pool.rs)
are pure type / API checks and run on any host with `--features
cudarc-backend` (cudarc 0.13 dlopens `libcuda` lazily, so just having
the type in the binary does not trigger a driver call).
