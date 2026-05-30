# Hardware verification run — 2026-05-30

First real-silicon validation of the WASM→GPU stack. Until this run every
CUDA path was gated behind `#[ignore = "requires CUDA hardware"]` /
`#[cfg(feature = "cuda")]` and had **never executed on a GPU** — and, as it
turns out, the `--features cuda` host path had never even *compiled*.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 |
| Compute capability | **7.5 (Turing)** |
| Driver | 591.86 (CUDA 13.1 driver API) |
| CUDA Toolkit | 13.2 (`nvcc` 13.2.78, `cuda.lib`/`cudart.lib`/`nvrtc.lib` present) |
| Host | Windows 11, MSVC BuildTools 14.44, Rust nightly-2026-04-03 |
| libclang | 18.1.1 via `pip install --user libclang` (needed by `cust_raw` bindgen; not previously present) |

Reproduce with `scripts/run-gpu-tests.{sh,ps1}` (auto-detects CUDA + libclang,
runs every suite single-threaded, logs to `bench-results/gpu-run/`).

## Results

### ✅ `tensor-wasm-mem --features unified-memory` (cust path) — PASS
`cust` 0.3.2 / `cust_raw` 0.11.3 **build cleanly against CUDA 13.2 + libclang 18**
(the single biggest unknown). On hardware:
- `cust_unified_buffer_snapshot_round_trip_on_device` … **ok** — real
  `cuMemAllocManaged` allocate → write → snapshot → restore round-trip.

### ◑ `tensor-wasm-mem --features gpu-mem-pool` (cudarc path) — 4 PASS / 1 FAIL
- `cudarc_backend::allocate_and_drop_small_buffer` … ok
- `cudarc_backend::device_cache_returns_same_arc_for_same_ordinal` … ok
- `driver_pin_matches_requested_cap` … ok
- `under_cap_allocation_through_pool_succeeds` … ok
- **`over_cap_allocation_through_pool_is_rejected_by_driver` … FAIL** — see BUG-1.

`cudarc` 0.13.9 (`features = ["driver","cuda-12000"]`) loads the 13.1 driver
fine — the CUDA 12 driver-API bindings are forward-compatible.

### ✗ `tensor-wasm-wasi-gpu --features cuda` — DOES NOT COMPILE — see BUG-2

## BUG-1 — driver-level per-tenant GPU memory cap (T39) is not enforced

`tests/cuda_mem_pool_driver_pin.rs::over_cap_allocation_through_pool_is_rejected_by_driver`
asks for 128 MiB from a pool created with a 64 MiB "cap" and expects the driver
to refuse with `CUDA_ERROR_OUT_OF_MEMORY`. **The allocation succeeds.**

Root cause (`crates/tensor-wasm-mem/src/cuda_mem_pool.rs:215`): the cap is wired
as `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, cap)`. But
`RELEASE_THRESHOLD` is a *memory-retention hint* — it controls how much freed
memory the pool caches before returning it to the OS. **It is not an allocation
ceiling**, and CUDA memory pools expose no hard max-size attribute. So the T39
threat model (a tenant with a raw driver handle bypassing the in-process
`consume_gpu_bytes` counter) is **not** closed at the driver level, contrary to
the module's "driver pin LANDED" status note and `docs/GPU-QUOTAS.md`.

Severity: this is a security-relevant correctness defect in a multi-tenant
isolation feature, and it would have shipped silently — the test that catches it
only runs on hardware. Fix options: enforce the cap host-side in
`TenantMemPool::allocate` (reject when `live + size > cap` before
`cuMemAllocFromPoolAsync`), and/or back the pool with a fixed-size virtual-memory
reservation (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`) whose mapped
size is the hard cap. Either way, drop the claim that `RELEASE_THRESHOLD` is the
enforcement mechanism.

## BUG-2 — the `--features cuda` host path has bit-rotted (never compiled)

`cargo test -p tensor-wasm-wasi-gpu --features cuda` fails to build with six
`error[E0063]: missing field 'device_ptr' in initializer of
'device_mem::DeviceMemEntry'`:

```
crates/tensor-wasm-wasi-gpu/src/host.rs:2596
crates/tensor-wasm-wasi-gpu/src/host.rs:2641
crates/tensor-wasm-wasi-gpu/src/host.rs:2686
crates/tensor-wasm-wasi-gpu/src/host.rs:2870
crates/tensor-wasm-wasi-gpu/src/host.rs:2877
crates/tensor-wasm-wasi-gpu/src/host.rs:2925
```

A `device_ptr` field was added to `DeviceMemEntry` but the six `cfg(cuda)`-gated
constructors were never updated. Because there is no GPU CI runner, nothing ever
compiles this feature — so the headline path (Wasm → `wasi:cuda` →
`cuLaunchKernel` → readback) could not even be built, let alone run. This is the
strongest argument for the GPU CI lane: a plain `cargo build --features cuda` in
CI would have caught it.

Fix: initialize `device_ptr` at each site with the device pointer the
allocation/registration already has in scope. (In progress.)

## sm_75 support (this GPU is Turing, the kernels target Ampere)

`kernels/vector_add.ptx`, the test fixture, and `ptx_emit`'s `DEFAULT_TARGET`
all hard-code `.target sm_80`. sm_80 PTX is rejected by the driver JIT on this
sm_75 card, so `vector_add_end_to_end_real_ptx_real_kernel` self-skips. The
kernel body is capability-agnostic (no wmma/tensor-core ops), so sm_75 variants
were added:
- `kernels/vector_add_sm75.ptx`
- `crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add_sm75.ptx`

Remaining: make `register_real_kernel` / the emitter pick the sm_75 fixture when
the device's compute capability is < 8.0, so the end-to-end launch proof runs on
Turing-class hardware. (In progress, blocked on BUG-2 — the test crate must
compile first.)

## Status of the loop

| Stage | State |
|---|---|
| cust builds on CUDA 13.2 | ✅ proven |
| cudarc builds + runs on GPU | ✅ proven (1 real bug found) |
| cust unified-memory round-trip on GPU | ✅ proven |
| `--features cuda` host path compiles | ❌ BUG-2 (fix pending) |
| Real kernel launch + verified output | ⏳ blocked on BUG-2 + sm_75 wiring |
| `--features cuda` benches on GPU | ⏳ blocked on BUG-2 |
| Self-hosted GPU CI lane active | ⏳ `gpu.yml` still needs a registered `[self-hosted, gpu]` runner |
