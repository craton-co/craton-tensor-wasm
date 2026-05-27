# `cust` → `cudarc` migration spike

**Status (2026-05-27): superseded by RFC 0001. F4 toolchain bump (2026-04-03) closed the originally-blocking cust + bindgen path.**

**Original status:** spike landed for v0.2; full cutover deferred to v0.2 release cycle pending S22 runner validation. See [`RISKS.md`](RISKS.md) ("CUDA `cust` 0.3.x EOL" row) and the [Path to v1](PATH-TO-V1.md) ("Open decision #1") for the surrounding context.

**Concrete frictions surfaced (W5.9 build attempt):** building `--features cudarc-backend` against `cudarc 0.13.9` on `nightly-2026-03-15` fails because the spike code references several FFI symbols at `cudarc::driver::sys` paths that the released crate does not actually export:

- `cudarc::driver::sys::cuMemAllocManaged` — **not** exported at the `sys` root in 0.13.9 (the safe `CudaDevice::alloc_zeros` returns device-only memory; the managed-pointer FFI lives behind a different module path or requires the `cuda-12000` feature variant the spike picked).
- `cudarc::driver::sys::cuMemPrefetchAsync` — same issue.
- `cudarc::driver::sys::cuMemFree_v2` — same.
- `cudarc::driver::sys::cuMemAdvise` — same.
- `cudarc::driver::sys::CUmem_advise_enum` — exported under a different name in 0.13.x.
- `cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS` — exported under `CUresult::CUDA_SUCCESS` in 0.13.x.

The corollary on the `cust` side: `cust 0.3.2` itself fails to compile on `nightly-2026-03-15` because of a `bytemuck::PodCastError` reference removed in modern `bytemuck`. So **today, neither backend compiles cleanly on this host** — the spike's job was to surface this kind of friction before the cutover, and it has. The v0.2 cutover PR will need to (a) fix the symbol paths above against whatever cudarc minor we settle on, and (b) provide a smoke test that the orchestrator can run before flipping the default. Captured here so the cutover work has a starting point rather than re-deriving it.



**Scope of this document:** the version chosen, an API mapping table for the operations TensorWasm actually uses today, the known gaps where `cudarc`'s surface does not match `cust`'s 1:1, and a recommended cutover plan.

**Scope of the spike code:** a parallel implementation of `UnifiedBuffer` + `cuMemAdvise` + `cuMemPrefetchAsync` in `crates/tensor-wasm-mem/src/cudarc_backend.rs`, gated behind a new `cudarc-backend` feature flag. The default v0.2 build still uses `cust`; both backends coexist so a regression in one cannot mask a regression in the other.

---

## Version chosen

| Crate | Version | Default features | Enabled features | Rationale |
|---|---|---|---|---|
| `cudarc` | `0.13` | none | `driver`, `cuda-12000` | Latest 0.13.x line on crates.io as of 2026-05; `0.13` is the stable series the maintainer has been backport-fixing since late 2025. `driver` selects the CUDA Driver API surface (mirrors what `cust::sys` exposes today). `cuda-12000` pins the bindings against CUDA 12.0 headers, which matches the toolkit on the proposed S22 self-hosted runner (CUDA 12.4) and is forward-compatible with CUDA 12.x driver releases per NVIDIA's ABI policy. The `runtime` feature is deliberately *not* enabled — TensorWasm uses the Driver API exclusively (same as `cust`) so pulling in the Runtime API would double our `dlopen` surface for no benefit. |
| `cuda-host` (cuda-oxide) | `0.1` (alpha) | n/a | TBD per [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md) | Added to this table 2026-05-25. Gated behind a separate `cuda-oxide-backend` feature flag scaffolded at v0.3.1; coexists with `cudarc-backend` and the cust default through v0.4.x. Requires the `nightly-2026-04-03` toolchain override documented in `docs/CUDA-SETUP.md`. Default-pick at v0.5 is contingent on cuda-oxide reaching v0.2.0 with a stable host API; if it doesn't, `cudarc-backend` (this spike) becomes the v0.5 default. |

### Why not `0.14` / `0.15`?

cudarc's 0.14+ branch reworks the safe ergonomic layer (the `CudaSlice` / `LaunchAsync` types) and is still seeing churn on its master branch. The 0.13.x line is what every downstream crate (`candle`, `burn`, `dfdx`) has been pinning during the same window. Picking 0.13 keeps us on the well-trodden path; a follow-up bump to 0.14 once the ergonomic layer stabilises is a 1-commit change to the workspace `Cargo.toml`.

### Why not a `git = ...` pin?

The maintainer publishes releases regularly. We have no reason to ride master, and a registry pin is what `cargo-audit` / `cargo-deny` know how to check.

---

## API mapping table

The columns:

- **TensorWasm call site** — the file and (rough) function in the existing `cust` path.
- **`cust` call** — the API we use today.
- **`cudarc` equivalent** — the spike's equivalent in `cudarc_backend.rs`.
- **Mapping quality** — `1:1` clean; `wrapper` we wrote a thin adapter; `sys-level` cudarc's safe surface does not cover it so we drop to `cudarc::driver::sys`; `gap` no equivalent yet (see [Known gaps](#known-gaps)).

| TensorWasm call site | `cust` call | `cudarc` equivalent | Mapping quality |
|---|---|---|---|
| `unified::Backing::allocate` | `cust::memory::UnifiedBuffer::new(&0u8, size)` | `cudarc::driver::sys::cuMemAllocManaged(&mut raw, size, CU_MEM_ATTACH_GLOBAL)` | sys-level (cudarc 0.13 does not yet wrap `cuMemAllocManaged` in a safe API; its `CudaDevice::alloc_zeros` returns device-only memory, not managed memory) |
| `UnifiedBuffer::drop` (implicit via `cust::memory::UnifiedBuffer`'s Drop) | `cust::memory::UnifiedBuffer::drop` → `cuMemFree` | `cudarc::driver::sys::cuMemFree_v2` in our explicit `Drop` impl | sys-level (same reason: no safe wrapper for the managed-pointer path) |
| `unified::UnifiedBuffer::prefetch_to_device` | `cust::memory::UnifiedBuffer::prefetch_to_device(device_id)` | `cudarc::driver::sys::cuMemPrefetchAsync(ptr, size, device_id, null_stream)` | sys-level (cudarc exposes `cuMemPrefetchAsync` on the safe `CudaSlice`, not on raw managed pointers) |
| `unified::UnifiedBuffer::prefetch_to_host` | `cust::memory::UnifiedBuffer::prefetch_to_host()` | `cudarc::driver::sys::cuMemPrefetchAsync(ptr, size, CU_DEVICE_CPU /* = -1 */, null_stream)` | sys-level + magic-constant (cudarc 0.13 does not export `CU_DEVICE_CPU` as a named constant; we inline the documented `-1` value) |
| `advise::apply_cuda` | `cust::sys::cuMemAdvise(ptr, size, kind, device)` | `cudarc::driver::sys::cuMemAdvise(ptr, size, kind, device)` | 1:1 (same FFI shape; only the enum import path differs: `CUmem_advise` → `CUmem_advise_enum`) |
| Context / device init (today implicit in `cust::CurrentContext`) | `cust::quick_init()` at first allocation | `cudarc::driver::CudaDevice::new(ordinal)` (cached in a `OnceLock`) | wrapper (cudarc's model is explicit-device, cust's is implicit-context; the spike adapts by caching a single `Arc<CudaDevice>` for device 0) |

### Operations TensorWasm does *not* yet exercise (out of scope for the spike)

These are listed for completeness so the v0.2 cutover PR knows what else needs porting:

| `cust` call | `cudarc` equivalent | Notes |
|---|---|---|
| `cust::module::Module::from_ptx(...)` | `CudaDevice::load_ptx(ptx, module_name, &[fn_names])` | Used by `tensor-wasm-jit`. cudarc's API is more ergonomic (loads + extracts functions in one call). Will be a net simplification at the JIT call site. |
| `cust::function::Function::launch(...)` | `func.launch(LaunchConfig { ... }, params)` via the `LaunchAsync` trait | Used by `tensor-wasm-wasi-gpu`. cudarc's typed-params macro replaces the manual `void**` we build by hand; see the kernel-args marshalling work for the existing manual lowering. |
| `cust::stream::Stream::new(StreamFlags::NON_BLOCKING, None)` | `device.fork_default_stream()` or `CudaStream::new(device.clone())` | Per-instance stream isolation in `tensor-wasm-tenant`. |
| `cust::event::Event::new(EventFlags::DEFAULT)` | `device.new_event(None)` | Used in the back-pressure / future-sync path. |
| `cust::sys::cuLaunchKernel` (raw, for typed argv lowering) | `cudarc::driver::sys::cuLaunchKernel` | Identical FFI; just the import path changes. |

---

## Known gaps

Gaps where `cudarc` 0.13's surface does not match `cust`'s 1:1. None are blockers; all have a documented workaround.

### 1. No safe wrapper for `cuMemAllocManaged`

`cudarc::driver::CudaDevice::alloc_zeros::<T>(n)` returns a `CudaSlice<T>` whose backing is *device-only* memory (`cuMemAlloc`), not unified memory. There is no public `alloc_managed` / `alloc_unified` helper in 0.13. The spike drops to `cudarc::driver::sys::cuMemAllocManaged` directly and wraps the resulting pointer in our own `CudarcUnifiedBuffer`. This is the same shape `cust::memory::UnifiedBuffer` has internally, just with the safety wrapper moved into our crate. **Workaround cost:** ~30 LOC of `unsafe` in `cudarc_backend.rs`, isolated to one module, audited in this spike.

### 2. No exported `CU_DEVICE_CPU` constant

`cuMemPrefetchAsync` takes a destination device ordinal; the documented sentinel for "prefetch back to host" is `-1` (`CU_DEVICE_CPU`). cudarc 0.13 does not re-export this constant. The spike inlines the literal `-1` with a comment. **Workaround cost:** trivial; submit a follow-up PR upstream to export the constant.

### 3. Enum naming drift

`cust::sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY` vs `cudarc::driver::sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_READ_MOSTLY`. Same underlying value, different Rust path. Every `match` arm in the cust path needs a one-line edit. **Workaround cost:** mechanical sed across `advise.rs` at cutover.

### 4. Implicit-context vs explicit-device

`cust::quick_init()` retains the primary context as a thread-local; subsequent allocations work on whichever thread is current. cudarc's `CudaDevice::new(0)` returns an `Arc<CudaDevice>` that is `Send + Sync` and must be threaded through to allocation sites. The spike papers over this with a single process-wide `OnceLock<Arc<CudaDevice>>` so the public API of `CudarcUnifiedBuffer::new` matches `UnifiedBuffer::new` (no extra parameter). **Workaround cost:** at cutover, route the `Arc<CudaDevice>` through `tensor-wasm-tenant::TenantContext` so each tenant can pin to a different ordinal — this is the right long-term shape anyway. The `OnceLock` is a spike-only hack.

### 5. No `prefetch_to_host` on managed pointers in the safe API

As above (sys-level call). Once gap #1 is closed upstream, this collapses to a 1-liner on `CudaSlice`.

### 6. Drop cannot return an error

`cust::memory::UnifiedBuffer`'s Drop swallows `cuMemFree` errors. cudarc's `CudaSlice` does the same. The spike's `CudarcUnifiedBuffer::drop` calls `cuMemFree_v2` and `tracing::warn!`s on failure — same observable behaviour. Both backends share the same blind spot; nothing to migrate here, but worth noting for the v1.0 audit.

---

## Recommendation: cutover plan

> **Update (2026-05-25):** NVlabs published `cuda-oxide` v0.1.0 alpha on 2026-05-09, after this spike's recommendation was written. [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md) supersedes the v0.2-cutover suggestion below with a three-way live evaluation (`cust` / `cudarc-backend` / `cuda-oxide-backend`) across v0.3.x and v0.4.x, and a contingent default flip at v0.5. The cudarc recommendation here is **still valid as the v0.3.x default and as the v0.5 fallback if cuda-oxide isn't ready** — read the rest of this section with that framing.

**Recommendation: cut over in v0.2.** The mapping is mechanical, the safety story is no worse than today, and dragging `cust` through another release is a security-patch hazard (the EOL'd dep has no upstream owner). The spike has surfaced no architectural blocker.

**Sequencing (proposed PR breakdown for the v0.2 cycle):**

1. **PR-A — Land this spike (done).** `cudarc-backend` feature gate, parallel `CudarcUnifiedBuffer`, smoke test. Default build unchanged.
2. **PR-B — Validate on the S22 runner.** Once the S22 self-hosted CUDA runner is online, run the `#[ignore]`d tests in `tests/cudarc_smoke.rs` in `--features cudarc-backend` mode. Gate: all three ignored tests pass, no driver-leak warnings in `tracing` output.
3. **PR-C — Cut `UnifiedBuffer` over.** Replace the `cust::memory::UnifiedBuffer` backing inside `unified.rs::backing_impl` with a `CudarcUnifiedBuffer` wrapped under the *same* `Backing::Cuda` enum variant. Keep both `unified-memory` (cust) and `cudarc-backend` features valid; users can pick. This is the "soft cutover" — `unified-memory` becomes a thin alias that auto-enables `cudarc-backend`.
4. **PR-D — Port `advise::apply_cuda`.** Mechanical: change the `use cust::sys as cuda_sys;` line to `use cudarc::driver::sys as cuda_sys;` and update the enum path (gap #3).
5. **PR-E — Port `tensor-wasm-jit::ptx_emit` and `tensor-wasm-wasi-gpu` callsites.** Module load + kernel launch. This is the largest single PR; the kernel-args marshalling work (W1.1, already landed in v0.2) means the typed argv shape is already abstracted, so the swap is bounded.
6. **PR-F — Port `tensor-wasm-tenant` stream/event/context.** Route the `Arc<CudaDevice>` through `TenantContext` (closes gap #4).
7. **PR-G — Remove `cust` from the workspace.** Drop the `cust = "0.3"` line, drop the `unified-memory` feature alias, drop the conditional `Backing::Host` fallback's now-defunct cust branch. Update `RISKS.md` to mark the row **Resolved**.

**Alternative: punt to v0.3.** If the S22 runner slips or PR-E exposes a JIT/kernel-launch surprise, the spike is harmless to leave in place across a release. `cudarc-backend` stays an opt-in feature, no user-visible API change, and we get another release of soak time before cutting over the defaults.

**Plan B if cudarc itself becomes unmaintained:** the `cudarc::driver::sys` layer we depend on is a thin bindgen wrapper over `cuda.h` — vendoring it (or generating our own with `bindgen` against the toolkit on the runner) is a 1-day fallback. This is the same Plan B `cust` itself would warrant.

---

## How to exercise the spike locally

```bash
# Compile-only check that the feature graph is wired (does not link CUDA libs
# on hosts without a toolkit; cudarc's `driver` feature dlopens lazily).
cargo build -p tensor-wasm-mem --features cudarc-backend

# Run the smoke test without hardware (the unignored tests confirm the
# cudarc-backend code compiles and that the public types from
# `tensor_wasm_mem::cudarc_backend` are reachable):
cargo test -p tensor-wasm-mem --features cudarc-backend --test cudarc_smoke

# Run the hardware-dependent tests on a CUDA host:
cargo test -p tensor-wasm-mem --features cudarc-backend --test cudarc_smoke -- --ignored
```

---

## Related docs

- [`RISKS.md`](RISKS.md) — the cust EOL row (now marked "spike landed").
- [`PATH-TO-V1.md`](PATH-TO-V1.md) — Open decision #1, v0.2 exit criteria.
- [`CUDA-SETUP.md`](CUDA-SETUP.md) — toolkit / driver / runner expectations.
- `crates/tensor-wasm-mem/src/cudarc_backend.rs` — the spike implementation.
- `crates/tensor-wasm-mem/tests/cudarc_smoke.rs` — the smoke test.
