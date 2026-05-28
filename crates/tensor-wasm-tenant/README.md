# tensor-wasm-tenant

Multi-tenant CUDA context management for Craton TensorWasm. Provides `TenantContext`, which bundles a per-tenant CUDA context, stream, and memory pool, and `TenantRegistry`, which maps `TenantId` values to live contexts and handles their lifecycle. Designed to ride on NVIDIA MPS when available, falling back to per-context isolation otherwise so workloads from different tenants never observe one another's GPU state.

## Feature flags

| Flag | Default | Description |
|---|---|---|
| `mps` | no | Prefer NVIDIA MPS-backed shared contexts when `/tmp/nvidia-mps` is present. |
| `cuda` | no | Use real CUDA contexts (vs in-process stub for unit tests). |
| `loom` | no | Swap `std::sync::atomic::AtomicU64` for `loom::sync::atomic::AtomicU64` on the `consume_bytes` / `release_bytes` CAS hot path so `tests/loom_consume_release.rs` can exhaustively explore the two-thread interleavings. Pure model-checking — do not enable in production builds. |
| `strict-cap-binding` | no | Bind `RegistryAdminCapability` and `TenantCapability` to the `TenantRegistry` that minted them. With the flag off, caps are an opaque "you-hold-*some*-cap" token and a cap minted by registry A is accepted by registry B (the v0.3 default). With the flag on, foreign caps are rejected at the cap-check site. See [Cap binding](#cap-binding) below. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Cap binding

`TenantRegistry::new()` mints a `RegistryAdminCapability`; every successful `register_with_capability` call mints a per-tenant `TenantCapability`. Both gate the hot-path mutation methods (admin enumeration / eviction for the former, per-tenant `consume_bytes_with_capability` / `release_bytes_with_capability` for the latter). The default mode and the strict mode differ only in how those caps relate to the *specific* registry instance they came from.

**Default mode (no `strict-cap-binding`):** capabilities are opaque tokens. A cap minted by registry A is accepted by registry B, and a `TenantCapability` for `TenantId(7)` minted against registry A can drive the `TenantId(7)` context in registry B. The security boundary this enforces is "the caller must hold *some* cap minted by this crate" — sufficient to prevent unauthenticated callers (e.g. user-facing API handlers that never see a cap) from enumerating tenants, but **not** sufficient to separate two independent registries running in the same process. Embedders that host more than one `TenantRegistry` are responsible for keeping the registry handles per-trust-domain so a subsystem only ever sees the cap it should.

**Strict mode (`--features strict-cap-binding`):** every cap carries an `Arc<()>` token cloned from the minting registry's per-instance allocation. The admin / quota check compares with `Arc::ptr_eq`, so a cap minted by registry A is rejected against registry B even if both registries happen to allocate at the same heap address — each `Arc::new(())` is a distinct allocation. Foreign-cap admin calls surface as [`RegistryError::CapabilityFromForeignRegistry`] from the `*_strict` admin variants (and as `None` / `0` / empty `Vec` from the legacy `Option`/`usize`-returning ones, for backward source-compat). Foreign-cap quota calls surface as [`TensorWasmError::TenantIsolationViolation`] from `consume_bytes_with_capability` / `release_bytes_with_capability`. Recommended for multi-tenant deployments; v0.4 will flip this on by default.

## Dependencies

Internal workspace crates:
- `tensor-wasm-core` — shared `TenantId` / `InstanceId` newtypes and the `TensorWasmError` enum returned by quota enforcement.

External crates (pinned at workspace root):
- `tokio` — async runtime for context lifecycle tasks.
- `thiserror` — derive macro for tenant-level errors.
- `tracing` — structured spans/events for context create/destroy and the underflow / pop-failure warnings.
- `dashmap` — concurrent map of `TenantId` to `Arc<TenantContext>`; combined with the `AtomicU64` counters on `TenantContext`, this is the entirety of the registry's concurrency story — there are no mutexes on the hot path.
- `cust` (optional, behind the `cuda` feature) — CUDA driver-API bindings; provides the primary-context API used by `ContextIsolated` tenants.
