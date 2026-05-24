# Craton TensorWasm — Risk Register

Living document tracking architectural risks, upstream pinning decisions, and known limitations for maintainers. Updated alongside `CHANGELOG.md` releases.

Last updated: 2026-05-24 (v0.1.0)

---

## Wasmtime pin

**Status:** pinned at `25.x` in `Cargo.toml` workspace dependencies.

**Why:** Wasmtime 26+ revises the `LinearMemory` / `MemoryCreator` traits used by `tensor-wasm-mem`, and breaks the `cranelift-codegen` 0.111 ABI relied on by `tensor-wasm-jit`. A bump requires:

1. Re-validating `tensor-wasm-mem::wasm_memory::TensorWasmMemoryCreator` against the new trait shape.
2. Updating `tensor-wasm-jit::clif_lower` for any cranelift IR changes.
3. Re-running the snapshot round-trip suite (snapshot bytes may shift if metadata layout drifts).
4. Re-validating WASI Preview 2 component-model integration in `tensor-wasm-wasi-gpu`.

**Dependabot ignores major bumps** for `wasmtime` and `wasmtime-wasi`. See `.github/dependabot.yml`.

**Owner:** runtime maintainers.

---

## CUDA `cust` 0.3.x EOL

**Status:** workspace pins `cust = "0.3"`; upstream maintenance is stalled.

**Risk:** no security or compatibility patches; CUDA 13+ may break without warning.

**Mitigations under evaluation:**
- Replace with `cudarc` (active, well-maintained, slightly different API surface).
- Maintain an internal fork of `cust` for security backports if migration is deferred.

**Owner:** GPU integration maintainers.

---

## Kernel-args marshalling (v0.1.0)

**Status:** the `wasi:cuda/host@0.1.0` `launch` host function accepts an `(args_ptr, args_len)` pair describing an opaque byte blob in the guest's linear memory. In v0.1.0 only the **zero-argument launch shape** reaches `cuLaunchKernel`; a non-empty buffer is rejected with `AbiError::KernelArgsUnsupported` (wire code `-10`) after the bounds-check passes.

**Why the restriction exists:** `cust 0.3.x`'s `launch!` macro takes statically-typed kernel parameters at the call site. Synthesizing a dynamic argv from a raw guest byte buffer requires bypassing the macro and calling `cuLaunchKernel` directly with a `void**`-style argument array — and that needs a frozen per-arg packing format (type tags, alignment rules, GPU-side pointer translation for `device_ptr` arguments) that has not yet been designed.

**Contract:**

- A malformed pointer (negative, overflow, out-of-bounds) returns `AbiError::InvalidPointer` (`-2`). This check runs FIRST so a malicious guest cannot trade a `MemoryFault` for the friendlier "unsupported" code.
- A well-formed, in-bounds, non-empty buffer returns `AbiError::KernelArgsUnsupported` (`-10`).
- An empty (`args_len == 0`) buffer proceeds to launch normally — this is the only shape that exercises the CUDA path in v0.1.0.

`AbiError::KernelArgsUnsupported` is intentionally distinct from `AbiError::InvalidArgs` so guests debugging "my launch with parameters fails" see "host can't pass this to CUDA yet" rather than the misleading "your input is malformed."

**Mitigation:** dynamic argv marshalling (BAL-422) is a v0.2 effort. Tests `launch_with_inbounds_args_returns_kernel_args_unsupported` and `launch_with_oob_args_returns_invalid_pointer` in `crates/tensor-wasm-wasi-gpu/src/host.rs` pin both halves of the contract.

**Owner:** GPU integration maintainers.

---

## Auto-offload coverage

**Status:** `tensor-wasm-jit` implements arg-passing trampoline + real PTX register allocation (v0.1.0). Op-classification taxonomy is comprehensive for i32/i64/f32/f64 primitives only.

**Known gaps:**
- v128 (SIMD) types not yet marshalled — offload candidates using v128 are rejected by the detector.
- Reference types (externref/funcref) not supported.
- Multi-value returns beyond 1 are not yet handled by the trampoline.

**Mitigation:** the detector rejects unsupported shapes at analysis time; rejected functions stay on the CPU path (correct, just no speedup).

**Owner:** JIT maintainers.

---

## Snapshot decompressed-size cap

**Status:** `tensor-wasm-snapshot::reader` enforces a `MAX_DECOMPRESSED_BYTES` cap (default 256 MiB, configurable via `SnapshotReader::with_max_decompressed`).

**Why:** zstd ratios on adversarial input can reach 1000× — without a hard cap, a small malicious snapshot can OOM the host.

**Tuning:** raise via builder if legitimate snapshots exceed the default; never disable.

**Owner:** snapshot maintainers.

---

## tensor-wasm-api authentication surface

**Status:** v0.1.0 ships a bearer-token gate via `TENSOR_WASM_API_TOKENS` env var and a `X-TensorWasm-Tenant` header for tenant scoping.

**Limitations:**
- Static token allowlist (no token rotation, no per-token scopes).
- No mTLS, no OIDC, no rate limiting per token.
- `X-TensorWasm-Tenant` is trusted; cross-tenant isolation depends on operators not exposing the API to untrusted clients without an upstream auth proxy.

**Recommendation:** deploy behind an authenticating reverse proxy (Cloudflare Access, AWS ALB + Cognito, OAuth2 Proxy) for any non-internal use.

**Owner:** API + platform maintainers.

---

## S22 deferred work

The following items from the audit cycle remain open at v0.1.0:

- **Differential testing** of tensor-wasm-jit against the wasmtime CPU path beyond the unit-test surface.
- **Snapshot fuzz harness** for structure-aware mutation of valid snapshots (a byte-fuzz target exists at `fuzz/fuzz_targets/fuzz_snapshot_restore.rs`).
- **End-to-end multi-tenant load test** (1000 cold starts/sec SLO claim is unverified).
- **`cust` migration plan** to `cudarc`.

Each is tracked as a GitHub issue with the `risk-register` label.

---

## How to update this document

When a new risk surfaces (e.g., a CVE in a transitive dep, a new architectural constraint, a missed coverage gap):

1. Add a new `##` section with status, why, mitigations, owner.
2. Reference it from `CHANGELOG.md` under the relevant release.
3. If it touches dependency policy, update `.github/dependabot.yml` to match.

Stale entries should be marked **Resolved (date)** rather than deleted, so the historical context survives.
