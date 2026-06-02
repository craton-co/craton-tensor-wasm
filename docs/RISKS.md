# Craton TensorWasm — Risk Register

Living document tracking architectural risks, upstream pinning decisions, and known limitations for maintainers. Updated alongside `CHANGELOG.md` releases.

Last updated: 2026-05-28 (v0.3.7 — workspace version bump)

---

## Wasmtime pin

**Status:** pinned at `45.x` in `Cargo.toml` workspace dependencies (bumped from `25.x` to clear RUSTSEC-2026-0096, a critical aarch64 Cranelift sandbox-escape advisory affecting <36.0.7 / 37–41 / <42.0.2).

**Why bumps are handled deliberately:** a Wasmtime major bump can revise the `LinearMemory` / `MemoryCreator` traits used by `tensor-wasm-mem` and the `cranelift-codegen` ABI relied on by `tensor-wasm-jit`. Every bump requires:

1. Re-validating `tensor-wasm-mem::wasm_memory::TensorWasmMemoryCreator` against the new trait shape.
2. Updating `tensor-wasm-jit::clif_lower` for any cranelift IR changes.
3. Re-running the snapshot round-trip suite (snapshot bytes may shift if metadata layout drifts).
4. Re-validating WASI Preview 2 component-model integration in `tensor-wasm-wasi-gpu`.

**Dependabot ignores major bumps** for `wasmtime`. See `.github/dependabot.yml` and the cadence policy in `docs/WASMTIME-UPGRADE.md`.

**Owner:** runtime maintainers.

---

## bincode unmaintained (RUSTSEC-2025-0141)

**Status:** advisory acknowledged and ignored in both `deny.toml` (`[advisories].ignore`) and `.cargo/audit.toml` — keep the two lists in sync.

**Why:** `bincode` 2.x is unmaintained. The team permanently ceased development after a doxxing/harassment incident, and there is no safe upgrade (1.3.3 is the "complete" 1.x line; 2.x has no successor). It is a core, non-optional serialization dependency — snapshot v3/v4 payloads (`tensor-wasm-snapshot`) and JIT kernel envelopes (`tensor-wasm-jit`) encode through it.

**Risk:** low. The advisory is informational (unmaintained), not a vulnerability. No untrusted bytes are bincode-decoded without a prior HMAC/signature check (snapshots and kernel manifests are authenticated before decode), so a latent parser bug is not directly attacker-reachable.

**Mitigation / tracking:** migrate to a maintained codec (`postcard` or `bitcode`) the next time the snapshot / JIT wire formats take a breaking bump. Both ignore lists carry a back-reference to this entry.

**Owner:** runtime maintainers.

---

## CUDA `cust` 0.3.x EOL

**Status:** spike landed under `--features cudarc-backend`; full migration pending the v0.5 cutover decision per RFC 0001. Workspace pins `cust = "0.3"` as the default; `cudarc = "0.13"` is available behind the per-crate `cudarc-backend` feature flag in `tensor-wasm-mem`. Both backends coexist through v0.4 for backward compat; the default flip is re-scoped to v0.5. Upstream `cust` maintenance is stalled.

**Risk:** no security or compatibility patches on the cust path; CUDA 13+ may break without warning. The cudarc-backend spike confirms the migration is viable (API mapping is ~95% one-to-one — see `docs/CUDARC-SPIKE.md`), but full cutover still needs a runner pass on real hardware and a callsite sweep across `tensor-wasm-mem::unified`, `advise`, `pool`, and `wasm_memory`.

**Mitigations under evaluation:**
- Cut the default backend over (to `cudarc` or `cuda-oxide`) at v0.5 once the S22 runner validates the spike end-to-end (see `docs/CUDARC-SPIKE.md` and RFC 0001 for the proposed cutover plan and the contingent-default approach).
- Maintain an internal fork of `cust` for security backports if the cutover slips past v0.5.

A third option has appeared since W1.2 wrote the spike: NVlabs `cuda-oxide` v0.1.0 alpha (released 2026-05-09) is now under evaluation per [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md), which proposes a three-way live evaluation with a contingent v0.5 default flip.

**Cross-cutting verification gap:** all three backends (cust, cudarc, cuda-oxide) share the property that their allocation / advise / prefetch paths are exercised on hosted CI only against the link-time stub libraries — the real driver calls have never run in CI. The complete inventory of these written-but-unverified-on-hardware paths, and the gated [`.github/workflows/gpu.yml`](../.github/workflows/gpu.yml) lane that closes the gap, is in [`docs/HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md).

**Owner:** GPU integration maintainers.

---

## Kernel-args marshalling (v0.2)

**Status:** the `wasi:cuda/host@0.1.0` `launch` host function accepts an `(args_ptr, args_len)` pair describing a tagged-argv byte blob in the guest's linear memory. As of v0.2 a non-empty buffer is parsed into a typed `Vec<LoweredArg>` (see `crates/tensor-wasm-wasi-gpu/src/kernel_args.rs`) and lowered into a `void**` parameter array that flows directly into a raw `cust::sys::cuLaunchKernel` call. The earlier v0.1.0 "all non-empty args rejected" contract is gone.

**Supported argument types** (each carries a 1-byte tag + value bytes; full table in the `kernel_args` module docs):

- Scalars: `i32` (tag `0x01`), `i64` (`0x02`), `f32` (`0x03`), `f64` (`0x04`), `u32` (`0x05`), `u64` (`0x06`).
- Pointer args: `ptr` (tag `0x07`), encoded as `(u32 guest_offset, u32 byte_len)`. The host bounds-checks `[guest_offset, guest_offset + byte_len)` against the caller's linear memory and resolves the offset into a raw host pointer. Under CUDA Unified Memory that pointer doubles as a device address.

**Sanity caps** (in `kernel_args::MAX_KERNEL_ARGS` / `MAX_KERNEL_ARGS_BYTES`): 128 args per launch, 4 KiB of wire-format argv per launch. Buffers above either cap surface as `AbiError::KernelArgsUnsupported` — the variant is preserved as a *fallback for size-cap rejections only*, not for "any args at all" as in v0.1.0.

**Contract:**

- A malformed outer pointer (negative, overflow, OOB) → `AbiError::InvalidPointer` (`-2`). This check runs FIRST so a malicious guest cannot trade a `MemoryFault` for any of the softer codes.
- A buffer above the size or arg-count caps → `AbiError::KernelArgsUnsupported` (`-10`).
- A buffer with an unknown tag byte or a truncated record → `AbiError::InvalidArgs` (`-9`).
- A buffer whose pointer arg points outside guest linear memory → `AbiError::InvalidPointer` (`-2`).
- Otherwise the parsed argv flows into `cuLaunchKernel` on CUDA builds (returning `0`) or is recorded into `WasiCudaContext::last_lowered_args` and surfaces as `AbiError::NotAvailable` on no-CUDA builds.

**Out of scope for v0.2:** explicit device-only allocations (the guest cannot today get a `device_ptr` from `cuMemAlloc`; all pointer args reach the kernel via UVM); structured-tag formats beyond the listed primitives (no v128 SIMD, no `struct` args, no array of pointers). Those expand the format additively and reuse the `KernelArgsUnsupported` fallback for "host doesn't accept this tag yet."

**Tests:** `crates/tensor-wasm-wasi-gpu/src/kernel_args.rs::tests` cover the parser unit-by-unit; `crates/tensor-wasm-wasi-gpu/src/host.rs::tests` exercise the unknown-tag / OOB / oversized branches through the launch host fn; `crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs` covers scalar + pointer + mixed argv end-to-end through wasmtime (CUDA-only assertions are `#[ignore]`).

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

## Snapshot authenticity (no signature on snapshot bytes)

**Status:** **Closed (v0.3.6):** HMAC-SHA256 signing landed behind the `signed-snapshots` feature; opt-in via `with_hmac_sha256_key`.

**Why this was on the register:** the v0.3.5 audit flagged that the on-wire `crc32` field is integrity-only — it catches bit-flips but does not authenticate the byte source. A malicious snapshot crafted with a matching CRC could be restored by a v0.1.0–v0.3.5 reader without any way for the operator to tell. The audit graded this MEDIUM (mitigated in practice by storing snapshots behind authenticated transports + ACLs; not mitigated for an operator-side compromise of the snapshot store).

**What landed in v0.3.6:**

- New wire v3 = v2 + `[signature_kind: u8][signature: 32 bytes]` trailer (`signature_kind = 1` is `HMAC-SHA256(key, v2_payload)`).
- `SnapshotWriter::with_hmac_sha256_key(key)` opts in to v3 emission; the default writer still emits v2 for backward compatibility (existing archives remain readable).
- `SnapshotReader` accepts both v2 and v3 by default once a key is configured; `SnapshotReader::require_signature()` rejects v2 outright for deployments that have completed the rollout.
- CLI: `--hmac-key-file PATH` on `snapshot save`/`restore`; `--require-signature` on `restore`.
- API: `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` (hex) and `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE` (bool) env vars wired into `AppConfig`. The snapshot HTTP routes themselves are not yet exposed; the config picks the key up automatically when they ship.
- Feature gate: `signed-snapshots`, default on. Operators who explicitly do not want the codepath compiled in can `--no-default-features` it off.

**Migration path:** [`docs/SNAPSHOT-COMPATIBILITY.md` — v2 → v3 migration](./SNAPSHOT-COMPATIBILITY.md#v2--v3-migration-signed-snapshots) documents the four-step rollout (provision key → configure reader → configure writer → flip to strict mode) and the cross-tier ordering for key rotation.

**Residual risk:** the default writer still emits unsigned v2 blobs; an operator who never opts in keeps the v0.3.5 posture. The deployment-side recommendation is to provision a key and reach Step 4 (`require_signature = true`) before exposing the snapshot HTTP routes to untrusted networks.

**Owner:** snapshot maintainers.

---

## tensor-wasm-api authentication surface

**Status:** the v0.1.0 bearer-token gate (`TENSOR_WASM_API_TOKENS` + `X-TensorWasm-Tenant`) has been hardened over the v0.2–v0.3.7 line. The items the v0.1.0 register flagged as open are now mitigated:

**Mitigated since v0.1.0:**
- **Per-token rate limiting (W1.4, v0.2):** `TENSOR_WASM_API_RATE_LIMIT_QPS` / `TENSOR_WASM_API_RATE_LIMIT_BURST` enable a per-bearer-token token bucket that returns `429` with `error.kind = rate_limited` and a `Retry-After` header. This replaces the old process-wide `ConcurrencyLimitLayer(64)` workaround. Unset/`0` keeps the limiter disabled (v0.1 behaviour). See [`crates/tensor-wasm-api/API.md#per-token-rate-limiting`](../crates/tensor-wasm-api/API.md).
- **Per-tenant scoped bearer tokens (W2.1, v0.4):** tokens carry a `:tenant=*` or `:tenant=1,2,3` scope clause; invoke routes refuse cross-tenant access with `403` `error.kind = tenant_scope_denied`. `X-TensorWasm-Tenant` is now enforced against the token scope rather than blindly trusted. Bare (unscoped) tokens still authenticate but are coerced to wildcard scope and emit a startup deprecation warning (removal planned v1.0 — see [`docs/MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) §3).
- **Structured audit log (W2.2, v0.4):** every state-mutating route emits one JSON record to the sink selected by `TENSOR_WASM_API_AUDIT_LOG` (stdout / `file:<path>` / `none`); 403 scope denials and 429 rate-limit rejections are captured. Schema and rotation in [`docs/AUDIT-LOG.md`](AUDIT-LOG.md).
- **mTLS deployment guide (W2.8, v0.4):** an mTLS-terminating reverse proxy can front the gateway and forward the client-cert Subject DN via `X-Forwarded-Client-Cert`, which the audit middleware records as `client_cert_subject`. Recipe in [`docs/deployment/mtls.md`](deployment/mtls.md).

**Residual risk (still open):**
- No built-in token rotation and no OIDC; the allowlist is still static env-var configuration.
- mTLS is proxy-fronted only (Architecture B). Self-terminated mTLS (`serve_tls()`, Architecture A) is not yet implemented.
- The `X-Forwarded-Client-Cert` header is parsed unconditionally — there is no trusted-proxy CIDR allowlist yet, so a caller that can reach the listener directly can spoof `client_cert_subject` (same shape as `X-Forwarded-For` spoofing). The mitigation is to bind the gateway to a private network and let only the trusted proxy reach it. See [`docs/AUDIT-LOG.md`](AUDIT-LOG.md) §6.2.

**Recommendation:** for any non-internal use, still deploy behind an authenticating / mTLS-terminating reverse proxy (Cloudflare Access, AWS ALB + Cognito, OAuth2 Proxy, Envoy) and bind the gateway to a private network so the forwarded-header trust boundary holds.

**Owner:** API + platform maintainers.

---

## Kernel registry shared-read model (deployment-global, operator-curated)

**Status:** **Accepted design decision (signed off in the v0.4 certification audit).** The kernel registry is a *deployment-global, operator-curated namespace*, not a per-tenant store. Publishing is gated by `KernelPublishTokens`, but reading is **not** tenant-scoped: any authenticated tenant in the deployment can list every published manifest and resolve any kernel's full PTX source.

- `list_kernels` (`crates/tensor-wasm-api/src/kernels.rs`, the `GET /kernels` handler) admits any authenticated tenant and performs no tenant filtering — it captures the `TenantId` extension only to assert the routing stack, then returns `list_paginated(offset, limit)` across the whole registry.
- `resolve_kernel` (same file, the `GET /kernels/{name}/{version}` handler) likewise admits any authenticated tenant and returns the manifest **plus the full `ptx_text`** for any registered kernel, regardless of which tenant published it.

The registry is keyed by a single per-deployment HMAC key (`TENSOR_WASM_API_KERNEL_HMAC_KEY`); there is no per-tenant key and no tenant/visibility dimension on the manifest.

**Why (rationale):** the registry is intended as an operator-scoped, shared catalog of kernels for a deployment — a curated set of PTX that the operator (gated by publish tokens) makes available to the tenants it hosts. One HMAC key per deployment matches that model: the operator controls what gets published, and all tenants in the deployment draw from the same shared catalog. Per-tenant read filtering would contradict the "shared catalog" purpose for the deployments this is built for.

**Trust boundary:** this shared-read model is acceptable **only when all tenants in a deployment are mutually trusted to read each other's kernel PTX.** A tenant can enumerate and download the PTX source of every kernel any other tenant published. Deployments whose tenants are mutually untrusting **must not** rely on the registry to hold confidential kernels — published PTX in such a deployment must be treated as readable by every co-located tenant.

**Residual risk:** no per-tenant kernel confidentiality. Tenant B can list and resolve (including full PTX) any kernel Tenant A published. For mutually-trusting tenants this is by design and carries no isolation regression (it does not affect tenant memory or execution isolation — see SECURITY.md). For mutually-untrusting tenants it is a confidentiality gap that the operator must account for at deployment time (e.g., run a separate deployment, with its own HMAC key, per trust domain).

**Mitigation / future work:** if per-tenant kernel confidentiality is required within a single deployment, add a tenant/visibility dimension to the kernel manifest and filter reads in `list_kernels` / `resolve_kernel` against the caller's `TenantId` (the handlers already capture the tenant extension). Until that lands, per-trust-domain deployment separation is the recommended mitigation.

**Owner:** API + platform maintainers.

---

## S22 deferred work

The following items from the audit cycle remain open as of v0.3.7:

- **Differential testing** of tensor-wasm-jit against the wasmtime CPU path beyond the unit-test surface.
- **Snapshot fuzz harness** for structure-aware mutation of valid snapshots (a byte-fuzz target exists at `fuzz/fuzz_targets/fuzz_snapshot_restore.rs`).
- **End-to-end multi-tenant load test** (1000 cold starts/sec SLO claim is unverified).
- **`cust` migration plan** to `cudarc` — spike landed (see the cust EOL row above and `docs/CUDARC-SPIKE.md`); full cutover still pending.
- **Hardware verification of the CUDA paths** — the allocation / prefetch backends, the `cuStreamAddCallback` async dispatch, the device-memory host functions, `try_grow_in_place`, the experimental wmma MatMul lowering, and the cuda-oxide host backend are all written but unverified on real silicon. Inventory and the gated `gpu.yml` CI lane that validates them: [`docs/HARDWARE-GATED-WORK.md`](HARDWARE-GATED-WORK.md).

Each is tracked as a GitHub issue with the `risk-register` label.

---

## How to update this document

When a new risk surfaces (e.g., a CVE in a transitive dep, a new architectural constraint, a missed coverage gap):

1. Add a new `##` section with status, why, mitigations, owner.
2. Reference it from `CHANGELOG.md` under the relevant release.
3. If it touches dependency policy, update `.github/dependabot.yml` to match.

Stale entries should be marked **Resolved (date)** rather than deleted, so the historical context survives.
