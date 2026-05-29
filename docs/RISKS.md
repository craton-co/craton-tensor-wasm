# Craton TensorWasm — Risk Register

Living document tracking architectural risks, upstream pinning decisions, and known limitations for maintainers. Updated alongside `CHANGELOG.md` releases.

Last updated: 2026-05-28 (v0.3.7 — workspace version bump)

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

**Status:** spike landed under `--features cudarc-backend`; full migration pending the v0.5 cutover decision per RFC 0001. Workspace pins `cust = "0.3"` as the default; `cudarc = "0.13"` is available behind the per-crate `cudarc-backend` feature flag in `tensor-wasm-mem`. Both backends coexist through v0.4 for backward compat; the default flip is re-scoped to v0.5. Upstream `cust` maintenance is stalled.

**Risk:** no security or compatibility patches on the cust path; CUDA 13+ may break without warning. The cudarc-backend spike confirms the migration is viable (API mapping is ~95% one-to-one — see `docs/CUDARC-SPIKE.md`), but full cutover still needs a runner pass on real hardware and a callsite sweep across `tensor-wasm-mem::unified`, `advise`, `pool`, and `wasm_memory`.

**Mitigations under evaluation:**
- Cut the default backend over (to `cudarc` or `cuda-oxide`) at v0.5 once the S22 runner validates the spike end-to-end (see `docs/CUDARC-SPIKE.md` and RFC 0001 for the proposed cutover plan and the contingent-default approach).
- Maintain an internal fork of `cust` for security backports if the cutover slips past v0.5.

A third option has appeared since W1.2 wrote the spike: NVlabs `cuda-oxide` v0.1.0 alpha (released 2026-05-09) is now under evaluation per [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md), which proposes a three-way live evaluation with a contingent v0.5 default flip.

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
- **`cust` migration plan** to `cudarc` — spike landed (see the cust EOL row above and `docs/CUDARC-SPIKE.md`); full cutover still pending.

Each is tracked as a GitHub issue with the `risk-register` label.

---

## How to update this document

When a new risk surfaces (e.g., a CVE in a transitive dep, a new architectural constraint, a missed coverage gap):

1. Add a new `##` section with status, why, mitigations, owner.
2. Reference it from `CHANGELOG.md` under the relevant release.
3. If it touches dependency policy, update `.github/dependabot.yml` to match.

Stale entries should be marked **Resolved (date)** rather than deleted, so the historical context survives.
