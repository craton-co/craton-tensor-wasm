<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Continuation prompt — TensorWasm v0.3.7 → v0.4

> Hand-off document for the next orchestrator session. Picks up where the
> 2026-05-28 session ended.

## 1. State of the repo at hand-off

- **Branch**: `dev`
- **HEAD**: `167dda7 fix(api): openapi_validation_test EXPECTED_ROUTES -> expected_routes() call sites`
- **Workspace version**: `0.3.6` (was bumped in B1.7 from 0.3.5; never bumped again — `Cargo.toml.workspace.package.version` + `CITATION.cff` need to flip to `0.3.7` when you tag).
- **Build status** (as of the last verification this session):
  - `cargo build --workspace --no-default-features` → exit 0
  - `cargo check --workspace --tests --no-default-features` → exit 0
  - `cargo check --workspace --tests --all-features` → exit 0
- **Commits this session**: 79 (B1.1 → B7.9).
- **Workspace crates**: 11. Added `tensor-wasm-artifacts` in B6.6 (content-addressed signed artifact store).
- **Docs in `docs/`**: 49 markdown files. Added: `STREAMING.md`, `COOPERATIVE-YIELD.md`, `GPU-QUOTAS.md`, `INSTANCE-POOL.md`, `KERNEL-REGISTRY.md`, `DIFFERENTIAL-ORACLE.md`, `OPENAI-COMPAT.md`, `ARTIFACT-STORE.md`, `RELEASE.md`, `TESTING.md`, `FUZZING.md`, `CONFIG.md`.

## 2. First thing to do on resumption

**Bump to 0.3.7.** All of batches 5–7 documented themselves as "v0.3.7 scaffolds" but `Cargo.toml` is still pinned at `0.3.6`. Touch the following files:

- `Cargo.toml` → `workspace.package.version = "0.3.7"`, plus all 8 internal `[workspace.dependencies]` pins (`tensor-wasm-{core,mem,exec,wasi-gpu,jit,snapshot,tenant,api}`) from `version = "0.3.6"` to `version = "0.3.7"`. `tensor-wasm-artifacts = { path = "crates/tensor-wasm-artifacts", version = "0.3.7" }` (it was added at 0.3.6 too — bump in lockstep).
- `CITATION.cff` → `version: 0.3.7`, `date-released: <today>`.
- `README.md` → status banner line near the top.
- `ARCHITECTURE.md` → trailer "current as of v0.3.7".
- `ACTIONABLE-ITEMS-PENDING.md` → date header.
- `docs/RISKS.md` → "Last updated" trailer.
- `SECURITY.md` → supported-versions table row.
- `Dockerfile` → ARG WORKSPACE_VERSION and image-tag examples.
- `deploy/helm/tensor-wasm/Chart.yaml` → appVersion.
- `deploy/helm/tensor-wasm/values.yaml` + `README.md` → era-tag comments.
- `deploy/k8s/20-deployment.yaml` → app.kubernetes.io/version labels + image tag.
- `deploy/nomad/*.nomad.hcl` → Consul tag + default.
- `docs/runbooks/ghcr-registry-provisioning.md` → workspace-version note + image-tag examples.
- `crates/tensor-wasm-api/openapi.json` → `info.version`.
- `crates/tensor-wasm-cli/man/tensor-wasm.1` → .TH header + VERSION section.
- `crates/tensor-wasm-snapshot/FORMAT.md` → "pre-v0.3.6" parenthetical.

Pattern is exactly what B1.7 did for 0.3.5 → 0.3.6. Read commit `ad40bed` (`chore(release): reconcile v0.3.5 → v0.3.6`) for the full file inventory.

After the bump, **do NOT add a new CHANGELOG entry** — the existing `[0.3.7]` section (added in B7.8) is already populated and labeled correctly.

## 3. Deferred items (the actual carry-forward work)

### 3.1 B7.3 — Migrate `jit::cache::DiskCache` to `tensor_wasm_artifacts::DiskArtifactStore` [HIGH PRIORITY]

**Why deferred**: The B7.3 worktree was created on an older dev snapshot (commit `bb4bd99`) that predates B5.3 (key partition + parallel emit + zeroize) and B6.8 (`verify_on_get` opt-out). Both of those reworked `crates/tensor-wasm-jit/src/cache.rs` substantially. Merging the B7.3 worktree would have clobbered both — its rewrite of `DiskCache` came from a world where neither existed.

**Why this matters**: B7.4 already landed the parallel migration on the snapshot side (opt-in via `artifact-backing` feature). Until B7.3 lands, the JIT cache uses its own bespoke envelope (`magic(16) || fingerprint(8) || sm_version(4) || grid_x(4) || block_x(4) || ptx_len(8) || ptx_text(N) || hmac(32)` from B3.3) and the snapshot side uses the shared artifact store. The whole point of B6.6's artifact store was to converge these — having one consumer migrated and the other not is half a job.

**What needs to happen**:

1. Read the current `crates/tensor-wasm-jit/src/cache.rs` in full (~960 lines now — substantial after B3.3 / B5.3 / B6.8 layered on top of each other). Pay attention to:
   - `DiskCache::put` and `get` — the bespoke envelope writer/reader.
   - `DiskCache::path_for` — currently includes `blake3(hmac_key)[..8]` in the path stem (B5.3 partition).
   - `KernelCacheConfig.verify_on_get` (B6.8) — needs to survive the migration.
   - `KernelCacheConfig.registry` (B7.2) — also needs to survive.
   - The `Zeroizing<[u8; 32]>` HMAC key wrapper (B5.3).
2. Read `crates/tensor-wasm-artifacts/src/lib.rs` to see the `DiskArtifactStore` API.
3. Design the JIT-side payload struct (likely `JitPayload { fingerprint, sm_version, grid_x, block_x, ptx_text }`, bincode-encoded) that becomes the artifact store's opaque body.
4. Refactor `DiskCache` to hold a `DiskArtifactStore` + an in-memory `Mutex<HashMap<CacheKey, ContentHash>>` keymap. v0.4 follow-up: persist the keymap via a sidecar JSON so cache survives restart (today's in-memory keymap means an L2 cache rebuilt from disk is functionally empty until requests re-populate it).
5. Legacy magic detection: if `ArtifactError::BadMagic` (or whatever the current artifact store returns), emit `tracing::warn!(target: "tensor_wasm_jit::cache", "legacy on-disk cache format; entry will be re-emitted via the artifact store")` and return `Ok(None)`.
6. Existing tests that need updating:
   - `crates/tensor-wasm-jit/tests/cache_integrity.rs` — the forged-blob test flips a byte at a specific offset. With the artifact store envelope, the offset changes. Either recompute or just assert any single-byte flip yields BadHmac.
   - `crates/tensor-wasm-jit/tests/cache_launch_geometry_persisted.rs` — round-trips grid/block through disk; just verify it still works against the new envelope.
   - `crates/tensor-wasm-jit/tests/disk_cache_keyed_path.rs` — distinct-key partitioning; the artifact store does this natively via the key-fingerprint filename suffix, so the test should still pass with possibly minor path-format updates.
7. New test `crates/tensor-wasm-jit/tests/disk_cache_artifact_envelope.rs` — assert the resulting file starts with `tensor-wasm-artifacts`'s magic header (not the old JIT-specific one) and decodes via `DiskArtifactStore::get` directly.
8. Documentation: update `docs/ARTIFACT-STORE.md` to mark "JIT cache migration: LANDED in v0.3.7" (or whatever version it ends up in). Update `crates/tensor-wasm-jit/src/cache.rs` module doc.

**Estimated effort**: ~4-6 hours for one engineer. The tricky bit is preserving B5.3's parallel-emit thread-safety guarantees through the new envelope — `DiskArtifactStore::put` returns a `ContentHash`, and the keymap insertion must be atomic with respect to concurrent puts of the same `CacheKey`.

### 3.2 v0.4 work that's "scaffold landed" but needs real wiring

All eight of these are usable surface-area today but need follow-up to fulfill their v0.4 promise:

#### 3.2.1 Streaming `/invoke-stream` mid-execution flush
- **Today**: route at `crates/tensor-wasm-api/src/routes.rs::invoke_function_stream` returns a single `event: scaffold` frame with `{"status":"not_yet_wired"}`.
- **What's missing**: the `tensor-wasm-wasi-gpu::streaming::StreamingContext` channel from B6.1 needs to be plumbed into `SpawnConfig` (`tensor-wasm-exec::executor`), thence to `WasiCudaContext` (or a new `TensorStreamingContext`) attached to the Wasmtime store, and finally consumed by the route handler that maps it to the SSE/chunked response body.
- **Threat model**: the chunks already go through `sanitize_path`-equivalent log injection defence per `docs/STREAMING.md`; verify when wiring.
- **Tests to add**: end-to-end SSE with a guest that actually calls `wasi:tensor/host.emit-chunk`. Today's test only verifies the route mount + content-type.

#### 3.2.2 Signed kernel registry — production server-side storage
- **Today**: in-memory `InMemoryRegistry` only. `/kernels` POST publishes into a `Mutex<HashMap>` that vanishes on restart.
- **What's missing**: on-disk backing via `DiskArtifactStore` (item 3.1 above is a prerequisite). Multi-publisher allowlist via a startup config (today every caller with the HMAC key can publish). Pagination on `GET /kernels` (today returns the whole list).
- **Cross-link**: `docs/KERNEL-REGISTRY.md` § "v0.4 wiring plan".

#### 3.2.3 `cuMemPool` driver enforcement against real CUDA hardware
- **Today**: `crates/tensor-wasm-mem/src/cuda_mem_pool.rs` has the `TenantMemPool` type, FFI calls, and feature gating. Marked `#[ignore]` on integration tests.
- **What's missing**: actually verify the cudarc FFI calls compile against `cudarc 0.13`'s real bindings (the agent inferred the type names from convention). Run the `#[ignore]`d tests on the S22 self-hosted runner once it's online.
- **Cross-link**: `docs/GPU-QUOTAS.md` § "v0.4 follow-up", `ACTIONABLE-ITEMS-PENDING.md` items 2.1-2.3.

#### 3.2.4 Pliron auto-offload pipeline (roadmap #7 — NOT scaffolded yet)
- **Today**: `crates/tensor-wasm-jit/src/pliron_*.rs` modules exist with `NotYetWired` stubs only (B2.5 made them `#[doc(hidden)]`).
- **What's missing**: this is RFC 0001 wave 3+. Read `rfcs/0001-cuda-oxide-integration.md` and `docs/PLIRON-PIPELINE.md`. ~2-3 months of work.
- **Status**: `🔵 not started` in `ACTIONABLE-ITEMS-PENDING.md`.

#### 3.2.5 `InstancePool` warm-pool channel
- **Today**: `crates/tensor-wasm-exec/src/instance_pool.rs::InstancePool::acquire` falls through to `executor.spawn_instance`.
- **What's missing**: the per-`(tenant, module-hash)` `crossbeam_channel::Sender<TensorWasmInstance>`, the pre-spawn loop, and the reset-on-return semantics. Reset is the hard part — instance state (linear memory, globals, tables) must be scrubbed back to module-initial before re-use, OR the guest must guarantee reentrancy.
- **Cross-link**: `docs/INSTANCE-POOL.md` § "v0.4 implementation plan".

#### 3.2.6 `DifferentialOracle` two-path runner
- **Today**: `crates/tensor-wasm-jit/src/differential.rs::DifferentialOracle::compare` always returns `OracleVerdict::Skipped("no-cuda; v0.4 wires this against the S22 runner")`.
- **What's missing**: actually run the blueprint on both the Wasmtime CPU interpreter and the JIT'd PTX, capture outputs, compare bit-for-bit. Requires the S22 runner.
- **Cross-link**: `docs/DIFFERENTIAL-ORACLE.md` § "v0.4 implementation plan".

#### 3.2.7 OpenAI gateway shim — actual translation
- **Today**: `crates/tensor-wasm-api/src/openai.rs::completions_handler` and `chat_completions_handler` both return `501 openai_not_yet_wired` regardless of input.
- **What's missing**: the `model` → deployed-function translation table (config-driven, probably env or a YAML). Then call `executor.call_export_with_args` with the prompt as a `WasmArg::I32`-encoded pointer/length pair (or whatever the guest export ABI demands). Finally, marshal the result into the OpenAI response envelope.
- **Why this matters**: per the original review § 6, **this is the single highest-ROI item** in the roadmap — it shifts the addressable market from Wasmtime/Wasmer migrators to Modal/Replicate/Beam users with Python OpenAI clients. Build this before recruiting design partners.
- **Cross-link**: `docs/OPENAI-COMPAT.md` § "v0.4 implementation plan".

#### 3.2.8 Snapshot artifact-backing default cutover
- **Today**: opt-in via the `artifact-backing` feature on `tensor-wasm-snapshot`. The legacy v2/v3 inline envelope is the default.
- **What's missing**: once operators have migrated, flip the default. v0.4 milestone per `crates/tensor-wasm-snapshot/FORMAT.md`.

### 3.3 Speculative / R&D — roadmap items #11, #12, #13 (not started)

- **#11 WASI-NN compatibility** — 6 weeks; high risk (spec moving). Gives existing ONNX/llama.cpp/OpenVINO WASI-NN guests a CUDA-accelerated path.
- **#12 Direct guest-side GPU dispatch via SPIR-V** — 6 months+; very high risk. Speculative; depends on a CG proposal landing.
- **#13 Distributed dispatch sidecar over QUIC** — 2-3 months; medium risk. Single-hop GPU bursting; v1.x scope, not v1.0.

## 4. Known issues / gotchas

### 4.1 Linter warnings worth fixing (cosmetic)

```
warning: unused variable: `router`
   --> crates\tensor-wasm-api\tests\get_job_cross_tenant.rs:118:5
warning: unused variable: `router`
  --> crates\tensor-wasm-api\tests\scoped_tokens_test.rs:85:5
warning: variable does not need to be mutable
  --> crates\tensor-wasm-api\tests\openapi_validation_test.rs:90:9
```

All three: prefix with `_router` / drop the `mut`. Five-second fixes; not done because they're test-only and don't affect compile.

### 4.2 The `pliron_*` `cuda-oxide-backend`-gated module surface

Per B2.5 these are `#[doc(hidden)]` but still `pub mod`. Many of the internal items (`StubLowerer`, `NotYetWired` variants, etc.) are reachable from external crates if someone bypasses docs. Consider:

- Either feature-gate to `cuda-oxide-backend-unstable` and document as preview, OR
- Wrap in a `#[cfg(doc)]`-style attribute that hides the whole feature stack from generated docs.

### 4.3 130 locked agent worktrees in `.claude/worktrees/`

`git worktree list` shows 130 locked agent worktrees (from this session and prior). They are inside `.claude/worktrees/` which is gitignored, so they don't pollute the repo. But they consume disk space (~hundreds of MB total). The harness left them locked; a manual `git worktree remove --force --force <path>` per directory will clean them up. Optional.

### 4.4 `tensor-wasm-cli` `documentation` field references the wrong crate

`crates/tensor-wasm-cli/Cargo.toml` has `documentation = "https://docs.rs/tensor-wasm"`. The crate name is `tensor-wasm-cli`. Harmless while `publish = false`, but if you ever flip that bit, the URL must change. Tracked in the original review § 4.

### 4.5 `B6.4 / B6.5 / B6.6` cross-file dependencies

These three batches landed scaffolds that reference one another:
- `tensor-wasm-mem::cuda_mem_pool::TenantMemPool` is feature-gated on `cudarc-backend` and uses `cudarc::driver::sys::*` types that were inferred (not verified against a real cudarc 0.13 compile). When you light up the S22 runner, the first thing to do is `cargo check --features gpu-mem-pool` on a CUDA host. Expect type-name fixes.
- `tensor-wasm-tenant::TenantContext.mem_pool: Option<Arc<TenantMemPool>>` is the consumer.
- `tensor-wasm-mem::wasm_memory::TensorWasmMemoryCreator::with_tenant_context` accounts every `UnifiedBuffer` allocation against `TenantContext::consume_gpu_bytes` (in-process) but does NOT yet route through the `cuMemPool` (driver-level). The two layers are independent; the in-process counter is the only enforcement today.

### 4.6 OpenAPI yaml drift

`crates/tensor-wasm-api/openapi.json` and `openapi/tensor-wasm-api.yaml` are TWO copies of the same schema. The yaml is the authoritative one per `openapi_validation_test`, but the json gets regenerated from it (or hand-edited — be careful which). When you add/modify routes, edit the yaml and let the test guide whether the json needs an update.

### 4.7 Tests that don't run by default

The following test suites are `#[ignore]`d and require special setup:

- `crates/tensor-wasm-mem/tests/cudarc_multi_gpu.rs` — needs CUDA hardware
- `crates/tensor-wasm-mem/tests/cuda_mem_pool_scaffold.rs` — needs CUDA hardware
- `crates/tensor-wasm-snapshot/tests/compat.rs` — golden fixtures (4 tests now properly `#[ignore]`d per B3.5)
- `crates/tensor-wasm-wasi-gpu/tests/wasi_gpu_smoke.rs` — most tests `#[ignore]` for CUDA, but a few non-CUDA tests run
- `crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs` — same
- `crates/tensor-wasm-jit/tests/lowering_e2e.rs` — requires both `cuda-oxide-backend` and `test-utils` features
- `crates/tensor-wasm-jit/tests/differential_scaffold.rs` — requires `differential-oracle` feature
- `crates/tensor-wasm-tenant/tests/loom_consume_release.rs` — requires `loom` feature
- `crates/tensor-wasm-tenant/tests/cap_binding_strict.rs` — requires `strict-cap-binding` feature

Run them with `cargo test -- --ignored` (CUDA host) or per-feature.

## 5. Verification recipe for the next session

After making any change, run in order:

```bash
# 1. No-features build (most consumers' default)
cargo build --workspace --no-default-features

# 2. Check tests compile (catches integration-test breakage)
cargo check --workspace --tests --no-default-features

# 3. Feature-gated builds — exercise every opt-in path
cargo check --workspace --tests --all-features

# 4. Doc builds (catches missing-docs lints and broken intra-doc links)
cargo doc --workspace --no-deps --no-default-features

# 5. Linting
cargo clippy --workspace --tests --no-default-features -- -D warnings

# 6. Format check
cargo fmt --all -- --check

# 7. Security audit
cargo audit
cargo deny check
```

If you're working on a CUDA-touching change, add:

```bash
cargo check --features cudarc-backend
cargo check --features cuda-oxide-backend
cargo test --features cudarc-backend -- --ignored   # on CUDA host
```

## 6. Suggested next-batch priorities

In order of ROI (per the review's batch-7 closing recommendation, which still stands):

1. **Bump 0.3.6 → 0.3.7 and tag** (1 hour). Pre-req: nothing.
2. **B7.3 jit DiskCache → artifact store** (4-6 hours). Pre-req: bump done, or skip the version mess and just do it on 0.3.6.
3. **OpenAI gateway translation table** (~2 weeks). Highest ROI of any v0.4 feature.
4. **Streaming `/invoke-stream` wiring through `StreamingContext`** (~3 weeks). Composes with the OpenAI gateway — both need streaming for the LLM use case.
5. **`InstancePool` warm-pool channel** (~2 weeks). Pushes P99 below Modal/Beam.
6. **Differential JIT oracle real two-path runner** (~3 weeks). Audit credibility before the v0.5 pen-test.
7. **Pliron pipeline wave 3 onwards** (months). Long pole; start picking at it in parallel with the above.

## 7. Files / commits to re-read before starting

- `docs/PATH-TO-V1.md` § "Post-v0.3.6 strategic features" — the full feature dossier with each item's status, cross-links, and v0.4 plan.
- `ACTIONABLE-ITEMS-PENDING.md` — the operator-facing pending-items list, status-colored.
- `CHANGELOG.md` `[0.3.7]` section — what landed.
- `docs/RISKS.md` — the current risk register, including the cust 0.3.x EOL story that gates several of the cudarc / cuda-oxide moves.
- `rfcs/0001-cuda-oxide-integration.md` — the master plan for items #3 (kernel registry), #6 (differential oracle), #7 (pliron), #8 (GPU quotas).

## 8. Session-level orchestration notes

If you re-spawn agents in worktrees:

- **Always `git pull` / rebase the worktree's base on current `dev` before letting the agent start writing.** Several batch-7 agents based on `bb4bd99` (which predates batch 5 substantially) produced unmergeable rewrites — that's how B7.3 ended up deferred. Either (a) instruct the agent to `git checkout dev && git pull` first, or (b) accept that the harness creates worktrees from the current dev HEAD and verify before launch.
- **Conflicts on merge usually mean the agent worked from a stale base.** When two agents touch the same file (which happens for cross-cutting items like `api/src/lib.rs`, `api/src/routes.rs`, `api/src/server.rs`, `jit/Cargo.toml`, `docs/PATH-TO-V1.md`, `bench-results/baseline.json`), the cleanest resolution is usually `git checkout --ours` (keep dev's state) followed by re-applying the agent's specific additions by hand. The session did this many times.
- **Three Cargo.toml gotchas worth knowing**:
  1. `subtle` was added as a hard dep in B5.3 and re-added as `optional = true` by B6.3's `kernel-registry` feature. Cargo errored with "duplicate key" until manually resolved.
  2. `tensor-wasm-artifacts` is a workspace member, listed in both `[workspace.members]` and `[workspace.dependencies]`. When new crates depend on it, add `tensor-wasm-artifacts = { workspace = true }` to the consumer's deps.
  3. `bincode` is at workspace version 2 with the `serde` feature globally enabled. Per-crate Cargo.toml entries should be `bincode.workspace = true`, NOT `bincode = "2"`.

## 9. Recap of what this session added (just the numbers)

| Metric | Count |
|---|---|
| Commits to `dev` | 79 |
| Files newly created | 88 |
| Files modified | 176 |
| New crates | 1 (`tensor-wasm-artifacts`) |
| New `pub mod` modules | ~10 (`openai`, `streaming`, `scheduler`, `registry` x2, `instance_pool`, `differential`, `cuda_mem_pool`, `kernels`, `differential`) |
| New HTTP routes | `/v1/completions`, `/v1/chat/completions`, `/functions/:id/invoke-stream`, `/kernels` (3 methods) |
| New WIT packages | `wasi:tensor@0.1.0`, `wasi:scheduler@0.1.0` (in-crate copies) |
| New cargo features | `kernel-registry`, `kernel-registry-api`, `differential-oracle`, `artifact-backing`, `gpu-mem-pool`, `test-utils`, `strict-cap-binding`, `loom` |
| New env vars consumed | `TENSOR_WASM_API_TRUSTED_XFCC_PROXIES`, `TENSOR_WASM_API_KERNEL_HMAC_KEY` |
| New deprecation attributes | 2 (`call_export`, `call_export_then_terminate`) |
| Roadmap features scaffolded | 9 of 13 (items #1, #2, #3, #4, #5, #6, #8, #9, #10) |
| Roadmap features not started | 4 of 13 (items #7, #11, #12, #13) |
| Test files added | ~30 (integration, fuzz, property, scaffold) |
| Docs added | 12 |
| Stale `.claude/worktrees/agent-*` gitlinks scrubbed | 94 |

---

_End of continuation prompt. The tree is build-clean. Pick up wherever the
priorities in § 6 make the most sense for the next milestone._
