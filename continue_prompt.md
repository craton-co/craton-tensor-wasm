<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Continuation prompt — TensorWasm v0.3.7 security/perf/v0.4-feature orchestration handoff

> Hand-off document for the next orchestrator session. Picks up after the
> 2026-05-28 audit-driven multi-agent orchestration. The previous session
> dispatched ~40 parallel Opus agents that landed security fixes, perf
> wins, and v0.4 feature deliverables on the `dev` branch. **The tree has
> NOT been built or tested.** First action on resumption: run `cargo
> check --workspace` and fix.

## 1. State of the repo at hand-off

- **Branch**: `dev`
- **HEAD**: `c7c19d3 merge T33: wire typed --args end-to-end through CLI / HTTP / SpawnConfig`
- **Commits ahead of `main`**: 73 (40-ish feature/fix commits + 33 merge commits).
- **Build status**: **UNKNOWN — never invoked `cargo` during the orchestration.** Agents were instructed code-only, and the orchestrator deferred verification to this handoff. See § 5 below for the recipe.

## 2. What landed (28 of 32 tasks complete + 4 manual entries)

### From the v0.3.7 security audit (28 items, 100% complete)

| Task | Severity | Crate | Summary |
|---|---|---|---|
| T1 | **Critical** | api | `/kernels*` routes moved under `tenant_scope`; `publish` requires kernel-publish scope; dev-mode rejects publishes. |
| T2 | High | api | `/v1/completions` + `/v1/chat/completions` gated behind `tenant_scope`. |
| T3 | High | api | `From<ExecError> for ApiError` scrubs internal IDs / deadlines / quotas; logs forensics server-side. |
| T4 | High | mem | `UnifiedMemoryPool::reset(&mut self)`; `slab_ptr` → `pub(crate) unsafe fn`. |
| T5 | High | mem | `Backing` enum sealed into private `mod backing`; aliasing invariant documented. |
| T6 | High | mem | `PooledLinearMemory::Drop` calls `pool.release(offset, size)`. |
| T7 | High | exec | Re-applied `#[deprecated]` attrs on `call_export` / `call_export_then_terminate` (dropped in merge `66af7db`). |
| T8 | Medium | snapshot | v3 detection now requires 4-byte magic at `bytes[len-37..len-33]`; trailer grew to 37 bytes. **BREAKING for unmigrated v3 captures.** |
| T9 | Medium | snapshot | `MAX_INPUT_BYTES` 4 GiB → 1 GiB; opt-in `SnapshotReader::with_max_age(Duration)`; `SystemTime::now()` failure → `Serialization` error. |
| T10 | Medium | artifacts | `MAX_PAYLOAD_LEN = 256 MiB`, `MAX_DECOMPRESSED_LEN = 1 GiB`, streaming `zstd::stream::read::Decoder`, `checked_sub` on hmac_start. |
| T11 | Medium | tenant | `register_with_capability` orphan-check is now atomic under `inner.entry()`; `unregister` inserts tombstone before removing entry. |
| T12 | Medium | jit | Registry envelope v2: `b"twasm-kmf-v2"` magic + length-prefixed canonicalisation + covers `publisher` + `published_unix_ms`. **BREAKING** for v1 manifests. |
| T13 | Medium | jit | `DiskCacheConfig::hmac_key` wrapped in `Zeroize`; custom `Debug` redacts; rustdoc example sanitised. |
| T14 | Medium | mem | Dropped redundant cudarc full-allocation memset; only `init_zero_bytes` is zeroed (relies on `pool.rs` cross-tenant zero-fill). |
| T15 | Low | wasi-gpu | Removed stray `<<<<<<< HEAD` from `wit/wasi-cuda.wit:89`; added `wit_file_has_no_merge_conflict_markers` regression test. |
| T16 | Low | api | Startup `tracing::warn!` when `TENSOR_WASM_API_TOKENS` is set but `TENSOR_WASM_API_TRUSTED_HOSTS` is unset. |
| T17 | Low | cli | `bounded_text` / `bounded_bytes` helpers; `MAX_RESPONSE_BODY_BYTES = 16 MiB`; new `ApiClientError::ResponseTooLarge`. |
| T18 | Low | cli | `sanitise_terminal_output(&str) -> String` strips ASCII control bytes from server-returned strings before `println!`. |
| T19 | Perf | api | `host_validate` moved outermost (ahead of `trace_layer`); `BASE64_OFFLOAD_THRESHOLD` 256 KiB → 32 KiB. |
| T20 | Perf | jit | `KernelCache::get` returns `Option<Arc<CachedKernel>>`; `hex::encode_to_slice` in `path_for`; pre-sized `lower_body` String; `cache_hits_total`/`cache_misses_total` counters. |
| T21 | Perf | snapshot | Pre-size compressed `Vec` from `total_uncompressed_bytes / 4`; `SnapshotRef` borrow on artifact path (no `.to_vec()`). |
| T22 | Perf | artifacts | Streaming HMAC + zstd via `BufReader`/`Decoder` + `MacWriter` tee in `put`; direct-to-file (no intermediate `Vec`). |
| T23 | Perf | wasi-gpu | `parse_argv` takes `&[u8]` (no Vec copy); pre-size `KernelParamStorage::backing`. |
| T24 | Perf | cli | `BufWriter::with_capacity(64 KiB, ...)` around snapshot save tempfile (~16× fewer syscalls). |
| T25 | Perf | core | `String::with_capacity(8 KiB)` on `encode_text`. |
| T26 | Perf | mem | Cached per-ordinal `Arc<CudaDevice>` / `Arc<CudaContext>`; `UnifiedMemoryPool` bump/live/issued_total now `AtomicUsize`/`AtomicU64` (CAS loop). |
| T27 | Perf | tenant | `OnceLock`-cached MPS decision; opportunistic tombstone prune on `unregister`/`register`. |
| T28 | Hygiene | api | Regenerated `crates/tensor-wasm-api/openapi.json` from authoritative YAML; new `openapi_json_yaml_sync.rs` regression test; `scripts/regen-openapi-json.{sh,ps1}`. |
| T31 | Hygiene | api tests | Three trivial lint warnings silenced (`_router` rename, `#[allow(unused_mut)]` on cfg-feature-gated mut). |
| T32 | Hygiene | cli | `documentation = "https://docs.rs/tensor-wasm"` → `https://docs.rs/tensor-wasm-cli`. |

### From `docs/PATH-TO-V1.md` §3.1 v0.4 deliverables (6 of 9 complete)

| Task | PATH-TO-V1 # | Summary |
|---|---|---|
| T34 | #2 streaming | `/functions/{id}/invoke-stream` plumbed end-to-end via `StreamingContext`; guest `emit-chunk` → SSE `event: chunk`; deadline-elapsed → `event: error`. |
| T35 | #3 kernel registry | `DiskRegistry` (HMAC v2 envelope, `DiskArtifactStore`-backed, restart-safe, pagination, publisher allowlist). Selected when `TENSOR_WASM_API_KERNEL_REGISTRY_DIR` is set. |
| T36 | #4 cooperative deadlines | Executor's Instant deadline → `SchedulerContext` + `BackPressure` (DEADLINE_NEAR_WINDOW = 50 ms). New `BackPressureError::DeadlineNear` / `DeadlineElapsed`. |
| T38 | #6 differential oracle | proptest harness for vector_add/conv2d/matmul + per-blueprint `ToleranceTable` (1-4 ULP). CUDA verdicts marked `#[ignore]` pending S22 runner. |
| T39 | #8 cuMemPool quota | `TenantMemPool::new(ordinal, cap_bytes)` calls `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)`. `UnifiedBuffer::new_in_tenant_pool` routes via `cuMemAllocFromPoolAsync`. Requires `--features gpu-mem-pool` + CUDA 11.2+. |
| T40 | #9 snapshot artifact-backing | `artifact-backing` is now in `default` features; new writes use the unified envelope; reads still accept v2/v3. `with_legacy_envelope()` for opt-out. **BREAKING for default writer behavior.** |
| T33 | #1 typed args | `--args <JSON>` CLI flag; HTTP `/invoke{,-async,-stream}` body `args: [...]`; `SpawnConfig::with_args(Vec<WasmArg>)`. |

## 3. Currently in flight (3 background agents — may have landed by the time you read this)

| Task | Agent ID | Worktree branch | Description |
|---|---|---|---|
| T30 | `a22579d0b864ef3f7` | `worktree-agent-a22579d0b864ef3f7` | B7.3 — migrate `tensor-wasm-jit::DiskCache` to `DiskArtifactStore` backend. Heavy refactor preserving T12/T13/T20 invariants. |
| T41 | `a9ae79a3579510399` | `worktree-agent-a9ae79a3579510399` | OpenAI request translator into internal invoke protocol. Model-map env-var, streaming via T34. Depends on T34. |

**On resumption:** check `git branch | grep worktree-agent` and `git log --all --oneline | head -40` to find their final commit SHAs. Merge each (expect CHANGELOG conflicts — see § 6.2).

## 4. Queued / pending (not started this session)

### T29 — Workspace version bump 0.3.6 → 0.3.7

Long touchlist (per the previous `continue_prompt.md` §2 and commit `ad40bed` as the template):

- `Cargo.toml` → `workspace.package.version = "0.3.7"` + 9 internal `[workspace.dependencies]` `version = "0.3.7"` pins.
- `CITATION.cff`, `README.md`, `ARCHITECTURE.md`, `ACTIONABLE-ITEMS-PENDING.md`, `docs/RISKS.md`, `SECURITY.md`, `Dockerfile`.
- `deploy/helm/tensor-wasm/{Chart.yaml,values.yaml,README.md}`.
- `deploy/k8s/20-deployment.yaml`, `deploy/nomad/*.nomad.hcl`.
- `docs/runbooks/ghcr-registry-provisioning.md`.
- `crates/tensor-wasm-api/openapi.json` AND `openapi/tensor-wasm-api.yaml` (both have `info.version`).
- `crates/tensor-wasm-cli/man/tensor-wasm.1`.
- `crates/tensor-wasm-snapshot/FORMAT.md`.

This was deliberately deferred to last to avoid colliding with every concurrent agent's `Cargo.toml` edits. **Do it AFTER you've cleaned up any T30/T41 merge mess and confirmed `cargo check` is green.**

### T37 — `InstancePool` through executor `invoke` path (PATH-TO-V1 §3.1 #5)

Not launched because it conflicts with T34 (both touch `crates/tensor-wasm-exec/src/executor.rs` invoke-path) and T33 (both touch `SpawnConfig`). With T33 and T34 merged, T37 is now unblocked. Spec is in the original prompt — repeated here:

- `InstancePool::acquire` currently falls through to `executor.spawn_instance` (it's a scaffold per `instance_pool.rs:117-129`).
- Implement: per-`(tenant, module-hash)` `crossbeam_channel::Sender<TensorWasmInstance>`, pre-spawn loop, reset-on-return semantics. Reset is the hard part — instance state must be scrubbed back to module-initial before re-use.
- Cross-link: `docs/INSTANCE-POOL.md` § "v0.4 implementation plan".

### Other deferred items NOT tasked this session

- **PATH-TO-V1 §3.1 #7 — Pliron-based auto-offload pipeline.** Months of work; blocked on cuda-oxide v0.2 upstream. Not appropriate for agent dispatch.
- **PATH-TO-V1 §3.1 #11 WASI-NN compat layer.** 6 weeks, high risk; spec is moving.
- **PATH-TO-V1 §3.1 #12 SPIR-V guest-side dispatch.** Anti-goal per the doc.
- **PATH-TO-V1 §3.1 #13 QUIC sidecar.** v1.x territory.
- **v0.2 milestone — kernel-args marshalling `KernelArgsUnsupported` removal.** Partially done host-side (T23 made `parse_argv` slice-based); full removal requires real CUDA `cuLaunchKernel` dispatch with typed argv, which needs the S22 runner.

## 5. **Verification recipe — do this first on resumption**

```bash
git checkout dev

# 1. Confirm in-flight agents finished and merge their branches.
git branch | grep worktree-agent
#   For each remaining worktree branch, merge with --no-ff. Expect CHANGELOG conflicts (see §6.2).

# 2. No-features build (most consumers' default).
cargo build --workspace --no-default-features

# 3. Tests compile (catches integration-test breakage).
cargo check --workspace --tests --no-default-features

# 4. All-features check (exercises every opt-in: gpu-mem-pool, cudarc-backend,
#    cuda-oxide-backend, differential-oracle, artifact-backing, kernel-registry,
#    kernel-registry-api, strict-cap-binding, loom).
cargo check --workspace --tests --all-features

# 5. Doc builds.
cargo doc --workspace --no-deps --no-default-features

# 6. Lint + format.
cargo clippy --workspace --tests --no-default-features -- -D warnings
cargo fmt --all -- --check

# 7. Security audit.
cargo audit
cargo deny check
```

**If anything fails, dispatch fix agents** following the original orchestration pattern (worktree isolation, code-only). The orchestration history committed during this session includes failures resolved via 6-7 small fix-PR-shaped commits — the same loop applies.

## 6. Known issues and gotchas

### 6.1 The tree has never been built this session

Every commit is logically wired by the agent but has not been syntax-checked. Expect a non-zero count of compile errors at first `cargo check`. Likely failure modes:

- **`tensor-wasm-mem` `gpu-mem-pool` feature** (T39, manually applied as `b082f66`) — the cudarc 0.13.9 FFI symbol names were verified by the agent against the actual cargo registry source, but the call sites use `cuda_sys::lib()` wrapping which may differ from what the worktree base assumed. Build with `--features gpu-mem-pool` to surface.
- **`tensor-wasm-snapshot` (T40) — `artifact-backing` is now default.** Five v3-shape tests were updated to use `.with_legacy_envelope()`. If any test was missed, expect "no field `with_legacy_envelope`" or "expected v3 magic" failures.
- **`tensor-wasm-jit` (T20) — `KernelCache::get` returns `Option<Arc<CachedKernel>>`.** Every caller was reportedly updated via the agent's grep; if any was missed, expect "expected struct, found Arc" or "no method named `.ptx`" on an `Option`.
- **`tensor-wasm-jit` (T30, in-flight) — DiskCache rewritten to wrap DiskArtifactStore.** The agent's spec preserved the public API but the migration is the largest single change of this session. Multiple existing tests may need adjustments (per the spec it touches 3 existing tests + adds 1 new).
- **`tensor-wasm-mem` (T4 + T6 + T26) — pool.rs has accumulated three concurrent reworks.** `reset(&mut self)`, `Drop`-decrements-live, atomic counters. If T4's `&mut self` callers were updated for T4 but a NEW caller introduced by T6/T26 still uses `&self`, expect a borrow-checker error.

### 6.2 CHANGELOG.md merge conflicts are routine

Every code-changing agent added an entry to `[Unreleased]`. The branches all base-off-of-dev but the orchestrator merged them sequentially, so each merge after the first has CHANGELOG conflicts. Resolution pattern:

1. Keep BOTH agents' entries.
2. Maintain Keep-a-Changelog ordering: `### Added` → `### Changed` → `### Deprecated` → `### Removed` → `### Fixed` → `### Security`.
3. Watch for STACKED `<<<<<<< HEAD` markers — twice this session a conflict-resolution Edit failed mid-flight and the resulting commit included an unresolved marker. Always `grep -n '<<<<<<<\|=======\|>>>>>>>' CHANGELOG.md` AFTER each merge.

### 6.3 T39 was manually applied to dev (not via worktree merge)

Mid-session, the working tree had uncommitted changes matching T39's spec (mem cuMemPool driver pin + tenant `with_driver_enforced_gpu_cap` + `docs/GPU-QUOTAS.md`). The orchestrator committed those as `b082f66 feat(mem): T39 ... (manual apply)`. The T39 agent later confirmed its worktree commit `c91f54f` is content-equivalent. **Action:** the worktree branch `worktree-agent-ac66361ede9c1757c` exists but is NOT merged (it's content-equivalent to `b082f66`); safe to delete with `git worktree remove --force` + `git branch -D`.

### 6.4 Stale tests likely

- `crates/tensor-wasm-api/tests/invoke_envelope_shape.rs::invoke_envelope_matches_empty_body_response` — per the T33 agent's report, this test posts `args: [1.0, 2.0, "three"]` expecting 200, but the T33-landed string-arg rejection now legitimately returns `400 invalid_args`. Either delete the test or update the expectation.
- v3-shape snapshot tests that didn't get T40's `.with_legacy_envelope()` treatment.
- Any test that constructs `DiskCacheConfig { hmac_key: [0xAB; 32], .. }` as a struct literal — after T13, `DiskCacheConfig` has a `Drop` impl that requires `std::mem::take`-based construction in callers (cf. T13 report).

### 6.5 Items that the v0.3.7 audit found but were intentionally deferred

These were flagged as STUBS in the audit but are intentional v0.4 work, not bugs:

- `tensor-wasm-exec::auto_offload.rs` — "consultation-only", activation pending `tensor_wasm_jit::rewrite`.
- `tensor-wasm-mem::cuda_oxide_backend.rs` — entire backend is `NOT_YET_WIRED` until cuda-oxide v0.2.
- `tensor-wasm-jit::pliron_*.rs` — 19+ `NotYetWired` variants; depends on cuda-oxide v0.2 + Pliron stability.
- `tensor-wasm-bench/benches/streaming_invoke.rs` — `todo!()` inside `b.iter`; saved only by not being registered in `Cargo.toml`. If you ever add it to `[[bench]]`, replace the `todo!()` first.
- `tensor-wasm-bench/benches/call_export_args.rs` — on disk but unregistered AND depends on crates not in `[dev-dependencies]`.

### 6.6 Worktree cleanup

`git worktree list` will show many worktrees from this session (~25-30 in `.claude/worktrees/`). They are inside `.claude/worktrees/` which is gitignored, so they don't pollute the repo. They consume disk space. Clean with:

```bash
for path in $(git worktree list --porcelain | awk '/worktree /{print $2}' | grep '\.claude/worktrees/'); do
    git worktree remove --force --force "$path"
done
git branch | grep '^  worktree-agent-\|^  fix/' | xargs -r git branch -D
```

### 6.7 OpenAPI YAML / JSON sync

T28 added a regression test (`openapi_json_yaml_sync.rs`) and helper scripts (`scripts/regen-openapi-json.{sh,ps1}`). After landing T33 (added `args` field to invoke envelopes), T34 (replaced invoke-stream description), T40 (snapshot format change touches API documentation), and potentially T41 (replaces /v1/* 501 description), the YAML and JSON may have drifted. Run:

```bash
bash scripts/regen-openapi-json.sh
cargo test -p tensor-wasm-api openapi_json_yaml_sync
```

### 6.8 The audit's audit

The original 11-crate audit (start of session) had findings categorized as Critical / High / Medium / Low. Of those:

- **Critical (1)**: T1, fixed.
- **High (6)**: T2, T3, T4, T5, T6, T7 — all fixed.
- **Medium (7)**: T8-T14 — all fixed.
- **Low (4)**: T15-T18 — all fixed.
- **Code review themes (cross-cutting)** — most addressed: OpenAPI drift (T28), stale doc strings (T13 commit msg notes the HMAC vs BLAKE3 doc mismatch on jit cache; verify still accurate after T30), merge artifacts (T7, T15).
- **Perf wins (clear)**: T19-T27 — all dispatched and merged.

### 6.9 Tasks that don't run by default (test ignore-list, unchanged from prior session)

```
crates/tensor-wasm-mem/tests/cudarc_multi_gpu.rs                    [#[ignore], CUDA hw]
crates/tensor-wasm-mem/tests/cuda_mem_pool_scaffold.rs              [#[ignore], CUDA hw]
crates/tensor-wasm-mem/tests/cuda_mem_pool_driver_pin.rs            [#[ignore], CUDA hw — NEW from T39]
crates/tensor-wasm-mem/tests/cudarc_visible_window_only.rs          [#[ignore], CUDA hw — NEW from T14]
crates/tensor-wasm-snapshot/tests/compat.rs                         [golden fixtures]
crates/tensor-wasm-wasi-gpu/tests/wasi_gpu_smoke.rs                 [#[ignore] CUDA paths]
crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs                [#[ignore] CUDA paths]
crates/tensor-wasm-jit/tests/lowering_e2e.rs                        [cuda-oxide-backend + test-utils]
crates/tensor-wasm-jit/tests/differential_scaffold.rs               [differential-oracle]
crates/tensor-wasm-jit/tests/differential_proptest.rs               [differential-oracle — NEW from T38]
crates/tensor-wasm-tenant/tests/loom_consume_release.rs             [loom]
crates/tensor-wasm-tenant/tests/cap_binding_strict.rs               [strict-cap-binding]
crates/tensor-wasm-artifacts/tests/streaming_perf.rs                [#[ignore] perf — NEW from T22]
crates/tensor-wasm-artifacts/tests/size_caps.rs (one of two)        [#[ignore] needs 1+ GiB allocation]
crates/tensor-wasm-snapshot/tests/max_input_tightened.rs            [#[ignore] needs ~1.5 GiB RAM]
```

## 7. Suggested next-batch priorities

In order:

1. **Verify the tree builds** (§5). Likely 2-5 fix-PRs needed. Each is a small fix agent in a worktree.
2. **Merge T30 (JIT DiskCache → artifact store) + T41 (OpenAI translator)** if not already done. Both are sitting on background agents.
3. **Land T37 (InstancePool wiring through invoke)** — the last v0.4 PATH-TO-V1 deliverable in scope.
4. **Land T29 (version bump 0.3.6 → 0.3.7)** — paperwork, last.
5. **Tag v0.3.7.** All security, perf, and v0.4 feature work for this milestone is now LANDED.

After v0.3.7:
- **v0.4 milestone (per PATH-TO-V1.md)**: kernel-args marshalling against real CUDA hardware, baseline measurements from S22 runner, MPS E2E validation, CUDA-SETUP.md rewrite. ALL of these need the S22 self-hosted CUDA runner online. Block on that.
- **v0.5 milestone**: external pen-test, beta deploy, fuzz corpus 24h+, cross-version snapshot compat matrix.

## 8. Files / commits worth re-reading on resumption

- `docs/PATH-TO-V1.md` § 3.1 — the v0.4 feature dossier, now with most items landed.
- `ACTIONABLE-ITEMS-PENDING.md` — operator-facing pending list. After T35/T36/T34/T38/T39/T40/T33/T41, most rows flip from 🟡 scaffold-landed to 🟢 wired.
- `CHANGELOG.md` `[Unreleased]` — comprehensive list of what this session landed. 14-ish entries; reorder if you adopt T29's version bump (move them under `## [0.3.7]` and reset `[Unreleased]`).
- `docs/RISKS.md` — the cust 0.3.x EOL risk now has a partial mitigation: T26 cached device handles + T39 cuMemPool show that the cudarc-backend path is usable enough to ship a v0.4 default.

## 9. Session-level orchestration notes for the next operator

If you spawn agents in worktrees again:

- **Branch from current `dev`, not `main`.** Several agents this session branched from `main` (which lacks every commit in this hand-off) and produced commits that needed rebasing before merge. The first commit in each agent's prompt should be `git checkout dev && git pull` if not already on dev.
- **One executor.rs-touching agent at a time.** `crates/tensor-wasm-exec/src/executor.rs` was the most contended file this session — T1 (no), T7 (deprecation), T33 (args), T34 (streaming), T36 (deadline), T37 (instance pool, never launched) all want it. Sequence them.
- **Single CHANGELOG.md owner per merge wave.** Letting every agent write `[Unreleased]` produces N-1 conflicts. Alternative for future sessions: have agents write CHANGELOG entries to a per-task file (`changelog-fragments/T34.md` etc.) and have a final consolidating step.
- **Worktree branches' base detection.** Agents reported their worktree's HEAD relative to `dev` only sporadically. Twice this session an agent's "I worked from dev" was actually "I worked from main" and the resulting merge had to re-base. Tighten the prompt: "Run `git log --oneline -1 dev` and `git rev-parse HEAD` and INCLUDE both in your report. If they differ, `git reset --hard dev` BEFORE writing any code."
- **Manual application path.** When in doubt about an agent's correctness, `git diff` the worktree, copy the changed files into the main tree directly, and commit yourself. T39 went this route mid-session and it worked.

## 10. Recap of what this session added (by the numbers)

| Metric | Count |
|---|---|
| Commits to `dev` | ~73 |
| Tasks defined | 41 |
| Tasks completed | 35 (28 audit + 6 v0.4 feature + T33/T34 + T31/T32 hygiene) |
| Tasks in-flight at handoff | 2 (T30, T41) |
| Tasks pending at handoff | 2 (T29, T37) |
| Critical vulnerabilities closed | 1 (`/kernels` tenant gap — T1) |
| High vulnerabilities closed | 6 |
| Medium vulnerabilities closed | 7 |
| Low / hygiene items closed | 7 |
| Perf clear-wins shipped | 9 |
| v0.4 feature deliverables landed | 7 |
| New files | ~30 (mostly tests + one new script directory entry) |
| Lines added to dev (rough) | ~6000 |
| Worktree branches left in `.claude/worktrees/` | ~25-30 |

---

_End of continuation prompt. The tree has the security / perf / v0.4 work
ready in `dev` but has NOT been built. First action on resumption is
§ 5 (verification recipe). Then merge the in-flight T30/T41, do T37 +
T29, tag v0.3.7._
