# Changelog

All notable changes to Craton TensorWasm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-25

This release consolidates the v0.2, v0.3 and v0.4 PATH-TO-V1 milestones'
implementable items into a single shipped version. CUDA-hardware-bound
items (S22 runner, measured GPU dispatch numbers, MPS production
validation) and external-party items (pen-test, design-partner
deployments) remain deferred — see `docs/PATH-TO-V1.md` for the
ongoing roadmap.

### Upgrade notes

- **Bare-token entries in `TENSOR_WASM_API_TOKENS` are deprecated.**
  Rewrite as `token:tenant=*` (preserves current behavior) or
  `token:tenant=1,2,3` (per-tenant scope). Removal target: 1.0.
- **Per-tenant scope enforcement on `/invoke`.** Calls that worked
  under 0.1 with a bare token now require `:tenant=*` or an explicit
  scope match; out-of-scope calls return `403 tenant_scope_denied`.
- **Audit log emission on state-mutating routes.** Default sink is
  stdout; disable with `TENSOR_WASM_API_AUDIT_LOG=none` if undesired.
- **Per-token rate limiting** subject to
  `TENSOR_WASM_API_RATE_LIMIT_QPS` and `_BURST`. Disabled when either
  is unset/zero (matches pre-0.3 behavior).
- **HTTP request metrics** now emitted under
  `tensor_wasm_http_requests_total`, `*_duration_seconds`, and
  `*_in_flight`. Dashboards that previously had TODO markers for these
  series now have real data.
- **Snapshot reader** still enforces the 256 MiB decompression-bomb cap
  by default; trusted-snapshot callers should opt in via
  `SnapshotReader::with_max_decompressed`.

See `docs/MIGRATION-v0-to-v1.md` for the full deprecation /
behavioral-change table.

## [0.3.1] - 2026-05-25

Additive cuda-oxide scaffolding release. No breaking changes; default
workspace build untouched. Three new opt-in landings under the
new `cuda-oxide-backend` feature flag.

### Added
- `rfcs/0001-cuda-oxide-integration.md` — first real RFC under the
  W1.7 process. Proposes Option C (three backends side-by-side from
  v0.3.1 through v0.4; default flips to cuda-oxide at v0.5 contingent
  on cuda-oxide reaching v0.2.0; cudarc-backend is the documented
  fallback if it doesn't). Advances PATH-TO-V1 Open Decision #1 (O1).
- `tensor-wasm-mem`: new `cuda-oxide-backend` feature flag and
  `cuda_oxide_backend` scaffold module (`CudaOxideUnifiedBuffer`,
  stubbed `allocate` / `apply_advice`, `Drop` with tracing). Workspace
  deps for `cuda-host`, `cuda-core`, `cuda-async` pinned via git rev
  pointing at github.com/NVlabs/cuda-oxide (NOT yet on crates.io as of
  2026-05-25; the crates.io `cuda-oxide` name is a different,
  unrelated 2018-era project). Coexists with `unified-memory` (cust)
  and `cudarc-backend` (W1.2); the three are independent (O2).
- `tensor-wasm-jit`: new `cuda-oxide-backend` feature flag and
  `pliron_dialect` scaffold module (the future `cranelift_to_dialect_mir`
  lowering pass that targets cuda-oxide's Pliron-based IR). Includes a
  23-row mapping table from Cranelift IR ops to Pliron `dialect-mir`
  ops so the v0.4 author has a concrete target (O3).
- `docs/CUDA-KERNELS.md`: new Section 5 — "Path C: Rust kernels via
  cuda-oxide". Decision table for picking among Path A (hand-PTX),
  Path B (.cu via nvcc), and Path C (Rust via cuda-oxide) (O4).

### Changed
- `docs/RISKS.md`: cust EOL row cross-references RFC 0001 + cuda-oxide
  as the third option (O5).
- `docs/CUDARC-SPIKE.md`: Version-chosen table gains a `cuda-host`
  row; cudarc recommendation still valid as v0.3.x default + v0.5
  fallback (O5).
- `docs/PATH-TO-V1.md`: Open Decision #1 gains an inline 2026-05-25
  update line cross-linking RFC 0001 (O5).

### Toolchain note (NOT changed)
- Workspace pin remains `nightly-2026-03-15`. Enabling the new
  `cuda-oxide-backend` feature requires a local override to
  `nightly-2026-04-03+` (cuda-oxide's pin). The workspace default
  bump is scheduled for v0.4 per RFC 0001 "Toolchain plan" step 3
  and PATH-TO-V1 Open Decision #8 (quarterly cadence).

## [0.3.2] - 2026-05-25

cuda-oxide v0.4-prep wave. Four follow-ups to v0.3.1's scaffolding.
Builds clean on the bumped nightly with default features; CUDA tests
(W1.1 kernel_args_e2e, W5.9 cudarc_smoke) still pass on real RTX 2060.

### Added
- `deploy/helm/tensor-wasm/values.yaml` + Nomad job specs gain
  `image.backend` (cust | cudarc | cuda-oxide | "") toggle. Build-time
  selection via image-tag suffix. Same shape mirrored as a manual-swap
  comment in `deploy/k8s/20-deployment.yaml`. Resolves RFC 0001
  Unresolved question #4 (F1).
- `deny.toml` at workspace root. Posture: deny all unknown git +
  registry sources. Two `allow-git` allowlist entries — NVlabs/cuda-oxide
  and vaivaswatha/pliron — each with rationale and cross-ref to
  RFC 0001 (F2).
- `docs/REPRODUCIBLE-BUILDS.md` "Git-pinned sources" subsection +
  audit-trail table (Crate / Repository / Pinned rev / Rationale) (F2).
- `crates/tensor-wasm-bench/benches/dispatch_future_backends.rs` —
  new tail-latency bench comparing the busy-poll DispatchFuture
  against a stubbed cuda-async backend. First baseline numbers
  committed to `bench-results/dispatch-future-backends.json`
  (busy-poll P50 400 ns, P99.9 1.8 µs on RTX 2060 / WDDM).
  cuda-async slot returns the documented "not yet wired" sentinel
  until the v0.4 port. Resolves RFC 0001 Unresolved question #3 (F3).

### Changed
- `Cargo.toml`: cuda-host / cuda-core / cuda-async git deps switch
  from `branch = "main"` to the v0.1.0 tag SHA
  `4a56e4220aab8ce5d085a411e7f806cebb647d14`. Reproducible +
  dependabot-blind + auditable (F2).
- `rust-toolchain.toml`: bumped from `nightly-2026-03-15` to
  `nightly-2026-04-03` to match cuda-oxide's pin. Per RFC 0001
  Toolchain plan step 3 / PATH-TO-V1 Open Decision #8 (quarterly
  cadence). Workspace + benches + tests compile clean on the new
  pin. Rollback is a one-character revert if anything downstream
  breaks (F4).

## [0.3.3] - 2026-05-25

The v0.3.2 audit follow-through. Six surgical fixes closing the
highest-ROI gaps. **Headline: pitch points 1 + 2 + 3 now have real
end-to-end correctness proof on RTX 2060 silicon.**

### Fixed
- `DispatchFuture::poll` no longer busy-spins. On `EventStatus::NotReady`
  the future now clones the waker, spawns a 50 µs tokio sleep, and
  yields the worker. Pre-B1, every outstanding kernel pinned a tokio
  worker at 100 % CPU until the event completed — that broke the
  "Hyper-Scale Async" pitch under any non-trivial load. Real
  cuStreamAddCallback-driven future remains v0.4 cuda-async work (B1).
- CLI shell completions + man pages regenerated from the binary
  (W2.4 had committed hand-rolled approximations; net +1046 lines of
  drift caught) (B4).
- CI workflow toolchain pins synced to `nightly-2026-04-03` to match
  the F4 workspace bump (8 workflow files; previously CI would have
  failed on next push) (B3 + F4 follow-up).

### Added
- **`tests/kernel_args_e2e.rs::dispatch_pipeline_compiles_against_real_module_bytes`**:
  unignored on every CUDA host. Registers the canonical
  `kernels/vector_add.ptx` via `cust::module::Module::from_ptx` and
  asserts the launch result is in `{Ok, MalformedPtx, LaunchFailed,
  InvalidKernel}` — explicitly NOT `InvalidArgs` / `InvalidPointer`
  (those would mean marshalling regressed). PASSES on RTX 2060 +
  Windows WDDM + CUDA 13.2 (B2).
- **`vector_add_end_to_end_real_ptx_real_kernel`**: builds Wasm guest
  with f32[64] arrays in linear memory, dispatches via wasi-cuda host
  with typed argv pointing into linear memory, reads the `c` region
  back out, asserts `c[i] == a[i] + b[i]` for all 64 elements.
  Marked `#[ignore = "requires SM_80+"]` defensively — but in
  practice the CUDA driver JIT'd the `.target sm_80` PTX up to SM_75
  SASS on the local box and the test **passed end-to-end with
  correct output** (B2).
- `cargo-deny` enforcement job in `.github/workflows/ci.yml` running
  `cargo deny check --all-features`. The F2 `deny.toml` allowlist
  posture is now enforced per-PR (B3).
- `UnifiedBuffer::is_uvm_backed()` + `TensorWasmLinearMemory::is_uvm_backed()`
  probes and 5 new tests pinning the property that under
  `--features unified-memory` the wasm linear memory IS allocated via
  `cuMemAllocManaged` and `as_ptr()` is reachable as a device pointer
  by kernel args. Closes v0.3.2 audit Problem #5 — the wiring already
  existed; what was missing was provable assertion + documentation
  (B5).
- `bench-results/hyperfine-vs-wasmtime.json` — re-run of the
  dimension-1 wasmtime comparison post-W4.1 / W2.2 / W2.3.
  Pre: wasmtime 1.05× faster. Post: tensor-wasm 1.02× faster, CIs
  overlap = statistically tied. **Tracing + audit + metrics overhead
  not measurable on the CLI path** (the instrumentation lives in the
  HTTP layer, CLI bypasses it) (B6).

### Audit problems resolved this release

- Problem #1 (busy-poll) → B1
- Problem #2 (cargo deny not enforced) → B3
- Problem #3 (hand-rolled completions/man pages) → B4
- Problem #5 (linear memory UVM wiring unproven) → B5
- Problem #10 (stale hyperfine comparison) → B6
- Problem #14 (no end-to-end PTX correctness test) → B2 — **the
  pitch-validating one**

Problems still open: #4 (PATH-TO-V1 v0.2/v0.5 framing drift), #6
(jobs_active / tenant_gpu_memory_bytes still TODO), #7 (no CI fuzz
cron), #8 (no S22 self-hosted runner), #9 (bench numbers high CV),
#11 (tracing span double-count under load — untested), #12 (wit
docstring vs WIT version), #13 (Helm image tags reference
unprovisioned registry), #15 (TBD placeholders in MAINTAINERS /
GOVERNANCE / CVE dry-run).

## [0.3.4] - 2026-05-25

Audit-closure wave (C1-C9). All nine outstanding audit problems from
the v0.3.3 grade closed or resolved-with-disclosure.

### Fixed
- **Audit Problem #4** (PATH-TO-V1 v0.2/v0.5 framing drift): Open
  Decision #1 + risk register row updated to reflect RFC 0001
  re-scope to v0.5 (C5).
- **Audit Problem #11** (W4.1 tracing may double-count under load):
  **disproven.** New `trace_concurrent_load_test` runs 64 concurrent
  async invokes on 4 worker threads + counts spans via a custom
  `tracing_subscriber::Layer`; emits exactly `4 * N` spans with no
  orphans, no duplicate trace_ids. The audit concern was wrong; the
  test pins the property going forward (C2).

### Added
- `tensor_wasm_jobs_active` gauge + `tensor_wasm_gpu_memory_bytes_per_tenant`
  family. W2.5 dashboard panels that rendered `n/a` for these now
  render real data. Audit Problem #6 closed (C3).
- `.github/workflows/cuda.yml` self-hosted CUDA runner workflow + 4
  per-backend jobs; dormant until a runner registers with the
  `self-hosted,cuda` labels. New `docs/runbooks/self-hosted-cuda-runner.md`
  procedure runbook covers registration through required-check
  wiring. Audit Problem #8 closed (will activate when a runner
  registers) (C1).
- `.github/workflows/fuzz-long.yml` — weekly 5.5h-per-target fuzz cron
  with year-keyed corpus caching. After 5 weekly runs the cumulative
  per-target wall-clock clears the v0.5 "24+ hours" gate. Also added
  the two W4.7 targets to the existing nightly `fuzz.yml` matrix.
  Audit Problem #7 partially-incorrect (a nightly cron already
  existed); fully closed now (C4).
- `Dockerfile` at repo root producing all four backend variants via
  `--build-arg BACKEND={"",cust,cudarc,cuda-oxide}`. Helm chart README
  rewritten to point at it. Audit Problem #13 closed (C8).
- `scripts/run-quiet-bench.{sh,ps1}` — bench drivers that raise
  `--sample-size` from 100 to 500 and pin CPU governor / power plan.
  Audit Problem #9 partially closed (script delivered; publication-
  grade numbers still require the S22 runner from C1) (C6).

### Changed
- `MAINTAINERS.md`: new "Placeholders are by design" section near the
  top articulating the v0.x convention that the 19 TBD cells are
  intentional, each with a documented unblock trigger. No names
  invented. Audit Problem #15 closed (C9).
- `wit/wasi-cuda.wit`: package version bumped `wasi:cuda@0.1.0` →
  `wasi:cuda@0.2.0` to reflect the W1.1 typed-argv contract change.
  Inline "Version history" comment block added. Audit Problem #12
  closed (C7).
- `docs/BENCHMARKING.md` CV target section gains explicit disclosure
  that committed bench-results numbers were captured on a noisy
  developer host (CV > 5%); usable as regression-gate floors, not as
  publication-grade comparison data (C6).
- `.github/workflows/fuzz.yml` matrix grew the W4.7 targets
  (token_scope_parser, audit_json_round_trip) (C4).

### Audit status after C wave

All 9 problems from the v0.3.3 audit are now closed or have a closure
path landed (C1 + C8 close on first registration / first registry
provisioning respectively; both have the operator-side procedure
documented). 0 audit problems remain open as of v0.3.4.

## [0.3.5] - 2026-05-25

The "pre-stage the blocked" wave (D1-D6). The user asked us to take
the v0.4 + v0.5 forward-work list — items I'd previously labeled
"blocked on cuda-oxide v0.2 / hardware / sponsor procurement / BD" —
and proceed. The honest landing: real code where implementable, real
operator-facing runbooks + RFP for items where the blocker is
external action. Two items declared out-of-scope-with-rationale rather
than ship vapor.

### Added (code)
- `UnifiedBuffer::try_grow_in_place(new_size)` scaffold — returns
  the documented `"in-place grow not yet wired"` sentinel until v0.4
  cutover wires `cuMemAddressReserve + cuMemMap`. Rationale: ~300-500
  LOC of careful unsafe FFI that needs `concurrentManagedAccess`-
  capable hardware (Linux datacenter GPU) to verify. Scaffold gives
  the v0.4 author a target signature + the four known constraints
  rather than a blank canvas. `supports_in_place_grow()` const fn
  returns `false` today; flips at v0.4. New test pins the sentinel
  string (D1).
- `Backing::Cudarc(CudarcUnifiedBuffer)` — third `UnifiedBuffer`
  backing under `--features cudarc-backend` (when `unified-memory`
  is off). Precedence: `unified-memory` wins if both. Expands the
  module-level precedence table from 2 to 4 rows; `is_uvm_backed()`
  now true for cust + cudarc + future cuda-oxide; only the
  default `Box<[u8]>` returns false. New test
  `cudarc_unified_buffer_smoke.rs` exercises the path on real
  silicon (D2).

### Added (operator-facing artifacts for blocked items)
- `docs/CUDA-OXIDE-CUTOVER.md` — 680-line cutover runbook for the
  day cuda-oxide v0.2 ships. 8 numbered steps from dependency bump
  through default flip. Gated on four pre-conditions; if any fails
  cudarc-backend remains v0.5 default per RFC 0001 Option C
  contingent-no path. Longest step (Pliron-dialect lowering
  implementation): ~3 days / 200-400 LOC implementing the first 4
  of O3's 23 mapping rows (D3).
- `docs/runbooks/ghcr-registry-provisioning.md` — 449-line sponsor-
  side runbook to provision `ghcr.io/craton-co/tensor-wasm`. 7-step
  procedure. Includes the `release.yml` matrix snippet operators
  paste in (4 image variants: "", cust, cudarc, cuda-oxide). Audit
  Problem #13 closes the moment this runbook is executed (D4).
- `docs/SECURITY-AUDIT-RFP.md` — 479-line procurement-grade RFP a
  sponsor can send to Trail of Bits / NCC Group / Cure53 /
  Doyensec today (PATH-TO-V1 Open Decision #5) after filling in 11
  `[bracketed]` placeholders. v0.5 PATH-TO-V1 "External pen-test
  commissioned" gate is now sponsor-procurement-bound rather than
  code-bound (D5).
- `docs/DESIGN-PARTNER-PROGRAM.md` — 586-line outreach + program kit
  for recruiting v0.5 design partners (PATH-TO-V1 Open Decision #6).
  8 sponsor deliverables vs 7 partner asks. Application template
  (Section 10) is copy-pasteable BEGIN/END-marked form. Sponsor
  maintainer-sync MUST sign off on §6.4 before the kit ships to
  candidates: it explicitly REFUSES to promise a faster severity-1
  SLA than `SECURITY.md` backport policy (creating a partner-tier
  SLA better than ordinary users' would corrode trust). Instead
  promises "priority queue position within the published policy" (D6).

### Out of scope (D7 — declared, not implemented)
- **Pliron-dialect actual lowering** — implementable in principle,
  but committing 2-3 weeks of careful Rust to a dialect-mir shape
  that may change in cuda-oxide v0.2 is high-risk. The O3 scaffold
  defines the trait + 23-row mapping table; the v0.4 cutover
  runbook (D3 Step 4) is the right place to implement the first 4
  rows once v0.2 ships. Re-evaluate when cuda-oxide v0.2 lands.
- **Cross-version snapshot compat matrix expansion** — W1.3 already
  covers v0.1.0 → current with golden fixtures + 4 active tests.
  The matrix expands when format-bumping releases ship; there is
  exactly one format version today, so the "matrix" is structurally
  done. The W1.3 framework auto-extends when v0.2 format ships; no
  pre-work would add value.

### v0.5 PATH-TO-V1 status after D wave

Two of the four "blocked on external action" v0.5 gates now have a
sendable artifact: the security-audit RFP (D5) ready to send to
firms; the design-partner kit (D6) ready to hand to candidate orgs.
The remaining two gates — actually commissioning the pen-test and
recruiting partners — are sponsor-procurement / BD work that AI
agents cannot do.

The cuda-oxide v0.2 cutover (D3) is similarly pre-staged: an
executable 8-step runbook ready for the maintainer who runs `git
checkout` the day v0.2 ships.

## [Unreleased]
_No entries yet — open the next PR adding one._

## [0.3.6] - 2026-05-27

Hardening + correctness wave. Forty-plus surgical landings across
`mem`, `exec`, `jit`, `wasi-gpu`, `api`, `tenant`, `snapshot`, `cli`,
and the build / docs surface. No new feature surface area; existing
surfaces tightened. Headline items: a tenant-quota capability gate
that closes the cross-tenant quota-mutation attack surface, several
silent-corruption-class JIT and exec fixes, constant-time bearer
comparison on the API path, and the first batch of fuzz targets for
snapshot restore + argv parsing + wasm rewrite + pool allocation.

### Added
- `tensor-wasm-tenant` — `TenantCapability` newtype + `TenantRegistry::register_with_capability`.
  Unforgeable proof-of-authority gate on quota mutation; closes the
  cross-tenant attack surface where any holder of an
  `Arc<TenantContext>` could mutate any tenant's `bytes_in_use`. The
  unchecked `consume_bytes` / `release_bytes` are retained as
  `#[deprecated]` shims; targeted for removal in v0.4.
- `tensor-wasm-api` — `CorsLayer` with an explicit origin allowlist,
  configured via the new `TENSOR_WASM_API_CORS_ALLOWED_ORIGINS`
  environment variable. Empty / unset = no cross-origin requests
  permitted (previous default behavior; now explicit).
- `tensor-wasm-snapshot` — `SnapshotReader::restore` now validates
  `metadata.total_uncompressed_bytes` against the actual sum of
  decompressed blob sizes; mismatches fail the restore rather than
  silently returning a truncated state.
- `tensor-wasm-exec` — per-call epoch deadline re-arm. Second and
  subsequent calls on the same `TensorWasmInstance` no longer inherit
  the first call's residual epoch budget; each `call` re-arms the
  deadline so timeout numbers correspond to wall-clock per-call work.
- `fuzz/` — four new libfuzzer targets: `fuzz_snapshot_restore`
  (already had a target; corpus extended), `fuzz_parse_argv`,
  `fuzz_rewrite_wasm`, `fuzz_pool_allocate`. Each wired into the
  nightly `fuzz.yml` matrix and the weekly long-form cron from C4.
- `SUPPORT.md`, `CITATION.cff`, `docs/glossary.md` — community
  scaffolding: where to file what, how to cite the project in
  academic work, single-source glossary of project-specific terms.
- `.github/CODEOWNERS` — team-based review routing
  (`@craton-co/maintainers` default; `@craton-co/security` on
  security-sensitive paths; `@craton-co/release` co-owning `Cargo.toml`
  and `CHANGELOG.md`). Uses teams (not individuals) so the file does
  not need editing every time the maintainer roster changes.
- `tensor-wasm-tenant` — `isolation_downgrade_count()` process-wide
  counter exposed for operators that requested `ContextIsolated` and
  need to alert on any silent downgrade to `StreamIsolated`.

### Changed
- `wasi-cuda` — host import-module bumped from `wasi:cuda/host@0.1.0`
  to `wasi:cuda/host@0.2.0` to match the `wit/wasi-cuda.wit` package
  version bumped in v0.3.4 (C7). Guests built against the older
  module name will fail to link and must be re-built; the new
  module name is the W1.1 typed-argv contract.
- `tensor-wasm-cli` — `--max-decompressed` renamed to
  `--max-archive-bytes` on `snapshot restore`. The old flag name is
  retained as a deprecated alias that emits a one-line warning and
  forwards to the new flag; removal target v1.0. Documentation,
  shell completions, and help-output snapshots all updated.
- `tensor-wasm-exec` — module cache key widened from a truncated
  `u64` (first 8 bytes of BLAKE3) to the full `[u8; 32]` digest.
  Collision probability drops from ~birthday-bound at 2^32 cached
  modules to cryptographically infeasible.
- `tensor-wasm-mem` — `UnifiedBuffer` no longer `memset`s the entire
  allocation on the cust path. Only the Wasm-visible window
  (`0..len`) is zero-filled at construction; the post-`len` tail
  (rounded up to page boundary) is left uninitialized, matching
  what Wasmtime's `MemoryCreator` contract actually requires and
  saving up to one page of zeroing per allocation. Cust-path leak
  counter added so the test suite can pin no-leak invariants under
  the new code path.
- `tensor-wasm-api` — bearer-token comparison rewritten to use
  `subtle::ConstantTimeEq` instead of `==`. Timing-leakable byte-wise
  comparison is gone; tokens of different lengths now also
  constant-time-compared to the same length (the length itself is
  no longer a side channel).
- `tensor-wasm-api` — body-limit middleware standardized on axum's
  `DefaultBodyLimit::max(...)` (was a mix of axum's limit and a
  separate `tower-http` request-body limit; the tower-http variant
  had a 32 MiB default that silently undercut the documented 256 MiB
  cap). Documentation in `API.md` and the openapi schema aligned to
  the single value.
- `tensor-wasm-tenant` — `release_bytes` reimplemented as a CAS loop
  computing `saturating_sub`. The previous `fetch_sub` + post-hoc
  `store(0)` shape was racy under concurrent `consume_bytes`.

### Deprecated
- `tensor-wasm-tenant::TenantContext::consume_bytes` and
  `release_bytes` — use `consume_bytes_with_capability` and
  `release_bytes_with_capability` with the `TenantCapability` minted
  by `TenantRegistry::register_with_capability`. Removal target v0.4.

### Fixed
- `tensor-wasm-mem::pinned_host` — integer overflow guard on the
  `usable + 2 * page` size computation. Previously a sufficiently
  large `usable` could wrap around silently and allocate a buffer
  smaller than the caller asked for.
- `tensor-wasm-mem::cudarc_backend` — `OnceLock` race fix.
  Concurrent first-time initialisation could lose the CUDA primary
  context and silently release the device on the losing thread;
  now `set_or_take`-shaped so the winner's context is the one that
  sticks.
- `tensor-wasm-exec` — JIT scratch arena no longer overlaps the
  guest's static data sections. The arena was previously placed in
  the same linear-memory page range as guest globals; tight loops
  could observe scratch writes as data corruption.
- `tensor-wasm-exec` — `ResourceLimiter::table_growing` now caps
  growth at the configured `engine_max_table_bytes` value (was:
  caps at the per-table element count, which let large element
  sizes exceed the byte budget).
- `tensor-wasm-wasi-gpu` — pointer aliasing across `.await` fixed
  in the kernel-launch path. Argv lowering and dispatch now resolve
  inside the same critical section so a concurrent `wasi_cuda_launch`
  on the same instance cannot observe a half-lowered argv vector.
- `tensor-wasm-wasi-gpu` — every host function (`wasi_cuda_load_ptx`,
  `wasi_cuda_launch`, `wasi_cuda_sync`, `wasi_cuda_last_error_*`)
  now gated behind an explicit `WasiCudaCapability` rather than
  unconditionally linked into every instance. Workloads that did
  not opt into `wasi-cuda` no longer see the imports.
- `tensor-wasm-jit` — `MatMul` PTX emission returns
  `JitError::NotYetImplemented` instead of producing a structurally
  broken `wmma` kernel. The previous emitter generated PTX that
  passed `ptxas` but produced wrong results at SM_80; the explicit
  error surfaces the gap at compile time.
- `tensor-wasm-jit` — rewrite trampoline now traps on a nonzero
  dispatch result. Previously a nonzero return from the host
  dispatch was silently dropped and the guest continued as if the
  call had succeeded.
- `tensor-wasm-jit` — rewrite arithmetic uses `checked_add` /
  `checked_mul` throughout. Overflow in the trampoline math is
  now a trap rather than a silent wrap.
- `tensor-wasm-jit` — `KernelCache` eviction holds the LRU lock
  across the eviction loop instead of dropping and re-acquiring on
  every iteration. Closes a window where two threads could both
  observe the same victim and double-free.
- `tensor-wasm-snapshot` — `examples/generate_golden.rs` migrated
  to the bincode 2.x `Encode` / `Decode` derive API (was still on
  the 1.x `bincode::serialize` / `deserialize` shape, which silently
  no-op'd after the workspace bincode bump).
- `tensor-wasm-cli` — `Box::leak` in `cmd/man.rs` retained only
  where clap 4.6's `Str` API requires it (`Command::name` accepts
  `impl Into<Str>` and clap 4.6 only implements `From<&'static str>`).
  The bounded leak (≤10 subcommands × <32 B per process call) is
  documented in-line.
- `docs/FORMAT.md` — `instance_id` width corrected from `u64` to
  `u128` (16 bytes on the wire). The format-on-disk has always been
  `u128`; the doc was wrong.
- `Dockerfile` — runtime image now installs `curl` so the
  `HEALTHCHECK` directive actually has the binary it invokes
  (previously the `HEALTHCHECK` line referenced a `curl` that was
  not present in the distroless runtime layer; container runtime
  reported `unhealthy` immediately on start).
- `crates/tensor-wasm-api/openapi.json` regenerated as a hand-ported
  3.0.3 mirror of the canonical `openapi/tensor-wasm-api.yaml` (the
  file is referenced by both the CI `swagger-cli validate` job and
  the crate's `API.md` cross-link; the regenerated version carries
  v0.3.5 schemas, the `tenant_scope_denied` + `rate_limited` error
  kinds, and the 429 / 403 responses).

### Security
- **Cross-tenant quota mutation now requires `TenantCapability`** —
  the headline security landing of this release. Holding an
  `Arc<TenantContext>` is no longer sufficient to mutate that
  tenant's `bytes_in_use` counter; the caller must also present a
  `TenantCapability` minted at registration time. The unchecked
  variants are kept as deprecated shims for one minor cycle.
- **Timing-leakable bearer-token comparison removed** —
  `subtle::ConstantTimeEq` replaces `==` on the
  `TENSOR_WASM_API_TOKENS` matching path.
- **wasi-gpu host functions gated behind explicit capability** —
  workloads that did not opt into `wasi-cuda` no longer have the
  host functions linked into the instance, removing them from the
  guest's reachable surface area.
- **OnceLock race in cudarc backend** — prevents silent CUDA
  primary-context release on the losing thread of a first-time
  initialisation race, which would have left the process running
  with no device context and (until first kernel launch failed)
  no audible failure mode.

### Build
- Workspace path dependencies now carry an explicit `version =
  "0.3.5"` field alongside the `path = "..."` field, so
  `cargo publish` from a fresh checkout no longer fails with
  "missing version for path dep". Crates.io publish readiness.
- `Dockerfile` distroless runtime now runs as `USER nonroot:nonroot`
  (was: implicit `root`). Matches CIS Docker Benchmark §4.1.
- `tower-http` `"limit"` feature dropped — body-limit standardisation
  on axum's `DefaultBodyLimit` made the tower-http limiter dead code.
- `tensor-wasm-tenant` — unused `parking_lot` dependency dropped.






### Security
- Repository ownership transferred to Craton Software Company
- Added LICENSE / NOTICE files (Apache-2.0)
- `tensor-wasm-api` now supports per-tenant scoped bearer tokens via the
  `token:tenant=N,M` syntax in `TENSOR_WASM_API_TOKENS`; cross-tenant access
  is refused with a `tenant_scope_denied` 403 (W2.1, advances v0.3).
- Structured audit log for state-mutating routes, opt-in via
  `TENSOR_WASM_API_AUDIT_LOG`; format and rotation guidance in
  `docs/AUDIT-LOG.md` (W2.2, advances v0.3).
- `docs/DEPLOYMENT.md` gains an mTLS section covering both self-terminated
  TLS and reverse-proxy fronting (W2.8, advances v0.3).
### Added
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, MAINTAINERS.md, docs/RISKS.md
- GitHub issue/PR templates and dependabot configuration
- `tensor-wasm-wasi-gpu` — typed argv lowering for scalar and pointer kernel
  arguments; `KernelArgsUnsupported` is now reserved for sanity-cap
  rejections rather than the missing-marshaller case (W1.1, advances v0.2).
- `tensor-wasm-mem` — cudarc backend spike behind the new
  `--features cudarc-backend` flag; `cust` remains the default backend
  (W1.2, advances v0.2).
- `tensor-wasm-snapshot` — cross-version compatibility test framework with
  golden fixtures under `crates/tensor-wasm-snapshot/tests/fixtures/`
  (W1.3, advances v0.2).
- `tensor-wasm-api` — per-token QPS rate limiting middleware configured via
  `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST`;
  retires the global `ConcurrencyLimitLayer(64)` workaround called out in
  the 0.1.0 known limitations (W1.4, advances v0.2).
- `tensor-wasm-api` — HTTP request metrics middleware exporting
  `tensor_wasm_http_requests_total`, a request-duration histogram, and an
  `in_flight` gauge (W2.3, advances v0.3).
- `tensor-wasm-cli` — `tensor-wasm observe` subcommand for live metrics
  tailing (W1.5, advances v0.2).
- `tensor-wasm-cli` — generated shell completions and man pages, plus a
  `tensor-wasm man` subcommand that prints them at runtime (W2.4, advances
  v0.3).
- `deploy/` — Kubernetes manifests and a Helm chart for the API gateway
  (W2.7, advances v0.3).
- `docs/CUDA-SETUP.md` rewritten end-to-end with no hedging language
  (W1.6, advances v0.2).
- `rfcs/` — lightweight RFC process with template and index
  (W1.7, advances v0.2).
- `GOVERNANCE.md` — project governance scaffold (W1.8, advances v0.2).
- `docs/SLO.md` — SLO definitions and burn-rate alert recipes
  (W1.9, advances v0.2).
- `docs/dashboards/` — reference Grafana dashboard JSON
  (W2.5, advances v0.3).
- `docs/runbooks/` — per-alert operator runbooks keyed to the SLO alerts
  (W2.6, advances v0.3).
- `docs/WASMTIME-FORK.md` — wasmtime upgrade cadence policy
  (W2.9, advances v0.4).
### Changed
- Workspace authors and repository metadata updated to craton-co/craton-tensor-wasm
- Bumped `prometheus-client` from `0.22` to `0.24` so `tensor_wasm_gpu_memory_used_bytes`
  can use the native `Gauge<u64, AtomicU64>` (added upstream in 0.23.0 via
  prometheus/client_rust#226) instead of the previous signed-int workaround.
### Deprecated
- Bare-token entries in `TENSOR_WASM_API_TOKENS` (tokens without a
  `:tenant=...` scope clause) are deprecated in favour of the scoped form
  introduced in W2.1. Unscoped tokens still authenticate but are logged
  with a deprecation warning; targeted for removal in v1.0.

## [0.1.0] — 2026-05-23

The first scaffold release of TensorWasm — a GPU-accelerated serverless WebAssembly
runtime. Every subsystem in the architecture is present and tested; CUDA-
bound paths are feature-gated and exercised by a self-hosted CUDA CI runner
(deferred until the appropriate hardware lands).

### Added

#### Crates
- `tensor-wasm-core` — `TensorWasmError` enum, `TenantId`/`InstanceId`/`KernelId`
  newtypes, Prometheus metrics registry, `tracing-subscriber` init with an
  optional OTLP exporter (`otlp` feature).
- `tensor-wasm-mem` — `UnifiedBuffer` with feature-gated CUDA backing,
  `UnifiedMemoryPool` bump allocator, `cudaMemAdvise` hint helpers,
  `PinnedHostBuffer` fallback, Wasmtime `MemoryCreator` integration,
  `IsolationLevel` taxonomy.
- `tensor-wasm-exec` — `TensorWasmEngine` wrapping `wasmtime::Engine` with epoch-based
  interruption; `TensorWasmInstance` + `InstanceState`; `TensorWasmExecutor` with
  async spawn / call / terminate. 100-concurrent integration test plus
  epoch-timeout regression test.
- `tensor-wasm-wasi-gpu` — `wasi-cuda` host bridge (`wasi:cuda/host@0.1.0` ABI:
  `wasi_cuda_load_ptx`, `wasi_cuda_launch`, `wasi_cuda_sync`,
  `wasi_cuda_last_error_*`). Instance-scoped `KernelRegistry` with
  per-owner authorisation. Stub on non-CUDA hosts; real dispatch behind
  the `cuda` feature. Async dispatch + back-pressure semaphore.
- `tensor-wasm-jit` — Cranelift-free detector over a simplified `BlockIR`,
  `TensorWasmKernelBlueprint` IR, CLIF→IR lowering, PTX text emitter for
  sm_80 (including `wmma` for MatMul), LRU `KernelCache`, `DeoptGuard`.
- `tensor-wasm-snapshot` — `SnapshotWriter::capture` and `SnapshotReader::restore`
  with CRC32 integrity check, per-field size limits, zstd compression,
  format version 2.
- `tensor-wasm-tenant` — `TenantContext` and `TenantRegistry`; MPS-or-fallback
  decision based on `/tmp/nvidia-mps` existence.
- `tensor-wasm-api` — Axum 0.7 HTTP gateway: `GET /healthz`, `GET /metrics`,
  `POST /functions`, `DELETE /functions/{id}`, `POST /functions/{id}/invoke`,
  `POST /functions/{id}/invoke-async`, `GET /jobs/{id}`. Structured JSON
  error envelope; tower-http timeout, trace, and concurrency-limit middleware;
  W3C `traceparent` propagation.
- `tensor-wasm-cli` — `tensor-wasm` binary: `run`, `deploy`, `invoke`, `bench`,
  `snapshot save/restore`, `metrics`, `completions`.
- `tensor-wasm-bench` — Criterion bench harness with 5 bench targets:
  `kernel_dispatch`, `cold_start`, `memory_bandwidth`, `jit_compile`,
  `e2e_inference`.

#### Documentation
- `README.md`, `ARCHITECTURE.md`, `SECURITY.md`.
- `docs/`: `BUILD.md`, `CUDA-SETUP.md`, `MPS-SETUP.md`, `AUTO-OFFLOAD.md`,
  `COLD-START.md`, `CLI.md`, `PERFORMANCE.md`, `OBSERVABILITY.md`,
  `SECURITY-AUDIT.md`, `WASMTIME-FORK.md`, `GETTING-STARTED.md`,
  `WASM-DEVELOPER-GUIDE.md`, `DEPLOYMENT.md`.
- `crates/tensor-wasm-api/API.md` — REST API reference.
- `kernels/vector_add.ptx` — sm_80 PTX fixture.
- `tests/wasm-fixtures/matrix_multiply.wat`.

#### Infrastructure
- `Cargo.toml` workspace with 10 member crates plus excluded `fuzz/` subdir.
- `rust-toolchain.toml` pinned to `nightly-2026-03-15`.
- `Makefile` with `build`, `test`, `bench`, `fmt`, `fmt-check`, `lint`,
  `check`, `doc`, `clean`, `ci`, `ci-bench`, `help` targets.
- `.github/workflows/ci.yml` — fmt + clippy + test + actionlint on
  ubuntu-latest with CUDA stub libs.
- `.github/workflows/bench.yml` — Criterion run with 10% regression check
  (regression diff stub; concrete diff lands in v0.2).
- `.github/workflows/release.yml` — `cargo publish --dry-run`, x86_64-linux
  binary build, GitHub Release upload on tag push.
- `docker-compose.yml` + `docker/` — observability stack (tensor-wasm-api +
  Prometheus + Grafana + Jaeger).
- `fuzz/` — cargo-fuzz package with three targets (`fuzz_wasm_compile`,
  `fuzz_ptx_emit`, `fuzz_snapshot_restore`).

### Known limitations
- CUDA paths (`unified-memory`, `cuda`, `auto-offload`, `mps`) require
  hardware that the public CI doesn't yet have. Local tests on a no-CUDA
  Windows host pass; CUDA-only tests are marked `#[ignore]`.
- Per-tenant rate limiting in `tensor-wasm-api` is currently global
  (`ConcurrencyLimitLayer(64)`). Per-tenant quotas track for v0.2 (BA-005).
- `cargo audit` is not yet in CI (BA-008).
- The `epoch_timeout` test is gated `#[ignore]` on Windows because of a
  Wasmtime fiber unwinding panic on epoch interrupt; the test runs on
  Linux/macOS CI.
- Auto-offload pipeline works against a simplified `BlockIR`; full
  Cranelift integration is deferred (see `docs/WASMTIME-FORK.md`).

[0.3.6]: https://github.com/craton-co/craton-tensor-wasm/releases/tag/v0.3.6
[0.1.0]: https://github.com/craton-co/craton-tensor-wasm/releases/tag/v0.1.0
