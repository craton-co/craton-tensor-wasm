# Craton TensorWasm — Path to v1.0 (proposal)

A proposed, opinionated roadmap from the current **v0.1.0 preview** to a
**v1.0 production** release. This is a proposal — it commits no one to
dates and invites pushback on the milestone shape, the exit criteria,
and the cut-line between "v1.0" and "v2.0". Treat it as the strawman
that future PRs and maintainer discussion sand down.

If you only read one section: skip to [What v1.0 means](#what-v10-means)
and [Anti-goals](#anti-goals-what-v10-does-not-promise) — those two
together define the bar.

## Contents

1. [What v1.0 means](#what-v10-means)
2. [Where v0.1.0 stands today](#where-v010-stands-today)
3. [Anti-goals — what v1.0 does NOT promise](#anti-goals-what-v10-does-not-promise)
4. [The five-milestone path](#the-five-milestone-path)
5. [Per-area workstreams](#per-area-workstreams)
6. [Open decisions to resolve before v1.0](#open-decisions-to-resolve-before-v10)
7. [Risk register](#risk-register)
8. [Effort and timeline (caveated)](#effort-and-timeline-caveated)
9. [Out of scope — deferred to v2.0](#out-of-scope--deferred-to-v20)
10. [Post-v0.3.6 strategic features](#post-v036-strategic-features)

---

## What v1.0 means

For TensorWasm, v1.0 is the line at which the project takes on three
commitments it does not make today:

1. **SemVer stability across the public API surface.** The HTTP API
   (`crates/tensor-wasm-api/API.md`), the WIT interface
   (`wit/wasi-cuda.wit`), the CLI (`tensor-wasm ...`), and every public
   Rust item in `tensor-wasm-core`/`tensor-wasm-mem`/`tensor-wasm-exec` follow semver
   strictly. Breaking changes require a major bump.
2. **A published SLA that survives external review.** Performance
   numbers are measured (not modeled), the regression gate fails the
   build on real drift (not synthetic ceilings), the security posture
   has been validated by an outside party, and the operations docs are
   enough that a competent SRE who has never touched TensorWasm can run it
   in production with the existing runbooks.
3. **A deprecation policy and a fix-it pipeline.** Bug-fix releases
   on the v1.x line for at least 12 months. CVEs handled per the
   process in `SECURITY.md` with publicly stated timelines.

Everything below is the work needed to credibly make those three
commitments. If we can't make them, we're not at v1.0 — we're at
v0.x with more polish.

---

## Where v0.1.0 stands today

This section is a snapshot, not a promise. Each row references the
crate or doc that owns the gap.

| Area | v0.1.0 state | Gap to v1.0 |
|---|---|---|
| WASM execution (Wasmtime wrapper) | Solid; all 280+ tests green on host-only | None — the wrapper is the thinnest layer and is feature-complete |
| Cold-start (snapshot/restore) | Implemented + tested; bounds-checked against zip bombs (Batch H) | Real cold-disk numbers from S22 runner; cross-version snapshot compat policy |
| Kernel dispatch | Back-pressure semaphore + future scaffold; returns immediately on non-CUDA hosts | Real `cuLaunchKernel`-backed event sync on CUDA; measured P99 |
| Auto-offload JIT | End-to-end working for matmul/vector_add/conv2d blueprints (Batch G); BLAKE3 cache | Broader blueprint set; coverage report on which patterns get offloaded |
| Kernel-args marshalling | Returns `KernelArgsUnsupported` for `args_len > 0` (documented v0.1.0 contract — see [`RISKS.md`](RISKS.md)) | Full dynamic argv via `cuLaunchKernel`; v0.2 milestone |
| Multi-tenant (TenantRegistry) | Quota gate works, MPS feature-gated | MPS production-tested; tenant-level metric isolation verified |
| HTTP API | axum gateway with bearer auth + 64 MiB body limit (Batch J); async invoke via `JobRecord`; OpenAPI committed; per-token QPS rate limiting (W1.4, closed); scoped bearer tokens (W2.1, closed); structured audit log (W2.2, closed) | mTLS / OIDC remain v2 considerations |
| CLI | Snapshot save/restore wired against API (Batch K); 22 lib tests + 19 smoke + 10 snapshots; `observe` subcommand (W1.5, closed); shell completions + man pages (W2.4, closed) | None remaining for v1.0 |
| Snapshot subsystem | Streaming zstd + bincode with hard size caps (Batch H) | Schema versioning policy; cross-version migration test matrix |
| Observability | OpenTelemetry tracing + Prometheus metrics; OTLP opt-in; HTTP request metrics middleware (W2.3, closed); Grafana dashboards (W2.5, closed); SLO + runbooks (W1.9 / W2.6, closed) | None remaining for v1.0 |
| Performance baseline | Hand-picked conservative ceilings in `bench-results/baseline.json` | Measured medians from S22 runner; tightened tolerances |
| Security | Threat model documented; fuzz harness for snapshot + WAT parser; internal code review pass (D7, closed) | External pen-test; CVE disclosure pipeline exercised |
| OSS hygiene | LICENSE / NOTICE / SPDX / CoC / CONTRIBUTING / dependabot landed (Batch A, M); `GOVERNANCE.md` landed (W1.8, closed) | Trademark; release-signing keys |
| Supply chain | `cargo-audit` + `cargo-deny` in CI (Batch M) | SBOM published per release; reproducible builds |
| Platforms | Linux x86_64 primary; Windows MSVC builds and tests; macOS compile-tested in CI (Tier 3 — no CUDA) | Tier matrix documented in [`BUILD.md`](BUILD.md#platform-support-tiers); broaden macOS coverage to tests post-v1.0 |
| Dependencies | `cust 0.3.x` (EOL'd upstream — see [`RISKS.md`](RISKS.md)); `prometheus-client 0.24` (recently bumped); `wasmtime 25.0.3` | `cust` successor chosen and migrated; Wasmtime upgrade cadence policy |

---

## Anti-goals — what v1.0 does NOT promise

Saying these out loud now prevents scope creep later. v1.0 explicitly
does **not** include:

- **WASI Preview 3 / async components.** Wasmtime's component-model
  async story is still moving. v1.0 ships WASI Preview 2 only; P3 is
  v2.x.
- **WebGPU as a guest interface.** WASI-GPU (our existing surface) is
  v1.0. WebGPU shaders compiled to PTX is a v2 research item.
- **AMD / Intel / Apple GPU backends.** v1.0 is NVIDIA CUDA only. We
  leave room in the WIT for vendor abstraction but do not implement
  it. Reasonable readers can disagree — this is a deliberate
  scope-cut.
- **Hosted / managed-service offering.** v1.0 is the self-hosted
  runtime. Any "TensorWasm Cloud" is a separate product on a separate
  timeline.
- **Cross-cloud orchestration.** Single-host runtime with HTTP API.
  Multi-host, scheduling, autoscaling — out of scope; integrate with
  existing orchestrators (k8s, Nomad).
- **GUI / web console.** CLI + HTTP API only. A console is a v2 product
  decision, not a runtime concern.
- **Wasm execution speed parity with Wasmer-LLVM on tight loops.**
  See [`BENCHMARKING.md`](BENCHMARKING.md#where-tensor-wasm-wins-where-it-wont).
- **First-class JavaScript / Python guest runtimes.** Bring your own
  Wasm; we don't ship language runtimes.
- **A Rust-stable build.** TensorWasm is pinned to `nightly-2026-04-03` for
  reasons documented in `rust-toolchain.toml`. v1.0 stays on a pinned
  nightly with a documented upgrade cadence (quarterly). Moving to
  stable is a v2 effort gated on Wasmtime dropping its own nightly
  needs.

If any of the above lands before v1.0 it's a happy accident, not a
plan. If the maintainers decide one of these IS v1.0 scope, move it
out of this section in a separate PR with the rationale.

---

## The five-milestone path

Five releases between today and v1.0. Each is independently
shippable, each has hard exit criteria, and each unblocks the next.

### v0.2.0 — "Real CUDA"

**Theme.** The CUDA path moves from feature-gated stub to first-class
supported configuration. Anything labeled "modeled" or "v0.1.0
contract" in the v0.1.0 docs becomes "measured" or "implemented".

**Exit criteria.**

- [ ] **S22 self-hosted CUDA runner online** in CI. Workflow runs the
      `cuda` + `unified-memory` + `mps` + `auto-offload` feature matrix
      on every PR that touches the relevant crates.
- [ ] **Kernel-args marshalling implemented.** `KernelArgsUnsupported`
      is removed (or relegated to a fallback for malformed args only).
      Direct `cuLaunchKernel` path with typed argv lowering. Two new
      end-to-end tests: scalar args, pointer args.
- [ ] **`dispatch/serial` and `dispatch/concurrent_cap64` measured on
      real GPU.** Bench results in `bench-results/baseline.json` replace
      the modeled numbers. Tolerances tightened to ±10% from the
      current 50%.
- [ ] **`cold_start/restore` measured with real UVM page-migration
      cost.** Numbers in `PERFORMANCE.md` move from "modeled" to
      "measured (H100 PCIe gen5)" or equivalent SKU disclosure.
- [ ] **MPS path validated end-to-end.** A test that spins up 4
      tenants under MPS, runs the same workload, asserts isolation
      (one tenant's OOM does not kill another's launch).
- [ ] **`docs/CUDA-SETUP.md` rewrite** with the exact toolkit
      versions and driver versions the runner uses. Removes any
      "this is what you'd do if..." hedging.

**Out of scope for v0.2.** Anything in v0.3+ below. Don't expand
scope; the CUDA story alone is large.

### v0.3.0 — "Production observability"

**Theme.** A team running TensorWasm in production can see what's happening
and respond to incidents without reading source code.

**Exit criteria.**

- [ ] **Reference Grafana dashboard committed** under
      `docs/dashboards/tensor-wasm-overview.json`, importable as-is, covering:
      request rate, error rate, P50/P95/P99 latency per endpoint,
      tenant-level GPU memory consumption, snapshot capture/restore
      durations, JIT cache hit ratio, back-pressure permit utilization.
- [ ] **SLOs published** in `docs/SLO.md`: numeric availability,
      latency, and error-rate targets for the HTTP API and the
      dispatch path, with the burn-rate alerts that go with them.
- [ ] **Runbook for every alert** in `docs/runbooks/`. Each alert in
      the dashboard has a one-page runbook with: what it means, what
      to check, how to mitigate, when to page.
- [ ] **Distributed tracing end-to-end.** Trace ID flows from HTTP
      request → tenant lookup → snapshot restore → dispatch → response,
      visible in a single OTLP backend.
- [ ] **`tensor-wasm-cli observe` subcommand** that wraps `curl` against
      `/metrics` and `/healthz` and prints a one-screen status board
      for operators.

**Decision before exit.** Default metric backend — Prometheus
scrape, OTLP push, or both. Pick one, document the other as
supported-but-not-default.

### v0.4.0 — "API hardening"

**Theme.** The HTTP API and CLI are durable enough to support real
multi-tenant deployments and an outside security review.

**Exit criteria.**

- [ ] **Rate limiting per token.** Configurable QPS + burst per
      bearer token, enforced at the router layer. Tested under
      concurrent load.
- [ ] **mTLS support optional but documented.** A
      `docs/deployment/mtls.md` showing how to terminate TLS at the
      TensorWasm process, with the same auth model as bearer.
- [ ] **Scoped tokens.** Tokens grant per-tenant scopes, not just
      "all access". Backwards-compatible default (existing tokens get
      `tenant: *`) with deprecation warning.
- [ ] **Audit log.** Every state-mutating API call writes a structured
      audit record (who, when, what, request-id). Documented schema.
- [ ] **CLI shell completions** for bash/zsh/fish under
      `crates/tensor-wasm-cli/completions/`, installable via `tensor-wasm completions
      generate <shell>`.
- [ ] **Man pages** for every `tensor-wasm` subcommand. Generated from
      clap definitions, committed under `crates/tensor-wasm-cli/man/`.
- [ ] **OpenAPI spec validated** against the live router in CI (a
      generated client compiles + round-trips a synthetic request).

### v0.5.0-beta — "External validation"

**Theme.** The work is in a state where an outside party can audit
it, deploy it, and report back. No new feature work — just bug fixes
from beta feedback.

**Exit criteria.**

- [ ] **External security review commissioned and the high-severity
      findings closed.** Choice of auditor is a separate decision (see
      [Open decisions](#open-decisions-to-resolve-before-v10)).
      Findings published in `docs/SECURITY-AUDIT-v0.5.md` with
      `accepted / mitigated / rejected` per finding and rationale.
- [ ] **At least one external production deployment** willing to be
      named in v1.0 release notes (or two anonymized ones). The
      deployment runs TensorWasm for a full month with no severity-1
      incidents.
- [ ] **Fuzz corpus accumulates 24+ hours of clean run per target.**
      All targets: snapshot reader, WAT parser, WASI-GPU host fn
      argument lowering, JIT IR builder.
- [ ] **Cross-version snapshot compatibility tested.** Snapshots
      from v0.2, v0.3, v0.4 all restore cleanly under v0.5. Documented
      migration policy ("v1.0 will read all v0.5+ snapshots") goes
      into `docs/SNAPSHOT-COMPATIBILITY.md`.
- [ ] **Beta release notes** explicitly state what is frozen for v1.0
      and what may still change. After 0.5.0-beta, the only changes
      between betas are bug fixes and doc improvements.

### v1.0.0-rc1 → v1.0.0

**Theme.** API freeze, paperwork, release engineering. No new code
unless a beta-cycle bug demands it.

**Exit criteria.**

- [ ] **Two clean weeks** on `main` with no severity-1 bugs filed
      against the latest RC.
- [ ] **Release signing keys generated and published.** Cargo
      registry release signed; container images signed (cosign or
      equivalent); SBOM (CycloneDX) attached to every release artifact.
- [ ] **Reproducible builds documented.** A reader can rebuild a
      TensorWasm v1.0 artifact from source and get bit-identical output
      (modulo timestamps).
- [ ] **`docs/CHANGELOG.md` v1.0 entry** lists every public API
      change from v0.5.0 with the rationale.
- [ ] **`docs/MIGRATION-v0-to-v1.md`** for users on the v0.x line.
      Includes deprecation table, removed-API table, behavioral-change
      table.
- [ ] **`docs/UPGRADE.md`** with the operational steps to roll a
      TensorWasm fleet from v0.5 to v1.0.
- [ ] **Trademark policy** in `docs/TRADEMARK.md` (if applicable —
      see [Open decisions](#open-decisions-to-resolve-before-v10)).
- [ ] **Maintainer governance** documented in `GOVERNANCE.md`:
      decision process, RFC procedure, security-disclosure committee,
      maintainer onboarding/offboarding.
- [ ] **Backport policy.** v1.x will receive security patches and
      severity-1 fixes for at least 12 months. Documented in
      `SECURITY.md`.

---

## Post-v0.3.6 strategic features

Features that landed as v0.3.x _scaffolds_ — surface-area-stable, but
with the production wire (server endpoints, on-disk stores, control
plane) deferred to v0.4 so a design partner can target them ahead of
the parity port. Each item lives behind a feature flag in its owning
crate so the default build doesn't pay the dep cost until the wire
lands.

1. **OpenAI gateway shim** (`tensor-wasm-api`, v0.3.6). Wired into the
   router behind a route allowlist with an OpenAPI spec. See B5.6.
2. **Unified backing for tensor buffers** (`tensor-wasm-mem`, v0.3.5).
   `UnifiedBacking` trait + `UvmAdvice` impls for the three buffer
   shapes. See B5.4 and `docs/CUDA-OXIDE-CUTOVER.md`.
3. **Signed kernel registry** (`tensor-wasm-jit`, v0.3.7). HMAC-SHA256
   `KernelManifest` records + in-memory `InMemoryRegistry`. CLI
   surface (`tensor-wasm kernel publish|list|verify`) is staged but
   exits with code 3 (`FEATURE_NOT_EXPOSED`) until v0.4 wires the
   server-side `/kernels` route. See
   [KERNEL-REGISTRY.md](KERNEL-REGISTRY.md).

These items are deliberately NOT exit criteria for v0.4; they exist
to let design partners build against a stable Rust + CLI surface
ahead of the v0.4 wire. The "graduate from scaffold to wired" step
moves to the relevant v0.4 milestone above.

---

## Per-area workstreams

Cross-cuts the milestones above. These can be parallelized; each
contributor can pick a stream.

### Security

- External pen-test of the HTTP API (v0.5 gate)
- External audit of WASI-GPU bounds-check correctness (v0.5 gate)
- Fuzz corpus growth: keep `fuzz/` targets running 24×7 on dedicated
  hardware once available (v0.3 onwards)
- CVE disclosure pipeline exercised at least once (intentional
  rehearsal, not a real CVE) before v0.5
- Supply-chain attestation (SLSA level 3 target for v1.0)

### Performance

- Replace every "modeled" number in `PERFORMANCE.md` with measured
  (v0.2 gate)
- Tighten `baseline.json` tolerances from 30-100% to 10-30% (v0.2)
- Publish at least three external comparisons per
  `BENCHMARKING.md` methodology before v0.5
- Long-tail latency analysis: P99.9 measured for `dispatch/*` and
  `e2e/*` (v0.3 gate)
- Capacity-planning doc: tenants-per-host curves at fixed SLA (v0.4)

### API and ABI

- Wasmtime upgrade cadence policy (quarterly minor bumps, major bumps
  case-by-case)
- `cust` successor chosen and migrated (see Open decisions)
- WIT interface frozen at v0.5; any changes after that are v2
- HTTP API surface frozen at v0.5; deprecations land in v0.4 with
  warnings

### Operations

- Reference deployment manifests: docker-compose (have), k8s
  (W2.7, closed), Nomad (v0.4, stretch)
- Helm chart for k8s (W2.7, closed)
- Backup / restore procedure documented and tested (W3.7, closed)
- Upgrade playbook (W3.3, closed)
- Disaster-recovery runbook: lost host, lost storage, lost auth
  state (v0.4)

### Documentation

- "Production deployment" tutorial end-to-end (v0.3)
- "Migrating from Wasmtime/Wasmer to TensorWasm" guide (v0.4)
- "Writing CUDA kernels for TensorWasm" guide (v0.3, once kernel-args
  marshalling lands)
- API reference auto-generated from rustdoc + OpenAPI, published per
  release (v0.4)

### Governance

- `GOVERNANCE.md` (v0.5)
- `MAINTAINERS.md` reviewed and trimmed/expanded (already exists from
  Batch A; revisit at v0.5)
- RFC process (lightweight — a `rfcs/` directory and a template)
  established at v0.3, used in anger by v0.5
- Contributor License Agreement decision: required, optional, or
  none. Default proposal: none, rely on inbound=outbound Apache-2.0
  per the existing DCO model.

---

## Open decisions to resolve before v1.0

Each of these is a Y-fork that blocks at least one milestone exit
criterion. Assign owners and resolve before the milestone they gate.

### 1. `cust` successor (gates v0.5 — re-scoped 2026-05-25)

**Re-scope note (2026-05-25):** the v0.1.0-era framing of this decision said "gates v0.2". [RFC 0001](../rfcs/0001-cuda-oxide-integration.md) re-scoped it to v0.5: the W1.2 cudarc spike + the O1-O6 cuda-oxide scaffolding wave + the F2 Pliron pin together let all three candidate backends ship side-by-side from v0.3.1, with the default-flip held to v0.5 pending cuda-oxide v0.2 stability. The decision below is the original options-list; the binding recommendation is Option C in RFC 0001 (three backends side-by-side, cuda-oxide default at v0.5 contingent on v0.2.0 shipping, cudarc fallback if it doesn't).

`cust 0.3.x` is EOL upstream. Options:

- **`cudarc`** — actively maintained, similar API surface, ~80% drop-in.
- **Bespoke FFI** — write our own thin wrapper over the CUDA Driver API.
  Maximum control, maximum maintenance burden.
- **`rust-cuda` fork** — community pickup if one materializes; high risk.

Proposed: `cudarc`. Migration is a v0.2 PR. Spike first to confirm
WASI-GPU host-fn surface still maps cleanly.

_Update (2026-05-25): see [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md) — `cuda-oxide` added as a third option; default-pick contingent on its v0.2 release._

### 2. Default auth model (gates v0.4)

Today: bearer tokens via `TENSOR_WASM_API_TOKENS`. v1.0 options:

- **Bearer + scoped tokens** (current path, refined). Simple, familiar.
- **mTLS-first** with bearer as fallback. Enterprise-friendly but
  more deployment overhead.
- **OAuth/OIDC integration.** Heaviest but most flexible.

Proposed: bearer + scoped tokens as default, mTLS as supported alt,
OIDC deferred to v2.

### 3. Metric backend default (gates v0.3)

Pull (Prometheus scrape) vs push (OTLP) as the documented default.
Both supported either way; the question is which the quickstart docs
show first. Proposed: Prometheus scrape — easier for self-hosted, more
common in CNCF ecosystem.

### 4. Trademark policy (gates v1.0)

Is "TensorWasm" a registered trademark of Craton Software Company? If yes,
publish a usage policy. If no, document that explicitly. The choice
affects how the community can fork and rename. Proposed: leave
unregistered; permissive trademark, document policy in
`docs/TRADEMARK.md`.

### 5. External auditor for v0.5 review

Candidates: Trail of Bits, NCC Group, Cure53, Doyensec. Quote-gather
and pick by v0.4 so the audit can run during the v0.5-beta cycle.

### 6. Production design partners

Need at least one (preferably two or three) external organization
willing to deploy a v0.5 beta in production for a month and report
back. Recruit during v0.3/v0.4.

### 7. Backport window length

12 months proposed. Some users will want LTS-style 24. Decide at v0.5
based on design-partner feedback.

### 8. Rust toolchain pin policy

Quarterly nightly bumps proposed, aligned with Wasmtime releases.
Decision: how do we communicate breaking nightly changes to users?
Proposed: every nightly bump is a minor-version bump for v0.x; for
v1.x, nightly bumps that don't break user code are patch releases.

---

## Post-v0.3.6 strategic features

These are scoped items the v0.3.6 → v0.5 window opens up. None of them
block v0.5-beta exit on their own, but each closes a specific
"audit-bait" objection the external auditor is likely to raise. Items
are listed in expected scaffold-land order; the cross-link points at
the in-tree spec doc where the scaffold landed first.

1. **Kernel ABI freeze + versioning policy** — stable `.ptxbin`
   container with explicit ABI version byte; covered by the cache
   integrity tests today, formalised post-v0.3.6.
2. **Snapshot replay-protection cross-version matrix** — v0.3.6
   landed the per-snapshot nonce + tenant-scoped epoch fields. v0.4
   adds the end-to-end matrix that exercises the policy across
   N-1 / N / N+1 minor versions.
3. **Tenant-quota fairness model formalisation** — the back-pressure
   semaphore is the current implementation; the formalised model
   (proportional-share or weighted fair queueing) lands as an RFC.
4. **MPS production-readiness checklist** — the feature flag exists;
   the checklist that says "ship MPS as default at v1.0 or stay
   behind the flag" lands here.
5. **WASI-GPU surface lock** — freeze the host-fn signatures so
   third-party guests can ship against v0.5-beta without breakage.
6. **Differential JIT correctness oracle** — runs every
   `auto_offload` candidate on both the Wasmtime CPU interpreter and
   the JIT PTX path and asserts bit-identity. Lowest-cost,
   highest-credibility security item on the v0.5 pre-audit
   checklist. Scaffold lives in
   [`crates/tensor-wasm-jit/src/differential.rs`](../crates/tensor-wasm-jit/src/differential.rs);
   spec doc at [`docs/DIFFERENTIAL-ORACLE.md`](DIFFERENTIAL-ORACLE.md).
7. **Reproducible-build attestation** — SLSA Level 3 ambition;
   pre-staged by [`docs/REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md)
   and the W4.3 SBOM workflow.

Items 1-5 and 7 are tracked elsewhere in this document or in their own
spec docs; item 6 is new in v0.3.6 and is cross-linked from
[`docs/DIFFERENTIAL-ORACLE.md`](DIFFERENTIAL-ORACLE.md).

---

## Risk register

Risks that could push v1.0 right or force a milestone re-cut.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| S22 self-hosted CUDA runner delayed or unfunded | Medium | High — blocks v0.2 exit | Identify cloud GPU-host alternative (Lambda Labs, RunPod) as fallback; document cost; budget |
| `cudarc` (or `cuda-oxide`) migration uncovers semantic gaps | Medium | Medium — slips v0.5 default-flip by 4-8 weeks | W1.2 cudarc spike + O2 cuda-oxide-backend scaffold both already shipped (see RFC 0001 Option C). All three backends coexist; the risk is the cutover, not the spike. Plan B: hold the default at `cudarc-backend` if cuda-oxide v0.2 slips. |
| External pen-test surfaces critical findings | High | Medium — slips v0.5 by 2-6 weeks | Budget time; plan for ≥1 round of significant remediation |
| Wasmtime upstream breaking change between bumps | Medium | Low-Medium — costs a sprint per occurrence | Pin via Cargo.lock; only bump on documented stable releases; subscribe to wasmtime release notes |
| No design partners willing to run a beta | Low-Medium | High — v1.0 launches without real-world validation | Start outreach at v0.3; offer integration help; allow anonymous deployment in release notes |
| MPS production-readiness gap larger than expected | Medium | Medium — drops MPS from v1.0 default | Acceptable fallback: v1.0 ships MPS as supported-but-not-default, feature-gate stays |
| WASI Preview 2 / 3 churn invalidates current WIT | Medium | Low — well-bounded | Stay on Preview 2 for v1.0; treat any P3 work as v2; document the freeze |
| Auto-offload coverage doesn't grow beyond v0.1 blueprints | Medium | Low | Acceptable — auto-offload stays opt-in feature flag for v1.0; explicit-dispatch remains the primary surface |
| Trademark conflict discovered late | Low | High — forces rename | Search before v0.4; resolve before v0.5 freeze |
| Disk-space / build-time issues block contributor onboarding | Low | Low | Document `target/` cleanup; CI uses sccache; consider workspace split if it grows |

---

## Effort and timeline (caveated)

These are calendar-time estimates assuming a small core team (2-4
maintainers) plus opportunistic contributors. They're informed
guesses, not commitments — every estimate is wrong, but having a
strawman is more useful than not.

| Milestone | Calendar estimate | Contingent on |
|---|---|---|
| v0.2.0 ("Real CUDA") | 3-4 months | S22 runner online; `cust` successor chosen |
| v0.3.0 ("Production observability") | +2-3 months | Dashboard work; runbook authoring |
| v0.4.0 ("API hardening") | +2-3 months | Auth model decision; rate-limit design |
| v0.5.0-beta ("External validation") | +3-4 months | Auditor scheduled; design partners recruited |
| v1.0.0-rc1 → v1.0.0 | +2 months | No new severity-1 bugs; paperwork |
| **Total** | **12-16 months** | All of the above |

Multiply by 1.5× if the team is part-time, by 0.75× if the team
expands to 6+ full-time maintainers. Subtract 2-4 months if
external sponsorship covers the auditor and the self-hosted runner.

Don't quote these dates externally. Quote milestones instead:
"v0.2 lands when the exit criteria above pass." The criteria are
the commitment; the date is a guess.

---

## Post-v0.3.6 strategic features

A small list of strategic features lined up between v0.3.6 and v0.4.0.
Each is scoped, has a single owner module, and ships behind an opt-in
config knob so existing deployments are unaffected until they elect
to turn it on. Order is by expected impact on cold-start / hot-path
latency; numbering is stable for cross-linking.

1. **OpenAI-shim compatibility layer** — drop-in `/v1/completions`
   surface that forwards into a TensorWasm tenant. Scaffold landed
   in `tensor-wasm-api`; full streaming in v0.4.
2. **Snapshot replay-protection** — monotonic snapshot epoch + nonce,
   verified on restore. Scaffold landed in `tensor-wasm-snapshot`.
3. **Tenant-aware WASI-GPU back-pressure** — per-tenant queue depth
   limits surfacing as `QuotaExceeded`. Scaffold in
   `tensor-wasm-wasi-gpu`.
4. **Configurable per-instance linear-memory cap** — `max_linear_memory`
   in `EngineConfig`. Landed in `tensor-wasm-mem`.
5. **Pre-instantiated instance pool** — pre-spawn N instances per
   `(tenant, module-hash)` tuple under MPS; draw from a
   `crossbeam-channel` on `invoke`. Scaffold lives in
   `tensor-wasm-exec::instance_pool`; see
   [docs/INSTANCE-POOL.md](INSTANCE-POOL.md) for the v0.4
   implementation plan and the reset-on-return contract.

## Out of scope — deferred to v2.0

For visibility, the v2.x line is likely to include:

- WASI Preview 3 / async components
- AMD ROCm / Intel oneAPI / Apple Metal backends (vendor abstraction
  layer in WIT + at least two backends shipped)
- WebGPU shader → PTX path for browser-compatible kernels
- Hosted control plane (separate product)
- Web console / GUI
- First-class JavaScript guest (via a bundled QuickJS or similar)
- Rust-stable build target
- Multi-host scheduling primitives (or a documented k8s operator)

None of this blocks v1.0. Mentioning it here so it's clear we know it
exists and have a place to put it.

---

## Post-v0.3.6 strategic features

The items below come out of the comprehensive review conducted at the
v0.3.6 mark. They are *additions* to the workstreams above, not
replacements: the existing milestone exit criteria still gate v0.5 /
v1.0. These are the strategic bets that turn TensorWasm from "Wasmtime
plus a CUDA crate" into a credible GPU-Wasm platform.

Items are grouped into three tiers by horizon and confidence: high-
leverage near-term (v0.4), strategic medium-term (v0.5–v1.0), and
speculative / R&D. Cost estimates are engineer-weeks of focused work
by a single contributor familiar with the affected crates; they are
not calendar time.

### High-leverage near-term (v0.4)

#### 1. Typed multi-value guest export ABI

- **What:** Today the executor only invokes `() -> ()`. Wire `--args`
  JSON through to the guest so typed multi-value exports work end to
  end. Touches `tensor-wasm-exec`, the CLI, and the HTTP API.
- **Why:** Unlocks every non-trivial guest. Without typed args, every
  real workload has to smuggle inputs through preopens or env vars.
- **Cost:** ~2 weeks.
- **Risk:** Low. The Wasmtime side already supports it; the work is
  plumbing and a JSON ↔ `Val` codec with clear failure modes.

#### 2. Streaming HTTP `invoke` responses

- **What:** SSE or chunked transfer encoding for `invoke`, so guests
  can emit token-by-token output. Adds a new host function
  `wasi:tensor/host.emit-chunk` that the guest can call repeatedly.
- **Why:** Closes the LLM use case. Without streaming, every chat-
  style workload has to buffer to completion before the client sees
  anything, which is a non-starter against Modal / Beam / vLLM.
- **Cost:** ~3 weeks.
- **Risk:** Low–medium. Backpressure, cancellation, and per-tenant
  fair scheduling on the streaming path each need design notes.

#### 3. Signed kernel registry

- **What:** `tensor-wasm kernel publish --sign` mirroring the existing
  snapshot-signing pattern. Operators consume vetted kernels (matmul,
  attention, conv2d) as first-class signed artifacts instead of
  rebuilding from source.
- **Why:** Lets the kernel library evolve independently of the
  runtime, and gives operators a defensible supply-chain story for
  the GPU code path.
- **Cost:** ~3 weeks.
- **Risk:** Low. The signing primitives, HMAC trailer format, and
  on-disk layout already exist from the snapshot and JIT L2 work.

#### 4. Cooperative deadlines via WASI yield

- **What:** Well-behaved guests offer suspend points via a new WASI
  yield host function; the scheduler uses these to keep tail latency
  bounded under MPS contention. Landed as the
  `wasi:scheduler/host@0.1.0` interface — see
  [COOPERATIVE-YIELD.md](COOPERATIVE-YIELD.md) for the protocol guide,
  return-code semantics, and the embedder wiring snippet.
- **Why:** Today a long-running guest under MPS contention blocks
  other tenants until preemption. Cooperative yields close the gap
  without paying full preemption cost.
- **Cost:** ~1 week.
- **Risk:** Low. The fallback (uncooperative guests) is the status
  quo, so this is a strict improvement when adopted.

#### 5. Pre-instantiated instance pool

- **What:** Pre-spawn N instances per (tenant, module) tuple and draw
  from a channel on `invoke` instead of paying cold-start on every
  call.
- **Why:** Pushes P99 latency down materially. The current numbers
  are where Modal and Beam currently win the head-to-head benchmarks
  — this directly attacks that gap.
- **Cost:** ~2 weeks.
- **Risk:** Medium. Pool sizing, eviction, and pinned-resource
  accounting interact with the GPU memory quota work (#8).

### Strategic medium-term (v0.5–v1.0)

#### 6. Differential JIT correctness oracle

- **What:** Every `auto_offload` candidate runs on both the Wasmtime
  CPU path and the JIT GPU path under proptest; bit-identity is
  asserted across a generated input distribution.
- **Why:** Highest-credibility security-pitch item before the v0.5
  external audit. "Our JIT is bit-identical to the interpreter under
  random inputs" is much stronger than any test-suite claim alone.
- **Cost:** ~3 weeks.
- **Risk:** Medium. Floating-point determinism across CPU and GPU
  paths requires care; some kernels will need an explicit tolerance
  policy with a documented rationale.

#### 7. Pliron-based auto-offload pipeline

- **What:** A real compiler pipeline — Wasm → CLIF → Pliron
  `dialect-mir` → cuda-oxide → PTX — replacing the three hand-written
  offload blueprints.
- **Why:** **THE feature that distinguishes "Wasmtime + a CUDA crate"
  from "the way you run GPU Wasm".** Expands offload coverage from
  three named kernels to anything the pipeline can lower.
- **Cost:** 2–3 months.
- **Risk:** High. Pliron is still maturing; lowering quality on real
  Wasm workloads is unproven. Worth the bet because the alternative
  is shipping a permanent allow-list of kernels.

#### 8. Per-tenant GPU memory quotas via `cuMemPool`

- **What:** Hard per-tenant GPU memory caps enforced inside MPS using
  `cuMemPool` (CUDA 11.2+). Replaces today's soft accounting.
- **Why:** Makes the multi-tenant pitch defensible. Without hard
  quotas, any tenant can OOM the whole device and the isolation
  story falls apart on first contact.
- **Cost:** ~4 weeks.
- **Risk:** Medium. Older driver / minimum-CUDA requirements need to
  be enforced and documented in the support matrix.

#### 9. Unified content-addressed signed artifact store

- **What:** Fold the JIT L2 cache and the snapshot store into a
  single content-addressed, signed artifact primitive. Both already
  share the HMAC trailer format and on-disk layout.
- **Why:** One fewer concept in operator docs; one fewer code path
  to audit; consistent garbage collection and quota story.
- **Cost:** ~3 weeks.
- **Risk:** Low. The two stores already converged in format; this is
  collapsing the abstraction, not reinventing it.
- **Status:** v0.3.7 scaffold landed — trait + in-memory impl +
  fully-implemented disk impl in
  [`crates/tensor-wasm-artifacts`](../crates/tensor-wasm-artifacts).
  Design and v0.4 convergence plan in
  [`docs/ARTIFACT-STORE.md`](ARTIFACT-STORE.md). JIT cache and
  snapshot store still use their own formats today; migration of
  each consumer onto the unified envelope is the v0.4 follow-up.

#### 10. OpenAI-compatible inference gateway shim

- **What:** A thin gateway exposing `/v1/completions` and
  `/v1/chat/completions` that translates to the internal `invoke`
  protocol.
- **Why:** **Highest-ROI item on this list.** The addressable market
  of "things that speak the OpenAI API" is orders of magnitude
  bigger than "Wasmtime / Wasmer migrators". Cheapest possible way
  to put TensorWasm into a real LLM serving stack.
- **Cost:** ~2 weeks.
- **Risk:** Low. The spec is stable, the translation is mechanical,
  and #2 (streaming responses) is the hard prerequisite — once that
  ships, this is almost free.

### Speculative / R&D

#### 11. WASI-NN compatibility layer

- **What:** A compatibility shim that lets existing WASI-NN guests
  (compiled for ONNX, llama.cpp, OpenVINO) execute on TensorWasm
  with a CUDA-accelerated backend.
- **Why:** Inherits an existing guest ecosystem instead of asking
  authors to port to a TensorWasm-specific WIT.
- **Cost:** 6 weeks.
- **Risk:** High. The WASI-NN spec is still moving; building against
  a moving target risks landing a layer that ages out before the
  audience materializes.

#### 12. Direct guest-side GPU dispatch via SPIR-V

- **What:** A SPIR-V → PTX path that lets guests dispatch GPU work
  directly, rather than going through host kernels.
- **Why:** Speculative. WebGPU-as-guest-interface is explicitly
  anti-goal'd in this doc — but worth keeping a WIT door open for in
  case the calculus shifts.
- **Cost:** 6 months or more.
- **Risk:** Very high. Conflicts with the current anti-goal; security
  model for guest-issued PTX is open; SPIR-V → PTX lowering is its
  own multi-engineer project.

#### 13. Distributed dispatch sidecar over QUIC

- **What:** A single-hop sidecar that fronts a TensorWasm host and
  transparently bursts GPU work to peer hosts over QUIC when local
  capacity is exhausted.
- **Why:** Multi-host scheduling without committing to a full control
  plane. v1.x territory, not v1.0.
- **Cost:** 2–3 months.
- **Risk:** Medium–high. Failure modes, tenancy boundaries across
  hosts, and operator UX all need design work before any code.

---

**Top priority for the v0.5-beta external-deploy gate: #10
(OpenAI-compatible inference gateway shim).** Of the items on this
list, it has the highest ratio of addressable market to engineering
cost, depends only on #2 (which is already on the v0.4 critical
path), and converts the runtime's existing strengths — streaming,
multi-tenancy, signed artifacts — into a deliverable that an LLM
serving team can adopt without rewriting their client code. The
medium-term strategic items (#6, #7, #8) are what make the platform
*credible* once adopted; #10 is what gets it adopted in the first
place.

---

## How to give feedback on this proposal

- Open an RFC PR against this file proposing scope changes
  (add/remove milestones, change exit criteria, move items between
  versions).
- Open an issue per open decision in
  [Open decisions](#open-decisions-to-resolve-before-v10) with
  arguments for one branch of the fork.
- Bring the milestone shape to a maintainer sync before any
  large-scope changes land; this doc should reflect maintainer
  consensus, not one author's view.

---

## Related docs

- [README.md](../README.md) — status statement, current feature matrix
- [ARCHITECTURE.md](../ARCHITECTURE.md) — crate dependency graph
  (constraints on what can move where)
- [PERFORMANCE.md](PERFORMANCE.md) — what's measured today; what
  becomes measured in v0.2
- [BENCHMARKING.md](BENCHMARKING.md) — how external comparisons are
  expected to be conducted before v0.5
- [RISKS.md](RISKS.md) — v0.1.0 known limitations and tracked
  upstream issues
- [SECURITY.md](../SECURITY.md) — disclosure process (matures into
  the v1.0 CVE pipeline)
- [MAINTAINERS.md](../MAINTAINERS.md) — current maintainer list
  (expands into GOVERNANCE.md at v0.5)

---

_Status: proposal, v0.1.0 era. This document is itself v0.x — expect
it to change shape before v0.2 ships. Treat the milestone exit
criteria as the contract; the calendar dates as guesses; the open
decisions as the actual blockers._
