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
| HTTP API | axum gateway with bearer auth + 64 MiB body limit (Batch J); async invoke via `JobRecord`; OpenAPI committed | Auth model that's actually useful (mTLS? OAuth? scoped tokens?); rate limits |
| CLI | Snapshot save/restore wired against API (Batch K); 22 lib tests + 19 smoke + 10 snapshots | Production telemetry; shell completions; man pages |
| Snapshot subsystem | Streaming zstd + bincode with hard size caps (Batch H) | Schema versioning policy; cross-version migration test matrix |
| Observability | OpenTelemetry tracing + Prometheus metrics; OTLP opt-in | Reference dashboards (Grafana JSON); SLO definitions |
| Performance baseline | Hand-picked conservative ceilings in `bench-results/baseline.json` | Measured medians from S22 runner; tightened tolerances |
| Security | Threat model documented; fuzz harness for snapshot + WAT parser | External pen-test; CVE disclosure pipeline exercised |
| OSS hygiene | LICENSE / NOTICE / SPDX / CoC / CONTRIBUTING / dependabot landed (Batch A, M) | Maintainer governance model; trademark; release-signing keys |
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
- **A Rust-stable build.** TensorWasm is pinned to `nightly-2026-03-15` for
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
  (v0.3), Nomad (v0.4, stretch)
- Helm chart for k8s (v0.4)
- Backup / restore procedure documented and tested (v0.3)
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

### 1. `cust` successor (gates v0.2)

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

## Risk register

Risks that could push v1.0 right or force a milestone re-cut.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| S22 self-hosted CUDA runner delayed or unfunded | Medium | High — blocks v0.2 exit | Identify cloud GPU-host alternative (Lambda Labs, RunPod) as fallback; document cost; budget |
| `cudarc` migration uncovers semantic gaps | Medium | Medium — slips v0.2 by 4-8 weeks | Spike before committing; have a "stay on `cust` longer + vendor it" Plan B |
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
