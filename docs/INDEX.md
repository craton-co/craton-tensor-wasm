<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Documentation Index

The single-page sitemap for every Markdown document shipped with
Craton TensorWasm. The grouping mirrors how a reader actually
navigates the project: pick the section that matches your role,
follow the link, and the linked doc is the contract.

The wave tag in parentheses (W1.1, W2.3, etc.) records which v0.2–v0.4
hardening wave landed the document; docs without a tag predate the
wave program. The link path is relative to **this file** (i.e. relative
to `docs/`), so `../GOVERNANCE.md` points at the repository root.

If a doc is reachable from [`README.md`](../README.md) as well, it is
listed in the table at the bottom of that file. The
[Missing cross-links](#missing-cross-links) section at the foot of this
page enumerates the docs that are only reachable via this index — a
future PR may add anchors for them in `README.md`.

## What this index is and is not

This index **is** the canonical inventory of in-repository Markdown
documentation: every doc that ships in the source tree appears in
exactly one section below. The summary text is the single-sentence
abstract a reader needs to decide whether to open the doc.

This index **is not** a tutorial, a learning path, or a status page.
- For a learning path see [GETTING-STARTED.md](GETTING-STARTED.md)
  followed by the [Audience routing](#audience-routing) table below.
- For the v1.0 roadmap and status see
  [PATH-TO-V1.md](PATH-TO-V1.md) and the `[Unreleased]` section of
  [`../CHANGELOG.md`](../CHANGELOG.md).
- For the published rustdoc + OpenAPI archive (rendered, hosted, per
  release) see [API-REFERENCE.md](API-REFERENCE.md).

The index is also not the right surface to host long-form content.
Anything that needs more than the one-line summary below belongs in
the linked doc itself; if a section starts growing past the
"one row per doc" rule the section is wrong, not the rule.

## Contents

1. [What this index is and is not](#what-this-index-is-and-is-not)
2. [Conventions](#conventions)
3. [Getting started](#getting-started)
4. [Architecture and internals](#architecture-and-internals)
5. [API surface](#api-surface)
6. [Performance and benchmarking](#performance-and-benchmarking)
7. [CUDA](#cuda)
8. [Operations](#operations)
9. [Security](#security)
10. [Governance and supply chain](#governance-and-supply-chain)
11. [Snapshots](#snapshots)
12. [Missing cross-links](#missing-cross-links)
13. [Audience routing](#audience-routing)
14. [How to extend this index](#how-to-extend-this-index)

## Conventions

- **Link paths** are relative to this file (`docs/INDEX.md`). A leading
  `../` therefore steps up into the repository root; a bare filename
  resolves inside `docs/` itself.
- **Wave tags** trace each doc back to a workstream entry in
  [`PATH-TO-V1.md`](PATH-TO-V1.md): `W<wave>.<task>` matches the row
  in the per-area workstream tables. Docs that predate the wave program
  are marked with an em dash.
- **Summaries** are deliberately single-sentence. If a doc warrants
  more context the doc itself should open with that paragraph so this
  index can quote a one-liner from it.
- **No emoji, no badges.** This file is a flat sitemap, optimised for
  grep, not for visual scanning.

---

## Getting started

The narrow on-ramp for a new contributor or operator: from a clean
checkout to a function running against a deployed gateway. Start with
`GETTING-STARTED.md`, then branch into the role-specific guide
(`CLI.md` for the developer, the production-deployment tutorial for
the operator).

| Doc | Wave | One-sentence summary |
|---|---|---|
| [GETTING-STARTED.md](GETTING-STARTED.md) | — | Fifteen-minute onboarding tutorial that walks a Rust developer from a clean checkout to invoking a deployed Wasm function. |
| [CLI.md](CLI.md) | — | Complete reference for the `tensor-wasm` developer CLI, its subcommands, global flags, exit codes, and JSON argument conventions. |
| [tutorials/production-deployment.md](tutorials/production-deployment.md) | W3.8 | End-to-end tutorial that takes a competent SRE from a fresh Kubernetes cluster to a production-ready TensorWasm deployment with mTLS, Prometheus, Grafana, audit log, and a deployed function. |
| [MIGRATING-FROM-WASMTIME-WASMER.md](MIGRATING-FROM-WASMTIME-WASMER.md) | W3.9 | Honest evaluation guide for teams already running upstream Wasmtime, Wasmer, or a Spin/Wasmer-Edge FaaS deciding whether to move workloads onto TensorWasm. |
| [WASM-DEVELOPER-GUIDE.md](WASM-DEVELOPER-GUIDE.md) | — | Walkthrough for writing Wasm guests against TensorWasm, from a trivial `add(a, b)` through the `wasi:cuda` host imports and the auto-offload fast path. |
| [BUILD.md](BUILD.md) | — | Build matrix for the three supported configurations (no-CUDA, CUDA host, CUDA stub) plus the canonical feature-flag taxonomy. |

## Architecture and internals

Background for contributors changing the runtime itself: how the
crates fit together, the upstream-pinning decisions, the JIT pipeline
shape, and the cold-start latency model that the snapshot subsystem
exists to fight. The root `ARCHITECTURE.md` is the entry point;
everything else in this section is a deeper cut into one subsystem.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | W5.2 (refresh) | The ten-crate dependency graph, the layered execution model, and the trust boundaries between Wasm guest, host process, and CUDA driver. |
| [WASMTIME-FORK.md](WASMTIME-FORK.md) | — | The decision record that explains why TensorWasm does **not** fork Wasmtime, and which alternative simplified-IR path the JIT detector walks instead. |
| [RISKS.md](RISKS.md) | — | Living risk register tracking architectural risks, upstream pinning decisions, and known limitations, refreshed alongside every `CHANGELOG.md` release. |
| [AUTO-OFFLOAD.md](AUTO-OFFLOAD.md) | — | User-facing reference for the auto-offload pipeline: which Wasm patterns the detector recognises, which it rejects, and how to enable it. |
| [CUDARC-SPIKE.md](CUDARC-SPIKE.md) | W1.2 | The `cust` → `cudarc` migration spike record: version chosen, API mapping table, known gaps, and the recommended cutover plan. |
| [COLD-START.md](COLD-START.md) | — | The five-component additive model for cold-start latency on a TensorWasm node and the operator levers that affect each component. |
| [INSTANCE-POOL.md](INSTANCE-POOL.md) | B5.8 | Roadmap feature #5 (pre-instantiated instance pool): scaffold status, configuration knobs, the v0.4 implementation plan, and the reset-on-return contract. |
| [KERNEL-REGISTRY.md](KERNEL-REGISTRY.md) | B6.3 | Roadmap feature #3 (signed kernel registry): HMAC-SHA256 `KernelManifest` records, `InMemoryRegistry` scaffold, and the v0.4 server-wire + on-disk store deliverable. |
| [DIFFERENTIAL-ORACLE.md](DIFFERENTIAL-ORACLE.md) | B5.9 | Roadmap feature #6 (differential JIT correctness oracle): bit-identity assertion contract between the Wasmtime CPU path and the JIT PTX path, plus the per-kernel tolerance policy. |
| [ARTIFACT-STORE.md](ARTIFACT-STORE.md) | B6.6 | Roadmap feature #9 (unified content-addressed signed artifact store): the `tensor-wasm-artifacts` trait surface, on-disk envelope, and the v0.4 convergence plan for the JIT L2 cache + snapshot store. |
| [glossary.md](glossary.md) | — | Short paragraph definitions of recurring CUDA, Wasm, and TensorWasm-internal terms (UVM, MPS, MIG, PTX, WMMA, BLAKE3 fingerprint, deopt guard, dispatch future, etc.). |

## API surface

The stable wire and binary surfaces TensorWasm commits to: HTTP REST,
audit-log JSON, the published Rust + OpenAPI reference archive, and
the mTLS contract that fronts them all. The hand-written REST
reference in `crates/tensor-wasm-api/API.md` is the canonical surface
for humans; the per-release rustdoc + OpenAPI bundle described in
`API-REFERENCE.md` is the canonical surface for tooling.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [../crates/tensor-wasm-api/API.md](../crates/tensor-wasm-api/API.md) | — | Hand-written REST reference for every endpoint the `tensor-wasm-api` gateway serves, with request/response examples for each route. |
| [API-REFERENCE.md](API-REFERENCE.md) | W4.8 | Publication-policy for the per-release rustdoc + OpenAPI archive: what is in it, what is not, the URL contract, and the workflow that produces it. |
| [AUDIT-LOG.md](AUDIT-LOG.md) | W2.2 | Wire-format schema, sink configuration, rotation guidance, and stable-string contract for the structured audit log emitted on state-mutating routes. |
| [STREAMING.md](STREAMING.md) | B6.1 | Roadmap feature #2 (streaming HTTP `invoke` responses): the `wasi:tensor/host.emit-chunk` host-fn contract, the SSE / chunked-transfer wire shape, and the v0.4 route-wiring deliverable. |
| [OPENAI-COMPAT.md](OPENAI-COMPAT.md) | B4.9 / B5.6 | Roadmap feature #10 (OpenAI-compatible inference gateway shim): the `/v1/completions` and `/v1/chat/completions` route surface, the v0.3.5 scaffold returning `501 openai_not_yet_wired`, and the v0.4 translation-layer deliverable. |
| [deployment/mtls.md](deployment/mtls.md) | W2.8 | Two production deployment shapes — self-terminated rustls and reverse-proxy fronting — with a recommended path for the v0.4 binary that still binds plaintext. |

## Performance and benchmarking

How TensorWasm measures itself, where the published numbers come from,
the operator-side SLO contract, and the runbook/dashboard pair that
turns burn-rate alerts into mitigation steps. Internal regression
(`PERFORMANCE.md` + committed `baseline.json`) and external comparison
(`BENCHMARKING.md`) are split deliberately — the latter pulls in the
anti-cheating checklist a reader needs to reproduce a result a blog
post would publish.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | — | How TensorWasm measures performance, what the current reference numbers look like, and how the committed-`baseline.json` CI regression gate works. |
| [BENCHMARKING.md](BENCHMARKING.md) | — | Companion to PERFORMANCE.md focused on **external** comparisons: same-workload, same-hardware, same-statistics rules for honest competitive benchmarks. |
| [CAPACITY-PLANNING.md](CAPACITY-PLANNING.md) | W4.4 | Three reference SKUs, four sizing formulas, and tenants-per-host curves that translate the SLO targets and bench medians into a host-sizing answer. |
| [SLO.md](SLO.md) | W1.9 | The project's commitment to numeric availability, latency, and error-rate targets for the HTTP surface and kernel-dispatch path. |
| [dashboards/README.md](dashboards/README.md) | W2.5 | Index for the importable Grafana dashboard (`tensor-wasm-overview.json`) that gives one stat panel per SLI and one row per subsystem. |
| [runbooks/README.md](runbooks/README.md) | W2.6 | One-page-per-alert operator manual: the alert → runbook mapping from `SLO.md` §7 with shared mitigation-step structure across every page. |

## CUDA

The toolkit-install path, the multi-tenant MPS daemon contract, the
kernel-authoring guide, and the auto-offload reference for the JIT
pipeline. A reader new to TensorWasm's CUDA story should read
`CUDA-SETUP.md` first (matrix of toolkit + driver + arch), then
`CUDA-KERNELS.md` to write a kernel, then `MPS-SETUP.md` once they
need more than ~8 co-located tenants on one GPU.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [CUDA-SETUP.md](CUDA-SETUP.md) | W1.6 | The exact toolkit, driver, compiler, environment-variable, and verification matrix to bring a CUDA host online for TensorWasm development. |
| [MPS-SETUP.md](MPS-SETUP.md) | — | NVIDIA MPS daemon startup, capabilities, limits, and the runtime probe TensorWasm uses to decide between MPS-shared and per-tenant CUDA contexts. |
| [AUTO-OFFLOAD.md](AUTO-OFFLOAD.md) | — | User-facing reference for which Wasm patterns the auto-offload JIT recognises and how to enable it (also listed under Architecture). |
| [CUDA-KERNELS.md](CUDA-KERNELS.md) | W4.5 | Practical guide for developers writing CUDA kernels that load and dispatch under TensorWasm's `wasi:cuda` surface, covering both explicit and auto-offload paths. |
| [PLIRON-PIPELINE.md](PLIRON-PIPELINE.md) | — | Four-wave implementation plan for the Pliron-based auto-offload pipeline (Wasm to PTX via the interim `LoweredOp` IR and cuda-oxide), companion to RFC 0001. |

## Operations

Everything an operator running TensorWasm in production needs after
the gateway is alive: deployment topology, manifest sets for the three
supported orchestrators (Kubernetes, Helm, Nomad), upgrade and backup
playbooks, and the observability contract. The three orchestrator
deliverables (`deploy/k8s/`, `deploy/helm/tensor-wasm/`,
`deploy/nomad/`) describe the same single-instance runtime — an
operator can switch between them without re-learning the env-var
surface.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | — | The canonical production-topology reference: load balancer, gateway replicas, GPU pool, MPS, and disaster-recovery sequencing. |
| [../deploy/k8s/README.md](../deploy/k8s/README.md) | W2.7 | Plain-YAML Kubernetes reference manifests (namespace, configmap, deployment, service, ServiceMonitor) for self-managed installs. |
| [../deploy/helm/tensor-wasm/README.md](../deploy/helm/tensor-wasm/README.md) | W2.7 | Templated Helm chart for the same single-gateway topology as the plain manifests, with a values-driven install surface. |
| [../deploy/nomad/README.md](../deploy/nomad/README.md) | W5.6 | HashiCorp Nomad reference job specs (docker and raw_exec) for the same single-instance runtime as the k8s and Helm assets. |
| [UPGRADE.md](UPGRADE.md) | W3.3 | Operator-facing fleet upgrade playbook describing the opinionated sequence for rolling a running TensorWasm deployment from one release to another. |
| [BACKUP-RESTORE.md](BACKUP-RESTORE.md) | W3.7 | What a production TensorWasm deployment must back up, the tested strategies, the restore paths, and the validation procedure that confirms a backup is good. |
| [OBSERVABILITY.md](OBSERVABILITY.md) | — | The `tracing` span schema, the optional OTLP exporter stack, and how to wire a local collector for development. |
| [CONFIG.md](CONFIG.md) | B2.9 | Single-source reference for every environment variable consumed by `tensor-wasm`, grouped by crate, with default + type + effect columns. |
| [GPU-QUOTAS.md](GPU-QUOTAS.md) | B6.5 | Roadmap feature #8 (per-tenant GPU memory quotas): the v0.3.7 in-process counter scaffold, the `TenantContextBuilder` surface, and the v0.4 `cuMemPool` driver-level enforcement plan. |
| [COOPERATIVE-YIELD.md](COOPERATIVE-YIELD.md) | B6.4 | Roadmap feature #4 (cooperative deadlines via WASI yield): the `wasi:scheduler/host@0.1.0` protocol, the CONTINUE / DEADLINE-NEAR / DEADLINE-ELAPSED return codes, and the embedder wiring snippet. |

## Security

The threat model, the v0.1 audit findings, the backport policy that
governs how security fixes flow into supported release branches, and
the runbook a maintainer follows to rehearse a coordinated disclosure
end to end before a real CVE arrives. Reports go to
`security@craton.com.ar` (covered in `../SECURITY.md`).

| Doc | Wave | One-sentence summary |
|---|---|---|
| [../SECURITY.md](../SECURITY.md) | W3.5 (backport policy), M8.5 (snapshot HMAC) | TensorWasm's threat model, isolation strategy summary, the optional snapshot HMAC authentication (cross-linked to [the v2 → v3 migration](SNAPSHOT-COMPATIBILITY.md#v2--v3-migration-signed-snapshots)), and the backport policy that decides which security fixes land on which release branches. |
| [SECURITY-AUDIT.md](SECURITY-AUDIT.md) | — | The v0.1 security-audit findings: methodology (manual walk + `cargo-fuzz`), per-asset verdict, and the follow-up tracking for partially-mitigated items. |
| [TESTING.md](TESTING.md) | B2.9 | Testing conventions across the workspace: unit/integration/CUDA/fuzz layers, the `#[ignore]` policy for hardware-gated tests, and the CI matrix that runs them. |
| [FUZZING.md](FUZZING.md) | B2.9 | The `fuzz/` directory layout, per-target corpora, the nightly + weekly cron schedule, and the v0.5 24-hour gate that determines when a target counts as "covered". |
| [runbooks/cve-disclosure-dry-run.md](runbooks/cve-disclosure-dry-run.md) | W5.5 | Manual procedure for rehearsing the CVE disclosure pipeline end-to-end on a test repository before a real CVE arrives. |

## Governance and supply chain

The maintainer registry, the decision process, the RFC pipeline,
release engineering (CHANGELOG, MIGRATION, PATH-TO-V1), and the
supply-chain commitments (SBOM, reproducible builds, Wasmtime cadence,
trademark policy) that together ground the v1.0 gate in
`PATH-TO-V1.md`. The split is deliberate: `GOVERNANCE.md` is the
rules, `MAINTAINERS.md` is the registry, and the RFC pipeline is the
mechanism that produces every other change that touches them.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [../GOVERNANCE.md](../GOVERNANCE.md) | W1.8 | Lightweight, opinionated governance model for a small core team, modeled on Wasmtime/ripgrep/zellij rather than the CNCF TOC. |
| [../MAINTAINERS.md](../MAINTAINERS.md) | W5.4 | Source of truth for the current maintainer roster and the active-maintainer count used by the quorum math in `GOVERNANCE.md`. |
| [TRADEMARK.md](TRADEMARK.md) | W3.4 | Permissive trademark policy: forks and re-implementations may reuse the "Craton TensorWasm" and "TensorWasm" names; only substantive forks must rebrand for clarity. |
| [SBOM.md](SBOM.md) | W4.3 | What the CycloneDX SBOM shipped with every release contains, what it does not contain, how to regenerate it locally, and the maintainer contract. |
| [REPRODUCIBLE-BUILDS.md](REPRODUCIBLE-BUILDS.md) | W3.6 | Recipe for two independent builds of the same release tag producing bit-identical sha256 digests on Linux x86_64 with the pinned toolchain. |
| [WASMTIME-UPGRADE.md](WASMTIME-UPGRADE.md) | W2.9 | Cadence policy for Wasmtime version bumps: quarterly minor bumps, major bumps case-by-case, plus the per-bump maintainer checklist. |
| [RELEASE.md](RELEASE.md) | B2.9 | Release-engineering runbook: tag preconditions, the per-release CHANGELOG / SBOM / cosign step sequence, and the `@craton-co/release` ownership contract. |
| [../CHANGELOG.md](../CHANGELOG.md) | W3.1 | Keep-a-Changelog log of every notable change, grouped by semver release; the `[Unreleased]` section tracks the v0.2–v0.4 wave work staged on `main`. |
| [MIGRATION-v0-to-v1.md](MIGRATION-v0-to-v1.md) | W3.2 | Operational checklist a v0.x deployment follows to land on v1.0 cleanly, populated continuously between v0.1 and v1.0. |
| [PATH-TO-V1.md](PATH-TO-V1.md) | — | The proposed five-milestone roadmap from the current v0.1.0 preview to a v1.0 production release, with explicit anti-goals and open decisions. |
| [../rfcs/README.md](../rfcs/README.md) | W1.7 | Lightweight RFC process: one contributor writes a doc, opens a PR, gives reviewers a week, and a maintainer decides. |
| [../rfcs/TEMPLATE.md](../rfcs/TEMPLATE.md) | W1.7 | The required starting point for a new RFC; copy to `rfcs/0000-short-kebab-slug.md` and fill in the sections in order. |

## Snapshots

The on-disk format spec and the cross-version compatibility promise.
`SNAPSHOT-COMPATIBILITY.md` is the **which-version-restores-what**
contract; `crates/tensor-wasm-snapshot/FORMAT.md` is the **what-the-bytes-are**
spec. The two are kept in sync deliberately: a wire-format bump
touches both files in the same PR.

| Doc | Wave | One-sentence summary |
|---|---|---|
| [SNAPSHOT-COMPATIBILITY.md](SNAPSHOT-COMPATIBILITY.md) | W1.3, M8.5 (v2 → v3) | The cross-version compatibility promise: which TensorWasm versions can restore which on-disk snapshot versions, the format-bump procedure, and the [v2 → v3 signed-snapshot migration](SNAPSHOT-COMPATIBILITY.md#v2--v3-migration-signed-snapshots) (provision key → configure reader → configure writer → flip to strict mode). |
| [../crates/tensor-wasm-snapshot/FORMAT.md](../crates/tensor-wasm-snapshot/FORMAT.md) | — | The wire-format specification — the byte layout `SnapshotWriter::capture` produces and `SnapshotReader::restore` consumes, including the magic constant and current version. |

---

## Missing cross-links

The docs below ship in the repository and are reachable through this
index, but **do not currently have inbound links from
[`README.md`](../README.md)**. They are flagged here so a future PR can
add table rows for them. This index does **not** modify any other
document; the missing anchors are intentionally left for a separate
change.

The runbook README itself is linked from `README.md`, but the
individual runbook pages it lists are not — they are reachable only
through that index, which is the intended structure for an on-call
manual (operators navigate from the alert payload via the index, not
from the project landing page). Listed here for completeness.

### Discoverable only through this index

| Doc | Why it should appear in `README.md` |
|---|---|
| [../crates/tensor-wasm-snapshot/FORMAT.md](../crates/tensor-wasm-snapshot/FORMAT.md) | Snapshot wire-format spec — currently reachable only from `SNAPSHOT-COMPATIBILITY.md` and crate-internal docs; the format is part of the public contract per `SNAPSHOT-COMPATIBILITY.md` and warrants a top-level anchor. |
| [../deploy/nomad/README.md](../deploy/nomad/README.md) | Nomad reference manifests (W5.6) ship alongside the k8s and Helm assets, both of which are already in the `README.md` Operations table. Adding a row keeps the three orchestrators symmetric. |
| [../rfcs/TEMPLATE.md](../rfcs/TEMPLATE.md) | Linked from the Contributing section of `README.md` (line 235); already covered, listed here only so the inventory is complete. |
| [RISKS.md](RISKS.md) | Living risk register, referenced from `CHANGELOG.md` but not directly from `README.md`'s Architecture & reference table. |

### Discoverable only via runbooks/README.md (by design)

These pages are intentionally reached through their index because the
alert payload, not the project landing page, is what an operator opens
during a page:

| Doc | Wave | Summary |
|---|---|---|
| [runbooks/availability-fast-burn.md](runbooks/availability-fast-burn.md) | W2.6 | Runbook for the 14.4× burn-rate alert against `availability_http`. |
| [runbooks/availability-slow-burn.md](runbooks/availability-slow-burn.md) | W2.6 | Runbook for the 6× burn-rate alert against `availability_http`. |
| [runbooks/availability-very-slow-burn.md](runbooks/availability-very-slow-burn.md) | W2.6 | Runbook for the 1× sustained-burn alert against `availability_http`. |
| [runbooks/dispatch-latency-spike.md](runbooks/dispatch-latency-spike.md) | W2.6 | Runbook for the kernel-dispatch P95 latency SLO breach (`tensor_wasm_kernel_latency_seconds`). |
| [runbooks/invoke-latency-spike.md](runbooks/invoke-latency-spike.md) | W2.6 | Runbook for the `POST /functions/{id}/invoke` P95 latency SLO breach. |
| [runbooks/healthz-slow.md](runbooks/healthz-slow.md) | W2.6 | Runbook for the `GET /healthz` P95 latency SLO breach. |
| [runbooks/rollback.md](runbooks/rollback.md) | W2.6 | Manual procedure for reverting a TensorWasm node from a bad release. |
| [runbooks/oncall-paging.md](runbooks/oncall-paging.md) | W2.6 | Manual procedure for escalating from operator-handling to waking the on-call maintainer. |
| [runbooks/trace-id.md](runbooks/trace-id.md) | W2.6 | Non-alert reference for finding logs that share a given trace id. |
| [runbooks/disaster-recovery.md](runbooks/disaster-recovery.md) | W3.7 | Manual procedure for bringing a TensorWasm deployment back online after the host kernel, snapshot store, or CUDA driver has been lost. |
| [runbooks/cve-disclosure-dry-run.md](runbooks/cve-disclosure-dry-run.md) | W5.5 | Manual procedure for rehearsing the CVE disclosure pipeline end-to-end on a test repository (also listed under Security). |

---

## Audience routing

A quick lookup table for "which docs should I read first?" by role.
Each row lists three documents in the order a fresh reader should
take them on.

| Role | First | Second | Third |
|---|---|---|---|
| Wasm developer | [GETTING-STARTED.md](GETTING-STARTED.md) | [WASM-DEVELOPER-GUIDE.md](WASM-DEVELOPER-GUIDE.md) | [../crates/tensor-wasm-api/API.md](../crates/tensor-wasm-api/API.md) |
| CUDA kernel author | [CUDA-SETUP.md](CUDA-SETUP.md) | [CUDA-KERNELS.md](CUDA-KERNELS.md) | [AUTO-OFFLOAD.md](AUTO-OFFLOAD.md) |
| SRE / operator | [DEPLOYMENT.md](DEPLOYMENT.md) | [tutorials/production-deployment.md](tutorials/production-deployment.md) | [runbooks/README.md](runbooks/README.md) |
| Capacity planner | [SLO.md](SLO.md) | [CAPACITY-PLANNING.md](CAPACITY-PLANNING.md) | [PERFORMANCE.md](PERFORMANCE.md) |
| Security reviewer | [../SECURITY.md](../SECURITY.md) | [SECURITY-AUDIT.md](SECURITY-AUDIT.md) | [AUDIT-LOG.md](AUDIT-LOG.md) |
| Release engineer | [../CHANGELOG.md](../CHANGELOG.md) | [UPGRADE.md](UPGRADE.md) | [REPRODUCIBLE-BUILDS.md](REPRODUCIBLE-BUILDS.md) |
| Compliance / auditor | [SBOM.md](SBOM.md) | [AUDIT-LOG.md](AUDIT-LOG.md) | [SECURITY-AUDIT.md](SECURITY-AUDIT.md) |
| Maintainer (new) | [../GOVERNANCE.md](../GOVERNANCE.md) | [../MAINTAINERS.md](../MAINTAINERS.md) | [../rfcs/README.md](../rfcs/README.md) |
| Wasmtime/Wasmer evaluator | [MIGRATING-FROM-WASMTIME-WASMER.md](MIGRATING-FROM-WASMTIME-WASMER.md) | [WASMTIME-FORK.md](WASMTIME-FORK.md) | [WASMTIME-UPGRADE.md](WASMTIME-UPGRADE.md) |
| On-call (paged) | runbook from alert payload | [runbooks/README.md](runbooks/README.md) | [runbooks/trace-id.md](runbooks/trace-id.md) |

## How to extend this index

When a new doc lands, add one row to the most-relevant section above
and (if appropriate) an inbound link from `README.md`. Keep summaries
to a single sentence; if a doc warrants a paragraph, the doc itself
should carry that introduction in its first line so this index can
quote it. The wave tag is the W-number from
[`PATH-TO-V1.md`](PATH-TO-V1.md)'s workstream tables; leave it as an
em dash if the doc predates the wave program.

If an existing doc moves or is renamed, update both the row here and
the corresponding row in `README.md`. The two surfaces are kept in
sync deliberately: this index is the complete inventory, the
`README.md` table is the curated landing surface. When the inventory
and the landing surface drift apart, the
[Missing cross-links](#missing-cross-links) section above is the
ledger that records the gap.
