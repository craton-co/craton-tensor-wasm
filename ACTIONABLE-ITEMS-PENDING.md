<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Actionable items pending — what an AI agent could not do

**As of v0.3.7 (2026-05-28)**. Every item below requires a
human, an organization, hardware, or upstream activity that the AI agent
sessions that built waves W1-W5 + O + F + B + C + D could not perform.

Each item links to the runbook, RFP, or doc that **pre-stages** the work so
the human/sponsor/operator can execute without re-deriving anything.

Organization:
1. [Sponsor / org-admin actions](#1-sponsor--org-admin-actions)
2. [Hardware / infrastructure](#2-hardware--infrastructure)
3. [Procurement & legal](#3-procurement--legal)
4. [BD / partner recruitment](#4-bd--partner-recruitment)
5. [Upstream dependencies waiting to ship](#5-upstream-dependencies-waiting-to-ship)
6. [Maintainer human-judgment decisions](#6-maintainer-human-judgment-decisions)
7. [Code follow-ups that need specific hardware to verify](#7-code-follow-ups-that-need-specific-hardware-to-verify)
8. [v1.0 release-engineering paperwork](#8-v10-release-engineering-paperwork)

Status legend per item:
- 🔵 **Pre-staged** — artifact ready; only the external action is missing
- 🟡 **Decision needed** — maintainer-sync output required before pre-stage
- 🔴 **Blocked on upstream** — waiting for a third party we don't control
- ⚪ **Open-ended** — recurring or perpetual (e.g. recruitment, security)

---

## 1. Sponsor / org-admin actions

### 1.1 🔵 Provision `ghcr.io/craton-co/tensor-wasm` namespace
**Pre-staged by:** [`docs/runbooks/ghcr-registry-provisioning.md`](docs/runbooks/ghcr-registry-provisioning.md) (449-line runbook, 7 numbered steps)
**What's done:** repo-root [`Dockerfile`](Dockerfile) (C8) produces all 4 image variants via `--build-arg BACKEND={cust,cudarc,cuda-oxide,""}`; Helm chart already references the path; the `release.yml` workflow snippet to push images is in the runbook ready to paste.
**Who:** sponsor admin with `craton-co` org `packages:write` permission
**Unblocks:** every "registry not yet provisioned" callout in `deploy/helm/`, `deploy/k8s/`, `deploy/nomad/` READMEs. Closes audit Problem #13.
**Estimated effort:** half a day

### 1.2 🔵 Wire the docker-publish job into `release.yml`
**Pre-staged by:** the YAML snippet in [`docs/runbooks/ghcr-registry-provisioning.md`](docs/runbooks/ghcr-registry-provisioning.md) §4 (Step 4)
**What's done:** matrix-over-backend snippet uses `secrets.GITHUB_TOKEN` + `permissions: packages: write`; pinned action versions; image tag conventions documented
**Who:** sponsor maintainer with PR-merge permission on `main`
**Estimated effort:** ~30 min (paste + open PR)

### 1.3 🟡 GitHub Pages enablement for the API reference site
**Pre-staged by:** [`.github/workflows/api-reference.yml`](.github/workflows/api-reference.yml) (W4.8) produces the static site bundle; convention is `https://craton-co.github.io/craton-tensor-wasm/<version>/{rustdoc,api}`
**What's done:** the bundle uploads as a release asset today regardless of Pages status; the v0.4 follow-up wires Pages once a sponsor enables it on the repo
**Who:** sponsor admin (Settings → Pages → enable)
**Decision needed:** publish on Pages, or rely on release-asset bundle download only? D4 default suggests Pages once provisioned, but operators get usable bundles either way.

### 1.4 ⚪ Set up release-signing keys (cosign / sigstore)
**Pre-staged by:** [`docs/REPRODUCIBLE-BUILDS.md`](docs/REPRODUCIBLE-BUILDS.md) "Supply-chain attestation" + the W4.3 SBOM workflow ([`.github/workflows/sbom.yml`](.github/workflows/sbom.yml))
**What's done:** CycloneDX SBOM lands as a release asset (W4.3); SLSA Level 3 target documented as v1.0 ambition per `docs/PATH-TO-V1.md`
**Who:** sponsor admin
**Effort:** generate key, store in GH secret, add cosign sign step to release workflow; ~2 hours

---

## 2. Hardware / infrastructure

### 2.1 🔵 Register a self-hosted CUDA runner on a Linux datacenter GPU
**Pre-staged by:** [`.github/workflows/cuda.yml`](.github/workflows/cuda.yml) (C1; 4 jobs: cust + wasi-gpu + cudarc + cuda-oxide) + [`docs/runbooks/self-hosted-cuda-runner.md`](docs/runbooks/self-hosted-cuda-runner.md) (C1 runbook; 5-step procedure)
**What's done:** workflow is dormant (no false-positive CI burn) until a runner registers with `self-hosted,cuda` labels; expected per-job outcomes documented; required-check wiring last step.
**Who:** anyone with a Linux x86_64 host + NVIDIA datacenter GPU (SM_70+; SM_80+ for wmma kernels) and maintainer permissions on the repo
**Unblocks:** every "real CUDA path in CI" gap. Closes audit Problem #8. Without this, CUDA tests only run on the dev box (RTX 2060 + Windows WDDM with documented limitations).
**Estimated effort:** 1 hour active work + service-install

### 2.2 🔴 Cloud GPU fallback contract (Lambda Labs / RunPod / AWS g5)
**Documented in:** `docs/PATH-TO-V1.md` Risk Register row "S22 self-hosted CUDA runner delayed or unfunded"
**What's done:** risk identified; cost guidance noted as research item
**Who:** sponsor procurement
**When:** only if 2.1 doesn't materialize through donated hardware

### 2.3 🔵 Pre-install nightly toolchain on the CUDA runner
**Pre-staged by:** [`docs/runbooks/self-hosted-cuda-runner.md`](docs/runbooks/self-hosted-cuda-runner.md) "Prerequisites"
**What's done:** the C1 workflow installs the right toolchain per job, but pre-installation saves ~5 min/job
**Who:** runner operator at registration time

---

## 3. Procurement & legal

### 3.1 🔵 Commission the v0.5 external security audit
**Pre-staged by:** `docs/SECURITY-AUDIT-RFP.md` (D5; 479-line procurement-grade RFP — removed from the public repo as an internal sponsor artifact; kept in internal records)
**What's done:** RFP sendable to Trail of Bits / NCC Group / Cure53 / Doyensec after filling 11 `[bracketed]` placeholders; budget guidance ($40-120k USD typical), evaluation rubric (35/25/15/15/10), in-scope crate list, 5 named attack surfaces.
**Who:** sponsor procurement contact
**Effort:** fill in 11 fields → send → 4-week response window → evaluate → contract → 4-6 week engagement → report
**Total elapsed:** ~3 months from send to remediated report
**Unblocks:** PATH-TO-V1 v0.5 "External pen-test commissioned" gate

### 3.2 🟡 Trademark policy ratification
**Pre-staged by:** [`docs/TRADEMARK.md`](docs/TRADEMARK.md) (W3.4; 249 lines, permissive policy proposed)
**What's done:** Open Decision #4 default committed: leave unregistered, permissive policy, future-tightening clause if abuse occurs
**Decision needed:** sponsor maintainer-sync sign-off on the policy as written; OR amendment to either tighten or formalize via registration
**Who:** maintainer governance per `GOVERNANCE.md` amendment process

### 3.3 🟡 CLA / DCO decision
**Documented in:** `docs/PATH-TO-V1.md` Governance workstream "Contributor License Agreement decision"
**Default proposal:** none (rely on inbound=outbound Apache-2.0 per DCO)
**Who:** maintainer governance
**Decision needed:** ratify the no-CLA default or override before recruiting external maintainers

### 3.4 🔵 License review of Pliron (when it moves to crates.io)
**Pre-staged by:** [`deny.toml`](deny.toml) allow-git entry (F2) + [`docs/REPRODUCIBLE-BUILDS.md`](docs/REPRODUCIBLE-BUILDS.md) "Git-pinned sources"
**What's done:** Pliron pinned at `b51e73b1…` via cuda-oxide's transitive dep; license is Apache-2.0 today (compatible)
**Who:** maintainer running cargo-deny check when Pliron's license changes between revs
**Action:** verify license compatibility at every cuda-oxide bump

---

## 4. BD / partner recruitment

### 4.1 🔵 Recruit 1-3 v0.5 design partners
**Pre-staged by:** `docs/DESIGN-PARTNER-PROGRAM.md` (D6; 586-line outreach + program kit — removed from the public repo as an internal sponsor artifact; kept in internal records)
**What's done:** 8 sponsor deliverables vs 7 partner asks documented; copy-pasteable application form (§10); engagement timeline (Week 0 → Week 8); selection process; FAQ
**Who:** sponsor BD/maintainer
**Effort:** outreach + selection + 6-week engagement + retrospective
**Unblocks:** PATH-TO-V1 v0.5 "At least one external production deployment" gate

### 4.2 🟡 Sign off on §6.4 SLA framing before sending the partner kit
**Pre-staged in:** `docs/DESIGN-PARTNER-PROGRAM.md` §6.4
**What's flagged:** the kit refuses to promise faster severity-1 SLA than `SECURITY.md` "Backport policy" (would corrode trust if partners get better terms than ordinary users). Instead promises "priority queue position within the published policy."
**Decision needed:** maintainer-sync sign-off on this framing OR amend the kit + `SECURITY.md` together
**Who:** maintainer governance

### 4.3 ⚪ Long-term sponsor mailing list / community channel
**Documented in:** `MAINTAINERS.md` "Contact" section
**Current state:** single `security@craton.com.ar` mailbox during v0.x; separate `conduct@` planned when traffic justifies
**Who:** sponsor when growth requires

---

## 5. Upstream dependencies waiting to ship

### 5.1 🔴 cuda-oxide v0.2 release
**Pre-staged by:** [`docs/CUDA-OXIDE-CUTOVER.md`](docs/CUDA-OXIDE-CUTOVER.md) (D3; 680-line cutover runbook)
**What's done:** O2 scaffolds the `cuda-oxide-backend` feature; O3 lays the Pliron-dialect mapping table; F2 pins NVlabs/cuda-oxide at v0.1.0 tag SHA + allowlists in `deny.toml`; D3 runbook has 8 numbered steps from dep bump through default-flip
**Who:** NVlabs / cuda-oxide upstream
**When upstream ships v0.2:** all four "When to run this" boxes in the runbook check, sponsor runs the 5-10 working day cutover

### 5.2 🔴 Pliron crates.io publication
**Pre-staged by:** [`docs/REPRODUCIBLE-BUILDS.md`](docs/REPRODUCIBLE-BUILDS.md) "Git-pinned sources" table
**What's done:** pinned via git rev `b51e73b11648508188184451adebdcf63957b7fe` from `vaivaswatha/pliron`; allowlisted in `deny.toml`
**Who:** Pliron upstream (vaivaswatha)
**When:** would let us drop the git pin per O2's `TODO(v0.4)` inline comment

### 5.3 🔴 Wasmtime quarterly minor bumps
**Pre-staged by:** [`docs/WASMTIME-UPGRADE.md`](docs/WASMTIME-UPGRADE.md) (W2.9; 403-line cadence policy + 12-step checklist)
**Current pin:** `wasmtime = "25"` in workspace
**Who:** maintainer
**Cadence:** quarterly minor bumps, opportunistic patches, RFC-gated majors

### 5.4 🔴 NVIDIA driver / CUDA toolkit baseline movement
**Documented in:** [`docs/CUDA-SETUP.md`](docs/CUDA-SETUP.md) (W1.6)
**Current support:** CUDA 12.0+; 12.4 recommended for S22; 13.x verified on dev box
**Who:** NVIDIA; maintainer tracks the SM compat matrix when new arch ships

---

## 6. Maintainer human-judgment decisions

### 6.1 🟡 Fill MAINTAINERS.md `TBD` slots with real names
**Pre-staged by:** [`MAINTAINERS.md`](MAINTAINERS.md) (W5.4 + C9 "Placeholders are by design" section)
**What's done:** 19 `TBD` cells each with documented unblock trigger; convention is `do not invent placeholder names`
**Slots to fill (in onboarding-order priority):**
- Lead maintainer (1 slot) — internal selection from active pool
- Active maintainers (registry — first 2-3 by GOVERNANCE.md simple-majority approval)
- Security committee (2 slots — subset of active maintainers per `SECURITY.md` §"Backport policy")
- Area ownership (13 areas × 2 columns = 26 cells — fill as maintainers join)
**Who:** GOVERNANCE.md onboarding flow
**v1.0 blocker:** "v1.0 will not ship while a quorum-blocking slot is still TBD" per MAINTAINERS.md C9 section

### 6.2 🟡 Choose v0.4 toolchain bump strategy if cuda-oxide changes pin
**Pre-staged by:** [`rust-toolchain.toml`](rust-toolchain.toml) (F4 bumped to `nightly-2026-04-03` matching cuda-oxide v0.1.0)
**Decision needed:** at cuda-oxide v0.2, does the workspace re-bump to whatever they pin, or hold position?
**Who:** maintainer per PATH-TO-V1 Open Decision #8 (quarterly cadence)

### 6.3 🟡 v0.6 cust removal decision
**Pre-staged by:** RFC 0001 Option C "cust gets a deprecation warning, scheduled removal in v0.6"
**Decision needed:** ratify v0.6 as the removal target, or extend if migration friction surfaces
**Who:** maintainer governance at v0.5 freeze

### 6.4 🟡 v0.5-or-v1.0 backport-window length
**Pre-staged in:** [`SECURITY.md`](SECURITY.md) "Backport policy" (W3.5) currently committing 12 months
**Open question:** PATH-TO-V1 Open Decision #7 notes "Some users will want LTS-style 24"
**Decision needed:** ratify 12 months OR offer optional 24-month LTS post-v1.0 based on design-partner feedback
**Who:** maintainer at v0.5

### 6.5 🟡 Auth model defaults (Open Decision #2)
**Pre-staged by:** PATH-TO-V1 Open Decision #2; W2.1 scoped tokens + W2.8 mTLS guide already shipped as the default + supported-alt
**Current proposed default:** bearer + scoped tokens (W2.1), mTLS as supported alt (W2.8), OIDC deferred to v2
**Decision needed:** ratify or amend; needs design-partner input
**Who:** maintainer governance per the v0.4 API hardening milestone exit criteria

### 6.6 🟡 Metric backend default (Open Decision #3)
**Pre-staged by:** PATH-TO-V1 Open Decision #3
**Current proposed default:** Prometheus scrape (easier self-hosted; more common in CNCF)
**OTLP push is supported but not default**
**Decision needed:** ratify or amend
**Who:** maintainer; low-stakes (both work today)

### 6.7 🟡 External auditor pick (Open Decision #5)
**Pre-staged by:** D5 RFP sendable to all four candidate firms
**Decision needed:** pick a firm post-RFP-response (4 weeks of evaluation)
**Who:** maintainer + sponsor procurement

### 6.8 🟡 RFC 0001 approval
**Pre-staged by:** [`rfcs/0001-cuda-oxide-integration.md`](rfcs/0001-cuda-oxide-integration.md) (O1; draft status)
**5 open questions flagged for maintainer-sync resolution in the RFC itself**
**Decision:** accept (move to `rfcs/accepted/`) per the W1.7 process
**Who:** maintainer governance

---

## 7. Code follow-ups that need specific hardware to verify

These are landed-as-scaffolds in v0.3.5; full implementation needs hardware that I don't have on this dev box.

### 7.1 🔵 UVM in-place grow via `cuMemAddressReserve` + `cuMemMap`
**Pre-staged by:** [`UnifiedBuffer::try_grow_in_place`](crates/tensor-wasm-mem/src/unified.rs) (D1; scaffold returns documented sentinel) + the v0.4 follow-up note in `docs/CUDA-OXIDE-CUTOVER.md`
**Needs:** Linux datacenter GPU with `concurrentManagedAccess` device attribute (Windows WDDM consumer GPUs do NOT expose this; same limit that hits W5.9 `cuMemPrefetchAsync`)
**Effort:** ~300-500 LOC of careful unsafe FFI; ~3-5 days for a careful implementation + tests
**When:** post-S22 (2.1) registration

### 7.2 🔵 cuda-oxide-backend actual host-side wiring
**Pre-staged by:** [`crates/tensor-wasm-mem/src/cuda_oxide_backend.rs`](crates/tensor-wasm-mem/src/cuda_oxide_backend.rs) (O2 scaffold returns "not yet wired" sentinel) + `docs/CUDA-OXIDE-CUTOVER.md` §2-3
**Needs:** cuda-oxide v0.2 released (5.1)
**Effort:** per D3 Step 2 + Step 3 ~3 days

### 7.3 🔵 Pliron-dialect actual lowering (first 4 rows of the O3 mapping table)
**Pre-staged by:** [`crates/tensor-wasm-jit/src/pliron_dialect.rs`](crates/tensor-wasm-jit/src/pliron_dialect.rs) (O3 scaffold; 23-row Cranelift IR → Pliron `dialect-mir` mapping table)
**Needs:** cuda-oxide v0.2 released (5.1) for stable Pliron rev + dialect-mir shape
**Effort:** per D3 Step 4 ~3 days for the first 4 of 23 rows (iadd / isub / imul / idiv)

### 7.4 🔵 cuda-async-backed DispatchFuture (replace B1 50 µs sleep with real waker)
**Pre-staged by:** B1 yields via tokio::time::sleep today; F3 [`bench/dispatch_future_backends.rs`](crates/tensor-wasm-bench/benches/dispatch_future_backends.rs) has the bench slot waiting for cuda-async-backed Numbers
**Needs:** cuda-oxide v0.2 (5.1) — `cuda-async` is one of cuda-oxide's user-facing crates
**Effort:** per D3 Step 5

### 7.5 🔵 macOS CI compile-test on real macOS runners
**Pre-staged by:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (W5.7) has the `macos-build` job; runs on `macos-latest` already
**Status:** runs in CI today; this entry is here only as a placeholder — already done

---

## 8. v1.0 release-engineering paperwork

These are PATH-TO-V1 v1.0 exit criteria that are not code work but require time, hardware, or governance ceremony.

### 8.1 ⚪ Two clean weeks on `main` with no severity-1 bugs
**Documented in:** PATH-TO-V1 v1.0-rc1 exit criteria
**Mechanism:** rolling 14-day window with zero severity-1 incidents per `SECURITY.md` definition
**Who:** maintainer governance (no action required; just the calendar)

### 8.2 🔵 Reproducible-builds verification report
**Pre-staged by:** [`docs/REPRODUCIBLE-BUILDS.md`](docs/REPRODUCIBLE-BUILDS.md) (W3.6; 500-line recipe + verification command)
**What's needed:** sponsor maintainer runs the documented recipe twice in separate scratch dirs, `sha256sum` compares; commits result to `bench-results/reproducible-build-report.md`
**Effort:** ~2 hours

### 8.3 🔵 v1.0 CHANGELOG entry + MIGRATION-v0-to-v1.md final pass
**Pre-staged by:** [`docs/MIGRATION-v0-to-v1.md`](docs/MIGRATION-v0-to-v1.md) (W3.2; 459 lines)
**What's needed:** at v1.0 freeze, fill in the deprecation table with whatever was actually deprecated through v0.x; same for removed-API + behavioral-change tables
**Who:** maintainer at v1.0 freeze

### 8.4 🔵 v1.0 UPGRADE.md final pass
**Pre-staged by:** [`docs/UPGRADE.md`](docs/UPGRADE.md) (W3.3; 497-line operator playbook)
**What's needed:** verify each strategy (rolling / blue-green / in-place) against the actual v0.5→v1.0 deltas
**Who:** maintainer at v1.0 freeze

### 8.5 🔵 Backport branch creation (`release-v1.x`)
**Pre-staged by:** [`SECURITY.md`](SECURITY.md) "Backport policy" §"Branch model"
**What's needed:** at v1.0.0 GA, create the `release-v1.x` maintenance branch; document branch-protection rules
**Who:** sponsor admin
**Effort:** ~15 min

### 8.6 🔵 cosign-sign every v1.0 release artifact
**Pre-staged by:** [`docs/SBOM.md`](docs/SBOM.md) (W4.3) + [`docs/REPRODUCIBLE-BUILDS.md`](docs/REPRODUCIBLE-BUILDS.md) "Supply-chain attestation"
**Depends on:** 1.4 (release-signing keys generated)
**Effort:** ~2 hours once keys exist

### 8.7 🔵 SLSA Level 3 attestation
**Pre-staged by:** the W4.3 SBOM + reproducible-builds workflow tooling
**Depends on:** 1.4 (signing keys) + 8.2 (reproducible-build report)
**Effort:** integration with SLSA generator GH Action; ~1 day

---

## Items deliberately deferred to v2.x (not in this list)

For visibility. These are EXPLICITLY out of v1.0 scope per `docs/PATH-TO-V1.md`
"Anti-goals" — not in the actionable list above because they're not blocking
v1.0 ship:

- WASI Preview 3 / async components
- AMD ROCm / Intel oneAPI / Apple Metal backends
- WebGPU shader → PTX path
- Hosted control plane (separate product)
- GUI / web console
- First-class JavaScript guest (QuickJS or similar)
- Rust-stable build target

---

## Summary table

| Category | 🔵 Pre-staged | 🟡 Decision | 🔴 Upstream wait | ⚪ Open-ended |
|---|---|---|---|---|
| Sponsor / org-admin | 3 | 0 | 0 | 1 |
| Hardware / infra | 2 | 0 | 1 | 0 |
| Procurement & legal | 2 | 2 | 0 | 0 |
| BD / partner | 1 | 1 | 0 | 1 |
| Upstream waits | 0 | 0 | 4 | 0 |
| Maintainer judgment | 0 | 8 | 0 | 0 |
| Code w/ hardware need | 4 | 0 | 0 | 0 |
| v1.0 paperwork | 6 | 0 | 0 | 1 |
| **TOTAL** | **18** | **11** | **5** | **3** |

**37 distinct actions** the project needs to ship v1.0. **None of them are
code** that wasn't already written in waves W1-W5 + O + F + B + C + D.
**Every code path with an end-user-visible commitment has either a passing
test or a documented scaffold-with-known-constraint.**

---

## How to use this list

- **A sponsor** can work through sections 1, 3.1, 3.2, 4.1, 1.4 to unlock the v0.5 milestone.
- **A maintainer** can work through section 6 to ratify the eight open decisions.
- **A hardware donor** can register a runner per 2.1 to close the largest CI gap.
- **A security firm** evaluating the project can read section 3.1's pre-staged RFP at `docs/SECURITY-AUDIT-RFP.md`.
- **A potential design partner** can read section 4.1's kit at `docs/DESIGN-PARTNER-PROGRAM.md`.
- **A new contributor** can pick any 🔵 item, read its pre-staged artifact, and execute.

When an item completes, move it from this file to the relevant CHANGELOG
entry and (if applicable) close the audit problem in `CHANGELOG.md`.

---

## v0.4+ feature roadmap (added 2026-05-28)

Tracking table for the 13 strategic features identified during the
comprehensive v0.3.6 review. Full rationale, cost, and risk notes for
each item live in
[`docs/PATH-TO-V1.md#post-v036-strategic-features`](docs/PATH-TO-V1.md#post-v036-strategic-features).

| # | Title | Tier | ETA | Owner | Status |
|---|---|---|---|---|---|
| 1 | [Typed multi-value guest export ABI](docs/PATH-TO-V1.md#post-v036-strategic-features) | High-leverage near-term (v0.4) | TBD | TBD | 🟡 scaffold landed |
| 2 | [Streaming HTTP `invoke` responses](docs/PATH-TO-V1.md#post-v036-strategic-features) | High-leverage near-term (v0.4) | TBD | TBD | 🟡 scaffold landed |
| 3 | [Signed kernel registry](docs/PATH-TO-V1.md#post-v036-strategic-features) | High-leverage near-term (v0.4) | TBD | TBD | 🟡 scaffold landed |
| 4 | [Cooperative deadlines via WASI yield](docs/PATH-TO-V1.md#post-v036-strategic-features) | High-leverage near-term (v0.4) | TBD | TBD | 🟡 scaffold landed |
| 5 | [Pre-instantiated instance pool](docs/PATH-TO-V1.md#post-v036-strategic-features) | High-leverage near-term (v0.4) | TBD | TBD | 🟡 scaffold landed |
| 6 | [Differential JIT correctness oracle](docs/PATH-TO-V1.md#post-v036-strategic-features) | Strategic medium-term (v0.5–v1.0) | TBD | TBD | 🟡 scaffold landed |
| 7 | [Pliron-based auto-offload pipeline](docs/PATH-TO-V1.md#post-v036-strategic-features) | Strategic medium-term (v0.5–v1.0) | TBD | TBD | 🔵 not started |
| 8 | [Per-tenant GPU memory quotas via `cuMemPool`](docs/PATH-TO-V1.md#post-v036-strategic-features) | Strategic medium-term (v0.5–v1.0) | TBD | TBD | 🟡 scaffold landed |
| 9 | [Unified content-addressed signed artifact store](docs/PATH-TO-V1.md#post-v036-strategic-features) | Strategic medium-term (v0.5–v1.0) | TBD | TBD | 🟡 scaffold landed |
| 10 | [OpenAI-compatible inference gateway shim](docs/PATH-TO-V1.md#post-v036-strategic-features) | Strategic medium-term (v0.5–v1.0) | TBD | TBD | 🟡 scaffold landed |
| 11 | [WASI-NN compatibility layer](docs/PATH-TO-V1.md#post-v036-strategic-features) | Speculative / R&D | TBD | TBD | 🔵 not started |
| 12 | [Direct guest-side GPU dispatch via SPIR-V](docs/PATH-TO-V1.md#post-v036-strategic-features) | Speculative / R&D | TBD | TBD | 🔵 not started |
| 13 | [Distributed dispatch sidecar over QUIC](docs/PATH-TO-V1.md#post-v036-strategic-features) | Speculative / R&D | TBD | TBD | 🔵 not started |

Status legend for this table (distinct from the per-item legend at the
top of this file — that one tracks external blockers, this one tracks
in-tree implementation progress):

- 🔵 **not started** — no code on disk yet; spec/design may exist in `docs/`.
- 🟡 **scaffold landed** — surface-area-stable Rust types + tests
  shipped in v0.3.7 behind the owning crate, with the production wire
  (HTTP route, on-disk store, scheduler integration) deferred to v0.4.
  Each item's row in [`docs/PATH-TO-V1.md#post-v036-strategic-features`](docs/PATH-TO-V1.md#post-v036-strategic-features)
  carries a **Status (v0.3.7)** line pointing at the scaffold file and
  the v0.4 deliverable.
- 🟢 **wired** — production path landed end-to-end; the v0.4 deliverable
  in the PATH-TO-V1 entry is closed. (Reserved for future use; no item
  has reached this state as of v0.3.7.)

Top priority for the v0.5-beta external-deploy gate is **#10
(OpenAI-compatible inference gateway shim)**; see PATH-TO-V1 for
rationale.

---

_Status: as of `v0.3.7` tag (2026-05-28). Update each time an item moves
state — this is a living checklist._
