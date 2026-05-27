<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Request For Proposal — External Security Audit of Craton TensorWasm v0.5

A procurement document. This file is the template the sponsor sends to
candidate audit firms; the sponsor renames it to
`SECURITY-AUDIT-RFP-v0.5.md` per release, completes every `[bracketed]`
placeholder below, and emails the resulting PDF to each firm individually.

The fill-in placeholders use the `[ALL-CAPS-WITH-DASHES]` convention so
they can be grep-replaced before sending.

---

## 1. Cover

| Field | Value |
|---|---|
| Project | Craton TensorWasm |
| Repository | `https://github.com/craton-co/craton-tensor-wasm` |
| Version under review | `v0.5.0-beta` (commit SHA fixed at engagement kick-off; see §5) |
| License | Apache-2.0 |
| Sponsor | Craton Software Company |
| Sponsor security contact | `security@craton.com.ar` |
| Sponsor procurement contact | `[SPONSOR-PROCUREMENT-CONTACT]` |
| RFP issued | `[RFP-ISSUE-DATE]` |
| Proposal deadline | `[PROPOSAL-DEADLINE-DATE]` (4 weeks from issuance) |
| Engagement target start | `[ENGAGEMENT-START-DATE]` (8-12 weeks from issuance) |
| Engagement duration | 4-6 weeks of audit time (calendar dates negotiable) |
| Addressed to | `[FIRM-NAME]`, attention `[FIRM-BUSINESS-DEVELOPMENT-CONTACT]` |

Submission instructions and confidentiality terms are in §10.

---

## 2. Project background

Craton TensorWasm is a GPU-accelerated serverless WebAssembly runtime
written in Rust. It executes untrusted Wasm guests under
[Wasmtime](https://wasmtime.dev/) on a Tokio async runtime, with explicit
GPU kernel dispatch via a WASI proposal (`wasi:cuda`) that bridges the
guest to NVIDIA CUDA through one of three feature-gated host backends:
the legacy `cust 0.3.x` path, the `cudarc-backend` clean-room
implementation (W1.2), and the new `cuda-oxide-backend` (NVlabs/cuda-oxide
v0.1.x alpha, git-pinned through Pliron — see
[`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md)).

The runtime ships a public HTTP API gateway built on axum, with bearer
authentication, per-token scoped tenant authorisation (W2.1), per-token
rate limiting (W1.4), a structured audit log of state-mutating requests
(W2.2), HTTP request metrics with route-template cardinality control
(W2.3), and W3C trace-context propagation (W4.1). Multi-tenant isolation
is implemented in `crates/tensor-wasm-tenant/` via per-instance CUDA
streams (default `StreamIsolated` mode) or, under NVIDIA MPS, per-tenant
CUDA contexts (`ContextIsolated` mode).

The threat model the auditor inherits is documented in full in
[`SECURITY.md`](../SECURITY.md). In one paragraph: the adversary controls
the Wasm bytecode and any uploaded PTX, may submit crafted snapshot
files, and may issue crafted HTTP requests; the host kernel, CUDA driver,
and Wasmtime runtime are trusted; the GPU L2 timing side channel is a
known unmitigated gap whose long-term answer is NVIDIA MIG. The audit
firm is asked to validate this model, surface gaps in the defences
described in `SECURITY.md`, and red-team the attack surfaces listed in
§3.

The lifecycle stage at audit time is `v0.5.0-beta`: the API surface and
on-disk formats are frozen pending audit feedback, the v0.4 hardening
wave has landed (rate limit, scoped tokens, audit log, mTLS doc, OpenAPI
CI validation, SBOM, capacity planning), and the v1.0 release is gated
on external validation per [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) v0.5
exit criteria. The runtime is not currently deployed at sponsor scale in
production; the audit drives the v1.0 production-readiness commitment.

The v0.5 audit closes one of two named v0.5 exit criteria. The findings
the firm produces will be published verbatim in
`docs/SECURITY-AUDIT-v0.5.md` under
`accepted / mitigated / rejected / accepted-with-mitigation` per finding,
with sponsor rationale beside each disposition, per the standing
commitment in `PATH-TO-V1.md`. The disposition file is public; the firm's
report is published in full unless the firm requests redaction of a
specific finding for coordinated-disclosure reasons (see §5 on embargo).

---

## 3. Scope of review

### 3.1 In scope — required coverage

The auditor is expected to walk each of the following components and
deliver findings under the categories in §4. Component paths are stable
at the v0.5 tag; the engagement kick-off will pin the exact commit SHA.

| Component | Crate | What to audit |
|---|---|---|
| WASI-GPU host bridge | `crates/tensor-wasm-wasi-gpu/` | Host-function bounds checking; W1.1 typed `(tag, value)` kernel-args lowering in `kernel_args.rs`; pointer translation from guest linear memory to device pointers; kernel-registry lifetime invariants; argv length and per-arg size limits |
| HTTP API gateway | `crates/tensor-wasm-api/` | Bearer auth (`middleware.rs`); W2.1 scoped tokens (`token_scope.rs`); W1.4 per-token rate limit (`rate_limit.rs`); W2.2 audit log (`audit.rs`); W2.3 HTTP request metrics (`http_metrics.rs`); W4.1 W3C trace-context propagator install (`trace_propagation.rs`); body-size caps; tenant header parsing |
| Snapshot subsystem | `crates/tensor-wasm-snapshot/` | zstd decompression bomb cap (`MAX_TOTAL_PAYLOAD_BYTES`); bincode per-field size limits; CRC32 integrity check; schema versioning and rejection of unknown versions; W1.3 cross-version compatibility fixtures; restore-side trust boundary |
| Executor | `crates/tensor-wasm-exec/` | Wasmtime async epoch-based interruption wiring; per-invocation deadline enforcement (`SpawnConfig`); panic containment; instance-store lifetime |
| Memory subsystem | `crates/tensor-wasm-mem/` | `UnifiedBuffer` over `cudaMallocManaged` (cust + cudarc parallel paths); host-fallback `PinnedHostBuffer` with `region`-based guard pages; D1 in-place grow scaffold; C3 tenant accounting (`tenant_gpu_memory_bytes`); isolation level taxonomy (`isolation.rs`) |
| JIT pipeline | `crates/tensor-wasm-jit/` | PTX text generation (`ptx_emit::emit`); BLAKE3 fingerprint cache-key construction and lookup; blueprint detector pure-compute proofs; deopt path; O3 `pliron_dialect` scaffold (in-tree only, no codegen reaches PTX at v0.5) |
| Tenant subsystem | `crates/tensor-wasm-tenant/` | Quota enforcement on instance spawn; MPS context-isolated mode (`ContextIsolated`); tenant registry lifetime and removal; per-tenant metric isolation under W2.3 + C3 |
| Supply chain config | `deny.toml`, `Cargo.lock`, `.github/workflows/audit.yml`, `.cargo/config.toml` | `cargo-deny` allowlist, including the `git`-pinned `NVlabs/cuda-oxide` + `vaivaswatha/pliron` revs documented under F2; B3 `cargo deny check` CI step; SBOM generation pipeline under W4.3 |
| Build reproducibility | `docs/REPRODUCIBLE-BUILDS.md` + tooling | Verifying the reproducible-build claim under W3.6; assessing the toolchain pin policy (W2.9 Wasmtime cadence, F4 workspace bump) for supply-chain implications |

### 3.2 Out of scope — explicitly excluded

The following are not part of the audit deliverable. The auditor may
note observations in these areas as informational findings but is not
asked to spend chargeable hours on them; the sponsor will not credit
findings here against the engagement scope.

| Area | Reason |
|---|---|
| Upstream Wasmtime itself | Bytecode Alliance owns Wasmtime's security posture; TensorWasm wraps and pins a specific release. Findings in Wasmtime are reported upstream by us, not by the firm. |
| Upstream `cust`, `cudarc`, `cuda-oxide`, `cuda-host`, `cuda-core`, `cuda-device`, `cuda-macros`, `cuda-async`, Pliron | Each has its own audit posture; TensorWasm depends. Trust boundary at the FFI shim is in scope (the shim is ours); the upstream crate internals are not. |
| NVIDIA CUDA Driver (`libcuda.so`), CUDA Runtime, `ptxas`, NVIDIA kernel module | NVIDIA-closed source; out of OSS audit scope. The host's interaction with these (argument validation, return-code handling, error containment) is in scope. |
| Guest Wasm modules | User-supplied. Auditing guest code is the user's responsibility; auditing the sandbox the guest runs in is in scope. |
| Physical security of self-hosted CI runners | Operational concern owned by the sponsor and the runner-host operator. The C1 self-hosted CUDA runner registration runbook documents the operational posture. |
| Wasmtime upstream fuzz corpus | Bytecode Alliance runs continuous fuzzing on Wasmtime. We do not ask the firm to re-fuzz Wasmtime; we ask for review of our `fuzz/` targets and their seed corpora (see `fuzz/README.md`). |
| Performance-only findings | Latency or throughput regressions that have no security implication are not in scope. The W4.6 P99.9 latency bench owns performance regressions; the auditor is asked for security findings, not perf findings. |

### 3.3 Specific attack surfaces to red-team

The sponsor expects the proposed methodology in §7 to address each of
the following surfaces explicitly. These are the attacks the sponsor
loses sleep over; a proposal that does not name them is unlikely to be
selected.

1. **Wasm-to-host escape via WASI-GPU bounds checks.** Pointer forgery
   through integer overflow in the W1.1 `kernel_args.rs` argv length
   field; off-by-one in `read_bytes` `(ptr, len)` translation; aliasing
   of a guest pointer to a sibling instance's `UnifiedBuffer`; bypass of
   the `MAX_KERNELS_PER_INSTANCE` (256) or `MAX_PTX_BYTES` (8 MiB) caps.
2. **Cross-tenant data leakage.** Via UVM (CUDA Unified Memory) page
   migration; via the JIT `KernelCache` BLAKE3 fingerprint (collision
   resistance and cache-key ownership — can tenant A construct a Wasm
   module whose cache key collides with tenant B's hot kernel?); via
   trace-context leakage in audit-log records under W2.2.
3. **Snapshot replay attack.** Tampered snapshot bytes restoring a
   different instance's state; CRC32 spoofing through bincode field
   manipulation; version downgrade by replacing the schema version field;
   decompression-bomb amplification beyond the documented
   `MAX_TOTAL_PAYLOAD_BYTES` cap.
4. **HTTP API exploitation.** Bearer-token enumeration via timing side
   channel in the constant-time comparison path; rate-limit bypass by
   varying the token-bucket shard key; audit-log injection (terminal
   escapes, JSON injection in the structured record); OpenAPI spec drift
   between the W4.2 CI-validated spec and the live router; cross-tenant
   access via crafted `X-Tenant` headers under W2.1 scoped tokens.
5. **Supply chain.** Whether the `cargo-deny` allowlist under `deny.toml`
   is sufficient given the F2 git-pinned `NVlabs/cuda-oxide` and
   `vaivaswatha/pliron` revisions; whether the SBOM under W4.3 captures
   the git pins; whether the W3.6 reproducible-build process holds under
   the F4 toolchain bump; whether `cargo audit` + `cargo deny check`
   coverage extends to the cuda-oxide-backend feature set.

The auditor is welcome to surface attack surfaces beyond these five; the
list is a floor, not a ceiling.

---

## 4. Deliverables expected from the auditor

### 4.1 Report

A single written report covering every finding, structured as follows.
The format is fixed; the content is the firm's. Markdown or PDF
acceptable; if PDF, source `.md` or `.tex` also delivered for diffing
across the optional retest in §4.4.

For each finding:

| Field | Description |
|---|---|
| ID | Firm-prefixed identifier (e.g. `[FIRM-PREFIX]-001`) |
| Severity | CVSS 3.1 base score + vector + qualitative label (`CRITICAL` / `HIGH` / `MEDIUM` / `LOW` / `INFO`) |
| Component | Crate path or subsystem name |
| Description | One-paragraph statement of the issue |
| Reproduction | Numbered steps; PoC code where non-trivial (see §4.3) |
| Impact | Concrete consequence in the deployed runtime, not the abstract weakness |
| Exploitability | Attacker prerequisites: local guest, network attacker, privileged operator, etc. |
| Recommended remediation | Specific fix proposal; cite the file and function where applicable |
| References | CWE, prior-art CVEs, upstream issues, related OSS audit reports |

### 4.2 Executive summary

A 2-4 page summary suitable for sponsor leadership (non-security
executives). Includes: the engagement scope as actually walked, the
finding count by severity, the three highest-priority findings in plain
language, and an assessment of whether the v0.5 security posture meets
the bar implied by the v1.0 commitments in
[`docs/PATH-TO-V1.md`](PATH-TO-V1.md).

### 4.3 Per-finding worksheet

A machine-readable table (CSV or JSON) of all findings with the columns
in §4.1 plus an empty `disposition` column. The sponsor populates this
column as `accepted / mitigated / rejected / accepted-with-mitigation`
during remediation; the populated worksheet is published verbatim as
`docs/SECURITY-AUDIT-v0.5.md` per the PATH-TO-V1 v0.5 exit criterion.

### 4.4 Optional retest

A 30-day follow-up retest after sponsor remediates all CRITICAL and HIGH
findings. The retest verifies that remediations close the original
findings and have not introduced regressions in the same attack surface.
The retest produces a delta report against §4.1, with each finding
marked `closed`, `partially-closed`, `regressed`, or `unchanged`.

The sponsor may decline the retest if no CRITICAL or HIGH findings land;
proposals should price the retest separately so the sponsor can decide
after the initial report.

### 4.5 Optional PoC code

For findings whose exploitability is non-trivial, PoC code in Rust,
Python, or shell as appropriate. Delivered in a private repository the
sponsor will provision; not bundled with the public report. The sponsor
runs PoC code only in the non-prod sandbox per §5.

---

## 5. Engagement parameters

| Parameter | Value |
|---|---|
| Audit duration | 4-6 weeks of audit time; sponsor flexible on calendar dates |
| Kick-off | Joint call with sponsor security committee (per [`GOVERNANCE.md`](../GOVERNANCE.md)); commit SHA pinned at kick-off; access provisioned within 5 business days |
| Coordination | Weekly status sync (45 minutes, video, recorded by sponsor); written weekly summary from lead consultant |
| Communication channel | Private Slack or equivalent stood up by sponsor; embargoed findings via PGP-encrypted email to `security@craton.com.ar` (PGP key published under SECURITY.md) |
| Embargo | 90-day coordinated disclosure per [`SECURITY.md`](../SECURITY.md) "Backport policy" and the W5.5 [`docs/runbooks/cve-disclosure-dry-run.md`](runbooks/cve-disclosure-dry-run.md) procedure. The firm's findings flow through the same pipeline as any external CVE report; the sponsor commits to the same 72-hour triage SLO. |
| Access | Read-only GitHub repository access (provisioned by sponsor); non-production sandbox environment for exploitation testing (sponsor provides, mirrors the production reference deployment under W3.8); no production system access requested or granted |
| Hardware | Sponsor provides a self-hosted CUDA runner SKU equivalent to the C1 environment (NVIDIA A100 or H100, CUDA 12.4+, Linux x86_64); the auditor may request a second SKU for cross-validation |
| Public release | Final report published in full at `docs/SECURITY-AUDIT-v0.5.md` after embargo lapses or all HIGH+ findings are remediated, whichever is later. The firm receives credit in the published document; firm logo and lead-consultant name included on request. |
| Liability | Per the firm's standard engagement letter; sponsor's preferred limits are limited liability for direct damages capped at the engagement fee, with no consequential damages |

The disclosure pipeline the firm's findings traverse is rehearsed
quarterly by the sponsor's security committee per the W5.5 dry-run
runbook; the firm should expect the pipeline to behave as documented
when the first real CRITICAL or HIGH finding lands.

---

## 6. Required auditor qualifications

The sponsor will evaluate proposals against the following minimum
qualifications. A firm that cannot demonstrate all four should still
respond; the sponsor will weigh strength in three of four against the
evaluation criteria in §8.

### 6.1 Rust unsafe-code audit experience

Demonstrated experience auditing Rust codebases with non-trivial
`unsafe` blocks. Cite at least one public report (firm-published, OSS
client, or client-published) in which the deliverable was a Rust audit
with `unsafe` findings. Examples of comparable work the sponsor has read
include Trail of Bits' audits of `wasmtime`, `firefox` parts in Rust,
and `solana-program`; NCC Group's audit of `rustls`; Cure53's audits of
Rust HTTP stacks; Doyensec's audits of Rust crypto crates. The sponsor
does not require any specific firm or specific report — these are
calibration examples.

### 6.2 WebAssembly runtime familiarity

Demonstrated familiarity with at least one production WebAssembly
runtime (Wasmtime, Wasmer, Lucet, WAMR, or equivalent). The audit
spends significant time at the Wasmtime ↔ host trust boundary; a
proposal that does not name the firm's WebAssembly experience will
score poorly under §8.

### 6.3 CUDA host-side FFI familiarity

Demonstrated familiarity with the CUDA Driver API (not just the
Runtime API), Unified Memory semantics, MPS context-isolation
properties, and the PTX validation path through `ptxas`. The audit
touches each of these. A proposal that focuses entirely on the HTTP
surface and skips CUDA is incomplete.

### 6.4 OSS-publication reputation

Public-by-default audit reports. The sponsor publishes the report in
full per §5; the firm should be comfortable with that. A firm whose
standard engagement model is private-only deliverables can still
respond, but should price the public-release modifier explicitly.

---

## 7. Proposal response format

Proposals are expected as a single PDF, 10-20 pages, structured as
follows. The sponsor will skim a proposal that does not follow this
structure; the structure exists to make cross-firm comparison fair.

### 7.1 Firm overview

One page. Firm name, founding year, headcount in the Rust + Wasm + CUDA
practice areas, public OSS report archive URL, primary jurisdictions of
operation.

### 7.2 Relevant past engagements

Up to three. For each: client name (or "anonymized — sector"),
engagement scope, deliverable shape, year. The sponsor weights public
reports more heavily than private engagements; an anonymized private
engagement is fine if the work is described in sufficient detail to
evaluate relevance.

### 7.3 Lead consultant CV summary

One paragraph plus a bullet list. The named lead consultant is the
person the sponsor expects to do the majority of the audit work; a
proposal that lists a senior lead and silently substitutes a junior on
the actual engagement will fail the §5 weekly-sync sanity check.

### 7.4 Proposed methodology

2-4 pages. The sponsor expects this section to address each of the five
attack surfaces in §3.3 explicitly, plus the firm's standard
methodology for Rust-unsafe review and WebAssembly host-boundary
review. A proposal that lists "manual code review" and "fuzzing" as the
entire methodology will not be selected.

### 7.5 Proposed schedule

Calendar week-by-week breakdown from kick-off through final delivery.
Includes: kick-off, scope walk, weekly checkpoints, draft report
delivery, sponsor review window, final delivery, optional retest. The
sponsor's preference is a 6-week schedule with draft at week 5 and
final at week 6; a 4-week schedule that the firm is confident in is
also acceptable.

### 7.6 Pricing

Fixed-fee strongly preferred over time-and-materials. If the firm
prefers T&M, include a not-to-exceed cap and the rate sheet. Itemise:

- Base audit
- Executive summary
- Per-finding worksheet (§4.3) — typically rolls into base
- PoC code (§4.5) — per finding or bundled
- Retest (§4.4) — separately priced so sponsor can decline
- Travel, if any (sponsor does not require on-site presence; remote
  engagement is the default)

All prices in USD. The sponsor's currency-conversion handling for non-USD
quotes is per §9.

### 7.7 References

At least two references from prior comparable OSS engagements. References
should be named contacts at the client side who agreed to discuss the
firm's work; the sponsor will reach out to each before final selection.

---

## 8. Evaluation criteria

The sponsor's selection committee scores each proposal against the
following weights. Total is 100. The committee notes are recorded but
not published; the selected firm is named publicly in
`docs/SECURITY-AUDIT-v0.5.md` along with the report.

| Criterion | Weight | What gets scored |
|---|---|---|
| Relevant past work in Rust + Wasm + GPU | 35% | §7.2 engagement list; §6.1 / §6.2 / §6.3 qualifications; reviewer's read of the firm's public report archive |
| Proposed methodology and depth | 25% | §7.4 methodology; whether the five attack surfaces in §3.3 are addressed explicitly; whether the methodology mentions the W4.7 fuzz targets as a starting point |
| Schedule fit | 15% | §7.5 schedule; alignment with the sponsor's `[ENGAGEMENT-START-DATE]` and v0.5-beta cycle |
| Price | 15% | §7.6 pricing; value for scope, not cheapest-wins |
| References | 10% | §7.7 references; the sponsor's reference-check conversations |

A proposal scoring below 50 in any single category is unlikely to be
selected regardless of total. The sponsor will respond to every firm
that submitted a proposal within 14 days of the §1 deadline, with a
short rationale for the selection or non-selection.

---

## 9. Sponsor information

| Field | Value |
|---|---|
| Sponsor entity | Craton Software Company |
| Security contact (technical) | `security@craton.com.ar` |
| Procurement contact | `[SPONSOR-PROCUREMENT-CONTACT]` |
| Sponsor jurisdiction | `[SPONSOR-JURISDICTION]` |
| Tax treatment | `[SPONSOR-TAX-TREATMENT]` (e.g. VAT/IVA handling for non-domestic firms) |
| Currency-conversion policy | Quotes in non-USD currencies are evaluated at the spot rate on the §1 deadline date |
| Maximum budget | `[SPONSOR-BUDGET-CEILING]` USD. The sponsor's planning range for this scope is USD 40,000 - 120,000; the actual ceiling is a sponsor-procurement decision and is filled in before sending. |
| Payment terms | `[SPONSOR-PAYMENT-TERMS]` (e.g. 50% on kick-off, 50% on final delivery; or per firm's standard terms) |
| Legal review | Sponsor's legal counsel reviews the firm's engagement letter before signature; turnaround target 5 business days |

---

## 10. Submission instructions

Email a single PDF to `security@craton.com.ar` by `[PROPOSAL-DEADLINE-DATE]`
UTC end-of-day. Subject line: `RFP response — [FIRM-NAME] — Craton TensorWasm v0.5 audit`.

PGP encryption optional but supported; the sponsor's PGP key is published
in [`SECURITY.md`](../SECURITY.md) under "Reporting vulnerabilities".

The sponsor treats all received proposals as confidential and will not
share them outside the selection committee (sponsor security committee
plus the sponsor procurement contact). Unsuccessful firms' proposals are
deleted within 90 days of selection per the sponsor's standard
record-retention policy.

Late proposals are evaluated only if the sponsor has not yet selected a
firm. The sponsor will acknowledge receipt of every submitted proposal
within 2 business days; non-acknowledgement after 5 business days is a
delivery failure — resend.

Questions about the RFP can be sent to the same address. The sponsor will
publish anonymized answers to substantive questions in an appendix to
the public report after selection, so that future RFPs benefit from the
clarification.

---

## Appendix A — relevant prior documents

The auditor is encouraged to read the following before drafting the
methodology in §7.4. Each is committed to the repository at the same
commit SHA the audit will pin.

| Document | What it covers |
|---|---|
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Workspace layout, ten-crate dependency graph, build matrix, feature-flag interactions for `cudarc-backend` / `cuda-oxide-backend` / `otlp` |
| [`SECURITY.md`](../SECURITY.md) | Threat model, defences, isolation-level taxonomy, known gaps (incl. GPU L2 timing side channel), authentication, vulnerability reporting, backport policy |
| [`docs/SECURITY-AUDIT.md`](SECURITY-AUDIT.md) | Prior internal audit at v0.1.0 — methodology, findings (BA-001 through BA-008 with disposition), audit checklist, fuzz coverage. The v0.5 external audit builds on this foundation; the firm should review which v0.1 findings landed `Fixed` versus `Documented gap`. |
| [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) | The v0.5 exit criterion this audit satisfies; the v1.0 commitments the audit feeds into; the open decisions still under consideration (incl. Open Decision #5, "External auditor for v0.5 review", which this RFP resolves) |
| [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md) | The v0.5 decision context for the CUDA backend selection; the firm should be aware that three backends ship in v0.5 (`cust` default, `cudarc-backend` opt-in, `cuda-oxide-backend` opt-in) with the default-flip held to v0.5 contingent on cuda-oxide v0.2 stability |
| [`fuzz/README.md`](../fuzz/README.md) | Existing `cargo-fuzz` targets: `fuzz_wasm_compile`, `fuzz_ptx_emit`, `fuzz_snapshot_restore`, `fuzz_wasi_cuda_abi`, `token_scope_parser`, `audit_json_round_trip`. The firm is asked to review the targets' invariants and seed corpora, and to propose additions where coverage is weak. |
| [`docs/runbooks/cve-disclosure-dry-run.md`](runbooks/cve-disclosure-dry-run.md) | The W5.5 dry-run procedure the firm's CRITICAL and HIGH findings will flow through. Rehearsed quarterly by the security committee; the firm should expect the documented timings (72-hour triage SLO, 90-day fix-or-workaround commitment) to hold in practice. |
| [`GOVERNANCE.md`](../GOVERNANCE.md) | Sponsor governance model, security committee composition and standing commitments, lead maintainer role, RFC amendment procedure |
| [`MAINTAINERS.md`](../MAINTAINERS.md) | Current maintainer roster, security committee roster, area ownership (the firm coordinates with crate owners for area-specific findings) |
| [`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md) | HTTP API wire format; the OpenAPI spec validated in CI under W4.2 lives alongside this doc |
| [`docs/AUDIT-LOG.md`](AUDIT-LOG.md) | W2.2 audit log schema and the actor/action/resource/outcome/latency fields; relevant to attack surface §3.3 finding-class 4 (audit-log injection) |
| [`docs/MPS-SETUP.md`](MPS-SETUP.md) | NVIDIA MPS deployment for `ContextIsolated` mode; relevant to attack surface §3.3 finding-class 2 (cross-tenant data leakage) |
| [`docs/REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md) | W3.6 reproducible-build claim and verification procedure; relevant to attack surface §3.3 finding-class 5 (supply chain) |
| [`docs/SBOM.md`](SBOM.md) | W4.3 SBOM generation pipeline; the firm verifies that git-pinned dependencies (cuda-oxide, Pliron) are captured |
| [`deny.toml`](../deny.toml) | `cargo-deny` allowlist, including F2 git-pin exceptions |

---

## Appendix B — fill-in checklist

A maintainer preparing this RFP for sending must complete each of the
following placeholders. The `[bracketed-with-dashes]` convention is
deliberate so each can be grep-replaced.

| Placeholder | Section | Notes |
|---|---|---|
| `[RFP-ISSUE-DATE]` | §1 | The date this RFP is sent to the named firm |
| `[PROPOSAL-DEADLINE-DATE]` | §1, §10 | 4 weeks after issuance; same date in both sections |
| `[ENGAGEMENT-START-DATE]` | §1 | 8-12 weeks after issuance |
| `[FIRM-NAME]` | §1, §10 | Per-firm; one RFP per firm |
| `[FIRM-BUSINESS-DEVELOPMENT-CONTACT]` | §1 | Named contact at the firm; obtained by sponsor outreach before RFP send |
| `[SPONSOR-PROCUREMENT-CONTACT]` | §1, §9 | Name + email of sponsor procurement lead |
| `[SPONSOR-JURISDICTION]` | §9 | Legal jurisdiction for the engagement contract |
| `[SPONSOR-TAX-TREATMENT]` | §9 | Tax handling for non-domestic firms (VAT/IVA, withholding) |
| `[SPONSOR-BUDGET-CEILING]` | §9 | Maximum USD figure; sponsor-procurement decision |
| `[SPONSOR-PAYMENT-TERMS]` | §9 | Payment schedule |
| `[FIRM-PREFIX]` | §4.1 | Optional — auditor's standard finding-ID prefix; left blank if the firm uses one by default |

Eleven placeholders total. A maintainer should grep the prepared file
for `[` before sending to confirm none remain.

---

_Document status: template, ready to populate. The most recent
modification is on the cover (§1) when a maintainer fills in the per-firm
fields. The body sections (§2 through Appendix B) should not change
between firms — fairness of evaluation depends on every firm receiving
the same scope and the same criteria._
