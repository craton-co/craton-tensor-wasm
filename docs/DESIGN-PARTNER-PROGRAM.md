<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Design Partner Program (v0.5)

A reciprocal engagement kit for organizations willing to run TensorWasm
v0.5-beta in production for one calendar month, in exchange for early
access, a roadmap voice, and named (or anonymized) credit in the v1.0
release notes. This document is the offer your sponsor representative
hands to your engineering lead. Read it end-to-end and decide in one
sitting whether the trade is worth a month of your SRE team's
attention.

The program closes one of two named v0.5 exit criteria on the project's
path to v1.0 — see [`PATH-TO-V1.md`](PATH-TO-V1.md) v0.5 §"External
validation" and open decision #6, "Production design partners". The
other v0.5 criterion is the external security audit
([`SECURITY-AUDIT-RFP.md`](SECURITY-AUDIT-RFP.md)); the two land in
parallel.

---

## Contents

1. [One-page pitch](#1-one-page-pitch)
2. [What you get](#2-what-you-get)
3. [What we ask](#3-what-we-ask)
4. [Partner-fit qualifications](#4-partner-fit-qualifications)
5. [Engagement timeline](#5-engagement-timeline)
6. [Reciprocal commitments (contract)](#6-reciprocal-commitments-contract)
7. [Anonymization policy](#7-anonymization-policy)
8. [Selection process](#8-selection-process)
9. [FAQ](#9-faq)
10. [Application template](#10-application-template)
11. [Appendix — documents you will consume](#11-appendix--documents-you-will-consume)

---

## 1. One-page pitch

**What this program is.** A formal 8-week engagement in which one to three external organizations deploy TensorWasm v0.5-beta into a real production (or staging-with-real-load) environment, run it for a calendar month under the published SLOs in [`SLO.md`](SLO.md), and exchange weekly telemetry + issue feedback for early access, roadmap influence, and named credit in the v1.0 release notes. Reciprocal, not paid. Runs against a frozen v0.5-beta commit, ends with a retrospective + partner-shareable case study.

**What we offer.** Early access to v0.5-beta four to six weeks before public cut. A dedicated maintainer-to-partner channel with a 24-hour business-day response SLA. A free 2-3 hour architecture review session before you go live, plus pre-baked deploy artifacts (Helm values, k8s manifests, Nomad job spec) templated to your environment.

**What we ask.** Run v0.5-beta for one calendar month with no severity-1 incidents (data loss, security regression, complete outage). A weekly 15-30 minute check-in (sync or async) sharing latency, GPU memory, and incident summary against [`SLO.md`](SLO.md). File issues for every reproducible bug, observability gap, or doc that confused you. Permission to cite the engagement in v1.0 release notes — named or anonymized, your choice, fixable up to release day.

**30-second elevator pitch (for your VP / SRE lead):**

> An open-source GPU-WASM runtime (Apache-2.0, sponsored by Craton Software Company) is recruiting up to three external production users for a one-month validation engagement before its v1.0 cut. They ship us the beta four to six weeks early, an architecture review, and a direct line to the maintainers; we run it on real workload for a month, file issues, share telemetry, and either get named in the v1.0 release notes or stay anonymous (our pick). Free. No SaaS lock-in; Apache-2.0; we keep what we deploy.

---

## 2. What you get

Eight concrete deliverables. Every item is a sponsor commitment that lives or dies in [§6](#6-reciprocal-commitments-contract).

**2.1 Early access to v0.5-beta.** Container image, Helm chart, CLI binary, and snapshot format four to six weeks before public cut. The commit SHA is frozen at Week 0 and held stable for the month; beta-cycle bug fixes you flag are cherry-picked onto a partner-tagged build the sponsor maintains for you.

**2.2 Dedicated maintainer channel.** A private Slack channel (or email thread, or Discord — your choice) with the sponsor's on-call maintainer rep. Response SLA: **24 hours during sponsor business days** (Mon-Fri, UTC-3 office hours; nights/weekends best-effort). Persists Week 0 through Week 8; archived to your possession afterwards.

**2.3 Named or anonymized credit.** In v1.0 release notes, named ("Production validation: [Your Org]. Thank you.", optional one-paragraph quote) or anonymous ("Production validation: a partner whose name is withheld at their request."). Your choice at engagement start, changeable up to the day v1.0 ships. See [§7](#7-anonymization-policy).

**2.4 Roadmap voice.** For the engagement duration, you have **first refusal** on RFCs that affect your declared use case. Any RFC landing in [`rfcs/`](../rfcs/) (see [`rfcs/README.md`](../rfcs/README.md)) that touches a subsystem you depend on is shared at draft stage; your objections are blocking until resolved on the RFC thread or escalated to a maintainer vote under [`GOVERNANCE.md`](../GOVERNANCE.md) §"Voting rules". Not a veto — only the three grounds in `GOVERNANCE.md` §"Vetoes" qualify — but your objection cannot be merged-over without explicit maintainer response.

**2.5 Priority bug fixes during the engagement.** Severity-1 bugs you discover Week 0 - Week 8 are queued ahead of the maintainer backlog: written triage ack within 24 business hours, documented mitigation or patch within the timeline in [`SECURITY.md`](../SECURITY.md) §"Backport policy" (14-day patch cadence for ordinary severity-1; 90-day fix-or-workaround for security-class). See [§6.4](#64-the-severity-1-timeline-discussion) for the explicit framing.

**2.6 Free architecture review session.** Before production cutover, a 2-3 hour working session with the relevant maintainers covering your workload shape, the deployment topology drafted from [`tutorials/production-deployment.md`](tutorials/production-deployment.md), the values you chose against [`CAPACITY-PLANNING.md`](CAPACITY-PLANNING.md) formulas, open RFC questions intersecting your use case, and the runbook set in [`runbooks/`](runbooks/). Output: a one-page written summary within five business days. Yours; not published.

**2.7 Pre-baked deploy artifacts.** Templated to your environment: a Helm `values.yaml` (from [`deploy/helm/tensor-wasm/`](../deploy/helm/tensor-wasm/)) with your token allowlist, GPU node selectors, ingress class, and Prometheus scrape labels; k8s manifests (`Deployment`, `Service`, `ServiceMonitor`, `NetworkPolicy`) from [`deploy/k8s/`](../deploy/k8s/) for non-Helm partners; a Nomad HCL2 job spec from [`deploy/nomad/`](../deploy/nomad/); a worked-example `Dockerfile` selecting your preferred backend (`cust`, `cudarc`, or `cuda-oxide`). Apache-2.0; modify freely.

**2.8 Bench + SLO baseline against your workload.** Before cutover, the sponsor runs the in-repo Criterion suite (see [`BENCHMARKING.md`](BENCHMARKING.md)) plus a synthetic load matched to your declared traffic envelope, producing a per-route latency baseline (P50, P95, P99, P99.9) recorded against the SLO thresholds in [`SLO.md`](SLO.md) §3, a capacity-planning recipe per [`CAPACITY-PLANNING.md`](CAPACITY-PLANNING.md) §4 with tenants-per-host curves at your target QPS, and a `bench-results/partner-<engagement-id>.json` overlay you can rerun yourself to detect drift across the month. Delivered privately; anonymized aggregates may be published per [§3.6](#36-permission-to-publish-anonymized-bench-numbers).

---

## 3. What we ask

Seven concrete asks. Every item is a partner commitment that lives or dies in [§6](#6-reciprocal-commitments-contract). The asymmetry (8 sponsor deliverables vs 7 partner asks) is intentional — production validation is what we cannot generate ourselves.

**3.1 Deploy v0.5-beta to a production (or realistic-load-test) environment.** Must run real workload or a high-fidelity load test that exercises the same code paths as production. A demo in a sandboxed namespace with synthetic curl traffic does not count. Acceptable substitute: shadow-traffic mirroring (production traffic copied onto the TensorWasm deployment with responses discarded) sustained for the full month.

**3.2 Run for a full calendar month with no severity-1 incidents.** The 30-day clock starts on the day you cut production traffic to v0.5-beta. **Severity-1** is defined per [`SECURITY.md`](../SECURITY.md) §"Backport policy":

- **Data loss.** Snapshot corruption, tenant memory overwrite, silently dropped writes, audit-log torn segment across a restart.
- **Security regression.** Any breach of the isolation boundaries documented in [`SECURITY.md`](../SECURITY.md) §"Defences".
- **Complete service outage.** Runtime fails to start, panics on first request, or hangs indefinitely under expected load (`availability_http` = 0 for >5 minutes).

A severity-1 inside the window does not invalidate the engagement: a maintainer-side RCA is triggered, the 30-day clock resets from the day the fix lands on your build, and you remain a design partner. The engagement only ends without a v1.0 citation if you withdraw or the sponsor cannot close the root cause within the engagement window.

**3.3 Weekly check-in.** A 15-30 minute sync (or async, in writing) every calendar week Week 1 - Week 6, reporting:

- **Latency** P50, P95, P99 for `/healthz` and `/functions/:id/invoke`, against the SLOs in [`SLO.md`](SLO.md) §3. Paste the PromQL output; no pretty graphs needed.
- **GPU memory utilization** per tenant, from the `tensor_wasm_gpu_memory_bytes_per_tenant` series documented in [`SLO.md`](SLO.md) §8.
- **Incidents** in the past week: severity, root cause (if known), time-to-recover, link to your internal post-mortem if you have one.
- **Feature gaps observed** — things you wanted that the runtime did not give you, so the maintainers can decide v0.5-vs-v1.0-vs-v2.0.

A single weekly email or PR comment satisfies this ask.

**3.4 File issues.** For every reproducible bug, every observability gap, every doc that confused your team: open a GitHub issue. Triaged within 24 business hours. The bar is intentionally low: "I read [`tutorials/production-deployment.md`](tutorials/production-deployment.md) §5 and got stuck because the values.yaml example uses a TLS field that doesn't exist" is a good issue. Bugs containing your production secrets, customer names, or internal IP go to the maintainer channel from [§2.2](#22-dedicated-maintainer-channel); the sponsor sanitizes and files on your behalf.

**3.5 Permission to cite the engagement in v1.0 release notes.** Named or anonymously, your choice at engagement kickoff and re-confirmable in writing up to the day v1.0 ships. See [§7](#7-anonymization-policy). The permission is to cite the engagement, not to endorse the product; even an anonymous mention requires this grant.

**3.6 Permission to publish anonymized bench numbers.** The baseline from [§2.8](#28-bench--slo-baseline-against-your-workload) contains performance data the project would like to publish stripped of identifying detail — workload shape as "ML inference at ~N req/s, P95 input size ~K bytes" rather than "Acme Corp's chatbot moderation pipeline". Specific revenue, customer counts, internal SLAs, and identifying details are never published. Declining does not disqualify you; the sponsor simply does not publish numbers from your engagement.

**3.7 Acknowledge the embargo discipline.** Any defect, performance regression, or unreleased feature you observe in v0.5-beta that has not been announced publicly is embargoed until coordinated disclosure, mirroring the maintainer embargo in [`GOVERNANCE.md`](../GOVERNANCE.md) §"Security disclosures". No blog posts, conference talks, vendor comparisons, or social-media discussion of unreleased v0.5-beta behaviour until the corresponding v0.5.0 or v1.0 release notes are public.

---

## 4. Partner-fit qualifications

The hard filter the sponsor applies during selection. Self-screen
before applying.

### 4.1 We are looking for

- **Serverless or batch workloads that would benefit from GPU
  acceleration.** ML inference, real-time inference, video/image
  processing pipelines, scientific compute, RAG embedding generation,
  or anything else that already runs untrusted-ish code against a
  fleet of GPUs. A CPU-only workload does not exercise the parts
  v0.5-beta needs validated.
- **At least one in-house SRE or platform engineer** comfortable with
  Helm, Prometheus, kubectl, and PromQL. The 2-3 hour architecture
  review is designed against an SRE-level audience.
- **Production traffic levels at or above 100 req/sec aggregate**
  across the TensorWasm deployment. Below that, the SLO data is too
  sparse to be statistically meaningful — 100 req/sec for 30 days is
  ~260,000,000 requests, which gives the burn-rate alerts in
  [`SLO.md`](SLO.md) §5 enough signal to fire (or not) meaningfully.
- **Linux x86_64 hosts with NVIDIA GPUs** in the SM matrix supported
  by v0.5: see [`CUDA-SETUP.md`](CUDA-SETUP.md) §"SM-level
  compatibility matrix" for the exact compute-capability table. Both
  datacenter SKUs (H100, A100, L4, A10) and workstation SKUs (RTX
  3000/4000 series) are in scope.
- **A willingness to share telemetry** at the depth in
  [Section 3.3](#33-weekly-check-in) (latency histograms, GPU memory
  per tenant, incident summaries). Partners who can share only "it
  worked, mostly" should apply for the v0.6 program later instead.

### 4.2 We are NOT looking for

**Pilot-of-pilot organizations with no production intent.** The
single most common false-positive in early-access programs is the
"we'd love to evaluate it but our path to production runs through six
months of internal review" applicant. The program is sized for
organizations that already deploy infrastructure changes weekly or
faster, that already operate at least one GPU-adjacent service in
production, and that have an identified, named project for which
v0.5-beta is the candidate. If your organization needs a six-month
procurement review to deploy an Apache-2.0 binary, the v0.5 program
is the wrong fit; v1.0 GA is the right one.

Additionally not in scope for v0.5: Windows hosts (WDDM driver-time
accounting issues — see [`CUDA-SETUP.md`](CUDA-SETUP.md)); macOS
hosts (no CUDA); AMD or Intel GPUs (v2 scope per
[`PATH-TO-V1.md`](PATH-TO-V1.md) §"Anti-goals"); single-tenant
trusted deployments running only your own code (works fine, but does
not exercise multi-tenant isolation).

### 4.3 Borderline cases — apply anyway

If you are close on traffic volume (50-100 req/sec) but exercise the
dispatch path heavily, apply. If you are on AMD but the CPU-only
host-only path is your actual production target, the engagement
still exercises WASM execution and snapshot subsystems usefully. If
you are on a non-stock NVIDIA SKU not listed in
[`CUDA-SETUP.md`](CUDA-SETUP.md), apply.

---

## 5. Engagement timeline

Eight weeks, gated. Each gate is a "stop here unless the previous
gate is closed" checkpoint, not a hard calendar deadline. Slipping a
gate by a week is acceptable; skipping a gate is not.

| Week | Phase | Activities |
|---|---|---|
| 0 | Kickoff | Kickoff call (60-90 min) with maintainer team + your eng lead + SRE. Deploy-artifact handoff per [§2.7](#27-pre-baked-deploy-artifacts). Architecture review scheduled per [§2.6](#26-free-architecture-review-session). Maintainer channel opened. v0.5-beta commit SHA frozen and recorded in the MOU. |
| 1 | Staging deploy + debugging | You deploy v0.5-beta to staging. Bench + SLO baseline session per [§2.8](#28-bench--slo-baseline-against-your-workload). Architecture review held. First weekly check-in. |
| 2-5 | Production deployment + weekly check-ins | Production cutover at start of Week 2 (your choice of day). Weekly check-ins per [§3.3](#33-weekly-check-in). The 30-day "no severity-1" clock from [§3.2](#32-run-for-a-full-calendar-month-with-no-severity-1-incidents) starts on cutover. |
| 6 | Retrospective | Final weekly check-in. Retrospective call (60-90 min) walking through every issue you filed. Sponsor drafts the partner-shareable case study (1-2 pages) within five business days. |
| 7 | Case-study review | You review the case-study draft and edit freely. Sponsor sends the v1.0 release-note line (named or anonymized) for your sign-off. Sign-off deadline: end of Week 7. |
| 8 | v1.0 release notes ship | v1.0 release notes cite the engagement per the language signed off in Week 7. Maintainer channel archived to your possession. Engagement formally closed. |

**Calendar drift.** Week numbers are nominal. Real engagements drift
by one or two weeks. The contract is the gate ordering, not the
calendar; if your 30-day clock has only run 7 days at Week 5, the
engagement extends.

---

## 6. Reciprocal commitments (contract)

The contractual core. The MOU signed at Week 0 cites this section by
reference.

### 6.1 Sponsor commits to

1. **24-hour business-day response SLA** on the dedicated channel
   from [§2.2](#22-dedicated-maintainer-channel). Acknowledge-within-24-hours
   is the commitment; resolve-within-24-hours is not.
2. **Severity-1 triage acknowledgment within 24 business hours** of
   filing.
3. **Severity-1 fix or documented workaround** within the
   [`SECURITY.md`](../SECURITY.md)-published timelines (see
   [§6.4](#64-the-severity-1-timeline-discussion) for the explicit
   framing — this program does NOT promise tighter SLAs than the
   published policy).
4. **Free architecture review session** per [§2.6](#26-free-architecture-review-session),
   with the one-page written summary delivered within five business
   days of the session.
5. **Pre-baked deploy artifacts** per [§2.7](#27-pre-baked-deploy-artifacts).
6. **Bench + SLO baseline** per [§2.8](#28-bench--slo-baseline-against-your-workload).
7. **Named or anonymized credit** per [§7](#7-anonymization-policy),
   anonymization revocable only by your written permission.
8. **Roadmap voice** per [§2.4](#24-roadmap-voice).
9. **Partner-shareable case-study draft** within five business days
   of the Week 6 retrospective.

### 6.2 Partner commits to

1. **Production (or realistic-load-test) deployment** per
   [§3.1](#31-deploy-v05-beta-to-a-production-or-realistic-load-test-environment).
2. **30 days of operation** under [§3.2](#32-run-for-a-full-calendar-month-with-no-severity-1-incidents).
3. **Weekly check-in** per [§3.3](#33-weekly-check-in), Weeks 1-6.
   Asynchronous form acceptable; skipping a week is not.
4. **Issue filing** per [§3.4](#34-file-issues) for every
   reproducible bug, observability gap, or confusing doc.
5. **Citation permission** for v1.0 release notes per
   [§3.5](#35-permission-to-cite-the-engagement-in-v10-release-notes),
   named or anonymized at your option.
6. **Anonymized bench publication permission** per
   [§3.6](#36-permission-to-publish-anonymized-bench-numbers),
   declinable without engagement penalty.
7. **Embargo discipline** for unannounced v0.5-beta findings per
   [§3.7](#37-acknowledge-the-embargo-discipline).

### 6.3 Mutual commitments

- **(a) Confidentiality of unannounced findings** until coordinated
  disclosure, mirroring the maintainer embargo in
  [`GOVERNANCE.md`](../GOVERNANCE.md) §"Security disclosures".
- **(b) Apache-2.0 license respected on both sides.** The runtime,
  the deploy artifacts, any sponsor patches, and any contributions
  you push back to the project (inbound=outbound DCO per
  [`CONTRIBUTING.md`](../CONTRIBUTING.md)) are all Apache-2.0.
- **(c) Good-faith effort.** Neither side is liable for missed
  commitments caused by reasonable circumstances. The MOU is not a
  SaaS contract; if things go wrong, both sides talk before either
  walks.
- **(d) No fee in either direction.** Reciprocal, not paid.
  Commercial support is a separate arrangement
  ([§9](#9-faq) item 3).

### 6.4 The severity-1 timeline discussion

This program does **not** promise "severity-1 patch within 7 days"
unconditionally. The sponsor-side commitment matches the published
backport policy in [`SECURITY.md`](../SECURITY.md):

- **Security-class severity-1** (matching
  [`SECURITY.md`](../SECURITY.md) §"What backports cover"
  item-security): 72-hour acknowledgment, 90-day fix or documented
  workaround.
- **Non-security severity-1 backport-eligible fix**: cherry-picked
  to the maintenance branch alongside main, patch release cut
  within 14 days of the confirmed backport-eligible merge.

Promising a faster SLA than the published policy would create a tier
of partner who gets better SLAs than ordinary users — which corrodes
the trust the published policy is built on, and is unsustainable
past the first few partners. What this program **does** promise is
*priority queue position*: your severity-1 issues are triaged ahead
of the ordinary backlog, and the maintainer team coordinates the fix
actively with you rather than passively waiting for triage. Within
the `SECURITY.md` envelope, this is meaningfully faster than the
non-partner default; outside the envelope, it is not. A partner who
needs a tighter contractual SLA than [`SECURITY.md`](../SECURITY.md)
is having a commercial-support conversation
([§9](#9-faq) item 3), not a design-partner-program conversation.

This nuance is flagged explicitly because the program PATH-TO-V1
framing left it ambiguous; the sponsor representative who hands you
this kit **must confirm the [`SECURITY.md`](../SECURITY.md)-policy
framing at kickoff, on the record**, as part of the v0.5 sign-off
referenced in [`PATH-TO-V1.md`](PATH-TO-V1.md) §"Open decisions"
item 6.

---

## 7. Anonymization policy

**Default: named.** When you sign the MOU, the default is that v1.0
release notes will name your organization. Recorded in the
application form ([§10](#10-application-template)).

**Switch to anonymized: any time pre-v1.0 release.** You may opt to
anonymized credit any time from engagement kickoff through the day
v1.0 ships. The sponsor switches the release-note language on
receipt of your written request — email to the maintainer channel
suffices. No questions asked; no penalty.

**Sponsor commits to never identifying an anonymized partner without
written permission.** Once anonymized, your organization is not
named in release notes, blog posts, conference talks, vendor
comparisons, the sponsor's website, customer references, the v2
design-partner program's "previous partners" list, or anywhere else,
indefinitely. The commitment outlives the engagement.

**Aggregated anonymized statistics are permitted.** Sentences like
"1 of 3 v0.5 design partners ran on Kubernetes" or "the program
covered ~10,000 cumulative tenant-hours" are about the program, not
about you, and are not gated by your anonymization choice — provided
no statement is granular enough to identify a specific partner. The
sponsor judges granularity in good faith; if you flag a statistic
as too granular, the sponsor amends.

**The named-vs-anonymized choice is not a tier.** Anonymized
partners get exactly the same deliverables in [§2](#2-what-you-get)
as named partners. The only difference is the release-note text.

---

## 8. Selection process

A lightweight pipeline; the bottleneck is the sponsor's review
bandwidth, not paperwork.

**8.1 Apply.** Email `security@craton.com.ar` with the completed
application template from [§10](#10-application-template). The single
mailbox is intentional during the v0.x window (see
[`MAINTAINERS.md`](../MAINTAINERS.md) §"Contact"). Subject line:
`Design Partner Program: <your-org-name>`.

**8.2 Sponsor responds within 7 business days** with one of:

- **Accepted.** Proposed engagement kickoff date (within 30 days),
  draft MOU citing this document, and the Week 0 kickoff call
  calendar.
- **Waitlisted.** Good fit; program at capacity. Sponsor will
  revisit when a slot opens or invite you to v0.6 later.
- **Declined.** Feedback letter explaining the fit gap referencing
  [§4](#4-partner-fit-qualifications). Final for v0.5; you remain
  eligible for v0.6 and v2.

**8.3 If accepted: MOU, kickoff, engagement.** MOU is a one-page
document citing this kit. Engagement starts within 30 days. Week 0
kickoff happens on the engagement start date.

**8.4 If declined.** Honest, short feedback letter. Typical reasons:
traffic volume below [§4.1](#41-we-are-looking-for) threshold by
enough that SLO data would be noise; platform mismatch (Windows /
macOS / AMD); procurement-review timeline too long for the 8-week
window; program full. A declined applicant may reply with new
information; the sponsor will reconsider once.

---

## 9. FAQ

**1. We're not ready for production yet — can we still join?** No,
not for v0.5. The program exists to validate production behaviour;
staging-only does not exercise what needs validating. Flag the
sponsor for the v0.6 or v1.x design-partner program.

**2. We use Windows / macOS / AMD GPUs.** No. Windows is verified
for build but not a v0.5 production target (WDDM accounting issues
per [`CUDA-SETUP.md`](CUDA-SETUP.md)); macOS has no CUDA; AMD is v2
scope. If you operate Linux NVIDIA infrastructure separately, apply
against that fleet specifically.

**3. Can we pay for priority support beyond the program?** The
program itself has no paid tier. Commercial support is a separate
arrangement with **Craton Software Company**, the project's
commercial sponsor (see [`MAINTAINERS.md`](../MAINTAINERS.md)
§"Lead maintainer" and [`GOVERNANCE.md`](../GOVERNANCE.md)
§"Roles"). Contact `security@craton.com.ar` and ask for the
commercial-support contact.

**4. What happens after the engagement ends?** You are cited (named
or anonymized per [§7](#7-anonymization-policy)) in v1.0 release
notes. You may apply for the v2 design-partner program when it
opens; a prior v0.5 partner has a fast-track in v2 selection (not a
guaranteed slot). The maintainer channel is archived to your
possession; the runtime, deploy artifacts, and bench baseline are
yours under Apache-2.0.

**5. Is there a fee?** No. The engagement is reciprocal; neither
side invoices. The sponsor gets production feedback; the partner
gets early access, roadmap voice, and named (or anonymized) credit.
Anything involving money is commercial-support under FAQ 3.

**6. What if we discover a security vulnerability in v0.5-beta?**
Report it through the standard disclosure path in
[`SECURITY.md`](../SECURITY.md) §"Reporting vulnerabilities", **not**
the design-partner channel. The 72-hour committee acknowledgment and
the 90-day fix-or-workaround commitment apply identically to partner
and non-partner reports. You receive disclosure credit per the
coordinated-disclosure preference, in addition to your
design-partner credit.

**7. What if a severity-1 bug is discovered (by anyone) mid-engagement?**
You are notified through the maintainer channel, your engagement
clock pauses, and a patched build lands on your partner tag per the
timeline in [§6.4](#64-the-severity-1-timeline-discussion). The
pause does not invalidate the engagement; the 30-day clock resumes
when the patched build is in production.

**8. Can multiple teams in our organization participate as one
partner?** Yes, if the deployments are coordinated under a single
engineering-lead contact and the weekly check-in aggregates across
them. One contact for the channel; data from any number of
deployments under that contact.

---

## 10. Application template

Copy from `>>> BEGIN >>>` to `<<< END <<<` into an email to
`security@craton.com.ar` with subject `Design Partner Program:
<your-org-name>`. Fill every field; "n/a" is acceptable where a field
genuinely does not apply.

```
>>> BEGIN >>>

Subject: Design Partner Program: <your-org-name>

A. Identification
-----------------

Organization name:        ___________________________________________
Organization URL:         ___________________________________________
Primary contact name:     ___________________________________________
Primary contact email:    ___________________________________________
Primary contact role:     ___________________________________________
Backup contact name:      ___________________________________________
Backup contact email:     ___________________________________________

B. Workload
-----------

Workload description (one paragraph; what does TensorWasm run for
you, what are your tenants, what are inputs and outputs):

___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

Current production traffic level (aggregate req/sec):
  Peak:    ____________ req/s
  Median:  ____________ req/s

Number of tenants (current):           ____________
Number of tenants (target at v0.5):    ____________

GPU fleet inventory:
  SKU(s):                _______________________________________
  Count per host:        _______________________________________
  Host count:            _______________________________________
  Driver version:        _______________________________________
  CUDA toolkit version:  _______________________________________

Target deployment platform (Helm / k8s manifest / Nomad / other):
  ___________________________________________________________________

C. Engagement
-------------

Target v0.5-beta deployment date:                _____________________

Named or anonymized preference (changeable any time before v1.0):
  _____________________

If named: one-sentence draft of release-note line you would prefer
(sponsor may edit; final language is signed off in Week 7):

___________________________________________________________________
___________________________________________________________________

D. Anything specific you want from the program beyond §2 standard
deliverables:

___________________________________________________________________
___________________________________________________________________

E. Confirmations (please mark [x] for each)
-------------------------------------------

[ ] I have authority to commit my organization to the partner
    commitments in §6.2.
[ ] My organization grants permission to cite the engagement in
    v1.0 release notes per §3.5 (named or anonymized per Section C).
[ ] My organization grants permission to publish anonymized bench
    numbers per §3.6 (write "decline" here to opt out): __________
[ ] My organization will respect the embargo discipline per §3.7.
[ ] My organization understands the engagement is reciprocal and
    not paid (§6.3.d), and is not commercial support (§9 item 3).
[ ] We have read this document end-to-end, including the severity-1
    timeline discussion in §6.4.

Signature:  __________________________________________
Date:       __________________________________________

<<< END <<<
```

Sponsor reply: within 7 business days per
[§8.2](#82-sponsor-responds-within-7-business-days).

---

## 11. Appendix — documents you will consume

Cross-references to project documents the engagement reads.

**Before you apply.** [`PATH-TO-V1.md`](PATH-TO-V1.md) (the v0.5
criterion this program closes); [`README.md`](../README.md) (project
status); [`ARCHITECTURE.md`](../ARCHITECTURE.md) (crate dependency
graph); [`RISKS.md`](RISKS.md) (v0.x known limitations);
[`LICENSE`](../LICENSE) (Apache-2.0).

**At engagement kickoff (Week 0).**
[`tutorials/production-deployment.md`](tutorials/production-deployment.md)
(end-to-end deploy walkthrough); [`CUDA-SETUP.md`](CUDA-SETUP.md)
(toolkit + driver matrix + SM-level compatibility);
[`deploy/helm/tensor-wasm/README.md`](../deploy/helm/tensor-wasm/README.md)
(Helm values reference); [`deploy/k8s/`](../deploy/k8s/) (raw
manifests); [`deploy/nomad/`](../deploy/nomad/) (HCL2 job spec).

**During production deployment (Weeks 2-5).** [`SLO.md`](SLO.md)
(the SLI/SLO definitions your check-in reports against);
[`CAPACITY-PLANNING.md`](CAPACITY-PLANNING.md) (sizing formulas and
tenants-per-host curves); [`BACKUP-RESTORE.md`](BACKUP-RESTORE.md)
(what to back up); [`UPGRADE.md`](UPGRADE.md) (fleet upgrade
playbook for partner-tagged builds);
[`OBSERVABILITY.md`](OBSERVABILITY.md) (tracing schema and OTLP
config); [`dashboards/README.md`](dashboards/README.md) (reference
Grafana dashboard import).

**The runbook set (Weeks 2-6, bookmark before you need it).**
[`runbooks/oncall-paging.md`](runbooks/oncall-paging.md);
[`runbooks/availability-fast-burn.md`](runbooks/availability-fast-burn.md);
[`runbooks/availability-slow-burn.md`](runbooks/availability-slow-burn.md);
[`runbooks/availability-very-slow-burn.md`](runbooks/availability-very-slow-burn.md);
[`runbooks/invoke-latency-spike.md`](runbooks/invoke-latency-spike.md);
[`runbooks/healthz-slow.md`](runbooks/healthz-slow.md);
[`runbooks/dispatch-latency-spike.md`](runbooks/dispatch-latency-spike.md);
[`runbooks/rollback.md`](runbooks/rollback.md);
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md);
[`runbooks/trace-id.md`](runbooks/trace-id.md);
[`runbooks/cve-disclosure-dry-run.md`](runbooks/cve-disclosure-dry-run.md);
[`runbooks/README.md`](runbooks/README.md) (index).

**For reference.** [`BENCHMARKING.md`](BENCHMARKING.md) (bench
methodology); [`PERFORMANCE.md`](PERFORMANCE.md) (bench targets and
CI regression gate); [`SECURITY.md`](../SECURITY.md) (threat model,
disclosure process, backport policy);
[`SECURITY-AUDIT-RFP.md`](SECURITY-AUDIT-RFP.md) (external audit
running in parallel); [`GOVERNANCE.md`](../GOVERNANCE.md) (lazy-
consensus model and voting rules behind your roadmap voice);
[`MAINTAINERS.md`](../MAINTAINERS.md) (registry of sponsor-side
contacts); [`CONTRIBUTING.md`](../CONTRIBUTING.md) (DCO and patch
flow); [`rfcs/README.md`](../rfcs/README.md) (RFC process behind
your first-refusal voice); [`CHANGELOG.md`](../CHANGELOG.md)
(release-by-release log).

---

_Status: v0.5 program kit. Revisions land as PRs against this file
under the standard lazy-consensus review in
[`GOVERNANCE.md`](../GOVERNANCE.md). The application template in
[§10](#10-application-template) is the partner-facing interface;
changes require sponsor sign-off because every in-flight applicant
uses the version live at the moment they applied._

_The severity-1 timeline framing in
[§6.4](#64-the-severity-1-timeline-discussion) is flagged for
explicit sponsor sign-off at the next maintainer sync before this
kit is sent to candidate partners; the published
[`SECURITY.md`](../SECURITY.md) backport policy is the binding
commitment, and this kit must not be sent if the sponsor disagrees
with §6.4's framing of "priority queue position within the published
policy, not a tighter SLA than the policy"._
