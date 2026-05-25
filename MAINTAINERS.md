<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton TensorWasm — Maintainers

_Status: living document. This file is the source of truth for who
currently holds maintainer rights on Craton TensorWasm. The procedural
rules — what a maintainer is, how decisions are made, how people are
onboarded and offboarded — live in [`GOVERNANCE.md`](GOVERNANCE.md);
this file is the registry, not the rulebook._

This file is also the source of truth for the **active-maintainer
count** used by the quorum and vote-threshold math in
[`GOVERNANCE.md`](GOVERNANCE.md#voting-rules). When the active list
below changes, every threshold that depends on "active maintainers"
(simple majority for onboarding, 2/3 for contested votes and for-cause
offboarding, half-rounded-up for quorum) moves with it.

Craton TensorWasm is commercially sponsored by **Craton Software
Company**, but maintainer status is not contingent on employment with
the sponsor. See [`GOVERNANCE.md`](GOVERNANCE.md#maintainer) for the
role definition.

## Placeholders are by design

Every `TBD` cell in this file is **intentional**, not an unfinished
draft. Each placeholder has a documented unblock trigger:

- **Lead maintainer** (1 slot): filled by internal selection from the
  active-maintainer pool per
  [`GOVERNANCE.md`](GOVERNANCE.md#maintainer-onboarding); see the
  "Lead maintainer" section below for the exact recruitment flow and
  what changes about decision-making while the slot is empty.
- **Active maintainers** (registry): filled by the standard onboarding
  RFC + simple-majority approval process in `GOVERNANCE.md`.
- **Security committee** (2 slots): a subset of active maintainers;
  filled after the active list has at least two members and the
  committee is constituted per `SECURITY.md` §"Backport policy".
- **Area ownership** (13 cells × 2 columns each = 26 placeholders):
  filled as maintainers join. Until then, every area is jointly
  reviewed; the absence of named owners is the binding rule, not a
  documentation gap.

The v0.x convention is: do not invent placeholder names. When a real
person fills a slot, this file is amended in the same PR that adds
them to `GOVERNANCE.md`-recognized rights (GH team, security
mailing list, etc.). v1.0 will not ship while a quorum-blocking
slot is still `TBD`; see `PATH-TO-V1.md` v1.0 exit criteria.



## Contact

- General project mail and Code-of-Conduct reports: `security@craton.com.ar`
- Security disclosures (embargoed): `security@craton.com.ar` — see
  [`SECURITY.md`](SECURITY.md) for the disclosure contract.

The single mailbox is intentional during the v0.x window. When the
project grows past one channel's worth of traffic, a separate
conduct@ address will be added; that change is an amendment to this
file.

## Lead maintainer

The Lead Maintainer holds tiebreaker authority on contested
decisions, per
[`GOVERNANCE.md`](GOVERNANCE.md#lead-maintainer). The slot is a
singleton.

| Lead maintainer | Since |
|---|---|
| TBD | n/a |

While this slot reads `TBD`, the project has no lead. Contested
decisions in that state require a 2/3 majority of active maintainers
and cannot be broken by tiebreaker; see
[`GOVERNANCE.md`](GOVERNANCE.md#voting-rules) for the consequence on
even-split votes.

Recruitment for the Lead Maintainer slot happens through the standard
onboarding flow in
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer-onboarding) followed by
internal selection from among the active maintainers; the Lead is not
elected from outside the active list.

## Active maintainers

A maintainer is anyone with review and merge rights on at least one
crate area (see [Area ownership](#area-ownership) below). The list
below is the registry; the role definition is in
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer).

| Maintainer | GitHub | Areas | Since |
|---|---|---|---|
| TBD | TBD | TBD | n/a |

This list is currently empty pending the first round of maintainer
onboardings under the GOVERNANCE.md criteria. Until at least one
active maintainer is listed, every PR is reviewed and merged by the
commercial sponsor under the bootstrap arrangement described in
[`GOVERNANCE.md`](GOVERNANCE.md#purpose-and-scope). The bootstrap
arrangement is not a substitute for the onboarding process; it is a
holding pattern that ends as soon as the first nominee clears
the criteria in
[`GOVERNANCE.md`](GOVERNANCE.md#criteria).

## Security committee

The Security Committee is a subset of the active maintainers
responsible for handling embargoed disclosures under
[`SECURITY.md`](SECURITY.md) and the standing commitments in
[`GOVERNANCE.md`](GOVERNANCE.md#security-disclosures). Membership is
deliberate and additive: a maintainer joins the committee only by the
onboarding flow described in
[`GOVERNANCE.md`](GOVERNANCE.md#criteria), after accepting the
embargo discipline in writing.

The committee must have at least two members at all times. If it
drops below two, recruiting a third is the next governance priority
and blocks any non-security release, per
[`GOVERNANCE.md`](GOVERNANCE.md#security-committee).

| Security committee member | Backup contact | Since |
|---|---|---|
| TBD | TBD | n/a |
| TBD | TBD | n/a |

Both slots are vacant pending maintainer recruitment. While the
committee is below the two-member floor, embargoed reports sent to
`security@craton.com.ar` are handled by the commercial sponsor under
the same 72-hour acknowledgement and 90-day fix-or-workaround
commitments documented in
[`GOVERNANCE.md`](GOVERNANCE.md#standing-commitments). This is a
holding pattern; populating both slots is a prerequisite for the v0.5
PATH-TO-V1 gate.

The committee owns the backport coordination described in
[`SECURITY.md`](SECURITY.md#backport-policy): every backport-eligible
fix is cherry-picked by the committee onto the relevant `release-vN.x`
branch, alongside the public release on disclosure day.

## Emeritus maintainers

Emeritus maintainers are former active maintainers who left
voluntarily or via the inactive path in
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer-offboarding). They have no
commit or vote rights; the listing is acknowledgement, not
authority. Emeritus maintainers remain bound by the embargo
discipline for any disclosures they were briefed on while active.

| Emeritus maintainer | Active period | Areas (while active) |
|---|---|---|
| _(none yet)_ | | |

The list is expected to remain empty for the v0.x window.

## Area ownership

Each crate area has a primary **Owner** maintainer and a **Backup**
maintainer. The owner is the default reviewer for PRs touching the
area; the backup handles reviews when the owner is unavailable and
holds the recall path if the owner becomes inactive under
[`GOVERNANCE.md`](GOVERNANCE.md#inactive).

The thirteen areas below partition the repository. A maintainer may
own or back up more than one area; the bound is honest review
capacity, not a per-area headcount.

| Area | Owner | Backup |
|---|---|---|
| `tensor-wasm-core` (error model, telemetry, public traits) | TBD | TBD |
| `tensor-wasm-mem` (linear memory, unified buffer, isolation enum) | TBD | TBD |
| `tensor-wasm-exec` (Wasmtime wrapper, dispatch, epoch timers) | TBD | TBD |
| `tensor-wasm-wasi-gpu` (WASI-GPU host functions, WIT surface) | TBD | TBD |
| `tensor-wasm-jit` (auto-offload, BLAKE3 cache, blueprints) | TBD | TBD |
| `tensor-wasm-snapshot` (zstd+bincode capture/restore, schema) | TBD | TBD |
| `tensor-wasm-tenant` (TenantRegistry, quotas, MPS plumbing) | TBD | TBD |
| `tensor-wasm-api` (HTTP gateway, auth, OpenAPI) | TBD | TBD |
| `tensor-wasm-cli` (`tensor-wasm` binary, completions, man pages) | TBD | TBD |
| `tensor-wasm-bench` (criterion harness, baseline.json) | TBD | TBD |
| `fuzz/` (cargo-fuzz targets, corpus management) | TBD | TBD |
| `docs/` (PATH-TO-V1, SECURITY-AUDIT, runbooks, dashboards) | TBD | TBD |
| `deploy/` (docker-compose, k8s manifests, Helm, CI workflows) | TBD | TBD |

Every cell currently reads `TBD`. Owners are assigned as maintainers
are onboarded under
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer-onboarding); the area is
named on the nomination PR and recorded in this table on merge of the
follow-up MAINTAINERS.md update.

Until an area has a named owner, PRs touching that area follow the
bootstrap arrangement noted under [Active maintainers](#active-maintainers).

## How to become a maintainer

Maintainer status is granted through the onboarding process in
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer-onboarding). The criteria,
in summary:

1. At least six months of activity on the project, measured from
   the candidate's first merged PR.
2. At least three merged substantive PRs.
3. A sponsoring active maintainer who opens the nomination PR
   against this file and commits to onboarding the candidate.
4. A simple-majority approval of active maintainers within the
   seven-day vote window.

The summary above is informational; the controlling text is in
[`GOVERNANCE.md`](GOVERNANCE.md#criteria), and any discrepancy
between the two is a drafting bug in this file to be fixed.

Candidates do not self-nominate. The sponsor relationship is part of
the process: a candidate who is not in regular contact with at least
one active maintainer should establish that contact first by
contributing PRs and participating in review discussion.

Joining the Security Committee is a separate step on top of
maintainer status, with the additional criteria documented in
[`GOVERNANCE.md`](GOVERNANCE.md#security-committee).

## How to step down

Offboarding routes are documented in
[`GOVERNANCE.md`](GOVERNANCE.md#maintainer-offboarding). In summary:

- **Voluntary.** A maintainer opens a PR removing themselves from
  the active list above (and optionally adding themselves to
  [Emeritus maintainers](#emeritus-maintainers)). Treated as
  lazy-consensus; should not be blocked.
- **Inactive.** A maintainer who has merged no contributions and
  reviewed no PRs for 12 consecutive months may be moved to
  emeritus by another maintainer's PR, after 30 days' notice and a
  simple-majority vote.
- **For-cause.** Removal for a confirmed Code-of-Conduct violation
  or a breach of maintainer trust; requires a 2/3 vote of active
  maintainers and sign-off from the Security Committee. See
  [`GOVERNANCE.md`](GOVERNANCE.md#for-cause).
- **Emeritus.** The terminal state of voluntary and inactive
  departures. No commit or review rights; the listing is permanent
  acknowledgement.

A maintainer who plans to be away for an extended period (parental
leave, sabbatical, deployment) is encouraged to preemptively move
themselves to emeritus and rejoin via the standard onboarding flow
when they return. That is cleaner than the 30-day inactive-notice
path.

When an active or Security-Committee maintainer departs by any route,
the area-ownership table above is updated in the same PR (or a tight
follow-up) so no area is left without a named owner. If an area would
become ownerless, the departing maintainer's backup steps up as
interim owner until a permanent owner is named.

## Reporting issues with a maintainer

Issues that concern the conduct or trustworthiness of an individual
maintainer — Code-of-Conduct violations, undisclosed conflicts of
interest, suspected embargo breaches, or any behavior that warrants
review under
[`GOVERNANCE.md`](GOVERNANCE.md#for-cause) — should be reported to
`security@craton.com.ar`.

The same mailbox handles security disclosures and conduct reports
during the v0.x window; reports are routed to the appropriate
recipient on receipt. A reporter who does not want the subject
maintainer to see the report should say so explicitly in the first
message; reports are handled with the same confidentiality discipline
as embargoed security reports.

For reports that involve a Security-Committee member, the report is
handled by the non-committee active maintainers under the
split-handling rule in
[`GOVERNANCE.md`](GOVERNANCE.md#for-cause), so that the subject of
the report is not in the review chain. The reporter does not need to
arrange this routing; it is the receiving group's responsibility.

Reports do not require a particular format. A one-paragraph
description of the incident, with dates and links to public artifacts
where possible, is enough to start the process. Reporters are not
required to propose a remedy.

## Cross-references

- [`GOVERNANCE.md`](GOVERNANCE.md) — role definitions, decision
  process, onboarding/offboarding rules (the controlling document
  for this file).
- [`SECURITY.md`](SECURITY.md) — external disclosure contract, backport
  policy, Security-Committee standing commitments.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — conduct standard
  applicable to all participants, maintainers included.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how contributors interact
  with the project; complementary to this file.
- [`docs/PATH-TO-V1.md`](docs/PATH-TO-V1.md) — v0.5 exit criterion
  "MAINTAINERS.md reviewed and trimmed/expanded" that motivated this
  revision.
