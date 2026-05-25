<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton TensorWasm — Governance

_Status: living document. First written for the v0.5 PATH-TO-V1 gate
(see [`docs/PATH-TO-V1.md`](docs/PATH-TO-V1.md), v1.0 exit criterion:
"Maintainer governance documented in `GOVERNANCE.md`")._

This is a lightweight, opinionated governance model for a small core
team. It is deliberately not modeled on the CNCF Technical Oversight
Committee process or the Apache Software Foundation's voting rules —
those are sized for projects with dozens of maintainers and a parent
foundation, and the overhead would slow this project down without
adding value. The shape here is closer to Wasmtime, ripgrep, or
zellij: a small set of maintainers, lazy consensus for most things,
explicit escalation paths when it fails.

If this doc grows past two screens of process for any single
decision, the doc is wrong and should be cut back.

## Purpose and scope

This document covers how decisions get made on Craton TensorWasm: who
holds review and merge rights, how disagreements are resolved, how
maintainers come and go, and how security disclosures are handled
inside the maintainer group. It is the procedural complement to
[`CONTRIBUTING.md`](CONTRIBUTING.md), which covers how external
contributors interact with the project.

This document does **not** modify the [Apache License 2.0](LICENSE)
under which TensorWasm is distributed. The license is sovereign: nothing
in here grants additional rights to maintainers, removes rights from
contributors, or alters the inbound=outbound licensing model. If a
clause here ever appears to conflict with the license, the license
wins and the clause is treated as a drafting bug to be fixed.

It also does not modify the [Code of Conduct](CODE_OF_CONDUCT.md);
the CoC applies to everyone equally, maintainers included.

The audience is current and prospective maintainers. Contributors
making one-off PRs do not need to read this — `CONTRIBUTING.md` is
enough.

## Roles

There are four roles. They are deliberately few. Adding roles is a
governance amendment (see the last section); inflating role
hierarchies for status reasons is the kind of thing this document
exists to prevent.

### Contributor

Anyone who has had at least one PR merged into the repository. No
obligations beyond the Code of Conduct and the DCO sign-off required
by `CONTRIBUTING.md`. No commit or review rights. Contributor status
is permanent — once a PR has merged, the author is a contributor
forever, including for the purposes of release notes and attribution.

### Maintainer

A person with review and merge rights on at least one crate area
(for example, `tensor-wasm-mem`, `tensor-wasm-api`, the `docs/` tree, or the
CI workflows). Maintainers are listed in [`MAINTAINERS.md`](MAINTAINERS.md);
that file is the source of truth for who currently holds the role.
This document does not duplicate the list.

Maintainers are expected to:

- Review PRs in their area within a reasonable window — not a hard
  SLA, but stale review queues are a sign the maintainer group is
  understaffed and should be discussed.
- Apply the lazy-consensus rule below honestly: object when they
  disagree, approve when they don't, and not sit silently on PRs they
  have an opinion about.
- Recuse themselves from review of PRs where they have a conflict of
  interest (see the Conflict of interest section).
- Respect the CVE embargo process for any security disclosure they
  are briefed on, whether or not they sit on the Security committee.

There is no requirement that a maintainer be employed by Craton
Software Company. The current commercial sponsor is Craton (see
`MAINTAINERS.md`), but the role does not depend on that.

### Lead maintainer

A single maintainer holds tiebreaker authority on contested
decisions. The role exists only to break ties; it does not confer
veto power, override the vote rules below, or give the holder any
special review rights on day-to-day PRs.

The lead maintainer rotates only on explicit step-down. There is no
fixed term. If the lead steps down, the active maintainer group picks
a successor by simple majority before the step-down takes effect; if
no successor is named within 30 days, the most senior active
maintainer (by `MAINTAINERS.md` ordering) acts as lead pro tem until
a successor is chosen.

The current lead maintainer is named in `MAINTAINERS.md`. If that
file's "Lead Maintainer" field reads `TBD`, the project has no lead;
contested decisions in that state require a 2/3 majority of active
maintainers and cannot be broken by tiebreaker, on the theory that a
project that has not named a lead should not be making contested
decisions until it does.

### Security committee

A subset of the maintainers responsible for handling embargoed
security reports under the process described in
[`SECURITY.md`](SECURITY.md). The committee is the only group with
access to embargoed reports before public disclosure; other
maintainers are briefed on a need-to-know basis.

Membership is a deliberate maintainer-group decision, not automatic.
A maintainer becomes a member of the Security committee only by the
onboarding flow below, after explicitly accepting the embargo
discipline that comes with the role. The committee should include at
least two maintainers at all times; if it drops below two, recruiting
a third is the next governance priority and blocks any non-security
release.

The Security committee's standing commitments are listed under
[Security disclosures](#security-disclosures) below.

## Decision making

The default is lazy consensus. Voting and tiebreakers exist for the
cases where lazy consensus does not resolve the question, not as the
normal mode.

### Lazy consensus (the default)

A PR that has at least one approving review from a maintainer other
than the author and has stood for 72 hours with no maintainer
objections may be merged. The 72-hour clock starts when the second
maintainer's approval lands, and resets if the PR is force-pushed in
a way that materially changes the diff (typo fixes and rebase-only
pushes do not reset).

This is the rule that governs ~95% of merges. It is intentionally
lightweight. The 72-hour window is the cost paid for not requiring a
formal vote on every PR; maintainers who want to block a PR are
expected to say so within the window, not retroactively.

Documentation-only PRs, dependency bumps with green CI, and
straightforward bug fixes with tests can merge sooner than 72 hours
at the reviewing maintainer's discretion. Anything touching the WIT
surface, the HTTP API contract, the snapshot format, or the security
posture should sit the full window even if it looks small.

### Substantive design changes require an RFC

If a change alters a public API surface, the snapshot format, the WIT
interface, the dispatch model, the auth model, or any commitment
listed in `docs/PATH-TO-V1.md`, it requires an RFC merged before the
implementation PR lands. The mechanical process for RFCs is documented
in [`rfcs/README.md`](rfcs/README.md) (which is the source of truth
for the procedure; this file only references it).

The judgment of "substantive" lives with the reviewing maintainers.
The intent is to catch designs that need cross-cutting review, not to
gate every refactor on paperwork.

### When lazy consensus fails

If a maintainer objects within the 72-hour window, lazy consensus
does not apply and the PR cannot merge until the objection is
resolved. Resolution paths, in order of preference:

1. The objector and the author iterate on the PR until the objection
   is withdrawn.
2. The PR is closed or paused while the author writes an RFC to work
   through the design disagreement in long form.
3. The active maintainers hold a vote.

A vote is a last resort, not a first move. Most contested changes
should die quietly in path (2) — the RFC either convinces the
objector or reveals that the change isn't worth the cost.

### Voting rules

When a vote is called:

- The eligible voters are the **active maintainers**: any maintainer
  listed in `MAINTAINERS.md` who has either reviewed at least one PR
  or merged at least one PR in the trailing 90 days. Inactive
  maintainers are not eligible to vote but are not removed by the
  vote-eligibility check — see [Maintainer offboarding](#maintainer-offboarding)
  for the separate inactive-removal process.
- A vote passes with a **2/3 supermajority** of votes cast within
  seven days of the vote being called. Abstentions do not count
  toward either side. Quorum is half of active maintainers, rounded
  up.
- Votes are public: cast as a comment on the PR or the RFC under
  vote, signed by the maintainer. No proxies; if a maintainer is
  unavailable for the seven days they abstain by default.
- If the vote ties (only possible with an even number of active
  maintainers and no abstentions), the **lead maintainer** breaks the
  tie. The lead's tiebreaker vote is not a veto: it only applies on
  an exact even split. If the project has no lead at the time of the
  vote (see [Lead maintainer](#lead-maintainer)), an even split fails
  the vote.

A 2/3 threshold is high enough that contested decisions need clear
majority support before merging, low enough that a single dissenting
maintainer cannot block work the rest of the team agrees on. This is
the same threshold used for inactive-to-emeritus transitions and for
for-cause offboarding below, so there is one number to remember.

### Vetoes

No maintainer has a unilateral veto in normal operation. A single
maintainer may block a PR from merging only on one of three grounds:

1. **Apache-2.0 incompatibility.** The change would introduce code
   that cannot be redistributed under the project's license — for
   example, a GPL-licensed dependency, a patent-encumbered algorithm
   with no Apache grant, or text without clear provenance.
2. **Irreparable security regression.** The change would publicly
   re-introduce a previously fixed CVE, leak an embargoed disclosure,
   or weaken a security boundary that the Security committee has
   committed to maintaining.
3. **Trademark violation.** The change misuses the project name, the
   Craton Software Company name, or any third-party trademark in a
   way that creates legal risk for the project or its contributors.

A veto must be explicit: the blocking maintainer states which of the
three grounds applies and provides a one-paragraph justification.
Vetoes are reviewable: if the vetoed party disputes the ground, the
question goes to a vote of active maintainers under the rules above,
and a 2/3 majority can override the veto. Vetoes that don't fit one
of the three grounds are not vetoes — they are objections, and they
follow the normal lazy-consensus-fail path.

The three grounds are narrow on purpose. A maintainer who feels they
need a veto for any other reason is asking for a vote.

## RFC procedure

The Request-for-Comments process is documented in
[`rfcs/README.md`](rfcs/README.md). That file describes the template,
the directory layout, the numbering scheme, and the merge mechanics.

The short version: substantive design changes (see
[Substantive design changes require an RFC](#substantive-design-changes-require-an-rfc)
above) land as a markdown file under `rfcs/`, go through the same
lazy-consensus review as any other PR, and are merged in `accepted`
state before the implementation PR lands. RFCs may also be merged in
`rejected` state to record the decision and the reasoning, so future
contributors don't re-propose the same design without knowing why it
was turned down last time.

The `rfcs/` directory will be present by the time this GOVERNANCE.md
merges. If you are reading this and that directory does not exist
yet, the cross-reference is to a planned doc landing in the same
batch as this one.

## Security disclosures

The external-facing disclosure process — the reporting address, the
triage SLO, the supported versions, and the threat model — lives in
[`SECURITY.md`](SECURITY.md). That document is the contract with
people who find vulnerabilities; this section is the contract with
the maintainer group about how the Security committee operates
internally.

### Standing commitments

When the Security committee receives an embargoed report at
`security@craton.com.ar`, the committee commits to:

1. **Acknowledge within 72 hours.** A human reply confirming receipt
   and naming a committee member as the primary point of contact for
   the report. The clock starts when the report lands in the inbox,
   not when a committee member next checks email; the committee is
   responsible for arranging coverage that meets this commitment.
2. **Provide a fix or a documented workaround within 90 days** of
   acknowledgement, or explain in writing why the timeline is being
   extended and what the new target is. The 90-day figure aligns
   with industry-standard responsible-disclosure windows (Google
   Project Zero, CERT/CC) and is the longest the committee will let
   a confirmed vulnerability sit without remediation, unless the
   reporter explicitly agrees to a longer embargo.
3. **Prefer coordinated disclosure.** The committee works with the
   reporter on a disclosure date, gives credit in the advisory unless
   the reporter prefers anonymity, and publishes the advisory through
   the GitHub Security Advisory mechanism in addition to any
   reporter-driven channel. Reporter-driven publication on the same
   day as the fix release is welcome; the committee will not request
   gag clauses.

### Internal handling

While a report is under embargo:

- Discussion happens on a private channel known only to the Security
  committee. Other maintainers are briefed only when their crate area
  is affected and they need to review the fix, and only after they
  acknowledge the embargo discipline in writing (an email reply is
  enough).
- Fix PRs are developed against private forks or with the diff
  shared only via the private channel until disclosure day. The fix
  lands on `main` as part of the public release that ships the
  advisory; backports to supported branches land at the same moment.
- If a Security committee member resigns or is offboarded while a
  report is under embargo, their access to the private channel is
  revoked at the moment of offboarding. The embargo discipline
  continues to bind them under the same Code of Conduct rules that
  apply to any contributor.

A breach of the embargo by a maintainer — publishing details before
the coordinated date, sharing the report outside the briefed group,
or filing a public issue with reproducer details — is grounds for
for-cause offboarding (see below).

## Maintainer onboarding

Adding maintainers is how the project grows past a small core team
without burning the small core team out. The bar is sustained
contribution, not heroic single contributions; the goal is to find
people who will be reviewing PRs and shaping the project in 12
months, not people who will fade after one big PR.

### Criteria

A candidate is eligible for maintainer status when all four hold:

1. **At least six months** of activity on the project, measured from
   their first merged PR. The intent is to have seen at least one
   release cycle and one round of issue triage with them.
2. **At least three merged substantive PRs.** "Substantive" means
   the PR shaped a non-trivial part of a crate, a workflow, or a doc
   — not a typo fix or a dependency bump. Quality matters more than
   raw count; the number is a floor, not a target.
3. **A sponsor maintainer nominates them** by opening a PR against
   `MAINTAINERS.md` with the candidate's name and a one-paragraph
   rationale (areas they would maintain, why they're a good fit,
   pointers to the PRs that demonstrate it). The sponsor commits to
   onboarding the candidate — answering their questions for the
   first month or two — so this is not a step to take lightly.
4. **A simple majority of active maintainers approves** the
   nomination PR. This is the one place lazy consensus does not
   apply: maintainer additions require positive assent, not just
   absence of objection. The seven-day vote window from
   [Voting rules](#voting-rules) applies.

The Security committee has an extra step: a candidate joining the
committee must additionally accept the embargo discipline in writing
in the nomination PR, and the existing committee must reach internal
consensus before the nomination is opened. The intent is to keep the
committee tight.

### Mechanical steps on approval

Once the nomination PR merges, the sponsor (or the lead) carries out
the following, ideally in a single PR or a tight follow-up:

1. **Add the new maintainer to [`MAINTAINERS.md`](MAINTAINERS.md)**
   in the appropriate section, with the area(s) they cover.
2. **Add them to the GitHub team** that holds review/merge rights on
   the repository, and confirm they have 2FA enabled on their GitHub
   account.
3. **Add them to the security-disclosures group** if and only if
   they are joining the Security committee. This is a separate
   credentialing step; default-merge rights and embargoed-report
   access are not the same thing.
4. **Announce the addition in `CHANGELOG.md`** under the next
   release's entry, with a one-line welcome. This is the public
   record of who joined when.

If any of these steps cannot be carried out (the candidate doesn't
have a GitHub account, refuses 2FA, etc.) the maintainer status is
not effective. The nomination doesn't roll back; the mechanical
prerequisites just have to be satisfied first.

## Maintainer offboarding

People leave projects. The exit paths exist so that leaving is clean
— no ambiguity about whether someone is still a maintainer, no
festering inactive accounts holding merge rights, no awkward
conversations when someone needs to be removed for cause.

There are four exit paths.

### Voluntary

A maintainer who wants to step down opens a PR removing themselves
from `MAINTAINERS.md`. The PR is effective on merge and is treated
as lazy-consensus by default — no vote needed. The departing
maintainer may also move themselves to the emeritus list in the same
PR, which preserves their history without retaining merge rights.

A voluntary offboarding PR should not be blocked. A maintainer who
wants to leave is leaving; the question is whether to record their
contributions as emeritus or to remove them from the file entirely,
and that's the departing person's choice.

### Inactive

A maintainer who has merged no contributions and reviewed no PRs for
12 consecutive months may be moved to emeritus status by any other
maintainer, via a PR moving them from the active list to the
emeritus list in `MAINTAINERS.md`. The PR requires a simple majority
of active maintainers (under the [Voting rules](#voting-rules)) and
must give the inactive maintainer 30 days' notice — typically by
@-mentioning them on the PR — before the vote closes.

The 12-month window is generous on purpose. Contributors have lives;
parental leave, illness, job changes, and sabbaticals all happen, and
the project would rather wait a year and welcome someone back than
push them out at month four. A maintainer who knows they're going to
be away for an extended period can preemptively move themselves to
emeritus and rejoin via the standard onboarding flow when they're
back; that's the cleanest path and avoids the awkward 30-day notice.

### For-cause

A maintainer may be removed for cause on one of two grounds:

1. **A confirmed Code of Conduct violation** under
   [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), where confirmed
   means the enforcement process referenced in the CoC has concluded
   with a finding that warrants removal of trust roles.
2. **A breach of maintainer trust** — publishing an embargoed CVE
   before its disclosure date, sharing an embargoed report outside
   the briefed group, deliberately landing code that violates the
   Apache-2.0 license obligations, intentionally introducing a
   backdoor, or similar acts that compromise the project's integrity.

For-cause removal requires:

- A 2/3 majority of active maintainers under the
  [Voting rules](#voting-rules).
- Sign-off from the Security committee (or, if the cause involves
  the Security committee itself, sign-off from the non-committee
  active maintainers). The Security committee sign-off exists to
  catch the case where the cause is a security breach and the rest
  of the maintainer group may not have the full picture.

For-cause removal takes effect immediately on the vote closing:
merge rights, GitHub team membership, and Security committee access
are revoked the same day. The removed maintainer is moved to neither
the active list nor the emeritus list; their entry is removed from
`MAINTAINERS.md` with the PR commit message recording the date and
the ground.

For-cause removal is a serious action and should be rare. The
threshold is intentionally high — both the 2/3 vote and the Security
committee sign-off — because the reputational cost to the removed
person is permanent. Maintainers who think a colleague's behavior is
heading toward for-cause territory should raise it privately first;
the CoC enforcement path exists for that.

### Emeritus

Maintainers who have left voluntarily or via the inactive path are
listed in `MAINTAINERS.md` under an emeritus section. They have:

- No commit rights and no review-merge rights.
- No vote in maintainer decisions.
- Permanent acknowledgement of their contributions in the file —
  this is a thank-you, not a punishment.

An emeritus maintainer who wants to come back follows the standard
onboarding flow above. They do not get a fast-track; the six-month
and three-PR thresholds reset. The reasoning is symmetric with the
incoming-maintainer process: time away changes context, and the
project may have moved in ways that warrant a fresh look at fit
rather than a presumption of continuity. In practice, an emeritus
maintainer who returns and contributes consistently will clear the
thresholds quickly.

Emeritus maintainers are bound by the same embargo discipline as
active maintainers for any disclosures they were briefed on while
active. The role ends; the obligation to not leak does not.

## Conflict of interest

Maintainers who have a personal or financial interest in a PR recuse
themselves from review and merge of that PR. Examples that trigger
recusal:

- The maintainer's employer (or a company in which they hold equity)
  is the author of the PR, sponsored the work, or sells a directly
  competing product whose roadmap the PR affects.
- The PR materially advances a product or paid service the
  maintainer owns or holds a financial stake in.
- The PR was authored by a close family member or by someone the
  maintainer has a non-arms-length personal relationship with.

Recusal means: the conflicted maintainer does not approve, does not
merge, and does not vote if the PR ends up in a vote. They may
comment to provide technical context — recusal is about not steering
the decision, not about going silent — but their comments are
labeled as conflicted so other reviewers can weight them
appropriately.

Disclosure is the maintainer's responsibility. The project does not
audit conflicts proactively; the trust model is that maintainers
will disclose in good faith, and that other maintainers will raise
it gently if a conflict appears to have been missed. A pattern of
undisclosed conflicts is grounds for for-cause review under the
breach-of-trust ground above.

This rule is intentionally narrower than the IRS or SEC definitions
of conflict of interest. The intent is to prevent a maintainer from
unilaterally landing changes that benefit them or their employer at
the project's expense, not to require recusal every time a
maintainer works on something they care about. If in doubt,
disclose; the worst outcome of over-disclosing is a brief slack
thread, while the worst outcome of under-disclosing is a governance
crisis.

## Conduct

All participants in the project — contributors, maintainers, the
lead, and the Security committee — are bound by
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Maintainers are
additionally expected to model the behavior described there in their
public interactions on behalf of the project: reviews, RFC
discussion, issue triage, and any communication where they are
recognizable as project representatives.

Enforcement goes through the channel named in the CoC. Maintainer
status does not grant immunity from CoC enforcement; if anything,
the standard is higher for maintainers because their conduct sets
the tone for the contributor base.

## Amendments to this doc

Changes to GOVERNANCE.md itself require an RFC under
[`rfcs/README.md`](rfcs/README.md). The amendment RFC is voted on
under the same [Voting rules](#voting-rules) as any other contested
decision: 2/3 supermajority of active maintainers, seven-day window,
lead maintainer tiebreaks an exact split.

The RFC requirement exists because governance changes affect
everyone who interacts with the project, not just the maintainer who
wrote the change. Time-pressured tweaks ("we need a new rule by
Friday") are exactly the kind of change that should sit through the
RFC process so the rule is well-thought-through before it's binding.

Typo fixes, link repairs, and reformatting are not amendments and
follow the normal lazy-consensus PR flow. The dividing line is
whether the change alters the meaning of any rule: if a reasonable
reader's behavior would change because of the edit, it's an
amendment.

---

_Cross-references:_
[`CONTRIBUTING.md`](CONTRIBUTING.md) ·
[`MAINTAINERS.md`](MAINTAINERS.md) ·
[`SECURITY.md`](SECURITY.md) ·
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) ·
[`rfcs/README.md`](rfcs/README.md) (lands in the same batch as this doc) ·
[`docs/PATH-TO-V1.md`](docs/PATH-TO-V1.md) ·
[`LICENSE`](LICENSE)
