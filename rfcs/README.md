<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm RFC process

Substantive changes to TensorWasm — new subsystems, breaking API
changes, on-disk format bumps, anything that would land a
`## Decision` line in [`CHANGELOG.md`](../CHANGELOG.md) — go through a
lightweight RFC before the implementation PR. The goal is to surface
disagreement early, in writing, while a proposal is still cheap to
change.

This is deliberately not a heavy process. There is no committee, no
template-of-templates, no required pre-meeting. One contributor writes
a doc, opens a PR, gives reviewers a week, and a maintainer decides.
The bureaucracy is the four files under this directory; that is all of
it.

This document is the artifact behind the v0.3 governance workstream in
[`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md): *"RFC process
(lightweight — a `rfcs/` directory and a template) established at
v0.3, used in anger by v0.5."*

## When to write an RFC

Open an RFC before the implementation PR if the change is any of:

- **A breaking change to a public API.** Any item in
  [`tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md), the WIT
  surface under [`wit/`](../wit/), the `tensor-wasm` CLI subcommand
  shape, or any public Rust item in the workspace crates.
- **A new subsystem.** A new crate, a new background task family, a
  new on-disk artifact (snapshot variant, cache layout, etc.), a new
  external dependency category (a second GPU backend, a metric
  exporter, an auth provider).
- **An on-disk or wire format bump.** Snapshot version, HTTP request
  schema, OpenAPI changes that aren't additive, JIT cache key layout.
- **A policy change.** Anything maintainers would want to point at
  later when explaining "we decided X because Y" — deprecation
  windows, toolchain pin cadence, feature-flag defaults, supported
  platform matrix changes.
- **Anything that would warrant a `## Decision` line in CHANGELOG.**
  If the answer to "why did we do it this way?" is non-obvious six
  months from now, write it down once.

If you are unsure, err toward writing one. RFCs are cheap; rolling
back a merged design is not.

## When NOT to write an RFC

Most PRs do not need an RFC. Skip it for:

- Bug fixes, including security fixes (use the disclosure process in
  [`SECURITY.md`](../SECURITY.md) instead).
- Refactors that do not change observable behaviour or public APIs.
- Test additions, doc tweaks, typo fixes.
- Performance work that does not change an API or a guarantee.
- Adding a non-default feature flag that does not touch the existing
  default build path.
- Anything contained inside a single crate's private internals.

A reviewer can ask for an RFC on a PR they think warrants one; that is
not a rejection, it is a request to split the design discussion from
the code review.

## Lifecycle

1. **Draft.** Copy [`TEMPLATE.md`](TEMPLATE.md) to
   `rfcs/0000-short-slug.md` (the `0000` is a placeholder — it gets a
   real number on acceptance). Fill in every section; sections you
   genuinely cannot answer yet go under "Unresolved questions".
2. **PR.** Open a pull request that adds only the RFC file. The PR
   title should be `RFC: <slug>`. Link it to any related issue or
   PATH-TO-V1 line item.
3. **Call for comments.** Reviewers have at least **7 calendar days**
   to comment from the time the PR is marked ready for review.
   Substantive comments restart the clock at the author's discretion —
   the clock exists to prevent rushed merges, not to force them.
4. **Maintainer decision.** A maintainer (see
   [`MAINTAINERS.md`](../MAINTAINERS.md)) closes the discussion with
   one of three outcomes:
    - **Accept.** The RFC gets the next sequential number, moves to
      `rfcs/accepted/`, and the PR merges. Implementation can begin
      immediately and need not wait for further sign-off.
    - **Reject.** The RFC moves to `rfcs/rejected/` with a final
      comment explaining why. Rejected RFCs are kept on purpose — they
      document what the project considered and decided against.
    - **Defer.** The PR is closed without merging; the author can
      reopen when the blocker (a prerequisite RFC, a missing
      milestone, etc.) clears. Deferred RFCs do not get a number.
5. **Implementation.** The implementation PR(s) reference the RFC
   number in the commit message and PR description. If implementation
   reveals the design is wrong, open an amendment PR against the
   accepted RFC — the merged file is the spec.

There is no separate "final comment period" gate. The 7-day clock is
the comment period; if a maintainer is ready to decide before then and
there are no open objections, they wait the rest of the week and then
decide. If there are open objections, the clock restarts.

## Numbering

RFCs are numbered sequentially in **four-digit zero-padded** form:
`0001`, `0002`, `0003`. The number is assigned at acceptance, not at
draft time — drafts use `0000` in the filename. Take the next unused
number by inspecting `rfcs/accepted/` and `rfcs/rejected/` at the
moment of acceptance; if two RFCs accept in the same hour, the second
one rebases.

Filenames are `NNNN-short-kebab-slug.md`. Example:
`0001-snapshot-format-v2.md`. Keep the slug under ~40 characters and
specific enough that the directory listing is self-describing.

`0000-template-example.md` lives in this directory (not under
`accepted/`) as a worked example; it is **not** an accepted RFC.

## Decision authority

The default is **lazy consensus**: if no maintainer objects within the
comment period, the RFC is accepted. A single maintainer can accept on
their own if there are no open objections.

If maintainers disagree, the decision escalates to a vote of the
maintainers listed in [`MAINTAINERS.md`](../MAINTAINERS.md). Quorum is
a simple majority of currently active maintainers; ties go to the lead
maintainer. The vote, with each maintainer's stated position, is
recorded as the final comment on the PR before merge.

Authors are not maintainers for the purpose of their own RFC.
Maintainers who wrote the RFC abstain from the vote.

This will tighten when [`GOVERNANCE.md`](../GOVERNANCE.md) lands at
v0.5; until then, the maintainer roster in
[`MAINTAINERS.md`](../MAINTAINERS.md) is the authoritative list.

## Directory layout

```
rfcs/
├── README.md                       (this file)
├── TEMPLATE.md                     (copy this when drafting)
├── 0000-template-example.md        (worked example — NOT an accepted RFC)
├── accepted/                       (numbered, merged, decisions of record)
└── rejected/                       (numbered, considered and declined)
```

Accepted and rejected RFCs are immutable in their final state.
Substantive changes to an accepted RFC require a new RFC that
supersedes it (with a `Supersedes: NNNN` line in the new RFC's
Summary).

## Style

- Match the prose style of the existing docs under
  [`docs/`](../docs/): measured, specific, no marketing voice.
- No emojis.
- Prefer concrete numbers over hedges. "P99 ≤ 50 ms at 100 RPS" beats
  "fast enough for most users".
- Show the alternatives you considered, not just the winner.
- Apache-2.0 SPDX header at the top of every RFC file.

## Cross-references

- [`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md) — the workstream that
  this process feeds into.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — general contributor flow
  (DCO, PR template, commit style).
- [`MAINTAINERS.md`](../MAINTAINERS.md) — current decision-makers for
  acceptance.
- [`CHANGELOG.md`](../CHANGELOG.md) — where accepted RFCs surface as
  `## Decision` lines on release.
