<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# RFC NNNN: <title>

- **Author(s):** <name <email>>
- **Status:** Draft <!-- Draft | Accepted | Rejected | Superseded by NNNN -->
- **Created:** YYYY-MM-DD
- **Discussion PR:** <link to PR once opened>
- **Related:** <PATH-TO-V1 line, prior RFCs, issues>

<!--
Copy this file to `rfcs/0000-short-kebab-slug.md` and fill in the
sections below. Drop sections you genuinely cannot answer under
"Unresolved questions"; do not delete section headings. See
[`README.md`](README.md) for the lifecycle and decision process.
-->

## Summary

One paragraph, plain English. Somebody who reads only this section
should know what changes, who is affected, and roughly when. Avoid
jargon that isn't already in [`ARCHITECTURE.md`](../ARCHITECTURE.md).

## Motivation

Why now. What is broken, missing, or limiting today, with specifics.
Reference the file, the metric, the bug, or the user report that
prompted this. Answer the "why not just keep doing what we do?"
question directly.

If the motivation is a milestone exit criterion in
[`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md), name the milestone and
the bullet.

## Detailed design

The actual proposal. This section must be specific enough that another
contributor could implement it from the RFC alone. Include:

- The new (or changed) API surface — function signatures, HTTP routes,
  CLI subcommands, configuration keys, on-disk schemas, as applicable.
- Behavioural changes — what does the system do differently after this
  lands?
- Error handling and edge cases — what fails, how, with what message?
- Compatibility implications — does this break callers? What is the
  deprecation/migration path?
- Feature-flag story — does this gate behind a new flag, change a
  default, remove a flag?
- Test plan — what new tests demonstrate the proposal works? Which
  existing tests change?
- Rollout — does this land in one PR or several? Any sequencing
  constraints with other work?

Diagrams, code blocks, and worked examples are welcome and usually
helpful. Aim for "as detailed as the implementation PR description
would have been, written first".

## Drawbacks

Be honest. Every design has costs. List the ones you know about:

- Maintenance burden the project takes on.
- Surface area added to the public API.
- Performance regressions (with rough magnitudes).
- Dependencies pulled in, especially anything not already in
  `Cargo.lock`.
- Backwards-compatibility commitments this locks in.
- Onboarding tax for new contributors.

If you genuinely believe there are no drawbacks, say so — but expect
reviewers to push back, and prepare to update this section after the
PR discussion.

## Rationale and alternatives

Compare this proposal to at least **two** other approaches you
considered. For each alternative, state:

1. **What it is** — enough that a reader who has not heard of it can
   evaluate the comparison.
2. **Why it was rejected** — specific to this project, not generic
   trade-offs.
3. **What would change the calculus** — under what future condition
   would this alternative become the better choice?

"Do nothing" is always an alternative; address it explicitly. If the
status quo is acceptable for some users, say which users.

## Unresolved questions

Open issues that the RFC does not answer. Frame each as a question
with a proposed answer where you have one. Examples:

- *What is the default rate-limit burst capacity?* Proposed: 10×
  steady-state. Open until benchmark data lands.
- *Do we expose this via the CLI or only the HTTP API?* Proposed: HTTP
  only for v1.0, CLI in v1.1. Open until CLI authors weigh in.

Unresolved questions block acceptance only if a maintainer says they
do. Most RFCs accept with a non-empty list here; the implementation
PRs close the questions one by one.

## Prior art

Other projects, papers, or standards that informed the design. For each,
note what TensorWasm takes from it and what it leaves on the table.
Examples of useful prior art:

- Rust RFCs (`rust-lang/rfcs`) for the process shape itself.
- Wasmtime RFCs (`bytecodealliance/rfcs`) for runtime-specific
  precedent.
- WASI proposals (`WebAssembly/WASI`) for guest-facing interfaces.
- Academic papers, with a one-line summary and a citation.
- Commercial systems (k8s, Firecracker, Lambda, etc.) when their
  public docs describe a relevant design.

Lack of prior art is itself worth noting — it means this RFC is the
project's first decision in this area and reviewers should weigh
novelty risk accordingly.

## Future possibilities

Things this RFC does not propose but enables. Out-of-scope for this
discussion, but worth flagging so the design does not foreclose them.

Examples:

- *Multi-region replication for snapshots.* Not in this RFC; the
  format we land here is region-agnostic so a future RFC can layer
  replication on top.
- *Auto-tuned defaults.* This RFC ships static defaults; a follow-up
  could expose them to a control loop.

Be brief here. The point is to mark adjacent space as "we noticed and
chose not to expand scope", not to design the follow-ups.
