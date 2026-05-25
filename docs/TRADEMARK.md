<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton TensorWasm — Trademark Policy

_Status: living document. First written for the v1.0 PATH-TO-V1 gate
(see [`PATH-TO-V1.md`](PATH-TO-V1.md), open decision 4: "Trademark
policy"). The policy here is the **permissive** branch of that
decision._

## Summary

The names **"Craton TensorWasm"** and **"TensorWasm"** are project names,
not registered trademarks. Forks, redistributions, and compatible
re-implementations may reuse the names; we only ask that substantive
forks rebrand for clarity, and that nobody claims an official
affiliation they do not have. Concerns about misuse go to
`security@craton.com.ar`.

## Names covered

This document covers three names. Each has a distinct status.

| Name | Use | Status |
|---|---|---|
| **Craton TensorWasm** | The full project name | Unregistered. Permissive use (see below). |
| **TensorWasm** | Short form of the project name | Unregistered. Permissive use (see below). |
| **Craton** | The corporate identity of Craton Software Company | Corporate name. Not granted as a project name. See [What's protected](#whats-protected). |

"Craton Software Company" is the current commercial sponsor of the
project and the copyright holder on the `LICENSE` and `NOTICE` files.
The project name and the company name are intentionally distinct, and
the policies that apply to them differ.

## Registration status

As of the v0.1.x era, **none of the three names above are registered
trademarks** in any jurisdiction. This is a deliberate choice, not an
oversight or a to-do.

The maintainers considered registering "TensorWasm" and decided the
costs (legal filings across jurisdictions, ongoing enforcement
obligations, the chilling effect on community forks) outweigh the
benefits at the project's current scale. The decision is recorded in
[`PATH-TO-V1.md`](PATH-TO-V1.md) under "Open decisions to resolve
before v1.0" → "Trademark policy" and is revisited as part of any
future governance amendment.

No `(R)` or `TM` symbol should appear next to any of these names in
the repository, the website, the release artifacts, or any official
communication, because no such claim is being made.

## What forks and derivatives MAY do

The project is Apache-2.0 licensed and the trademark stance is
intentionally permissive. Forks, redistributions, and compatible
implementations may, without asking permission:

1. **Reuse the names as-is when describing compatible derivatives.**
   Phrases like "Acme Inc's TensorWasm-compatible runtime", "a fork
   of Craton TensorWasm", or "drop-in replacement for TensorWasm" are
   fine. The names function descriptively in those contexts and we
   encourage clear attribution of compatibility.

2. **Fork and keep the original name short-term** while developing
   patches that will be upstreamed. Carrying the original name during
   a development branch that intends to merge back is normal practice
   and is welcome.

3. **Rename freely.** A fork that diverges substantively from the
   upstream codebase is encouraged to pick a new name; the maintainers
   will not contest the rename and have no naming approval over forks.
   Renaming is the cleanest path for forks that intend to live
   long-term as separate projects.

4. **Redistribute the source under Apache-2.0** under the original
   names, subject to the license terms (in particular Section 4 of
   `LICENSE` on retention of notices and Section 6 on the absence of
   any trademark grant).

5. **Build commercial products on top of the project** without
   royalty, attribution surcharge, or permission. Apache-2.0 grants
   that explicitly; this trademark policy does not narrow it.

## What we ASK derivatives to do

The following are requests, not legal requirements. We make them
because they keep the ecosystem easier to navigate for users; we do
not enforce them and have no mechanism to do so.

1. **Rebrand substantive forks.** If a fork diverges to the point
   that user-visible behavior differs (CLI flags removed, API
   semantics changed, security defaults loosened), please pick a new
   name. The risk we want to avoid is a user reading our docs,
   running someone else's binary, and being confused about which
   project's documentation applies — especially around security
   posture.

2. **Do not claim official affiliation in marketing copy.** Phrases
   like "the official TensorWasm distribution", "endorsed by Craton",
   or "the supported version" should be used only by the upstream
   project itself. "Built on TensorWasm", "compatible with
   TensorWasm", or "a fork of TensorWasm" are all fine — they
   accurately describe a relationship without implying an
   endorsement that does not exist.

3. **Note that you are a fork in your README.** A single sentence
   near the top of the fork's README pointing users back to the
   upstream project (and explaining why the fork exists) is enough.
   See [How to attribute upstream](#how-to-attribute-upstream) for
   suggested phrasing.

4. **Coordinate on security advisories.** If a fork inherits a
   vulnerability from upstream and an upstream advisory is published,
   please reference the upstream CVE in any advisory the fork issues.
   The reverse is also welcome — fork-discovered vulnerabilities
   reported back upstream are credited per `SECURITY.md`.

These are the requests; none of them carry legal weight. They are
the same kind of community norm that operates around other
Apache-2.0 projects (Wasmtime, Cranelift, OpenTelemetry) and we
expect they will be honored in good faith.

## What's protected

Two things are not subject to the permissive stance above.

1. **The Craton Software Company name and logo.** Craton Software
   Company is a corporate identity separate from the project name.
   The permissive trademark posture on "TensorWasm" does not grant
   anyone the right to operate under the Craton Software Company
   name, to imply that a fork is a Craton product, or to use the
   Craton logo on a derivative work's marketing materials. Use of
   the company name for descriptive attribution — "the project is
   sponsored by Craton Software Company", "originally developed at
   Craton" — is fine.

2. **The Apache-2.0 license terms.** Nothing in this document
   modifies the `LICENSE`. In particular, Section 6 of the Apache
   License explicitly states that the license does not grant
   permission to use the Licensor's trade names, trademarks, service
   marks, or product names except for descriptive use and the
   `NOTICE` file. This policy is consistent with that — it documents
   the maintainers' permissive stance on enforcement of the project
   name, but it does not pre-grant any rights the license withholds.

## Future tightening

This policy is a choice, not a permanent commitment. It can be
revisited via the governance amendment process described in
[`GOVERNANCE.md`](../GOVERNANCE.md#amendments-to-this-doc), which
requires an RFC under [`rfcs/README.md`](../rfcs/README.md) and a
2/3 supermajority of active maintainers.

Plausible reasons for revisiting:

- **A fork mimics the upstream name to deceive users on security
  advisories** — for example, distributing a backdoored binary
  labeled as the upstream project, or publishing fake CVE
  advisories under the project name. This is the case the
  trademark-veto language in
  [`GOVERNANCE.md`](../GOVERNANCE.md#vetoes) anticipates: a single
  maintainer may veto code or doc changes that misuse the project
  name to create legal risk.
- **A commercial entity uses the name in a way that systematically
  confuses paying customers** — for example, selling a "TensorWasm
  Pro" subscription with no relationship to upstream.
- **A jurisdiction-specific legal need arises** that registering the
  mark would resolve more cheaply than ad-hoc dispute resolution.

In each case, the response would be an RFC proposing a tightened
policy (registration, an explicit usage guide, a brand book) with
the rationale spelled out. The default until such an RFC merges is
the permissive stance documented above.

## How to attribute upstream

Suggested README phrasing for forks and derivatives. Use, adapt, or
ignore — these are starting points, not required language.

For a compatible re-implementation:

> Acme Runtime is a TensorWasm-compatible serverless Wasm runtime.
> It implements the [Craton TensorWasm HTTP API](https://github.com/craton-co/craton-tensor-wasm)
> and aims for binary compatibility with upstream TensorWasm
> snapshots. It is an independent implementation, not affiliated
> with or endorsed by Craton Software Company.

For a fork:

> This is a fork of [Craton TensorWasm](https://github.com/craton-co/craton-tensor-wasm).
> It exists to <one-sentence reason>. Patches that apply upstream
> are sent to the upstream tracker; fork-specific changes stay
> here. The maintainers of this fork are independent of Craton
> Software Company.

For a downstream distribution that tracks upstream releases:

> Acme TensorWasm Distribution packages upstream
> [Craton TensorWasm](https://github.com/craton-co/craton-tensor-wasm)
> releases for <platform>. We add packaging, init scripts, and
> security hardening defaults. The runtime itself is unmodified
> from upstream.

The phrase "not affiliated with or endorsed by Craton Software
Company" (or its equivalent) is the load-bearing piece in each
case — it sets reader expectations clearly and prevents confusion
about who is on the hook when something breaks.

## Reporting concerns

If you see a use of the project name that you believe is genuinely
misleading users — not "I disagree with their fork's technical
direction", but "this is being used to deceive" — send a note to
`security@craton.com.ar`. This is the same address used for
security disclosures (see [`../SECURITY.md`](../SECURITY.md)); the
Security committee triages incoming mail and forwards trademark
concerns to the maintainer group.

We do not run a takedown pipeline and we are not equipped to
litigate. The response to most reports will be a conversation with
the entity in question, asking them to clarify their attribution.
Where that fails, the maintainer group may publish a clarifying
statement in the project repository naming the situation. That is
the extent of our enforcement at v0.1; the future-tightening
section above describes when that posture might change.

## Cross-references

- [`PATH-TO-V1.md`](PATH-TO-V1.md) — open decision 4 records the
  decision to leave the names unregistered with a permissive policy.
- [`GOVERNANCE.md`](../GOVERNANCE.md#vetoes) — the trademark-veto
  ground that lets a maintainer block a PR misusing the project name.
- [`GOVERNANCE.md`](../GOVERNANCE.md#amendments-to-this-doc) — the
  amendment process this policy is revisited under.
- [`../LICENSE`](../LICENSE) — Section 6 (trademarks) of Apache-2.0.
- [`../NOTICE`](../NOTICE) — the attribution notices retained per
  Section 4 of the license.
- [`../SECURITY.md`](../SECURITY.md) — the `security@craton.com.ar`
  reporting channel shared with this policy.
- [`../README.md`](../README.md) — project status and the canonical
  upstream URL.

---

_This document codifies a permissive stance toward the project's
unregistered names. It is not legal advice. Contributors and
downstream users with jurisdiction-specific questions should consult
their own counsel; nothing here creates an attorney-client
relationship or supersedes applicable trademark law._
