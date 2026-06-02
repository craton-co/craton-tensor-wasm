<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# oncall-paging

Manual procedure for escalating from "operator handling" to "wake the
on-call rotation". Not an alert; referenced by [`SLO.md`](../SLO.md)
§4.5 and by the "When to page" section of every alert runbook in
this directory. Severity: **manual** (the operator decides when to
escalate based on the criteria in the calling runbook).

## When to use this procedure

This document covers escalation to the **next** tier — typically a
maintainer or on-call lead who is not the first responder. It does
**not** describe how to react to an automated page; that is the job
of the per-alert runbooks.

Escalate via this procedure when:

- An alert runbook's "When to page" section names a threshold and
  the threshold has been crossed.
- An incident has been worked for **more than 30 minutes** by the
  first responder without convergent progress.
- The cause is identified but the fix requires permissions or
  knowledge the first responder does not have (e.g. release-signing
  keys, datastore credentials, vendor escalation).
- A security-relevant incident is suspected (follow
  [`SECURITY.md`](../../SECURITY.md) in parallel; the security
  pipeline and the operational pipeline both need to know).
- More than one severity-1 alert is firing simultaneously (one
  responder is not enough).

If none of these are true, do not page — work the alert runbook.

## How to page

The exact channel depends on the deployment's on-call setup. This
document does not pin a vendor (PagerDuty, Opsgenie, VictorOps,
homegrown). The reference deployment uses the following pattern;
adapt to local tooling.

### 1. Decide the severity

- **Sev-1** — user-visible outage or sustained SLO miss; the
  on-call lead joins the call within 15 minutes.
- **Sev-2** — degraded service that is not a full outage, or a
  ticket-severity alert that has recurred too often; on-call lead
  joins within 60 minutes.
- **Sev-3** — operational nuisance; file an issue, do not page.

The threshold each alert runbook names in its "When to page"
section maps directly to one of these levels. If the calling
runbook didn't specify, default to sev-2.

### 2. Open the incident channel

```sh
# Reference: a chat-ops bot that creates a shared incident channel.
# Replace with the equivalent in the operator's environment.
incident open --severity sev-1 \
              --summary "tensor-wasm: <alert name> firing on <host>" \
              --runbook docs/runbooks/<calling-runbook>.md
```

The channel is the durable record of the incident — every
mitigation step, every dashboard screenshot, every decision lands
there.

### 3. Page the on-call

```sh
# Reference: a pager bot that resolves the "tensor-wasm-oncall"
# rotation to the current on-call person and sends them an
# acknowledgeable page.
page --rotation tensor-wasm-oncall \
     --severity sev-1 \
     --message "tensor-wasm <alert> firing; incident channel: <link>"
```

The page must contain:

- The alert name (matches a filename in `docs/runbooks/`).
- The host or deployment identifier.
- A link to the incident channel.
- A link to the dashboard time range showing the alert.

A page without the runbook link makes the on-call do extra work
under stress; do not omit it.

### 4. Notify the maintainers list

For sev-1 incidents, send a short email or chat to the maintainers
list with the same information. Maintainers do not necessarily join
the response, but they need to know an incident is live in case
related work is in flight or a release needs holding.

For sev-2 incidents, notification is optional; use judgement based
on the maintainers' usual preference.

### 5. Confirm the page was received

The on-call's pager system should acknowledge within 5 minutes. If
no acknowledgement arrives:

1. Page the **secondary** rotation (`page --rotation tensor-wasm-oncall-secondary ...`).
2. If still no acknowledgement, page the on-call **lead** directly
   by name.
3. If still nothing, the rotation itself is broken — file that as
   an incident in its own right and route to whoever manages the
   pager system.

## After the on-call joins

Hand off cleanly. The first responder is the better-informed
participant for the first 15-30 minutes; do not abandon the incident
on handover.

1. Summarise in the incident channel: what alert fired, when, what
   was tried, what worked, what did not.
2. Confirm the on-call has read the calling runbook.
3. Hand off the dashboard tabs, the `journalctl` window, and the
   shell sessions you have open.
4. Stay on the channel for at least 15 minutes after the handover in
   case the on-call has questions about earlier context.
5. The on-call decides when the incident is over and writes the
   incident summary in the channel.

## Sev-1 deferral criteria

Some sev-1 incidents do **not** warrant waking a human in the
middle of the night. The on-call may defer to morning if:

- The mitigation is in place (e.g. rollback completed, alert
  cleared) and the postmortem is the only remaining work.
- The incident is contained to a single tenant and that tenant has
  been notified.
- No customers are user-visibly affected and the SLO budget impact
  is below 10% of the monthly allowance.

Defer by setting the incident status to "mitigated, postmortem
pending" and scheduling the review for the next business day.
Document the deferral decision in the incident channel.

## Pre-incident readiness checks (operator hygiene)

Once per month, run these to make sure the paging path itself works:

- Send a **test page** to the rotation: `page --test --rotation tensor-wasm-oncall`.
  Confirm the on-call receives and acknowledges within 5 minutes.
- Confirm the secondary rotation has an actual person (rotations
  drift as team membership changes).
- Confirm the maintainers list has current addresses.
- Verify each runbook in `docs/runbooks/` opens from the alert
  payload's runbook link without 404ing.
- Verify the dashboard URL in the alert payload resolves to the
  correct dashboard.

A paging path that fails its monthly test is a sev-2 issue in its
own right.

## Related

- [`SLO.md`](../SLO.md) §4.5 — the error-budget threshold that
  triggers paging.
- Every alert runbook in this directory has a "When to page"
  section that references this procedure.
- [`rollback.md`](rollback.md) — the most common action taken
  during a paged incident.
- [`SECURITY.md`](../../SECURITY.md) — the parallel disclosure
  pipeline for security-relevant incidents.
- [`MAINTAINERS.md`](../../MAINTAINERS.md) — the current
  maintainer list, which feeds the on-call rotation and the
  maintainer notification list.
