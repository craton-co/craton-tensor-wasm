<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Operator runbooks

This directory is the one-page-per-alert operator manual that makes the
v0.3 ("Production observability") gate of [`PATH-TO-V1.md`](../PATH-TO-V1.md)
real. Every alert defined in [`SLO.md`](../SLO.md) §5 has a runbook
here, and every runbook follows the same structure so an operator who
has paged once knows the shape of every other page on the rotation.

If you got paged and you have 60 seconds: open the runbook named in
the alert payload, scroll to **Mitigation steps**, do step 1, and
only then keep reading.

Status: **v0.3 gate**. The runbooks describe a single-host
self-hosted runtime as that is the only deployment topology TensorWasm
v0.3 supports. Multi-host failover, k8s operators, and cross-region
procedures are out of scope until at least v0.4.

## Alert → runbook index

The table below matches [`SLO.md`](../SLO.md) §7 exactly. If a row in
that document changes, this table changes with it in the same PR.

| Alert (SLO.md §5) | Severity | Runbook |
|---|---|---|
| Availability fast burn (14.4×, 5 m + 1 h) | Page | [`availability-fast-burn.md`](availability-fast-burn.md) |
| Availability slow burn (6×, 30 m + 6 h) | Page | [`availability-slow-burn.md`](availability-slow-burn.md) |
| Availability very-slow burn (1×, 6 h + 3 d) | Ticket | [`availability-very-slow-burn.md`](availability-very-slow-burn.md) |
| `/invoke` latency spike (P95 > SLO, 5 m + 1 h) | Page | [`invoke-latency-spike.md`](invoke-latency-spike.md) |
| `/healthz` slow (P95 > 10 ms, 30 m) | Ticket | [`healthz-slow.md`](healthz-slow.md) |
| Dispatch latency spike (P95 > 50 µs, 5 m + 1 h) | Page (host-only) | [`dispatch-latency-spike.md`](dispatch-latency-spike.md) |
| Rollback procedure | (manual) | [`rollback.md`](rollback.md) |
| On-call paging procedure | (manual) | [`oncall-paging.md`](oncall-paging.md) |

A runbook in this directory without a corresponding entry in
[`SLO.md`](../SLO.md) §7 is a documentation bug; report it in the
sibling repo issue tracker rather than leaving it stranded.

## Procedure / companion runbooks

These documents are not triggered by an SLO alert, so they have no
row in the table above. They are manual procedures or companion
recipes the on-call runs deliberately. A runbook in this directory
without a corresponding entry — in the table above *or* the list
below — is a documentation bug.

| Runbook | Purpose |
|---|---|
| [`disaster-recovery.md`](disaster-recovery.md) | Bring a deployment back online after a lost host, lost storage, or lost auth state. |
| [`cve-disclosure-dry-run.md`](cve-disclosure-dry-run.md) | Rehearse the CVE disclosure pipeline end-to-end without an actual vulnerability. |
| [`trace-id.md`](trace-id.md) | Companion recipe: pivot from a captured `x-trace-id` to the related logs and distributed trace. |
| [`ghcr-registry-provisioning.md`](ghcr-registry-provisioning.md) | Sponsor-only procedure to provision the `ghcr.io/craton-co/tensor-wasm` container-registry namespace. |
| [`self-hosted-cuda-runner.md`](self-hosted-cuda-runner.md) | Register the self-hosted GitHub Actions runner the `cuda` CI workflow requires. |

(The two procedure runbooks already listed in the table above —
[`rollback.md`](rollback.md) and [`oncall-paging.md`](oncall-paging.md)
— remain there because [`SLO.md`](../SLO.md) §7 references them.)

## Runbook contract

Every alert runbook in this directory uses the same nine H2 sections,
in the same order, so an on-call reading their fifth page of the week
does not have to relearn the layout. Sections in italics may be empty
when genuinely not applicable, but the heading itself must be present:

1. **What this alert means** — three to five sentences. Plain English.
   No source-code reading required. Names the SLO it is defending.
2. **Symptoms users see** — bullet list, from the outside in. What a
   caller of `POST /functions/{id}/invoke` notices before the alert
   wakes the operator.
3. **First-look queries** — two to four ready-to-paste PromQL queries
   the operator runs first to confirm and scope the incident, each
   annotated with what the result means.
4. **Mitigation steps** — numbered list, fastest fix first. Each step
   names a concrete tool: `tensor-wasm observe`, `systemctl`,
   `journalctl`, `curl`, `nvidia-smi`, `docker`. No invented commands.
5. **Root-cause hypotheses** — table mapping hypothesis to confirmation
   query/log to remediation. The table is not exhaustive; it is the
   five most likely causes ranked by base rate on a single-host
   deployment.
6. **When to page** — explicit severity-1 criteria for escalating from
   "operator handling" to "wake the on-call rotation". Names a numeric
   threshold or a duration, not a feeling.
7. **Postmortem checklist** — five to seven bullets covering what to
   capture before the evidence rotates out of `journalctl`/Prometheus
   retention, who to notify, and how to file the follow-up issue.
8. **Related** — cross-references to other runbooks, the relevant
   dashboard panels in [`dashboards/README.md`](../dashboards/README.md),
   and the SLO clauses being defended.

Two of the documents in this directory — [`rollback.md`](rollback.md)
and [`oncall-paging.md`](oncall-paging.md) — are *procedure* runbooks
rather than *alert* runbooks. They do not follow the nine-section
contract because there is no alert firing; they document a manual
operation the on-call performs. They are still referenced from
[`SLO.md`](../SLO.md) §4.5 and §7 so they live here for proximity.

## Voice and style

These runbooks borrow their voice from [`PATH-TO-V1.md`](../PATH-TO-V1.md)
and [`SLO.md`](../SLO.md): declarative, conservative, no marketing,
no emojis. A runbook is not the place to discover the right mitigation
through inspiration; it is the place to execute a step that was
agreed on calmly, weeks before the incident, by people who could
think clearly.

When a step requires a judgement call (e.g. "decide whether to roll
back or keep debugging"), the runbook names the threshold that should
tip the decision, not the operator's intuition.

## Honest gaps

Several of the alerts in [`SLO.md`](../SLO.md) §5 reference metrics
that are not yet emitted by `tensor-wasm-api` (see SLO.md §8 — the
"TODO" inventory at the bottom). Those alerts cannot fire today; the
runbooks are written against the metric names the alerts will use
once the instrumentation lands in W2.3. Each affected query is
flagged inline with the same `<!-- TODO: emit this metric -->`
comment used in [`SLO.md`](../SLO.md), so the gap is visible at the
point of reading.

A runbook that references a not-yet-emitted metric is still useful
today: it tells the operator what to do when the metric does exist
and the alert does fire. It is also useful as a code-review prompt
when the instrumentation PR lands — every query in this directory
should resolve cleanly against the new metric without further
editing.

## Cross-references

- [`SLO.md`](../SLO.md) — SLI/SLO definitions and the burn-rate alert
  expressions the runbooks defend.
- [`dashboards/README.md`](../dashboards/README.md) — panel inventory;
  every "First-look queries" section in a runbook should have a
  matching dashboard panel.
- [`PATH-TO-V1.md`](../PATH-TO-V1.md) — the v0.3 exit criterion
  "Runbook for every alert" is satisfied by the runbooks listed in
  the table above.
- [`OBSERVABILITY.md`](../OBSERVABILITY.md) — the tracing schema;
  several runbooks reference the spans there for the "deep dive"
  step after mitigation.

---

_Status: v0.3 gate. The runbooks are conservative for a single-host
self-hosted runtime; revisit at v0.4 once rate limiting, mTLS, and
multi-host topologies enter scope._
