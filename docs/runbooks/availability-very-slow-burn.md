<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# availability-very-slow-burn

Alert: `availability_http` is burning the 30-day error budget at **1×
the budgeted rate**, sustained across both a 6-hour and a 3-day
window. Severity: **ticket** (does not page).

## What this alert means

The HTTP API is returning 5xx responses at exactly the rate the SLO
allows — 0.5% — for long enough that the system will run out of
monthly error budget exactly at the 30-day mark. This is not an
incident; it is a warning that there is no cushion left. If anything
makes the failure rate worse the SLO will be missed, and if a deploy
introduces a new failure mode there is no slack to absorb it. The
alert files a ticket so the team investigates during normal hours
rather than at 3 AM. Defends the `availability_http: 99.5%` target in
[`SLO.md`](../SLO.md) §3 by giving the team a multi-day heads-up
before the budget is gone.

## Symptoms users see

- Almost certainly **none directly**. 0.5% failure rates are below
  the noise floor of most callers — a retry policy with two attempts
  reduces user-visible failures to ~25 parts per million.
- Internal dashboards may show baseline 5xx counts that have been
  creeping up over weeks.
- The dashboard's `availability_http (30d)` stat is exactly on the
  99.5% line, neither passing nor failing comfortably.

## First-look queries

```promql
# 1. Confirm: is the 6-hour error rate at or above 0.005?
# TODO: emit tensor_wasm_http_requests_total{route,method,status}
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h]))
  /
sum(rate(tensor_wasm_http_requests_total[6h]))
```

A value at or above `0.005` (= 1 × 0.005) confirms the alert.

```promql
# 2. Compare to a week ago: is the burn growing?
# TODO: emit tensor_wasm_http_requests_total{route,method,status}
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h]))
  /
sum(rate(tensor_wasm_http_requests_total[6h]))
  -
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h] offset 1w))
  /
sum(rate(tensor_wasm_http_requests_total[6h] offset 1w))
```

A positive delta says the failure rate has drifted up week-over-week.
A near-zero delta says the system has been steady at 0.5% — that
might be acceptable baseline if no specific cause exists, but it
means there's no room for any regression.

```promql
# 3. Which routes dominate the 5xx tail?
# TODO: emit tensor_wasm_http_requests_total{route,method,status}
topk(5,
  sum by (route, status) (
    rate(tensor_wasm_http_requests_total{status=~"5.."}[24h])
  )
)
```

The top five (route, status) pairs over a day. Useful for spotting a
specific handler/error combination that contributes most of the
budget consumption.

## Mitigation steps

This alert files a ticket; there is no on-call action under time
pressure. Treat the steps below as a triage checklist for the next
business day.

1. **Read the ticket against the dashboard.** Confirm the
   `availability_http (30d)` panel and the per-route error-rate
   panel. The dashboard view over 30 days is more informative than
   the alert's raw 6-hour window.
2. **Categorise the dominant failure**. From query 3, decide
   whether the dominant failures are:
   - **Bugs** — file specific issues, fix in the normal release
     cadence.
   - **Capacity** — the system is over-subscribed; plan a scale-up
     or a tenant offboarding.
   - **Acceptable baseline** — the failure is intrinsic (e.g. a
     known flaky guest module) and the team accepts it. Document
     the acceptance in the issue and move on.
3. **Decide whether the SLO target itself is right.** A 1× burn
   sustained over weeks may indicate the SLO is too tight for the
   actual workload, or that the workload mix has shifted. Either
   open an RFC to revise the SLO ([`SLO.md`](../SLO.md) §9 governs
   the process) or accept the burn as a known cost.
4. **Freeze risky deploys until the burn drops.** Per
   [`SLO.md`](../SLO.md) §4.5, when the budget is largely consumed
   the operator should freeze non-rollback deploys. At 1× burn there
   is no headroom, so it is reasonable to defer feature deploys
   until the rate falls back below 0.005.
5. **Schedule a follow-up review** in two weeks. If the alert is
   still firing, treat it as a sustained problem and escalate to a
   sev-2 incident (see **When to page**).

## Root-cause hypotheses

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| Genuine baseline of expected client errors masquerading as 5xx (e.g. handler returns 500 on malformed input it should reject as 400) | `journalctl -u tensor-wasm \| grep '500' \| head -50` and inspect what the handler logged | Fix the handler to return the correct 4xx status; the SLI excludes 4xx per [`SLO.md`](../SLO.md) §2.1 |
| One tenant generating a slow drip of failures (e.g. a buggy guest that crashes on 1% of inputs) | Per-tenant 5xx rate (requires the `tenant` label on `tensor_wasm_http_requests_total`, on the W2.3 TODO list); fall back to per-tenant termination rate today | Engage the tenant; help them debug their guest; raise a tenant-level SLO contract conversation |
| Snapshot subsystem occasionally times out under disk pressure (a slow tail of `504`) | Snapshot capture/restore P95 panels in the dashboard; correlate timestamps with the 5xx counts | Tune snapshot concurrency or move to faster storage; document if accepted |
| Recent dependency bump introduced an edge-case panic that fires rarely | `git log --since '2 weeks ago' -- Cargo.lock`; `journalctl -u tensor-wasm \| grep -i panic` | Revert the dependency or patch the call site |
| Network flap to an upstream auth backend producing intermittent 502s | Auth-backend health metrics; tracing spans showing intermittent failures on auth lookups | Improve auth-backend stability or add caching/retry in TensorWasm middleware |

## When to page

This is a ticket-severity alert; it does **not** page automatically.
Manually escalate to a page-equivalent incident if any of the following:

- The alert has fired continuously for **more than 14 days** — the
  baseline drift is now a sustained problem and warrants a focused
  investigation.
- Investigation reveals a security-relevant cause (e.g. failed
  responses correlate with attempted exploitation) — follow
  [`SECURITY.md`](../../SECURITY.md) disclosure procedure.
- The 30-day rolling availability drops below 99.5% — the SLO has
  been missed. The fast and slow burn alerts should have fired first;
  if they did not, fix the alerting before re-opening this ticket.

## Postmortem checklist

There is no incident retrospective for a ticket-severity alert, but a
follow-up note is still useful:

- Record the dominant failure category from step 2 in the ticket.
- If a tenant was engaged, link the tenant ticket from the runbook
  ticket so the resolution is visible.
- If the SLO target was revised, link the RFC that did so.
- If deploys were frozen, record when they were unfrozen and what
  metric triggered the unfreeze.
- Update this runbook's hypothesis table if the actual cause was
  novel — the very-slow-burn alert is most valuable as a long-tail
  detector and the table should accumulate the long tail over time.
- Close the ticket only when the 6-hour error rate has been below
  0.005 for at least 24 consecutive hours.

## Related

- [`SLO.md`](../SLO.md) §3 (target), §4.1 (budget), §5.3 (alert query),
  §4.5 (deploy freeze policy).
- [`availability-fast-burn.md`](availability-fast-burn.md) and
  [`availability-slow-burn.md`](availability-slow-burn.md) — the
  page-severity siblings. The very-slow alert is meant to fire
  *before* either of the others becomes likely.
- [`rollback.md`](rollback.md) — referenced if step 3 concludes the
  baseline is caused by a release rather than a workload shift.
- [`dashboards/README.md`](../dashboards/README.md) — the SLO summary
  row's `availability_http (30d)` stat and the HTTP traffic row's
  error-rate panel are the two views to track.
