<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# availability-slow-burn

Alert: `availability_http` is burning the 30-day error budget at **6×
the budgeted rate**, sustained across both a 30-minute and a 6-hour
window. Severity: **page**.

## What this alert means

The HTTP API is returning 5xx responses to roughly 3.0% of all
incoming requests, and has been doing so steadily across at least the
last 30 minutes. At this rate, the full 30-day budget of ~216 minutes
of allowed downtime is consumed in about five days. This is the
"something is wrong but the building isn't on fire" alert: slow enough
that it would not have fired the 14.4× fast-burn page, fast enough
that ignoring it for a working day bleeds half the monthly budget.
Defends the same `availability_http: 99.5%` SLO as the fast-burn
alert ([`SLO.md`](../SLO.md) §3) but catches degradations that the
faster alert misses.

## Symptoms users see

- A noticeable but non-catastrophic minority of API calls fail —
  retries usually succeed, but the client error rate is above
  baseline.
- CLI clients see intermittent `Error: server returned 500` or
  occasional connection-reset errors that clear on retry.
- Long-running batch consumers that don't retry start logging
  elevated failure counts in their own dashboards.
- The dashboard's **HTTP error rate** panel sits visibly above the
  1% threshold line for a sustained period, without spiking high
  enough to be visually alarming.

## First-look queries

```promql
# 1. Confirm: is the 30-minute error rate above the alert threshold?
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[30m]))
  /
sum(rate(tensor_wasm_http_requests_total[30m]))
```

A value above `0.030` (= 6 × 0.005) confirms the alert. The
6-hour-window query (same expression with `[6h]`) confirms the
incident is steady-state, not a recent spike.

```promql
# 2. Scope: which route is failing?
topk(3,
  sum by (route) (
    rate(tensor_wasm_http_requests_total{status=~"5.."}[30m])
  )
)
```

The top three failing routes. A single dominant route narrows the
search to one handler; an even split across many routes points at a
shared dependency (snapshot disk, GPU, auth, DB).

```promql
# 3. Compare against the same window yesterday: is this new?
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[30m]))
  /
sum(rate(tensor_wasm_http_requests_total[30m]))
  -
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[30m] offset 1d))
  /
sum(rate(tensor_wasm_http_requests_total[30m] offset 1d))
```

A large positive delta means the elevated error rate is new in the
last 24 hours and worth correlating to a deploy or config change. A
near-zero delta means the system has been like this for a while and
the alert just crossed its window threshold.

```promql
# 4. Are the failures concentrated on one tenant?
sum by (tenant) (rate(tensor_wasm_instance_terminations_total[30m]))
```

This uses an existing metric to spot a tenant that is repeatedly
spawning and failing — a tenant-specific bug pattern is a common
slow-burn cause.

## Mitigation steps

The slow-burn rate is patient enough to leave room for diagnosis
before mitigation. Stop after step 2 if the dashboard recovers.

1. **Open the dashboard alongside the queries.** Confirm the SLO
   summary `availability_http (30d)` panel is dropping, and identify
   the dominant failing route from query 2.
2. **If the cause is obvious from logs, fix forward.**
   `journalctl -u tensor-wasm --since '6 hours ago' | grep -iE 'error|warn' | tail -200`
   often surfaces a recurring message that points at the cause (e.g.
   a tenant exceeding quota repeatedly, an upstream timeout). A
   targeted config change is faster than a rollback if the root cause
   is genuinely known.
3. **If a recent deploy correlates with the start of the burn, roll
   back.** Compare deploy timestamps (`git log --since '6 hours ago'`
   on the release branch, or your deployment system's audit log) to
   the 6-hour query window. Use [`rollback.md`](rollback.md) if the
   correlation is plausible — the slow burn at 6× still consumes
   budget quickly enough that an unnecessary rollback is cheaper than
   a missed real cause.
4. **If a single tenant is the cause, isolate them.** Use the
   `tensor-wasm-cli` admin commands to revoke or rate-limit the
   offending tenant's bearer token: `tensor-wasm admin token revoke <token-id>`
   or set a stricter QPS in the per-tenant config. Document the
   action for the postmortem.
5. **If no clear cause emerges in 30 minutes, escalate.** This alert
   is patient but not infinite. See **When to page**.

## Root-cause hypotheses

Ranked by base rate; the slow-burn alert tends to surface different
root causes than the fast-burn one.

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| One tenant is repeatedly hitting a quota or timeout, failing 100% of their calls | Per-tenant termination rate (query 4); `journalctl -u tensor-wasm \| grep '<tenant_id>'` | Revoke or rate-limit their token; engage the tenant; raise quota if legitimate |
| Snapshot disk degraded (slow but not yet failed) — restore calls increasingly time out at 30 s | `iostat -x 5` shows high `%util` on the snapshot device; `tensor-wasm observe --once` shows restore P95 climbing | Move snapshots to a faster volume; investigate the storage backend's own metrics |
| Memory leak slowly approaching the per-process limit; per-call allocations start failing before OOM kill | RSS climbing in `systemctl status tensor-wasm` or `cat /proc/$(pidof tensor-wasm)/status \| grep VmRSS`; `tensor_wasm_active_instances` not draining | Restart `tensor-wasm`; file an issue with the RSS-over-time graph |
| Wasmtime engine accumulating compiled modules without eviction | `tensor_wasm_active_instances` high and stable; restart drops error rate immediately | Tune the engine's instance limit downward; consider periodic warm restart until the leak is fixed |
| Upstream dependency (auth backend, OTLP collector) slow and triggering tower middleware timeouts | `curl` directly against the dependency; tracing spans show long `http.request` parent with no child progress | Restore or temporarily bypass the dependency; tighten the middleware timeout if it is too loose |

## When to page

Escalate to sev-1 if any of the following:

- The 30-minute error rate stays above 0.030 for more than **two
  hours** without identifying a hypothesis to act on.
- The slow burn upgrades to a fast burn (the
  [`availability-fast-burn.md`](availability-fast-burn.md) alert
  starts firing). Handle that page first — it is louder for a reason.
- The dashboard's `availability_http (30d)` panel crosses below
  99.3% (half the budget consumed in the rolling window).
- A single tenant is identified as the cause but cannot be isolated
  via token revocation (e.g. the auth subsystem is also degraded).

## Postmortem checklist

- Save `journalctl -u tensor-wasm --since '<incident_start - 30m>' --until '<incident_end + 30m>' > /tmp/tensor-wasm-slow-burn.log`.
- Capture the Prometheus snapshot for the full 6-hour window plus 30 minutes either side.
- If a tenant was isolated during mitigation, file an issue against that tenant's onboarding ticket (or notify their owner channel) so the action is undone or formalised.
- Note the actual root cause in the issue's body even if the dashboard recovered before it was confirmed — a slow burn that auto-resolves is still worth attributing.
- Update the per-tenant config or quota defaults if the cause was a quota that was set too high.
- If the cause traces back to a release, follow up with a regression test added before the next release.
- Cross-reference the incident in the next CHANGELOG entry under "Operator-visible behaviour change" per [`SLO.md`](../SLO.md) §9.

## Related

- [`SLO.md`](../SLO.md) §3 (target), §4.1 (budget), §5.2 (alert query).
- [`availability-fast-burn.md`](availability-fast-burn.md) — escalate
  here if this alert upgrades.
- [`availability-very-slow-burn.md`](availability-very-slow-burn.md)
  — the long-window ticket-severity variant of the same SLI.
- [`invoke-latency-spike.md`](invoke-latency-spike.md) — a slow
  upstream often surfaces here as both elevated latency AND elevated
  5xx rate; cross-check that runbook's queries.
- [`rollback.md`](rollback.md) — the rollback procedure step 3
  references.
- [`oncall-paging.md`](oncall-paging.md) — escalation path used in
  "When to page".
- [`dashboards/README.md`](../dashboards/README.md) — the HTTP traffic
  row's "Error rate (5xx) by route" panel is the primary visual.
