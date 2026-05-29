<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# availability-fast-burn

Alert: `availability_http` is burning the 30-day error budget at **14.4×
the budgeted rate**, sustained across both a 5-minute and a 1-hour
window. Severity: **page**.

## What this alert means

The HTTP API is returning 5xx responses to roughly 7.2% of all
incoming requests right now, and it has been doing so for at least
five consecutive minutes — long enough that a transient blip has been
ruled out. At this rate, the entire 30-day availability budget of
~216 minutes of allowed downtime is consumed in about 50 hours.
This is the loudest availability alert TensorWasm defines because
sustained 14.4× burn is incompatible with the 99.5% SLO; if it is
real, the binary is failing requests faster than the budget can
absorb. Defends the `availability_http: 99.5%` target documented in
[`SLO.md`](../SLO.md) §3.

## Symptoms users see

- `POST /functions/{id}/invoke` returns HTTP 500, 502, 503, or 504
  for many or most calls.
- CLI clients using `tensor-wasm invoke` print
  `Error: server returned 500 internal error` or a connection-reset.
- Synthetic monitors hit the same error rate from multiple regions
  (rules out a single-client network problem).
- The dashboard's **SLO summary** top-row stat for
  `availability_http (30d)` is below 99.5% and shrinking visibly
  between refreshes.

## First-look queries

Paste these into the Prometheus query bar in the order shown. The
first confirms the alert is real, the second scopes it by route, the
third scopes it by status family, and the fourth shows the burn
graphically.

```promql
# 1. Confirm: is the 5-minute error rate above the alert threshold?
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[5m]))
  /
sum(rate(tensor_wasm_http_requests_total[5m]))
```

A value above `0.072` (= 14.4 × 0.005) means the alert is genuine.
Below it means the alert is in the process of resolving or is a
flapping false positive.

```promql
# 2. Scope: which route is failing?
sum by (route) (
  rate(tensor_wasm_http_requests_total{status=~"5.."}[5m])
)
```

One dominant route means a handler-local bug or a downstream
dependency that route uses. Many routes failing together means a
process-wide problem (panic in a shared layer, OOM, file descriptor
exhaustion).

```promql
# 3. Scope: which status code dominates?
sum by (status) (
  rate(tensor_wasm_http_requests_total{status=~"5.."}[5m])
)
```

`500` points at a handler panic or an `internal` error envelope.
`502/503` points at the process being down or refusing connections.
`504` points at the per-call deadline tripping — usually upstream
(GPU, snapshot disk, JIT) is slow, not failed.

```promql
# 4. Trend: how long has this been bad?
sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[1h]))
  /
sum(rate(tensor_wasm_http_requests_total[1h]))
```

If the 1-hour rate is also above 0.072 the incident has been running
for the full hour — not a spike. If the 1-hour rate is much lower than
the 5-minute rate, the failure started recently and the burn is
accelerating.

## Mitigation steps

Do these in order. Stop as soon as the 5-minute error rate (query 1
above) drops below 0.005.

1. **Confirm process is alive.** `systemctl status tensor-wasm` (or
   `docker ps | grep tensor-wasm` for the container deployment). If
   the process is dead or restart-looping, skip to step 4.
2. **Drain new traffic.** If the deployment sits behind a load
   balancer, set the backend's health check to fail intentionally
   (e.g. `curl -X POST http://localhost:9000/admin/drain` if your LB
   exposes that, or pull the host out of the LB's target group
   manually). This buys time without bouncing the process.
3. **Capture state, then restart.** Before restarting, run
   `tensor-wasm observe --once > /tmp/tensor-wasm-state-$(date +%s).json`
   to snapshot the metric exposition for the postmortem, then
   `systemctl restart tensor-wasm`. A fresh process clears most
   stuck-state failure modes (file descriptor leak, accumulated
   wasmtime engine churn, deadlocked tokio reactor).
4. **If restart does not help, roll back.** Follow
   [`rollback.md`](rollback.md) to revert to the last known-good
   release. The fast-burn alert at 14.4× means the budget is being
   consumed faster than a debug-and-fix cycle can return — rollback
   first, debug after.
5. **Reduce blast radius if rollback is unsafe.** If the current
   release cannot be rolled back (schema migration, snapshot format
   change), narrow exposure: scale the process down to handle only
   a fraction of tenants via the auth-token allowlist, or take the
   node offline entirely and serve maintenance from upstream.

## Root-cause hypotheses

Ranked by base rate on a single-host self-hosted runtime.

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| Bad deploy: a recent release introduced a panic on a common code path | `journalctl -u tensor-wasm --since '1 hour ago' \| grep -i panic`; compare deploy time to alert-firing time | Roll back via [`rollback.md`](rollback.md) |
| OOM kill loop: the host is out of memory and the kernel keeps killing the process | `journalctl -k --since '1 hour ago' \| grep -i 'killed process'`; `dmesg \| tail -50` | Reduce per-tenant memory caps in `tensor-wasm-mem` config; restart; investigate guest with `tensor-wasm observe` |
| Snapshot disk full or read-only | `df -h` on the snapshot directory; `journalctl -u tensor-wasm \| grep -i 'snapshot.*ENOSPC\|read-only'` | Free space or remount; restart `tensor-wasm` |
| GPU driver hang propagating into 5xx | `nvidia-smi` hangs or shows `Unable to determine`; dispatch span timings in tracing UI flatlined | Restart the NVIDIA driver (`sudo systemctl restart nvidia-persistenced && sudo nvidia-smi -r`); restart `tensor-wasm`; consider host reboot |
| Upstream auth backend down (if bearer tokens are validated against a remote source) | `curl` directly against the auth backend; check `journalctl` for `auth backend unreachable` | Restore the auth backend; TensorWasm should recover automatically once auth round-trips succeed |

## When to page

The alert itself is severity-page. Escalate to the **next** tier
(wake the on-call lead, file a sev-1 incident, notify the
maintainers list) if any of the following are true:

- The 5-minute error rate from query 1 stays above 0.072 for more
  than **15 minutes** after a rollback attempt.
- The process keeps restart-looping (more than three restarts within
  10 minutes per `systemctl status`).
- The error rate jumps from a single-route problem to "every route is
  failing" while the operator is working the incident.
- The dashboard's `availability_http (30d)` panel crosses below
  99.0% — that's twice the SLO floor, and qualifies as a
  budget-emergency under [`SLO.md`](../SLO.md) §4.5.

## Postmortem checklist

- Save `journalctl -u tensor-wasm --since '<incident_start - 10m>' --until '<incident_end + 10m>' > /tmp/tensor-wasm-incident.log` before journald rotates the relevant lines.
- Save the `/tmp/tensor-wasm-state-*.json` captures taken during steps 3 and 4 of mitigation.
- Snapshot the Prometheus metrics for the incident window (Grafana Explore → CSV export on the SLO summary panels).
- File a follow-up issue in the repo: title `incident: availability-fast-burn YYYY-MM-DD`, body containing the root-cause hypothesis from the table above plus the actual cause.
- If rollback was used, note the from/to release versions; the rollback procedure asks for this.
- Notify the maintainers list per [`SECURITY.md`](../../SECURITY.md) if the root cause is suspected to be a vulnerability rather than a bug.
- Update this runbook if the actual cause was not in the hypothesis table — add a row with what to check next time.

## Related

- [`SLO.md`](../SLO.md) §3 (target), §4.1 (budget), §5.1 (alert query).
- [`availability-slow-burn.md`](availability-slow-burn.md) — the slower
  cousin of this alert; same SLI, lower threshold, larger window.
- [`availability-very-slow-burn.md`](availability-very-slow-burn.md)
  — the ticket-severity variant.
- [`rollback.md`](rollback.md) — the rollback procedure step 4
  references.
- [`oncall-paging.md`](oncall-paging.md) — escalation path used in
  "When to page".
- [`dashboards/README.md`](../dashboards/README.md) — the SLO summary
  row and the HTTP-traffic row are the two panel groups operators
  watch during this incident.
- [`OBSERVABILITY.md`](../OBSERVABILITY.md) — span schema; if the
  metric-only investigation doesn't surface root cause, drop down to
  traces via the configured OTLP collector.
