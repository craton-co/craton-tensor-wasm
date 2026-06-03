<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# healthz-slow

Alert: P95 latency of `GET /healthz` is above the 10 ms SLO threshold,
sustained over a 30-minute window. Severity: **ticket** (does not
page).

## What this alert means

The liveness endpoint `/healthz` is meant to be the cheapest possible
HTTP call TensorWasm exposes — no auth, no business logic, just a
"the axum router is alive" probe. When its P95 climbs above 10 ms it
means something is congesting the request path even though the
handler itself does no work: the tokio reactor is stalled, the
process is paging, the load balancer is sleeping, or the host is
saturated. This is a slow leading indicator rather than a user-facing
problem — `/healthz` slowness typically precedes `/invoke` slowness
by minutes to hours. Defends `latency_http_healthz_P95` from
[`SLO.md`](../SLO.md) §3 and exists primarily to give the team
warning before the harder-to-recover-from latency alerts fire.

## Symptoms users see

- None directly — `/healthz` is operator-facing.
- Load balancer dashboards may mark the TensorWasm instance "degraded"
  or "yellow" without taking it out of rotation.
- Synthetic monitors hitting `/healthz` report slightly elevated
  response times.
- Operators tail-watching `tensor-wasm observe` see the response
  field for `/healthz` print larger numbers than usual.

## First-look queries

```promql
# 1. Confirm: is the 30-minute P95 of /healthz above 10 ms?
# TODO: emit tensor_wasm_http_request_duration_seconds_bucket{route,method,status}
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/healthz",
      method="GET"
    }[30m])
  )
)
```

A value above `0.01` confirms. Compare against `0.030` and `0.100` to
classify the severity — at 100 ms `/healthz`, the host is in
trouble.

```promql
# 2. Is this isolated to /healthz or is the whole router slow?
# TODO: emit tensor_wasm_http_request_duration_seconds_bucket{route,method,status}
histogram_quantile(
  0.95,
  sum by (le, route) (
    rate(tensor_wasm_http_request_duration_seconds_bucket[30m])
  )
)
```

If only `/healthz` is slow, suspect a problem in the health-check
path itself. If every route is proportionally slower, the cause is
process-wide (CPU starvation, reactor stall, paging).

```promql
# 3. Is the host paging or short of CPU?
node_cpu_seconds_total{mode="iowait"}
```

If `node_exporter` is co-installed, a high `iowait` ratio explains
slow `/healthz` even when nothing in TensorWasm is the cause. If
`node_exporter` is not installed, fall back to `top` and `vmstat 5`
on the host.

```promql
# 4. Are active instances unusually high?
tensor_wasm_active_instances
```

Slow `/healthz` correlated with very high instance counts usually
means wasmtime engine churn is monopolising tokio executor threads.

## Mitigation steps

This is a ticket-severity alert; do not interrupt anyone for it. Work
the steps below in the next business day.

1. **Rule out the host.** `top`, `vmstat 5`, `iostat -x 5`. If the
   host is under load from a non-TensorWasm process (a backup job,
   another tenant on a shared host, a runaway log shipper), fix the
   host-level cause first and re-evaluate.
2. **Rule out paging.** `cat /proc/$(pidof tensor-wasm)/status | grep VmSwap`
   should report `0 kB`. If TensorWasm is paging, add memory, reduce
   the per-process memory cap, or move other workloads off the host.
3. **Restart during a planned window.** `systemctl restart tensor-wasm`
   often clears reactor-stall situations from accumulated wasmtime
   engine state. Because this is a ticket, schedule the restart for
   a low-traffic window — there is no urgency.
4. **Check for reactor-blocking handlers added recently.** Review
   `git log --since '2 weeks ago' -- crates/tensor-wasm-api/`. A
   handler that calls a synchronous I/O API without `spawn_blocking`
   stalls the reactor and slows every request, including
   `/healthz`. Fix forward in the next release.
5. **Verify the load balancer's own health is not the cause.** If
   `/healthz` is fast when measured directly on the host but slow
   through the LB, the LB itself is the issue, not TensorWasm.

## Root-cause hypotheses

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| Co-tenant on the host is consuming CPU or I/O | `top`, `iostat -x 5`, `nvidia-smi` (if a co-tenant uses the same GPU) | Move the co-tenant; isolate via cgroups; document the host's single-tenancy requirement |
| Tokio reactor blocked by a sync handler that should be `spawn_blocking` | `tokio-console` if attached; otherwise `perf top -p $(pidof tensor-wasm)` shows long stacks in a single handler | Refactor the handler to use async I/O or `spawn_blocking`; restart |
| TensorWasm process paging due to memory pressure | `cat /proc/$(pidof tensor-wasm)/status \| grep VmSwap` non-zero; `free -m` shows low free memory | Add RAM; reduce per-tenant memory cap; restart to drop accumulated state |
| Disk I/O saturation by snapshot capture/restore traffic | `iostat -x 5` shows high `%util` on the snapshot device; correlate with `tensor_wasm_active_instances` spikes | Move snapshots to a faster volume; throttle snapshot concurrency |
| Load balancer health-check interval too aggressive, sampling under a thundering herd | LB config; correlate spike timestamps with LB health-check schedule | Increase health-check interval (5-15 s is plenty); reduce probe concurrency |
| Process accumulating compiled wasmtime modules without eviction | `tensor_wasm_active_instances` very high and not draining; restart immediately fixes the symptom | Tune wasmtime engine module-cache size; investigate module-eviction policy |

## When to page

This alert does not page automatically. Manually escalate if any of
the following:

- `/healthz` P95 climbs above **100 ms** sustained for 10 minutes —
  the host is in trouble, not just slow.
- `/healthz` slowness coincides with an
  [`invoke-latency-spike.md`](invoke-latency-spike.md) alert
  firing — the leading indicator and the trailing indicator now both
  agree.
- The slowness coincides with availability burn — handle the burn
  first.
- The process is paging and free memory is below 100 MiB — risk of
  OOM kill is high.

## Postmortem checklist

There is no incident retrospective for a ticket alert, but file the
ticket with enough context to act on:

- `tensor-wasm observe --once` output captured during the slow
  window.
- `top -bn1`, `vmstat 5 3`, `iostat -x 5 3`, `free -m` from the host
  during the window.
- The dominant hypothesis from the table above.
- A pointer to the dashboard time range showing `/healthz` elevated.
- If a restart was used, note the time and the next-check timestamp.
- Close the ticket only after `/healthz` P95 has been below 10 ms
  for at least 24 hours.
- If the ticket recurs more than twice in a month, promote to a
  sev-2 incident — there is a structural cause that needs design
  attention.

## Related

- [`SLO.md`](../SLO.md) §3 (target), §5.4 (alert query).
- [`invoke-latency-spike.md`](invoke-latency-spike.md) — the
  page-severity sibling on the `/invoke` route; `/healthz` slow
  often foreshadows it.
- [`availability-fast-burn.md`](availability-fast-burn.md) — if
  `/healthz` slowness is severe enough that the LB takes the host
  out of rotation, this alert fires shortly after.
- [`rollback.md`](rollback.md) — referenced if step 4 traces back to
  a recent deploy.
- [`dashboards/README.md`](../dashboards/README.md) — the HTTP
  latency P50/P95/P99 panel renders `/healthz` alongside `/invoke`.
