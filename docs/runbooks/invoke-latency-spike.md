<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# invoke-latency-spike

Alert: P95 latency of `POST /functions/{id}/invoke` is above the SLO
threshold, sustained over both a 5-minute and a 1-hour window.
Severity: **page**.

## What this alert means

The 95th-percentile latency of invocation calls has exceeded its SLO
bound — `100 ms` for host-only invocations or `500 ms` for
invocations that issue at least one GPU dispatch (see
[`SLO.md`](../SLO.md) §3). The threshold has held for both a recent
5-minute window and the trailing 1-hour window, which rules out brief
GC stalls or single-request outliers. At this point, one in twenty
users is waiting longer than the SLO permits; perceived performance
is degraded even though the system is not returning errors. Defends
`latency_http_invoke_P95` and is the latency counterpart to the
availability burn alerts.

## Symptoms users see

- `tensor-wasm invoke` calls feel slow but still succeed; a CLI user
  notices the prompt takes noticeably longer than usual.
- Synchronous SDK callers see their P95 RPC times climb; downstream
  services that wrap TensorWasm calls in their own SLO start tripping
  their own latency budgets.
- Async batch consumers that use `JobRecord` polling see longer end-
  to-end completion times.
- Dashboard's **HTTP latency P50/P95/P99 by route** panel for
  `/functions/:id/invoke` crosses its threshold line and stays there.

## First-look queries

```promql
# 1. Confirm: is the 5-minute P95 over the SLO threshold?
# TODO: emit tensor_wasm_http_request_duration_seconds_bucket{route,method,status}
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/functions/:id/invoke",
      method="POST"
    }[5m])
  )
)
```

Compare to `0.1` (host-only SLO) or `0.5` (GPU-dispatch SLO). A value
above the relevant threshold confirms the alert.

```promql
# 2. Where in the stack is the latency? Compare against dispatch P95.
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_kernel_latency_seconds_bucket[5m])
  )
)
```

If kernel-dispatch P95 is also elevated, the GPU path is the cause.
If dispatch is healthy, the latency is in instance spawn, the WASM
execution body, or the HTTP middleware — not in the kernel.

```promql
# 3. Are spawns piling up faster than terminations?
sum(rate(tensor_wasm_instance_spawns_total[5m]))
  -
sum(rate(tensor_wasm_instance_terminations_total[5m]))
```

A positive value sustained over the alert window means instances are
accumulating (a leak or stuck-instance situation), which inflates
both latency and memory.

```promql
# 4. Is the GPU under memory pressure?
sum(tensor_wasm_gpu_memory_used_bytes)
```

Compare to the device's total. UVM page migration thrashing under
memory pressure can multiply dispatch latency by orders of magnitude
without producing any errors.

## Mitigation steps

Stop after the first step that brings P95 back under the SLO
threshold.

1. **Identify host-only vs GPU-dispatch path.** Run query 2. If
   dispatch P95 is fine but invoke P95 is bad, jump to step 3 (host-
   path issue). If dispatch P95 is also bad, follow
   [`dispatch-latency-spike.md`](dispatch-latency-spike.md) in
   parallel — the dispatch alert may also be firing.
2. **Check GPU memory pressure.** `nvidia-smi` shows used/total
   memory and any throttling. If memory is near saturation, identify
   the tenant consuming the most via the dashboard's "GPU memory by
   tenant" panel, and either:
   - terminate that tenant's instances via
     `tensor-wasm admin instance terminate <instance_id>`, or
   - revoke their token until they fix the workload.
3. **Look for instance leaks.** If query 3 shows spawns exceeding
   terminations, run `tensor-wasm observe --once` and inspect
   `tensor_wasm_active_instances`. A monotonically rising count over
   the alert window means a leak; restart `tensor-wasm` via
   `systemctl restart tensor-wasm` to clear and capture the
   correlated journalctl lines for the postmortem.
4. **Check the snapshot disk.** `iostat -x 5` for at least 30
   seconds. If `%util` is above 80% on the snapshot device, restores
   are blocking on I/O and inflating invoke latency. Move snapshots
   to a faster volume or reduce snapshot churn (raise the cache
   TTL).
5. **Roll back if a recent deploy correlates.** A deploy that added
   per-call work (extra middleware, expensive validation, a new
   wasmtime version with worse JIT) shows up here as a latency
   shift. Follow [`rollback.md`](rollback.md).

## Root-cause hypotheses

Ranked by base rate.

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| Snapshot restore on a cold function dominates per-call time | Snapshot capture/restore P95 panels elevated; `iostat -x 5` shows high `%util` on the snapshot volume | Faster snapshot storage; raise the warm-cache TTL; pre-warm hot functions on startup |
| GPU memory pressure causing UVM page migration thrash | `nvidia-smi` shows memory near full; `tensor_wasm_gpu_memory_used_bytes` climbing | Reduce per-tenant GPU memory cap; evict idle tenants; investigate the heaviest tenant's workload |
| One tenant issuing very large payloads (close to the 64 MiB body limit) | Per-route P99 climbing on `/invoke`; `journalctl -u tensor-wasm \| grep 'body too large\|large payload'` | Rate-limit the tenant or coach them to use chunked requests once that lands in v0.4 |
| Wasmtime JIT cold-compile on first-call of new modules | `tensor_wasm_active_instances` climbing alongside spawn rate; dashboard's JIT row shows cold-miss spike | Pre-warm modules; raise the module cache size |
| Back-pressure semaphore at or near saturation | `tensor_wasm_backpressure_permits_used / available` near 1.0 (this metric is in the W2.3 TODO list; until then, infer from dispatch P95 climbing without kernel-side cause) | Raise the back-pressure permit count; scale out (one node per N tenants) |
| Tokio reactor stalled by a synchronous handler call | Tracing shows long gaps inside `http.request` spans with no child progress; CPU usage low but latency high | Audit recent handler changes for blocking I/O; restart as a stopgap |

## When to page

The alert itself is page-severity. Escalate to the next tier if:

- P95 stays above `1.0 s` for any sustained 10-minute window — that
  is 2× the loosest SLO variant and indicates a structural problem.
- The invoke P50 (not just P95) crosses 500 ms, meaning the median
  user is now affected, not just the tail.
- The latency spike is accompanied by a fast-burn availability alert
  ([`availability-fast-burn.md`](availability-fast-burn.md)) — handle
  availability first.
- GPU memory is at or above 95% and not draining within 5 minutes of
  terminating the heaviest tenant.

## Postmortem checklist

- Capture `tensor-wasm observe --once > /tmp/tensor-wasm-latency-$(date +%s).json` before any restart.
- Save `journalctl -u tensor-wasm --since '<incident_start - 10m>'`.
- Snapshot the Prometheus metrics around the spike, especially the kernel-latency histogram and the snapshot capture/restore histograms.
- If `nvidia-smi` showed memory pressure, capture its output (`nvidia-smi --query-gpu=memory.used,memory.total --format=csv > /tmp/gpu-mem.csv`).
- File a follow-up issue with the dominant root cause from the table and the steps that worked; if no step worked and the spike auto-recovered, file the issue anyway with that observation.
- If a tenant was identified, link the incident from the tenant's onboarding ticket.
- Update this runbook's hypothesis table if the real cause was not listed.

## Related

- [`SLO.md`](../SLO.md) §3 (`latency_http_invoke_P95` target),
  §5.4 (alert query).
- [`dispatch-latency-spike.md`](dispatch-latency-spike.md) — if the
  cause is in the dispatch path, work both runbooks in parallel.
- [`healthz-slow.md`](healthz-slow.md) — if `/healthz` is also slow
  the problem is in the tower middleware or tokio reactor, not in
  the handler.
- [`availability-fast-burn.md`](availability-fast-burn.md) — a deep
  latency spike often correlates with a 5xx burn; check the
  availability alert first if both are firing.
- [`rollback.md`](rollback.md) — referenced in step 5.
- [`dashboards/README.md`](../dashboards/README.md) — the HTTP latency
  P50/P95/P99 panel and the tenant row's GPU memory panel are the
  primary visuals.
- [`OBSERVABILITY.md`](../OBSERVABILITY.md) — once the alert is open,
  the call tree in the span schema documents where to look for the
  latency hotspot.
