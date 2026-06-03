<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# dispatch-latency-spike

Alert: P95 of `tensor_wasm_kernel_latency_seconds` is above the 50 µs
SLO threshold, sustained over both a 5-minute and a 1-hour window.
Severity: **page on host-only deployments**; informational on
CUDA-host deployments until the v0.4 CUDA-host SLO is set.

## What this alert means

Kernel dispatch — the path from a guest's `wasi_cuda.launch` call
through the back-pressure semaphore to dispatch completion — has a
95th-percentile latency above 50 µs over the last five minutes and
the last hour. On the host-only stub path, dispatch is essentially a
tokio semaphore round-trip plus a few microseconds of bookkeeping; a
P95 above 50 µs means the tokio scheduler is stalled, the semaphore
is saturated, or a recent change has added work to the path that
wasn't there before. Defends `latency_dispatch_serial_P95` from
[`SLO.md`](../SLO.md) §3, which is the only SLO in this directory
that is enforceable today (the underlying histogram exists in
`tensor-wasm-core` and does not require the W2.3 HTTP-metric work).

## Symptoms users see

- Guest programs that call `wasi_cuda.launch` in a tight loop see
  per-iteration time inflate proportionally.
- Invocations that issue many dispatches (e.g. a guest doing a
  thousand small kernels) take noticeably longer end-to-end.
- The dashboard's **Kernel latency** panel — particularly the P95
  trace — crosses its 50 µs threshold line and stays there.
- If the spike is severe, the
  [`invoke-latency-spike.md`](invoke-latency-spike.md) alert fires
  shortly after, because the dispatch latency is a component of the
  invoke latency.

## First-look queries

```promql
# 1. Confirm: is dispatch P95 above 50 µs over 5 minutes?
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_kernel_latency_seconds_bucket[5m])
  )
)
```

A value above `0.00005` confirms the alert.

```promql
# 2. Is the dispatch rate unusually high?
sum(rate(tensor_wasm_kernel_dispatches_total[5m]))
```

If the rate is normal but latency is elevated, the path itself has
slowed. If the rate is much higher than usual, the system is being
hammered and contention on the back-pressure semaphore is the likely
cause.

```promql
# 3. Are dispatches concentrated on one tenant?
topk(3,
  sum by (tenant) (
    rate(tensor_wasm_kernel_dispatches_total[5m])
  )
)
```

A single tenant dominating dispatches is a common cause — one tenant
saturating the back-pressure permits starves every other tenant's
calls. Note: the `tenant` label is on the W2.3 TODO list; until it
ships, this query falls through to a single aggregate series.

```promql
# 4. Compare dispatch latency to invoke latency.
histogram_quantile(0.95,
  sum by (le) (rate(tensor_wasm_kernel_latency_seconds_bucket[5m]))
)
/
on() group_left
histogram_quantile(0.95,
  sum by (le) (rate(tensor_wasm_http_request_duration_seconds_bucket{
    route="/functions/:id/invoke",
    method="POST"
  }[5m]))
  > 0
)
```

The ratio approaches 1 when dispatch dominates invoke latency — that
means the kernel path is the bottleneck. A small ratio means the
dispatch is fine and the invoke alert (if firing) has a different
cause. (TODO on the HTTP histogram.)

## Mitigation steps

Stop after the first step that brings dispatch P95 back below 50 µs.

1. **Identify the dominant tenant.** Run query 3. If one tenant
   dominates and the workload is unexpected (a runaway loop, a
   misconfigured client), terminate their instances:
   `tensor-wasm admin instance list --tenant <id>` then
   `tensor-wasm admin instance terminate <instance_id>`.
2. **Check the back-pressure permits.** On host-only deployments,
   the semaphore is sized at `concurrent_cap64` by default. If
   utilization is at 100% sustained, the system is under-provisioned.
   Raise the permit count via the config (`TENSOR_WASM_DISPATCH_PERMITS`)
   and restart; or scale horizontally (more nodes).
3. **Rule out tokio reactor stall.** If `/healthz` is also slow
   ([`healthz-slow.md`](healthz-slow.md)), the cause is process-wide
   reactor congestion, not dispatch-specific. Work the healthz
   runbook in parallel — the dispatch path uses the same reactor.
4. **On CUDA-host deployments, check GPU health.**
   `nvidia-smi -q -d UTILIZATION,MEMORY,ECC,PERFORMANCE` shows
   utilization, ECC errors, and clock throttling. A GPU at thermal
   limits or with ECC errors will dispatch slowly even on a healthy
   host path.
5. **Restart `tensor-wasm` if the dispatch path is wedged.** A
   semaphore deadlock or stuck future will not self-clear; capture
   `tensor-wasm observe --once` first, then
   `systemctl restart tensor-wasm`. The wedge usually traces back to
   a guest panic or a host-fn bug.
6. **Roll back if a recent deploy correlates.** A change in
   `tensor-wasm-exec` or `tensor-wasm-wasi-gpu` is the most common
   cause of a dispatch-path regression. Follow
   [`rollback.md`](rollback.md).

## Root-cause hypotheses

| Hypothesis | How to confirm | How to fix |
|---|---|---|
| Back-pressure semaphore saturated by one tenant | Per-tenant dispatch rate (query 3); permits-used metric (W2.3 TODO) at 100% | Throttle the tenant; raise permit count; scale out |
| Tokio reactor stalled by an unrelated handler | `/healthz` also slow ([`healthz-slow.md`](healthz-slow.md)); CPU low but latency high | Fix the blocking handler; restart as stopgap |
| GPU thermal throttling or ECC degradation (CUDA-host only) | `nvidia-smi` shows `Throttle Reasons` or non-zero ECC counters | Improve cooling; replace the device; revisit the SLO if the device is genuinely degraded |
| Recent deploy added work to the dispatch path (e.g. extra span attributes, new validation) | `git log --since '1 week ago' -- crates/tensor-wasm-exec/ crates/tensor-wasm-wasi-gpu/`; correlate to alert start | Roll back; reland the change with the cost measured |
| Snapshot capture happening on the same disk that backs PTX cache, contending for I/O | `iostat -x 5` shows high `%util` during dispatch spikes | Move snapshots to a separate volume; serialize snapshot capture |
| Guest stuck in a wasi_cuda.launch tight loop, dominating dispatch count without doing meaningful work | Tracing shows one instance issuing very many `wasi_cuda.launch` spans per second | Terminate the instance; engage the tenant about workload shape |

## When to page

The alert is page-severity on host-only deployments. Escalate to the
next tier if any of the following:

- Dispatch P95 stays above **500 µs** (10× the SLO) for any
  sustained 10-minute window — the path is structurally broken.
- The spike is accompanied by an
  [`invoke-latency-spike.md`](invoke-latency-spike.md) alert that
  cannot be mitigated by tenant isolation or restart.
- On CUDA-host nodes, `nvidia-smi` shows persistent ECC errors or
  the driver is unresponsive (the dispatch alert is informational on
  CUDA-host today, but a driver-level GPU problem warrants paging in
  its own right).
- Restart does not return dispatch P95 to baseline within 5 minutes
  of process startup — there is a workload or environment cause that
  the restart did not clear.

## Postmortem checklist

- Capture `tensor-wasm observe --once > /tmp/tensor-wasm-dispatch-$(date +%s).json` before any restart.
- Save `journalctl -u tensor-wasm --since '<incident_start - 10m>'`.
- On CUDA-host deployments, save `nvidia-smi -q > /tmp/nvidia-state.txt` for the same window.
- Snapshot the Prometheus metrics for the kernel-latency histogram and the dispatch counter across the incident window plus 10 minutes either side.
- File a follow-up issue with the dominant hypothesis from the table; if the cause is a tenant, link the tenant ticket.
- If a deploy was rolled back, note the from/to version per [`rollback.md`](rollback.md)'s requirements.
- Update this runbook if the actual cause was not in the hypothesis table — dispatch is the most varied incident surface in TensorWasm and the table benefits from accumulation.

## Related

- [`SLO.md`](../SLO.md) §3 (target), §5.5 (alert query); this is the
  one alert in [`SLO.md`](../SLO.md) that fires against an existing
  metric today.
- [`invoke-latency-spike.md`](invoke-latency-spike.md) — the `/invoke`
  latency alert often fires shortly after this one; they share root
  causes when the dispatch path is the bottleneck.
- [`healthz-slow.md`](healthz-slow.md) — process-wide reactor stalls
  surface in both runbooks; if `/healthz` is also slow the cause is
  shared.
- [`rollback.md`](rollback.md) — referenced in step 6.
- [`dashboards/README.md`](../dashboards/README.md) — the Kernel
  latency panel and the Back-pressure permit-utilization panel are
  the two primary visuals.
- [`OBSERVABILITY.md`](../OBSERVABILITY.md) — `wasi_cuda.launch` and
  `wasi_cuda.sync` spans cover the dispatch path; if metrics are
  inconclusive, drop down to traces.
