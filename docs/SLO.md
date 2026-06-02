<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Service Level Objectives (v0.3)

This document is the project's commitment to numeric availability,
latency, and error-rate targets for the public HTTP surface and the
kernel-dispatch path. It is the operator-facing contract: every alert
in `docs/runbooks/` and every panel in
`docs/dashboards/tensor-wasm-overview.json` traces back to a Service
Level Indicator (SLI) and a Service Level Objective (SLO) defined
here.

Status: **v0.3 gate**. The targets are conservative, intentionally
under-promised, and tied to the host-only baselines measured today
(`bench-results/baseline.json`). They tighten in v0.4-v0.5 once the
S22 self-hosted CUDA runner replaces the modeled CUDA estimates with
measured numbers.

## Contents

1. [Purpose](#1-purpose)
2. [SLI definitions](#2-sli-definitions)
3. [SLO targets](#3-slo-targets)
4. [Error budget calculations](#4-error-budget-calculations)
5. [Burn-rate alerts](#5-burn-rate-alerts)
6. [Dashboards](#6-dashboards)
7. [Runbook mapping](#7-runbook-mapping)
8. [Disclosure: measured vs modeled](#8-disclosure-measured-vs-modeled)
9. [How to change an SLO](#9-how-to-change-an-slo)

---

## 1. Purpose

An **SLO** in this document is an externally-visible promise about
the behaviour of a running TensorWasm node: "the HTTP API is up at
least X% of the time, and 95% of `/invoke` calls finish within Y
milliseconds". When an SLO is missed for long enough that the error
budget is consumed, the operator is expected to act — roll back the
release, page the on-call, freeze deployments, or open an incident.
SLOs are about user-visible behaviour, not about internal
implementation details.

This is **not the same as a bench target.** The numbers in
[`PERFORMANCE.md`](PERFORMANCE.md) are CI regression guards measured
in micro-benchmarks under controlled conditions: a single host with
no contention, Criterion warm-up, outlier rejection. The numbers in
this document are operational bounds under real load: noisy
neighbours, network jitter, OS scheduler stalls, log-rotation
hiccups, snapshot capture happening on the same disk. As a rule of
thumb, an SLO target is one to three orders of magnitude looser than
the corresponding bench median, because production includes
everything the bench excludes.

If a bench regression fires but an SLO does not, the regression is
worth investigating but is not necessarily a user-visible problem.
If an SLO fires but the benches are green, the production stack is
hiding behaviour the benches do not exercise — that is a higher-
priority signal.

---

## 2. SLI definitions

Each SLI is a single Prometheus expression returning a unitless
ratio (for availability and error-rate) or a duration in seconds (for
latency). The expressions assume the standard Prometheus scrape
interval (15 s) and standard tower-http instrumentation. As of W2.3,
every metric referenced below is emitted by the gateway — the PromQL
runs as-is against a live `/metrics` scrape.

The metrics that **do** exist today, audited against
`crates/tensor-wasm-core/src/metrics.rs`, are:

- `tensor_wasm_active_instances` (gauge)
- `tensor_wasm_gpu_memory_used_bytes` (gauge)
- `tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id}` (gauge family,
  C3)
- `tensor_wasm_kernel_dispatches_total` (counter)
- `tensor_wasm_kernel_latency_seconds` (histogram, 14 buckets,
  10 µs – 10 s)
- `tensor_wasm_instance_spawns_total` (counter)
- `tensor_wasm_instance_terminations_total` (counter)
- `tensor_wasm_offload_success_total` (counter)
- `tensor_wasm_offload_fallback_total` (counter)
- `tensor_wasm_jobs_active` (gauge, single series, C3)

The HTTP-level metrics referenced below
(`tensor_wasm_http_requests_total`,
`tensor_wasm_http_request_duration_seconds`) are emitted by the
`tensor_wasm_api::http_metrics` middleware as of W2.3. Labels are
`method` (`GET`, `POST`, `DELETE`), `route` (the matched axum route
template, e.g. `/functions/:id/invoke`, never the substituted value),
and `status` (the numeric HTTP status code as a string). The
PromQL expressions below are executable against a running gateway.

### 2.1 `availability_http`

Ratio of non-5xx HTTP responses to all HTTP responses over the
trailing 30 days.

```promql
sum(rate(tensor_wasm_http_requests_total{status!~"5.."}[30d]))
  /
sum(rate(tensor_wasm_http_requests_total[30d]))
```

Rationale for excluding `4xx`: a `404` on `GET /functions/<unknown>`
or a `401` on a missing bearer token is a client-side error, not a
TensorWasm fault. We report client errors in the dashboard but do not
spend availability budget on them. `429` is similarly excluded: the
rate limiter doing its job is not a service failure.

### 2.2 `latency_http_healthz`

P95 latency of `GET /healthz` over the trailing 5 minutes.

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/healthz",
      method="GET"
    }[5m])
  )
)
```

`/healthz` is the liveness probe; it does no work and exists to
detect that the process is serving. Its P95 is a pure proxy for
"is the axum router stuck".

### 2.3 `latency_http_invoke`

P95 end-to-end latency of `POST /functions/{id}/invoke` over the
trailing 5 minutes. The 30-second per-call deadline (see
`API.md` middleware section) is the worst-case ceiling; the SLO
is far tighter than that for the common path.

```promql
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

Two flavours of this SLO are tracked separately (see
[SLO targets](#3-slo-targets)): one for host-only invocations
(guest never touches WASI-CUDA), one for invocations that issue at
least one kernel dispatch. They are distinguished post-hoc in the
dashboard via tenant-tagged join on
`tensor_wasm_kernel_dispatches_total`; the underlying histogram is
a single series.

### 2.4 `latency_dispatch_serial`

P95 of single-stream kernel dispatch latency (back-pressure permit
acquire → future poll → completion) over the trailing 5 minutes.
This is measured against the existing
`tensor_wasm_kernel_latency_seconds` histogram.

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_kernel_latency_seconds_bucket[5m])
  )
)
```

Note: the existing histogram does not distinguish serial from
concurrent dispatch, nor host-stub from real CUDA. Once the S22
runner lands and a real CUDA event-based sync replaces the immediate-
resolve stub described in `RISKS.md`, this SLI will split into
`{dispatch_kind="serial"}` and `{dispatch_kind="concurrent"}` series.
Until then, the SLO covers the aggregate.

### 2.5 `error_rate_invoke`

Ratio of 5xx responses on `/invoke` to all responses on `/invoke`,
trailing 5 minutes. Aligned with the API error envelope in
`crates/tensor-wasm-api/API.md` — `wasmtime`, `internal`, and
`invoke_timeout` (a 504 — counted here because it indicates a TensorWasm
failure to meet its own deadline) all roll up under "5xx for the
purposes of this SLI".

```promql
(
  sum(rate(tensor_wasm_http_requests_total{
    route="/functions/:id/invoke",
    method="POST",
    status=~"5..|504"
  }[5m]))
)
/
(
  sum(rate(tensor_wasm_http_requests_total{
    route="/functions/:id/invoke",
    method="POST"
  }[5m]))
)
```

---

## 3. SLO targets

Targets are evaluated per calendar quarter. Each target is the floor
the project commits to publicly; we expect to exceed it on healthy
days, and we treat sustained miss as an incident.

| SLI | Target (v0.3) | Window | Headroom vs measured today |
|---|---|---|---|
| `availability_http` | 99.5% | 30 d rolling | No production telemetry yet; conservative for a pre-1.0 release |
| `latency_http_healthz_P95` | ≤ 10 ms | 5 m rolling | Bench P50 ~30–60 µs (PERFORMANCE.md) → ~150× headroom |
| `latency_http_invoke_P95` (host-only) | ≤ 100 ms | 5 m rolling | Bench `e2e/create_function` P50 ~40–80 µs → ~1000× headroom |
| `latency_http_invoke_P95` (with GPU dispatch) | ≤ 500 ms | 5 m rolling | Modeled — CUDA event-sync path lands in v0.2; SLO ratifies in v0.3 |
| `latency_dispatch_serial_P95` (host-only) | ≤ 50 µs | 5 m rolling | Baseline median 50 µs at `dispatch/serial/10` → exactly the floor; see disclosure |
| `latency_dispatch_serial_P95` (CUDA-host) | **TBD — v0.4** | 5 m rolling | S22 numbers must land before this SLO can be set |
| `error_rate_invoke` | ≤ 1.0% | 5 m rolling | No production telemetry; budget chosen to align with availability |

### 3.1 Rationale

**`availability_http: 99.5%`** is "two nines and a half". It allows
~3 h 36 m of unavailability per 30 days. For v0.x this is deliberate
under-promising: TensorWasm has no high-availability story yet (single-
host runtime, no built-in failover), so the SLO should not pretend
that the binary's worst-case downtime is bounded by anything other
than the operator's deploy and restart cadence. v1.0 will tighten
this to 99.9% (≈43 m / month), once `docs/UPGRADE.md` lands the
rolling-restart procedure.

**`latency_http_healthz_P95: 10 ms`** is the operationally meaningful
bound — it includes the network hop from the load balancer, OS
scheduler, and the axum middleware stack. The bench P50 of ~30-60 µs
is the lower bound of what `/healthz` could ever achieve under
ideal conditions; the SLO covers the realistic case of a noisy
neighbour and a load balancer that sometimes sleeps.

**`latency_http_invoke_P95: 100 ms (host-only) / 500 ms (with GPU
dispatch)`** is sized to comfortably cover a small-payload synchronous
invocation including instance spawn, single export call, and
terminate. The 500 ms variant adds room for one PTX load + one kernel
launch + one sync, modelled against the v0.2 CUDA event-sync target
of 5-20 µs per dispatch plus PCIe transfer cost for inputs/outputs.
These will tighten significantly once we have production telemetry.

**`latency_dispatch_serial_P95: 50 µs` (host-only)** is exactly the
baseline median from `bench-results/baseline.json`
(`dispatch/serial/10`, median 50 000 ns). The SLO matches the bench
floor on purpose: dispatch on the host-stub path is essentially a
Tokio semaphore round-trip, and an SLO that allows worse than 50 µs
P95 on the stub path would mask scheduler regressions. Note that the
bench is P50 and the SLO is P95 — even at the same numeric value,
the SLO is a strictly weaker promise. The CUDA-host variant of this
SLO is **TBD** because no measured CUDA dispatch latency exists in
the repo today; setting a number now would be a wild guess and we
prefer the honest "TBD" to a wrong number.

**`error_rate_invoke: ≤ 1.0%`** complements the availability SLO at
finer time resolution. 1% over a 5-minute window catches a fault
that an availability burn-rate alarm would not page on for hours.

### 3.2 Out of scope for v0.3

Quantities we deliberately do **not** set SLOs on at this gate:

- Snapshot capture/restore latency. Sized by payload, dominated by
  disk I/O. Measured in `cold_start/*` benches, alarmed via dashboard
  thresholds, but no SLO until v0.5.
- JIT compile latency. Cache-hit on a warm path is the common case;
  cache-miss is rare and bursty. Tracked in dashboard.
- Tenant context-switch latency. Below the 5-µs done-when target
  documented in PERFORMANCE.md; no SLO until multi-tenant deployments
  are common enough to need one.
- GPU memory pressure. A gauge, not a rate; alarmed via threshold,
  not via SLO.

---

## 4. Error budget calculations

The error budget is the amount of bad behaviour the SLO permits
before it is considered violated. It is the operator's accounting
unit for risk: if there is budget left, ship. If there is not,
freeze deploys until the budget refills.

### 4.1 Availability budget

`availability_http: 99.5%` over 30 days = **0.5% downtime allowed**.

```
30 d × 24 h × 60 m = 43 200 minutes per window
0.5% × 43 200 m   =      216 minutes      = 3 h 36 m
```

Per-day amortised budget: `216 / 30 = 7.2 minutes/day`.

### 4.2 Latency budget

For the latency SLOs, "budget consumption" is the fraction of
requests that fell above the SLO threshold during the rolling 5 m
window. Example: if 5% of `/invoke` calls exceeded 100 ms P95 in
the last 5 m, but the SLO allows the P95 itself to be ≤ 100 ms,
then the SLO is met by definition (the 5% is the tail above P95,
not a violation). Latency SLOs are violated by the P95 of the
window itself exceeding the bound, not by individual slow calls.

### 4.3 Error-rate budget

`error_rate_invoke: ≤ 1.0%` over 5 minutes is the
**simultaneous-window** view: at any moment, the past 5 m of
`/invoke` traffic must have fewer than 1% server-side failures.

For monthly accounting we report budget consumption as
`1 - (1 - observed_5m_max) / (1 - SLO_target)` summed over windows,
but the alerts care only about the rolling number.

### 4.4 Spending the budget

The budget exists to be spent on calculated risks. Examples of
legitimate spend:

- Rolling out a feature flag to 10% of traffic and accepting up to
  X minutes of degradation in that slice.
- Running a destructive migration that briefly serves `503` while a
  tenant's state moves.
- Doing a planned maintenance restart during a low-traffic window.

Examples of illegitimate spend (these are bugs, not features):

- An unintentional infinite loop in a handler.
- An OOM crash because a deploy raised a memory limit without
  testing.
- A regression in a dependency picked up by a routine `cargo update`.

### 4.5 Rollback when budget consumed

If a release consumes more than **50% of the monthly availability
budget in a single 24-hour window** (i.e. >108 minutes of downtime
or >50% of error budget on the latency/error SLOs), the operator
should:

1. Page the on-call (`docs/runbooks/oncall-paging.md`).
2. Roll back to the last known-good release using the procedure in
   `docs/runbooks/rollback.md`.
3. Open an incident retrospective; do not redeploy until the
   regression's root cause is identified and tested for.
4. If the budget is fully consumed, **freeze all non-rollback
   deploys** until the next 30-day window opens. Security patches
   are the only exception; document them in the incident retro.

This is a self-imposed gate, not an automatic enforcement. The
project trusts the operator to follow it.

---

## 5. Burn-rate alerts

Following the [Google SRE workbook](https://sre.google/workbook/alerting-on-slos/)
multi-window multi-burn-rate pattern, three alert pairs cover fast,
slow, and very-slow consumption of the availability budget. Each
pair fires when **both** windows exceed the burn rate
simultaneously, which suppresses brief spikes and short outages that
auto-recover.

The burn rate is `actual_error_rate / SLO_error_budget_rate`. For a
99.5% SLO, the budget rate is 0.5% = 0.005; a burn rate of 14.4×
means errors at 7.2%, which would exhaust the 30-day budget in
~50 hours if sustained.

### 5.1 Fast burn (page)

Errors at 14.4× the budgeted rate, sustained over both a 5-minute
and a 1-hour window. At this rate, the monthly budget is gone in
50 hours.

```promql
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[1h]))
    /
  sum(rate(tensor_wasm_http_requests_total[1h]))
  > (14.4 * 0.005)
)
and
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[5m]))
    /
  sum(rate(tensor_wasm_http_requests_total[5m]))
  > (14.4 * 0.005)
)
```

Severity: page. Runbook: [`docs/runbooks/availability-fast-burn.md`](runbooks/availability-fast-burn.md).

### 5.2 Slow burn (page)

Errors at 6× budgeted rate, sustained over 30 m and 6 h windows.
Catches a degradation that is not fast enough to trigger the
14.4× page but is bleeding the monthly budget over a working day.

```promql
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h]))
    /
  sum(rate(tensor_wasm_http_requests_total[6h]))
  > (6 * 0.005)
)
and
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[30m]))
    /
  sum(rate(tensor_wasm_http_requests_total[30m]))
  > (6 * 0.005)
)
```

Severity: page. Runbook: [`docs/runbooks/availability-slow-burn.md`](runbooks/availability-slow-burn.md).

### 5.3 Very-slow burn (ticket)

Errors at 1× the budgeted rate sustained over 6 h and 3 d windows.
At this rate the budget is depleted exactly at 30 days, so this is
a "you are using all of your budget" warning rather than an
emergency. Files a ticket, does not page.

```promql
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[3d]))
    /
  sum(rate(tensor_wasm_http_requests_total[3d]))
  > (1 * 0.005)
)
and
(
  sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h]))
    /
  sum(rate(tensor_wasm_http_requests_total[6h]))
  > (1 * 0.005)
)
```

Severity: ticket. Runbook: [`docs/runbooks/availability-very-slow-burn.md`](runbooks/availability-very-slow-burn.md).

### 5.4 Latency burn-rate alerts

Latency SLOs use a single-window threshold rather than the
multi-burn-rate pattern, because the 5-m P95 is itself the
burn-rate analogue (a single window already smooths the per-request
noise).

#### Fast latency-burn on `/invoke` (page)

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/functions/:id/invoke",
      method="POST"
    }[5m])
  )
) > 0.5
and
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/functions/:id/invoke",
      method="POST"
    }[1h])
  )
) > 0.5
```

The threshold `0.5` is in seconds, matching the 500 ms SLO for
invocations that issue GPU dispatch. The host-only 100 ms variant
fires a separate alert keyed off the same metric.

Severity: page. Runbook: [`docs/runbooks/invoke-latency-spike.md`](runbooks/invoke-latency-spike.md).

#### Sustained latency-burn on `/healthz` (ticket)

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_http_request_duration_seconds_bucket{
      route="/healthz",
      method="GET"
    }[30m])
  )
) > 0.01
```

`/healthz` exceeding 10 ms P95 sustained over 30 m is usually a sign
of a stuck event loop, not an outage; ticket rather than page.

Severity: ticket. Runbook: [`docs/runbooks/healthz-slow.md`](runbooks/healthz-slow.md).

### 5.5 Dispatch-latency burn (page)

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_kernel_latency_seconds_bucket[5m])
  )
) > 0.00005
and
histogram_quantile(
  0.95,
  sum by (le) (
    rate(tensor_wasm_kernel_latency_seconds_bucket[1h])
  )
) > 0.00005
```

`0.00005` is 50 µs, matching the host-only `latency_dispatch_serial_P95`
SLO. This alert is meaningful only on host-only deployments; on
CUDA-host nodes the threshold must be widened to the CUDA-host
SLO once that number is set (v0.4 work).

Severity: page on host-only deployments. Runbook:
[`docs/runbooks/dispatch-latency-spike.md`](runbooks/dispatch-latency-spike.md).

### 5.6 Why no separate `error_rate_invoke` alert

The fast/slow/very-slow burn alerts above operate on
`tensor_wasm_http_requests_total` across all routes. A
`/invoke`-specific burn would page on the same conditions a few
seconds earlier and would mask the same underlying incidents.
v0.4 may split the alerts per-route if the operational data
justifies it.

---

## 6. Dashboards

The reference Grafana dashboard committed at
`docs/dashboards/tensor-wasm-overview.json` (added in a sibling task
under the v0.3 milestone) renders each SLI above and overlays the
SLO threshold as a horizontal line. Panels:

- **Availability** — 30-day rolling, with the 99.5% target as a
  reference line and the consumed-budget bar in the corner.
- **HTTP P50/P95/P99 latency by route** — one panel per route in
  the API, with the SLO threshold marked for `/healthz` and
  `/invoke`.
- **Error rate by route and status family** — stacked, with the 1%
  threshold marked on `/invoke`.
- **Kernel latency** — P50/P95/P99 from
  `tensor_wasm_kernel_latency_seconds`, with the 50-µs host-only
  SLO threshold marked.
- **Burn-rate** — three panels, one per alert pair, with the alert
  threshold marked. A panel that crosses its threshold is the
  pre-page warning the operator sees in the dashboard.
- **Active instances / GPU memory / spawn-terminate rate** —
  capacity panels, no SLO threshold.
- **JIT cache hit ratio / offload success vs fallback** — efficiency
  panels, no SLO threshold.

Dashboard import is a single JSON upload; the file is meant to be
checked in alongside this document and updated under the same PR
process whenever an SLO changes.

---

## 7. Runbook mapping

Each alert in [Section 5](#5-burn-rate-alerts) links to a one-page
runbook with: what the alert means, what to check first, common
causes, mitigation steps, and the criteria for paging the next
escalation tier. The runbooks themselves are out of scope for this
document — they are tracked under the v0.3 milestone in
[`PATH-TO-V1.md`](PATH-TO-V1.md#v030--production-observability).

| Alert | Severity | Runbook |
|---|---|---|
| Availability fast burn (14.4×, 5 m + 1 h) | Page | `docs/runbooks/availability-fast-burn.md` |
| Availability slow burn (6×, 30 m + 6 h) | Page | `docs/runbooks/availability-slow-burn.md` |
| Availability very-slow burn (1×, 6 h + 3 d) | Ticket | `docs/runbooks/availability-very-slow-burn.md` |
| `/invoke` latency spike (P95 > SLO, 5 m + 1 h) | Page | `docs/runbooks/invoke-latency-spike.md` |
| `/healthz` slow (P95 > 10 ms, 30 m) | Ticket | `docs/runbooks/healthz-slow.md` |
| Dispatch latency spike (P95 > 50 µs, 5 m + 1 h) | Page (host-only) | `docs/runbooks/dispatch-latency-spike.md` |
| Rollback procedure | (manual) | `docs/runbooks/rollback.md` |
| On-call paging procedure | (manual) | `docs/runbooks/oncall-paging.md` |

A runbook landing without a corresponding alert is fine; an alert
landing without a runbook is a PR blocker.

---

## 8. Disclosure: measured vs modeled

The honesty bar mirrors [`PERFORMANCE.md`](PERFORMANCE.md): every
SLO target above is annotated with whether the headroom claim is
grounded in measurement, extrapolation, or guess.

| SLO | Grounded in | v0.3 follow-up |
|---|---|---|
| `availability_http` (99.5%) | **Modeled.** No production deployment exists yet; the number is a conservative pre-1.0 choice. | Recruit a v0.5 design partner per [PATH-TO-V1 §6](PATH-TO-V1.md#6-production-design-partners); replace the model with one month of observed data. |
| `latency_http_healthz_P95` (10 ms) | **Measured floor, modeled SLO.** Bench `e2e/healthz/get` shows P50 ~30-60 µs (PERFORMANCE.md). The 10 ms SLO is a 150× ceiling chosen to bound real-world tail behaviour (network + scheduler), not the bench floor. | None — this is a stable v0.3 commitment. |
| `latency_http_invoke_P95` host-only (100 ms) | **Measured floor, modeled SLO.** Bench `e2e/create_function/post` P50 ~40-80 µs. SLO at 100 ms gives ~1000× headroom for real payloads and instance spawn cost. | None — stable. |
| `latency_http_invoke_P95` with GPU (500 ms) | **Fully modeled.** No GPU-host invocation latency has been measured; the number combines the modeled CUDA event-sync target (5-20 µs/dispatch) with a worst-case PCIe transfer estimate. | Replace with measured P95 once S22 runner produces real numbers; tighten in v0.4. |
| `latency_dispatch_serial_P95` host-only (50 µs) | **Measured.** Baseline median 50 000 ns at `dispatch/serial/10` in `bench-results/baseline.json`. The SLO equals the bench median, which is honest: it means we promise no worse than today's measured floor. | Tighten once `tolerance_pct` drops from 50% to 10% in v0.2 re-baseline. |
| `latency_dispatch_serial_P95` CUDA-host | **TBD.** No measurement exists. | S22 runner produces a measured baseline → number set in v0.4 SLO update. Until then, the column reads "TBD". |
| `error_rate_invoke` (1.0%) | **Modeled.** No production telemetry. Chosen to align with the availability budget at finer time resolution. | Re-evaluate against design-partner data before v1.0. |

**Metrics landed in W2.3** (`crates/tensor-wasm-api/src/http_metrics.rs`,
shipped alongside this document under the v0.3 milestone):

- `tensor_wasm_http_requests_total{route,method,status}` (counter)
- `tensor_wasm_http_request_duration_seconds_bucket{route,method,status}` (histogram)
- `tensor_wasm_http_requests_in_flight{route,method}` (gauge — capacity panel only, not an SLI)

The instrumentation point is a tower middleware (`http_metrics_middleware`)
wired into `build_router` *outside* `bearer_auth`, so `401` responses
are also counted — consistent with `availability_http` and the
burn-rate alerts in [Section 5](#5-burn-rate-alerts), which evaluate
the ratio over every HTTP response. The `route` label always carries
the axum route template (e.g. `/functions/:id/invoke`); the substituted
UUID is never emitted as a label value, and any unmatched path collapses
to `route="unknown"`. Cardinality is bounded by a runtime allow-list
initialised at startup from the route templates registered in
`build_router_with_audit`; adding a new route to that builder requires
adding the same template to `tensor_wasm_api::http_metrics::DEFAULT_ROUTE_ALLOWLIST`
or the panel falls through to `route="unknown"`. The PromQL above is
therefore executable today; the alerts in [Section 5](#5-burn-rate-alerts)
can be loaded into Prometheus and will fire once production traffic
exists to exercise them.

**Metrics confirmed present today** (audited against
`crates/tensor-wasm-core/src/metrics.rs` and
`crates/tensor-wasm-api/src/http_metrics.rs`):

- `tensor_wasm_active_instances`
- `tensor_wasm_gpu_memory_used_bytes`
- `tensor_wasm_kernel_dispatches_total`
- `tensor_wasm_kernel_latency_seconds` (histogram)
- `tensor_wasm_instance_spawns_total`
- `tensor_wasm_instance_terminations_total`
- `tensor_wasm_offload_success_total`
- `tensor_wasm_offload_fallback_total`
- `tensor_wasm_http_requests_total{route,method,status}` (counter, W2.3)
- `tensor_wasm_http_request_duration_seconds{route,method,status}` (histogram, W2.3)
- `tensor_wasm_http_requests_in_flight{route,method}` (gauge, W2.3)
- `tensor_wasm_jobs_active` (gauge, single series, C3 — number of
  async-invocation jobs in `Pending` state in the API-layer job
  registry; not an SLI but feeds the dashboard capacity row)
- `tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id}` (gauge family,
  C3 — additive per-tenant breakdown of GPU memory reservation; the
  pre-existing single-series total at
  `tensor_wasm_gpu_memory_used_bytes` is preserved alongside)

Every SLO in this document is now enforceable against the metrics the
gateway emits. The `latency_dispatch_serial_P95` SLO sits on
`tensor_wasm_kernel_latency_seconds`; the four HTTP-keyed SLOs
(`availability_http`, `latency_http_healthz`, `latency_http_invoke`,
`error_rate_invoke`) sit on the HTTP families landed in W2.3.

---

## 9. How to change an SLO

SLO targets are part of the project's public contract. Tightening
an SLO is a non-breaking change to users (we promise more) but a
potentially breaking change to operators (a node that met the old
SLO may fail the new one). Loosening an SLO is the opposite. Both
directions require:

1. **An RFC** under `rfcs/` per the process documented in
   `rfcs/README.md`. The RFC must include:
   - the current target, the proposed target, and the delta;
   - measured data supporting the change (a month of production
     telemetry, or new bench results, or both);
   - a list of operators consulted and their feedback;
   - the migration plan for any in-flight deployment that meets the
     old SLO but would miss the new one.
2. **A `CHANGELOG.md` entry** under the next release's section,
   classified as "Operator-visible behaviour change". Tightening
   reads: "SLO `<name>` tightened from `X` to `Y`; see RFC #NNN."
   Loosening reads the same with a justification.
3. **Updated dashboard threshold** in
   `docs/dashboards/tensor-wasm-overview.json` and the corresponding
   runbook revisions, all landing in the same PR as the SLO change.

Adding a new SLO follows the same process. Removing an SLO is the
heaviest change — it requires the RFC plus a six-month deprecation
window during which the SLO continues to be evaluated and reported
but no longer pages.

Adjusting the burn-rate alert thresholds **without** changing the
SLO target itself does not require an RFC, only a PR with the
rationale. The alert is an operational knob; the SLO is the
contract.

---

## Related docs

- [PATH-TO-V1.md](PATH-TO-V1.md) — milestone gates; the v0.3
  "SLOs published" criterion is satisfied by this document.
- [PERFORMANCE.md](PERFORMANCE.md) — bench targets and the CI
  regression gate; the *floor* this document's SLOs sit above.
- [OBSERVABILITY.md](OBSERVABILITY.md) — tracing schema, OTLP
  setup, and the existing Prometheus exposition.
- [`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md)
  — HTTP surface this document covers.
- [`crates/tensor-wasm-core/src/metrics.rs`](../crates/tensor-wasm-core/src/metrics.rs)
  — source of truth for currently-emitted metric names.
- [BENCHMARKING.md](BENCHMARKING.md) — external comparison
  methodology; out of scope for SLOs but cross-checks the floors.
- [RISKS.md](RISKS.md) — v0.1.0 known limitations relevant to
  several "modeled" disclosures here.

---

_Status: v0.3 gate. Targets are conservative pre-1.0 commitments and
will tighten once production telemetry exists. The CUDA-host
dispatch SLO is intentionally left TBD pending the S22 runner;
prefer "TBD" to a guess._

---

**Dashboards.** The reference Grafana dashboard described in
[Section 6](#6-dashboards) is committed at
[`docs/dashboards/tensor-wasm-overview.json`](dashboards/tensor-wasm-overview.json)
with an importer-facing companion at
[`docs/dashboards/README.md`](dashboards/README.md). The dashboard's
top row renders the five SLIs defined in
[Section 2](#2-sli-definitions) — `availability_http`,
`error_rate_invoke`, `latency_http_healthz_P95`,
`latency_http_invoke_P95`, and `latency_dispatch_serial_P95` — as
Stat panels with thresholds matching the targets in
[Section 3](#3-slo-targets). Panels whose backing metric is in the
"TODO" column of the dashboard's metric inventory (snapshot
histograms, JIT cache counters, back-pressure gauges, per-tenant
labelling on existing series) render "No data" until those follow-ups
land; no dashboard edit is required to bring them online. The HTTP
request counter, HTTP duration histogram, and HTTP in-flight gauge
landed in W2.3 and render real data today.
