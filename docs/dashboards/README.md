<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Reference Grafana dashboards

This directory holds the importable Grafana dashboards that ship with
TensorWasm. They are the operator-facing companion to
[`docs/SLO.md`](../SLO.md) and the `docs/runbooks/` set: every alert
in those documents has at least one panel here, and every SLI defined
in `docs/SLO.md` has a top-row stat panel.

Status: **v0.3 gate**. The dashboard is checked in alongside
[`PATH-TO-V1.md`](../PATH-TO-V1.md) milestone v0.3 ("Production
observability"). Several panels reference metrics that have not yet
been emitted by the code; those panels will show "No data" / "N/A"
until the metric ships. See [Metric inventory](#metric-inventory).

## Files

| File | Purpose |
|---|---|
| `tensor-wasm-overview.json` | The main overview dashboard. SLO summary + HTTP + tenant + snapshot + JIT + back-pressure rows. Imports cleanly into any Grafana with a Prometheus datasource configured. |
| `README.md` | This file. |

## How to import

1. Open Grafana.
2. Navigate to **Dashboards → New → Import**.
3. Either:
   - **Upload JSON file:** select
     `docs/dashboards/tensor-wasm-overview.json` from your local
     checkout, or
   - **Paste JSON / URL:** paste the raw contents of the file, or the
     raw-file URL (e.g.
     `https://github.com/craton-co/craton-tensor-wasm/raw/main/docs/dashboards/tensor-wasm-overview.json`).
4. Grafana prompts for the `DS_PROMETHEUS` datasource variable —
   select the Prometheus datasource that scrapes your TensorWasm
   process's `GET /metrics` endpoint. Any Prometheus datasource works;
   the dashboard does not depend on Mimir, Cortex, or any
   distribution-specific feature.
5. Click **Import**. Grafana assigns a fresh `uid` and `id`; the JSON
   file does not pin either, so two Grafana instances will not clash
   if they both import this dashboard.

To pull updates after the dashboard JSON changes upstream, re-import
the new JSON over the existing dashboard. Grafana offers a "Replace
existing" toggle that preserves the dashboard's URL.

## Prometheus scrape interval

The dashboard uses `$__rate_interval` in every `rate()` and
`histogram_quantile()` expression, which automatically picks a window
appropriate for the current panel resolution and the configured scrape
interval. The recommended minimum scrape interval is **15 seconds**
(Prometheus default); going below 5 seconds will make per-panel resolution
choppy without adding signal, and going above 60 seconds will smear
the latency P95 panels.

Configure the scrape in your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: tensor-wasm
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ['tensor-wasm.your-host:8080']
```

## Dashboard variables

Three templating variables are defined at the dashboard level.

| Name | Type | Source | Default | Purpose |
|---|---|---|---|---|
| `DS_PROMETHEUS` | datasource | Grafana | (prompt at import) | The Prometheus datasource feeding every panel. Selected once at import. |
| `tenant` | query | `label_values(tensor_wasm_active_instances, tenant)` | All | Filters the **Tenant** row panels by tenant. Multi-select with "All" included. |
| `route` | query | `label_values(tensor_wasm_http_requests_total, route)` | All | Filters the **HTTP traffic** row panels by axum route template (e.g. `/healthz`, `/functions/:id/invoke`). Multi-select with "All" included. |

The `tenant` and `route` variables degrade gracefully: if the
underlying label does not yet exist on the metric (per W2.3), the
variable's option list will be empty and the panels will fall through
to whatever series exists (typically a single un-labeled aggregate).

## Panel inventory

The grid is 24 columns wide. Rows from top to bottom:

### SLO summary (top row, stat panels — thresholds match `docs/SLO.md` §3)

- `availability_http (30d)` — 99.5% target
- `error_rate_invoke (5m)` — ≤ 1.0% target
- `healthz P95 (5m)` — ≤ 10 ms target
- `invoke P95 (5m)` — ≤ 100 ms host-only / ≤ 500 ms with GPU dispatch
- `dispatch P95 (5m)` — ≤ 50 µs host-only; CUDA-host TBD (v0.4)

### Build identity (header row, stat panel)

- `Binary version` — Stat panel reading the `tensor_wasm_build_info`
  info-style gauge. Recommended PromQL — render the gauge's
  `version` label directly with no aggregation, so a heterogeneous
  fleet shows one row per running binary:

  ```promql
  tensor_wasm_build_info
  ```

  In Grafana, set the panel's value mapping to the `version` label
  via **Value options → Fields → Labels → version** (or use
  `label_values(tensor_wasm_build_info, version)` in a templated
  caption). The same metric carries `git_sha`, `rustc_version`,
  `profile`, and `target` labels; surface whichever are useful as
  secondary stat panels or as the panel tooltip. The gauge is
  always `1` — the payload is the label set, not the number. Useful
  during an upgrade window to confirm every replica reports the
  expected version; cross-references the `tensor_wasm_build_info`
  check in `docs/UPGRADE.md` §6.

### HTTP traffic

- `Requests/sec by route` — timeseries
- `Error rate (5xx) by route` — timeseries, 1% threshold line
- `HTTP latency P50 / P95 / P99 by route` — timeseries, log Y axis

### Tenant

- `Active instances by tenant` — stacked timeseries
- `GPU memory by tenant` — stacked timeseries, bytes
- `Kernel dispatches/sec by tenant` — timeseries

### Snapshot

- `Snapshot capture P95` — timeseries
- `Snapshot restore P95` — timeseries
- `Snapshot disk round-trip P95` — timeseries

### JIT / auto-offload

- `JIT cache hit ratio` — timeseries; carries two queries (the
  v0.3 proxy via `offload_success`/`offload_fallback` and the
  intended `jit_cache_warm_hit`/`cold_miss_then_insert` ratio that
  fills in once the metric pair ships)
- `JIT emit_text P95 by blueprint` — timeseries

### Back-pressure

- `Back-pressure permits (used vs available)` — timeseries
- `Back-pressure permit utilization` — timeseries, 80% / 95%
  threshold lines

## Metric inventory

Audited against `crates/tensor-wasm-core/src/metrics.rs` and
`crates/tensor-wasm-api/src/http_metrics.rs` (the sources of truth
for currently-emitted metrics).

### Exists today

These metrics back panels that render real data immediately after
import:

- `tensor_wasm_active_instances` (gauge)
- `tensor_wasm_gpu_memory_used_bytes` (gauge)
- `tensor_wasm_kernel_dispatches_total` (counter)
- `tensor_wasm_kernel_latency_seconds_bucket` /
  `tensor_wasm_kernel_latency_seconds_count` /
  `tensor_wasm_kernel_latency_seconds_sum` (histogram, 14 buckets
  spanning 10 µs to 10 s)
- `tensor_wasm_instance_spawns_total` (counter)
- `tensor_wasm_instance_terminations_total` (counter)
- `tensor_wasm_offload_success_total` (counter)
- `tensor_wasm_offload_fallback_total` (counter)
- `tensor_wasm_http_requests_total{route,method,status}` (counter,
  landed in W2.3 via `tensor_wasm_api::http_metrics`)
- `tensor_wasm_http_request_duration_seconds_bucket` /
  `tensor_wasm_http_request_duration_seconds_count` /
  `tensor_wasm_http_request_duration_seconds_sum` (histogram, 12
  buckets spanning 1 ms to 10 s, labels `route`/`method`/`status`,
  landed in W2.3)
- `tensor_wasm_http_requests_in_flight{route,method}` (gauge,
  landed in W2.3; capacity panel only, not an SLI)
- `tensor_wasm_build_info{version,git_sha,rustc_version,profile,target}`
  (info-style gauge, value always `1`; landed in W4.9). Powers the
  **Build identity** stat panel and is referenced by the UPGRADE.md
  post-upgrade verification step that confirms every replica reports
  the expected `version` after a rolling deploy.
- `tensor_wasm_jobs_active` (gauge, single series; landed in C3).
  Incremented when `POST /functions/:id/invoke-async` records a
  `JobRecord` in the API-layer registry, decremented when the
  spawned task transitions the job to `Completed` or `Failed`. The
  Async-invocation row's "Pending jobs" panel renders this series
  directly. v0.3.x is intentionally a single series; the v0.4
  follow-up will switch the field to a `Family<TenantLabels, ...>`
  and the panel's PromQL will tolerate the relabel without edit.
- `tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id}` (gauge family;
  landed in C3). Updated by `tensor-wasm-tenant` on every
  `consume_bytes` / `release_bytes` accounting transition when the
  context is constructed with `TenantContextBuilder::with_metrics`.
  Additive to the pre-existing single-series total at
  `tensor_wasm_gpu_memory_used_bytes`; `sum by ()
  (tensor_wasm_gpu_memory_bytes_per_tenant)` is expected to track
  that total within scrape jitter. The Tenant row's "GPU memory by
  tenant" panel switches from the single-series total to this
  family automatically once at least one tenant has been observed.

Caveat: the gauges and counters listed above are emitted today as
**aggregates with no `tenant` label**. The Tenant-row panels query
`sum by (tenant) (...)`, which collapses to a single series until
per-tenant labeling lands. That is intentional — the panel layout
should not change when the labels appear, only the number of series.

### TODO, will fill in when remaining follow-ups land

These metrics are referenced by panels but are **not yet emitted** by
the code. The corresponding panels will show "No data" / "N/A" until
the listed follow-up ships the missing instrumentation. Each is
flagged inline in the panel description.

(The HTTP gateway counter / duration / in-flight metrics that were
previously in this list landed in W2.3 — see [Exists today](#exists-today).)

Tenant labeling on existing metrics (a relabeling, not a new metric):

- `tensor_wasm_active_instances{tenant}` (add label)
- `tensor_wasm_kernel_dispatches_total{tenant}` (add label)

Per-tenant GPU memory has been replaced by the additive family
`tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id}` rather than a
relabeling — see [Exists today](#exists-today). The pre-existing
single-series `tensor_wasm_gpu_memory_used_bytes` total stays, so
existing alerts that aggregate against it do not break.

Snapshot subsystem (proposed instrumentation point: `tensor-wasm-mem`
snapshot save/restore paths):

- `tensor_wasm_snapshot_capture_seconds_bucket` (histogram)
- `tensor_wasm_snapshot_restore_seconds_bucket` (histogram)
- `tensor_wasm_snapshot_disk_round_trip_seconds_bucket` (histogram)

JIT / auto-offload subsystem (proposed instrumentation point:
`tensor-wasm-jit` cache and emit paths):

- `tensor_wasm_jit_cache_warm_hit_total` (counter)
- `tensor_wasm_jit_cache_cold_miss_then_insert_total` (counter)
- `tensor_wasm_jit_emit_text_seconds_bucket{blueprint}` (histogram)

Back-pressure semaphore (proposed instrumentation point:
`tensor-wasm-exec` dispatch semaphore):

- `tensor_wasm_backpressure_permits_used` (gauge)
- `tensor_wasm_backpressure_permits_available` (gauge)

When a metric in this TODO list ships, **no dashboard edit is
required** — the panel that references it will start populating with
data on the next scrape. The panel descriptions should then be
updated in a follow-up PR to remove the "shows N/A until …" hedge.

## Editing the dashboard

The JSON is the source of truth. If you edit the dashboard in
Grafana's UI:

1. Use **Dashboard settings → JSON Model** to copy the updated JSON.
2. Strip the `id` and `uid` fields from the top level — they are
   instance-specific and should not be committed.
3. Replace the contents of `tensor-wasm-overview.json` with the
   stripped JSON.
4. Commit per the rules in `docs/SLO.md` §9 (any threshold or query
   change that affects an SLO panel must land in the same PR as the
   SLO change).

## Cross-references

- [`docs/SLO.md`](../SLO.md) — SLI/SLO definitions and burn-rate
  alert thresholds that this dashboard visualizes.
- [`docs/OBSERVABILITY.md`](../OBSERVABILITY.md) — tracing schema and
  OTLP setup; the metrics half of the same observability story.
- [`crates/tensor-wasm-core/src/metrics.rs`](../../crates/tensor-wasm-core/src/metrics.rs)
  — source of truth for currently-emitted metric names.
- [`PATH-TO-V1.md`](../PATH-TO-V1.md) — v0.3 exit criterion
  "Reference Grafana dashboard committed" is satisfied by this
  directory.

---

_Status: v0.3 gate. HTTP request rate, error rate, and latency panels
render real data as of W2.3. The async-invocation "Pending jobs"
panel and the per-tenant GPU-memory breakdown render real data as of
C3 (`tensor_wasm_jobs_active`,
`tensor_wasm_gpu_memory_bytes_per_tenant`). The remaining "No data"
panels (snapshot histograms, JIT cache counters, back-pressure
gauges, and tenant labeling on the remaining counter/gauge pair) are
tracked in [Metric inventory](#metric-inventory)._
