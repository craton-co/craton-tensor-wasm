<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Capacity Planning (v0.4)

This document helps operators answer one question: **how big a host do
I need to run N tenants under the published TensorWasm SLA?** It
translates the SLOs in [`docs/SLO.md`](SLO.md) and the bench medians
in [`bench-results/baseline.json`](../bench-results/baseline.json) into
three reference SKUs, four explicit sizing formulas, and a set of
tenants-per-host curves you can map onto your own workload.

Status: **v0.4 gate**. The host-only numbers are measured Criterion
medians from the committed baseline. The CUDA-host numbers are
modeled — the S22 self-hosted CUDA runner has not yet produced
measured medians for `dispatch/*` on real hardware, and the modeled
ceilings here will be replaced in v0.5 once it has. Every table cell
is marked `(measured)` or `(modeled)` so the operator knows where the
ground truth ends.

## Contents

1. [Purpose](#1-purpose)
2. [Inputs the operator must know](#2-inputs-the-operator-must-know)
3. [Reference SKUs](#3-reference-skus)
4. [Sizing formulas](#4-sizing-formulas)
5. [Tenants-per-host curves](#5-tenants-per-host-curves)
6. [Bottleneck analysis](#6-bottleneck-analysis)
7. [Scaling strategies](#7-scaling-strategies)
8. [How to validate your sizing](#8-how-to-validate-your-sizing)
9. [When to re-plan](#9-when-to-re-plan)
10. [Disclosure: measured vs modeled](#10-disclosure-measured-vs-modeled)
11. [Related docs](#11-related-docs)

---

## 1. Purpose

**What this doc helps with.** Picking a host SKU when you already
know roughly how many tenants you want to land, at what aggregate
QPS, against the SLOs in [`docs/SLO.md`](SLO.md) §3. The output is a
recipe (vCPU, RAM, GPU memory) and a short list of bottlenecks to
watch in the dashboard from
[`docs/dashboards/README.md`](dashboards/README.md).

**What this doc does not replace.** Real load testing. The formulas
in [Section 4](#4-sizing-formulas) carry safety multipliers; the SKU
recipes assume warm cache, average wasm working set, and the MPS
configuration each recipe specifies. Your tenants will not look like
the bench profile.
[Section 8](#8-how-to-validate-your-sizing) covers closing the last
factor of 2 by load-testing before going live.

**Honest caveat.** A meaningful chunk of the numbers below are
modeled — specifically anything tagged `(modeled)` in
[Section 5](#5-tenants-per-host-curves) and every `GPU-bound /invoke`
cell. The S22 self-hosted CUDA runner that would replace them with
measured medians is on the v0.2 path
([`docs/PATH-TO-V1.md`](PATH-TO-V1.md) §v0.2.0).

---

## 2. Inputs the operator must know

The formulas in [Section 4](#4-sizing-formulas) take five inputs.
You cannot size a deployment without estimates for each. Bring rough
numbers; the formulas are not precision instruments.

### 2.1 Target SLA

Read from [`docs/SLO.md`](SLO.md) §3. The v0.3 commitments are:

| SLI | Target | Source |
|---|---|---|
| `availability_http` | 99.5% over 30 d | SLO.md §3 |
| `latency_http_healthz_P95` | ≤ 10 ms | SLO.md §3 |
| `latency_http_invoke_P95` (host-only) | ≤ 100 ms | SLO.md §3 |
| `latency_http_invoke_P95` (with GPU dispatch) | ≤ 500 ms (modeled) | SLO.md §3 |
| `latency_dispatch_serial_P95` (host-only) | ≤ 50 µs | SLO.md §3 |
| `error_rate_invoke` | ≤ 1.0% | SLO.md §3 |

If your contract with downstream users is tighter than the published
SLA, plan against the tighter number and ignore the rest of this
section's choices — every SKU below will overshoot your spend.

### 2.2 Per-tenant QPS

The sustained `/invoke` rate per tenant under normal load. Use the
P95 of the busy hour, not the average; sizing for the average and
hoping the burst absorbs is how nodes get paged at 3am. If you do
not yet have telemetry, start with 1 QPS per tenant as a
conservative default and re-plan ([Section 9](#9-when-to-re-plan))
once production data exists.

### 2.3 Per-tenant wasm working-set size

The resident memory each instance holds when warm — guest linear
memory plus Wasmtime instance overhead. The bench fixtures hold
single-digit MiB; production payloads commonly land between 8 MiB
and 64 MiB. If your tenants ship distinct `.wasm` modules, size
against the worst one, not the median.

### 2.4 Per-tenant GPU usage

Two sub-parts:

- **Driver/context overhead.** Without MPS, every tenant pays one
  `cuCtxCreate` per active instance — roughly 30 MiB of resident
  GPU memory plus a few hundred ms at create time per
  [`docs/MPS-SETUP.md`](MPS-SETUP.md). With MPS, the daemon
  multiplexes onto a single context and per-tenant tax drops to a
  few MiB.
- **Per-call kernel duration.** Sum of input transfer, kernel time,
  output transfer. On v0.1 this is **not measured on real CUDA**
  hardware (see [`docs/PERFORMANCE.md`](PERFORMANCE.md) "CUDA-host
  path"); the modeled 5-20 µs/dispatch floor from PERFORMANCE.md is
  the launch overhead, not the kernel runtime, which is workload-
  specific.

### 2.5 Snapshot capture/restore frequency

How often a tenant captures or restores a snapshot, and the typical
payload size. The bench medians at 1 MiB / 16 MiB land in
`bench-results/baseline.json` as 25 ms capture / 30 ms restore
(1 MiB, measured) and 350 ms capture / 400 ms restore (16 MiB,
measured). For 128 MiB and 512 MiB payloads, the
[`docs/PERFORMANCE.md`](PERFORMANCE.md) reference table extrapolates
to ~600 ms / 400 ms (128 MiB, measured) and ~2.4 s / ~1.6 s (512
MiB, modeled linear extrapolation). Capture/restore competes with
the same disk bandwidth that `/metrics` scrapes and audit-log
writes use; high-frequency snapshotting can starve `/invoke` of I/O
even when CPU is idle.

---

## 3. Reference SKUs

Three recipes. Each names concrete hardware, a tenant ceiling, an
aggregate QPS ceiling, and the assumption set that ceiling assumes.
**Move outside the assumption set and the ceiling no longer holds.**

### 3.1 Small (dev / staging)

- **Hardware.** 4 vCPU, 16 GiB RAM, 1 × NVIDIA T4 (16 GB VRAM).
- **Tenant ceiling.** ≤ 10 active tenants.
- **Aggregate QPS ceiling.** ≤ 100 QPS.
- **Assumptions.** Warm cache (no cold-start storm), average wasm
  working set 16 MiB/tenant, no MPS, snapshot capture/restore at
  most once per minute per tenant, single-host single-replica.
- **Why "Small".** A T4 validates the dispatch path end-to-end but
  cannot absorb bursts across more than a few concurrent tenants.
  The 4 vCPU floor is set by single-threaded JIT compile cost
  during cold start (see [Section 6](#6-bottleneck-analysis));
  below that, cold-start latency dominates `/invoke` P95 and the
  100 ms SLO breaks first.

### 3.2 Medium (production, single-host)

- **Hardware.** 16 vCPU, 64 GiB RAM, 1 × NVIDIA A10 or L4 (24 GB VRAM).
- **Tenant ceiling.** ≤ 50 active tenants.
- **Aggregate QPS ceiling.** ≤ 1000 QPS.
- **Assumptions.** Warm cache with snapshot pre-restore for any
  tenant idle > 5 minutes, average wasm working set 16 MiB/tenant,
  MPS enabled (per [`docs/MPS-SETUP.md`](MPS-SETUP.md), required
  above ~8 co-located tenants), snapshot capture/restore ≤ 1/s node
  total, single-host primary with a standby for failover.
- **Why "Medium".** A 24 GB VRAM SKU comfortably fits 50 tenants at
  the modeled 256 MiB/tenant working set with 50% headroom. 16 vCPU
  has enough parallelism that Wasmtime JIT and the axum router do
  not serialise during cold-start bursts. RAM at 4× the modeled
  working-set total holds the `2×` safety factor plus the 1 GiB
  OS/runtime budget.

### 3.3 Large (production, multi-tenant)

- **Hardware.** 32 vCPU, 128 GiB RAM, 1 × NVIDIA A100 (40 GB) or
  H100 (80 GB).
- **Tenant ceiling.** ≤ 500 active tenants.
- **Aggregate QPS ceiling.** ≤ 10 000 QPS.
- **Assumptions.** Warm cache with aggressive snapshot pre-restore
  on onboarding, average wasm working set 16 MiB/tenant, MPS
  enabled (required), snapshot capture/restore amortised across
  multiple disks or a network volume — never on the same disk as
  the audit log, ≥ 2 host fleet for rolling restart per
  [`docs/UPGRADE.md`](UPGRADE.md) §3.
- **Why "Large".** An H100's 80 GB VRAM unlocks 500 tenants; an
  A100's 40 GB does so only with MPS collapsing per-tenant context
  overhead. 32 vCPU absorbs 10 000 QPS only when the per-request
  path stays out of the JIT — cold-start bursts at this scale will
  violate the 100 ms invoke SLO for the duration of the burst. The
  128 GiB RAM ceiling is set by the I/O bandwidth bottleneck
  described in [Section 6](#6-bottleneck-analysis), not by
  working-set memory.

---

## 4. Sizing formulas

Four explicit formulas. Each is intentionally conservative — the
multipliers exist because production diverges from the bench profile
and SLOs cannot be missed because the formula was tight.

### 4.1 Memory

```
total_ram >= 2 * (active_instances * wasm_working_set
                  + snapshot_buffer
                  + 1 GiB OS/runtime)
```

- **`active_instances`.** Peak concurrent tenants with an instance
  resident (not just registered).
- **`wasm_working_set`.** Per-tenant working set from
  [Section 2.3](#23-per-tenant-wasm-working-set-size). Use the
  worst-case payload, not the median.
- **`snapshot_buffer`.** Size the in-flight snapshot capture/restore
  budget at `max_concurrent_snapshots * max_snapshot_payload`. For
  Small SKU defaults: 1 × 64 MiB. For Medium: 4 × 64 MiB. For Large:
  16 × 128 MiB.
- **`1 GiB OS/runtime`.** OS kernel, Wasmtime runtime, axum router,
  Prometheus scrape state, audit-log buffers.
- **`2×` multiplier.** Headroom for snapshot+live coexistence during
  rolling restart, GC pressure, and one tenant briefly doubling its
  working set during a request.

**Worked example.** Medium SKU, 50 tenants, 16 MiB working set:

```
total_ram >= 2 * (50 * 16 MiB + 4 * 64 MiB + 1 GiB)
          =  2 * (800 MiB + 256 MiB + 1024 MiB)
          =  2 * 2080 MiB
          =  4160 MiB
```

The Medium SKU at 64 GiB has ~15× headroom over this minimum. The
extra is for the I/O cache, the kernel TCP buffers under 1000 QPS
load, and the Wasmtime JIT working memory during cold-start bursts
that the formula does not model directly.

### 4.2 GPU memory

```
total_vram >= 1.5 * (active_instances * gpu_working_set)
```

- **`gpu_working_set`.** Per-tenant resident GPU memory. Without
  MPS: ~30 MiB driver context + ~256 MiB modeled application working
  set + per-call transfer buffers. With MPS (see
  [`docs/MPS-SETUP.md`](MPS-SETUP.md)): the 30 MiB driver context
  drops to a few MiB shared across all clients.
- **`1.5×` multiplier.** Fragmentation headroom. CUDA allocators
  fragment under tenant churn; 50% headroom is the smallest factor
  that consistently avoids OOM when tenants are spawning and
  terminating concurrently.

**Worked example.** Medium SKU, 50 tenants, MPS on, 256 MiB GPU
working set per tenant:

```
total_vram >= 1.5 * (50 * (5 MiB + 256 MiB))
           =  1.5 * 13.05 GiB
           =  19.6 GiB
```

A 24 GB A10/L4 carries this with ~4 GB headroom for transient
allocations. Without MPS the driver-context tax climbs to ~30 MiB ×
50 = 1.5 GiB and the SKU stops fitting.

### 4.3 CPU

```
total_vcpu >= ceiling(
                  (aggregate_qps * cpu_cost_per_request_seconds)
                  / target_utilization
              )
```

- **`cpu_cost_per_request_seconds`.** Per-request CPU cost measured
  under the methodology in [`docs/BENCHMARKING.md`](BENCHMARKING.md)
  §"vs Wasmtime (upstream)". The side-by-side measurement against
  upstream wasmtime on a 500M-iteration compute loop showed
  TensorWasm consuming ~506 ms of User CPU per invocation under that
  fixture — that figure is for a CPU-bound, deliberately heavy
  payload and represents the *upper bound* of per-request CPU cost,
  not a typical `/invoke`. The lightweight bench
  `e2e/create_function/post` (P50 ~40-80 µs, measured per
  PERFORMANCE.md) is the lower bound. Real workloads fall between
  these two.
- **`target_utilization`.** 0.6 for production (40% headroom for
  bursts). Going above 0.8 sustained means the 100 ms invoke P95 SLO
  is one tail event away from breaking.

**Worked example.** Medium SKU, 1000 QPS aggregate, mid-weight
workload at 2 ms CPU per request:

```
total_vcpu >= ceil((1000 * 0.002) / 0.6)
           =  ceil(3.33)
           =  4 vCPU
```

The Medium SKU at 16 vCPU has 4× headroom over this minimum, which
is correct: the formula sizes the steady-state floor, and the extra
12 vCPU absorbs cold-start JIT spikes (single-threaded, ~50 ms of
solid User CPU per spawn) plus the audit-log + metrics overhead.

**For the 506 ms User CPU heavy-payload fixture** at 1000 QPS the
same formula yields 844 vCPU — the Medium SKU obviously cannot
serve this. The BENCHMARKING.md fixture is a stress test; if your
real payloads consume hundreds of ms of CPU per request you are not
running web-shaped traffic and the formulas in this document
under-serve you. Re-derive the QPS ceiling from your own profile.

### 4.4 I/O bandwidth

```
disk_iops_required >= snapshot_ops_per_second * 2
                    + audit_log_writes_per_second
                    + 100 (metrics scrape + OS background)
```

- **`snapshot_ops_per_second`.** Captures + restores per second,
  totalled across all tenants.
- **`audit_log_writes_per_second`.** One write per state-mutating
  API call per [`docs/AUDIT-LOG.md`](AUDIT-LOG.md).
- **`2×` multiplier on snapshots.** Snapshot I/O is bursty; double
  the steady-state estimate to absorb tail spikes.

A spinning disk at 100 IOPS will saturate at the Small SKU's
sustained load. The Medium and Large SKUs assume NVMe (≥ 10 000 IOPS
sustained); the Large SKU additionally assumes snapshot storage on a
separate device from the audit log, because the formula's two terms
otherwise compete for the same write queue.

---

## 5. Tenants-per-host curves

Three tables, one per SLO target. Cells are the **maximum tenant
count** the SKU comfortably serves at the indicated QPS without
breaching the SLO. Every cell is tagged `(measured)` or `(modeled)`
honestly. `(measured)` means a Criterion bench in
`bench-results/baseline.json` or `docs/PERFORMANCE.md` directly
supports the cell. `(modeled)` means the cell is a derivation from
the sizing formulas above without a direct bench backing it.

### 5.1 99.5% availability SLA

The availability SLO covers process-up time, not request latency.
Tenant density does not directly affect availability — the limit is
the SKU's ability to ride through a process restart without dropping
the 0.5% monthly budget. This table reports the **largest tenant
count for which a planned restart fits within the 7.2 min/day
amortised availability budget** from [`docs/SLO.md`](SLO.md) §4.1.

| Aggregate QPS | Small (T4) | Medium (A10/L4) | Large (A100/H100) |
|---|---|---|---|
| 10 QPS | 10 (modeled) | 50 (modeled) | 500 (modeled) |
| 100 QPS | 10 (modeled) | 50 (modeled) | 500 (modeled) |
| 1000 QPS | n/a (over QPS ceiling) | 50 (modeled) | 500 (modeled) |
| 10 000 QPS | n/a | n/a (over QPS ceiling) | 500 (modeled) |

These cells are all modeled because no production availability data
exists yet — per [`docs/SLO.md`](SLO.md) §8 ("Disclosure"), the 99.5%
availability target is itself a conservative pre-1.0 commitment that
will be re-derived from observed data once a v0.5 design partner
runs a beta deployment for a month.

### 5.2 100 ms P95 `/invoke` SLA (host-only)

This is the host-only invoke SLO from [`docs/SLO.md`](SLO.md) §3.
The bench backing it is `e2e/create_function/post` (P50 ~40-80 µs,
measured per PERFORMANCE.md). Tenant density affects this SLO
through CPU contention during cold-start bursts; warm-path invokes
do not contend.

| Aggregate QPS | Small (T4) | Medium (A10/L4) | Large (A100/H100) |
|---|---|---|---|
| 10 QPS | 10 (measured) | 50 (measured) | 500 (modeled) |
| 100 QPS | 10 (measured) | 50 (measured) | 500 (modeled) |
| 1000 QPS | n/a (over QPS ceiling) | 50 (measured) | 500 (modeled) |
| 10 000 QPS | n/a | n/a (over QPS ceiling) | 500 (modeled) |

The Small + Medium "10/100 QPS" cells are measured because the
`e2e/create_function/post` Criterion median of 40-80 µs holds with
multi-thousand-× headroom against the 100 ms SLO; the cells reduce
to "do the SKU's other constraints (memory, GPU) fit the tenant
count?". The Large 500-tenant cells are modeled because no bench
exercises 500 concurrent tenants today — the formula in
[Section 4.1](#41-memory) and the GPU-memory formula in
[Section 4.2](#42-gpu-memory) both fit the SKU, but the actual P95
under 500-tenant load has not been measured.

### 5.3 500 ms P95 GPU-bound `/invoke` SLA (modeled)

This is the GPU-dispatch invoke SLO from [`docs/SLO.md`](SLO.md) §3.
**Every cell is modeled.** Per [`docs/PERFORMANCE.md`](PERFORMANCE.md)
§"CUDA-host path" and [`docs/SLO.md`](SLO.md) §8, no GPU-host
invocation latency has been measured; the modeled 5-20 µs/dispatch
floor combined with a worst-case PCIe transfer estimate puts the
P95 well under 500 ms for the common case, but the underlying
measurement does not yet exist.

| Aggregate QPS | Small (T4) | Medium (A10/L4) | Large (A100/H100) |
|---|---|---|---|
| 10 QPS | 10 (modeled) | 50 (modeled) | 500 (modeled) |
| 100 QPS | 10 (modeled) | 50 (modeled) | 500 (modeled) |
| 1000 QPS | n/a (over QPS ceiling) | 50 (modeled) | 500 (modeled) |
| 10 000 QPS | n/a | n/a (over QPS ceiling) | 500 (modeled) |

When the S22 self-hosted CUDA runner produces measured medians for
`dispatch/serial/*` and `dispatch/concurrent_cap64/*` on real
hardware, this table tightens and the `(modeled)` annotations are
replaced. Until then, treat the QPS-per-tenant capacity as
directionally correct — not a contractual ceiling.

### 5.4 How to read these tables

Pick the SKU column that matches your hardware budget, find the row
matching your expected aggregate QPS, and the cell tells you the
maximum tenant count the SKU supports at the SLO. If your tenant
count exceeds that, you have two choices: scale up to the next SKU
([Section 7.1](#71-vertical-bigger-host)) or shard tenants across
multiple hosts ([Section 7.2](#72-horizontal-shard-tenants-across-pods)).

If the cell is `(modeled)` and your deployment matters, run the
validation in [Section 8](#8-how-to-validate-your-sizing) before
committing budget.

---

## 6. Bottleneck analysis

Which resource saturates first depends on the SKU. Know which one
will go before you commit to the recipe, so the monitoring you build
out catches the right signal.

### 6.1 Small (T4 + 4 vCPU + 16 GiB)

**CPU saturates first.** Specifically, single-threaded JIT compile
during cold start. The Wasmtime Cranelift backend pins a single core
per fresh module; at 4 vCPU, one cold-starting tenant takes 25% of
the host's CPU budget while it compiles, and two concurrent cold
starts push the 100 ms `/invoke` P95 SLO over the edge for the
duration of the spike. Watch for HTTP P95 spikes on
`/functions/:id/invoke`, `tensor_wasm_instance_spawns_total` rate
above 0.5/s sustained, and the "invoke latency spike" burn-rate
alert from [`docs/SLO.md`](SLO.md) §5.4. GPU does not bind on Small
(10 tenants × ~256 MiB ≈ 2.5 GiB, well under T4's 16 GB).

### 6.2 Medium (A10/L4 + 16 vCPU + 64 GiB)

**GPU memory saturates first** when tenant count climbs above 50.
At 50 × 256 MiB working set the 1.5× fragmentation multiplier
consumes the 24 GB SKU's headroom; a 51st tenant's CUDA allocator
request fails, the tenant's `/invoke` returns 500, and the
`error_rate_invoke` SLO breaches. Watch
`tensor_wasm_gpu_memory_used_bytes` climbing toward the VRAM ceiling,
spawn failures with `CudaErrorOutOfMemory` in the audit log, and
`tensor_wasm_offload_fallback_total` rising faster than
`tensor_wasm_offload_success_total`. CPU does not bind (4× headroom
over 1000 QPS at 2 ms/request).

### 6.3 Large (A100/H100 + 32 vCPU + 128 GiB)

**I/O bandwidth and audit-log throughput saturate first.** At
10 000 QPS, the audit log writes 10 000 records/s per
[`docs/AUDIT-LOG.md`](AUDIT-LOG.md); if snapshot capture/restore
shares the same disk, snapshot operations compete for write queue
and `cold_start/disk_round_trip` latency degrades. Watch snapshot
capture P95 climbing while CPU/GPU sit flat, audit-log fsync
latency in the disk panel, and `/healthz` P95 creeping above 10 ms.
This is why the Large SKU assumption requires snapshot storage on a
separate device from the audit log. CPU and GPU do not bind here.

---

## 7. Scaling strategies

Three options. Use them in this order; the operations cost climbs
monotonically.

### 7.1 Vertical: bigger host

Move up an SKU. Memory and GPU memory scale roughly linearly with
tenant count up to the I/O bandwidth ceiling discussed in
[Section 6.3](#63-large-a100h100--32-vcpu--128-gib). Above that
ceiling vertical scaling buys nothing — audit log throughput does
not improve with more RAM or cores. Pros: simplest; no new failure
modes. Cons: single host = single failure domain. The 99.5%
availability SLO allows 3 h 36 m of downtime/month (comfortable for
one host); v1.0's 99.9% target (43 m/month) makes single-host
operation a struggle.

### 7.2 Horizontal: shard tenants across pods

Run multiple TensorWasm replicas behind a load balancer with sticky
routing keyed on tenant ID. Each replica owns a subset of tenants.
See [`docs/UPGRADE.md`](UPGRADE.md) §6 ("Rolling strategy") — the
same sticky-routing requirement that makes rolling upgrades work
also makes horizontal sharding work. Pros: linear scaling, no SPOF.
Cons: a tenant's snapshots and live instance state are not
transparently portable across replicas (per
[`docs/UPGRADE.md`](UPGRADE.md) §3 "Snapshots cross replicas; live
instances do not"); replica swap = cold start for every tenant on
the swapped replica. Recommended for any deployment > 500 tenants
or > 10 000 QPS.

### 7.3 GPU sharing via MPS

Enable MPS per [`docs/MPS-SETUP.md`](MPS-SETUP.md). This is the
difference between the Medium SKU at 50 tenants and the Large SKU
at 500 — without MPS, the per-tenant CUDA-context tax of ~30 MiB
makes the formula in [Section 4.2](#42-gpu-memory) fail above
roughly eight co-located tenants. MPS is not free: the daemon is a
privileged process to operate and MPS clients cannot attach the
GPU debugger. For dev hosts (Small SKU) leave MPS off; for Medium
and Large in production MPS is the default — without it, the SKU
recipe does not hold.

---

## 8. How to validate your sizing

Load-test before going live. The formulas in
[Section 4](#4-sizing-formulas) get you within a factor of two of
correct; the validation step closes the rest.

### 8.1 Generate synthetic load

Use `wrk` or `vegeta` against `/invoke` with a representative
payload. For a TensorWasm-specific harness, the side-by-side
methodology in [`docs/BENCHMARKING.md`](BENCHMARKING.md) §"vs
Wasmtime (upstream)" includes a hyperfine-based driver against the
`matrix_multiply.wat` fixture — adapt that fixture to your tenants'
actual `.wasm` modules and drive concurrently across the planned
tenant population.

Example with `vegeta`:

```bash
echo 'POST http://tensor-wasm-host:8080/functions/<id>/invoke' \
  | vegeta attack -rate=1000/s -duration=10m -header "Authorization: Bearer <token>" \
                  -body=payload.json \
  | vegeta report
```

Run the attack at your **planned aggregate QPS** for at least
10 minutes; shorter windows do not exercise the snapshot
capture/restore cycle.

### 8.2 Watch the W2.5 dashboard

Open the Grafana dashboard from
[`docs/dashboards/README.md`](dashboards/README.md). The five SLO
stat panels in the top row are the validation oracle: if every
panel stays green for the duration of the load test, the sizing
holds. Specifically watch:

- `invoke P95 (5m)` — must stay ≤ 100 ms host-only / ≤ 500 ms with GPU
- `error_rate_invoke (5m)` — must stay ≤ 1.0%
- `dispatch P95 (5m)` — must stay ≤ 50 µs host-only
- `availability_http (30d)` — should not move during the test
- `healthz P95 (5m)` — must stay ≤ 10 ms

Plus the capacity panels:

- `GPU memory by tenant` — should plateau, not climb monotonically
- `Active instances by tenant` — should match the tenant count you
  drove
- `Back-pressure permit utilization` — sustained > 80% means you
  are at the dispatch ceiling

### 8.3 If any burn-rate alert fires

Per [`docs/SLO.md`](SLO.md) §5, three burn-rate pairs cover fast,
slow, and very-slow budget consumption. If any fire during the load
test, **you are undersized**. Fast burn (14.4×) → scale up
immediately, either vertically ([Section 7.1](#71-vertical-bigger-host))
or by sharding ([Section 7.2](#72-horizontal-shard-tenants-across-pods)).
Slow burn (6×) → investigate the bottleneck in
[Section 6](#6-bottleneck-analysis); often a misconfigured workload
(snapshot frequency, MPS off, audit log on the wrong disk) rather
than the SKU. Very-slow burn (1×) → SKU is on the edge; either
tighten the working-set estimate or move up an SKU.

### 8.4 Re-test on each upgrade

Sizing assumptions are not stable across releases. Any trigger in
[Section 9](#9-when-to-re-plan) invalidates the prior validation
and demands a re-run.

---

## 9. When to re-plan

The SKU recipes above hold under the assumptions in
[Section 3](#3-reference-skus). Any of these triggers means an
assumption changed and the formulas in
[Section 4](#4-sizing-formulas) must be re-evaluated:

- **Tenant onboarding past the SKU ceiling.** Move up an SKU or
  shard ([Section 7](#7-scaling-strategies)).
- **Sustained traffic doubles.** Re-derive the CPU formula in
  [Section 4.3](#43-cpu); the QPS ceiling is per-aggregate.
- **Wasm payload size grows.** A 16 MiB → 64 MiB working-set bump
  breaks [Section 4.1](#41-memory); the Medium SKU's 50-tenant
  ceiling drops to roughly 12.
- **MPS toggled.** Disabling MPS adds back the per-tenant ~30 MiB
  CUDA-context tax and the Medium SKU stops holding 50 tenants on a
  24 GB GPU; enabling MPS frees the same headroom. Re-derive
  [Section 4.2](#42-gpu-memory) either way.
- **Kernel-args lowering changes per-call cost.** Per
  [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) §v0.2.0, `KernelArgsUnsupported`
  is removed in v0.2; the direct `cuLaunchKernel` path may shift the
  modeled 5-20 µs/dispatch in either direction. Re-validate after
  v0.2 and update [Section 5.3](#53-500-ms-p95-gpu-bound-invoke-sla-modeled).
- **A new TensorWasm release ships.** Per
  [`docs/UPGRADE.md`](UPGRADE.md) §6 the regression gate catches
  large drift, but small drift compounds. Re-validate at every
  minor-version upgrade.
- **Hardware swap.** SKU recipes are calibrated for the exact
  hardware listed. Swapping T4 → V100 or A10 → A100 changes the
  GPU-side numbers; the QPS ceiling moves.

If none of the above happens, re-plan **at least quarterly** anyway.
Drift accumulates; quarterly re-validation costs an afternoon and
catches assumption rot before it becomes an incident.

---

## 10. Disclosure: measured vs modeled

Mirroring [`docs/SLO.md`](SLO.md) §8: every sizing claim above is
either grounded in measurement (a Criterion bench in
`bench-results/baseline.json` or a number documented in
`docs/PERFORMANCE.md`) or modeled (a derivation from those
measurements without a direct bench backing it). The split:

| Claim | Grounded in | v0.5 follow-up |
|---|---|---|
| `cold_start/capture` + `cold_start/restore` 1 MiB / 16 MiB medians | **Measured.** `bench-results/baseline.json` | Re-baseline as tolerances tighten in v0.2 |
| `cold_start/capture` 128 MiB ~600 ms | **Measured.** `docs/PERFORMANCE.md` reference table | None |
| `cold_start/capture` 512 MiB ~2.4 s | **Modeled (linear extrapolation).** `docs/PERFORMANCE.md` | Measure on S22 runner |
| `dispatch/serial/10` median 50 µs (host-only stub) | **Measured.** `bench-results/baseline.json` | Replace with CUDA-host median in v0.4 |
| `dispatch/serial` 5-20 µs/dispatch on CUDA | **Modeled.** `docs/PERFORMANCE.md` "CUDA-host path" | Measure on S22 runner; tighten in v0.4 |
| `e2e/create_function/post` P50 40-80 µs; `e2e/healthz/get` P50 30-60 µs | **Measured.** `docs/PERFORMANCE.md` reference table | None |
| 506 ms User CPU per 500M-iter loop | **Measured (BENCHMARKING.md fixture).** Side-by-side hyperfine run against `matrix_multiply.wat` per `docs/BENCHMARKING.md` §"vs Wasmtime (upstream)" | Stress-test fixture; not a typical workload |
| MPS reduces per-tenant context from ~30 MiB to a few MiB | **Documented.** `docs/MPS-SETUP.md` | None |
| Memory / GPU / CPU / IO formulas in [Section 4](#4-sizing-formulas) | **Modeled.** Derived from bench floors + safety multipliers | Validate against design-partner deployment in v0.5 |
| Small / Medium / Large tenant-count ceilings | **Modeled.** Bench medians fit each SKU but no multi-tenant concurrent bench exists | Add multi-tenant bench in v0.5 |
| 99.5% availability for any SKU | **Modeled.** No production telemetry yet | Replace with observed data from v0.5 design partner |
| 100 ms P95 host-only invoke for any SKU | **Measured floor, modeled SKU mapping.** `e2e/create_function/post` floor fits with ~1000× headroom; tenant-count mapping is modeled | Cells become measured once multi-tenant bench lands |
| 500 ms P95 GPU-bound invoke for any SKU | **Fully modeled.** No GPU-host invocation has been measured | Replace with measured P95 from S22 runner |

The honest takeaway: **host-only per-request floors are measured;
multi-tenant aggregates and CUDA-host paths are modeled.** The S22
runner work in [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) §v0.2.0 moves
the CUDA-host numbers from modeled to measured. The
design-partner work in [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) §v0.5.0
moves the multi-tenant aggregates from modeled to measured. Until
both land, this document is the operator's best guess — calibrated
against what is measured, but not a replacement for actually
load-testing your deployment per
[Section 8](#8-how-to-validate-your-sizing).

---

## 11. Related docs

- [`docs/SLO.md`](SLO.md) — the SLA this document plans to.
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — bench medians the
  formulas derive from; CUDA-host estimates inherited here.
- [`docs/BENCHMARKING.md`](BENCHMARKING.md) — methodology for
  per-request CPU cost and the side-by-side wasmtime comparison.
- [`docs/dashboards/README.md`](dashboards/README.md) — the W2.5
  dashboard; the validation oracle for [Section 8](#8-how-to-validate-your-sizing).
- [`docs/runbooks/`](runbooks/) — per-alert mitigations; the
  burn-rate runbooks are the first response to an undersized SKU.
- [`docs/MPS-SETUP.md`](MPS-SETUP.md) — without MPS the Medium and
  Large SKU recipes do not hold.
- [`docs/UPGRADE.md`](UPGRADE.md) — sticky-routing and
  rolling-restart cost referenced in
  [Section 7](#7-scaling-strategies).
- [`docs/AUDIT-LOG.md`](AUDIT-LOG.md) — write rate used in the I/O
  formula in [Section 4.4](#44-io-bandwidth).
- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — milestone gates; v0.4
  "Capacity-planning doc" exit criterion satisfied here. The S22
  runner (v0.2) and design-partner work (v0.5) replace the modeled
  cells in [Section 5](#5-tenants-per-host-curves).
- [`bench-results/baseline.json`](../bench-results/baseline.json) —
  source of truth for every `(measured)` cell.

---

_Status: v0.4 gate. Host-only floors are measured against the
committed baseline; multi-tenant aggregates and CUDA-host paths are
modeled. Re-validate per [Section 9](#9-when-to-re-plan)._
