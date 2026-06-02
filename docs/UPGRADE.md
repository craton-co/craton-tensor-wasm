<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Fleet Upgrade Playbook

The operator-facing instructions for rolling a running TensorWasm
deployment from one release to another. This is the W3.3 artifact
behind the v1.0 gate line in
[`docs/PATH-TO-V1.md`](PATH-TO-V1.md#v100-rc1--v100): "the operational
steps to roll a TensorWasm fleet from v0.5 to v1.0". The same playbook
applies to any v0.X → v0.Y step on the path there.

The doc is opinionated about sequencing. Skip a step, you may still
land on the new release; skip it on a busy fleet, you will eventually
land on [`docs/runbooks/rollback.md`](runbooks/rollback.md) at 03:00.

---

## 1. When to use this doc

Three release-engineering documents work together. Pick the one that
matches the question you are answering:

| Doc | Question it answers |
|---|---|
| **`docs/UPGRADE.md` (this doc)** | "How do I roll my fleet from v0.X to v0.Y without taking the SLO down?" |
| [`docs/MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) | "What public APIs / env vars / behaviours changed, and what do I have to edit in my code?" |
| [`docs/runbooks/rollback.md`](runbooks/rollback.md) | "The upgrade is failing — how do I get back to the last known-good release right now?" |

This doc is strictly **operational**: drain, upgrade, verify, resume.
API-surface changes live in the migration doc, recovery from a bad
upgrade in the rollback runbook — both cross-referenced inline, never
duplicated.

---

## 2. Pre-flight checklist

Complete every box before you touch a binary. Each item, if skipped,
removes a layer of safety net.

### 2.1 Read the changelog

- [ ] Read [`CHANGELOG.md`](../CHANGELOG.md) for the target release
      **and every intermediate release** between your current pin and
      the target. A v0.2 → v0.5 hop reads four changelog sections,
      not one.
- [ ] Note every `### Changed`, `### Deprecated`, and `### Removed`
      entry. Those are the ones that need operator action.

### 2.2 Read the migration guide

- [ ] Read [`docs/MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) §3
      (deprecation table) and §5 (behavioural-change table) for every
      version in the range. The "How to detect" column tells you what
      to look for in production *before* you upgrade — many entries
      are visible as deprecation warnings the current binary already
      emits.
- [ ] If §4 (removed-API table) has a row pinned to a release inside
      your hop range, you cannot skip-upgrade past it. Either land on
      the release where the removal happens with the migration already
      applied, or stop one release short and migrate first.

### 2.3 Snapshot the state you would need for rollback

- [ ] Capture every active instance per
      [`MIGRATION-v0-to-v1.md` §2.1](MIGRATION-v0-to-v1.md#21-snapshot-a-working-baseline)
      using `tensor-wasm-cli snapshot save`, tagged with the source
      version.
- [ ] Archive the current audit-log segment per
      [`MIGRATION-v0-to-v1.md` §2.2](MIGRATION-v0-to-v1.md#22-archive-audit-log-streams).
      A torn audit segment across an upgrade is the most common
      compliance gap reported by design-partner deployments.
- [ ] Dump current env vars and Helm values / systemd unit per
      [`MIGRATION-v0-to-v1.md` §2.3](MIGRATION-v0-to-v1.md#23-snapshot-current-configuration).

### 2.4 Quiesce write traffic if possible

- [ ] Pause batch jobs that create new functions or invoke async via
      `POST /functions/{id}/invoke-async`. Read traffic
      (`GET /functions`, `GET /jobs/{id}`) can keep flowing — those
      do not generate audit records or contend for the rate-limit
      bucket.
- [ ] Drain in-flight async jobs. Poll `GET /jobs/{id}` until the set
      of jobs in `pending`/`running` is empty. The `tensor-wasm-cli
      observe` subcommand has a `--once` mode you can grep for the
      `jobs_in_flight` line.

### 2.5 Confirm SLO budget headroom

The single best predictor of a bad upgrade window is upgrading into an
already-degraded SLO. Look at the dashboard before you do anything
else.

- [ ] Open the reference Grafana dashboard at
      [`docs/dashboards/tensor-wasm-overview.json`](dashboards/README.md).
      The top row renders the five SLIs from
      [`docs/SLO.md` §2](SLO.md#2-sli-definitions).
- [ ] Read the consumed-budget bar on the **Availability** panel.
      **Do not upgrade with less than 30% error budget remaining.** A
      bad upgrade will burn through what is left in minutes; the
      [`availability-fast-burn`](runbooks/availability-fast-burn.md)
      alert fires at 14.4× the budgeted rate (see
      [`docs/SLO.md` §5.1](SLO.md#51-fast-burn-page)) and you will not
      have headroom to recover.
- [ ] If the budget is below 30%, defer the upgrade unless it is
      itself the fix for a budget-consuming bug. Document the
      rationale in the change ticket.

### 2.6 Pre-open the response tools

Every minute spent finding a runbook is a minute the SLO is burning.
Pre-open the dashboard
([`docs/dashboards/README.md`](dashboards/README.md)),
[`docs/runbooks/rollback.md`](runbooks/rollback.md), and the
page-severity burn-rate runbooks listed in
[`docs/SLO.md` §7](SLO.md#7-runbook-mapping)
([`availability-fast-burn`](runbooks/availability-fast-burn.md),
[`invoke-latency-spike`](runbooks/invoke-latency-spike.md),
[`dispatch-latency-spike`](runbooks/dispatch-latency-spike.md)).

---

## 3. Upgrade strategies

TensorWasm is a single-instance-stateful runtime: per-tenant
rate-limit buckets, the function registry, the JIT cache, and the
active-instance set all live in process memory. That shapes which
upgrade strategies work. Pick **one** strategy and apply it end-to-end
— mixing them (rolling restart + blue/green LB) is how partial
outages happen.

### 3.1 Strategy A: Rolling upgrade (multi-replica)

Applies when more than one `tensor-wasm-api` replica sits behind an
LB. The Helm chart's `replicaCount` (see
[`deploy/helm/tensor-wasm/README.md`](../deploy/helm/tensor-wasm/README.md))
makes this possible but not automatic.

Constraints:

- **Sticky routing is mandatory.** Rate-limit buckets and warm JIT
  caches are per-process; without stickiness, the observed QPS limit
  drifts and cache hit rate halves.
- **Snapshots cross replicas; live instances do not.** If a tenant's
  active instance is on the replica being restarted, the next invoke
  may land on a different replica with no instance and fail with
  `instance_not_found` until the client retries through a snapshot
  restore. Treat replica swap as a hard cold-start for every tenant
  pinned to it.
- **The Helm chart defaults to `strategy.type: Recreate`.** Switch to
  `RollingUpdate` only when `replicaCount > 1` and the above are
  acceptable.

Wall-clock cost: ~5 min for a 4-tenant deployment with three replicas.

### 3.2 Strategy B: Blue/green (recommended)

Stand up vNext as a separate Deployment alongside vCurrent, smoke-test
out-of-band via port-forward, then cut the Service selector over. If
anything misbehaves, cut the selector back. This is the default
recommendation for production fleets and the strategy assumed by the
k8s walkthrough in [§4](#4-kubernetes-upgrade-walkthrough). The Helm
chart supports it via `--set nameOverride` or a second Release.

Wall-clock cost: ~10 min for a 4-tenant deployment; revert is
sub-30-second (Service selector flip).

### 3.3 Strategy C: In-place restart

Single host (dev, staging, small production) that can tolerate a 5–15
second 503 window. Drain (if possible), stop, swap binary, start,
verify — the same step list as
[`docs/runbooks/rollback.md` §A](runbooks/rollback.md#a-systemd-the-reference-deployment)
run forward.

Wall-clock cost: ~2 min for a 4-tenant deployment.

---

## 4. Kubernetes upgrade walkthrough

The W2.7 Helm chart
([`deploy/helm/tensor-wasm/`](../deploy/helm/tensor-wasm/README.md))
and plain manifests
([`deploy/k8s/`](../deploy/k8s/README.md)) are the reference shapes.
Helm is shorter; plain YAML is auditable.

### 4.1 Helm path

Assumes a `values.yaml` from the current release with `image.tag`
bumped to the target. Replace `0.2.0` with your target.

```bash
# Pre-flight (Section 2) first.

# Diff before applying. helm-diff plugin is cheapest; helm template +
# kubectl diff also works.
helm diff upgrade tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm -f values.yaml --set image.tag=0.2.0

# Apply. The chart's checksum/config + checksum/secret annotations
# re-roll the pod on value changes (see deploy/helm/tensor-wasm/README.md).
helm upgrade tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm -f values.yaml --set image.tag=0.2.0

# Watch the rollout.
kubectl rollout status deployment/tensor-wasm -n tensor-wasm --timeout=5m
kubectl -n tensor-wasm get pods -l app.kubernetes.io/name=tensor-wasm

# Verify /healthz from the cluster network.
kubectl -n tensor-wasm port-forward svc/tensor-wasm 8080:8080 &
PF_PID=$!
sleep 2
curl -sf http://localhost:8080/healthz
kill $PF_PID
```

If `kubectl rollout status` returns non-zero or `/healthz` does not
return 200 within the timeout, fall through to
[`docs/runbooks/rollback.md`](runbooks/rollback.md):

```bash
helm history tensor-wasm -n tensor-wasm
helm rollback tensor-wasm <previous-revision> -n tensor-wasm
```

### 4.2 Plain-manifest path

For installs from `deploy/k8s/`. Edit the `image:` line in
`20-deployment.yaml`, commit, and re-apply:

```bash
kubectl apply -f deploy/k8s/20-deployment.yaml
kubectl rollout status deployment/tensor-wasm-api -n tensor-wasm --timeout=5m

kubectl -n tensor-wasm port-forward deploy/tensor-wasm-api 8080:8080 &
PF_PID=$!
sleep 2
curl -sf http://localhost:8080/healthz
kill $PF_PID
```

Rollback: revert the `image:` edit and re-apply.

### 4.3 GPU-node-specific notes

If the target bumps `CUDA_ARCH`, the `cudarc-backend` feature default,
or any GPU prerequisite documented in
[`deploy/k8s/README.md` "GPU-node prerequisite checklist"](../deploy/k8s/README.md#gpu-node-prerequisite-checklist),
the upgrade is not just a binary swap. Re-walk the checklist and
confirm driver / device-plugin / `nvidia-container-toolkit` versions
remain compatible before applying.

---

## 5. Docker / systemd upgrade walkthrough

The non-k8s deployment shapes.

### 5.1 systemd

The reference layout (see
[`docs/runbooks/rollback.md` §A](runbooks/rollback.md#a-systemd-the-reference-deployment))
keeps versioned binaries under `/usr/local/lib/tensor-wasm/vX.Y.Z/bin/`
with a symlink at `/usr/local/bin/tensor-wasm`. The upgrade is the
rollback procedure run forward.

```sh
# Pre-flight (Section 2). Capture state the rollback runbook expects.
tensor-wasm --version | tee /tmp/tensor-wasm-upgrade-from.txt
tensor-wasm observe --once > /tmp/tensor-wasm-pre-upgrade-$(date +%s).json
systemctl status tensor-wasm > /tmp/tensor-wasm-status-pre.txt

# Install new binary alongside old; symlink swap is then atomic.
sudo install -D -m 0755 ./target/release/tensor-wasm \
  /usr/local/lib/tensor-wasm/vNEW/bin/tensor-wasm

# Drain (site-specific), stop, swap, sanity-check, start.
sudo /usr/local/sbin/tensor-wasm-drain
sudo systemctl stop tensor-wasm
sudo ln -sfn /usr/local/lib/tensor-wasm/vNEW/bin/tensor-wasm \
              /usr/local/bin/tensor-wasm
/usr/local/bin/tensor-wasm --version
sudo systemctl start tensor-wasm

# Wait for /healthz, then re-add to LB.
until curl -sf http://localhost:8080/healthz > /dev/null; do sleep 1; done
sudo /usr/local/sbin/tensor-wasm-undrain
```

If `/healthz` does not return 200 within ~60 seconds, abort and follow
[`docs/runbooks/rollback.md` §A](runbooks/rollback.md#a-systemd-the-reference-deployment) —
the symlink swap is one `ln -sfn` away from being reversed.

### 5.2 Docker / docker-compose

For installs from the repo root `docker-compose.yml` referenced in
[`docs/DEPLOYMENT.md` §4](DEPLOYMENT.md#4-docker-compose-stack):

```sh
docker inspect tensor-wasm --format '{{.Config.Image}}' \
  | tee /tmp/tensor-wasm-upgrade-from.txt

# Edit docker-compose.yml image: tag for tensor-wasm and commit.
docker compose pull tensor-wasm
docker compose up -d --no-deps tensor-wasm   # --no-deps preserves siblings

until curl -sf http://localhost:8080/healthz > /dev/null; do sleep 1; done
docker inspect tensor-wasm --format '{{.Config.Image}}'
```

Rollback: revert the tag edit and re-run
`docker compose up -d --no-deps tensor-wasm` — the recipe in
[`docs/runbooks/rollback.md` §C](runbooks/rollback.md#c-container-with-docker-compose-file-checked-into-the-repo).

---

## 6. Post-upgrade verification

A green health check is necessary but not sufficient; the metric +
audit + smoke trio is what catches regressions that pass `/healthz`
and fail under load.

- [ ] **Liveness.** `curl -sf http://<host>:8080/healthz` returns 200
      on every replica (k8s: `kubectl -n tensor-wasm get pods` shows
      all `Ready 1/1`).
- [ ] **Metrics.** `curl -s http://<host>:8080/metrics | grep
      tensor_wasm_http_requests_total` returns lines, and the
      duration histogram (`tensor_wasm_http_request_duration_seconds_bucket`)
      has at least one observation. The W2.3 HTTP-metrics family
      documented in
      [`docs/SLO.md` §8](SLO.md#8-disclosure-measured-vs-modeled)
      is the canary that the gateway started and processed a request.
- [ ] **Binary identity.** Confirm the binary identity:
      `curl -s http://<host>:8080/metrics | grep tensor_wasm_build_info`
      should show `version=<new-version>` (and matching `git_sha`,
      `rustc_version`, `profile`, `target` labels). The
      `tensor_wasm_build_info` gauge is the W4.9 info-style metric
      primed at process start; the value is always `1` and the
      payload is the label set. Run this against every replica
      (k8s: loop over `kubectl -n tensor-wasm get pods -l
      app.kubernetes.io/name=tensor-wasm -o name`) — a single replica
      reporting the old version is the canary that the rolling
      restart skipped a pod.
- [ ] **Dashboard + SLO sanity.** Open the dashboard
      ([`docs/dashboards/README.md`](dashboards/README.md)) and watch
      for 15 minutes. The
      [`availability-fast-burn`](runbooks/availability-fast-burn.md)
      alert evaluates over 5 m + 1 h windows
      ([`docs/SLO.md` §5.1](SLO.md#51-fast-burn-page)); 15 minutes
      gives the rate windows enough samples to be meaningful. If
      burn-rate panels exceed their thresholds, treat that as the
      page and follow [`rollback`](runbooks/rollback.md).
- [ ] **Audit-log continuity.** If `TENSOR_WASM_API_AUDIT_LOG` is set
      per the v0.4 wave (see
      [`MIGRATION-v0-to-v1.md` §5 row "Structured audit log"](MIGRATION-v0-to-v1.md#5-behavioural-change-table)),
      tail the destination and confirm new records are landing.
      Cross-check that no records from the pre-flight §2.3 rotation
      were lost.
- [ ] **Smoke test.** Run `tensor-wasm-cli observe --once`; confirm
      `active_instances`, `jobs_in_flight`, and
      `kernel_dispatches_total` panels render. Optionally deploy a
      fixture function and invoke it once. Let `observe` run for one
      minute and watch for monotonic counter advance — a counter flat
      for a full minute under live traffic means the gateway is
      serving but the executor is wedged.

---

## 7. Schema migrations

TensorWasm's only durable on-disk artifact is the snapshot store. The
compatibility promise lives in
[`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) and the
upgrade-relevant subset in
[`MIGRATION-v0-to-v1.md` §6](MIGRATION-v0-to-v1.md#6-snapshot-compatibility).
Key points:

- **v0.5+ snapshots are forward-portable to v1.0+.** Once on v0.5+,
  the upgrade does not touch snapshot storage.
- **Pre-v0.5 snapshots may bump `SNAPSHOT_VERSION`** on a minor
  release. The supported path is "restore under the binary that wrote
  it, re-capture under the new binary" (see
  [`MIGRATION-v0-to-v1.md` §6 "Pre-v0.5 captures"](MIGRATION-v0-to-v1.md#pre-v05-captures)).
- **No standalone `snapshot migrate` subcommand** is planned for v1.0.
  For snapshots predating v0.5 destined for v1.0, the safe sequence
  is stepwise: upgrade to v0.5, re-capture, then proceed.

Consult the reader matrix in
[`docs/SNAPSHOT-COMPATIBILITY.md#format-version--behavior-matrix`](SNAPSHOT-COMPATIBILITY.md#format-version--behavior-matrix)
before any upgrade that crosses a `SNAPSHOT_VERSION` bump.

---

## 8. Rollback

If post-upgrade verification ([§6](#6-post-upgrade-verification))
fails or the burn-rate alerts in
[`docs/SLO.md` §5](SLO.md#5-burn-rate-alerts) fire in the watch
window, defer to
[`docs/runbooks/rollback.md`](runbooks/rollback.md). That runbook is
the single source of truth for reverting a node: systemd, Docker, and
docker-compose shapes; the "rollback itself fails" branch; and the
postmortem-capture requirements that close out the incident.

It is intentionally a separate document — its change cadence
(alert-runbook revisions) differs from this doc's
(release-engineering revisions), and version-locking the two would
let them drift. Every step in
[§4](#4-kubernetes-upgrade-walkthrough) and
[§5](#5-docker--systemd-upgrade-walkthrough) above has a
corresponding revert action in `rollback.md`.

---

## 9. Time budget per strategy

Wall-clock estimates for a four-tenant single-region deployment with
healthy SLO budget and pre-flight already done.

| Strategy | Wall-clock | Worst-case revert | Risk profile |
|---|---|---|---|
| **A. Rolling** (3 replicas) | ~5 min | ~5 min (re-roll previous tag) | Per-tenant cold starts when their pinned replica restarts; sticky routing required |
| **B. Blue/green** (recommended) | ~10 min | < 30 s (Service selector flip) | Lowest risk; requires double capacity during the cut window |
| **C. In-place** (single host) | ~2 min | ~1 min (symlink swap-back) | 5–15 s of 503s during the swap; acceptable for dev / staging / non-paying-user prod |

Guidance, not commitments. Multiply by 1.5× for the first upgrade
against a new deployment shape. Verification window in
[§6](#6-post-upgrade-verification) is the same length regardless of
operator experience.

---

## 10. Communications template

Send before the upgrade window. Adjust the square-bracketed placeholders.

```
Subject: TensorWasm upgrade window: [DATE] [TIME] [TZ] (~[DURATION])

We are upgrading TensorWasm from [CURRENT_VERSION] to [TARGET_VERSION]
in the [REGION] region during the maintenance window above.

What you will see:
- Synchronous invokes (POST /functions/{id}/invoke) may return 5xx
  briefly during the cut. Retry-safe clients should back off.
- Async invokes in flight when the window starts complete on the old
  binary. New async invokes submitted during the cut are queued and
  processed on the new binary.
- No data loss expected. Snapshots are preserved per
  docs/SNAPSHOT-COMPATIBILITY.md.

What you should do:
- If your client does not retry on 5xx, defer non-time-sensitive
  batches until [DATE+1].
- If you rely on a deprecated API surface (see
  docs/MIGRATION-v0-to-v1.md §3), confirm your migration is in place
  before the window. Bare-token TENSOR_WASM_API_TOKENS entries are
  the most common gotcha.

Status updates in [CHANNEL] every 15 minutes during the window.
Degraded behaviour after [END_TIME] → ticket against [TEAM].

— Platform Operations
```

Adapt to your channel norms. The three-bullet structure (what you
will see / what you should do / where to ask) is the load-bearing
part.

---

## 11. Related

- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — v1.0 gate criteria; W3.3
  artifact behind the "roll a TensorWasm fleet from v0.5 to v1.0" line.
- [`docs/MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) — API-surface
  counterpart; deprecation, removed-API, and behavioural-change tables.
- [`docs/runbooks/rollback.md`](runbooks/rollback.md) — reverse of
  this playbook; single source of truth for reverting a bad upgrade.
- [`docs/SLO.md`](SLO.md) — burn-rate alerts the post-upgrade watch
  window monitors; source of the 30% error-budget rule in §2.5.
- [`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) —
  cross-version snapshot reader matrix; referenced in [§7](#7-schema-migrations).
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — production topology and sizing.
- [`docs/dashboards/README.md`](dashboards/README.md) — reference
  Grafana dashboard.
- [`deploy/k8s/README.md`](../deploy/k8s/README.md),
  [`deploy/helm/tensor-wasm/README.md`](../deploy/helm/tensor-wasm/README.md) —
  sources for the [§4](#4-kubernetes-upgrade-walkthrough) walkthroughs.
- [`CHANGELOG.md`](../CHANGELOG.md) — per-release diff; §2.1 pre-flight.
- [`docs/AUDIT-LOG.md`](AUDIT-LOG.md) — audit-log schema referenced in §6.
- [`docs/BACKUP-RESTORE.md`](BACKUP-RESTORE.md) *(planned, W3.7)* —
  full backup and disaster-recovery; §2.3 is the pre-upgrade subset.

---

_Status: W3.3 deliverable, v0.3.7. The strategy and verification
shape is stable for the v0.x line; the time-budget numbers in §9 will
re-baseline once an external design partner (see
[`docs/PATH-TO-V1.md` §6 open decision #6](PATH-TO-V1.md#6-production-design-partners))
reports real-fleet upgrade durations. Updated whenever the W2.7
deploy assets or the W2.6 rollback runbook change shape._
