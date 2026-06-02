<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — v0.x to v1.0 Migration Guide

This document is the operational checklist a v0.x deployment follows to
land on v1.0 cleanly. It is one of two release-engineering artifacts
required by the v1.0.0 gate in [`PATH-TO-V1.md`](PATH-TO-V1.md); the
companion is [`docs/UPGRADE.md`](UPGRADE.md) *(planned, W3.3)* covering
the per-fleet operational sequence.

**Status — template, populated continuously.** This file is shipped
starting in the v0.1.0 era and is updated **on every release** between
now and v1.0. Each minor release adds one row per behavioural change,
deprecation, or removed API to the tables in §3, §4, and §5. By the
time v1.0 ships, the tables together are the exhaustive diff between
v0.1.0 and v1.0.

If you only read one section, skip to [§3 Deprecation table](#3-deprecation-table)
and [§5 Behavioural-change table](#5-behavioural-change-table) — those
two cover everything you have to act on.

## Contents

1. [How to use this doc](#1-how-to-use-this-doc)
2. [Pre-flight](#2-pre-flight)
3. [Deprecation table](#3-deprecation-table)
4. [Removed-API table](#4-removed-api-table)
5. [Behavioural-change table](#5-behavioural-change-table)
6. [Snapshot compatibility](#6-snapshot-compatibility)
7. [Configuration migration](#7-configuration-migration)
8. [Upgrade order](#8-upgrade-order)
9. [Rollback](#9-rollback)
10. [Cross-references](#10-cross-references)

---

## 1. How to use this doc

The audience is a developer or operator who is running a v0.x release
of TensorWasm and needs to land on v1.0 with no surprises. The doc is
organised as a set of **append-only tables**: each row pins itself to
the version it was introduced in, and every row is preserved through
v1.0 so the same checklist works whether you are upgrading from v0.1
direct to v1.0 or stepping through every intermediate release.

### Per-version row policy

For every release on the v0.x line up to v1.0, a corresponding row
lands in the relevant table:

- **Deprecation table** (§3) — a row when a feature is first marked
  deprecated, with the release that flagged it and the release that
  will remove it.
- **Removed-API table** (§4) — a row when the feature is actually
  removed. Per project policy a feature deprecated in v0.X is removed
  no earlier than v0.(X+2) or v1.0, whichever is later, so the
  removed-API table stays empty until late on the v0.x line.
- **Behavioural-change table** (§5) — a row when the *observed*
  behaviour of a stable API changes (status code, header, log emission,
  performance class) even though the API shape did not.

The doc is updated as part of every release-engineering pass; do not
expect to find rows for releases that have not yet shipped. If you are
reading this in a checkout, the tables reflect what was true at the
time of that commit.

### Cross-link reading order

For a clean v0.x → v1.0 upgrade, read in order:

1. This document — actionable diff.
2. [`docs/UPGRADE.md`](UPGRADE.md) *(planned, W3.3)* — fleet-rollout
   sequencing.
3. [`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) — the
   v0.5+ snapshot-format compatibility promise.
4. [`docs/runbooks/rollback.md`](runbooks/rollback.md) — what to do if
   the upgrade misbehaves.
5. [`CHANGELOG.md`](../CHANGELOG.md) — full per-release diff.

### What this doc is not

- **Not a release-notes substitute.** Release notes live in
  [`CHANGELOG.md`](../CHANGELOG.md). This doc is the *migration*
  surface — a strict subset of the changelog filtered to "things a
  v0.x operator must do or know".
- **Not a tutorial.** First-time setup belongs in
  [`docs/GETTING-STARTED.md`](GETTING-STARTED.md).
- **Not a policy doc.** Deprecation, removal, and SemVer policy live
  in [`GOVERNANCE.md`](../GOVERNANCE.md); this file cites those
  policies but does not redefine them.

---

## 2. Pre-flight

Before any upgrade, complete the items in this section. Each item is
something that, if skipped, leaves you without a clean rollback path.

### 2.1 Snapshot a working baseline

A v0.x to v1.0 upgrade is in principle reversible, but the safest
posture is to start with a known-good capture you can restore against.

- [ ] **Capture every active instance** via
      `tensor-wasm-cli snapshot save <instance-id> <path>`. Store
      under a directory tagged with the source version, e.g.
      `~/tensor-wasm-backups/pre-upgrade-v0.4-to-v1.0/`.
- [ ] **Verify each snapshot restores** under the *current*
      (pre-upgrade) binary before proceeding. A snapshot that cannot
      round-trip under the binary that wrote it will not round-trip
      under the new one either.
- [ ] **Record the source version** with `tensor-wasm --version` and
      pin it next to the backups. This is what you roll back to if §9
      becomes necessary.

The v0.5+ snapshot-format compatibility promise (see §6) guarantees
v1.0 can read v0.5 captures; in the interim, **re-capture under each
intermediate version** rather than trying to skip across format
versions. The cross-version reader matrix in
[`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md#format-version--behavior-matrix)
is the source of truth.

### 2.2 Archive audit log streams

If you configured `TENSOR_WASM_API_AUDIT_LOG=file:...` or rely on the
stdout audit stream, rotate the current segment and archive it before
the upgrade. The audit-log schema is stable across patch releases (see
[`docs/AUDIT-LOG.md`](AUDIT-LOG.md#8-stability-guarantees)) but the
file inode changes on restart, and rolling the segment manually keeps
the per-version partition tidy for compliance retention.

- [ ] **Force-rotate** the current `audit.log` using your existing
      `logrotate` job (or its equivalent) and confirm the rotated
      segment compressed cleanly.
- [ ] **Confirm the SIEM caught up** to the rotated segment before
      tearing the old process down — a torn segment is the most
      common source of audit-trail gaps across upgrades.

### 2.3 Snapshot current configuration

Capture the running environment so a rollback restores not just the
binary but the operator-visible knobs.

- [ ] **Dump current env vars** (`TENSOR_WASM_API_*`,
      `TENSOR_WASM_*`) to a file. The `env | grep TENSOR_WASM > env.before`
      idiom is sufficient; for a systemd-managed deployment the
      `EnvironmentFile=` is the source of truth.
- [ ] **Capture the systemd unit** / Helm values / docker-compose
      file. These are what the rollback procedure in §9 expects.
- [ ] **Record the active token allowlist shape.** The deprecation in
      §3 about bare tokens is *only* relevant if your allowlist
      currently contains bare entries — confirm via `grep -c ':tenant='`
      in your secret store.

### 2.4 Confirm rollback target

- [ ] The source binary (`vX.Y.Z`) is available as a re-deployable
      artifact (installed binary, container image, or release tarball).
- [ ] The §9 rollback runbook applies to your deployment shape.

A full disaster-recovery procedure — including data loss scenarios,
lost-host recovery, and corrupted-state recovery — will live in
[`docs/BACKUP-RESTORE.md`](BACKUP-RESTORE.md) *(planned, W3.7)*. Until
that ships, the checklist above is the documented backup procedure;
treat the snapshot and config capture as mandatory rather than
advisory.

---

## 3. Deprecation table

Items that still work today but will stop working at a future
release. Migrate before the **Removal Planned** column to avoid a
hard break.

| Item | Deprecated In | Removal Planned | Migration Path |
|---|---|---|---|
| Bare-token entries in `TENSOR_WASM_API_TOKENS` (no `:tenant=` scope clause) | v0.4 (W2.1) | v1.0 | Rewrite each entry as `token:tenant=*` for explicit wildcard scope, or as `token:tenant=1,2,3` for an explicit allowlist. The API doc has a worked diff: [`crates/tensor-wasm-api/API.md#migration-from-bare-tokens`](../crates/tensor-wasm-api/API.md#migration-from-bare-tokens). |
| `KernelArgsUnsupported` returned for any `args_len > 0` (v0.1.0 contract) | v0.2 (W1.1 — typed-argv lowering lands) | v1.0 | No application change required for callers that pass valid argv; the error is now reserved for sanity-cap rejections only. Callers that *relied on* the unconditional rejection as a feature gate must switch to the explicit `cuda` feature flag or refuse the host capability via WIT. |
| `cust` 0.3.x as the default CUDA backend | v0.2 (W1.2 — `cudarc-backend` flag lands) | v0.3 (target) once cudarc cutover completes | Build with `--features cudarc-backend` to opt into the maintained backend. `cust` 0.3.x is EOL upstream; see [`docs/RISKS.md`](RISKS.md) "Wasmtime/cust pin" and [`docs/CUDARC-SPIKE.md`](CUDARC-SPIKE.md) for the migration plan. The default flips before v0.3 ships. |

**Maintainer note** — the `KernelArgsUnsupported` and `cust 0.3.x`
rows above are deprecations the W1 wave *implies* (W1.1 reclassifies
the error to a sanity-cap; W1.2 introduces the replacement backend)
but are **not yet listed under a `### Deprecated` heading in
[`CHANGELOG.md`](../CHANGELOG.md)**. Both rows should be reconciled
into the next changelog edit; until then this doc is the
operator-facing source of truth for the deprecation.

---

## 4. Removed-API table

| Item | Last Release | Removed In | Replacement |
|---|---|---|---|
| _(empty — no public APIs have been removed yet)_ |   |   |   |

### Removal policy

An API deprecated in v0.X is removed no earlier than **v0.(X+2)** or
**v1.0**, whichever is later. This is the SemVer-compatible cadence
the v1.0 commitment requires (see
[`PATH-TO-V1.md`](PATH-TO-V1.md#what-v10-means)). Removals that ship
inside the v0.x line are themselves treated as minor bumps because
the v0.x line is pre-stability; v1.x removals require a major bump
under SemVer.

Every removal must be preceded by:

1. A merged RFC under [`rfcs/`](../rfcs/) documenting the API being
   removed, the alternatives, and the migration window. See
   [`rfcs/README.md`](../rfcs/README.md) (W1.7) for the template.
2. At least one release in which the API is deprecated with a
   runtime warning (log line, deprecation header, or
   compile-time `#[deprecated]`).
3. A row in §3 above flagging the deprecation, and a row in this
   table when the removal lands.

The formal authority for the removal-RFC requirement is
[`GOVERNANCE.md`](../GOVERNANCE.md).

---

## 5. Behavioural-change table

Items whose API *shape* did not change but whose *observable
behaviour* did. These do not break the public contract under SemVer,
but they will surface in dashboards, alert thresholds, or test
fixtures.

| Behaviour | Before | After | How to detect | What to do |
|---|---|---|---|---|
| HTTP request metrics emission | v0.1 — only executor-side counters were exported; dashboards used `<TODO: emit metric>` markers for request-rate / latency. | v0.3 (W2.3) — gateway exports `tensor_wasm_http_requests_total` (counter, labelled by method/path/status), a request-duration histogram, and an `in_flight` gauge. | Scrape `/metrics` and search for `tensor_wasm_http_`. New series appear with stable label cardinality. | Wire the new series into your dashboards; remove the `<TODO: emit metric>` placeholders. The reference Grafana dashboard at [`docs/dashboards/`](dashboards/README.md) (W2.5) already consumes them. |
| Per-tenant scope enforcement on `/invoke` | v0.1–v0.3 — any token in the allowlist could address any tenant; the `X-TensorWasm-Tenant` header was advisory. | v0.4 (W2.1) — invoke routes refuse cross-tenant access. Bare tokens are coerced to wildcard scope and emit a deprecation warning; scoped tokens enforce strictly. | Application-level: HTTP `403` with `{"error":{"kind":"tenant_scope_denied", ...}}`. Operator-level: one-shot startup warning `bare bearer tokens in TENSOR_WASM_API_TOKENS are deprecated ...`. | Update `TENSOR_WASM_API_TOKENS` to use the `token:tenant=*` or `token:tenant=1,2,3` shape per [`crates/tensor-wasm-api/API.md#per-tenant-scopes`](../crates/tensor-wasm-api/API.md#per-tenant-scopes-tensor_wasm_api_tokens-entry-forms). |
| Structured audit log on state-mutating routes | v0.1–v0.3 — no per-request audit stream; only `tracing` spans and Prometheus counters. | v0.4 (W2.2) — every `POST /functions`, `DELETE /functions/{id}`, `POST /functions/{id}/invoke`, and `POST /functions/{id}/invoke-async` emits one JSON line to the sink selected by `TENSOR_WASM_API_AUDIT_LOG` (default stdout). | Log volume on stdout increases by one record per state-mutating call; container log shippers see a new structured stream. | Either accept the increased stdout volume and consume the records (the schema is stable — see [`docs/AUDIT-LOG.md`](AUDIT-LOG.md#1-record-schema)), pipe to a file with `TENSOR_WASM_API_AUDIT_LOG=file:/path`, or disable entirely with `TENSOR_WASM_API_AUDIT_LOG=none`. |
| Per-token rate limiting | v0.1–v0.1 — no per-token limit; a single process-wide `ConcurrencyLimitLayer(64)` capped total in-flight requests, which was a known limitation (BA-005). | v0.2 (W1.4) — `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST` enable a per-bearer-token token bucket. Unset / `0` keeps the old behaviour (limiter disabled). | HTTP `429` with `{"error":{"kind":"rate_limited", ...}}` and a `Retry-After: <seconds>` header. The integer is ceil((1 − bucket_tokens) / qps), clamped to ≥ 1. | Tune `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST` to your real per-token QPS; see [`crates/tensor-wasm-api/API.md#per-token-rate-limiting`](../crates/tensor-wasm-api/API.md#per-token-rate-limiting) for the defaulting rules. Clients should honour `Retry-After` and back off. |

---

## 6. Snapshot compatibility

The on-disk snapshot format is the artifact most likely to outlive a
binary upgrade. The compatibility promise is documented end-to-end in
[`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md); the
v1.0-relevant clauses are reproduced here.

### The v0.5+ promise

**v1.0 will read every snapshot produced by v0.5+.** Concretely, once
the v0.5 line freezes the snapshot format, every later release on the
v1.x line must accept a snapshot blob byte-for-byte identical to one
produced by any v0.5+ writer. See
[`docs/SNAPSHOT-COMPATIBILITY.md#the-promise`](SNAPSHOT-COMPATIBILITY.md#the-promise).

### Pre-v0.5 captures

In the v0.x pre-freeze window, each minor bump may bump
`SNAPSHOT_VERSION` and refuse older blobs. If you have a snapshot
produced by v0.1 – v0.4 that you want to land on v1.0, the supported
path is:

1. Restore the snapshot under the binary that wrote it.
2. Re-capture under v0.5 (or later).
3. Carry the v0.5 capture forward to v1.0.

A standalone `tensor-wasm-cli snapshot migrate` subcommand is *not*
planned for v1.0; if cross-version migration proves common in beta
deployments it becomes a v1.x item (see
[`docs/SNAPSHOT-COMPATIBILITY.md#migration-paths-supported`](SNAPSHOT-COMPATIBILITY.md#migration-paths-supported)).

### Reader matrix

The version-to-behaviour matrix lives in
[`docs/SNAPSHOT-COMPATIBILITY.md#format-version--behavior-matrix`](SNAPSHOT-COMPATIBILITY.md#format-version--behavior-matrix).
The golden-fixture compatibility suite under
[`crates/tensor-wasm-snapshot/tests/compat.rs`](../crates/tensor-wasm-snapshot/tests/compat.rs)
(W1.3) is the machine-checkable evidence that the promise holds.

---

## 7. Configuration migration

Side-by-side comparison of the env-var surface as of v0.1 and as of
the current (`Unreleased`) line. Operators upgrading from v0.1 to a
post-v0.4 release should expect every `New in` knob below to be
defined explicitly rather than left at its default.

### 7.1 New since v0.1

| Env var | First release | Purpose |
|---|---|---|
| `TENSOR_WASM_API_RATE_LIMIT_QPS` | v0.2 (W1.4) | Per-bearer-token steady-state QPS. Unset/0 disables the limiter (v0.1 behaviour). |
| `TENSOR_WASM_API_RATE_LIMIT_BURST` | v0.2 (W1.4) | Per-bearer-token burst capacity. Unset/0 disables; if one of QPS/BURST is set, the missing knob defaults to `100`/`200` respectively. |
| `TENSOR_WASM_API_AUDIT_LOG` | v0.4 (W2.2) | Audit sink selector. Unset or `stdout` → stdout JSONL; `file:/path` → append-only file; `none` → disabled. |
| `TENSOR_WASM_API_REQUIRE_TENANT` | v0.1 (clarified at v0.4) | When `1`, requires `X-TensorWasm-Tenant` on every request; otherwise defaults to tenant 0. |

### 7.2 Renamed

| Old name | New name | Renamed in |
|---|---|---|
| _(none)_ |   |   |

No env vars have been renamed on the v0.x line. Any future rename
will land here with a transitional release in which both names work
and the old name emits a `tracing::warn!`.

### 7.3 Deprecated entry shapes (env-value level)

The `TENSOR_WASM_API_TOKENS` variable name is unchanged, but the
*shape* of its entries has a deprecated form. See the bare-token row
in §3 and the worked diff in
[`crates/tensor-wasm-api/API.md#migration-from-bare-tokens`](../crates/tensor-wasm-api/API.md#migration-from-bare-tokens).

### 7.4 Side-by-side example

A v0.1-era production config:

```bash
export TENSOR_WASM_API_TOKENS=secret-prod-token,canary-token
```

The same config on the current line (after applying §3 deprecation,
§7.1 new knobs, §5 behavioural changes):

```bash
# Tokens — scoped per the W2.1 surface.
export TENSOR_WASM_API_TOKENS='secret-prod-token:tenant=*,canary-token:tenant=*'

# Rate limit — pick numbers your downstream can absorb.
export TENSOR_WASM_API_RATE_LIMIT_QPS=100
export TENSOR_WASM_API_RATE_LIMIT_BURST=200

# Audit log — explicit destination, even if it equals the default.
export TENSOR_WASM_API_AUDIT_LOG=file:/var/log/tensor-wasm/audit.log
```

---

## 8. Upgrade order

The sequence below is the recommended fleet rollout for a single
TensorWasm node. The fleet-level orchestration (canary, percentage
rollout, blue/green) lives in [`docs/UPGRADE.md`](UPGRADE.md)
*(planned, W3.3)*.

1. **Back up.** Complete every item in §2 — snapshots, audit-log
   archive, config dump.
2. **Drain traffic.** Stop sending new `/invoke` and `/invoke-async`
   requests to the node. The recommended mechanism is the
   load-balancer health check — flip the node's `/healthz` upstream
   weight to zero and wait for in-flight async jobs to settle (poll
   `GET /jobs/{id}` until none are `pending`).
3. **Upgrade the binary.** Replace the binary or container image
   atomically. Do not start the new process until step 4 is staged.
4. **Reload tokens with scope syntax** (if upgrading from a v0.4-and-
   earlier deployment that still uses bare tokens). Apply the
   `:tenant=*` or `:tenant=1,2,...` form per §3 and §7.1. This is the
   one configuration change that *must* happen in the upgrade
   window — leaving bare tokens in place under v1.0 will refuse to
   start once the deprecation removal lands (see the §3 row).
5. **Set new env vars** for the knobs introduced since your previous
   pin (§7.1). Defaults are safe — explicit values are preferred for
   the compliance trail.
6. **Start the new process.** Watch the startup log for:
   - The audit-log `info` line confirming the chosen sink.
   - The rate-limit `info` line confirming the bucket configuration.
   - Absence of the bare-token deprecation warning.
7. **Verify metrics.** Hit `/metrics` and confirm:
   - `tensor_wasm_http_requests_total` is present (the W2.3 metric).
   - `tensor_wasm_active_instances` reads `0`.
   - The HTTP-duration histogram has emitted at least one observation
     for `/healthz`.
8. **Verify a synthetic invocation.** Deploy a fixture function,
   invoke it, confirm a 2xx, and confirm the audit stream caught the
   record (if enabled).
9. **Resume traffic.** Flip the load-balancer weight back to normal
   and watch the SLO burn-rate dashboards
   ([`docs/SLO.md`](SLO.md)) for the next 15 minutes. The
   `availability-fast-burn` alert
   ([`docs/runbooks/availability-fast-burn.md`](runbooks/availability-fast-burn.md))
   is the canary signal for a bad rollout.
10. **Close out.** Archive the §2 pre-upgrade config dump alongside
    the new version's config; this is the artifact §9 rollback
    expects to find.

---

## 9. Rollback

If steps 6–9 above fail or the SLO burn rate exceeds the v1.0
fast-burn threshold within the watch window, fall back to the
operational rollback procedure documented in
[`docs/runbooks/rollback.md`](runbooks/rollback.md) (W2.6). That
runbook is the single source of truth for:

- When to roll back (correlation with availability and latency
  alerts).
- How to roll back without losing audit records (re-mounts the
  pre-upgrade segment).
- How to restore snapshots produced by the previous binary.
- How to record the postmortem signals.

This document defers to the runbook intentionally — the rollback
procedure has its own change cadence and version-locking it inside a
migration doc would let the two drift.

---

## 10. Cross-references

Within this repository:

- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — v1.0 gate criteria; this
  doc is the artifact behind the "MIGRATION-v0-to-v1.md" line item.
- [`CHANGELOG.md`](../CHANGELOG.md) — full per-release diff; this
  doc is a filtered, action-oriented view.
- [`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md) —
  authoritative reference for auth, rate-limit, audit-log, and the
  error-envelope `kind` strings.
- [`docs/AUDIT-LOG.md`](AUDIT-LOG.md) — audit-record schema, sink
  configuration, log-rotation guidance.
- [`docs/SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) —
  cross-version snapshot reader matrix and the v0.5+ promise.
- [`docs/WASMTIME-UPGRADE.md`](WASMTIME-UPGRADE.md) — Wasmtime pin
  cadence; relevant because a Wasmtime major bump always ships a
  companion migration doc of its own (`MIGRATION-<old>-to-<new>.md`).
- [`docs/SLO.md`](SLO.md) — burn-rate thresholds the post-upgrade
  watch window monitors.
- [`docs/runbooks/rollback.md`](runbooks/rollback.md) — operational
  rollback procedure referenced by §9.
- [`GOVERNANCE.md`](../GOVERNANCE.md) — RFC and removal policy
  governing entries in §4.
- [`rfcs/README.md`](../rfcs/README.md) — RFC template required for
  every removal.

Planned cross-references (will resolve as the corresponding wave
items land):

- [`docs/UPGRADE.md`](UPGRADE.md) *(planned, W3.3)* — fleet-rollout
  operations doc; §8 above is the per-node subset.
- [`docs/BACKUP-RESTORE.md`](BACKUP-RESTORE.md) *(planned, W3.7)* —
  full backup and disaster-recovery procedure; §2 above is the
  pre-upgrade subset.

External:

- [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) —
  the policy underlying §4.

---

_Status: template, v0.3.7. Tables in §3 and §5 reflect
deprecations and behavioural changes that have landed in waves W1
and W2 as of the current `Unreleased` line in
[`CHANGELOG.md`](../CHANGELOG.md). §4 is empty by design until a
removal lands. Updated on every release through v1.0._
