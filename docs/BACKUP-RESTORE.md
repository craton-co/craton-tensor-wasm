<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Backup and restore

This document is the v0.4 *"Backup / restore procedure documented and
tested"* exit criterion from [`PATH-TO-V1.md`](PATH-TO-V1.md) (the
Operations workstream). It enumerates what a production TensorWasm
deployment must back up, the strategies the maintainers test
themselves, the supported restore paths, and the validation procedure
that confirms a backup is good before you discover otherwise during a
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md) event.

The document is deliberately narrow: it covers the artefacts a
TensorWasm process owns. Everything else (cluster manifests,
ingress-controller config, the cloud account itself) is the operator's
infrastructure concern and is mentioned only when it crosses into the
TensorWasm surface.

## Contents

1. [Purpose](#1-purpose)
2. [What to back up](#2-what-to-back-up)
3. [What NOT to back up](#3-what-not-to-back-up)
4. [Backup strategies](#4-backup-strategies)
5. [Cadence](#5-cadence)
6. [Restore procedures](#6-restore-procedures)
7. [Test procedure](#7-test-procedure)
8. [Retention policy](#8-retention-policy)
9. [Related](#9-related)

---

## 1. Purpose

TensorWasm v0.4 stores most state in-process. The function registry,
the jobs registry, and the per-token rate-limit buckets are `DashMap`
instances that live and die with the binary (see
[`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md)).
A pod restart or a host loss clears all three by design.

The narrow set of state that *does* benefit from backup is:
snapshot blobs (slow to reconstruct), audit-log archives (no
external regeneration path), auth and TLS secrets (block every
client when lost), and operator configuration (turns a recoverable
host loss into a long outage).

A deployment that backs up none of these is supported but accepts a
longer recovery time during a
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md)
event. This document is the menu of strategies that shorten it.

---

## 2. What to back up

### 2.1 Snapshot blobs

`tensor-wasm snapshot save` writes a zstd-compressed bincode archive
per the wire format in
[`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md).
The compatibility promise is in
[`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md): v1.0 reads
every snapshot produced by v0.5+; within v0.x, minor bumps may break
backwards compatibility and the upgrade path is re-capture on the
older binary then re-import on the newer one.

Snapshots live under `/var/lib/tensor-wasm` (the chart's
`persistence` mount; see
[`deploy/helm/tensor-wasm/templates/pvc.yaml`](../deploy/helm/tensor-wasm/templates/pvc.yaml)).
Conventional layout:

```
/var/lib/tensor-wasm/
    snapshots/<instance-id>-<unix-ms>.tensor-wasm
    audit.log
    audit.log.1.gz
    ...
    jit-cache/                # PTX blobs; NOT a backup target (sec 3)
```

Each `.tensor-wasm` is self-contained: the bincode envelope carries
`tenant_id`, `instance_id`, `created_unix_ms`, and per-blob sizes
without needing an external schema registry. Per-blob caps (see
`FORMAT.md` §"Size caps") bound a single snapshot to 256 MiB
decompressed; the typical compressed on-disk size is 10-200 MiB per
snapshot, dominated by the GPU-memory blob.

### 2.2 Audit-log archives

When `TENSOR_WASM_API_AUDIT_LOG=file:/var/lib/tensor-wasm/audit.log`
the gateway writes one JSON object per state-mutating request (see
[`AUDIT-LOG.md`](AUDIT-LOG.md) §3.2). Each record is ~300-500 bytes;
a node serving 100 state-mutating calls per second produces ~4 GiB
per day before gzip (`AUDIT-LOG.md` §5.3).

`logrotate` with `copytruncate` (`AUDIT-LOG.md` §5.2) leaves a
directory of `audit.log.N.gz` archives alongside the current
`audit.log`. Back the current file up too, accepting that
last-second records will be missing from the captured copy.

When the sink is `stdout` (the chart default), the durable copy is
the log shipper's responsibility — Loki, Vector, Fluent Bit. Treat
that shipper's storage as the audit-log backup in that topology;
this section is then informational only.

### 2.3 Function Wasm payloads

Functions are stored in the gateway's in-memory registry as
`Arc<[u8]>` and discarded on restart (API.md §"POST /functions"). If
the deployment uses a Wasm source-of-record (CI artefact store, S3,
git LFS), that is the canonical source and needs no separate backup.

If the deployment uploads Wasm directly from operator workstations
without an external store, the backup target is the local Wasm
cache. Restore is the `POST /functions` re-upload loop in
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md)
§"Lost host procedure" step 7. Re-uploaded functions get new ids;
clients that pinned ids must be refreshed in lockstep.

### 2.4 Auth secrets and TLS material

The auth surface is:

- `TENSOR_WASM_API_TOKENS` — comma-separated `token:tenant=...`
  entries (API.md §"Per-tenant scopes"). The entire credential
  database; no on-disk session store. Lose it and every downstream
  client breaks until restored or rotated.
- `TENSOR_WASM_API_REQUIRE_TENANT` — boolean policy flag; restore
  from values.yaml.
- TLS server cert + key, if Architecture A from
  [`deployment/mtls.md`](deployment/mtls.md) §3 is in use.
- TLS CA bundle for client-cert validation, if mTLS is enabled.
- Reverse-proxy TLS material under Architecture B (Envoy / nginx) —
  belongs to the proxy, not the gateway.

Treat these as **out-of-band secret-manager state**: back up on the
schedule the secret-manager owner operates (Vault snapshots, AWS
Secrets Manager replication). Do not put plaintext copies in the
same backup as the snapshot blobs — a non-production restore must
not also restore production secrets.

### 2.5 Operator configuration

`deploy/helm/tensor-wasm/values.yaml`, `/etc/tensor-wasm/env` for
systemd, the Grafana JSON in
[`docs/dashboards/`](dashboards/README.md), and the `logrotate(8)`
config for the audit-log sink all belong in the same git repo as
the rest of the deployment's infrastructure code — the "backup" is
git history. If they live only on a single operator's laptop, that
is itself a finding to fix before the next DR rehearsal.

---

## 3. What NOT to back up

The following are intentionally excluded; backing them up is either
useless work or actively harmful.

- **In-memory function registry.** Reconstructed by re-uploading
  Wasm payloads (§2.3). The live `DashMap` is not a documented
  format and function ids are handler-assigned at upload time.
- **In-memory jobs registry.** Async-invoke job state is
  per-process by design; in-flight jobs are lost on failure and
  callers polling `GET /jobs/{id}` against a restored host get
  `404`. Documented limitation, not a recovery target.
- **Rate-limit buckets.** Intentionally ephemeral per API.md
  §"Per-token rate limiting"; a restored host issues a fresh burst,
  same as a normal restart.
- **`tensor_wasm_http_requests_total` and the rest of the
  Prometheus registry.** Prometheus persists the time-series itself
  (W2.3 / `OBSERVABILITY.md`); the gateway only exposes current
  counter values, so backing them up is double-counting.
- **`jit-cache/`** (BLAKE3-keyed PTX blobs). Regenerates on first
  invocation per blueprint — a small one-time warm-up cost.
- **OS-level state** (systemd unit overrides outside
  `/etc/tensor-wasm/`, generic syslog, kernel logs). Host
  playbook, not application backup.

If a future feature persists state to disk that does not appear in
§2, this section must be updated in the same PR.

---

## 4. Backup strategies

The three patterns below are what the maintainers test against. Pick
the one that matches the deployment topology; all three are equally
supported.

### 4.1 PVC volume snapshots (k8s)

The most direct path when the chart's `persistence.enabled=true`
and the cluster has a CSI driver with `VolumeSnapshot` support
(EBS-CSI, GCE-PD, Azure-Disk-CSI, Ceph-CSI, Longhorn).

```sh
# One-time: confirm a VolumeSnapshotClass exists for the PVC's
# storage class.
kubectl get volumesnapshotclass

# On the cadence in sec 5:
cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: tensor-wasm-state-snap-$(date +%Y%m%d-%H%M)
  namespace: tensor-wasm
spec:
  volumeSnapshotClassName: csi-aws-ebs
  source:
    persistentVolumeClaimName: tensor-wasm-state
EOF
kubectl get volumesnapshot -n tensor-wasm -w   # wait for ReadyToUse
```

The snapshot lives in the CSI driver's underlying storage; off-host
durability is whatever that provider promises (usually multi-AZ).
For cross-region durability, copy via the provider's tool
(`aws ec2 copy-snapshot --destination-region ...`).

### 4.2 rsync / restic to off-host storage

For systemd deployments or k8s deployments that want a
filesystem-level off-host copy independent of the CSI driver.

```sh
# One-time:
export RESTIC_REPOSITORY=s3:s3.amazonaws.com/tensor-wasm-backups/host-N
export RESTIC_PASSWORD_FILE=/etc/tensor-wasm/restic-password
sudo restic init

# On the cadence in sec 5:
sudo restic backup /var/lib/tensor-wasm \
    --exclude /var/lib/tensor-wasm/jit-cache \
    --tag tensor-wasm --tag host-N

# Prune per sec 8 retention.
sudo restic forget --keep-hourly 24 --keep-daily 30 --keep-monthly 12 \
    --prune
```

`restic` deduplicates across snapshots, encrypts client-side, and
verifies content-addressed chunks on every read.
`rsync --link-dest` to a sibling host is a supported alternative
when the operator already manages an rsync sink.

### 4.3 Object-store sync (S3 / GCS / Azure Blob)

Simpler than `restic` but no dedup. Storage cost scales linearly
with retention.

```sh
sudo aws s3 sync /var/lib/tensor-wasm/ \
    s3://tensor-wasm-backups/host-N/state/ \
    --exclude "jit-cache/*" --delete --storage-class STANDARD_IA

sudo gsutil -m rsync -r -d -x "jit-cache/.*" \
    /var/lib/tensor-wasm/ gs://tensor-wasm-backups/host-N/state/

sudo az storage blob sync --account-name tensorwasmbackups \
    --container 'host-N' --source /var/lib/tensor-wasm/ \
    --exclude-pattern 'jit-cache/*'
```

---

## 5. Cadence

Recommended schedule; tune to the deployment's RPO target.

| Artefact          | Cadence                              | Rationale                                                                              |
|-------------------|--------------------------------------|-----------------------------------------------------------------------------------------|
| Audit log         | Hourly incremental + daily full      | Compliance windows treat audit gaps as findings; an hourly cadence bounds the gap.     |
| Snapshot blobs    | Daily full                           | Workloads change slowly; the per-blob caps in §2.1 bound the volume per backup run.    |
| Auth secrets      | On change (out of band)              | Rotation is rare; backing up unchanged copies adds risk without benefit.                |
| Helm values.yaml  | On change (git commit)               | git history is the backup; no separate pipeline needed.                                |
| TLS material      | On rotation (out of band)            | Same logic as auth secrets. Tie to the cert renewal cron.                              |

The audit-log and snapshot cadences are the two worth tuning. An
RPO of 5 minutes shrinks the audit-log cadence to 5 minutes;
storage cost rises in proportion (see §8). A backup pipeline that
has not run in the last 25 hours for a "daily" target should fire
a sev-2 alert — a stale backup is functionally no backup.

---

## 6. Restore procedures

The high-level orchestration lives in
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md). This
section documents the artefact-by-artefact "how to put the bytes
back" steps that runbook calls into.

### 6.1 Snapshot restore

Use the existing CLI subcommand
([`crates/tensor-wasm-cli/src/cmd/snapshot.rs`](../crates/tensor-wasm-cli/src/cmd/snapshot.rs)).
The CLI does local-side size validation (`--max-decompressed`)
before any upload.

```sh
tensor-wasm snapshot restore \
    --input /restore/snapshots/instance-abc-1716491220123.tensor-wasm \
    --as-instance instance-abc \
    --server http://localhost:8080

# Bulk restore on a recovered host:
for snap in /restore/snapshots/*.tensor-wasm; do
    inst=$(basename "$snap" | cut -d- -f1-2)
    tensor-wasm snapshot restore --input "$snap" \
        --as-instance "$inst" --server http://localhost:8080 \
        || echo "FAILED: $snap" >> /tmp/restore-failures.txt
done
```

Notes:

- The CLI exits **3** (`FEATURE_NOT_EXPOSED`) if the API server
  has not yet wired the `/instances/restore` route; that endpoint
  is planned but not merged as of v0.4. Until it lands, CLI
  restore is a no-op and the deployment must accept the snapshot
  loss.
- The binary running the restore must satisfy
  [`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) — within
  v0.x the reader version must equal the writer version.
- The envelope CRC32 (FORMAT.md §"CRC32") catches bit-flips on the
  restore path; a mismatch surfaces as a `Serialization` error
  from `SnapshotReader::restore`. Fetch a different backup copy
  rather than trying to "fix" the file.
- A non-empty `restore-failures.txt` is a postmortem finding per
  `runbooks/disaster-recovery.md`.

### 6.2 Audit-log restore

The audit log is append-only JSONL with no "import" semantic. To
restore, copy the archived files back into place:

```sh
sudo install -d -m 0750 -o tensor-wasm -g tensor-wasm /var/lib/tensor-wasm
sudo restic restore latest --include /var/lib/tensor-wasm/audit.log \
    --target /
sudo restic restore latest --include '/var/lib/tensor-wasm/audit.log.*' \
    --target /
```

The current `audit.log` needs no special "merge" — the gateway
holds it open with `O_APPEND` (`AUDIT-LOG.md` §3.2). Restored
historical records appear before process-local ones, which is the
correct ordering for SIEM ingest by `ts_unix_ms`. If the SIEM
already ingested some of the restored window from the live
`stdout` mirror, expect duplicates and dedupe on `request_id`
(UUIDv4, stable per `AUDIT-LOG.md` §1).

### 6.3 Secrets restore

Re-apply the env var (systemd) or the k8s Secret. End-to-end in
`runbooks/disaster-recovery.md` §"Lost auth state procedure"; the
artefact-level call is one of:

```sh
# k8s:
vault kv get -field=tokens secret/tensor-wasm/prod | \
    kubectl create secret generic tensor-wasm-tokens \
        --from-literal=TENSOR_WASM_API_TOKENS="$(cat -)" \
        --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deploy/tensor-wasm -n tensor-wasm

# systemd: rewrite /etc/tensor-wasm/env then `systemctl restart tensor-wasm`.
```

### 6.4 Helm values restore

`git checkout` the values file from the infrastructure repo and
`helm upgrade --install ... -f values.yaml`. The chart is
idempotent on re-apply.

---

## 7. Test procedure

A backup that has never been restored is not a backup. The
maintainers test every strategy in §4 monthly against a
non-production deployment.

### 7.1 Snapshot integrity check (offline, no restore)

The fastest "is this snapshot file good?" check. The
`SnapshotReader::restore` API is the canonical parser; build a tiny
harness once and run it against the backup:

```rust
// examples/snapshot-verify.rs (not yet shipped — see footer).
use tensor_wasm_snapshot::SnapshotReader;
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path)?;
    let snap = SnapshotReader::new().restore(&bytes)?;
    println!("ok: tenant={} instance={} wasm={}B gpu={}B regs={}B",
        snap.metadata.tenant_id, snap.metadata.instance_id,
        snap.wasm_memory.len(), snap.gpu_memory.len(),
        snap.registers.len());
    Ok(())
}
```

The reader validates the CRC32 and the per-blob size caps from
`FORMAT.md`; a parse failure means the file is corrupt and must be
re-fetched from a different backup. Do not try to "fix" it.

### 7.2 Audit-log JSON validation

Every restored `audit.log*` file must round-trip cleanly through
`jq` line-by-line. A failure means the file was truncated,
corrupted, or rotated mid-write.

```sh
# Validate every line parses as JSON and carries the required fields.
for f in /restore/audit.log /restore/audit.log.*.gz; do
    case "$f" in *.gz) reader="gzip -dc" ;; *) reader="cat" ;; esac
    if ! $reader "$f" | jq -ce \
        'has("ts_unix_ms") and has("request_id") and has("actor") and
         has("action") and has("outcome") and has("latency_ms")' \
        > /dev/null; then
        echo "BAD: $f"
    fi
done
```

### 7.3 End-to-end restore drill

Once per quarter, do a full restore into a sandbox cluster:
`helm install` (or systemd equivalent) with production values,
`persistence.existingClaim` pointing at a fresh PVC restored from
the latest backup, the auth secret from a vault-of-record copy
(not the production secret), then run the verification block from
[`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md)
§"Verification". Time the run vs the RTO target; document any
gap. Vary the scenario each quarter so 2 and 3 from
`disaster-recovery.md` get exercised, not just scenario 1.

---

## 8. Retention policy

Default: **30 days hot, 1 year cold**. The split is the
storage-tier boundary, not the data lifecycle.

| Tier  | Window      | Where                                                  |
|-------|-------------|--------------------------------------------------------|
| Hot   | 30 days     | S3 Standard / GCS Standard / Azure Hot, or primary CSI |
| Cold  | 11 months   | S3 Glacier / GCS Coldline / Azure Cool                 |
| —     | > 12 months | Deleted                                                |

Audit-log retention is the compliance window — if your auditor
requires 24 months, extend the cold tier to 23 months. Snapshot-blob
retention is usually driven by storage cost rather than compliance;
30 days hot + delete is enough since snapshots become semantically
stale (source instances drift) within days.

Lifecycle is configurable per storage backend. For AWS S3 use
`aws s3api put-bucket-lifecycle-configuration` with a Transition to
`GLACIER` at day 30 and an `Expiration` at day 365. For `restic`,
use the `--keep-*` flags shown in §4.2 and run
`restic forget --prune` on the same cron as the backup.

---

## 9. Related

- [`PATH-TO-V1.md`](PATH-TO-V1.md) — v0.4 Operations exit
  criterion this document satisfies.
- [`runbooks/disaster-recovery.md`](runbooks/disaster-recovery.md)
  — the playbook consuming the backups defined here.
- [`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) and
  [`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md)
  — cross-version rules and wire format for snapshot blobs.
- [`AUDIT-LOG.md`](AUDIT-LOG.md) — schema, sink, rotation for
  audit-log artefacts.
- [`deployment/mtls.md`](deployment/mtls.md) — TLS material
  covered by §2.4 / §6.3 when mTLS is enabled.
- [`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md)
  — auth surface and in-memory registries determining §2 vs §3.
- [`deploy/helm/tensor-wasm/values.yaml`](../deploy/helm/tensor-wasm/values.yaml)
  — the `persistence`, `auth`, and `extraEnv` keys.
- [`UPGRADE.md`](UPGRADE.md) — version-skew guarantees
  constraining the snapshot-restore section.

---

_Status: v0.4 release. The strategies in §4 and the cadence in §5
are what the maintainers test; the test procedure in §7 is the
contract a production deployment should mirror. A
`tensor-wasm snapshot verify` subcommand to replace the §7.1
ad-hoc Rust harness is recommended for v0.5 — track in the
operations workstream._
