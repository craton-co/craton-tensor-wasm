<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# disaster-recovery

Manual procedure for bringing a TensorWasm deployment back online after the
host, the persistent storage, or the auth state has been destroyed. Not an
alert; referenced from [`SLO.md`](../SLO.md) §4.5 (catastrophic budget
loss) and from the v0.4 exit criterion *"Disaster-recovery runbook: lost
host, lost storage, lost auth state"* in [`PATH-TO-V1.md`](../PATH-TO-V1.md).
Severity: **manual** (invoked by the operator after another runbook, an
infrastructure event, or an external report establishes that recovery is
required).

This runbook is a *procedure* runbook — it does not follow the nine-section
alert template described in [`README.md`](README.md) §"Runbook contract". It
sits alongside [`rollback.md`](rollback.md) and
[`oncall-paging.md`](oncall-paging.md) as the third manual playbook in the
directory.

## When to use this runbook

Use this runbook when one of the three scenarios below applies. In every
case, the underlying assumption is that **a normal `tensor-wasm serve`
restart will not recover the deployment** because some piece of state the
process depends on is no longer reachable.

1. **Lost host.** The machine running `tensor-wasm-api` is gone, unreachable,
   or has degraded past the point where `systemctl start tensor-wasm` will
   succeed. Examples: hardware failure, hypervisor loss, cloud-provider
   instance terminated, disk-corruption rendering the OS unbootable. The
   binary, config, and any local snapshots on the failed host are
   assumed lost. The fix is to re-provision a new host from clean
   infrastructure and re-bind it into the deployment.
2. **Lost storage.** The host is intact but the data volume mounted at
   `/var/lib/tensor-wasm` (the PVC backing `persistence.enabled=true` in
   the Helm chart, or the equivalent local directory in the systemd
   reference deployment) is gone or unreadable. Examples: PVC accidentally
   deleted, EBS volume detached and reformatted, disk replaced without
   restoring its contents, ransomware encryption. The fix is to restore
   the volume from off-host backup (per [`BACKUP-RESTORE.md`](../BACKUP-RESTORE.md))
   and remount.
3. **Lost auth state.** The `TENSOR_WASM_API_TOKENS` env var is gone from
   wherever it was kept (k8s Secret deleted, env-file overwritten, secret
   manager rotated without a copy, operator-of-record left the team). The
   binary will start, but no caller can authenticate. The fix is to
   restore the secret from its backup or rotate to fresh tokens and
   redistribute. **There is no scope state to lose** — scopes live in the
   `token:tenant=...` entries inside `TENSOR_WASM_API_TOKENS` itself,
   parsed at startup per [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md#per-tenant-scopes-tensor_wasm_api_tokens-entry-forms);
   they are not persisted to disk.

If none of the three scenarios applies, this runbook is not the right one.
A scrolling outage with the host and storage intact is usually a
[`rollback.md`](rollback.md) candidate, or one of the burn-rate alerts.

## Severity assessment

Walk this decision tree before you start typing commands. The wrong
scenario diagnosis turns a recoverable outage into a worse one.

1. **Is the host reachable on SSH or `kubectl exec`?**
   - **No** → host is gone. Scenario 1 (lost host).
   - **Yes** → continue.
2. **Does `ls /var/lib/tensor-wasm` (or the chart's PVC mount) return
   the expected snapshot / audit-log directory contents?**
   - **No** (directory empty, permission denied, mount missing) →
     Scenario 2 (lost storage).
   - **Yes** → continue.
3. **Does a known-good bearer token return `200` against `/healthz` via
   `Authorization: Bearer <token>`?** (`/healthz` itself does not check
   auth, but any state-mutating call does — use the manifest-listing
   procedure in `tensor-wasm observe --once` if you have it.)
   - **No** (every request is `401 unauthorized`) → Scenario 3 (lost
     auth state).
   - **Yes** → this is not a DR scenario; check the burn-rate runbooks.

It is possible to be in **more than one scenario simultaneously**
(e.g. a host loss that also destroyed the local secret cache). When that
happens, work scenarios in the order **1 → 2 → 3**: bring up the host
first, then restore storage, then restore auth state. The order matters
because each step's verification depends on the previous one being done.

## Lost host procedure

The reference deployment runs as a systemd unit on a Linux host with a
versioned binary at `/usr/local/lib/tensor-wasm/vX.Y.Z/bin/tensor-wasm`
and a config file at `/etc/tensor-wasm/env`. Adapt the paths if the
deployment uses different conventions.

```sh
# 1. Provision a new host of the same shape (instance type, kernel,
#    GPU SKU if applicable). The exact command depends on the
#    infrastructure tool — terraform, pulumi, or a cloud-provider CLI.
#    Example for AWS:
aws ec2 run-instances --image-id ami-... --instance-type ... \
    --key-name ... --security-group-ids ... --subnet-id ...

# 2. Install the TensorWasm binary at the same version the lost host
#    was serving. Pull the artefact from the release store; do NOT
#    install a newer version during DR (version-mismatch surprises
#    compound the incident).
curl -fSL https://releases.example.com/tensor-wasm/vX.Y.Z/tensor-wasm-linux-x86_64.tar.gz \
    | sudo tar -C /usr/local/lib/tensor-wasm/vX.Y.Z -xz
sudo ln -sfn /usr/local/lib/tensor-wasm/vX.Y.Z/bin/tensor-wasm \
              /usr/local/bin/tensor-wasm
/usr/local/bin/tensor-wasm --version    # must print vX.Y.Z

# 3. Restore the env file from the config-manager of record. The file
#    contains TENSOR_WASM_API_TOKENS, TENSOR_WASM_API_AUDIT_LOG,
#    TENSOR_WASM_API_RATE_LIMIT_*, OTEL_EXPORTER_OTLP_ENDPOINT, and
#    any extra knobs from values.yaml::extraEnv. Replace the example
#    below with the actual fetch:
sudo install -d -m 0755 /etc/tensor-wasm
sudo vault kv get -field=env secret/tensor-wasm/prod \
    | sudo tee /etc/tensor-wasm/env > /dev/null
sudo chmod 0640 /etc/tensor-wasm/env
sudo chown tensor-wasm:tensor-wasm /etc/tensor-wasm/env

# 4. Restore the persistent data directory.
#    4a. If the deployment uses a PVC / external volume, attach it now.
#        Skip to step 5 once mounted at /var/lib/tensor-wasm.
#    4b. If the deployment does not use persistence, OR the PVC was
#        also lost, see "Lost storage procedure" below before
#        continuing. An empty /var/lib/tensor-wasm is acceptable
#        provided you accept that all snapshots and the local audit
#        log archive are gone.
sudo install -d -m 0750 -o tensor-wasm -g tensor-wasm /var/lib/tensor-wasm

# 5. Install the systemd unit and start the service.
sudo cp /usr/local/lib/tensor-wasm/vX.Y.Z/share/tensor-wasm.service \
        /etc/systemd/system/tensor-wasm.service
sudo systemctl daemon-reload
sudo systemctl enable --now tensor-wasm

# 6. Wait for /healthz.
until curl -sf http://localhost:8080/healthz > /dev/null; do
    sleep 1
done

# 7. Re-deploy functions. The function registry is in-memory only (see
#    crates/tensor-wasm-api/API.md "POST /functions"); a host loss
#    means every previously-deployed function id is gone. Re-upload
#    each function's Wasm bytes from the source-of-record (CI artefact
#    store, git LFS, S3 bucket). Note that the new function ids will
#    differ from the old ones; clients that pinned ids must be
#    refreshed in lockstep.
for wasm in /opt/tensor-wasm-functions/*.wasm; do
    name=$(basename "$wasm" .wasm)
    b64=$(base64 -w0 < "$wasm")
    curl -sf -X POST http://localhost:8080/functions \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H 'Content-Type: application/json' \
        -d "{\"name\":\"$name\",\"wasm_b64\":\"$b64\"}"
done

# 8. Re-add to the load-balancer rotation. The exact command depends
#    on the LB; example for a generic out-of-rotation script:
sudo /usr/local/sbin/tensor-wasm-undrain
```

The k8s-Helm equivalent is shorter (`kubectl delete pod` plus a fresh
`helm upgrade`), but the function re-upload step is identical — the
function registry is per-pod in-memory state and a Pod restart loses
it. If the deployment depends on a pre-populated function set, run
the function-deploy loop above as a Helm post-install hook or as a
Job pinned to the same Service.

## Lost storage procedure

The PVC (or local data directory) carries three things worth thinking
about, all documented in [`BACKUP-RESTORE.md`](../BACKUP-RESTORE.md) §2:

- **Snapshots** — zstd-compressed bincode blobs produced by
  `tensor-wasm snapshot save`. Format documented in
  [`crates/tensor-wasm-snapshot/FORMAT.md`](../../crates/tensor-wasm-snapshot/FORMAT.md).
  Cross-version compatibility per
  [`SNAPSHOT-COMPATIBILITY.md`](../SNAPSHOT-COMPATIBILITY.md).
- **Audit-log archive** — append-only JSONL files written when
  `TENSOR_WASM_API_AUDIT_LOG=file:/var/lib/tensor-wasm/audit.log`
  (see [`AUDIT-LOG.md`](../AUDIT-LOG.md) §3.2). Rotated by `logrotate`
  with `copytruncate` per `AUDIT-LOG.md` §5.2.
- **JIT cache** — BLAKE3-keyed PTX blobs. Regenerated on first
  invocation; safe to lose, expect a one-time warm-up cost on the
  first call per blueprint.

```sh
# 1. Identify what was on the lost volume. If the deployment kept an
#    inventory (advised — file a separate task to add one if not),
#    consult it now. Otherwise infer from the values.yaml /
#    /etc/tensor-wasm/env paths:
#       /var/lib/tensor-wasm/snapshots/*.tensor-wasm
#       /var/lib/tensor-wasm/audit.log            (current, if file: sink)
#       /var/lib/tensor-wasm/audit.log.*.gz       (rotated archives)
#       /var/lib/tensor-wasm/jit-cache/           (PTX cache; disposable)

# 2. Restore from off-host backup per BACKUP-RESTORE.md sec 6.
#    The exact command depends on the backup strategy chosen there.
#    Examples (pick the one that matches your strategy):

# 2a. PVC / VolumeSnapshot restore.
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tensor-wasm-state-restored
  namespace: tensor-wasm
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
  dataSource:
    name: tensor-wasm-state-snap-YYYYMMDD
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
helm upgrade tensor-wasm ./deploy/helm/tensor-wasm \
    -n tensor-wasm --reuse-values \
    --set persistence.existingClaim=tensor-wasm-state-restored

# 2b. restic restore.
sudo restic -r s3:s3.amazonaws.com/tensor-wasm-backups/host-N \
    restore latest --target /var/lib/tensor-wasm

# 2c. aws s3 sync restore.
sudo aws s3 sync s3://tensor-wasm-backups/host-N/state/ \
    /var/lib/tensor-wasm/

# 3. If no backup exists, accept the loss and replay from upstream
#    sources where possible:
#    - Snapshots: gone. Affected instances must be re-warmed by
#      re-invoking the function from scratch; first-call latency
#      will spike. See COLD-START.md for the cost.
#    - Audit log: gone for the missing window. Note this in the
#      incident postmortem; downstream SIEM may have a partial
#      copy from the live stdout/OTLP forwarders.
#    - JIT cache: not a problem — regenerates on first use.

# 4. Restart the service so the new volume is picked up.
sudo systemctl restart tensor-wasm
# Or, for k8s:
kubectl rollout restart deploy/tensor-wasm -n tensor-wasm
```

Snapshots restored from off-host backup must satisfy the version
constraints in [`SNAPSHOT-COMPATIBILITY.md`](../SNAPSHOT-COMPATIBILITY.md):
the reader version must be greater than or equal to the writer version,
and within v0.x the formats may not be backwards-compatible across
minor bumps. If the backup is from a release line older than the
current binary, the recommended path is to restore on the matching
older binary, re-capture, and import the re-captured snapshot under
the current binary.

## Lost auth state procedure

`TENSOR_WASM_API_TOKENS` is the only piece of auth state the gateway
keeps. There is no on-disk credential database, no session table, no
refresh-token store. The variable lives in one of:

- A k8s `Secret` (rendered by the chart's `templates/secret.yaml`,
  or referenced by `auth.existingSecret`).
- A systemd `EnvironmentFile=` (e.g. `/etc/tensor-wasm/env`).
- An external secret manager (Vault, AWS Secrets Manager, GCP Secret
  Manager) pulled into one of the above at boot.

```sh
# Case A: the source-of-record still has a valid copy.
# 1. Re-fetch and reapply.
vault kv get -field=tokens secret/tensor-wasm/prod | \
    kubectl create secret generic tensor-wasm-tokens \
        --from-literal=TENSOR_WASM_API_TOKENS="$(cat -)" \
        --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deploy/tensor-wasm -n tensor-wasm

# Case B: the source-of-record is also gone. Issue fresh tokens.
# 1. Generate replacement tokens (high-entropy strings; the gateway
#    does not constrain shape beyond non-empty).
ADMIN_TOKEN=$(openssl rand -base64 48 | tr -d '/=+')
TENANT_TOKEN=$(openssl rand -base64 48 | tr -d '/=+')

# 2. Install the new allowlist.
kubectl create secret generic tensor-wasm-tokens \
    --from-literal=TENSOR_WASM_API_TOKENS="${ADMIN_TOKEN}:tenant=*,${TENANT_TOKEN}:tenant=7" \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deploy/tensor-wasm -n tensor-wasm

# 3. Distribute the new tokens to the downstream clients that need
#    them. This is the painful step: every consumer must update its
#    Bearer header before traffic resumes successfully. Coordinate
#    via the incident channel.

# 4. Record the rotation in the audit / incident log so the next
#    operator knows why the token ids change post-recovery (token_id
#    is a SipHash of the token string — see AUDIT-LOG.md sec 4.3).
```

Rotating to fresh tokens is **destructive of existing client
authentication** by design — the whole point of the v0.4 scoped-token
surface is that the gateway cannot tell a "lost" token from a stolen
one. If you cannot coordinate the client cutover, the only safe
alternative is to leave the deployment unauthenticatable until the
original token is recovered. Do not start the gateway with an empty
`TENSOR_WASM_API_TOKENS` to "unblock" callers — that drops the
deployment into **dev mode** (see API.md "Authentication"), which
admits every request including unauthenticated ones. The audit log
will fingerprint the misconfiguration with `actor.scope.kind: "dev"`
records (per [`AUDIT-LOG.md`](../AUDIT-LOG.md) §1) but the damage will
already be done.

## Verification

After completing the relevant scenario above, before declaring the
incident over:

1. **Confirm `/healthz`.** `curl -sf http://localhost:8080/healthz`
   must return `{"status":"ok"}`. The endpoint should respond well
   under 10 ms per [`healthz-slow.md`](healthz-slow.md).
2. **Confirm `/metrics` is incrementing.** `curl -s http://localhost:8080/metrics`
   should show `tensor_wasm_http_requests_total` advancing across two
   consecutive fetches.
3. **Smoke-test via `tensor-wasm observe`.** The CLI dashboard is
   the fastest end-to-end check that auth, metrics, and the router
   are all healthy:
   ```sh
   TENSOR_WASM_TOKEN="$ADMIN_TOKEN" tensor-wasm observe \
       --server http://localhost:8080 --once
   ```
4. **Confirm an authenticated state-mutating call.** Deploy a tiny
   throwaway Wasm module to verify the `POST /functions` path works
   end-to-end with the restored token; then `DELETE` it to clean up:
   ```sh
   ID=$(curl -sf -X POST http://localhost:8080/functions \
       -H "Authorization: Bearer $ADMIN_TOKEN" \
       -H 'Content-Type: application/json' \
       -d '{"name":"dr-smoke","wasm_b64":"AGFzbQEAAAA="}' | jq -r .id)
   curl -sf -X DELETE "http://localhost:8080/functions/$ID" \
       -H "Authorization: Bearer $ADMIN_TOKEN"
   ```
5. **Confirm the audit sink is writing.** If `TENSOR_WASM_API_AUDIT_LOG`
   is `file:...`, `tail -n 1 /var/lib/tensor-wasm/audit.log` must show
   the smoke-test create/delete pair. If it is `stdout`, `kubectl logs`
   or `journalctl -u tensor-wasm -n 20` must show the same JSONL lines.
6. **Watch any alerts that fired during the incident clear.** Burn-rate
   windows take 5-30 minutes to refill after the underlying signal
   recovers — do not declare the incident closed until the originating
   alert resolves.

If verification step 4 fails (auth wrong) but step 1 succeeds, the
binary is up but the new tokens were not loaded — check
`kubectl describe pod` / `systemctl show tensor-wasm` for the env
the process actually sees.

## Postmortem

A disaster-recovery event always produces a follow-up. Capture, in the
incident issue:

- Which of the three scenarios applied (or which combination).
- The trigger event with timestamp (`aws health describe-events`,
  `kubectl get events`, the on-call paging timestamp).
- The recovery-time-objective (RTO) target vs the observed recovery
  time. If the RTO was missed, the gap is a finding in its own right.
- Whether the most recent backup was usable. If not — backup is
  broken; this is a sev-2 finding even if the recovery succeeded by
  other means.
- The function-registry replay output (which functions were
  re-uploaded, with which new ids, which clients were notified).
- For scenario 3: the rotation timestamps, the new token ids
  (hashes, not the tokens themselves), the downstream-client cutover
  list.
- A pre-incident control that would have prevented or detected
  the scenario earlier (multi-AZ replication, off-host backup
  verification, secret-manager replication), filed as a follow-up.

The incident channel and postmortem are the durable record; DR
events do not need a [`CHANGELOG.md`](../../CHANGELOG.md) entry.

## Related

- [`BACKUP-RESTORE.md`](../BACKUP-RESTORE.md) — what to back up
  and how to restore it; this runbook assumes that document's
  strategy is in place.
- [`SNAPSHOT-COMPATIBILITY.md`](../SNAPSHOT-COMPATIBILITY.md) —
  version-skew rules constraining which snapshots a recovered
  binary can read.
- [`AUDIT-LOG.md`](../AUDIT-LOG.md) — schema and rotation guidance
  for the file-sink archives in the off-host backup.
- [`rollback.md`](rollback.md), [`oncall-paging.md`](oncall-paging.md)
  — sibling procedure runbooks (less-catastrophic recovery; paging
  escalation).
- [`SLO.md`](../SLO.md) §4.5 — error-budget consumption thresholds;
  a DR event consumes the monthly budget in full.
- [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md)
  — auth model and in-memory registry shape this runbook
  re-populates.
- [`deploy/helm/tensor-wasm/values.yaml`](../../deploy/helm/tensor-wasm/values.yaml)
  — the `persistence`, `auth`, and `extraEnv` keys re-applied here.
