<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# rollback

Manual procedure for reverting a TensorWasm node from a bad release
back to the last known-good one. Not an alert; referenced by
[`SLO.md`](../SLO.md) §4.5 and by the availability and latency alert
runbooks in this directory. Severity: **manual** (invoked by the
operator after another runbook decides a rollback is warranted).

## When to roll back

Roll back when any of the following is true, in this order of
preference:

1. The [`availability-fast-burn.md`](availability-fast-burn.md) alert
   is firing and a recent deploy plausibly correlates.
2. The [`availability-slow-burn.md`](availability-slow-burn.md) alert
   has been firing for 30+ minutes and is not abating.
3. The [`invoke-latency-spike.md`](invoke-latency-spike.md) or
   [`dispatch-latency-spike.md`](dispatch-latency-spike.md) alerts
   are firing and a deploy in the last 24 hours touched
   `tensor-wasm-api`, `tensor-wasm-exec`, or `tensor-wasm-wasi-gpu`.
4. A monthly error budget (per [`SLO.md`](../SLO.md) §4.5) has been
   consumed at more than 50% in a 24-hour window and the cause is
   suspected to be a release.

If none of these are true, do not roll back — investigate first.
Rollback is destructive of in-flight in-memory state (active
instances, warm JIT cache, GPU context) and should not be used as a
diagnostic.

## Prerequisites

Before invoking this procedure, confirm:

- The last known-good release version (call it `vX.Y.Z`) is
  available as an installed binary, a downloadable artifact, or a
  container image. Check `~/.tensor-wasm/releases/` or your release
  artefact store.
- A maintenance window is acceptable. Rollback drops all active
  instances; in-flight `/invoke` calls return 5xx for the duration
  of the restart (typically 5-15 seconds).
- The current bad release version (call it `vA.B.C`) is recorded
  somewhere durable. `tensor-wasm --version` captures it; record it
  for the postmortem before doing anything else.

## Procedure

The exact commands vary by deployment topology. Pick the section
that matches the environment.

### A. systemd (the reference deployment)

```sh
# 1. Record current state for the postmortem.
tensor-wasm --version | tee /tmp/tensor-wasm-rollback-from.txt
tensor-wasm observe --once > /tmp/tensor-wasm-pre-rollback-$(date +%s).json
systemctl status tensor-wasm > /tmp/tensor-wasm-status-pre.txt

# 2. Drain new traffic if behind a load balancer. The exact command
# depends on the LB; example for a generic out-of-rotation script:
sudo /usr/local/sbin/tensor-wasm-drain  # or equivalent for your LB

# 3. Stop the current binary.
sudo systemctl stop tensor-wasm

# 4. Swap binaries. The reference layout uses a symlink at
# /usr/local/bin/tensor-wasm pointing at the versioned binary.
sudo ln -sfn /usr/local/lib/tensor-wasm/vX.Y.Z/bin/tensor-wasm \
              /usr/local/bin/tensor-wasm

# 5. Verify the symlink resolved correctly before starting.
/usr/local/bin/tensor-wasm --version  # should print vX.Y.Z

# 6. Start the old binary.
sudo systemctl start tensor-wasm

# 7. Wait for /healthz to report ready.
until curl -sf http://localhost:8080/healthz > /dev/null; do
  sleep 1
done

# 8. Re-add to LB rotation.
sudo /usr/local/sbin/tensor-wasm-undrain  # or equivalent
```

### B. Docker / docker-compose

```sh
# 1. Record current state.
docker inspect tensor-wasm --format '{{.Config.Image}}' | tee /tmp/tensor-wasm-rollback-from.txt

# 2. Drain traffic at the LB as above.

# 3. Restart with the previous image tag.
docker stop tensor-wasm
docker rm tensor-wasm
docker run -d --name tensor-wasm \
  --env-file /etc/tensor-wasm/env \
  -p 8080:8080 \
  -v /var/lib/tensor-wasm:/var/lib/tensor-wasm \
  ghcr.io/craton-co/tensor-wasm:vX.Y.Z

# 4. Wait for /healthz.
until curl -sf http://localhost:8080/healthz > /dev/null; do
  sleep 1
done

# 5. Re-add to LB.
```

### C. Container with docker-compose file checked into the repo

If a `docker-compose.yml` pins the image tag, edit the tag, then:

```sh
docker compose pull tensor-wasm
docker compose up -d --no-deps tensor-wasm
```

`--no-deps` ensures only TensorWasm is recreated, not adjacent
services (Prometheus, Jaeger).

## Verification

After the rollback, before declaring the incident over:

1. **Confirm version.** `tensor-wasm --version` (or
   `docker exec tensor-wasm tensor-wasm --version`) must print
   `vX.Y.Z`.
2. **Confirm health.** `curl -sf http://localhost:8080/healthz` must
   return 200. The endpoint should respond in well under 10 ms per
   [`healthz-slow.md`](healthz-slow.md).
3. **Confirm metrics resume.** `curl -s http://localhost:8080/metrics | head -20`
   should show counter values incrementing across two consecutive
   fetches.
4. **Watch the originating alert clear.** The alert that triggered
   the rollback should resolve within 5-15 minutes (the time it
   takes for the rate-window expressions to refill with healthy
   data). If it does not, the rollback did not fix the cause and
   the procedure was a misdiagnosis.
5. **Confirm the SLO summary recovers.** The dashboard's top-row
   stats for `availability_http`, `error_rate_invoke`, and the
   relevant latency P95s should return to or below their threshold
   values.

## What to do if the rollback fails

If `/healthz` does not return 200 within 60 seconds of starting the
old binary:

1. Check `journalctl -u tensor-wasm --since '5 minutes ago'` for
   binary-incompatible state — typically a snapshot or schema format
   that the old version cannot read.
2. If the snapshot format is the issue and the rollback cannot
   proceed, take the node offline (`systemctl stop tensor-wasm` and
   leave it stopped) and escalate per
   [`oncall-paging.md`](oncall-paging.md). A bad release that cannot
   be rolled back is a sev-1 incident in its own right.
3. Do not try to "roll forward" to a third version under time
   pressure — that compounds the incident. Page the on-call lead
   instead and serve maintenance from upstream.

## Postmortem requirements

A rollback always produces a follow-up. Capture, in the incident
issue:

- `vA.B.C` (the bad release) and `vX.Y.Z` (the rolled-back-to
  release).
- The trigger alert (which runbook called this procedure).
- The `/tmp/tensor-wasm-pre-rollback-*.json` observe capture from
  step 1.
- `git log vX.Y.Z..vA.B.C` — the changeset that introduced the
  problem.
- The hypothesis for which specific commit caused the regression,
  if identifiable. If not identifiable, that is itself a finding —
  the release process needs better staging coverage.
- A pre-merge test or canary check that would have caught the
  regression, added in a follow-up PR.

The next release notes ([`CHANGELOG.md`](../CHANGELOG.md)) must
acknowledge the rollback under "Operator-visible behaviour change"
per [`SLO.md`](../SLO.md) §9.

## Related

- [`SLO.md`](../SLO.md) §4.5 — error-budget consumption thresholds
  that justify a rollback.
- [`availability-fast-burn.md`](availability-fast-burn.md),
  [`availability-slow-burn.md`](availability-slow-burn.md),
  [`invoke-latency-spike.md`](invoke-latency-spike.md),
  [`dispatch-latency-spike.md`](dispatch-latency-spike.md) —
  alert runbooks that call this procedure.
- [`oncall-paging.md`](oncall-paging.md) — escalation if the
  rollback itself fails.
- [`UPGRADE.md`](../UPGRADE.md) — the forward-rolling counterpart;
  documents the version-skew guarantees that constrain how far back
  a rollback can go.
