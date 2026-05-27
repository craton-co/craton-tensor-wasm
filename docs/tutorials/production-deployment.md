<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Production deployment tutorial

This is the v0.3 *"Production deployment tutorial end-to-end"* artifact
from [`docs/PATH-TO-V1.md`](../PATH-TO-V1.md) (Documentation workstream).
It walks a competent SRE who has never touched Craton TensorWasm from a
*fresh* Kubernetes cluster to a *production-ready* deployment: GPU
scheduling, mTLS at the ingress, Prometheus scrape with burn-rate
alerts, Grafana dashboard, audit log to durable storage, a deployed
function, and a smoke test against the live `/metrics` surface.

The tutorial is deliberately thin on duplication. Every step calls into
an existing document — the Helm chart README, the CUDA setup guide, the
SLO definitions, the mTLS deployment guide, the backup playbook — and
asks you to read it rather than re-stating its content here. A delta in
one of those documents is then automatically a delta for this tutorial;
nothing drifts.

Read end-to-end before you start. The first time is roughly **2-3 hours
of wall-clock work** on a cluster you already operate; subsequent
deployments collapse to ~30 minutes once you have the values file and
the secrets in place.

## Contents

1. [Audience and prerequisites](#1-audience-and-prerequisites)
2. [Architecture](#2-architecture)
3. [Step 1: Prepare the cluster](#3-step-1-prepare-the-cluster)
4. [Step 2: Generate auth tokens](#4-step-2-generate-auth-tokens)
5. [Step 3: Author `values.yaml`](#5-step-3-author-valuesyaml)
6. [Step 4: Install the chart](#6-step-4-install-the-chart)
7. [Step 5: Configure mTLS at the ingress](#7-step-5-configure-mtls-at-the-ingress)
8. [Step 6: Import the Grafana dashboard](#8-step-6-import-the-grafana-dashboard)
9. [Step 7: Apply burn-rate alert rules](#9-step-7-apply-burn-rate-alert-rules)
10. [Step 8: Deploy your first function](#10-step-8-deploy-your-first-function)
11. [Step 9: Smoke test with `tensor-wasm observe`](#11-step-9-smoke-test-with-tensor-wasm-observe)
12. [Step 10: Set up backups](#12-step-10-set-up-backups)
13. [Step 11: Read the runbook before you need it](#13-step-11-read-the-runbook-before-you-need-it)
14. [Common pitfalls](#14-common-pitfalls)
15. [Where to go next](#15-where-to-go-next)
16. [Related](#16-related)

---

## 1. Audience and prerequisites

You are a **competent SRE** who is *new to TensorWasm specifically*.
You know Kubernetes, Helm, Prometheus, and cert-manager. You can read a
YAML diff and a PromQL expression. You have operated at least one piece
of GPU-adjacent infrastructure before.

If any of the above is not yet true, read
[`docs/GETTING-STARTED.md`](../GETTING-STARTED.md) first — it covers
the laptop-scale loop and the conceptual model.

### 1.1 What you need running before you start

| Requirement | How to confirm |
|---|---|
| Kubernetes 1.23+ cluster, kubectl context configured | `kubectl version --short` returns server >= 1.23 |
| At least one GPU-enabled node with `nvidia.com/gpu` capacity | [Step 1.1](#31-confirm-gpu-nodes); see [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) "SM-level compatibility matrix" |
| `nvidia-device-plugin` DaemonSet healthy in `kube-system` | [Step 1.2](#32-confirm-the-nvidia-device-plugin); [`deploy/k8s/README.md`](../../deploy/k8s/README.md) "GPU-node prerequisite checklist" |
| Prometheus Operator with `monitoring.coreos.com/v1` CRDs | [Step 1.4](#34-confirm-prometheus-operator); [`docs/dashboards/README.md`](../dashboards/README.md) "Prometheus scrape interval" |
| Grafana with a Prometheus datasource | [Step 6](#8-step-6-import-the-grafana-dashboard); [`docs/dashboards/README.md`](../dashboards/README.md) "How to import" |
| cert-manager with a cluster `Issuer` / `ClusterIssuer` | [Step 1.3](#33-confirm-cert-manager); [`docs/deployment/mtls.md`](../deployment/mtls.md) §5.3 |
| An ingress controller (nginx, Traefik, Envoy/Istio, ALB) | `kubectl get ingressclass` returns >= 1 row; [`docs/deployment/mtls.md`](../deployment/mtls.md) §4 |
| `helm` 3.8+, `kubectl`, `openssl`, `jq`, `curl`, `base64` on your workstation | `helm version --short`; shell-builtins on Linux/macOS; Git Bash or WSL on Windows |

### 1.2 What this tutorial does NOT cover

Cluster provisioning (you bring the cluster); external secret
management (substitute your ESO / Vault / SOPS pattern at
[Step 2](#4-step-2-generate-auth-tokens)); multi-cluster federation
(out of scope per
[`docs/PATH-TO-V1.md`](../PATH-TO-V1.md#anti-goals--what-v10-does-not-promise));
host-level CUDA install (read [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md)
end-to-end for bare-metal deployments); workload-specific tuning
(values in [Step 3](#5-step-3-author-valuesyaml) are defaults — re-measure
per [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md) §7).

---

## 2. Architecture

End-state: one TensorWasm replica behind an mTLS-terminating ingress,
scraped by Prometheus, fronted by Grafana, spilling audit records to a
durable PVC.

```
   external caller --HTTPS+mTLS--> Ingress controller
                                   (nginx / Envoy / ALB; cert-manager Secret;
                                    validates client cert; forwards plaintext
                                    + X-Forwarded-Client-Cert header)
                                          |
                                          v
                                   Service: tensor-wasm (ClusterIP :8080)
                                          |
                                          v
                                   Pod: tensor-wasm (Deployment replicas=1)
                                     tensor-wasm-api: axum
                                       + bearer_auth + tenant_scope
                                       + rate_limit + audit middleware
                                       -> file:/var/lib/tensor-wasm/audit.log
                                          |                  |
                                          v                  v
                                   nvidia.com/gpu (1)   PVC (50Gi, CSI w/
                                          |             VolumeSnapshot)
                                          v
                                   GPU node (driver + device-plugin
                                             + nvidia-container-toolkit)

       Prometheus (Operator)  <--scrape GET /metrics (15 s)-- Pod
        + PrometheusRule (fast / slow / very-slow burn alerts)
                        |
                        v
       Grafana: tensor-wasm-overview.json
        (SLO summary + HTTP / tenant / snapshot / JIT / back-pressure rows)

       Pod stdout (audit mirror) --> Log shipper (Fluent Bit / Vector / Loki)
                                     -> SIEM / object store / BigQuery
```

Three properties to internalize: (1) **the pod is stateless apart
from the PVC** — state that must survive a pod cycle lives on the
PVC; rate-limit bucket, function registry, and in-flight jobs are
per-process by design per
[`docs/BACKUP-RESTORE.md`](../BACKUP-RESTORE.md) §3. (2) **mTLS
terminates at the ingress, not the pod** — Architecture B from
[`docs/deployment/mtls.md`](../deployment/mtls.md) §4, because
Architecture A is not implemented in v0.4. (3) **Bearer auth still
runs after mTLS** — mTLS authenticates the caller; the bearer selects
the tenant scope per
[`docs/deployment/mtls.md`](../deployment/mtls.md) §2. Do not skip
[Step 2](#4-step-2-generate-auth-tokens).

---

## 3. Step 1: Prepare the cluster

All commands below are read-only.

### 3.1 Confirm GPU nodes

```bash
kubectl get nodes -o json \
  | jq -r '.items[] | select(.metadata.labels["nvidia.com/gpu.product"]) | "\(.metadata.name)\t\(.metadata.labels["nvidia.com/gpu.product"])"'
```

You want at least one row. If the filter prints nothing, the node has
no GPU or the GPU Operator / device plugin has not labelled it. Confirm
with `kubectl describe node <name>` looking for `nvidia.com/gpu` under
`Capacity:`/`Allocatable:`. Use [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md)
"SM-level compatibility matrix" to map your GPU SKU to a `CUDA_ARCH`
value (L4 = `sm_89`, A100 = `sm_80`, H100 = `sm_90`).

If the label is absent altogether, install the NVIDIA GPU Operator or
`nvidia/gpu-feature-discovery` chart before continuing.

### 3.2 Confirm the nvidia-device-plugin

```bash
kubectl -n kube-system get ds nvidia-device-plugin-daemonset \
  -o jsonpath='{.status.numberReady}/{.status.desiredNumberScheduled}'; echo
```

Should be `N/N`. If `0/N` or `not found`, the device plugin is missing
or wedged — revisit
[`deploy/k8s/README.md`](../../deploy/k8s/README.md) "GPU-node
prerequisite checklist". The DaemonSet name varies (GPU Operator vs
standalone chart); adjust accordingly.

### 3.3 Confirm cert-manager

```bash
kubectl get crd certificates.cert-manager.io issuers.cert-manager.io clusterissuers.cert-manager.io
kubectl -n cert-manager get pods
```

All CRDs should exist and the pods `Running 1/1`. cert-manager renews
the ingress server cert; the mTLS doc shows a worked
`Issuer` + `Certificate` pair at
[`docs/deployment/mtls.md`](../deployment/mtls.md) §5.3.

### 3.4 Confirm Prometheus operator

```bash
kubectl get crd servicemonitors.monitoring.coreos.com prometheusrules.monitoring.coreos.com
kubectl get prometheus -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\t"}{.spec.serviceMonitorSelector}{"\n"}{end}'
```

Note the Prometheus CR's `serviceMonitorSelector` labels — you need
them for the chart's `prometheus.additionalLabels` in
[Step 3](#5-step-3-author-valuesyaml) and the PrometheusRule's
`release:` label in [Step 7](#9-step-7-apply-burn-rate-alert-rules).
kube-prometheus-stack default is `release: prometheus`.

### 3.5 Host-level CUDA (only outside Kubernetes)

For bare-metal hosts running TensorWasm under systemd, read
[`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) end-to-end and run its §9
verification script. Inside Kubernetes you only need the NVIDIA driver
and `nvidia-container-toolkit` on the node; the container image ships
the toolkit it needs.

---

## 4. Step 2: Generate auth tokens

The token grammar is documented in
[`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md)
"Per-tenant scopes (`TENSOR_WASM_API_TOKENS` entry forms)". Read that
section once before continuing.

### 4.1 Decide your token-scope strategy

Three patterns cover most deployments:

| Pattern | Token shape | Use when |
|---|---|---|
| Admin + per-service scoped | `admin:tenant=*`, `svc-N:tenant=N` | Common production case; least-privilege per service. |
| Per-environment scoped | `prod:tenant=1,2,3` | Small fleet, no per-tenant service identity. |
| Wildcard-only | `infra:tenant=*` | Dev / staging only. Not for production. |

The audit log
([`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §1) hashes the bearer into
`token_id` and records `actor.scope.kind`. A `wildcard` actor doing a
high-blast-radius action is a finding; a `tenant_set` actor is
expected. That is the operational case for scoped tokens.

### 4.2 Generate the token strings

`openssl rand -hex 32` is the recommended generator (256 bits, URL-safe):

```bash
ADMIN_TOKEN=$(openssl rand -hex 32)
SVC7_TOKEN=$(openssl rand -hex 32)
SVC8_TOKEN=$(openssl rand -hex 32)
TENSOR_WASM_API_TOKENS="${ADMIN_TOKEN}:tenant=*,${SVC7_TOKEN}:tenant=7,${SVC8_TOKEN}:tenant=8"
```

Store each token in your secret manager **before** continuing — the
next command writes the assembled value into a k8s Secret and you
cannot recover the plaintext from there. Treat each like a database
password ([`deploy/k8s/README.md`](../../deploy/k8s/README.md)
"Provision the bearer-token secret").

### 4.3 Create the Kubernetes Secret

```bash
kubectl create namespace tensor-wasm
kubectl -n tensor-wasm create secret generic tensor-wasm-tokens \
  --from-literal=TENSOR_WASM_API_TOKENS="${TENSOR_WASM_API_TOKENS}"
```

Rotation is documented at
[`deploy/k8s/README.md`](../../deploy/k8s/README.md) "Provision the
bearer-token secret" — delete the Secret, recreate, re-roll the
Deployment. There is no graceful overlap window in v0.4; old and new
tokens must coexist in the env value during hand-off, then the old
entry is removed in a second roll. Document this today.

---

## 5. Step 3: Author `values.yaml`

The chart's value reference lives at
[`deploy/helm/tensor-wasm/README.md`](../../deploy/helm/tensor-wasm/README.md)
"Values reference" with full defaults in
[`deploy/helm/tensor-wasm/values.yaml`](../../deploy/helm/tensor-wasm/values.yaml).
This step only documents the production overrides.

Create `tensor-wasm.values.yaml` next to your other infrastructure
code. Commit it to git; the secret stays out (`auth.existingSecret`
references the Secret from [Step 2](#4-step-2-generate-auth-tokens)).

```yaml
# tensor-wasm.values.yaml -- production overrides for the v0.1.0
# reference workload on an NVIDIA L4 (sm_89) GPU node.

image:
  # Placeholder registry per deploy/helm/tensor-wasm/README.md "Image
  # registry is a placeholder" -- replace with your built image until
  # ghcr.io/craton-co is provisioned.
  repository: my-registry.example.com/tensor-wasm
  tag: "0.1.0"
  pullPolicy: IfNotPresent

replicaCount: 1
strategy: { type: Recreate }

service:
  type: ClusterIP
  port: 8080

# Architecture B (mTLS at the ingress) per docs/deployment/mtls.md sec 4.
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    # Match the API's 64 MiB body cap; see API.md "Request limits".
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
    # mTLS. The CA Secret is created in Step 5.
    nginx.ingress.kubernetes.io/auth-tls-verify-client: "on"
    nginx.ingress.kubernetes.io/auth-tls-secret: "tensor-wasm/tensor-wasm-client-ca"
    nginx.ingress.kubernetes.io/auth-tls-pass-certificate-to-upstream: "true"
    nginx.ingress.kubernetes.io/auth-tls-verify-depth: "2"
  hosts:
    - host: tensor-wasm.example.com
      paths:
        - { path: /, pathType: Prefix }
  tls:
    - hosts: [tensor-wasm.example.com]
      secretName: tensor-wasm-server-tls

# GPU scheduling per deploy/k8s/README.md "GPU-node prerequisite checklist".
gpu:
  enabled: true
  count: 1
  nodeSelector:
    nvidia.com/gpu.present: "true"
  tolerations:
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  runtimeClassName: nvidia

# Match the node SM level. See docs/CUDA-SETUP.md "SM-level compatibility matrix".
cuda: { arch: "sm_89" }

# Production sizing for the v0.1.0 reference workload. Re-measure per
# docs/DEPLOYMENT.md sec 7 "Multi-tenant capacity planning".
resources:
  requests: { cpu: "500m", memory: "2Gi" }
  limits:   { cpu: "4",    memory: "8Gi" }

# Reference the Secret from Step 2; chart-rendered Secret is suppressed
# and the weak `auth.tokens` default is ignored.
auth:
  existingSecret: "tensor-wasm-tokens"
  requireTenant: true

# Per-bearer-token rate limit. Tune for traffic; see API.md "Per-token
# rate limiting" for token-bucket semantics and 429 behaviour.
rateLimit: { qps: 100, burst: 200 }

# Prometheus Operator integration. The `release: prometheus` label must
# match your Prometheus CR's serviceMonitorSelector (Step 1.4).
prometheus:
  enabled: true
  interval: 15s
  scrapeTimeout: 10s
  additionalLabels: { release: prometheus }
  honorLabels: true

# /var/lib/tensor-wasm holds audit log + snapshots per
# docs/BACKUP-RESTORE.md sec 2.1.
persistence:
  enabled: true
  size: 50Gi
  storageClass: ""        # cluster default; pin to a CSI w/ VolumeSnapshot
  accessModes: [ReadWriteOnce]

# Audit-log destination. Env grammar: docs/AUDIT-LOG.md sec 3.
extraEnv:
  - name: TENSOR_WASM_API_AUDIT_LOG
    value: "file:/var/lib/tensor-wasm/audit.log"

log: { level: "info" }

# OTLP push exporter; empty disables.
otlp:
  endpoint: "http://otel-collector.observability.svc.cluster.local:4317"
```

Field notes not in the chart README: `auth.requireTenant: true`
surfaces missing-tenant bugs at the gateway rather than silently
landing on tenant 0. `rateLimit.qps: 100`, `burst: 200` is sensible for
internal multi-tenant; re-tune after a week of
`tensor_wasm_http_requests_total{status="429"}` observations.
`persistence.size: 50Gi` holds ~12 days of audit log at 100 qps per
[`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §5.3 plus a small snapshot
inventory; scale linearly with traffic. `extraEnv` is the only way to
set the audit sink in v0.4; the chart does not yet expose a
first-class `audit:` block.

---

## 6. Step 4: Install the chart

With the namespace pre-created in [Step 2.3](#43-create-the-kubernetes-secret):

```bash
helm install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm -f tensor-wasm.values.yaml
```

Watch the rollout:

```bash
kubectl -n tensor-wasm rollout status deployment/tensor-wasm --timeout=5m
kubectl -n tensor-wasm get pods -l app.kubernetes.io/name=tensor-wasm
kubectl -n tensor-wasm logs deploy/tensor-wasm --tail=50
```

You want `1/1 Ready` and a `tensor-wasm-api listening on 0.0.0.0:8080`
log line. Failure modes (see [Common pitfalls](#14-common-pitfalls)):
`Pending` with `Insufficient nvidia.com/gpu` -> device plugin issue;
crash-loop with `secret "tensor-wasm-tokens" not found` -> Secret in
the wrong namespace.

Probe the pod via port-forward before exposing the ingress:

```bash
kubectl -n tensor-wasm port-forward svc/tensor-wasm 8080:8080 &
PF=$!; sleep 2
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/metrics | head -20
kill $PF
```

---

## 7. Step 5: Configure mTLS at the ingress

We use Architecture B from
[`docs/deployment/mtls.md`](../deployment/mtls.md) §4 — read that
section before continuing. The chart's `ingress` block from
[Step 3](#5-step-3-author-valuesyaml) already references the two
Secrets this step provisions: `tensor-wasm-server-tls` (cert-manager)
and `tensor-wasm-client-ca` (manual).

### 7.1 Issue the server cert via cert-manager

```yaml
# server-cert.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: tensor-wasm-server
  namespace: tensor-wasm
spec:
  secretName: tensor-wasm-server-tls
  duration: 2160h    # 90 days; cert-manager renews at 2/3 lifetime
  renewBefore: 720h
  commonName: tensor-wasm.example.com
  dnsNames: [tensor-wasm.example.com]
  issuerRef: { name: letsencrypt-prod, kind: ClusterIssuer }
```

```bash
kubectl apply -f server-cert.yaml
kubectl -n tensor-wasm get certificate tensor-wasm-server -w
```

If stuck `READY=False`, `kubectl describe certificate` and the
underlying `CertificateRequest` reveal the cause; the most common is
an unreachable HTTP-01 challenge endpoint.

### 7.2 Provision the client CA bundle

For internal-only mTLS, run a private CA — Vault PKI, an internal
cert-manager `CA`-kind Issuer, or the offline OpenSSL pipeline in
[`docs/deployment/mtls.md`](../deployment/mtls.md) §3.3. Wrap the PEM
bundle in a Secret the ingress reads:

```bash
kubectl -n tensor-wasm create secret generic tensor-wasm-client-ca \
  --from-file=ca.crt=ca.pem
```

The nginx-ingress annotations expect the data key `ca.crt`; other
ingresses expect different keys
([`docs/deployment/mtls.md`](../deployment/mtls.md) §4 covers Envoy
and Caddy).

### 7.3 Verify

```bash
HOST=tensor-wasm.example.com
# Should fail w/ SSL_ERROR_BAD_CERT_REQUIRED (mTLS demands a cert).
curl -v "https://${HOST}/healthz" 2>&1 | tail -20
# Should succeed with a CA-signed client cert.
curl --cacert /path/to/server-trust.pem \
     --cert  /path/to/client.pem --key /path/to/client.key \
     "https://${HOST}/healthz"
```

The XFCC header the ingress forwards lands as `client_cert_subject` in
the audit log ([`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §6); confirm in
[Step 9](#11-step-9-smoke-test-with-tensor-wasm-observe).

### 7.4 XFCC trust caveat

The gateway does **not** today validate that XFCC came from the ingress
([`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §6.2;
[`docs/deployment/mtls.md`](../deployment/mtls.md) §7.4). Mitigation is
the topology you have just built: `ClusterIP` Service, so the only
caller that reaches :8080 is the ingress pod. Leave it that way until
`TENSOR_WASM_API_TRUSTED_PROXY_CIDRS` lands in v0.5.

---

## 8. Step 6: Import the Grafana dashboard

Read [`docs/dashboards/README.md`](../dashboards/README.md). The "How
to import" section is canonical; below is the Kubernetes variant.

### 8.1 Sidecar discovery (kube-prometheus-stack)

If Grafana runs with a `grafana-sc-dashboards` sidecar, drop the JSON
into a labelled ConfigMap:

```bash
kubectl -n monitoring create configmap tensor-wasm-overview \
  --from-file=tensor-wasm-overview.json=./docs/dashboards/tensor-wasm-overview.json
kubectl -n monitoring label configmap tensor-wasm-overview grafana_dashboard=1
```

The sidecar imports within ~30 s. If your Grafana is in a different
namespace or uses a different label, adjust.

### 8.2 Manual import

Follow [`docs/dashboards/README.md`](../dashboards/README.md) "How to
import" verbatim.

### 8.3 Confirm panels render

The top-row SLI stat panels
([`docs/dashboards/README.md`](../dashboards/README.md) "Panel
inventory") should show numeric values within ~30 s of the
ServiceMonitor activating. "No data" on HTTP rows usually means a
`serviceMonitorSelector` mismatch ([Common pitfalls](#14-common-pitfalls)).

Several panels (snapshot histograms, JIT cache, back-pressure permits)
*intentionally* render "No data" today — the metrics are W3+
follow-ups tracked in
[`docs/dashboards/README.md`](../dashboards/README.md) "Metric inventory"
-> "TODO". They light up when the missing metric ships, no dashboard
edit needed.

---

## 9. Step 7: Apply burn-rate alert rules

The SLO targets and the burn-rate alert PromQL are defined in
[`docs/SLO.md`](../SLO.md) §3 and §5. Read those before applying;
runbook links live in [`docs/SLO.md`](../SLO.md) §7.

The PrometheusRule below is a thin wrapper that copies the PromQL from
[`docs/SLO.md`](../SLO.md) verbatim so any SLO tightening there is a
single-source edit and this CR follows.

```yaml
# tensor-wasm-rules.yaml -- thresholds and PromQL track docs/SLO.md.
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: tensor-wasm-slo
  namespace: tensor-wasm
  labels: { release: prometheus }   # match your Prometheus CR ruleSelector
spec:
  groups:
    - name: tensor-wasm.availability
      interval: 30s
      rules:
        - alert: TensorWasmAvailabilityFastBurn   # docs/SLO.md sec 5.1
          expr: |
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[1h])) / sum(rate(tensor_wasm_http_requests_total[1h])) > (14.4 * 0.005))
            and
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[5m])) / sum(rate(tensor_wasm_http_requests_total[5m])) > (14.4 * 0.005))
          for: 2m
          labels: { severity: page }
          annotations:
            summary: "TensorWasm error budget burning at 14.4x"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/availability-fast-burn.md"
        - alert: TensorWasmAvailabilitySlowBurn   # docs/SLO.md sec 5.2
          expr: |
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h])) / sum(rate(tensor_wasm_http_requests_total[6h])) > (6 * 0.005))
            and
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[30m])) / sum(rate(tensor_wasm_http_requests_total[30m])) > (6 * 0.005))
          for: 15m
          labels: { severity: page }
          annotations:
            summary: "TensorWasm error budget burning at 6x"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/availability-slow-burn.md"
        - alert: TensorWasmAvailabilityVerySlowBurn   # docs/SLO.md sec 5.3
          expr: |
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[3d])) / sum(rate(tensor_wasm_http_requests_total[3d])) > (1 * 0.005))
            and
            (sum(rate(tensor_wasm_http_requests_total{status=~"5.."}[6h])) / sum(rate(tensor_wasm_http_requests_total[6h])) > (1 * 0.005))
          for: 1h
          labels: { severity: ticket }
          annotations:
            summary: "TensorWasm error budget being consumed"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/availability-very-slow-burn.md"
    - name: tensor-wasm.latency
      interval: 30s
      rules:
        - alert: TensorWasmInvokeLatencySpike   # docs/SLO.md sec 5.4
          expr: |
            histogram_quantile(0.95, sum by (le) (rate(tensor_wasm_http_request_duration_seconds_bucket{route="/functions/:id/invoke",method="POST"}[5m]))) > 0.5
            and
            histogram_quantile(0.95, sum by (le) (rate(tensor_wasm_http_request_duration_seconds_bucket{route="/functions/:id/invoke",method="POST"}[1h]))) > 0.5
          for: 5m
          labels: { severity: page }
          annotations:
            summary: "TensorWasm /invoke P95 > 500 ms"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/invoke-latency-spike.md"
        - alert: TensorWasmHealthzSlow   # docs/SLO.md sec 5.4
          expr: |
            histogram_quantile(0.95, sum by (le) (rate(tensor_wasm_http_request_duration_seconds_bucket{route="/healthz",method="GET"}[30m]))) > 0.01
          for: 30m
          labels: { severity: ticket }
          annotations:
            summary: "TensorWasm /healthz P95 > 10 ms"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/healthz-slow.md"
        - alert: TensorWasmDispatchLatencySpike   # docs/SLO.md sec 5.5 (host-only)
          expr: |
            histogram_quantile(0.95, sum by (le) (rate(tensor_wasm_kernel_latency_seconds_bucket[5m]))) > 0.00005
            and
            histogram_quantile(0.95, sum by (le) (rate(tensor_wasm_kernel_latency_seconds_bucket[1h]))) > 0.00005
          for: 5m
          labels: { severity: page }
          annotations:
            summary: "TensorWasm dispatch P95 > 50 us"
            runbook_url: "https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/runbooks/dispatch-latency-spike.md"
```

```bash
kubectl apply -f tensor-wasm-rules.yaml
kubectl -n tensor-wasm get prometheusrule tensor-wasm-slo
```

Wait one evaluation interval (30 s) then confirm Prometheus loaded the
rules:

```bash
kubectl -n monitoring port-forward svc/prometheus-operated 9090:9090 &
PF=$!; sleep 2
curl -s http://localhost:9090/api/v1/rules \
  | jq '.data.groups[] | select(.name | startswith("tensor-wasm")) | .name'
kill $PF
```

The dispatch alert is calibrated for host-only deployments
([`docs/SLO.md`](../SLO.md) §5.5); on GPU-host the threshold widens
once the v0.4 measured CUDA-host dispatch SLO lands.

---

## 10. Step 8: Deploy your first function

End-to-end: upload a Wasm module, invoke it, watch the audit log and
metrics. The full HTTP surface is in
[`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md).

### 10.1 Build a Wasm module

The `hello.wasm` from
[`docs/GETTING-STARTED.md`](../GETTING-STARTED.md) §2 works; the
shortest path uses `wabt`:

```bash
cat > hello.wat <<'EOF'
(module (func (export "_start")))
EOF
wat2wasm hello.wat -o hello.wasm
```

### 10.2 Deploy

```bash
HOST=tensor-wasm.example.com
TOKEN="${SVC7_TOKEN}"
WASM_B64=$(base64 -w0 < hello.wasm)

DEPLOY_RESP=$(curl -sf \
  --cacert /path/to/server-trust.pem \
  --cert   /path/to/client.pem --key /path/to/client.key \
  -X POST "https://${HOST}/functions" \
  -H "authorization: Bearer ${TOKEN}" \
  -H "x-tensor-wasm-tenant: 7" \
  -H "content-type: application/json" \
  -d "{\"name\":\"hello\",\"wasm_b64\":\"${WASM_B64}\"}")

FUNCTION_ID=$(echo "$DEPLOY_RESP" | jq -r '.id')
echo "deployed: ${FUNCTION_ID}"
```

Request shape is canonical in
[`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md)
`POST /functions`.

### 10.3 Invoke

```bash
curl -sf \
  --cacert /path/to/server-trust.pem \
  --cert   /path/to/client.pem --key /path/to/client.key \
  -X POST "https://${HOST}/functions/${FUNCTION_ID}/invoke" \
  -H "authorization: Bearer ${TOKEN}" \
  -H "x-tensor-wasm-tenant: 7" \
  -H "content-type: application/json" \
  -d '{}'
```

Expected: `{"result":"ok","function_id":"<FUNCTION_ID>"}`.

### 10.4 Confirm the audit log

```bash
kubectl -n tensor-wasm exec deploy/tensor-wasm -- tail -1 /var/lib/tensor-wasm/audit.log | jq
```

The record should match the schema in
[`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §1, with
`action: "invoke_function"`, `outcome.status_code: 200`,
`resource.tenant_id: 7`, and a populated `client_cert_subject` if
mTLS XFCC reached the gateway. A `null` `client_cert_subject` despite
working mTLS means the ingress is not forwarding XFCC — re-check
`auth-tls-pass-certificate-to-upstream: "true"` per
[`docs/deployment/mtls.md`](../deployment/mtls.md) §4.1.

### 10.5 Confirm the metric counter

```bash
kubectl -n tensor-wasm port-forward svc/tensor-wasm 8080:8080 &
PF=$!; sleep 2
curl -s http://localhost:8080/metrics \
  | grep -E 'tensor_wasm_http_requests_total\{.*invoke.*200' | head
kill $PF
```

Expect a non-zero counter. The Availability and Invoke-latency stat
panels from [Step 6](#8-step-6-import-the-grafana-dashboard) should
reflect it within one scrape (15 s).

---

## 11. Step 9: Smoke test with `tensor-wasm observe`

The W1.5 `tensor-wasm observe` CLI is the operator's one-screen status
board — wraps `GET /metrics` and `GET /healthz` and prints a live
table. Perfect for a deploy window.

```bash
# Build from source if not already (no published binary in v0.1.0):
cargo build --release -p tensor-wasm-cli

# Port-forward avoids needing a client cert for every poll.
kubectl -n tensor-wasm port-forward svc/tensor-wasm 8080:8080 &
PF=$!
tensor-wasm observe --server http://localhost:8080
```

Leave it running. In another shell, fire 50 invokes through the
ingress (curl loop from [Step 8.3](#103-invoke)) and watch
`http_requests_total`, `kernel_dispatches_total`, and `active_instances`
tick. A *flat* counter under live load is the "passes /healthz but
fails under load" regression
[`docs/UPGRADE.md`](../UPGRADE.md) §6 calls out.

`observe --once` prints a single snapshot and exits, suitable for CI.

In parallel, open the Grafana dashboard from
[Step 6](#8-step-6-import-the-grafana-dashboard). The SLI summary row
should report `availability_http` ~100% and `invoke P95` well under
100 ms on the host-only path. Unhealthy on a fresh deployment -> the
linked runbook for the failing SLI.

---

## 12. Step 10: Set up backups

[`docs/BACKUP-RESTORE.md`](../BACKUP-RESTORE.md) is the source of
truth. Action items: (1) pick a strategy from §4 — for the CSI-backed
PVC built here, §4.1 ("PVC volume snapshots") is the most direct, with
§4.2 (`restic`) or §4.3 (object-store sync) on top for off-cluster
durability; (2) schedule on the §5 cadence; (3) run the §7 validation
*today* before the first DR drill (§7.1 snapshot integrity and §7.2
audit-log JSONL round-trip are the fastest); (4) confirm your secret
manager backs up `tensor-wasm-tokens`, `tensor-wasm-server-tls`, and
`tensor-wasm-client-ca` — these are *not* in the PVC, so PVC snapshots
do not cover them ([`docs/BACKUP-RESTORE.md`](../BACKUP-RESTORE.md)
§2.4). Backups you have never restored are not backups; plan the first
end-to-end drill before the deployment ships.

---

## 13. Step 11: Read the runbook before you need it

Bookmark these three before calling the deployment production:
[`rollback.md`](../runbooks/rollback.md) (revert in a hurry; single
source of truth referenced from [`docs/UPGRADE.md`](../UPGRADE.md) §8),
[`availability-fast-burn.md`](../runbooks/availability-fast-burn.md)
(response to the 14.4x burn-rate alert), and
[`disaster-recovery.md`](../runbooks/disaster-recovery.md) (host-loss
playbook; consumes the backups from [Step 10](#12-step-10-set-up-backups)).

Then do a *planned* upgrade rehearsal: bump `image.tag` to a synthetic
version (a re-tag of the same binary works); walk
[`docs/UPGRADE.md`](../UPGRADE.md) §2 (pre-flight), §4.1 (Helm path),
§6 (post-upgrade verification); roll back per
[`runbooks/rollback.md`](../runbooks/rollback.md) §B; time both passes
against [`docs/UPGRADE.md`](../UPGRADE.md) §9. A team that has
rehearsed gets it right at 03:00.

---

## 14. Common pitfalls

| Symptom | Likely cause | Where to look |
|---|---|---|
| Pod `Pending` with `Insufficient nvidia.com/gpu` | Device plugin not advertising capacity | [Step 1.2](#32-confirm-the-nvidia-device-plugin); [`deploy/k8s/README.md`](../../deploy/k8s/README.md) "GPU-node prerequisite checklist" |
| Pod `Pending` with `node(s) had taints` | GPU nodes have a different taint than `nvidia.com/gpu=true:NoSchedule` | Adjust `gpu.tolerations`; [`deploy/helm/tensor-wasm/values.yaml`](../../deploy/helm/tensor-wasm/values.yaml) |
| Crash-loop `secret "tensor-wasm-tokens" not found` | Secret in wrong namespace or `auth.existingSecret` typo | [Step 2.3](#43-create-the-kubernetes-secret) |
| All requests return 401 | Token format wrong — missing `:tenant=...` or wrong bearer | [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md) "Per-tenant scopes" |
| All requests return 403 `tenant_scope_denied` | Token scoped to wrong tenants for the `x-tensor-wasm-tenant` value | [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md) "Tenant scoping" |
| All requests return 400 `missing_tenant` | `auth.requireTenant: true` but caller did not send the header | [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md) "Tenant scoping" |
| `Certificate` stuck `READY=False` | cert-manager cannot complete ACME challenge | `kubectl describe certificate/certificaterequest`; [`docs/deployment/mtls.md`](../deployment/mtls.md) §7.5 |
| curl fails with `SSL_ERROR_BAD_CERT_REQUIRED` | mTLS demanded a client cert, you did not present one — correct behaviour | [`docs/deployment/mtls.md`](../deployment/mtls.md) §7.2 |
| Dashboard HTTP rows show "No data" | ServiceMonitor `release:` label does not match Prometheus CR selector | [Step 1.4](#34-confirm-prometheus-operator); adjust `prometheus.additionalLabels` |
| Dashboard snapshot / JIT / back-pressure rows "No data" | Metrics not yet emitted; expected v0.4 state | [`docs/dashboards/README.md`](../dashboards/README.md) "Metric inventory" -> "TODO" |
| Burn-rate alerts never fire under induced errors | PrometheusRule `release:` label does not match `ruleSelector` | [Step 7](#9-step-7-apply-burn-rate-alert-rules); adjust `metadata.labels` |
| `client_cert_subject` `null` despite mTLS working | nginx-ingress missing `auth-tls-pass-certificate-to-upstream` | [`docs/deployment/mtls.md`](../deployment/mtls.md) §4 / §7.4 |
| `actor.scope.kind` is `dev` in audit log | `TENSOR_WASM_API_TOKENS` env not set — Secret mistyped | [`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §1; `kubectl describe pod` |
| Crash with `nvcc fatal : Unsupported gpu architecture` | `CUDA_ARCH` does not match node SM level | [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) "SM-level compatibility matrix" |
| `tensor-wasm observe` connects but counters flat | Executor wedged on a stuck dispatch | [`docs/UPGRADE.md`](../UPGRADE.md) §6; dispatch-latency runbook |
| Audit log fills the PVC in days | Traffic exceeds the 100-qps sizing or rotation not running | [`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §5.2 / §5.3 |

If the symptom is not above, triage in order:

1. `kubectl describe pod` and `kubectl logs deploy/tensor-wasm`.
2. The runbook for any alert that fired
   ([`docs/SLO.md`](../SLO.md) §7).
3. [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) "Troubleshooting" for any
   GPU-flavoured failure.

---

## 15. Where to go next

1. **Make the upgrade muscle real.** Read
   [`docs/UPGRADE.md`](../UPGRADE.md) end to end and rehearse a
   blue/green per §3.2. Pair with
   [`docs/MIGRATION-v0-to-v1.md`](../MIGRATION-v0-to-v1.md).
2. **Wire runbooks to paging.** Every alert in
   [Step 7](#9-step-7-apply-burn-rate-alert-rules) carries a
   `runbook_url`; confirm Alertmanager surfaces it and fill in
   [`docs/runbooks/oncall-paging.md`](../runbooks/oncall-paging.md).
3. **Plan for v0.4 audit-log limitations.** Review
   [`docs/AUDIT-LOG.md`](../AUDIT-LOG.md) §8 and plan to tighten when
   v0.5 ships `TENSOR_WASM_API_TRUSTED_PROXY_CIDRS`.
4. **Tighten SLOs as your data lands.**
   [`docs/SLO.md`](../SLO.md) §3 marks every CUDA-host target "TBD" or
   "modeled"; replace with measurements after a month of traffic via
   the §9 RFC process.
5. **Walk the v1.0 path with your team** via
   [`docs/PATH-TO-V1.md`](../PATH-TO-V1.md).

---

## 16. Related

- [`docs/PATH-TO-V1.md`](../PATH-TO-V1.md) — v0.3 documentation
  workstream this tutorial satisfies.
- [`docs/GETTING-STARTED.md`](../GETTING-STARTED.md) — laptop-scale
  conceptual onboarding before this tutorial.
- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md) — production topology,
  sizing, capacity planning.
- [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) — GPU node prerequisites,
  SM-level matrix, troubleshooting.
- [`deploy/helm/tensor-wasm/README.md`](../../deploy/helm/tensor-wasm/README.md),
  [`values.yaml`](../../deploy/helm/tensor-wasm/values.yaml), and
  [`deploy/k8s/README.md`](../../deploy/k8s/README.md) — chart and
  plain-YAML alternatives.
- [`docs/SLO.md`](../SLO.md) and
  [`docs/dashboards/README.md`](../dashboards/README.md) — SLO
  definitions and dashboard import.
- [`docs/AUDIT-LOG.md`](../AUDIT-LOG.md),
  [`docs/deployment/mtls.md`](../deployment/mtls.md) — audit log and
  mTLS deployment guides.
- [`docs/BACKUP-RESTORE.md`](../BACKUP-RESTORE.md),
  [`docs/UPGRADE.md`](../UPGRADE.md),
  [`docs/runbooks/`](../runbooks/) — DR, upgrade, alert playbooks.
- [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md)
  — HTTP surface, token grammar, request limits.
- [`docs/OBSERVABILITY.md`](../OBSERVABILITY.md) — tracing schema and
  OTLP setup.

---

_Status: v0.3 release. End-to-end runnable against the v0.1.0 binary
plus the W2.7 Helm chart; every `TODO (v0.4)` / `TODO (v0.5)` marker in
a referenced doc is reflected here as a limitation, not a gap. Re-validate
when the chart values, the SLO targets, or the mTLS doc change shape._
