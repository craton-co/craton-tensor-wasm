# tensor-wasm Helm chart

Helm chart for Craton TensorWasm, the sandboxed Wasm runtime with WASI-CUDA
host functions for multi-tenant GPU compute. Use this chart when you want a
templated, upgradable, value-driven install; for a hand-rolled install see
the plain YAML manifests at `../../k8s/`.

| Field | Value |
|---|---|
| Chart version | `0.1.0` |
| App version | `0.1.0` |
| Default image | `ghcr.io/craton-co/tensor-wasm:0.1.0` |
| Kubernetes | `>= 1.23` |

> **Image registry is a placeholder.** The `ghcr.io/craton-co/*` registry is
> not yet provisioned (v0.1.0 era). Build and push from
> `../../docker/tensor-wasm-api.Dockerfile` and override
> `--set image.repository=my-registry/tensor-wasm` until the public registry
> exists.

## Install

```bash
helm install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm --create-namespace
```

Override the token allowlist on every production install:

```bash
helm install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm --create-namespace \
  --set auth.tokens='prod-token:tenant=*,tenant-7:tenant=7'
```

Or point at a pre-provisioned Secret containing the
`TENSOR_WASM_API_TOKENS` key:

```bash
kubectl -n tensor-wasm create secret generic my-tokens \
  --from-literal=TENSOR_WASM_API_TOKENS='prod-token:tenant=*'

helm install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm --create-namespace \
  --set auth.existingSecret=my-tokens
```

## Upgrade

```bash
helm upgrade tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm \
  --set image.tag=0.2.0
```

The Deployment carries `checksum/config` and `checksum/secret` annotations
that change when their backing object changes, so a value-only upgrade
(e.g. flipping `rateLimit.qps`) re-rolls the pod automatically.

## Uninstall

```bash
helm uninstall tensor-wasm -n tensor-wasm
kubectl delete namespace tensor-wasm   # optional
```

## Values reference

All keys live in `values.yaml` and are documented inline; the table below is
a high-level summary.

| Key | Default | Notes |
|---|---|---|
| `image.repository` | `ghcr.io/craton-co/tensor-wasm` | OCI repository. |
| `image.tag` | `""` (uses `.Chart.AppVersion`) | |
| `image.pullPolicy` | `IfNotPresent` | |
| `imagePullSecrets` | `[]` | |
| `replicaCount` | `1` | Runtime keeps no shared state across pods. |
| `strategy.type` | `Recreate` | Use `RollingUpdate` only with `replicaCount > 1`. |
| `service.type` | `ClusterIP` | `NodePort` / `LoadBalancer` also valid. |
| `service.port` | `8080` | |
| `service.nodePort` | `0` | Honored only when `service.type=NodePort`. |
| `ingress.enabled` | `false` | |
| `ingress.className` | `""` | |
| `ingress.hosts` | one example host | |
| `ingress.tls` | `[]` | |
| `resources.requests` | `cpu 500m / mem 1Gi` | |
| `resources.limits` | `cpu 2 / mem 4Gi` | |
| `gpu.enabled` | `false` | |
| `gpu.count` | `1` | **Must be an integer.** |
| `gpu.nodeSelector` | `nvidia.com/gpu.present: "true"` | |
| `gpu.tolerations` | one toleration | |
| `gpu.runtimeClassName` | `nvidia` | |
| `auth.tokens` | `"changeme:tenant=*"` | Override on every prod install. |
| `auth.existingSecret` | `""` | Wins over `auth.tokens` when set. |
| `auth.requireTenant` | `false` | Sets `TENSOR_WASM_API_REQUIRE_TENANT`. |
| `rateLimit.qps` | `100` | Per-token QPS. |
| `rateLimit.burst` | `200` | Per-token burst. |
| `otlp.endpoint` | `""` | OTLP collector URL; empty disables export. |
| `log.level` | `"info"` | |
| `cuda.arch` | `"sm_80"` | Match node SM level when GPU is enabled. |
| `livenessProbe.enabled` | `true` | Targets `/healthz`. |
| `readinessProbe.enabled` | `true` | |
| `startupProbe.enabled` | `true` | |
| `persistence.enabled` | `false` | `emptyDir` when false. |
| `persistence.size` | `10Gi` | |
| `prometheus.enabled` | `false` | Requires prometheus-operator CRDs. |
| `prometheus.interval` | `15s` | |
| `prometheus.additionalLabels` | `release: prometheus` | Match your Prometheus CR. |
| `serviceAccount.create` | `true` | |
| `extraEnv` / `extraEnvFrom` | `[]` | Pass through to PodSpec. |

## Common overlay recipes

### Enable GPU scheduling

```yaml
# values-gpu.yaml
image:
  repository: my-registry/tensor-wasm-gpu
  tag: "0.1.0"
gpu:
  enabled: true
  count: 1
cuda:
  arch: "sm_89"   # Match the node SKU
```

```bash
helm install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm --create-namespace -f values-gpu.yaml
```

### Expose via Ingress with TLS

```yaml
# values-ingress.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
  hosts:
    - host: tensor-wasm.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - hosts: [tensor-wasm.example.com]
      secretName: tensor-wasm-tls
```

The `proxy-body-size: 64m` annotation matches the in-process 64 MiB body
cap (see `crates/tensor-wasm-api/API.md` "Request limits").

### Hand off metrics to Prometheus Operator

```yaml
# values-prom.yaml
prometheus:
  enabled: true
  additionalLabels:
    release: prometheus   # Match your Prometheus CR's serviceMonitorSelector
```

## Notes for chart maintainers

- The chart bumps version independently from the application. Chart-only
  changes (template fixes, new values) bump the chart patch version;
  breaking value-key changes bump the chart major.
- `Chart.yaml` `appVersion` must move in lockstep with the default
  `values.yaml` `image.tag` (or with the chart's appVersion fallback in
  `_helpers.tpl` `tensor-wasm.image`).
- Helm `lint` and `template` checks should run before tagging a release:

  ```bash
  helm lint deploy/helm/tensor-wasm
  helm template tensor-wasm deploy/helm/tensor-wasm -n tensor-wasm > /tmp/rendered.yaml
  kubectl apply --dry-run=client -f /tmp/rendered.yaml
  ```

## Cross-references

- `../../k8s/README.md` — Plain-YAML alternative.
- `../../../crates/tensor-wasm-api/API.md` — Env vars, endpoints, error
  envelope.
- `../../../docs/DEPLOYMENT.md` — Production topology and DR.
- `../../../docs/SLO.md` — `/healthz` SLO; informs the probe thresholds.
- `../../../docs/CUDA-SETUP.md` — GPU node prerequisites.
