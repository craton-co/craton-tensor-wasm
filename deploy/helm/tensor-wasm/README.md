# tensor-wasm Helm chart

Helm chart for Craton TensorWasm, the sandboxed Wasm runtime with WASI-CUDA
host functions for multi-tenant GPU compute. Use this chart when you want a
templated, upgradable, value-driven install; for a hand-rolled install see
the plain YAML manifests at `../../k8s/`.

| Field | Value |
|---|---|
| Chart version | `0.1.0` |
| App version | `0.3.3` |
| Default image | `ghcr.io/craton-co/tensor-wasm:0.3.3` (host-only); `…:0.3.3-cust` / `…-cudarc` / `…-cuda-oxide` when `image.backend` is set |
| Kubernetes | `>= 1.23` |

> **Image registry is not yet provisioned.** The `ghcr.io/craton-co/*` path is
> aspirational as of v0.3.3 — operators must build + push locally until the
> v0.4 release-engineering pipeline lands. The repo root [`Dockerfile`](../../../Dockerfile)
> produces all four variants via `--build-arg BACKEND={"",cust,cudarc,cuda-oxide}`.
> Override `--set image.repository=my-registry/tensor-wasm` to point the chart
> at your registry until the public one exists.
>
> Build commands (run from repo root):
> ```sh
> docker build                                  -t my-registry/tensor-wasm:0.3.3            .
> docker build --build-arg BACKEND=cust         -t my-registry/tensor-wasm:0.3.3-cust       .
> docker build --build-arg BACKEND=cudarc       -t my-registry/tensor-wasm:0.3.3-cudarc     .
> docker build --build-arg BACKEND=cuda-oxide   -t my-registry/tensor-wasm:0.3.3-cuda-oxide .
> ```

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
| `image.backend` | `""` | One of `cust` \| `cudarc` \| `cuda-oxide` \| `""`. Appends `-<backend>` to the tag. See "Backend selection". |
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

## Backend selection

The GPU host runtime is selected at **build time** via a Cargo feature flag
(`unified-memory` / `cudarc-backend` / `cuda-oxide-backend` — see RFC 0001
at `../../../rfcs/0001-cuda-oxide-integration.md` "Feature-flag layout").
Different builds ship as different image tags; the chart picks between them
by suffixing `image.tag` with `-<image.backend>`. The runtime env-var
surface (`rateLimit.qps`, `auth.tokens`, `cuda.arch`, …) is identical
across backends — only the binary inside the image differs.

Three values are accepted for `image.backend`:

- `cust` — the legacy default through v0.2.x. `cust 0.3.x` is EOL upstream
  (see `../../../docs/RISKS.md` "CUDA `cust` 0.3.x EOL") and the W1.2
  spike confirmed it no longer builds against the workspace's pinned
  nightly. Pick this only if you are pinning to a pre-v0.3 image tag for a
  reproducibility window.
- `cudarc` — the W1.2 spike, the **recommended-stable** choice for
  v0.3.x. Clean-room maintained, used by `candle` / `burn` / `dfdx`. The
  CHANGELOG `0.3.0` entry promotes this to the recommended-stable backend
  for the v0.3.x line; the default flips to it (or to `cuda-oxide`,
  contingent on cuda-oxide v0.2 shipping) at v0.5 per RFC 0001 "Rollout
  (PR sequencing)".
- `cuda-oxide` — the v0.5 target, **alpha today**. NVIDIA Labs' Rust →
  PTX compiler + host runtime, v0.1.0 released 2026-05-09. The scaffold
  landed in v0.3.1; parity work is v0.4; default-flip is v0.5 contingent
  on a v0.2 stable host API. Operators evaluating the migration ahead of
  v0.5 set `backend: cuda-oxide` and accept the alpha-churn risk
  documented in RFC 0001 "Drawbacks".

The empty default (`backend: ""`) leaves the tag untouched, so any
out-of-tree registry that does not adopt the `-<backend>` suffix
convention keeps working — useful while
`ghcr.io/craton-co/tensor-wasm:<tag>-<backend>` is still aspirational
(see the "Image registry is a placeholder" callout above; the
backend-suffixed variants share that aspirational status).

**Ambiguous case worth flagging.** The three-way pick does not encode
hardware fit. An operator on a Linux datacenter GPU where the `cust` path
historically worked fine may stay on `backend: cust` for a release cycle
even though `cuda-oxide` would (post-v0.5) be the faster choice; the
chart will happily render either, and there is no upgrade hint. Watch the
CHANGELOG and RFC 0001 "Rollout (PR sequencing)" for the v0.5 cutover —
once the default flips, the absence of an explicit `backend:` value lands
you on `cuda-oxide`. If you want pinning that survives the cutover, set
`backend:` explicitly to the value you actually want, even if it matches
today's default.

A backend flip on its own — same `image.tag`, different `image.backend`
— still re-rolls the pod: the chart hashes the resolved image reference
into a `checksum/image` annotation alongside the existing
`checksum/config` and `checksum/secret` triggers.

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
- `../../nomad/README.md` — Nomad alternative; mirrors the same
  `image.backend` toggle as a Nomad variable.
- `../../../crates/tensor-wasm-api/API.md` — Env vars, endpoints, error
  envelope.
- `../../../docs/DEPLOYMENT.md` — Production topology and DR.
- `../../../docs/SLO.md` — `/healthz` SLO; informs the probe thresholds.
- `../../../docs/CUDA-SETUP.md` — GPU node prerequisites.
- `../../../rfcs/0001-cuda-oxide-integration.md` — The cust → cudarc →
  cuda-oxide rollout that motivates the `image.backend` toggle.
- `../../../CHANGELOG.md` — Version-by-version backend-status notes.
