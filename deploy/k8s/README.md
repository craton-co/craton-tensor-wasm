# Kubernetes reference manifests for Craton TensorWasm

Plain-YAML reference deployment for self-managed installs. For templated
installs use the Helm chart at `../helm/tensor-wasm/` instead.

> **Image tag is a placeholder.** The manifests pin
> `ghcr.io/craton-co/tensor-wasm:0.1.0`. The `ghcr.io/craton-co/*`
> registry is not yet provisioned (v0.1.0 era of the project). Until it
> exists you must build and push the image yourself from the Dockerfile at
> `../../docker/tensor-wasm-api.Dockerfile` and adjust the `image:` field
> in `20-deployment.yaml` to point at your registry.

## Contents

| File | Purpose |
|---|---|
| `00-namespace.yaml` | `tensor-wasm` namespace |
| `10-configmap.yaml` | Non-secret env vars (`TENSOR_WASM_API_*`, `TENSOR_WASM_LOG`, `CUDA_ARCH`, optional OTLP endpoint) |
| `20-deployment.yaml` | Single-replica Deployment with `/healthz` probes, non-root securityContext, GPU resource request commented out |
| `30-service.yaml` | ClusterIP Service on port 8080 (NodePort variant commented out) |
| `40-servicemonitor.yaml` | Optional ServiceMonitor for prometheus-operator users |

## Install order

The files are numerically prefixed so a single apply does the right thing:

```bash
kubectl apply -f deploy/k8s/
```

If your cluster is not running `prometheus-operator`, the ServiceMonitor
apply will fail. Skip it with:

```bash
kubectl apply \
  -f deploy/k8s/00-namespace.yaml \
  -f deploy/k8s/10-configmap.yaml \
  -f deploy/k8s/20-deployment.yaml \
  -f deploy/k8s/30-service.yaml
```

## Provision the bearer-token secret

The Deployment references a Secret named `tensor-wasm-tokens` with a
single key, `TENSOR_WASM_API_TOKENS`. Until you create it the pod will
crash-loop with `secret "tensor-wasm-tokens" not found`. Create one:

```bash
kubectl -n tensor-wasm create secret generic tensor-wasm-tokens \
  --from-literal=TENSOR_WASM_API_TOKENS='changeme:tenant=*'
```

For multi-tenant deployments issue scoped tokens (see
`crates/tensor-wasm-api/API.md` "Per-tenant scopes"):

```bash
kubectl -n tensor-wasm create secret generic tensor-wasm-tokens \
  --from-literal=TENSOR_WASM_API_TOKENS='admin:tenant=*,svc-7:tenant=7,svc-8:tenant=8'
```

Rotate by recreating the Secret and deleting the pod:

```bash
kubectl -n tensor-wasm delete secret tensor-wasm-tokens
kubectl -n tensor-wasm create secret generic tensor-wasm-tokens \
  --from-literal=TENSOR_WASM_API_TOKENS='new-token:tenant=*'
kubectl -n tensor-wasm rollout restart deployment/tensor-wasm-api
```

> Do **not** commit a real token value to git or into the ConfigMap. The
> token is the only thing standing between the open internet and your
> compute; treat it like a database password.

## Verify the install

```bash
# Pods up and ready
kubectl -n tensor-wasm get pods

# Probe readiness directly
kubectl -n tensor-wasm get deploy tensor-wasm-api

# Logs
kubectl -n tensor-wasm logs deploy/tensor-wasm-api -f

# Hit the API from your laptop
kubectl -n tensor-wasm port-forward svc/tensor-wasm-api 8080:8080
# In another shell:
curl -s -H 'Authorization: Bearer changeme' http://localhost:8080/healthz
```

## Expose the service externally

### Option A: Ingress (production)

Adjust to your cluster's IngressClass:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tensor-wasm-api
  namespace: tensor-wasm
  annotations:
    # nginx example; substitute traefik / istio / etc.
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
spec:
  ingressClassName: nginx
  rules:
    - host: tensor-wasm.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: tensor-wasm-api
                port:
                  number: 8080
  tls:
    - hosts: [tensor-wasm.example.com]
      secretName: tensor-wasm-tls
```

Save as `ingress.yaml`, then `kubectl apply -f ingress.yaml`. The
`proxy-body-size` annotation matches the in-process 64 MiB body cap
(see `crates/tensor-wasm-api/API.md` "Request limits"); larger bodies
are rejected before the handler runs, so allowing them at the ingress
is wasted bandwidth.

### Option B: LoadBalancer (cloud-managed)

Switch the Service type in `30-service.yaml` to `LoadBalancer` and
re-apply, or layer your own `Service`:

```bash
kubectl -n tensor-wasm expose deployment tensor-wasm-api \
  --type=LoadBalancer --name=tensor-wasm-api-lb --port=8080
```

### Option C: NodePort (bare metal)

Uncomment the NodePort variant in `30-service.yaml` and re-apply.

## GPU-node prerequisite checklist

The reference Deployment ships with `nvidia.com/gpu` resources commented
out. Before uncommenting them, the cluster needs:

- [ ] **NVIDIA driver** on every GPU node, version pinned per
      `docs/CUDA-SETUP.md` (the S22 runner uses driver 550.54.15 with
      CUDA 12.4; minimum is driver 525.60.13 with CUDA 12.0).
- [ ] **`nvidia-container-toolkit`** installed on the node and
      configured as the default container runtime (or via a
      `RuntimeClass` named `nvidia`).
- [ ] **`nvidia-device-plugin` DaemonSet** running in `kube-system`
      (typically installed via the helm chart from
      `nvdp/nvidia-device-plugin`). Verify with
      `kubectl get pods -n kube-system -l app=nvidia-device-plugin-daemonset`.
- [ ] **Node label** so the scheduler picks a GPU node — most installs
      use `nvidia.com/gpu.present=true`. Apply with
      `kubectl label node <node> nvidia.com/gpu.present=true`.
- [ ] **A GPU-enabled tensor-wasm image** built with the `unified-memory`
      / `cuda` feature set. The default image at
      `ghcr.io/craton-co/tensor-wasm:0.1.0` is host-only and will
      treat the GPU as unavailable.
- [ ] **Compute capability matches the image's `CUDA_ARCH`.** Set
      `CUDA_ARCH` in `10-configmap.yaml` to the node's SM level
      (`sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`). See
      `docs/CUDA-SETUP.md` "SM-level compatibility matrix" for the
      full list.

Once all six rows are checked, uncomment the `nvidia.com/gpu: 1` line in
`20-deployment.yaml` and the `nodeSelector` / `tolerations` /
`runtimeClassName` block at the bottom of the same file, then re-apply.

## Uninstall

```bash
kubectl delete namespace tensor-wasm
```

The namespace delete cascades to every object in this directory. Snapshots
written to the `emptyDir` volume are lost — provision a PVC via the Helm
chart if you need durable state.

## Cross-references

- `../helm/tensor-wasm/README.md` — Helm-managed alternative.
- `../../crates/tensor-wasm-api/API.md` — env-var reference, endpoint
  contracts, error envelope.
- `../../docs/DEPLOYMENT.md` — production topology, capacity planning,
  disaster recovery.
- `../../docs/SLO.md` — `/healthz` semantics (10 ms P95 target) and the
  full SLO surface the probes correspond to.
- `../../docs/CUDA-SETUP.md` — GPU prerequisites referenced in the
  checklist above.
