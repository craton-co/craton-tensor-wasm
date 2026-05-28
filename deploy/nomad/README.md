# Nomad reference manifests for Craton TensorWasm

Reference job specs for HashiCorp Nomad clusters. For Kubernetes use the
plain YAML at `../k8s/` or the Helm chart at `../helm/tensor-wasm/`;
those two and these three files describe the same single-instance
runtime, so an operator can switch back and forth without re-learning
the env-var surface.

This directory is the v0.4 "ops stretch" deliverable in
[`../../docs/PATH-TO-V1.md`](../../docs/PATH-TO-V1.md). It targets
single-instance deployments. The multi-instance constraints are
documented under "Multi-instance constraints" below.

> **Image and artifact placeholders.** The docker spec pins
> `ghcr.io/craton-co/tensor-wasm:0.3.7`. The raw_exec spec fetches
> `https://example.invalid/.../tensor-wasm-x86_64-linux`. Neither URL
> resolves yet (v0.3.7 era). Until the `ghcr.io/craton-co/*` registry
> and the GitHub Releases page are provisioned you must build and
> publish artifacts yourself — see `../k8s/README.md` "Placeholder
> image" for the build-and-push workflow.

## Contents

| File | Driver | Use when |
|---|---|---|
| `tensor-wasm.nomad.hcl` | `docker` | The Nomad client runs `dockerd` (the common case). |
| `tensor-wasm-raw-exec.nomad.hcl` | `raw_exec` | The client has no container runtime; bare-metal HPC nodes; sub-second cold-start evaluation. |
| `README.md` | n/a | This file. |

## Quickstart

### Docker driver

```bash
# Submit the job
nomad job run deploy/nomad/tensor-wasm.nomad.hcl

# Watch placement
nomad job status tensor-wasm

# Tail logs from the running allocation
nomad alloc logs -f $(nomad job allocs -t '{{(index . 0).ID}}' tensor-wasm)

# Probe /healthz through Consul DNS (assumes consul-dns on :8600)
dig @127.0.0.1 -p 8600 tensor-wasm-api.service.consul
curl -s -H 'Authorization: Bearer changeme' \
  http://tensor-wasm-api.service.consul:8080/healthz
```

### raw_exec driver

```bash
nomad job run deploy/nomad/tensor-wasm-raw-exec.nomad.hcl
nomad job status tensor-wasm
```

The raw_exec driver is **disabled by default** in Nomad 1.5 and later.
Enable it explicitly in the client config:

```hcl
plugin "raw_exec" {
  config {
    enabled = true
  }
}
```

Restart the Nomad client after the change. Create the service-account
user on every client node before running the job (raw_exec runs as
the Nomad agent user — root in the default install — unless `user`
is set):

```bash
sudo useradd --system --shell /usr/sbin/nologin \
  --home /var/lib/tensor-wasm tensor-wasm
sudo mkdir -p /var/lib/tensor-wasm
sudo chown tensor-wasm:tensor-wasm /var/lib/tensor-wasm
```

## Status, logs, exec

```bash
# Job-level status
nomad job status tensor-wasm

# Per-allocation detail
nomad alloc status <alloc-id>

# Logs (stdout + stderr)
nomad alloc logs <alloc-id>
nomad alloc logs -stderr <alloc-id>
nomad alloc logs -f <alloc-id>

# Open a shell in the docker-driver task (works only on docker driver)
nomad alloc exec -task tensor-wasm-api <alloc-id> /bin/sh
```

## Bearer-token allowlist

Default `TENSOR_WASM_API_TOKENS = "changeme:tenant=*"` is baked into both
specs as a placeholder. Override on every production deployment via
one of:

1. **Vault template (recommended).** Un-comment the `template { ... }`
   stanza in the task and the `vault { policies = [...] }` stanza on
   the group. See "Vault integration" below.
2. **Job-spec edit.** Replace the literal `TENSOR_WASM_API_TOKENS`
   value in the `env { ... }` block before `nomad job run`. Acceptable
   for single-tenant evaluation; not recommended in production.
3. **Nomad variables (Nomad 1.4+).** Reference a `nomad_var` inside
   the `template` stanza instead of Vault. The token-rotation story
   is identical; the access-control surface is Nomad ACLs instead of
   Vault policies.

The token grammar (`token:tenant=...`) is documented in
`../../crates/tensor-wasm-api/API.md` "Per-tenant scopes".

## Vault integration

The reference specs include a `vault { policies = [] }` stub on the
group and a commented `template { ... }` stanza on the task that
renders `TENSOR_WASM_API_TOKENS` from a Vault KV secret. To wire it up:

1. Write the allowlist into Vault (KV v2 path shown):

   ```bash
   vault kv put kv/tensor-wasm/tokens \
     allowlist='prod:tenant=*,svc-7:tenant=7,svc-8:tenant=8'
   ```

2. Author a policy granting read on that path:

   ```hcl
   # tensor-wasm-tokens.policy
   path "kv/data/tensor-wasm/tokens" {
     capabilities = ["read"]
   }
   ```

   ```bash
   vault policy write tensor-wasm-tokens tensor-wasm-tokens.policy
   ```

3. In the job spec, change:

   ```hcl
   vault {
     policies = ["tensor-wasm-tokens"]
   }
   ```

4. Un-comment the `template { ... }` stanza in the task. The default
   `change_mode = "restart"` re-rolls the task on rotation; switch to
   `change_mode = "signal", change_signal = "SIGHUP"` only if your
   process supports it (v0.3.7 does not — restart is the safe choice).

Rotate the secret with `vault kv put kv/tensor-wasm/tokens
allowlist=...`; the template renderer notices within Vault's lease TTL
and triggers the restart.

## Consul service registration

Both specs register a Consul service named **`tensor-wasm-api`** on
port `8080`. The HTTP health check at `/healthz` runs every 10 s with
a 2 s timeout — 200x the 10 ms P95 SLO documented in
`../../docs/SLO.md` sec 2.2. Failed checks above `check_restart.limit`
restart the allocation.

DNS resolution via consul-dns:

```
tensor-wasm-api.service.consul        →  A record(s) of every healthy alloc
_http._tcp.tensor-wasm-api.service.consul  →  SRV records with port
```

Downstream services should reference the service name, not the
allocation IP. The service name is the documented anchor; the
allocation IPs change on every rollout.

If you do not run Consul, change `provider = "consul"` to
`provider = "nomad"` to use Nomad's built-in service registry instead.
The Nomad provider does not currently support DNS resolution, so
clients must query the Nomad API (`nomad service info tensor-wasm-api`)
or be on Nomad's bridge network.

## GPU prerequisites

Both specs ship with `device "nvidia/gpu" { ... }` commented out. Before
un-commenting the cluster needs:

1. **NVIDIA driver** on every client node, version per
   `../../docs/CUDA-SETUP.md` (minimum 525.60.13 with CUDA 12.0;
   the S22 runner uses 550.54.15 with CUDA 12.4).
2. **`nvidia-container-toolkit`** installed on the host. For the
   docker driver, also configure `nvidia-container-runtime` as a
   runtime (`docker info | grep -i runtime`) so the
   `runtime = "nvidia"` line in `tensor-wasm.nomad.hcl` resolves.
3. **`nomad-device-nvidia` plugin** installed and enabled on every
   GPU client. Drop the binary into the Nomad plugin dir
   (`/opt/nomad/plugins/` is conventional) and add to client config:

   ```hcl
   plugin "nvidia-gpu" {
     config {
       enabled            = true
       ignored_gpu_ids    = []
       fingerprint_period = "1m"
     }
   }
   ```

   Restart the Nomad client; verify with
   `nomad node status -self -verbose | grep nvidia`. The node's
   fingerprint should now list one `nvidia/gpu/*` device per detected
   GPU.

4. **GPU-enabled image** built with the `unified-memory` / `cuda`
   feature set. The default `ghcr.io/craton-co/tensor-wasm:0.3.7` is
   host-only — flipping the device request on with a host-only image
   wastes GPU minutes and treats the GPU as unavailable at runtime.

5. **`CUDA_ARCH` set to the node's SM level** (`sm_75`, `sm_80`,
   `sm_86`, `sm_89`, `sm_90`). See
   `../../docs/CUDA-SETUP.md` "SM-level compatibility matrix" for
   the full table.

Once all five rows are checked, un-comment in `tensor-wasm.nomad.hcl`:

- The `runtime = "nvidia"` line inside `config { ... }`.
- The full `device "nvidia/gpu" { ... }` stanza inside `resources { ... }`.

Then `nomad job run` again. The job re-rolls per the update strategy
below.

## Persistence

`/var/lib/tensor-wasm` holds the snapshot store and the JIT cache. The
docker spec mounts a Nomad **host_volume** there; provision it on every
client that may run the job:

```hcl
# Nomad client config
client {
  host_volume "tensor-wasm-state" {
    path      = "/var/lib/tensor-wasm"
    read_only = false
  }
}
```

```bash
sudo mkdir -p /var/lib/tensor-wasm
sudo chown 65532:65532 /var/lib/tensor-wasm
```

The chown matches the `user = "65532:65532"` setting in the task. If
you change that UID, change the chown.

For ephemeral state (CI smoke tests, single-node evaluation), comment
out both the group-level `volume "state" { ... }` stanza and the
task-level `volume_mount { ... }` stanza. The binary will write into
the allocation scratch dir and lose state on restart — equivalent to
the k8s `emptyDir` default.

For CSI-backed durable storage (cloud blocks, NetApp, Ceph), declare a
CSI volume at the group level instead:

```hcl
volume "state" {
  type            = "csi"
  source          = "tensor-wasm-state"
  read_only       = false
  attachment_mode = "file-system"
  access_mode     = "single-node-writer"
}
```

`single-node-writer` is the correct mode: above one writer the
in-process registry's view of disk diverges.

## Upgrade procedure

Both specs set:

```hcl
update {
  max_parallel      = 1
  health_check      = "checks"
  min_healthy_time  = "10s"
  healthy_deadline  = "5m"
  progress_deadline = "10m"
  auto_revert       = true
}
```

`max_parallel = 1` matches the single-replica constraint discussed in
W3.3 (`../../docs/UPGRADE.md` "Single-replica constraint"): the
runtime keeps no shared state across allocations, so a rolling update
with two live allocs would briefly split the registry and the
rate-limit buckets.

Procedure:

```bash
# 1. Edit the image tag (or any other value) in the job spec.
$EDITOR deploy/nomad/tensor-wasm.nomad.hcl

# 2. Plan to preview the diff and capture the job-modify-index.
nomad job plan deploy/nomad/tensor-wasm.nomad.hcl

# 3. Submit with the modify index from the previous step.
nomad job run -check-index <modify-index> deploy/nomad/tensor-wasm.nomad.hcl

# 4. Watch the deployment.
nomad deployment status -monitor $(nomad deployment list -json \
  | jq -r '.[0].ID')
```

`auto_revert = true` means a failed health check inside
`healthy_deadline` rolls back to the previous job version
automatically. Manual revert is below.

## Rollback procedure

```bash
# List job versions (most recent first).
nomad job history tensor-wasm

# Revert to a specific prior version (the integer in column 1 of
# `nomad job history`).
nomad job revert tensor-wasm <previous-version>

# Watch the revert deployment.
nomad job status tensor-wasm
```

`nomad job revert` is itself an update, so the same `update { ... }`
constraints apply (max_parallel = 1, auto_revert on failure).

If the auto-revert ladders into a revert-of-a-revert loop, stop the
job and re-submit a known-good spec from disk:

```bash
nomad job stop tensor-wasm
nomad job run deploy/nomad/tensor-wasm.nomad.hcl  # from a known-good ref
```

## Backend selection

Both job specs expose an `image_tag` and a `backend` variable at the top
of the file. The backend choice — `cust` | `cudarc` | `cuda-oxide` | `""`
(default; no suffix) — is a **build-time** decision (a Cargo feature
flag, not a runtime knob), so the variable affects only which artifact
the driver fetches: the docker driver appends `-<backend>` to the image
tag, the raw_exec driver appends `-<backend>` to the binary filename in
the artifact URL. The env-var surface below the variable blocks
(`TENSOR_WASM_API_*`, `CUDA_ARCH`, …) is identical across backends.

Override on submit:

```bash
nomad job run -var backend=cudarc -var image_tag=0.3.1 \
  deploy/nomad/tensor-wasm.nomad.hcl
```

`cust` is the legacy default and is EOL upstream (see
`../../docs/RISKS.md` "CUDA `cust` 0.3.x EOL"); `cudarc` is the W1.2
spike and the recommended-stable choice for the v0.3.x line; `cuda-oxide`
is the v0.5 target and is alpha today. The default flips to `cuda-oxide`
(or to `cudarc` as fallback) at v0.5 per RFC 0001 "Rollout (PR
sequencing)". The empty default leaves the URL untouched, so any
self-hosted artifact endpoint that does not adopt the `-<backend>`
suffix convention keeps working. The full trade-off, ambiguous-case
notes, and CHANGELOG cross-references live in
`../helm/tensor-wasm/README.md` "Backend selection"; this section is the
Nomad-specific surface only.

## Multi-instance constraints

The runtime is single-host today. Above `count = 1` requires either:

- **Sticky load-balancer routing** so a given bearer token always
  reaches the same allocation. The per-token rate-limit buckets are
  in-process; without sticky routing the effective limit per token
  scales with the number of allocations.
- **Acceptance that registry / rate-limit / audit state is
  per-allocation.** Acceptable for stateless health checks and
  observability scrapes; not acceptable for tenants who depend on
  a single global state machine.

The Nomad spec keeps `count = 1` and `max_parallel = 1` as the
v0.1.0 contract. Multi-host scheduling is explicitly v2 scope per
`../../docs/PATH-TO-V1.md` "Anti-goals". Revisit when the runtime
ships shared state.

## Uninstall

```bash
nomad job stop -purge tensor-wasm
```

Without `-purge` the job stays in the registry in `dead` status (so
`nomad job history` keeps working). Snapshots written to the
host_volume persist after stop; clean them up out-of-band:

```bash
sudo rm -rf /var/lib/tensor-wasm/*
```

## Cross-references

- `../k8s/README.md` — Plain-YAML Kubernetes alternative; same env-var
  surface, same probe semantics.
- `../helm/tensor-wasm/README.md` — Helm-chart Kubernetes alternative;
  see the "Values reference" table for the canonical knob inventory
  this spec mirrors.
- `../../crates/tensor-wasm-api/API.md` — Env-var reference, endpoint
  contracts, error envelope.
- `../../docs/DEPLOYMENT.md` — Production topology, capacity planning,
  disaster recovery.
- `../../docs/SLO.md` — `/healthz` semantics (10 ms P95 target) and
  the full SLO surface the Nomad check thresholds correspond to.
- `../../docs/CUDA-SETUP.md` — GPU prerequisites cross-linked from the
  GPU section above.
- `../../docs/UPGRADE.md` — Single-replica rolling-update guidance
  referenced by the `update { max_parallel = 1 }` stanza.
- `../../rfcs/0001-cuda-oxide-integration.md` — The cust → cudarc →
  cuda-oxide rollout that motivates the `backend` variable.
- `../../CHANGELOG.md` — Version-by-version backend-status notes.
