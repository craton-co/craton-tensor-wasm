# Deploying Craton TensorWasm

This document covers running **TensorWasm** in production: topology, sizing, configuration, and disaster recovery. If you're still onboarding, read [GETTING-STARTED.md](./GETTING-STARTED.md) first; if you're writing the functions, [WASM-DEVELOPER-GUIDE.md](./WASM-DEVELOPER-GUIDE.md) is your starting point.

## 1. Production topology

A typical TensorWasm production deployment looks like:

```
            +---------------------+
            |   Load Balancer     |
            +----------+----------+
                       |
        +--------------+--------------+
        |              |              |
   +----+----+    +----+----+    +----+----+
   | tensor-wasm-api|    | tensor-wasm-api|    | tensor-wasm-api|
   |  + GPU  |    |  + GPU  |    |  + GPU  |
   |  + MPS  |    |  + MPS  |    |  + MPS  |
   +----+----+    +----+----+    +----+----+
        |              |              |
        +--------------+--------------+
                       |
        +--------------+--------------+
        |                             |
   +----+-------+              +------+------+
   | Prometheus |              | Jaeger /    |
   | + Grafana  |              | Tempo       |
   +------------+              +-------------+
```

- **`tensor-wasm-api` replicas** sit behind a load balancer (any L7 proxy will do — nginx, Caddy, an ALB). Each replica is stateless.
- **MPS (Multi-Process Service)** runs as a daemon on each host that has a GPU, providing context isolation between concurrent tenants. See [MPS-SETUP.md](./MPS-SETUP.md).
- **Prometheus and Grafana** are shared across replicas for metrics.
- **Jaeger or Tempo** are shared for traces.

The gateway tier holds **no durable state** — the only durable artifact is the snapshot store (covered below).

## 2. System requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | sm_70, CUDA 11.8 | sm_80+, CUDA 12.0+ |
| Disk | HDD | SSD (snapshot I/O dominates cold-start at large payloads) |
| Network | 1 Gb | 10 Gb between gateway and snapshot storage |

The SSD recommendation is not cosmetic. At payload sizes above ~32 MiB, snapshot read latency dominates the end-to-end cold-start budget; HDDs add tens of milliseconds you can't get back. See [COLD-START.md](./COLD-START.md) for the numbers.

## 3. Configuration

Craton TensorWasm is configured by environment variables. The full set:

| Variable | Default | Purpose |
|---|---|---|
| `TENSOR_WASM_LOG` | `info` | `tracing-subscriber` filter directive. |
| `TENSOR_WASM_OTLP_ENDPOINT` | unset | OpenTelemetry collector endpoint (e.g. `http://otel:4317`). |
| `TENSOR_WASM_LISTEN_ADDR` | `0.0.0.0:8080` | HTTP bind address for `tensor-wasm serve`. |
| `CUDA_ROOT` | autodetected | Override CUDA toolkit location (build- and run-time). |
| `CUDA_ARCH` | none (required for GPU builds) | Target GPU compute capability for emitted PTX; set to match your GPU (e.g. `sm_80` for A100, `sm_89` for L4). See [CUDA-SETUP.md](./CUDA-SETUP.md). |

For zero-trust environments, set `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` (32-byte hex HMAC key) to authenticate snapshot bytes, and `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE=true` to refuse unsigned snapshots (see [SNAPSHOT-COMPATIBILITY.md](./SNAPSHOT-COMPATIBILITY.md)).

## 4. Docker Compose stack

The repo ships a `docker-compose.yml` at its root that brings up `tensor-wasm-api`, Prometheus, Grafana, and Jaeger with a single command:

```sh
docker compose up
```

The compose file pins versions, wires up the OTLP exporter, and pre-loads a Grafana dashboard for the four key SLI metrics (see Monitoring below). It's the recommended way to evaluate a real TensorWasm deployment locally before promoting to Kubernetes.

## 5. Kubernetes

A Helm chart ships at [`deploy/helm/tensor-wasm/`](../deploy/helm/tensor-wasm/) — it is the recommended way to install on Kubernetes (`helm install tensor-wasm ./deploy/helm/tensor-wasm -n tensor-wasm --create-namespace`; see the chart's [`README.md`](../deploy/helm/tensor-wasm/README.md) for the full values reference). If you prefer a hand-rolled manifest, a minimal `Deployment` looks like:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensor-wasm-api
spec:
  replicas: 3
  selector:
    matchLabels: { app: tensor-wasm-api }
  template:
    metadata:
      labels: { app: tensor-wasm-api }
    spec:
      containers:
        - name: tensor-wasm-api
          image: ghcr.io/craton-co/craton-tensor-wasm-api:0.1.0
          args: ["serve", "--addr", "0.0.0.0:8080"]
          ports: [{ containerPort: 8080 }]
          env:
            - { name: TENSOR_WASM_LOG,           value: "info" }
            - { name: TENSOR_WASM_OTLP_ENDPOINT, value: "http://otel-collector:4317" }
            - { name: CUDA_ARCH,          value: "sm_80" }
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "8"
```

Pair with the standard `nvidia-device-plugin` DaemonSet for GPU scheduling.

## 6. TLS

Do **not** terminate TLS inside `tensor-wasm-api` for v0.1 — the axum stack is configured for plaintext H2C. Put a TLS-terminating proxy in front of it: nginx, Caddy, an ALB, or a service mesh sidecar. Any of these is more battle-tested than what we'd ship inline.

A minimal Caddyfile:

```
api.example.com {
    reverse_proxy tensor-wasm-api:8080
}
```

That's it — Caddy will provision an ACME certificate on startup.

## 7. Multi-tenant capacity planning

Capacity is bounded by **GPU HBM**, not CPU or system RAM. The arithmetic:

- Each instance reserves up to `EngineConfig::max_memory_bytes` (default **256 MiB**) of linear memory, which lives in unified memory and is page-backed on the device.
- A 40 GiB A100 therefore supports a theoretical ceiling of `40 GiB / 256 MiB ≈ 160` concurrently resident instances, minus headroom for driver state and PTX caches. In practice plan for **120**.
- Per-tenant quotas are tracked through `tensor-wasm-tenant` via the `consume_bytes` budget on each tenant principal. Set quotas conservatively; you can always raise them.

If you need stronger isolation than MPS provides, run one `tensor-wasm-api` replica per tenant on a dedicated GPU. The replica is stateless, so this scales horizontally without coordination.

## 8. Backup and restore

Snapshots are the **only** durable artifact in a TensorWasm deployment. They contain:

- The Wasm module bytes
- The serialized `TensorWasmLinearMemory`
- The `KernelRegistry` state (loaded PTX, kernel ids)
- Engine config and tenant attribution metadata

Snapshots are **portable across hosts of the same architecture** — `sm_80` to `sm_80` is fine; `sm_80` to `sm_90` requires a rebuild because PTX is JITed per arch.

The snapshot schema is versioned. The reader does **not** migrate snapshots in place: a `version` mismatch is a hard error (within v0.x the reader version must equal the writer version). The supported upgrade path is to re-capture from the live instance under the new format — or, if the source instance is gone, run an older `tensor-wasm` binary against the old snapshot, restore the instance, then re-capture under the current binary. See [SNAPSHOT-COMPATIBILITY.md](./SNAPSHOT-COMPATIBILITY.md). Always rehearse this on a staging snapshot before bumping the binary in production.

Recommended rotation: snapshot every active instance hourly, with a 7-day retention window in your object store. Pair with periodic integrity checks on a random 1% sample — see the offline integrity check in [BACKUP-RESTORE.md](./BACKUP-RESTORE.md) §7.1, which parses each blob through `SnapshotReader::restore` (validating the CRC32 and per-blob size caps) without restoring it to a live instance.

## 9. Monitoring

TensorWasm exposes Prometheus metrics on `/metrics`. The **four SLI metrics** to alert on:

| Metric | What it tells you |
|---|---|
| `tensor_wasm_active_instances` | Capacity headroom. Alert when within 20% of your planned ceiling. |
| `tensor_wasm_kernel_dispatches_total` | Throughput. Sudden drops indicate a stuck instance or scheduler. |
| `tensor_wasm_kernel_latency_seconds` | Tail latency. p99 regressions usually mean GPU contention. |
| `tensor_wasm_offload_success_total / tensor_wasm_offload_fallback_total` | Auto-offload health. A growing fallback ratio means promoted kernels are failing on the device. |

The bundled Grafana dashboard groups these into a single overview page. For the full catalog — including the per-tenant breakdown — see [OBSERVABILITY.md](./OBSERVABILITY.md).

## 10. Disaster recovery

Because the gateway tier is stateless, DR is essentially **snapshot DR**:

1. **Ship snapshots off-host.** Mirror to S3, GCS, or your equivalent. The snapshot format is content-addressed by sha256, so the mirror is naturally deduplicating.
2. **Verify periodically.** Run the offline snapshot integrity check ([BACKUP-RESTORE.md](./BACKUP-RESTORE.md) §7.1) against a sample of stored snapshots on a schedule.
3. **Practice restore.** A quarterly DR drill that restores a snapshot to a fresh region is cheaper than discovering schema drift the day you actually need it.
4. **Keep the binary in sync with the snapshot schema.** A binary that can read your latest snapshots should be available in your registry at all times — not just rebuildable from source.

If both the gateway and a snapshot are intact, recovery is a `tensor-wasm snapshot restore --input <file>.tensor-wasm --as-instance <instance-id> --server <url>` away (see [BACKUP-RESTORE.md](./BACKUP-RESTORE.md) §6.1). If only the Wasm module bytes survive, you can redeploy clean — you lose accumulated linear-memory state, but the function is back online.

For broader operational guidance, see the other runbooks in this directory.
