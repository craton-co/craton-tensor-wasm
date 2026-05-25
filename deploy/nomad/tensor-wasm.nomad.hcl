# Nomad job spec for Craton TensorWasm (docker driver).
#
# Single-instance reference deployment that mirrors the k8s manifests under
# deploy/k8s/ and the Helm chart under deploy/helm/tensor-wasm/. An operator
# should be able to switch between the k8s and the Nomad path without
# re-learning the env-var surface.
#
# Submit with:
#   nomad job run deploy/nomad/tensor-wasm.nomad.hcl
#
# The image tag `0.1.0` is a placeholder. The ghcr.io/craton-co/* registry is
# not yet provisioned (v0.1.0 era); see README.md "Placeholder image" for the
# build-and-push workflow against your own registry.
#
# Single-instance constraint: the runtime is single-host today (in-process
# function registry, per-token rate-limiter buckets, in-process audit log
# sink). Above count = 1 needs sticky load-balancer routing OR an operator
# who accepts that registry / rate-limit / audit state is per-allocation.
# See docs/UPGRADE.md "Single-replica constraint" and the matching note in
# deploy/helm/tensor-wasm/values.yaml `replicaCount`.

job "tensor-wasm" {
  datacenters = ["dc1"]
  type        = "service"

  # Rolling-update strategy. With count = 1 (the v0.1 reference), the only
  # meaningful update strategy is max_parallel = 1; this matches the
  # `strategy: Recreate` choice in deploy/k8s/20-deployment.yaml and the
  # single-replica rollout described in docs/UPGRADE.md (W3.3).
  update {
    max_parallel      = 1
    health_check      = "checks"
    min_healthy_time  = "10s"
    healthy_deadline  = "5m"
    progress_deadline = "10m"
    auto_revert       = true
  }

  group "api" {
    count = 1

    # /var/lib/tensor-wasm holds snapshots and the JIT cache. With docker
    # driver, the host-volume mount below points at a Nomad host_volume
    # provisioned on the client (see README.md "Persistence").
    # For ephemeral state, comment out the volume + volume_mount stanzas
    # and the binary will write into the allocation scratch dir.
    volume "state" {
      type      = "host"
      source    = "tensor-wasm-state"
      read_only = false
    }

    network {
      mode = "bridge"
      port "http" {
        to           = 8080
        host_network = "default"
      }
    }

    # Automatic Consul service registration. The service name is the anchor
    # other workloads use to discover the API; downstream consul-template /
    # connect mesh configs should reference `tensor-wasm-api`.
    service {
      name     = "tensor-wasm-api"
      port     = "http"
      provider = "consul"
      tags = [
        "tensor-wasm",
        "api",
        "v0.1.0",
      ]

      # /healthz semantics per docs/SLO.md sec 2.2: returns 200 while the
      # axum router is serving; P95 SLO is 10 ms. The 2 s timeout below
      # is 200x the SLO, matching the k8s readinessProbe timeoutSeconds.
      check {
        name     = "healthz-http"
        type     = "http"
        path     = "/healthz"
        port     = "http"
        interval = "10s"
        timeout  = "2s"

        check_restart {
          limit           = 3
          grace           = "30s"
          ignore_warnings = false
        }
      }
    }

    # Vault stub. Populate `policies` with the Vault policy that grants read
    # access to the path where the bearer-token allowlist is stored, then
    # un-comment the `template` stanza in the task below. The policy itself
    # is documented in README.md "Vault integration".
    vault {
      policies = []
    }

    restart {
      attempts = 3
      interval = "5m"
      delay    = "15s"
      mode     = "fail"
    }

    task "tensor-wasm-api" {
      driver = "docker"

      config {
        image = "ghcr.io/craton-co/tensor-wasm:0.1.0"
        ports = ["http"]

        args = [
          "serve",
          "--addr",
          "0.0.0.0:8080",
        ]

        # readOnlyRootFilesystem equivalent. Combined with the host volume
        # below for /var/lib/tensor-wasm and the alloc scratch dir for /tmp.
        readonly_rootfs = true

        # Drop all Linux capabilities and run as the same numeric UID the
        # k8s securityContext uses (65532).
        cap_drop = ["ALL"]

        # GPU device assignment via the nvidia-container-runtime path.
        # Uncomment when the Nomad client has the nvidia container runtime
        # configured as a plugin (`plugin "nvidia"` in client config), the
        # node has driver + nvidia-container-toolkit installed per
        # docs/CUDA-SETUP.md, and the image variant has the `cuda` /
        # `unified-memory` features compiled in.
        #
        # runtime = "nvidia"
      }

      # Run as the same numeric non-root UID as the k8s deployment so a
      # host-volume bind-mount provisioned for one driver works for the
      # other.
      user = "65532:65532"

      # Env vars matching the k8s ConfigMap at deploy/k8s/10-configmap.yaml
      # and the Helm values at deploy/helm/tensor-wasm/values.yaml. Keep
      # this list in sync with the configmap.yaml template when adding
      # knobs upstream.
      env {
        # HTTP listen address. Must match the `--addr` arg and the port
        # mapping above; the binary reads the env var when the CLI arg is
        # absent.
        TENSOR_WASM_API_LISTEN_ADDR = "0.0.0.0:8080"

        # tracing-subscriber EnvFilter directive. trace / debug / info /
        # warn / error or a per-target filter like
        # "info,tensor_wasm_exec=debug".
        TENSOR_WASM_LOG = "info"

        # Per-token rate limit. Both QPS and burst must be > 0 for the
        # limiter to engage; see crates/tensor-wasm-api/API.md
        # "Per-token rate limiting".
        TENSOR_WASM_API_RATE_LIMIT_QPS   = "100"
        TENSOR_WASM_API_RATE_LIMIT_BURST = "200"

        # If "1", every request must carry X-TensorWasm-Tenant. Leave at
        # "0" for the permissive default that maps a missing header to
        # tenant 0.
        TENSOR_WASM_API_REQUIRE_TENANT = "0"

        # Audit-log destination. "stdout" (default), "file:/path/to.log"
        # for an append-only file, or "none" to disable. The file path
        # must be writable; under readonly_rootfs that means either a
        # host_volume mount or the alloc scratch dir
        # (NOMAD_ALLOC_DIR/data/audit.log).
        TENSOR_WASM_API_AUDIT_LOG = "stdout"

        # OpenTelemetry OTLP collector. Empty disables OTLP export.
        # Common values:
        #   http://otel-collector.service.consul:4317
        #   http://jaeger.service.consul:4317
        TENSOR_WASM_OTLP_ENDPOINT = ""

        # CUDA target compute capability for JIT-emitted PTX. Only
        # consulted on GPU-enabled builds; harmless on host-only.
        # Match the node's SM level: sm_70, sm_75, sm_80, sm_86, sm_89,
        # sm_90 (see docs/CUDA-SETUP.md "SM-level compatibility matrix").
        CUDA_ARCH = "sm_80"

        RUST_BACKTRACE = "1"

        # Bearer-token allowlist. The static default is intentionally
        # weak; on every production deployment override via the Vault
        # template below.
        TENSOR_WASM_API_TOKENS = "changeme:tenant=*"
      }

      # Vault-managed token rotation. Uncomment and adapt to your Vault
      # layout. The rendered file is sourced as env vars (env = true) and
      # the task re-reads on rotation per change_mode = "restart".
      #
      # template {
      #   data = <<-EOH
      #     {{- with secret "kv/data/tensor-wasm/tokens" -}}
      #     TENSOR_WASM_API_TOKENS={{ .Data.data.allowlist }}
      #     {{- end -}}
      #   EOH
      #   destination = "secrets/tokens.env"
      #   env         = true
      #   change_mode = "restart"
      #   perms       = "0400"
      # }

      # Mount the host volume declared at the group level. With
      # readonly_rootfs = true the binary must be able to write its
      # snapshot + JIT cache somewhere — this is that somewhere. For an
      # ephemeral install, comment both this stanza and the group's
      # `volume "state" { ... }` stanza.
      volume_mount {
        volume      = "state"
        destination = "/var/lib/tensor-wasm"
        read_only   = false
      }

      resources {
        # CPU is in MHz. 2000 MHz is a conservative match for the
        # `cpu: "2"` limit in deploy/k8s/20-deployment.yaml on a 2 GHz
        # reference core; tune to your scheduler's CPU shares.
        cpu    = 2000
        # Memory in MiB.
        memory = 4096

        # GPU device request via the nomad-device-nvidia plugin. Uncomment
        # when the plugin is installed (see README.md "GPU prerequisites").
        # The reference deployment is host-only; flipping this on without
        # also flipping the `runtime = "nvidia"` in the docker config and
        # using a GPU-enabled image variant will fail at allocation time.
        #
        # device "nvidia/gpu" {
        #   count = 1
        #
        #   constraint {
        #     attribute = "${device.attr.compute_capability}"
        #     operator  = ">="
        #     value     = "7.0"
        #   }
        # }
      }

      # Plain-Prometheus scrape hint (consul-prometheus integration reads
      # tags of the form `prometheus.io/...`). Harmless if you use the
      # ServiceMonitor path on a k8s cluster behind a Nomad-managed
      # bridge.
      meta {
        "prometheus.io/scrape" = "true"
        "prometheus.io/port"   = "8080"
        "prometheus.io/path"   = "/metrics"
      }

      logs {
        max_files     = 10
        max_file_size = 10
      }

      kill_timeout = "30s"
    }
  }
}
