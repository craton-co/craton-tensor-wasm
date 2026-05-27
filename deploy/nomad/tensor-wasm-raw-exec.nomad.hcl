# Nomad job spec for Craton TensorWasm (raw_exec driver).
#
# Alternate spec for hosts that do not run Docker. The binary is fetched as
# an `artifact` block; substitute the URL + checksum for your release. Use
# this path when:
#   - The cluster does not run a container runtime (bare-metal HPC nodes).
#   - You want to bypass image-pull latency for sub-second cold starts.
#   - You are evaluating TensorWasm on a single-node Nomad agent before
#     standing up registry plumbing.
#
# Submit with:
#   nomad job run deploy/nomad/tensor-wasm-raw-exec.nomad.hcl
#
# SECURITY NOTE: raw_exec runs as the Nomad client agent user (root, unless
# you have set `user = ...` on the client). The driver is disabled by
# default in nomad-1.5+; enable it explicitly in client config:
#
#   plugin "raw_exec" {
#     config { enabled = true }
#   }
#
# The `user = "tensor-wasm"` line below downgrades the task to a service
# account; create it on every client node before running this job, or the
# task will fail to start.
#
# Single-instance constraint: see the comment in tensor-wasm.nomad.hcl.
# The runtime is single-host today; count = 1 is the only safe value
# without sticky LB routing.

# Release version. Override with `-var image_tag=0.3.1` on submit. Mirrors
# the `image_tag` variable in tensor-wasm.nomad.hcl so an operator can use
# the same -var flag for both drivers.
variable "image_tag" {
  type    = string
  default = "0.3.5"
}

# GPU backend. One of "cust" | "cudarc" | "cuda-oxide" | "" (default; no
# suffix). When non-empty the binary filename is suffixed with `-<backend>`
# in the artifact URL, matching the build-time feature-flag layout from
# RFC 0001 (rfcs/0001-cuda-oxide-integration.md "Feature-flag layout").
# `cust` = legacy default, `cudarc` = the v0.3.x recommended-stable choice,
# `cuda-oxide` = the v0.5 target (alpha today). See
# deploy/helm/tensor-wasm/README.md "Backend selection" for the trade-off.
variable "backend" {
  type    = string
  default = ""
}

job "tensor-wasm" {
  datacenters = ["dc1"]
  type        = "service"

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

    network {
      mode = "host"
      port "http" {
        static = 8080
      }
    }

    service {
      name     = "tensor-wasm-api"
      port     = "http"
      provider = "consul"
      tags = [
        "tensor-wasm",
        "api",
        "v0.3.5",
        "raw_exec",
      ]

      check {
        name     = "healthz-http"
        type     = "http"
        path     = "/healthz"
        port     = "http"
        interval = "10s"
        timeout  = "2s"

        check_restart {
          limit = 3
          grace = "30s"
        }
      }
    }

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
      driver = "raw_exec"

      # Download the prebuilt binary. The URL below is a placeholder; the
      # ghcr.io/craton-co/* registry and the GitHub Releases page are not
      # yet provisioned (v0.3.5 era). Either:
      #   (a) self-host the artifact (object storage, internal mirror) and
      #       replace `source` + `options.checksum` accordingly, or
      #   (b) bake the binary into a base AMI / machine image and replace
      #       this artifact stanza with a `command = "/usr/local/bin/..."`
      #       that points at the on-disk path.
      artifact {
        # The backend suffix is appended only when `var.backend` is set.
        # See the `variable "backend"` block at the top of this file and
        # README.md "Backend selection" for the three-way trade-off. The
        # backend choice is BUILD-TIME (a Cargo feature flag): different
        # backends ship as different binaries published under different
        # filenames; the runtime env-var surface below does not change.
        source      = "https://example.invalid/tensor-wasm/${var.image_tag}/tensor-wasm-x86_64-linux${var.backend != "" ? "-${var.backend}" : ""}"
        destination = "local/tensor-wasm"
        mode        = "file"

        options {
          # sha256 of the binary. Replace with the real digest published
          # alongside the release (per docs/REPRODUCIBLE-BUILDS.md the
          # release process emits a sha256sums.txt for every artifact).
          # The digest is per-binary, so each backend variant has its
          # own checksum — override on submit:
          #   nomad job run -var checksum=sha256:abc... \
          #     -var backend=cudarc deploy/nomad/tensor-wasm-raw-exec.nomad.hcl
          checksum = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        }
      }

      config {
        command = "local/tensor-wasm"
        args = [
          "serve",
          "--addr",
          "0.0.0.0:8080",
        ]
      }

      # Service-account user. Create on every client node:
      #   useradd --system --shell /usr/sbin/nologin --home /var/lib/tensor-wasm tensor-wasm
      # Without this, raw_exec runs as the Nomad client agent user (root
      # in the default install).
      user = "tensor-wasm"

      # Env vars — same surface as the docker job and the k8s ConfigMap.
      # Keep aligned with deploy/helm/tensor-wasm/values.yaml when adding
      # knobs upstream.
      env {
        TENSOR_WASM_API_LISTEN_ADDR = "0.0.0.0:8080"
        TENSOR_WASM_LOG             = "info"

        TENSOR_WASM_API_RATE_LIMIT_QPS   = "100"
        TENSOR_WASM_API_RATE_LIMIT_BURST = "200"
        TENSOR_WASM_API_REQUIRE_TENANT   = "0"

        # raw_exec has no readonly rootfs; the audit log can write directly
        # to the alloc dir. NOMAD_ALLOC_DIR is expanded at runtime.
        TENSOR_WASM_API_AUDIT_LOG = "stdout"

        TENSOR_WASM_OTLP_ENDPOINT = ""

        CUDA_ARCH      = "sm_80"
        RUST_BACKTRACE = "1"

        # Bearer-token allowlist. Override via the Vault template below
        # on every production deployment.
        TENSOR_WASM_API_TOKENS = "changeme:tenant=*"
      }

      # Vault-managed token rotation. Uncomment and adapt to your Vault
      # layout. The rendered file is sourced as env vars; the task
      # restarts on rotation per change_mode = "restart".
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

      resources {
        # CPU in MHz, memory in MiB. Same defaults as the docker job.
        cpu    = 2000
        memory = 4096

        # GPU device request. nomad-device-nvidia binds the chosen
        # device(s) into the task env (CUDA_VISIBLE_DEVICES); raw_exec
        # inherits the binding directly because no container isolation
        # is in the way. Uncomment when the plugin is installed.
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
