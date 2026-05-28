# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Multi-stage Dockerfile producing tensor-wasm-api images for each
# cust-successor backend:
#
#   docker build --build-arg BACKEND=cust       -t tensor-wasm:0.3.6-cust       .
#   docker build --build-arg BACKEND=cudarc     -t tensor-wasm:0.3.6-cudarc     .
#   docker build --build-arg BACKEND=cuda-oxide -t tensor-wasm:0.3.6-cuda-oxide .
#   docker build                                -t tensor-wasm:0.3.6            .  # host-only (no CUDA)
#
# The matching Helm chart `image.backend` value selects the right tag
# at deploy time -- see deploy/helm/tensor-wasm/values.yaml and the
# "Backend selection" section of deploy/helm/tensor-wasm/README.md.
#
# Closes audit Problem #13: the Helm toggle previously referenced
# registry tags that nothing actually built. This Dockerfile is what
# the v0.4 release-engineering workflow will invoke once the
# ghcr.io/craton-co/tensor-wasm registry path is provisioned.
#
# Registry status as of v0.3.6: the ghcr.io path is NOT yet
# provisioned -- operators consuming these images today must run
# `docker build` locally + push to their own registry, or wait for
# the v0.4 release pipeline. See deploy/helm/tensor-wasm/README.md
# "Backend selection" for the operator-facing wording.

ARG BACKEND=""

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM rust:1.84-slim-bookworm AS builder

ARG BACKEND
# Re-declare inside the stage so it's available in RUN.
ARG WORKSPACE_VERSION=0.3.6

# nightly-2026-04-03 matches rust-toolchain.toml + cuda-oxide's pin.
# We install via rustup rather than relying on rust:nightly image so
# the BACKEND-conditional cargo invocation can pick up the toolchain
# components consistently across all four variants.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain nightly-2026-04-03 \
        --component rust-src --component rustc-dev
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

# Per-backend feature selection. The empty BACKEND case ("") is the
# host-only build -- no CUDA features, no GPU linkage. This is what
# `cargo build --release` produces and matches what the CI default
# `ci` workflow tests.
RUN set -eux; \
    case "${BACKEND}" in \
        "")             FEATURES="" ;; \
        cust)           FEATURES="--features unified-memory" ;; \
        cudarc)         FEATURES="--features cudarc-backend" ;; \
        cuda-oxide)     FEATURES="--features cuda-oxide-backend" ;; \
        *) echo "unknown BACKEND='${BACKEND}'; valid: '', cust, cudarc, cuda-oxide" >&2; exit 1 ;; \
    esac; \
    cargo build --release --bin tensor-wasm ${FEATURES}

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

ARG BACKEND
LABEL org.opencontainers.image.title="tensor-wasm" \
      org.opencontainers.image.source="https://github.com/craton-co/craton-tensor-wasm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.craton.tensor-wasm.backend="${BACKEND:-host-only}"

# Non-root runtime user matching the k8s securityContext in
# deploy/k8s/20-deployment.yaml (UID/GID 65532).
RUN groupadd -g 65532 tensor-wasm \
 && useradd -u 65532 -g 65532 -s /usr/sbin/nologin -d /var/lib/tensor-wasm -m tensor-wasm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libcuda1 \
    && rm -rf /var/lib/apt/lists/* \
    || true
# `curl` is required by the HEALTHCHECK below; keeping it in the
# runtime image also makes `kubectl debug` / `docker exec` ergonomic
# for on-the-fly liveness checks against /healthz.
# libcuda1 is present only on hosts with a CUDA driver installed
# (typically via nvidia-container-toolkit injecting the host driver
# into the container at runtime). The `|| true` keeps the build from
# failing on Debian repos that don't carry libcuda1; the runtime
# resolves the symbol lazily on the CUDA path anyway.

COPY --from=builder /build/target/release/tensor-wasm /usr/local/bin/tensor-wasm

USER 65532:65532
WORKDIR /var/lib/tensor-wasm

EXPOSE 8080

ENV TENSOR_WASM_API_LISTEN_ADDR="0.0.0.0:8080" \
    RUST_BACKTRACE="1"

HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/tensor-wasm"]
CMD ["serve"]
