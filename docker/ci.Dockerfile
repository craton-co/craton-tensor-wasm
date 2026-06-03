# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Local-CI image: reproduces the Linux jobs of .github/workflows/ci.yml
# (fmt, clippy, test, doc, deny, cuda-oxide-backend-check, openapi,
# actionlint) so contributors can run "the CI" locally before pushing
# without waiting on GitHub-hosted runners.
#
# Driven by scripts/ci-local.sh and scripts/ci-local.ps1 — you normally
# do NOT build this by hand; the wrapper builds and caches it for you.
#
# The `macos-compile-test` CI job is intentionally NOT reproduced here:
# it needs a macOS host and cannot run in a Linux container.
#
# Manual build (the wrapper does this automatically, cached after the
# first run):
#
#   docker build -f docker/ci.Dockerfile -t tensor-wasm-ci:local docker
#
# KEEP IN LOCKSTEP with CI when bumping versions:
#   - RUST_NIGHTLY        -> rust-toolchain.toml `channel` AND every
#                            `toolchain:` field in .github/workflows/ci.yml
#   - NODE_MAJOR          -> ci.yml openapi job `setup-node` node-version
#   - CARGO_DENY_VERSION  -> tracks the cargo-deny-action used by ci.yml `deny`
#   - ACTIONLINT_VERSION  -> tracks reviewdog/action-actionlint used by ci.yml

FROM rust:1.84-slim-bookworm@sha256:f1c6e953d9cfe4bd8eb4512a82647ef68965484714eb63d925b955458a357133

# Keep in lockstep with rust-toolchain.toml + every ci.yml `toolchain:` field.
ARG RUST_NIGHTLY=nightly-2026-04-03
ARG NODE_MAJOR=20
# Must be new enough to parse edition-2024 cargo metadata: 0.16.1 chokes with
# "unknown variant 2024" once any crate in the graph uses edition 2024. Built
# with the nightly default set above, so cargo-deny's own MSRV is satisfied.
ARG CARGO_DENY_VERSION=0.19.0
ARG ACTIONLINT_VERSION=1.7.7

ENV DEBIAN_FRONTEND=noninteractive

# System deps: build toolchain + git/pkg-config (cargo build), curl/ca-certs
# (rustup + downloads), and Node.js (redocly / swagger-cli for the openapi job).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git pkg-config build-essential xz-utils \
    && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pinned nightly + the exact component set ci.yml requests. The base image
# ships a stable toolchain; we add and default to the workspace nightly so
# `cargo` (a rustup proxy) resolves it regardless of rust-toolchain.toml.
RUN rustup toolchain install "${RUST_NIGHTLY}" \
        --component rustfmt --component clippy \
        --component rust-src --component llvm-tools-preview \
    && rustup default "${RUST_NIGHTLY}"

# cargo-deny for the `deny` job. Installed to /usr/local (NOT $CARGO_HOME) so
# the binary survives the wrapper mounting a cache volume over CARGO_HOME.
RUN cargo install cargo-deny --version "${CARGO_DENY_VERSION}" --locked --root /usr/local \
    && rm -rf "${CARGO_HOME:-/usr/local/cargo}/registry"

# OpenAPI linters (ci.yml openapi job) + actionlint (ci.yml actionlint job).
RUN npm install --global @redocly/cli@1 @apidevtools/swagger-cli@4 \
    && curl -fsSL \
        "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz" \
        | tar -xz -C /usr/local/bin actionlint \
    && actionlint --version

# CUDA stub libraries — mirrors the "Install CUDA stub libraries" step shared
# by the clippy/test/doc/openapi/cuda-oxide jobs so default-feature crates
# that probe for libcuda resolve at build time. Real CUDA is never exercised
# in CI (no GPU on hosted runners); these are empty placeholder objects, same
# as the `touch` the workflow performs.
RUN mkdir -p /usr/local/cuda/lib64 \
    && : > /usr/local/cuda/lib64/libcuda.so \
    && : > /usr/local/cuda/lib64/libcudart.so
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    CUDA_ROOT=/usr/local/cuda

WORKDIR /work

# The wrapper script overrides CARGO_HOME / CARGO_TARGET_DIR onto named
# volumes at run time so the Linux build cache persists across runs and never
# collides with the host's (possibly Windows) target/ directory.
CMD ["bash"]
