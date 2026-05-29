#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Run the GitHub Actions CI (.github/workflows/ci.yml) locally inside Docker.
#
# Reproduces the Linux jobs — fmt, clippy, test, doc, deny,
# cuda-oxide(-backend-check), openapi, actionlint — in a container that pins
# the same nightly toolchain (rust-toolchain.toml / ci.yml) and the same
# RUSTFLAGS=-D warnings posture. The macOS compile-test job is not reproduced
# (it needs a macOS host).
#
# The Linux build cache lives on dedicated Docker volumes, so it persists
# across runs and never collides with the host's target/ directory (important
# on Windows, where host artifacts are MSVC and the container's are ELF).
#
# Usage:
#   scripts/ci-local.sh [options] [job ...]
#
# Jobs (default: all, run in this order, stopping at the first failure):
#   fmt          cargo fmt --all -- --check
#   clippy       cargo clippy --workspace --all-targets -- -D warnings
#   test         cargo build --workspace
#                cargo test --workspace --no-default-features
#                cargo test --workspace
#   doc          RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --no-default-features
#   deny         cargo deny check --all-features
#   cuda-oxide   cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend
#   openapi      redocly lint + swagger-cli validate + openapi_validation_test
#   actionlint   actionlint
#
# Options:
#   --rebuild-image   Force a rebuild of the local CI image before running.
#   --pull            Pass --pull to the image build (refresh the base image).
#   --keep-going      Run every selected job even if one fails (report at end).
#   --clean-cache     Remove the cargo/target cache volumes, then run.
#   -h | --help       Show this help.
#
# Examples:
#   scripts/ci-local.sh                 # full CI
#   scripts/ci-local.sh fmt clippy      # just the fast lints
#   scripts/ci-local.sh --rebuild-image test
set -euo pipefail

# --- locate the repo root (this script lives in <root>/scripts) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="tensor-wasm-ci:local"
DOCKERFILE="${REPO_ROOT}/docker/ci.Dockerfile"
TARGET_VOLUME="tensor-wasm-ci-target"
CARGO_VOLUME="tensor-wasm-ci-cargo"

# Keep this list aligned with the job dispatch in run_job() below and with
# .github/workflows/ci.yml.
ALL_JOBS=(fmt clippy test doc deny cuda-oxide openapi actionlint)

REBUILD_IMAGE=0
PULL=0
KEEP_GOING=0
CLEAN_CACHE=0
declare -a JOBS=()

usage() { sed -n '2,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild-image) REBUILD_IMAGE=1 ;;
        --pull)          PULL=1 ;;
        --keep-going)    KEEP_GOING=1 ;;
        --clean-cache)   CLEAN_CACHE=1 ;;
        -h|--help)       usage; exit 0 ;;
        fmt|clippy|test|doc|deny|cuda-oxide|openapi|actionlint) JOBS+=("$1") ;;
        all)             JOBS=("${ALL_JOBS[@]}") ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done
[[ ${#JOBS[@]} -eq 0 ]] && JOBS=("${ALL_JOBS[@]}")

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on PATH. Install Docker Desktop / Engine first." >&2
    exit 1
fi

# Stop MSYS/Git-Bash from rewriting the container-side /work path on Windows.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

# --- container-side command for a single job --------------------------------
# Echoed to stdout; executed inside the container by the runner below.
job_cmd() {
    case "$1" in
        fmt)        echo 'cargo fmt --all -- --check' ;;
        clippy)     echo 'cargo clippy --workspace --all-targets -- -D warnings' ;;
        test)       echo 'cargo build --workspace && cargo test --workspace --no-default-features && cargo test --workspace' ;;
        doc)        echo 'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --no-default-features' ;;
        deny)       echo 'cargo deny check --all-features' ;;
        cuda-oxide) echo 'cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend' ;;
        openapi)    echo 'redocly lint --config openapi/redocly.yaml openapi/tensor-wasm-api.yaml && swagger-cli validate crates/tensor-wasm-api/openapi.json && cargo test -p tensor-wasm-api --test openapi_validation_test --no-default-features' ;;
        actionlint) echo 'actionlint' ;;
    esac
}

# --- build the CI image if missing or requested -----------------------------
if [[ "${CLEAN_CACHE}" == "1" ]]; then
    echo ">> removing cache volumes ${TARGET_VOLUME}, ${CARGO_VOLUME}"
    docker volume rm -f "${TARGET_VOLUME}" "${CARGO_VOLUME}" >/dev/null 2>&1 || true
fi

if [[ "${REBUILD_IMAGE}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo ">> building ${IMAGE} (one-time; cached afterwards)"
    build_args=(build -f "${DOCKERFILE}" -t "${IMAGE}")
    [[ "${PULL}" == "1" ]] && build_args+=(--pull)
    build_args+=("${REPO_ROOT}/docker")
    docker "${build_args[@]}"
fi

# --- assemble the in-container program --------------------------------------
# One container, jobs run sequentially so the warm build cache is shared.
container_script='set -uo pipefail; rc=0;'
for job in "${JOBS[@]}"; do
    cmd="$(job_cmd "${job}")"
    container_script+="
printf '\n\033[1;36m===> ci job: %s\033[0m\n' '${job}';
if ${cmd}; then
    printf '\033[1;32m===> %s: OK\033[0m\n' '${job}';
else
    printf '\033[1;31m===> %s: FAILED\033[0m\n' '${job}'; rc=1;"
    if [[ "${KEEP_GOING}" != "1" ]]; then
        container_script+=" exit 1;"
    fi
    container_script+="
fi;"
done
container_script+=' exit $rc;'

echo ">> running CI jobs: ${JOBS[*]}"
docker run --rm -t \
    -v "${REPO_ROOT}:/work" \
    -v "${TARGET_VOLUME}:/cargo-target" \
    -v "${CARGO_VOLUME}:/cargo-home" \
    -e CARGO_TARGET_DIR=/cargo-target \
    -e CARGO_HOME=/cargo-home \
    -e CARGO_TERM_COLOR=always \
    -e RUSTFLAGS="-D warnings" \
    -w /work \
    "${IMAGE}" \
    bash -c "${container_script}"
