#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Run the GitHub Actions CI (.github/workflows/ci.yml) locally inside Docker.
#
# Reproduces the Linux jobs — fmt, clippy, test, doc, deny,
# cuda-oxide(-backend-check), openapi, actionlint — in containers that pin
# the same nightly toolchain (rust-toolchain.toml / ci.yml) and the same
# RUSTFLAGS=-D warnings posture. The macOS compile-test job is not reproduced
# (it needs a macOS host).
#
# Parallel layout (default): two containers run AT THE SAME TIME —
#   * lane "tests"  — the `test` job only (build + both test passes).
#   * lane "checks" — every other selected job (fmt, clippy, doc, deny,
#                     cuda-oxide, openapi, actionlint), run sequentially.
# The tests pass is the long pole, so overlapping it with the lint/doc/audit
# battery roughly halves wall-clock on a multi-core host. The two lanes use
# SEPARATE target-cache volumes (concurrent cargo on one target dir would
# lock-contend and serialise); they share the registry/git cache volume.
# Pass --serial to fall back to a single container with one shared cache.
#
# The Linux build cache lives on dedicated Docker volumes, so it persists
# across runs and never collides with the host's target/ directory (important
# on Windows, where host artifacts are MSVC and the container's are ELF).
#
# Usage:
#   scripts/ci-local.sh [options] [job ...]
#
# Jobs (default: all; every job runs even if an earlier one fails, and the
# script exits non-zero if any failed — see --fail-fast):
#   fmt          cargo fmt --all -- --check
#   clippy       cargo clippy --workspace --all-targets -- -D warnings
#   test         cargo build --workspace
#                cargo test --workspace --no-default-features
#                cargo test --workspace
#   doc          RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --no-default-features
#   deny         cargo deny --all-features check
#   cuda-oxide   cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend
#   openapi      redocly lint + swagger-cli validate + openapi_validation_test
#   actionlint   actionlint
#
# The local CI image is rebuilt on EVERY run by default (Docker layer cache
# keeps it near-instant when docker/ci.Dockerfile is unchanged, and any edit —
# e.g. a pinned tool-version bump — is picked up automatically). Pass
# --skip-image-build to reuse the existing image as-is.
#
# Options:
#   --serial          Run all jobs sequentially in ONE container (old
#                     behaviour; one shared warm cache, no parallelism).
#   --skip-image-build  Reuse the existing CI image; skip the default rebuild.
#   --rebuild-image     Accepted for compatibility; rebuild is now the default.
#   --pull            Pass --pull to the image build (refresh the base image).
#   --fail-fast       Stop a lane at its first failing job (default: run all).
#   --keep-going      Accepted for compatibility; now the default (no-op).
#   --clean-cache     Remove the cargo/target cache volumes, then run.
#   -h | --help       Show this help.
#
# Examples:
#   scripts/ci-local.sh                 # full CI, tests || checks in parallel
#   scripts/ci-local.sh fmt clippy      # just the fast lints (checks lane only)
#   scripts/ci-local.sh --serial        # old single-container behaviour
#   scripts/ci-local.sh --skip-image-build test   # reuse the cached image
set -euo pipefail

# --- locate the repo root (this script lives in <root>/scripts) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="tensor-wasm-ci:local"
DOCKERFILE="${REPO_ROOT}/docker/ci.Dockerfile"
# Per-lane target caches (parallel mode) + a single shared one for --serial.
TARGET_VOLUME_TESTS="tensor-wasm-ci-target-tests"
TARGET_VOLUME_CHECKS="tensor-wasm-ci-target-checks"
TARGET_VOLUME_SERIAL="tensor-wasm-ci-target"
CARGO_VOLUME="tensor-wasm-ci-cargo"

# On Windows/MSYS, MSYS_NO_PATHCONV=1 (exported below) stops Git-Bash from
# auto-converting Unix-style paths to Windows paths.  Docker needs native
# Windows paths for the build context, Dockerfile flag, and volume source,
# so convert them explicitly with cygpath when available.
if command -v cygpath >/dev/null 2>&1; then
    REPO_ROOT_WIN="$(cygpath -w "${REPO_ROOT}")"
    DOCKERFILE_WIN="$(cygpath -w "${DOCKERFILE}")"
else
    REPO_ROOT_WIN="${REPO_ROOT}"
    DOCKERFILE_WIN="${DOCKERFILE}"
fi

# Keep this list aligned with the job dispatch in job_cmd() below and with
# .github/workflows/ci.yml.
ALL_JOBS=(fmt clippy test doc deny cuda-oxide openapi actionlint)

SKIP_IMAGE_BUILD=0
PULL=0
# Run every job even if an earlier one fails (the script still exits non-zero
# when any failed). `--fail-fast` restores stop-at-first-failure (per lane).
FAIL_FAST=0
CLEAN_CACHE=0
SERIAL=0
declare -a JOBS=()

usage() { sed -n '2,65p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial)        SERIAL=1 ;;
        --skip-image-build) SKIP_IMAGE_BUILD=1 ;;
        --rebuild-image) : ;;  # rebuild is now the default; accepted for compatibility
        --pull)          PULL=1 ;;
        --fail-fast)     FAIL_FAST=1 ;;
        --keep-going)    : ;;  # now the default; accepted for compatibility
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
# Echoed to stdout; spliced into the per-lane container script below.
job_cmd() {
    case "$1" in
        fmt)        echo 'cargo fmt --all -- --check' ;;
        clippy)     echo 'cargo clippy --workspace --all-targets -- -D warnings' ;;
        test)       echo 'cargo build --workspace && cargo test --workspace --no-default-features && cargo test --workspace' ;;
        doc)        echo 'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --no-default-features' ;;
        deny)       echo 'cargo deny --all-features check' ;;
        cuda-oxide) echo 'cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend' ;;
        openapi)    echo 'redocly lint --config openapi/redocly.yaml openapi/tensor-wasm-api.yaml && swagger-cli validate crates/tensor-wasm-api/openapi.json && cargo test -p tensor-wasm-api --test openapi_validation_test --no-default-features' ;;
        actionlint) echo 'actionlint' ;;
    esac
}

# --- assemble the in-container program for a set of jobs --------------------
# `failed` accumulates the names of jobs that failed so the lane tail can print
# a single summary; the lane exits non-zero if any job failed.
make_container_script() {
    local script='set -uo pipefail; rc=0; failed="";'
    local job cmd
    for job in "$@"; do
        cmd="$(job_cmd "${job}")"
        script+="
printf '\n\033[1;36m===> ci job: %s\033[0m\n' '${job}';
if ${cmd}; then
    printf '\033[1;32m===> %s: OK\033[0m\n' '${job}';
else
    printf '\033[1;31m===> %s: FAILED\033[0m\n' '${job}'; rc=1; failed=\"\${failed} ${job}\";"
        if [[ "${FAIL_FAST}" == "1" ]]; then
            script+=" exit 1;"
        fi
        script+="
fi;"
    done
    script+='
if [[ -n "${failed}" ]]; then
    printf "\n\033[1;31m===> lane FAILED:%s\033[0m\n" "${failed}";
else
    printf "\n\033[1;32m===> lane OK (all jobs passed)\033[0m\n";
fi;
exit $rc;'
    printf '%s' "${script}"
}

# --- cache reset ------------------------------------------------------------
if [[ "${CLEAN_CACHE}" == "1" ]]; then
    echo ">> removing cache volumes ${TARGET_VOLUME_TESTS}, ${TARGET_VOLUME_CHECKS}, ${TARGET_VOLUME_SERIAL}, ${CARGO_VOLUME}"
    docker volume rm -f "${TARGET_VOLUME_TESTS}" "${TARGET_VOLUME_CHECKS}" \
        "${TARGET_VOLUME_SERIAL}" "${CARGO_VOLUME}" >/dev/null 2>&1 || true
fi

# --- build the CI image if missing or requested -----------------------------
if [[ "${SKIP_IMAGE_BUILD}" == "0" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo ">> building ${IMAGE} (Docker layer cache makes this near-instant when docker/ci.Dockerfile is unchanged)"
    build_args=(build -f "${DOCKERFILE_WIN}" -t "${IMAGE}")
    [[ "${PULL}" == "1" ]] && build_args+=(--pull)
    build_args+=("${REPO_ROOT_WIN}\\docker")
    docker "${build_args[@]}"
fi

# --- run a single lane (container) ------------------------------------------
# Backgrounded; output is streamed live, prefixed with the lane name; the
# container's exit code is written to a per-lane file so the caller can tally
# results after `wait`.
LANE_TMP="$(mktemp -d)"
trap 'rm -rf "${LANE_TMP}"' EXIT
declare -a LANE_PIDS=() LANE_NAMES=() LANE_RCFILES=()

start_lane() {
    local name="$1" color="$2" target_vol="$3"; shift 3
    local jobs=("$@")
    [[ ${#jobs[@]} -eq 0 ]] && return 0
    local rcfile="${LANE_TMP}/${name}.rc"
    local script; script="$(make_container_script "${jobs[@]}")"
    echo ">> lane [${name}] (vol ${target_vol}) jobs: ${jobs[*]}"
    {
        docker run --rm \
            -v "${REPO_ROOT_WIN}:/work" \
            -v "${target_vol}:/cargo-target" \
            -v "${CARGO_VOLUME}:/cargo-home" \
            -e CARGO_TARGET_DIR=/cargo-target \
            -e CARGO_HOME=/cargo-home \
            -e CARGO_TERM_COLOR=always \
            -e RUSTFLAGS="-D warnings" \
            -w /work \
            "${IMAGE}" \
            bash -c "${script}"
        echo "$?" > "${rcfile}"
    } 2>&1 | while IFS= read -r line; do
        printf '%b[%s]\033[0m %s\n' "${color}" "${name}" "${line}"
    done &
    LANE_PIDS+=("$!")
    LANE_NAMES+=("${name}")
    LANE_RCFILES+=("${rcfile}")
}

if [[ "${SERIAL}" == "1" ]]; then
    # Old behaviour: every selected job in one container, one shared cache.
    echo ">> running CI jobs serially (one container): ${JOBS[*]}"
    start_lane "ci" '\033[1;36m' "${TARGET_VOLUME_SERIAL}" "${JOBS[@]}"
else
    # Partition selected jobs: the `test` job goes to the tests lane, all
    # others to the checks lane. Both lanes run in parallel.
    declare -a TEST_JOBS=() REST_JOBS=()
    for j in "${JOBS[@]}"; do
        if [[ "${j}" == "test" ]]; then TEST_JOBS+=("${j}"); else REST_JOBS+=("${j}"); fi
    done
    echo ">> running CI in parallel containers: [tests] || [checks]"
    [[ ${#TEST_JOBS[@]} -gt 0 ]] && start_lane "tests"  '\033[1;35m' "${TARGET_VOLUME_TESTS}"  "${TEST_JOBS[@]}"
    [[ ${#REST_JOBS[@]} -gt 0 ]] && start_lane "checks" '\033[1;34m' "${TARGET_VOLUME_CHECKS}" "${REST_JOBS[@]}"
fi

# --- wait for all lanes, then tally -----------------------------------------
for pid in "${LANE_PIDS[@]}"; do
    wait "${pid}" || true
done

overall=0
summary=""
for i in "${!LANE_NAMES[@]}"; do
    name="${LANE_NAMES[$i]}"
    rc="$(cat "${LANE_RCFILES[$i]}" 2>/dev/null || echo 1)"
    if [[ "${rc}" == "0" ]]; then
        summary+=$'\n'"   [${name}] OK"
    else
        summary+=$'\n'"   [${name}] FAILED (rc=${rc})"
        overall=1
    fi
done

if [[ "${overall}" == "0" ]]; then
    printf '\n\033[1;32m===> CI OK (all lanes passed)\033[0m%s\n' "${summary}"
else
    printf '\n\033[1;31m===> CI FAILED\033[0m%s\n' "${summary}"
fi
exit "${overall}"
