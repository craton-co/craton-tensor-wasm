#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Quiet-mode bench driver -- target: CV < 5% per docs/BENCHMARKING.md
# anti-cheating checklist.
#
# What this does:
#   1. Pins the CPU governor to performance (best-effort; needs sudo)
#   2. Drops the page cache before each Criterion run
#   3. Runs each bench at 3-5x the default sample count
#   4. Writes results into bench-results/quiet/
#
# What this does NOT do (true scientific-bar):
#   - Disable Turbo Boost (operator-specific, varies by SKU)
#   - taskset to isolated cores (kernel boot param isolcpus=N required)
#   - Disable SMT / Hyper-Threading
#   - Throttle background daemons
# These are documented in docs/BENCHMARKING.md "Hardware and OS
# normalization"; the quiet script is a usable-but-honest middle
# ground for noise reduction without root-level OS reconfiguration.
#
# Usage:
#   bash scripts/run-quiet-bench.sh                  # all groups
#   bash scripts/run-quiet-bench.sh tail_latency     # one group
#
# Closes audit Problem #9 partially: the committed bench-results
# numbers (baseline.json, tail-latency.json, dispatch-future-
# backends.json) were captured with default Criterion sample count
# and high background noise (CV typically > 5%). Re-running with
# this script + on a quieter host should produce publishable numbers.

set -euo pipefail

# Run from repo root so relative paths (bench-results/, target/) resolve
# the same regardless of where the operator invoked the script from.
cd "$(git rev-parse --show-toplevel)"

OUT_DIR="bench-results/quiet"
mkdir -p "$OUT_DIR"

BENCH_FILTER="${1:-}"

# Compute the baseline stamp *once* per session, not per-bench. With the
# stamp inside the per-bench loop each Criterion bench in one run lands
# in a different baseline dir (target/criterion/<bench>/quiet-<ts>/),
# which defeats `--save-baseline` for cross-bench comparison: you cannot
# `cargo bench -- --baseline quiet-<ts>` against a single name. Hoisting
# the stamp here gives every bench in this invocation the same baseline
# tag so they form a coherent set.
STAMP="$(date +%Y%m%d-%H%M%S)"

# -- step 1: CPU governor (Linux only, best-effort) -------------------
if command -v cpupower >/dev/null 2>&1 && [ "$(uname -s)" = "Linux" ]; then
    echo "[quiet] pinning CPU governor to performance (sudo required)"
    sudo cpupower frequency-set -g performance >/dev/null || \
        echo "[quiet] governor pin failed (non-fatal)"
else
    echo "[quiet] cpupower not available -- skipping governor pin"
fi

# -- step 2: page cache drop ------------------------------------------
drop_cache() {
    if [ "$(uname -s)" = "Linux" ]; then
        sync
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
            echo "[quiet] page-cache drop failed (non-fatal)"
    fi
}

# -- step 3: per-bench runs at elevated sample count ------------------
# CRITERION_DEBUG_DRAWING off; CRITERION_SAMPLE_COUNT raises sample
# count from default 100 to 500 (5x). measurement_time uplift
# (default 5s -> 15s) is per-bench code, not env-controlled, so the
# committed bench files (W4.6 tail_latency, F3 dispatch_future_backends
# etc.) already pick the right value; nothing to override here.
export CARGO_TERM_COLOR=always
ELEVATED_SAMPLES=500

run_one() {
    local name="$1"
    # dispatch_future_backends is only meaningful with --features cuda: the
    # default build skips both backends (see bench-results/README.md). Pass
    # the feature so the busy-poll path actually exercises the DispatchFuture
    # loop rather than emitting two skip lines.
    local extra_args=()
    if [ "$name" = "dispatch_future_backends" ]; then
        extra_args=(--features cuda)
    fi
    echo "[quiet] -- bench: $name (samples=$ELEVATED_SAMPLES)"
    drop_cache
    cargo bench -p tensor-wasm-bench --bench "$name" "${extra_args[@]}" -- \
        --sample-size "$ELEVATED_SAMPLES" \
        --save-baseline "quiet-$STAMP" \
        2>&1 | tee "$OUT_DIR/${name}.log"
}

BENCHES=(
    cold_start
    e2e_inference
    jit_compile
    kernel_dispatch
    memory_bandwidth
    tail_latency
    dispatch_future_backends
    metrics_label_validation
    call_export_args
    streaming_invoke
)

if [ -n "$BENCH_FILTER" ]; then
    BENCHES=("$BENCH_FILTER")
fi

for b in "${BENCHES[@]}"; do
    run_one "$b"
done

echo "[quiet] done. Logs in $OUT_DIR/. Criterion baselines under target/criterion/."
echo "[quiet] inspect target/criterion/report/index.html for CV per metric."
