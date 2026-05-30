#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Reproducible local hardware-verification runner — the loop the dormant
# .github/workflows/gpu.yml describes, runnable on any CUDA box (incl. a
# Windows dev box with the CUDA Toolkit + a pip-installed libclang).
#
# It exercises the real-GPU ignored tests across every backend feature and
# the --features cuda benches, exactly as the S22 self-hosted runner would,
# and writes per-suite logs under bench-results/gpu-run/.
#
# Prerequisites it auto-detects (and fails loudly if missing):
#   * NVIDIA driver + nvidia-smi on PATH.
#   * CUDA Toolkit (cust links cuda.lib; cudarc dlopens the driver). Honors
#     $CUDA_PATH / $CUDA_ROOT, else probes the standard Windows install dir.
#   * libclang (cust_raw's bindgen needs it). Honors $LIBCLANG_PATH, else
#     locates the pip `libclang` package's bundled DLL
#     (`pip install --user libclang`).
#
# Usage:
#   scripts/run-gpu-tests.sh             # all suites
#   scripts/run-gpu-tests.sh mem         # tensor-wasm-mem only
#   scripts/run-gpu-tests.sh wasi        # tensor-wasm-wasi-gpu only
#   scripts/run-gpu-tests.sh jit         # differential-oracle only
#   scripts/run-gpu-tests.sh bench       # --features cuda benches only
#
# All GPU tests run single-threaded (--test-threads=1): cust 0.3's
# primary-context model does not survive parallel test execution.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/bench-results/gpu-run"
mkdir -p "$OUT"

log() { printf '\033[1;36m[gpu-tests]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[gpu-tests] ERROR:\033[0m %s\n' "$*" >&2; }

# --- CUDA toolkit ----------------------------------------------------------
if [ -z "${CUDA_PATH:-}" ]; then
  for d in "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"/v*; do
    [ -d "$d" ] && CUDA_PATH="$(cygpath -w "$d" 2>/dev/null || echo "$d")"
  done
fi
if [ -z "${CUDA_PATH:-}" ]; then err "CUDA toolkit not found; set CUDA_PATH"; exit 2; fi
export CUDA_PATH
export CUDA_ROOT="${CUDA_ROOT:-$CUDA_PATH}"
log "CUDA_PATH = $CUDA_PATH"

# --- libclang (cust_raw bindgen) ------------------------------------------
if [ -z "${LIBCLANG_PATH:-}" ]; then
  LC="$(find "$HOME/AppData" /usr "$HOME/.local" -ipath '*clang/native/libclang.dll' 2>/dev/null | head -1)"
  [ -z "$LC" ] && LC="$(find '/c/Program Files/LLVM' -iname libclang.dll 2>/dev/null | head -1)"
  if [ -n "$LC" ]; then
    LIBCLANG_PATH="$(cygpath -w "$(dirname "$LC")" 2>/dev/null || dirname "$LC")"
  fi
fi
if [ -z "${LIBCLANG_PATH:-}" ]; then
  err "libclang not found. Install it with: pip install --user libclang"
  err "or set LIBCLANG_PATH to a directory containing libclang.dll/.so."
  exit 2
fi
export LIBCLANG_PATH
log "LIBCLANG_PATH = $LIBCLANG_PATH"

# --- driver sanity ---------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then err "nvidia-smi not on PATH"; exit 2; fi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader \
  | tee "$OUT/gpu-info.txt"

CARGO_FLAGS="--release"
TEST_FLAGS="-- --include-ignored --test-threads=1"
rc=0
run() { # name, cargo args...
  local name="$1"; shift
  log "RUN $name: cargo $*"
  if cargo "$@" 2>&1 | tee "$OUT/$name.log"; then :; else rc=1; err "$name FAILED"; fi
}

what="${1:-all}"
case "$what" in
  mem|all)
    run mem-unified-memory  test -p tensor-wasm-mem $CARGO_FLAGS --features unified-memory  $TEST_FLAGS
    run mem-gpu-mem-pool    test -p tensor-wasm-mem $CARGO_FLAGS --features gpu-mem-pool     $TEST_FLAGS
    run mem-cuda-oxide      test -p tensor-wasm-mem $CARGO_FLAGS --features cuda-oxide-backend $TEST_FLAGS
    ;;& # fall through to next matching pattern when "all"
  wasi|all)
    run wasi-gpu-cuda       test -p tensor-wasm-wasi-gpu $CARGO_FLAGS --features cuda $TEST_FLAGS
    ;;&
  jit|all)
    run jit-differential    test -p tensor-wasm-jit $CARGO_FLAGS --features differential-oracle $TEST_FLAGS
    ;;&
  bench|all)
    log "RUN cuda-benches (smoke): cargo bench -p tensor-wasm-bench --features cuda"
    # `--benches` restricts to the criterion bench targets; without it `cargo
    # bench` also runs the lib unittest binary, which rejects criterion's
    # `--warm-up-time`/`--measurement-time` flags ("Unrecognized option").
    if cargo bench -p tensor-wasm-bench --features cuda --benches -- --warm-up-time 3 --measurement-time 5 \
        2>&1 | tee "$OUT/cuda-benches.log"; then :; else rc=1; err "cuda-benches FAILED"; fi
    ;;
esac

if [ "$rc" -eq 0 ]; then log "ALL GPU SUITES PASSED. Logs in $OUT"; else err "one or more suites failed; see $OUT"; fi
exit "$rc"
