# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Reproducible local hardware-verification runner (PowerShell) — the loop the
# dormant .github/workflows/gpu.yml describes, runnable on a Windows CUDA dev
# box with the CUDA Toolkit + a pip-installed libclang.
#
# Exercises the real-GPU ignored tests across every backend feature and the
# --features cuda benches, writing per-suite logs to bench-results\gpu-run\.
#
# Prerequisites (auto-detected; fails loudly if missing):
#   * NVIDIA driver + nvidia-smi on PATH.
#   * CUDA Toolkit. Honors $env:CUDA_PATH, else probes the standard install dir.
#   * libclang (cust_raw bindgen). Honors $env:LIBCLANG_PATH, else locates the
#     pip `libclang` package DLL (`pip install --user libclang`).
#
# Usage:
#   scripts\run-gpu-tests.ps1            # all suites
#   scripts\run-gpu-tests.ps1 mem        # tensor-wasm-mem only
#   scripts\run-gpu-tests.ps1 wasi       # tensor-wasm-wasi-gpu only
#   scripts\run-gpu-tests.ps1 jit        # differential-oracle only
#   scripts\run-gpu-tests.ps1 bench      # --features cuda benches only
#
# GPU tests run single-threaded (--test-threads=1): cust 0.3's primary-context
# model does not survive parallel test execution.

param([string]$What = "all")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$outDir = Join-Path $root "bench-results\gpu-run"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Log($m) { Write-Host "[gpu-tests] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[gpu-tests] ERROR: $m" -ForegroundColor Red }

# --- CUDA toolkit ---
if (-not $env:CUDA_PATH) {
  $cand = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" -Directory -ErrorAction SilentlyContinue |
          Sort-Object Name -Descending | Select-Object -First 1
  if ($cand) { $env:CUDA_PATH = $cand.FullName }
}
if (-not $env:CUDA_PATH) { Fail "CUDA toolkit not found; set CUDA_PATH"; exit 2 }
if (-not $env:CUDA_ROOT) { $env:CUDA_ROOT = $env:CUDA_PATH }
Log "CUDA_PATH = $($env:CUDA_PATH)"

# --- libclang (cust_raw bindgen) ---
if (-not $env:LIBCLANG_PATH) {
  $dll = Get-ChildItem -Path "$env:LOCALAPPDATA","$env:APPDATA" -Recurse -Filter libclang.dll -ErrorAction SilentlyContinue |
         Where-Object { $_.FullName -match 'clang\\native\\libclang.dll' } | Select-Object -First 1
  if (-not $dll) { $dll = Get-ChildItem "C:\Program Files\LLVM\bin\libclang.dll" -ErrorAction SilentlyContinue | Select-Object -First 1 }
  if ($dll) { $env:LIBCLANG_PATH = $dll.DirectoryName }
}
if (-not $env:LIBCLANG_PATH) {
  Fail "libclang not found. Install it with: pip install --user libclang  (or set LIBCLANG_PATH)"
  exit 2
}
Log "LIBCLANG_PATH = $($env:LIBCLANG_PATH)"

# --- driver sanity ---
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi not on PATH"; exit 2 }
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader |
  Tee-Object -FilePath (Join-Path $outDir "gpu-info.txt")

$rc = 0
function Run($name, $cargoArgs) {
  Log "RUN ${name}: cargo $($cargoArgs -join ' ')"
  & cargo @cargoArgs 2>&1 | Tee-Object -FilePath (Join-Path $outDir "$name.log")
  if ($LASTEXITCODE -ne 0) { $script:rc = 1; Fail "$name FAILED (exit $LASTEXITCODE)" }
}

$testTail = @("--","--include-ignored","--test-threads=1")
if ($What -in @("mem","all")) {
  Run "mem-unified-memory" (@("test","-p","tensor-wasm-mem","--release","--features","unified-memory") + $testTail)
  Run "mem-gpu-mem-pool"   (@("test","-p","tensor-wasm-mem","--release","--features","gpu-mem-pool") + $testTail)
  Run "mem-cuda-oxide"     (@("test","-p","tensor-wasm-mem","--release","--features","cuda-oxide-backend") + $testTail)
}
if ($What -in @("wasi","all")) {
  Run "wasi-gpu-cuda" (@("test","-p","tensor-wasm-wasi-gpu","--release","--features","cuda") + $testTail)
}
if ($What -in @("jit","all")) {
  Run "jit-differential" (@("test","-p","tensor-wasm-jit","--release","--features","differential-oracle") + $testTail)
}
if ($What -in @("bench","all")) {
  Log "RUN cuda-benches (smoke)"
  # `--benches` restricts to criterion bench targets; without it `cargo bench`
  # also runs the lib unittest binary, which rejects criterion's
  # `--warm-up-time`/`--measurement-time` flags ("Unrecognized option").
  & cargo bench -p tensor-wasm-bench --features cuda --benches -- --warm-up-time 3 --measurement-time 5 2>&1 |
    Tee-Object -FilePath (Join-Path $outDir "cuda-benches.log")
  if ($LASTEXITCODE -ne 0) { $rc = 1; Fail "cuda-benches FAILED" }
}

if ($rc -eq 0) { Log "ALL GPU SUITES PASSED. Logs in $outDir" } else { Fail "one or more suites failed; see $outDir" }
exit $rc
