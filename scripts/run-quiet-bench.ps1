# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Windows-side quiet-mode bench driver. Equivalent of the bash script
# (scripts/run-quiet-bench.sh) for hosts where CI/dev runs on
# nightly-2026-04-03 + PowerShell rather than bash.
#
# What this does:
#   1. Sets the active power scheme to "High performance" (no sudo needed)
#   2. Runs each bench at elevated sample count (--sample-size 500)
#   3. Writes logs into bench-results/quiet/
#
# What this does NOT do:
#   - Disable Hyper-Threading / Defender real-time scan / IDE indexing
#   - taskset equivalent (Set-ProcessorAffinity exists but requires
#     admin and is per-process not per-cargo-invocation friendly)
# These match the bash version's "not done" list -- the doc bar in
# docs/BENCHMARKING.md "Hardware and OS normalization" still applies
# for publishable numbers.
#
# Usage:
#   pwsh scripts/run-quiet-bench.ps1                    # all groups
#   pwsh scripts/run-quiet-bench.ps1 tail_latency       # one group

param(
    [string]$BenchFilter = ""
)

$ErrorActionPreference = "Stop"
$OutDir = "bench-results\quiet"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Step 1: High-performance power plan (8c5e7fda... is the well-known GUID)
$HighPerfGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
Write-Host "[quiet] setting active power scheme to High performance"
powercfg /setactive $HighPerfGuid

# Step 2: per-bench runs
$ElevatedSamples = 500
$env:CARGO_TERM_COLOR = "always"

function Invoke-Bench($Name) {
    Write-Host "[quiet] -- bench: $Name (samples=$ElevatedSamples)"
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $log = Join-Path $OutDir "$Name.log"
    & cargo bench -p tensor-wasm-bench --bench $Name -- `
        --sample-size $ElevatedSamples `
        --save-baseline "quiet-$stamp" 2>&1 | Tee-Object -FilePath $log
}

$Benches = @(
    "cold_start",
    "e2e_inference",
    "jit_compile",
    "kernel_dispatch",
    "memory_bandwidth",
    "tail_latency",
    "dispatch_future_backends"
)

if ($BenchFilter -ne "") {
    $Benches = @($BenchFilter)
}

foreach ($b in $Benches) {
    Invoke-Bench $b
}

Write-Host "[quiet] done. Logs in $OutDir\. Criterion baselines under target\criterion\."
Write-Host "[quiet] inspect target\criterion\report\index.html for CV per metric."
