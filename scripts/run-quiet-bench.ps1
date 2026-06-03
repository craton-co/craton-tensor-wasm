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
# What this does NOT do (true scientific-bar, Windows-specific
# additions over the bash version's list):
#   - Drop the page cache. Windows has no equivalent of Linux's
#     /proc/sys/vm/drop_caches; the working-set / standby list cannot
#     be flushed from userspace without third-party tools (EmptyStandbyList
#     / RAMMap). The bash version sync+drops between benches; this script
#     cannot, so cold-cache numbers on Windows are weaker. Reboot between
#     publishable runs if cold-start behaviour matters.
#   - Stop / exclude Windows Defender real-time scanning. Defender
#     scanning of cargo's target/ tree is a measurable source of jitter.
#     Run `Add-MpPreference -ExclusionPath $PWD` (admin shell) manually
#     before invoking this script for publishable numbers.
#   - Stop IDE / language-server indexing (VS Code, Rider, rust-analyzer
#     standalone). Close the editor before benching for tightest CV.
#   - Disable Hyper-Threading.
#   - taskset equivalent. Set-ProcessorAffinity exists but requires
#     admin and is per-process not per-cargo-invocation friendly.
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

# Run from repo root so relative paths (bench-results\, target\) resolve
# the same regardless of where the operator invoked the script from.
Set-Location (git rev-parse --show-toplevel)

$OutDir = "bench-results\quiet"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Compute the baseline stamp *once* per session, not per-bench. With the
# Get-Date call inside Invoke-Bench, each Criterion bench in one run
# landed in a different baseline dir (target\criterion\<bench>\quiet-<ts>\),
# which defeats `--save-baseline` for cross-bench comparison: you cannot
# `cargo bench -- --baseline quiet-<ts>` against a single name. Capture
# the stamp here at script entry so every bench in this invocation shares
# one baseline tag.
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"

# Step 1: High-performance power plan (8c5e7fda... is the well-known GUID)
$HighPerfGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
Write-Host "[quiet] setting active power scheme to High performance"
powercfg /setactive $HighPerfGuid

# Step 2: per-bench runs
$ElevatedSamples = 500
$env:CARGO_TERM_COLOR = "always"

function Invoke-Bench($Name) {
    Write-Host "[quiet] -- bench: $Name (samples=$ElevatedSamples)"
    # Use $script:Stamp computed once at script entry so every bench in
    # this run shares the same `quiet-<ts>` baseline tag.
    $log = Join-Path $OutDir "$Name.log"
    # dispatch_future_backends is only meaningful with --features cuda: the
    # default build skips both backends (see bench-results\README.md). Pass
    # the feature so the busy-poll path actually exercises the DispatchFuture
    # loop rather than emitting two skip lines.
    $ExtraArgs = @()
    if ($Name -eq "dispatch_future_backends") {
        $ExtraArgs = @("--features", "cuda")
    }
    & cargo bench -p tensor-wasm-bench --bench $Name @ExtraArgs -- `
        --sample-size $ElevatedSamples `
        --save-baseline "quiet-$script:Stamp" 2>&1 | Tee-Object -FilePath $log
}

$Benches = @(
    "cold_start",
    "e2e_inference",
    "jit_compile",
    "kernel_dispatch",
    "memory_bandwidth",
    "tail_latency",
    "dispatch_future_backends",
    "metrics_label_validation",
    "call_export_args",
    "streaming_invoke"
)

if ($BenchFilter -ne "") {
    $Benches = @($BenchFilter)
}

foreach ($b in $Benches) {
    Invoke-Bench $b
}

Write-Host "[quiet] done. Logs in $OutDir\. Criterion baselines under target\criterion\."
Write-Host "[quiet] inspect target\criterion\report\index.html for CV per metric."
