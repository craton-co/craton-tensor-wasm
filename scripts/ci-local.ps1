# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Run the GitHub Actions CI (.github/workflows/ci.yml) locally inside Docker.
#
# Windows-native (PowerShell) counterpart of scripts/ci-local.sh. Reproduces
# the Linux CI jobs — fmt, clippy, test, doc, deny, cuda-oxide(-backend-check),
# openapi, actionlint — in containers pinned to the same nightly toolchain
# (rust-toolchain.toml / ci.yml) with the same RUSTFLAGS=-D warnings posture.
# The macOS compile-test job is not reproduced (it needs a macOS host).
#
# Parallel layout (default): two containers run AT THE SAME TIME —
#   * lane "tests"  — the `test` job only (build + both test passes).
#   * lane "checks" — every other selected job (fmt, clippy, doc, deny,
#                     cuda-oxide, openapi, actionlint), run sequentially.
# The two lanes use SEPARATE target-cache volumes (concurrent cargo on one
# target dir would lock-contend and serialise); they share the registry cache.
# Each lane's output is captured and printed under its own header when both
# finish. Use -Serial for a single container with one shared warm cache.
#
# The Linux build cache lives on dedicated Docker volumes, so it persists
# across runs and never collides with the host's MSVC target/ directory.
#
# Usage:
#   scripts\ci-local.ps1 [-Serial] [-SkipImageBuild] [-Pull] [-FailFast] [-CleanCache] [job ...]
#
# Jobs (default: all; every job runs even if an earlier one fails, and the
# script exits non-zero if any failed — use -FailFast to stop a lane at its
# first failure):
#   fmt clippy test doc deny cuda-oxide openapi actionlint
#
# Examples:
#   scripts\ci-local.ps1                 # full CI, tests || checks in parallel
#   scripts\ci-local.ps1 fmt clippy      # just the fast lints (checks lane only)
#   scripts\ci-local.ps1 -Serial         # old single-container behaviour
#   scripts\ci-local.ps1 -SkipImageBuild test   # reuse the cached image
[CmdletBinding()]
param(
    # Run all jobs sequentially in ONE container (old behaviour: one shared
    # warm cache, no parallelism).
    [switch]$Serial,
    # The CI image is rebuilt on every run by default (Docker layer cache makes
    # it near-instant when docker\ci.Dockerfile is unchanged). Use
    # -SkipImageBuild to reuse the existing image as-is.
    [switch]$SkipImageBuild,
    # Accepted for compatibility; rebuild is now the default.
    [switch]$RebuildImage,
    [switch]$Pull,
    [switch]$FailFast,
    # Accepted for compatibility; running every job is now the default.
    [switch]$KeepGoing,
    [switch]$CleanCache,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Jobs
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$Image     = 'tensor-wasm-ci:local'
$Dockerfile = Join-Path $RepoRoot 'docker\ci.Dockerfile'
# Per-lane target caches (parallel mode) + a single shared one for -Serial.
$TargetVolumeTests  = 'tensor-wasm-ci-target-tests'
$TargetVolumeChecks = 'tensor-wasm-ci-target-checks'
$TargetVolumeSerial = 'tensor-wasm-ci-target'
$CargoVolume        = 'tensor-wasm-ci-cargo'

# Keep aligned with Get-JobCmd below and .github/workflows/ci.yml.
$AllJobs = @('fmt', 'clippy', 'test', 'doc', 'deny', 'cuda-oxide', 'openapi', 'actionlint')

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path |
        Select-Object -Skip 1 -First 38 |
        ForEach-Object { $_ -replace '^#\s?', '' }
    exit 0
}

# Resolve + validate the requested jobs.
$selected = @()
if ($null -eq $Jobs -or $Jobs.Count -eq 0) {
    $selected = $AllJobs
}
else {
    foreach ($j in $Jobs) {
        if ($j -eq 'all') { $selected = $AllJobs; break }
        if ($AllJobs -notcontains $j) {
            Write-Error "unknown job '$j' (valid: $($AllJobs -join ', '), all)"
            exit 2
        }
        $selected += $j
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error 'docker not found on PATH. Install Docker Desktop first.'
    exit 1
}

function Get-JobCmd([string]$job) {
    switch ($job) {
        'fmt'        { 'cargo fmt --all -- --check' }
        'clippy'     { 'cargo clippy --workspace --all-targets -- -D warnings' }
        'test'       { 'cargo build --workspace && cargo test --workspace --no-default-features && cargo test --workspace' }
        'doc'        { "RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --no-default-features" }
        'deny'       { 'cargo deny --all-features check' }
        'cuda-oxide' { 'cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend' }
        'openapi'    { 'redocly lint --config openapi/redocly.yaml openapi/tensor-wasm-api.yaml && swagger-cli validate crates/tensor-wasm-api/openapi.json && cargo test -p tensor-wasm-api --test openapi_validation_test --no-default-features' }
        'actionlint' { 'actionlint' }
    }
}

# Build the in-container bash program for a set of jobs. `failed` accumulates
# failing job names; the lane exits non-zero if any job failed.
function Build-LaneScript([string[]]$laneJobs) {
    $q  = [char]34  # " — avoids confusing PowerShell's own string parser
    $nl = "`n"
    $s  = "set -uo pipefail${nl}rc=0${nl}failed=${q}${q}${nl}"
    foreach ($job in $laneJobs) {
        $cmd = Get-JobCmd $job
        $s += "printf ${q}\n\033[1;36m===> ci job: $job\033[0m\n${q}${nl}"
        $s += "if $cmd; then${nl}"
        $s += "    printf ${q}\033[1;32m===> $($job): OK\033[0m\n${q}${nl}"
        $s += "else${nl}"
        $s += "    printf ${q}\033[1;31m===> $($job): FAILED\033[0m\n${q}${nl}"
        $s += "    rc=1${nl}"
        $s += "    failed=${q}`$failed $job${q}${nl}"
        if ($FailFast) { $s += "    exit 1${nl}" }
        $s += "fi${nl}"
    }
    $s += "if [ -n ${q}`$failed${q} ]; then${nl}"
    $s += "    printf ${q}\n\033[1;31m===> lane FAILED: %s\033[0m\n${q} ${q}`$failed${q}${nl}"
    $s += "else${nl}"
    $s += "    printf ${q}\n\033[1;32m===> lane OK (all jobs passed)\033[0m\n${q}${nl}"
    $s += "fi${nl}"
    $s += "exit `$rc${nl}"
    return $s
}

# Optional cache reset.
if ($CleanCache) {
    Write-Host ">> removing cache volumes $TargetVolumeTests, $TargetVolumeChecks, $TargetVolumeSerial, $CargoVolume"
    docker volume rm -f $TargetVolumeTests $TargetVolumeChecks $TargetVolumeSerial $CargoVolume 2>$null | Out-Null
}

# Rebuild the image on every run by default (skip with -SkipImageBuild); always
# build if it is missing.
$imageExists = $false
try { docker image inspect $Image *> $null; if ($LASTEXITCODE -eq 0) { $imageExists = $true } } catch {}
if (-not $SkipImageBuild -or -not $imageExists) {
    Write-Host ">> building $Image (Docker layer cache makes this near-instant when docker\ci.Dockerfile is unchanged)"
    $buildArgs = @('build', '-f', $Dockerfile, '-t', $Image)
    if ($Pull) { $buildArgs += '--pull' }
    $buildArgs += (Join-Path $RepoRoot 'docker')
    & docker @buildArgs
    if ($LASTEXITCODE -ne 0) { Write-Error 'CI image build failed'; exit 1 }
}

# Partition the selected jobs into lanes.
$laneDefs = @()
if ($Serial) {
    $laneDefs += [pscustomobject]@{ Name = 'ci'; Vol = $TargetVolumeSerial; Jobs = $selected }
}
else {
    $testJobs = @($selected | Where-Object { $_ -eq 'test' })
    $restJobs = @($selected | Where-Object { $_ -ne 'test' })
    if ($testJobs.Count -gt 0) { $laneDefs += [pscustomobject]@{ Name = 'tests';  Vol = $TargetVolumeTests;  Jobs = $testJobs } }
    if ($restJobs.Count -gt 0) { $laneDefs += [pscustomobject]@{ Name = 'checks'; Vol = $TargetVolumeChecks; Jobs = $restJobs } }
}

# The lane body: pipe the bash program to a container over stdin, capture all
# output to a per-lane log file, and emit docker's exit code as the job result.
$laneBody = {
    param($laneScript, $repoRoot, $vol, $cargoVol, $image, $logFile)
    $laneScript | & docker run --rm -i `
        -v "${repoRoot}:/work" `
        -v "${vol}:/cargo-target" `
        -v "${cargoVol}:/cargo-home" `
        -e CARGO_TARGET_DIR=/cargo-target `
        -e CARGO_HOME=/cargo-home `
        -e CARGO_TERM_COLOR=always `
        -e "RUSTFLAGS=-D warnings" `
        -w /work `
        $image bash *> $logFile
    $LASTEXITCODE
}

if ($Serial) {
    Write-Host ">> running CI jobs serially (one container): $($selected -join ', ')"
}
else {
    Write-Host ">> running CI in parallel containers: [tests] || [checks]"
}

$started = @()
foreach ($lane in $laneDefs) {
    $laneScript = Build-LaneScript $lane.Jobs
    $logFile = Join-Path ([System.IO.Path]::GetTempPath()) "tw-ci-$($lane.Name)-$PID.log"
    if (Test-Path $logFile) { Remove-Item $logFile -Force -ErrorAction SilentlyContinue }
    Write-Host ">> lane [$($lane.Name)] (vol $($lane.Vol)) jobs: $($lane.Jobs -join ', ')"
    $bg = Start-Job -Name "ci-$($lane.Name)" -ScriptBlock $laneBody `
        -ArgumentList $laneScript, $RepoRoot, $lane.Vol, $CargoVolume, $Image, $logFile
    $started += [pscustomobject]@{ Name = $lane.Name; Job = $bg; Log = $logFile }
}

# Wait for every lane, then print each lane's captured output and tally.
$started.Job | Wait-Job | Out-Null

$overall = 0
$summary = @()
foreach ($entry in $started) {
    Write-Host ""
    Write-Host "==== lane [$($entry.Name)] output ====" -ForegroundColor Cyan
    if (Test-Path $entry.Log) {
        Get-Content $entry.Log | ForEach-Object { Write-Host $_ }
    }
    $rc = Receive-Job $entry.Job | Select-Object -Last 1
    Remove-Job $entry.Job -Force | Out-Null
    if (Test-Path $entry.Log) { Remove-Item $entry.Log -Force -ErrorAction SilentlyContinue }
    if ($null -eq $rc) { $rc = 1 }
    if ([int]$rc -eq 0) {
        $summary += "   [$($entry.Name)] OK"
    }
    else {
        $summary += "   [$($entry.Name)] FAILED (rc=$rc)"
        $overall = 1
    }
}

Write-Host ''
if ($overall -eq 0) {
    Write-Host '===> CI OK (all lanes passed)' -ForegroundColor Green
}
else {
    Write-Host '===> CI FAILED' -ForegroundColor Red
}
$summary | ForEach-Object { Write-Host $_ }
exit $overall
