# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Craton Software Company
#
# Run the GitHub Actions CI (.github/workflows/ci.yml) locally inside Docker.
#
# Windows-native (PowerShell) counterpart of scripts/ci-local.sh. Reproduces
# the Linux CI jobs — fmt, clippy, test, doc, deny, cuda-oxide(-backend-check),
# openapi, actionlint — in a container pinned to the same nightly toolchain
# (rust-toolchain.toml / ci.yml) with the same RUSTFLAGS=-D warnings posture.
# The macOS compile-test job is not reproduced (it needs a macOS host).
#
# The Linux build cache lives on dedicated Docker volumes, so it persists
# across runs and never collides with the host's MSVC target/ directory.
#
# Usage:
#   scripts\ci-local.ps1 [-SkipImageBuild] [-Pull] [-FailFast] [-CleanCache] [job ...]
#
# Jobs (default: all, run in this order; every job runs even if an earlier one
# fails, and the script exits non-zero if any failed — use -FailFast to stop
# at the first failure):
#   fmt clippy test doc deny cuda-oxide openapi actionlint
#
# Examples:
#   scripts\ci-local.ps1                 # full CI
#   scripts\ci-local.ps1 fmt clippy      # just the fast lints
#   scripts\ci-local.ps1 -SkipImageBuild test   # reuse the cached image
[CmdletBinding()]
param(
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
$TargetVolume = 'tensor-wasm-ci-target'
$CargoVolume  = 'tensor-wasm-ci-cargo'

# Keep aligned with Get-JobCmd below and .github/workflows/ci.yml.
$AllJobs = @('fmt', 'clippy', 'test', 'doc', 'deny', 'cuda-oxide', 'openapi', 'actionlint')

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path |
        Select-Object -Skip 1 -First 26 |
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

# Optional cache reset.
if ($CleanCache) {
    Write-Host ">> removing cache volumes $TargetVolume, $CargoVolume"
    docker volume rm -f $TargetVolume $CargoVolume 2>$null | Out-Null
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

# Build a proper multi-line shell script and pipe it via stdin.
# This sidesteps PowerShell's double-quote mangling when passing arguments
# to external processes (the bash -c $prog approach breaks whenever $prog
# contains any " because PowerShell corrupts them on the way to docker).
$q  = [char]34  # " — avoids confusing PowerShell's own string parser
$nl = "`n"

$script  = "set -uo pipefail${nl}"
$script += "rc=0${nl}"
$script += "failed=${q}${q}${nl}"

foreach ($job in $selected) {
    $cmd = Get-JobCmd $job
    $script += "printf ${q}\n\033[1;36m===> ci job: $job\033[0m\n${q}${nl}"
    $script += "if $cmd; then${nl}"
    $script += "    printf ${q}\033[1;32m===> $($job): OK\033[0m\n${q}${nl}"
    $script += "else${nl}"
    $script += "    printf ${q}\033[1;31m===> $($job): FAILED\033[0m\n${q}${nl}"
    $script += "    rc=1${nl}"
    $script += "    failed=${q}`$failed $job${q}${nl}"
    if ($FailFast) { $script += "    exit 1${nl}" }
    $script += "fi${nl}"
}
$script += "if [ -n ${q}`$failed${q} ]; then${nl}"
$script += "    printf ${q}\n\033[1;31m===> CI FAILED: %s\033[0m\n${q} ${q}`$failed${q}${nl}"
$script += "else${nl}"
$script += "    printf ${q}\n\033[1;32m===> CI OK (all jobs passed)\033[0m\n${q}${nl}"
$script += "fi${nl}"
$script += "exit `$rc${nl}"

Write-Host ">> running CI jobs: $($selected -join ', ')"
$script | & docker run --rm -i `
    -v "${RepoRoot}:/work" `
    -v "${TargetVolume}:/cargo-target" `
    -v "${CargoVolume}:/cargo-home" `
    -e CARGO_TARGET_DIR=/cargo-target `
    -e CARGO_HOME=/cargo-home `
    -e CARGO_TERM_COLOR=always `
    -e "RUSTFLAGS=-D warnings" `
    -w /work `
    $Image bash

exit $LASTEXITCODE
