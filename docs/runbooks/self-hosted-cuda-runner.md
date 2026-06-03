<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Self-hosted CUDA runner — registration runbook

Procedure runbook for standing up the self-hosted GitHub Actions runner
the `cuda` workflow (`.github/workflows/cuda.yml`) requires. Closes
audit Problem #8 and unblocks PATH-TO-V1 v0.2 exit criterion "S22
self-hosted CUDA runner online in CI". Until a runner registers with
both the `self-hosted` and `cuda` labels, every job in `cuda.yml` is
queued indefinitely; the workflow does not surface as a required
check.

This runbook is a procedure runbook (not an alert runbook); follow
the [`runbooks/README.md`](README.md) contract section "Procedure
runbooks".

## When to run this

- Standing up the first runner for the project
- Replacing a runner whose GPU has been re-provisioned
- Onboarding a contributor or sponsor donating spare GPU capacity

## Prerequisites

- A host with at least one NVIDIA GPU, SM_70 or higher (`nvidia-smi`
  reports it). SM_80+ for wmma kernels; SM_75 is enough for the
  current bench + test corpus (the dev box that signed off
  `vector_add_end_to_end_real_ptx_real_kernel` is an RTX 2060 / SM_75
  / WDDM).
- CUDA Toolkit 12.0+ (12.4 recommended; 13.x verified). See
  [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) for the install matrix.
- Rust toolchain `nightly-2026-04-03` (will be pulled by the workflow
  on first job, but pre-installing saves ~5 min/job).
- `git`, `curl`, `tar`, basic build tools.
- Network egress to `github.com`, `crates.io`, and (transitively, via
  `Cargo.toml`) `https://github.com/NVlabs/cuda-oxide` +
  `https://github.com/vaivaswatha/pliron`.
- Maintainer permissions on the `craton-co/craton-tensor-wasm`
  repository (Settings → Actions → Runners).

## Procedure

### Step 1 — register the runner with GitHub

From a maintainer-permissioned account:

1. Open `https://github.com/craton-co/craton-tensor-wasm/settings/actions/runners/new`.
2. Pick the runner OS (Linux x86_64 strongly preferred; Windows works
   on the dev box but WDDM-specific test failures will need
   per-platform `#[ignore]` markers — see W5.9 + B5).
3. Copy the displayed registration token (single-use, ~1 h validity).

### Step 2 — install the runner on the host

Linux:

```sh
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -O -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
tar xzf actions-runner-linux-x64-2.319.1.tar.gz
./config.sh \
  --url https://github.com/craton-co/craton-tensor-wasm \
  --token <PASTE_TOKEN_FROM_STEP_1> \
  --name "$(hostname)-cuda" \
  --labels self-hosted,cuda \
  --work _work \
  --unattended
```

**The `self-hosted,cuda` labels are not optional.** `cuda.yml` jobs
target `runs-on: [self-hosted, cuda]`; a runner missing either label
will be ignored and the job will queue forever.

### Step 3 — install as a service

Linux (systemd):

```sh
sudo ./svc.sh install $USER
sudo ./svc.sh start
sudo ./svc.sh status
```

The runner appears in the Settings → Actions → Runners page within a
few seconds with status "Idle".

### Step 4 — verify against `cuda.yml`

Trigger the workflow via the GitHub UI: Actions → cuda →
"Run workflow" → branch `dev`. Within ~30 s the four jobs
(`cust-unified-memory`, `wasi-gpu-cuda`, `cudarc-backend`,
`cuda-oxide-backend`) should be picked up by the runner.

Expected outcomes (on a Linux datacenter GPU; WDDM consumer GPUs
diverge — see "Platform caveats" below):

| Job | Expected result |
|---|---|
| `cust-unified-memory` | 46/46 tests pass (W5.9 + B5; single-threaded due to cust 0.3 primary-context model) |
| `wasi-gpu-cuda` | 7/7 tests pass (B2 incl. real-PTX dispatch + readback) |
| `cudarc-backend` | 6/6 tests pass (the 1 WDDM failure does NOT appear on Linux datacenter) |
| `cuda-oxide-backend` | Compile only; scaffold returns "not yet wired" sentinels per O2 |

### Step 5 — wire the workflow to required-check policy

Once jobs pass green at least once:

1. Settings → Branches → `main` branch protection rule
2. Add `cuda / cust-unified-memory`, `cuda / wasi-gpu-cuda`,
   `cuda / cudarc-backend`, `cuda / cuda-oxide-backend` to the
   required status checks
3. Same for `dev` if the project enforces protection on that branch

After this step the CUDA path is no longer "CI is blind to CUDA
tests" — every PR is gated on the four jobs passing.

## Platform caveats

- **Windows WDDM consumer GPUs**: `cuMemAdvise(SET_PREFERRED_LOCATION)`
  and `cuMemPrefetchAsync` return `CUDA_ERROR_INVALID_DEVICE` because
  consumer Turing/Ampere cards in WDDM mode don't expose
  `concurrentManagedAccess`. The B5 + W5.9 wave documented this as
  "24/46 pass on Windows; 22 failures are platform-tier, not bugs".
  Recommend Linux for the production runner; if Windows is the only
  option, expect those 22 tests to fail and either accept it or wrap
  them in `#[cfg(not(target_os = "windows"))]`.

- **Driver model**: TCC (data-center cards) and Linux UVM expose
  `concurrentManagedAccess`; WDDM (consumer Windows) does not. Check
  with `nvidia-smi --query-gpu=driver_model.current --format=csv`.

- **PTX target architecture**: the `kernels/vector_add.ptx` fixture
  targets SM_80 but the modern CUDA driver JIT'd it up to SM_75 on the
  dev box. If the runner GPU is older than SM_70, `cust::module::
  Module::from_ptx` will reject with `CUDA_ERROR_NO_BINARY_FOR_GPU`
  and the B2 test will trip its skip path.

## Tearing down

```sh
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <REGISTRATION_TOKEN_OR_PAT>
```

Then delete the runner entry from Settings → Actions → Runners and
remove the required-check policy for the four `cuda /` jobs.

## Cost / scaling notes

- One runner is sufficient for the current PR volume. If queue depth
  becomes an issue, register additional runners with the same labels;
  GitHub round-robins across them.
- Concurrency is gated by the workflow's `concurrency:` group (one
  in-flight per ref) — adding runners helps cross-PR throughput, not
  per-PR latency.
- Cloud GPU rental (Lambda Labs / RunPod / AWS g5) is documented in
  the v0.2 risk register as a fallback if no donated host materialises.

## Related

- [`.github/workflows/cuda.yml`](../../.github/workflows/cuda.yml) —
  the workflow this runbook stands up
- [`docs/CUDA-SETUP.md`](../CUDA-SETUP.md) — toolkit + driver versions
- [`bench-results/cuda-rtx2060-tests.txt`](../../bench-results/cuda-rtx2060-tests.txt)
  — the dev box's per-test results (Windows WDDM tier)
- [`docs/PATH-TO-V1.md`](../PATH-TO-V1.md) — v0.2 exit criterion
  S22 runner
- [`docs/runbooks/README.md`](README.md) — runbook contract
