# CUDA Setup

Craton TensorWasm's GPU-resident crates — `tensor-wasm-mem`, `tensor-wasm-wasi-gpu`, `tensor-wasm-jit`, and `tensor-wasm-tenant` — link against the CUDA Driver API and the CUDA Runtime through the [`cust`](https://docs.rs/cust) crate (default backend) or the [`cudarc`](https://docs.rs/cudarc) crate (opt-in via `cudarc-backend`, see [`docs/CUDARC-SPIKE.md`](CUDARC-SPIKE.md)). This document states the exact toolkit version, exact driver version, exact compiler version, exact environment variables, exact verification commands, exact feature-flag combinations, and exact troubleshooting actions used to bring a TensorWasm development host online. It is the contract between a contributor's box and the S22 self-hosted CUDA runner: if your box matches the matrix below, a clean `cargo build --workspace --features tensor-wasm-mem/unified-memory` succeeds.

The S22 runner runs **CUDA Toolkit 12.4** on **driver 550.54.15** under **Ubuntu 22.04 x86_64** on an **NVIDIA L4 (SM_89)**. Active contributor dev boxes have been verified additionally on **CUDA Toolkit 13.2 + driver 591.86** under **Windows 11 x86_64** on an **RTX 2060 (SM_75)**, with the [SM_75 limitations called out below](#sm-level-compatibility-matrix).

## Contents

1. [Required versions](#required-versions)
2. [Install commands](#install-commands)
3. [Required environment variables](#required-environment-variables)
4. [Verification commands](#verification-commands)
5. [Feature-flag combinations](#feature-flag-combinations)
6. [Using the cuda-oxide-backend feature](#using-the-cuda-oxide-backend-feature)
7. [Using the experimental-cuda-oxide-host-backend feature](#using-the-experimental-cuda-oxide-host-backend-feature)
8. [SM-level compatibility matrix](#sm-level-compatibility-matrix)
9. [MPS quick-start](#mps-quick-start)
10. [Troubleshooting](#troubleshooting)
11. [One-shot verification script](#one-shot-verification-script)
12. [Stub libraries for CI](#stub-libraries-for-ci)
13. [Cross-references](#cross-references)

---

## Required versions

The numbers in this section are not aspirational. They are the versions installed on the S22 runner and on the contributor dev boxes that ship green PRs.

### CUDA Toolkit

| Component | Minimum | Recommended (S22 runner) | Maximum verified |
|---|---|---|---|
| CUDA Toolkit | 12.0 | **12.4** | 13.2 |
| Cudarc headers selector (`cuda-12000` feature) | 12.0 | 12.0 | 13.2 (forward-compatible) |

The `cudarc` workspace dependency is pinned at `0.13` with the `cuda-12000` feature, which compiles against CUDA 12.0+ headers and runs forward against any CUDA 12.x or 13.x toolkit installed on the host. The `cust 0.3` backend has no header selector — it loads the driver dynamically and accepts any toolkit at runtime that supplies a 12.0+ driver.

CUDA Toolkit 13.x is verified for builds but is not the S22 runner version. If you develop on a 13.x box, your PRs are still validated against 12.4 in CI.

### NVIDIA driver

Drivers are forward-compatible: the toolkit's runtime works against any driver at or above the row that matches the toolkit. Mismatches surface as `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (error 803) at the first `cuInit` call inside `tensor-wasm-mem`.

| CUDA Toolkit | Linux driver minimum | Windows driver minimum |
|---|---|---|
| 12.0 | 525.60.13 | 527.41 |
| 12.4 | 550.54.14 | 551.61 |
| 12.6 | 560.28.03 | 560.81 |
| 13.0 | 580.65.06 | 580.88 |
| 13.2 | 590.42.01 | **591.86** |

The contributor box noted in the header runs driver **591.86** on Windows 11 against a **13.2** toolkit and an RTX 2060. The S22 runner runs driver **550.54.15** on Ubuntu 22.04 against a 12.4 toolkit and an L4.

### Host compiler

`cust` and `cudarc` both invoke `nvcc` at build time to validate header parsing. `nvcc` calls the system host compiler. The matrix below is the supported set.

| OS | Host compiler | Exact version | Notes |
|---|---|---|---|
| Ubuntu 22.04 | GCC | 11.4.0 | Stock `apt install build-essential` |
| Ubuntu 24.04 | GCC | 13.2.0 | Stock `apt install build-essential` |
| Windows 11 | MSVC | Visual Studio 2022 Build Tools 17.10+ (`cl.exe` 19.40+) | "Desktop development with C++" workload, MSVC v143 |
| WSL2 (Ubuntu 22.04) | GCC | 11.4.0 | Same as Ubuntu 22.04; do not install a Windows toolchain inside WSL |

Clang as the host compiler is **not supported** by the project. `nvcc -ccbin=clang++` builds in isolation but the upstream `cust 0.3` build script hard-codes GCC/MSVC probes and panics under Clang. `cudarc` is Clang-agnostic but switching back to `cust` for the default build will fail; do not mix.

---

## Install commands

### Ubuntu 22.04 / 24.04 (x86_64)

Run as a user with `sudo`. The `cuda-keyring` package is the supported NVIDIA mechanism for adding the APT repo.

```bash
# Pick ONE distro line below
DISTRO=ubuntu2204     # for 22.04
# DISTRO=ubuntu2404   # for 24.04

wget "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the S22 runner version (12.4). Substitute cuda-toolkit-12-6,
# cuda-toolkit-13-0, or cuda-toolkit-13-2 if you want to develop against a
# newer toolkit; CI still validates against 12.4.
sudo apt-get install -y cuda-toolkit-12-4 build-essential

# Driver: install separately on bare metal (not needed inside WSL2).
sudo apt-get install -y cuda-drivers-550

sudo reboot
```

After reboot, `nvidia-smi` must report a populated GPU table before the toolkit is usable. Headless servers without `nvidia-modprobe` running need the device nodes created once at boot:

```bash
sudo apt-get install -y nvidia-modprobe
sudo nvidia-modprobe -u -c=0
```

### Windows 11 (x86_64)

Three options. Pick exactly one — do not stack them, the second installer will overwrite the first's `CUDA_PATH`.

**Option A — winget (recommended, scriptable):**

```powershell
winget install --id Nvidia.CUDA --version 12.4.1 --accept-package-agreements --accept-source-agreements
```

**Option B — Chocolatey:**

```powershell
choco install cuda --version=12.4.1.55100 -y
```

**Option C — official .exe installer:** Download `cuda_12.4.1_551.78_windows.exe` from [developer.nvidia.com/cuda-12-4-1-download-archive](https://developer.nvidia.com/cuda-12-4-1-download-archive), run it, accept the default component set. The installer bundles a compatible driver (551.78 with the 12.4.1 archive); do not deselect it unless you already have a newer driver from GeForce Experience or the NVIDIA Driver Downloads page.

After the installer finishes, **install Visual Studio 2022 Build Tools 17.10 or later** with the "Desktop development with C++" workload:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

Open a fresh **x64 Native Tools Command Prompt for VS 2022** (or run `vcvars64.bat` in your existing shell) before invoking `cargo build` so `cl.exe` and `link.exe` are on `PATH`.

### WSL2 (Ubuntu 22.04 inside Windows 11)

WSL2 has a non-obvious split: the **driver lives in the Windows host**, the **toolkit lives inside the WSL distro**, and the two communicate through `/usr/lib/wsl/lib/libcuda.so.1` which WSL bind-mounts from the host.

1. **Inside Windows**, install the NVIDIA driver via GeForce Experience or the `cuda-12-4` Windows installer (Option C above). Do not skip; WSL2 cannot use a Linux driver.
2. **Inside the WSL Ubuntu distro**, install only the toolkit (NOT the driver):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4 build-essential
```

3. Verify the bind-mount is present:

```bash
ls -l /usr/lib/wsl/lib/libcuda.so.1
nvidia-smi   # uses /usr/lib/wsl/lib/nvidia-smi; reports the Windows driver
```

If `/usr/lib/wsl/lib/libcuda.so.1` is missing, your Windows driver is too old. Update to driver 555.85 or later on the Windows side; the WSL GPU bind-mount became reliable starting there.

---

## Required environment variables

The build scripts read four variables. Set them in your shell profile, not just per-shell, so `rust-analyzer` and your IDE see them too.

| Variable | Linux value | Windows value | Purpose |
|---|---|---|---|
| `CUDA_ROOT` (alias: `CUDA_PATH`, `CUDA_HOME`) | `/usr/local/cuda` (12.x) or `/usr/local/cuda-12.4` (pinned) | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` | Toolkit install root. `cust` checks `CUDA_ROOT`, then `CUDA_PATH`, then `CUDA_HOME` in order. |
| `CUDA_ARCH` | `sm_75` (RTX 2060), `sm_80` (A100), `sm_86` (RTX 30xx), `sm_89` (L4 / RTX 40xx), `sm_90` (H100) | same | Target compute capability for PTX emission by `tensor-wasm-jit`. **The S22 runner uses `sm_89` for L4.** |
| `PATH` | prepend `$CUDA_ROOT/bin` | prepend `%CUDA_ROOT%\bin` | `nvcc` and `ptxas` must be reachable by `tensor-wasm-jit`. |
| `LD_LIBRARY_PATH` (Linux only) | prepend `$CUDA_ROOT/lib64` | not used | Dynamic loader finds `libcuda.so`, `libcudart.so`. Windows finds DLLs through `PATH` only. |

### Linux (bash / zsh) — append to `~/.bashrc` or `~/.zshrc`

```bash
export CUDA_ROOT=/usr/local/cuda
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDA_ARCH=sm_89
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
```

Then `source ~/.bashrc` (or open a new shell).

### Windows 11 (PowerShell, persistent)

```powershell
setx CUDA_ROOT "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
setx CUDA_ARCH "sm_75"
setx PATH "$env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
```

`setx` writes to the user registry; **close and reopen your shell** for the values to apply. The toolkit installer also adds `%CUDA_PATH%\bin` to the system `PATH` automatically; the `setx PATH` line above is belt-and-braces for shells that ignore the system path.

Set `CUDA_ARCH` to the value matching your installed GPU. On the dev box (RTX 2060) use `sm_75`; on the S22 runner (L4) use `sm_89`. See [SM-level compatibility matrix](#sm-level-compatibility-matrix) for the full list.

---

## Verification commands

Run each command in order. Every line must succeed before you start a long build.

### Linux / WSL2 / macOS

```bash
nvidia-smi                                   # driver loaded, GPU enumerated
nvcc --version                               # toolkit on PATH
ptxas --version                              # PTX assembler reachable
echo "$CUDA_ROOT"                            # non-empty, points to existing dir
ls "$CUDA_ROOT/lib64/libcuda.so" 2>/dev/null || \
  ls /usr/lib/x86_64-linux-gnu/libcuda.so   # libcuda visible to the loader
```

### Windows 11 (PowerShell)

```powershell
nvidia-smi                                   # driver loaded, GPU enumerated
nvcc --version                               # toolkit on PATH
ptxas --version                              # PTX assembler reachable
$env:CUDA_PATH                               # non-empty
Test-Path "$env:CUDA_PATH\bin\nvcc.exe"      # True
Test-Path "$env:CUDA_PATH\bin\cudart64_*.dll" # True
```

### What good output looks like

`nvidia-smi` on the dev box prints a table with a `NVIDIA GeForce RTX 2060` row, driver `591.86`, CUDA version `13.1` (this is the **driver-reported runtime**, not the toolkit). On the S22 runner the row reads `NVIDIA L4`, driver `550.54.15`, CUDA `12.4`.

`nvcc --version` ends with `Cuda compilation tools, release 12.4, V12.4.131` (S22 runner) or `release 13.2, V13.2.x` (dev box). The release line must match the toolkit you installed.

`ptxas --version` ends with the same release number as `nvcc`. A mismatch means two toolkits are layered on the same `PATH`; uninstall the older one or reorder `PATH`.

### Smoke build

From the repository root:

```bash
cargo build --workspace --features tensor-wasm-mem/unified-memory
```

This builds `tensor-wasm-mem` against `cust` and links `libcuda`. If this succeeds, the toolchain is fully wired. To exercise the JIT pipeline too:

```bash
cargo build --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-jit/auto-offload,tensor-wasm-wasi-gpu/cuda,tensor-wasm-tenant/cuda
```

---

## Feature-flag combinations

The workspace has no default features. Every CUDA-touching code path is opt-in. The table below lists the exact `cargo` commands; cross-reference [BUILD.md](BUILD.md) for the cross-crate feature taxonomy.

### Quick-reference commands

| Goal | Command |
|---|---|
| No-CUDA local check (no linker against libcuda) | `cargo build --workspace` |
| CUDA host build (default `cust` backend) | `cargo build --workspace --features tensor-wasm-mem/unified-memory` |
| CUDA host build via `cudarc` (spike backend) | `cargo build --workspace --features tensor-wasm-mem/cudarc-backend` |
| CUDA + auto-offload JIT | `cargo build --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-jit/auto-offload,tensor-wasm-wasi-gpu/cuda` |
| CUDA + multi-tenant MPS | `cargo build --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-tenant/cuda,tensor-wasm-tenant/mps` |
| macOS Metal Performance Shaders (placeholder) | `cargo build --workspace --features tensor-wasm-mem/mps` — **does not exist; the `mps` flag is on `tensor-wasm-tenant` and refers to NVIDIA Multi-Process Service, not Apple MPS. See [MPS-SETUP.md](MPS-SETUP.md).** |

### What each flag pulls in

- **`tensor-wasm-mem/unified-memory`** — Links the `cust 0.3` crate. Allocates GPU memory via `cudaMallocManaged`. Adds `libcuda` to the link line. **This is the production default.** Requires the toolkit and a working driver.

- **`tensor-wasm-mem/cudarc-backend`** — Links the `cudarc 0.13` crate with the `driver` and `cuda-12000` features. Exposes a parallel `UnifiedBuffer` implementation under `tensor_wasm_mem::cudarc_backend`. **Coexists with `unified-memory`** — both features may be enabled simultaneously during the migration spike (v0.2 milestone). Do not rely on `cudarc-backend` for production until the migration is committed; see [`docs/CUDARC-SPIKE.md`](CUDARC-SPIKE.md) for the cutover plan and [`docs/RISKS.md`](RISKS.md) for the timeline.

- **`tensor-wasm-mem/pinned-host-memory`** — Pure-Rust page-locked host buffers. Does **not** link `cust` or `cudarc`. Use this if you want fast host→device transfers without a CUDA toolkit on the build host.

- **`tensor-wasm-wasi-gpu/cuda`** — Links `cust`. Compiles the real `wasi_cuda_*` host functions (vs. the no-CUDA stubs that return `CudaUnavailable`). Required for `tensor-wasm-wasi-gpu` integration tests against real hardware.

- **`tensor-wasm-tenant/cuda`** — Links `cust`. Creates real per-tenant `cuCtx*` contexts instead of in-process stubs.

- **`tensor-wasm-tenant/mps`** — Pure-Rust feature (no extra crate dependency). Switches `TenantRegistry::mps_or_fallback()` to probe `/tmp/nvidia-mps` and use MPS-shared contexts when present. Combine with `tensor-wasm-tenant/cuda` for real production use.

- **`tensor-wasm-jit/auto-offload`** — Enables additional CUDA-side wiring in the JIT detector. The Cranelift→PTX pipeline itself is always compiled; this flag gates the runtime that actually dispatches generated PTX through `cust`. Combine with `tensor-wasm-mem/unified-memory` and `tensor-wasm-wasi-gpu/cuda` for a real end-to-end JIT path.

### Switching between cust and cudarc backends

The `unified-memory` and `cudarc-backend` features are not mutually exclusive at the Cargo level — both can compile in. At runtime, code paths under `tensor_wasm_mem::cudarc_backend::*` use cudarc; code paths under `tensor_wasm_mem::*` (the existing surface) use cust. To switch a single build between backends:

```bash
# cust only
cargo build --workspace --features tensor-wasm-mem/unified-memory

# cudarc only
cargo build --workspace --features tensor-wasm-mem/cudarc-backend

# both, for migration testing
cargo build --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-mem/cudarc-backend
```

The S22 runner builds with `unified-memory` only. The cudarc spike runner (when online) will build with `cudarc-backend` only. Do not enable both in CI until the cutover decision is made.

---

## Using the cuda-oxide-backend feature

The `cuda-oxide-backend` feature on `tensor-wasm-mem` is the third
host-side CUDA backend, sitting alongside `unified-memory` (cust,
production default) and `cudarc-backend` (the W1.2 spike). It compiles
against the [cuda-oxide](https://github.com/NVlabs/cuda-oxide) host
crates and is the v0.5 default candidate per
[RFC 0001](../rfcs/0001-cuda-oxide-integration.md) ("cuda-oxide as
the v0.5 cust successor"). The full Wasm→PTX kernel-compilation
pipeline that cuda-oxide enables is documented in
[`PLIRON-PIPELINE.md`](PLIRON-PIPELINE.md).

At v0.3.1, `cuda-oxide-backend` is a **dep-less scaffold**: enabling it
does not pull `cuda-host`, `cuda-core`, `cuda-async`, or `pliron` into
the resolved dependency graph yet. The scaffold exists to lock in the
feature name and the `CudaBackend` trait shape so call-sites in
`tensor-wasm-jit` / `tensor-wasm-wasi-gpu` / `tensor-wasm-tenant`
written against it during v0.3.x do not need to be re-typed when the
actual cuda-oxide deps land in v0.4 (per [RFC 0001 "Rollout"](../rfcs/0001-cuda-oxide-integration.md#rollout-pr-sequencing)).
Until v0.4, `cargo build --features cuda-oxide-backend` is therefore a
no-op on link behaviour but exercises the feature-flag plumbing.

### Toolchain pin

cuda-oxide pins `nightly-2026-04-03`. The TensorWasm workspace currently
pins the same nightly (see [`rust-toolchain.toml`](../rust-toolchain.toml)),
so on the **current workspace pin** no toolchain override is required;
a plain `cargo build --features cuda-oxide-backend` works.

The RFC nevertheless documents an explicit toolchain override as the
invocation pattern, for two reasons:

1. The workspace pin may bump at v0.4 (per [RFC 0001 "Toolchain plan"
   step 3](../rfcs/0001-cuda-oxide-integration.md#toolchain-plan))
   to a nightly that satisfies both cuda-oxide and the W2.9 Wasmtime
   cadence policy. If that nightly diverges from cuda-oxide's pin
   between v0.4 and a later refresh, the override becomes load-bearing
   again.
2. Local toolchain overrides (`rustup override set <nightly>` in the
   workspace, or a contributor running `--features cuda-oxide-backend`
   from a non-default checkout) want a documented, explicit form.

The documented invocation (matches [RFC 0001 "Toolchain plan" step 2](../rfcs/0001-cuda-oxide-integration.md#toolchain-plan)):

### Linux / WSL2 / macOS (bash / zsh)

```bash
# Override only for this invocation; does not touch rust-toolchain.toml.
RUSTUP_TOOLCHAIN=nightly-2026-04-03 \
  cargo build --workspace --features tensor-wasm-mem/cuda-oxide-backend

# Workspace check (no link, faster) — what CI runs:
RUSTUP_TOOLCHAIN=nightly-2026-04-03 \
  cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend
```

### Windows 11 (PowerShell)

```powershell
$env:RUSTUP_TOOLCHAIN = "nightly-2026-04-03"
cargo build --workspace --features tensor-wasm-mem/cuda-oxide-backend
Remove-Item Env:RUSTUP_TOOLCHAIN
```

### What CI runs

The `.github/workflows/ci.yml` workflow gains a single matrix entry
(`cuda-oxide-backend-check`) that runs
`cargo check --workspace --features tensor-wasm-mem/cuda-oxide-backend`
on ubuntu-latest with the pinned toolchain. The existing CUDA-stub
runners are untouched; the new entry is additive and only fails when
the cuda-oxide-backend wiring itself regresses. Tests that require
actual GPU hardware are not run on hosted runners — they live in
ignored tests under the cuda-oxide-backend gate, on the S22 self-hosted
runner once the v0.4 parity work lands.

### Cross-references

- [RFC 0001](../rfcs/0001-cuda-oxide-integration.md) — full design
  rationale for cuda-oxide as the v0.5 cust successor, the
  contingent-default approach, and the cudarc fallback.
- [`PLIRON-PIPELINE.md`](PLIRON-PIPELINE.md) — the Pliron-based
  Wasm→PTX pipeline that cuda-oxide unlocks (v0.6+ research goal in
  RFC 0001 "Future possibilities").
- [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md#git-pinned-sources)
  — the git-pin policy for the Pliron transitive dependency that
  cuda-oxide pulls in.
- [`CUDA-KERNELS.md`](CUDA-KERNELS.md) — "Path C: Rust kernels via
  cuda-oxide" — the author-side kernel surface that the
  `#[cuda_module]` macro enables once the backend is wired.

---

## Using the experimental-cuda-oxide-host-backend feature

`experimental-cuda-oxide-host-backend` (added in W4.1, 2026-05-27; renamed
from `cuda-oxide-host-backend` to carry the `experimental-` prefix) is the
**strict-superset** sibling of `cuda-oxide-backend`.

> **Experimental — not yet buildable.** This feature is intentionally
> non-building: the `cuda_oxide_backend` module opens with a
> `compile_error!`, so enabling
> `--features experimental-cuda-oxide-host-backend` will fail to compile
> today. The `compile_error!` is lifted only once the S22 self-hosted
> runner has actually compiled and validated the host port. The commands
> below document the intended invocation for when the port lands; they do
> not build on the current tree.

Enabling it pulls
in the four cuda-oxide host-side crates as git-pinned dependencies (pin
SHA `4a56e4220aab8ce5d085a411e7f806cebb647d14`, matching the v0.1.0 tag)
and is intended to switch `tensor_wasm_mem::cuda_oxide_backend::CudaOxideUnifiedBuffer`
from the `NOT_YET_WIRED` sentinel-error scaffold to a real
`cuMemAllocManaged`-backed allocation. The transitive crate set:

| Crate | Role |
|---|---|
| `cuda-host` | Kernel launch helpers (`cuda_launch!`, `LtoIR` loader). |
| `cuda-core` | RAII `CudaContext` / `CudaStream` / `CudaModule`. Re-exports the raw `cuda_bindings` as `cuda_core::sys` — the path `cuda_oxide_backend.rs` uses for `cuMemAllocManaged`, `cuMemPrefetchAsync`, `cuMemAdvise`, `cuMemFree_v2`. |
| `cuda-device` | Device-side primitives (`DisjointSlice`, kernel attribute). Linked here for v0.4+ kernel-authoring follow-ups; not directly imported from `cuda_oxide_backend.rs` today. |
| `cuda-macros` | `#[kernel]` and `cuda_launch!` / `cuda_launch_async!` proc-macros. Linked for the same v0.4+ rationale as `cuda-device`. |

The pattern mirrors W3.3's `pliron-llvm-backend` on `tensor-wasm-jit`:
the *base* feature (`cuda-oxide-backend`) is intentionally dep-less so
contributor boxes without a CUDA Toolkit or `libclang` can still build
the scaffold, and the *superset* feature (`cuda-oxide-host-backend`)
adds the heavyweight git deps that need a full toolchain.

### Toolchain prerequisites

The `cuda-bindings` build script invokes `bindgen` against `<cuda.h>`,
which needs **both** of:

| Prerequisite | Linux | Windows |
|---|---|---|
| CUDA Toolkit (provides `<cuda.h>`, `libcuda.so` / `nvcuda.dll`) | `cuda-toolkit-12-4` (see [Install commands](#install-commands)) | NVIDIA CUDA installer (Option A/B/C, see above) |
| `libclang` (for `bindgen`) | `sudo apt-get install -y libclang-dev` | `winget install LLVM.LLVM` (installs `libclang.dll` at `C:\Program Files\LLVM\bin\`) |
| `LIBCLANG_PATH` env var | usually unnecessary; `libclang-dev` puts the SO on `LD_LIBRARY_PATH` | required: `setx LIBCLANG_PATH "C:\Program Files\LLVM\bin"` |
| `CUDA_TOOLKIT_PATH` env var (`cuda-bindings` reads this; defaults to `/usr/local/cuda`) | usually unnecessary on the default Linux install | required: `setx CUDA_TOOLKIT_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"` |

The workspace nightly pin (`nightly-2026-04-03`, see
[`rust-toolchain.toml`](../rust-toolchain.toml)) is the same nightly
cuda-oxide itself pins, so no `RUSTUP_TOOLCHAIN` override is required
on the default workspace toolchain.

### Linux / WSL2 install

```bash
# CUDA Toolkit + driver — see Install commands above
sudo apt-get install -y cuda-toolkit-12-4 build-essential

# libclang for bindgen
sudo apt-get install -y libclang-dev

# verify
ls /usr/lib/llvm-*/lib/libclang.so* | head -1   # should print at least one path
echo "$CUDA_ROOT"                                # should resolve to /usr/local/cuda
```

If `libclang.so` lives outside the default search path, export
`LIBCLANG_PATH`:

```bash
export LIBCLANG_PATH=/usr/lib/llvm-14/lib
```

### Windows 11 install

```powershell
# CUDA Toolkit — see Install commands above for Option A/B/C
winget install --id Nvidia.CUDA --version 12.4.1 --accept-package-agreements --accept-source-agreements

# LLVM (provides libclang.dll)
winget install LLVM.LLVM --accept-package-agreements --accept-source-agreements

# Persistent env vars (close + reopen the shell after)
setx LIBCLANG_PATH "C:\Program Files\LLVM\bin"
setx CUDA_TOOLKIT_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

### Build invocation

From the repository root:

> The commands below are the *intended* invocation once the host port
> lands and the `compile_error!` guard is removed. On the current tree
> they fail to compile by design (see the experimental note above).

```bash
# Compile-only check (what CI's cuda-host runner runs)
cargo check -p tensor-wasm-mem --features experimental-cuda-oxide-host-backend

# Full build
cargo build -p tensor-wasm-mem --features experimental-cuda-oxide-host-backend

# Hardware-gated tests (requires a CUDA-capable GPU)
cargo test -p tensor-wasm-mem --features experimental-cuda-oxide-host-backend \
    --test cuda_oxide_smoke -- --ignored
```

The `--features tensor-wasm-mem/experimental-cuda-oxide-host-backend` form works
identically from the workspace root:

```bash
cargo build --workspace --features tensor-wasm-mem/experimental-cuda-oxide-host-backend
```

### Failure modes

| Error | Root cause | Fix |
|---|---|---|
| `Unable to find libclang: "couldn't find any valid shared libraries matching: ['clang.dll', 'libclang.dll']"` | `LIBCLANG_PATH` unset or points at a directory missing `libclang.dll`/`libclang.so`. | Install LLVM (see above) and `setx LIBCLANG_PATH ...` on Windows / `export LIBCLANG_PATH=...` on Linux. |
| `fatal error: 'cuda.h' file not found` from `bindgen` | `CUDA_TOOLKIT_PATH` not set (or `cuda.h` not under `$CUDA_TOOLKIT_PATH/include`). | `setx CUDA_TOOLKIT_PATH ...` on Windows; verify `ls $CUDA_TOOLKIT_PATH/include/cuda.h` on Linux. |
| `error: linker 'link.exe' not found` (Windows) | Visual Studio Build Tools not installed / not on `PATH`. | Open an `x64 Native Tools Command Prompt for VS 2022` (or run `vcvars64.bat`) before invoking `cargo`. |
| `error[E0432]: unresolved import cuda_core::sys` | Stale `Cargo.lock` from before W4.1; the pinned rev did not include `cuda-core`. | `cargo update -p cuda-core --precise <pin>` or delete `Cargo.lock` and let cargo resolve afresh. |

### What CI runs

The `experimental-cuda-oxide-host-backend-check` job in
`.github/workflows/ci.yml` runs
`cargo check -p tensor-wasm-mem --features experimental-cuda-oxide-host-backend`
on a runner image that pre-installs CUDA Toolkit 12.4 + LLVM 18. Because
the feature is currently guarded by a `compile_error!`, that job is
expected to fail-by-design and is kept non-required / allowed-to-fail
until the S22 host port lifts the guard. The existing CUDA-stub runners
are untouched. Hardware-gated tests (the
`#[ignore = "requires CUDA hardware"]` set in
`tests/cuda_oxide_smoke.rs`) run on the S22 self-hosted runner only.

---

## SM-level compatibility matrix

This matrix is the authoritative statement of what TensorWasm runs on what hardware. **wmma (tensor-core warp-matrix-multiply-accumulate) PTX kernels require SM_80 or newer.** Everything else — scalar kernels, vector kernels, `cudaMallocManaged` unified memory, `cuLaunchKernel` dispatch, snapshot/restore, MPS — runs on SM_70 (Volta) and up.

| Compute capability | GPU examples | Status | What works | What does NOT work |
|---|---|---|---|---|
| SM_70 (Volta) | V100, Titan V | **Supported** | Unified memory, kernel dispatch, JIT, snapshots, MPS | wmma; async-copy intrinsics from S12 PTX |
| SM_72 (Xavier) | Jetson AGX Xavier | Untested | Same as SM_70 in theory | Same as SM_70 |
| SM_75 (Turing) | **RTX 2060 (dev box)**, RTX 2070/2080, T4, Quadro RTX | **Supported with caveat** | Unified memory, kernel dispatch, **non-wmma JIT**, snapshots, MPS | wmma PTX paths; `cp.async.bulk`; tensor-memory-accelerator intrinsics |
| SM_80 (Ampere data-center) | A100, A30 | **Fully supported** | Everything | — |
| SM_86 (Ampere consumer) | RTX 30xx series | **Fully supported** | Everything | — |
| SM_89 (Ada Lovelace) | **L4 (S22 runner)**, RTX 40xx, L40S | **Fully supported** | Everything | — |
| SM_90 (Hopper) | H100, H200 | **Fully supported** | Everything | — |
| Pre-SM_70 (SM_60, SM_61, SM_62) | P100, GTX 10xx | **Not supported** | n/a — `cudaMallocManaged` lacks the page-migration support TensorWasm requires | All TensorWasm paths |

### The SM_75 caveat in detail

On an RTX 2060 (SM_75), the following commands work:

```bash
export CUDA_ARCH=sm_75
cargo build --workspace --features tensor-wasm-mem/unified-memory
cargo test --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-wasi-gpu/cuda -- --include-ignored
```

But if you set `CUDA_ARCH=sm_80` to compile the wmma path on a Turing host, `nvcc` and `ptxas` will error at JIT-compile time with `Unsupported gpu architecture 'compute_80'` from the **driver**, because Turing tensor cores don't have the wmma int8/bfloat16 API surface SM_80 adds. The fix is to **leave `CUDA_ARCH=sm_75`** and accept that the wmma JIT blueprints are skipped on Turing — the dispatcher falls back to scalar paths automatically and tests in `tests/wasm-fixtures/wmma_matmul.rs` are skipped with `#[ignore = "requires sm_80"]` when the host capability is below SM_80.

The S22 runner is SM_89 and exercises every blueprint. PRs that touch wmma kernels must be validated against CI, not against an RTX 2060 dev box.

---

## MPS quick-start

For multi-tenant production on Linux with more than ~8 co-located tenants on the same GPU, run the NVIDIA Multi-Process Service daemon. Below is the minimum to bring it up; the full operations guide is in [MPS-SETUP.md](MPS-SETUP.md).

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
sudo mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
sudo chown "$USER" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

Then build TensorWasm with the MPS feature:

```bash
cargo build --workspace --features tensor-wasm-mem/unified-memory,tensor-wasm-tenant/cuda,tensor-wasm-tenant/mps
```

To stop the daemon:

```bash
echo quit | nvidia-cuda-mps-control
```

MPS is **Linux-only**. On Windows the `mps` feature compiles but `TenantRegistry::mps_or_fallback()` returns `Fallback` unconditionally. See [MPS-SETUP.md](MPS-SETUP.md) for capability requirements (`CAP_SYS_NICE`), per-tenant quota configuration, the 16-client Volta+ limit, and the systemd unit template.

---

## Troubleshooting

Error strings are quoted verbatim from `cust`, `cudarc`, `nvcc`, `ptxas`, and the CUDA driver. Match the left column against your error output exactly.

### Linker / loader failures

| Error string | Root cause | Fix |
|---|---|---|
| `libcuda.so: cannot open shared object file: No such file or directory` | Driver not installed, or `LD_LIBRARY_PATH` does not include the directory that holds `libcuda.so`. | Linux: `sudo apt-get install nvidia-driver-550` and reboot. Verify `ldconfig -p \| grep libcuda`. If the file is present but not found, `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. |
| `cuda.dll was not found` (Windows) | Driver not installed or `PATH` does not include `C:\Windows\System32` where the user-mode driver DLL lives. | Reinstall the driver from the official .exe. Confirm `nvidia-smi.exe` runs from a fresh shell. |
| `failed to run custom build command for cust` | `CUDA_ROOT` (or `CUDA_PATH` / `CUDA_HOME`) unset, points to a non-existent directory, or points to a directory missing `bin/nvcc`. | Re-check the [Required environment variables](#required-environment-variables) section. The order is `CUDA_ROOT` → `CUDA_PATH` → `CUDA_HOME`; first match wins. |
| `LINK : fatal error LNK1181: cannot open input file 'cuda.lib'` (Windows MSVC) | MSVC linker cannot find the stub library. The CUDA installer's `lib/x64` directory is missing from the linker search path. | Run `cargo build` from an `x64 Native Tools Command Prompt for VS 2022`, or run `vcvars64.bat` first. The cust build script appends `%CUDA_PATH%\lib\x64` only if the MSVC environment is loaded. |

### Driver / toolkit mismatches

| Error string | Root cause | Fix |
|---|---|---|
| `CUDA driver version is insufficient for CUDA runtime version` (error 35) | Toolkit is newer than driver. | Upgrade the driver to the minimum row in the [driver matrix](#nvidia-driver), or downgrade the toolkit. |
| `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (error 803) at first `cuInit` | Toolkit and driver from different CUDA major versions (e.g. CUDA 13 toolkit, CUDA 11 driver). | Match the major versions. The driver should always be ≥ the toolkit. |
| `ptxas not found` or `error: nvcc fatal : Could not find ptxas` | Toolkit not on `PATH`. | `export PATH="$CUDA_ROOT/bin:$PATH"` (Linux) or restart the shell after `setx PATH ...` (Windows). |
| `nvcc fatal : Unsupported gpu architecture 'compute_80'` | `CUDA_ARCH=sm_80` or higher on a SM_75 (Turing) host. | Set `CUDA_ARCH=sm_75` for RTX 2060 / T4 / 20-series. See [SM-level compatibility](#sm-level-compatibility-matrix). |
| `nvcc fatal : Unsupported gpu architecture 'compute_90'` on a 12.0 toolkit | Toolkit too old for the requested arch. | SM_90 requires CUDA 12.0+. SM_89 requires CUDA 11.8+. Upgrade the toolkit. |

### Runtime device problems

| Error string | Root cause | Fix |
|---|---|---|
| `no CUDA-capable device is detected` (error 100) | Driver loaded but no usable device. Common causes: device permission denied; running inside Docker without `--gpus`; running inside a container missing `nvidia-container-toolkit`; another process holds the GPU in exclusive mode. | Verify `nvidia-smi` works in the same shell. In Docker, add `--gpus all`. In Kubernetes, use the NVIDIA device plugin and request `nvidia.com/gpu: 1`. Check `nvidia-smi -q -d COMPUTE` for the compute mode (`Default` is what you want). |
| `CUDA_ERROR_NO_DEVICE` (error 100) on a headless server | `/dev/nvidia*` device nodes not created at boot (no X session, no `nvidia-modprobe`). | `sudo apt-get install nvidia-modprobe && sudo nvidia-modprobe -u -c=0`. Make this systemd-persistent in production. |
| `cuda runtime error: out of memory` | The GPU is full. On the RTX 2060 (12 GB VRAM) at SM_75, the default `wasm_memory: 4 GiB` plus `gpu_memory: 8 GiB` per tenant exhausts the device with a single tenant. | Reduce `wasm_memory` in the tenant config, reduce `gpu_memory`, or both. The minimum useful values are 256 MiB Wasm + 512 MiB GPU. See [PERFORMANCE.md](PERFORMANCE.md) for sizing guidance. |
| `cudaErrorIllegalAddress` during a JIT-emitted kernel launch | Almost always a generated-PTX bug, not a host-side bug. | File an issue with the kernel blueprint name, the input shape, and the `CUDA_ARCH` setting. Workaround: disable auto-offload by removing `tensor-wasm-jit/auto-offload` from the feature set. |

### WSL2-specific

| Error string | Root cause | Fix |
|---|---|---|
| `WSL/Windows could not load the dynamic library 'libcuda.so.1'` | Windows host driver too old for the WSL bind-mount, or WSL was started before the driver upgrade. | Update the Windows driver to 555.85 or later. Then in PowerShell: `wsl --shutdown`. Restart WSL. |
| `nvidia-smi: command not found` inside WSL | Standard `nvidia-utils` package was installed inside WSL — it does not work; WSL uses the host driver only. | `sudo apt-get purge nvidia-utils-*`. The correct `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi` and is bind-mounted from Windows. |

---

## One-shot verification script

Run this before any long CUDA build. It exercises every prerequisite and fails fast if any one is missing.

### Linux / WSL2 (`bash`)

```bash
#!/usr/bin/env bash
# Save as scripts/verify-cuda.sh; run as `bash scripts/verify-cuda.sh`.
set -euo pipefail

echo "== nvidia-smi =="
nvidia-smi || { echo "FAIL: nvidia-smi not found or no GPU"; exit 1; }

echo "== nvcc --version =="
nvcc --version || { echo "FAIL: nvcc not on PATH; check CUDA_ROOT/bin"; exit 1; }

echo "== ptxas --version =="
ptxas --version || { echo "FAIL: ptxas not on PATH"; exit 1; }

echo "== env vars =="
: "${CUDA_ROOT:?FAIL: CUDA_ROOT not set}"
: "${CUDA_ARCH:?FAIL: CUDA_ARCH not set (e.g. sm_75 for RTX 2060, sm_89 for L4)}"
echo "CUDA_ROOT=$CUDA_ROOT"
echo "CUDA_ARCH=$CUDA_ARCH"
[ -d "$CUDA_ROOT" ] || { echo "FAIL: CUDA_ROOT does not exist"; exit 1; }
[ -x "$CUDA_ROOT/bin/nvcc" ] || { echo "FAIL: $CUDA_ROOT/bin/nvcc not executable"; exit 1; }

echo "== libcuda visible =="
if ! { ldconfig -p | grep -q libcuda.so; } && ! [ -f /usr/lib/x86_64-linux-gnu/libcuda.so ] && ! [ -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
  echo "FAIL: libcuda.so not found by ldconfig or in standard locations"
  exit 1
fi

echo "== rustup toolchain =="
rustup show active-toolchain | grep -q "nightly-2026-04-03" || \
  { echo "WARN: rust-toolchain.toml pins nightly-2026-04-03; you are on a different toolchain"; }

echo "== smoke build (no-CUDA workspace) =="
cargo build --workspace --quiet || { echo "FAIL: workspace does not build without CUDA"; exit 1; }

echo "== smoke build (--features unified-memory) =="
cargo build --workspace --features tensor-wasm-mem/unified-memory --quiet || \
  { echo "FAIL: --features unified-memory does not link; check libcuda.so"; exit 1; }

echo "OK: CUDA toolchain ready for TensorWasm builds."
```

### Windows 11 (`PowerShell`)

```powershell
# Save as scripts/verify-cuda.ps1; run as `powershell -File scripts/verify-cuda.ps1`.
$ErrorActionPreference = 'Stop'

Write-Host "== nvidia-smi ==" -ForegroundColor Cyan
nvidia-smi
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: nvidia-smi not found or no GPU" }

Write-Host "== nvcc --version ==" -ForegroundColor Cyan
nvcc --version
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: nvcc not on PATH; check CUDA_PATH\bin" }

Write-Host "== ptxas --version ==" -ForegroundColor Cyan
ptxas --version
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: ptxas not on PATH" }

Write-Host "== env vars ==" -ForegroundColor Cyan
if (-not $env:CUDA_PATH) { Write-Error "FAIL: CUDA_PATH not set" }
if (-not $env:CUDA_ARCH) { Write-Error "FAIL: CUDA_ARCH not set (e.g. sm_75 for RTX 2060)" }
Write-Host "CUDA_PATH=$env:CUDA_PATH"
Write-Host "CUDA_ARCH=$env:CUDA_ARCH"
if (-not (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { Write-Error "FAIL: nvcc.exe not at $env:CUDA_PATH\bin" }
if (-not (Test-Path "$env:CUDA_PATH\lib\x64\cuda.lib")) { Write-Error "FAIL: cuda.lib not at $env:CUDA_PATH\lib\x64" }

Write-Host "== MSVC linker reachable ==" -ForegroundColor Cyan
& link.exe /? | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Error "FAIL: link.exe not on PATH. Run vcvars64.bat or use an x64 Native Tools Command Prompt for VS 2022."
}

Write-Host "== rustup toolchain ==" -ForegroundColor Cyan
$active = (rustup show active-toolchain).Split(' ')[0]
if ($active -notlike "*nightly-2026-04-03*") {
  Write-Warning "rust-toolchain.toml pins nightly-2026-04-03; you are on $active"
}

Write-Host "== smoke build (no-CUDA workspace) ==" -ForegroundColor Cyan
cargo build --workspace --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: workspace does not build without CUDA" }

Write-Host "== smoke build (--features unified-memory) ==" -ForegroundColor Cyan
cargo build --workspace --features tensor-wasm-mem/unified-memory --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "FAIL: --features unified-memory does not link; check cuda.lib + CUDA_PATH" }

Write-Host "OK: CUDA toolchain ready for TensorWasm builds." -ForegroundColor Green
```

---

## Stub libraries for CI

GitHub-hosted runners have no GPU. The Craton TensorWasm CI workflow does **not** install the real CUDA toolkit on hosted runners. Instead, `.github/workflows/ci.yml` drops a directory of stub `.so` files at `/usr/local/cuda/lib64/` containing only the symbols `cust` resolves at link time (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`, etc.) — each a no-op exported from a tiny C shim. This is enough to satisfy the linker so the workspace builds and unit tests that do not launch kernels can run.

Tests that actually launch kernels are marked `#[ignore = "requires CUDA hardware"]` and skipped on hosted runners. They execute on the S22 self-hosted runner, which has the real toolkit installed per the matrix at the top of this document.

---

## Cross-references

- [`docs/BUILD.md`](BUILD.md) — full feature-flag taxonomy across all 10 crates, build matrix, test tiers, `make ci` parity.
- [`docs/MPS-SETUP.md`](MPS-SETUP.md) — full NVIDIA MPS operations guide (daemon, capabilities, limits, systemd unit).
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — measured numbers, sizing guidance for `wasm_memory` / `gpu_memory`, SKU-specific baselines.
- [`docs/RISKS.md`](RISKS.md) — v0.1.0 known limitations, the `cust → cudarc` migration timeline, and tracked upstream issues.
- [`docs/CUDARC-SPIKE.md`](CUDARC-SPIKE.md) — the cust → cudarc migration spike: API mapping, parallel-backend strategy, cutover gates.
- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — v0.2 milestone exit criteria, including the S22 runner provisioning that this document targets.
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) — upstream reference.
- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) — upstream reference.
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) — driver matrix.

---
_Updated for tensor-wasm v0.2 (PATH-TO-V1 milestone, S22 runner provisioning). Re-verify the driver matrix and the S22 runner toolkit version before every release; bump the recommended Linux GCC / Windows MSVC pins to match the S22 runner image when it is refreshed._
