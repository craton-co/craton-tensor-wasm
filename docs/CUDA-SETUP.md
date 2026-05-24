# CUDA Setup

Craton TensorWasm's GPU-resident crates — `tensor-wasm-mem`, `tensor-wasm-wasi-gpu`, `tensor-wasm-jit`, and `tensor-wasm-tenant` — all link against the CUDA driver API and CUDA runtime via the [`cust`](https://docs.rs/cust) crate. To build and test these crates locally you need a working CUDA toolkit, a matching NVIDIA driver, and a handful of environment variables wired up so `nvcc`, `ptxas`, and the `cust` build script can find the headers and stub libraries. This document walks through toolkit installation across the supported host operating systems, the driver-version matrix Craton TensorWasm validates against, the required environment variables, and a smoke test you can run before opening a PR.

## Toolkit installation

### Ubuntu 22.04 / 24.04

Follow NVIDIA's [APT repository setup](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu) to register the `cuda-keyring` package, then install the pinned toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

Reboot once after the driver package lands so `nvidia.ko` is loaded against the running kernel.

### Windows 11

Download the network installer from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and run it with the default component set. **The MSVC toolchain (Visual Studio 2022 Build Tools, "Desktop development with C++") is required** — `cust` and its `find_cuda_helper` build script invoke `link.exe` for the stub libraries and will fail under MinGW or Clang-Cl. After install, open a fresh "x64 Native Tools Command Prompt for VS 2022" before running `cargo build` so the MSVC environment is on `PATH`.

### Arch / Fedora notes

Arch users can install `cuda` from the `extra` repository (`sudo pacman -S cuda`); the package installs to `/opt/cuda`, so `CUDA_ROOT` must be set accordingly. Fedora 40+ uses NVIDIA's `rhel9` repo — install via `sudo dnf install cuda-toolkit-12-6`. Both distributions ship driver packages out-of-tree (`nvidia-dkms` on Arch, `akmod-nvidia` on Fedora/RPMFusion); rebuild the kernel module after every kernel upgrade.

## Driver version matrix

The CUDA runtime is forward-compatible only against drivers at or above the toolkit's minimum. Craton TensorWasm CI matrix-tests against CUDA 12.0 and 12.4 stubs; 12.6 is the recommended local-dev version.

| CUDA Toolkit | Linux driver  | Windows driver |
|--------------|---------------|----------------|
| 12.0         | >= 525.60.13  | >= 527.41      |
| 12.4         | >= 550.54.14  | >= 551.61      |
| 12.6         | >= 560.28.03  | >= 560.81      |

If `nvidia-smi` reports a driver below the row for your installed toolkit, either upgrade the driver or downgrade the toolkit — mismatches surface as `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (error 803) at the first `cuInit` call inside `tensor-wasm-mem`.

## Required environment variables

Craton TensorWasm's build scripts read four variables. Set them globally (shell profile, `setx`) rather than per-shell so `rust-analyzer` and your IDE pick them up too.

| Variable          | Purpose                                                                                       |
|-------------------|-----------------------------------------------------------------------------------------------|
| `CUDA_ROOT`       | Absolute path to the toolkit install root.                                                    |
| `CUDA_ARCH`       | Target compute capability for PTX emission (e.g. `sm_80`). S12 default is `sm_80`.            |
| `LD_LIBRARY_PATH` | Must include `$CUDA_ROOT/lib64` on Linux so the dynamic loader finds `libcuda.so`.            |
| `PATH`            | Must include `$CUDA_ROOT/bin` so `nvcc` and `ptxas` are reachable to the JIT crate.           |

`CUDA_ARCH` accepts the values you'd pass to `nvcc -arch=`: `sm_80` for A100, `sm_86` for RTX 30xx (Ampere consumer), `sm_89` for L4 / L40S / RTX 40xx, and `sm_90` for H100. The S12 PTX emitter defaults to `sm_80` because that's the lowest capability that supports the async-copy and tensor-memory-accelerator intrinsics `tensor-wasm-jit` lowers to.

### Linux (bash / zsh)

```bash
export CUDA_ROOT=/usr/local/cuda
export CUDA_ARCH=sm_80
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
```

Drop the above into `~/.bashrc` or `~/.zshrc`, then `source` it.

### Windows (PowerShell, persistent)

```powershell
setx CUDA_ROOT "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
setx CUDA_ARCH "sm_80"
setx PATH "%PATH%;%CUDA_ROOT%\bin"
```

`setx` writes to the user registry; close and reopen your shell for the values to take effect. `LD_LIBRARY_PATH` is not used on Windows — the toolkit installer adds `%CUDA_ROOT%\bin` (which holds the DLLs) to `PATH` for you.

## Verifying the toolchain

Run each command in order; every step should print a version banner without errors.

```bash
nvidia-smi             # driver loaded, GPU enumerated, processes column visible
nvcc --version         # toolkit on PATH; should match the version you installed
ptxas --version        # PTX assembler reachable; tensor-wasm-jit (S12) depends on this
```

Then from the repository root, build the workspace with the GPU features turned on:

```bash
cargo build --workspace --features unified-memory
```

A clean build of `tensor-wasm-mem`, `tensor-wasm-wasi-gpu`, and `tensor-wasm-jit` against a real toolkit is the canonical smoke test. If `cargo` fails at the `cust-build` stage with `could not find libcuda`, re-check that `CUDA_ROOT/lib64` (or `CUDA_ROOT\lib\x64` on Windows) is on the loader path.

## Stub libraries for CI

GitHub-hosted runners have no GPU, so the Craton TensorWasm CI workflow does **not** install the real CUDA toolkit. Instead, `.github/workflows/ci.yml` drops a directory of stub `.so` files at `/usr/local/cuda/lib64/` containing only the symbols `cust` resolves at link time (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`, etc.) — each as a no-op exported from a tiny C shim. This is enough to satisfy the linker so the workspace builds, but any test that actually launches a kernel is marked `#[ignore = "requires CUDA"]` and skipped on CI. Running real CUDA kernels in CI requires self-hosted runners with attached GPUs, which is out of scope until S22 (the perf-regression milestone).

## Common pitfalls

- **Mixing toolkit and driver from different major versions.** A CUDA 12 toolkit will not load against an 11.x driver, and vice versa. Always check the matrix above before upgrading either component in isolation.
- **Missing `nvidia-modprobe`.** On headless Linux servers without a logged-in desktop session, the first `cuInit` call fails with `CUDA_ERROR_NO_DEVICE` (error 100) because `/dev/nvidia*` nodes haven't been created. Install the `nvidia-modprobe` package (Ubuntu) or run `sudo nvidia-modprobe -u -c=0` once at boot.
- **WSL2 users.** Do not install a Linux NVIDIA driver inside the WSL distro. Install `nvidia-driver-windows-host` on the Windows side and `cuda-toolkit-wsl-ubuntu` inside WSL — the toolkit talks to the host driver through `/usr/lib/wsl/lib/libcuda.so.1`.
- **macOS is not supported.** NVIDIA stopped shipping macOS drivers in 2018 and Apple Silicon has no PCIe GPU path. Use a Linux workstation, a cloud VM (e.g. `g5.xlarge` on AWS, `n1-standard-4` + T4 on GCP), or remote into a Windows host.

## References

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [`cust` crate docs on docs.rs](https://docs.rs/cust)

---
_Updated for tensor-wasm v0.1.0 (S2 of plan). Re-verify driver matrix before each release._
