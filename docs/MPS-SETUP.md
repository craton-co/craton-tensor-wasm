# NVIDIA Multi-Process Service (MPS) Setup

Craton TensorWasm's `tensor-wasm-tenant` crate (S16) uses NVIDIA Multi-Process Service to back the `ContextIsolated` isolation tier with a low-overhead shared CUDA context. When the MPS control daemon is running, tenants share a single GPU context that the daemon time-slices between them; when MPS is not available — on Windows, on CI hosts without the toolkit, or on Linux hosts where the daemon was never started — the registry falls back to per-tenant `cuCtxCreate` calls. This document covers daemon startup, the operating-system capabilities the daemon needs, the practical limits of MPS, and how TensorWasm probes for it at runtime.

## When to use MPS vs the per-context fallback

Without MPS, each `ContextIsolated` tenant pays the cost of a full CUDA context: a few hundred milliseconds at create time, ~30 MB of resident GPU memory for driver bookkeeping, and a context-switch on every kernel launch that crosses tenants. That is fine for a handful of tenants but it does not scale — by the time you reach a couple of dozen co-located instances the context-switch overhead dominates dispatch latency. MPS collapses all of that onto a single context: the daemon multiplexes client work onto shared GPU hardware queues, eliminating the switch and reducing the per-tenant memory tax to a few megabytes. The trade-off is that an MPS client cannot use the GPU debugger and the daemon is itself a privileged process you must operate. For single-tenant deployments or development on a workstation, leave MPS off and let `TenantRegistry::mps_or_fallback()` return `MpsDecision::Fallback`. For multi-tenant production with more than ~8 co-located tenants on the same GPU, configure MPS.

## Starting the daemon (Linux)

The daemon is shipped with the CUDA toolkit as `nvidia-cuda-mps-control`. Pick a directory for the daemon's control pipe and log files, export the two environment variables MPS expects, then start the daemon in background mode:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

`tensor-wasm-tenant` probes `/tmp/nvidia-mps` (the value of [`tensor_wasm_tenant::MPS_CONTROL_PATH`]) at registry construction time; if you point `CUDA_MPS_PIPE_DIRECTORY` somewhere else you also need to symlink or override the probe path before deploying. To stop the daemon cleanly so it flushes per-client state, send `quit` on its control channel:

```bash
echo quit | nvidia-cuda-mps-control
```

A systemd unit that runs the daemon as a dedicated `nvidia-mps` user, redirects stderr to the journal, and depends on `nvidia-persistenced.service` is the recommended production setup; see the SRE handbook for the exact unit file.

## Required Linux capabilities

The MPS daemon needs `CAP_SYS_NICE` to lower its own scheduling priority and to manipulate the SIGSTOP/SIGCONT signals it uses to gate client processes. On a default Linux install running as root this is implicit; under hardening (e.g. an unprivileged user with `setcap`'d binaries, or a non-privileged container) you must grant the capability explicitly:

```bash
sudo setcap cap_sys_nice+ep "$(command -v nvidia-cuda-mps-control)"
sudo setcap cap_sys_nice+ep "$(command -v nvidia-cuda-mps-server)"
```

Inside containers, the orchestrator must pass `--cap-add SYS_NICE` (Docker) or set `securityContext.capabilities.add: ["SYS_NICE"]` (Kubernetes) on the pod running the daemon. The MPS *clients* — that is, `tensor-wasm-cli` and the `tensor-wasm-api` server processes — do not need this capability.

## Feature limitations

MPS is not a substitute for hardware partitioning. The most important constraints to remember when planning a deployment:

- **No on-the-fly resize.** Per-client thread-percent and memory limits are set when the client first connects. To change a tenant's share you must `quit` the daemon, edit `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` or per-client overrides, and restart — every existing client loses its context. The TensorWasm scheduler treats MPS quotas as immutable for the lifetime of a tenant registration.
- **Maximum 16 concurrent clients on Volta+ (V100, A100, H100).** Pre-Volta architectures cap at 48 clients with reduced parallelism. If you expect more than 16 co-located tenants, partition them across multiple GPUs with MIG or stripe across multiple MPS daemons on separate `CUDA_VISIBLE_DEVICES` masks.
- **Windows is not supported.** The daemon is Linux-only. On Windows, `TenantRegistry::mps_or_fallback()` always returns `Fallback`; `ContextIsolated` tenants get `cuCtxCreate` contexts directly. This is fine for development and small deployments but not viable for high-tenant-count production.
- **No CUDA-graphics interop.** OpenGL / DirectX interop is disabled under MPS. TensorWasm kernels are pure compute, so this does not affect us, but be aware if you embed TensorWasm in a larger application that wants to render with the same GPU.
- **Single-user by default.** All MPS clients must share a UID with the daemon unless you run the daemon in exclusive-process compute mode and configure per-user pipe directories. The TensorWasm deployment template runs the API server as the `nvidia-mps` user for this reason.

## Cross-references

- `docs/CUDA-SETUP.md` — toolkit installation, driver-version matrix, and environment variables consumed by `cust`. Read this first; MPS requires a working toolkit.
- `SECURITY.md` — threat model that motivates the `IsolationKind` taxonomy. In particular, MPS provides software-level isolation only; for hardware-enforced isolation use MIG, which is documented separately in the SRE handbook.
