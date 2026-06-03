<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Craton TensorWasm

### Run untrusted code. On the GPU. Safely. At serverless speed.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/release-v0.3.7-green.svg)](CHANGELOG.md)
[![Built in Rust](https://img.shields.io/badge/built_in-Rust-orange.svg)](https://www.rust-lang.org/)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76b900.svg)](docs/CUDA-SETUP.md)

> **Craton TensorWasm is a GPU-accelerated, multi-tenant serverless WebAssembly runtime.**
> It runs sandboxed `.wasm` workloads that talk directly to NVIDIA GPUs through a
> typed host interface — with auth, quotas, observability, and ops built in from day one.

<p align="center">
  <a href="#-get-started-in-5-minutes"><b>🚀 Get Started</b></a> &nbsp;·&nbsp;
  <a href="#-why-teams-choose-tensorwasm"><b>✨ Why TensorWasm</b></a> &nbsp;·&nbsp;
  <a href="#-where-it-runs"><b>📦 Deploy</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/craton-co/craton-tensor-wasm"><b>⭐ GitHub</b></a>
</p>

---

## The problem

Modern AI and data workloads want two things that have always been at odds:

- **The isolation of a sandbox** — so you can run untrusted or multi-tenant code without it touching the host or its neighbours.
- **The raw throughput of the GPU** — so the work actually finishes on time.

Traditional sandboxes give you safety but keep you on the CPU. Hand-written GPU
code gives you speed but no isolation. **TensorWasm gives you both, in one runtime.**

---

## ✨ Why teams choose TensorWasm

### 🛡️ Sandboxed by construction
Every workload is a WebAssembly module isolated by Wasmtime. Untrusted code stays
in its lane — memory-safe, capability-gated, and deadline-enforced. No escape hatches.

### ⚡ GPU-native, not GPU-adjacent
Guests reach the GPU through a typed `wasi:cuda` interface: explicit kernel
dispatch today, opt-in automatic offload tomorrow. Wasm linear memory is backed
by CUDA Unified Memory, so data is reachable from the GPU **without a copy**.

### 🏢 Multi-tenant from the first line of code
One process, many tenants — each with scoped bearer tokens, per-token rate limits,
and per-tenant GPU memory quotas. Isolation isn't a deployment pattern you bolt on;
it's the architecture.

### 📊 Production-ready, not just a demo
Prometheus metrics, OpenTelemetry traces propagated end-to-end, a drop-in Grafana
dashboard, structured audit logs, published SLOs, and one runbook per alert. The
boring parts that decide whether you can actually operate it — already done.

### 🚀 Fast cold-starts
A snapshot subsystem captures and restores Wasm + GPU state, so cycling many small
functions doesn't mean paying full instantiation cost every time.

### 🔓 Truly open source
Apache-2.0. Commercial use, modification, and redistribution all permitted, with a
permissive trademark policy. No open-core bait-and-switch.

---

## 🔬 Proven on real silicon

This isn't a whitepaper. The full path — **Wasm guest → `wasi:cuda` → `cuLaunchKernel`
→ read results back** — runs end-to-end on a real NVIDIA GPU (RTX 2060), with tests
asserting the GPU actually computed the right answer.

| What | Status |
|---|---|
| End-to-end GPU dispatch with typed arguments | ✅ Passing on real hardware |
| Pure-CPU execution speed | ✅ Statistically tied with upstream Wasmtime 45 |
| HTTP gateway, auth, audit, metrics | ✅ Shipped and exercised |
| 11-crate workspace, 9 tagged releases (`v0.1.0` → `v0.3.7`) | ✅ 0 open **external**-audit findings (the external audit is itself roadmap — see [PATH-TO-V1](docs/PATH-TO-V1.md)) |

We publish where we **win** and where we **lose** — see the
[honest benchmarking guide](docs/BENCHMARKING.md). Credibility is the marketing.

---

## 🧩 What you can build

- **Serverless GPU inference** — host small models behind an OpenAI-compatible
  `/v1/completions` and `/v1/chat/completions` endpoint, with streaming responses.
- **Untrusted compute marketplaces** — let third parties submit `.wasm` jobs that
  use the GPU, without ever trusting their code with your host.
- **Multi-tenant AI platforms** — give every customer isolated GPU budget,
  rate limits, and audit trails out of one fleet.
- **Edge / embedded AI** — compile workloads to `wasm32-wasip1` once, run them
  anywhere TensorWasm runs, with or without a GPU.

---

## 📦 Where it runs

Ship it the way your platform already works:

| Surface | What you get |
|---|---|
| **Kubernetes** | Plain manifests *and* a templated [Helm chart](deploy/helm/tensor-wasm/README.md) |
| **Nomad** | Reference job specs for both `docker` and `raw_exec` drivers |
| **Docker / Compose** | First-class `Dockerfile` and `docker-compose.yml` |
| **Bare binary** | A single `tensor-wasm` CLI with shell completions + man pages |

Observability is included, not sold separately: Prometheus, OpenTelemetry,
a Grafana dashboard, audit logs, SLOs, and runbooks all ship in the repo.

---

## 🚀 Get started in 5 minutes

```sh
git clone https://github.com/craton-co/craton-tensor-wasm
cd craton-tensor-wasm

# Build and test the whole workspace (no GPU required)
cargo build --workspace
cargo test  --workspace

# Run a Wasm function locally
cargo run -p tensor-wasm-cli -- run tests/wasm-fixtures/matrix_multiply.wat

# Spin up the HTTP API
cargo run --release --bin tensor-wasm -- serve --addr 0.0.0.0:8080
```

Have an NVIDIA GPU? Add `--features cuda` and watch a real kernel launch compute
on your hardware. Full setup in [CUDA-SETUP.md](docs/CUDA-SETUP.md).

> **Heads up:** the host-only path runs on **stable Rust ≥ 1.78**. The repository
> pins a Rust nightly only to align its dev/CI toolchain with the optional
> cuda-oxide backend — published crates build on stable. See [BUILD.md](docs/BUILD.md).

---

## 🛣️ Where it's headed

- **Today (v0.3.7)** — real GPU dispatch proven on hardware; auth, multi-tenancy,
  observability, and ops complete. A pre-certification internal audit found and
  **resolved on `dev`** a HIGH cross-tenant isolation issue plus several
  medium/low items — all tracked in [docs/RISKS.md](docs/RISKS.md). No open
  external-audit findings (the external audit is itself roadmap, below).
- **Next** — the cuda-oxide backend cutover and a Cranelift→PTX auto-offload
  pipeline, so more workloads reach the GPU with zero guest changes.
- **Toward v1.0** — an external security audit, design partners, and an API freeze.

The roadmap is public and honest about dates: see [PATH-TO-V1.md](docs/PATH-TO-V1.md).

---

## ❓ FAQ

**Is it production-ready?**
For pilots, staging, and internal tooling — yes, today. For v1.0-grade SLAs — the
beta arrives with the design-partner program. The GPU path runs on real hardware now.

**How is this different from Wasmtime / Wasmer / Spin?**
TensorWasm *wraps* Wasmtime (it's not a fork) and adds the GPU, multi-tenancy,
snapshots, the HTTP gateway, and observability on top. FaaS platforms like Spin and
Wasmer Edge are products you could build *on* a runtime like this one.

**What hardware do I need?**
NVIDIA CUDA (12.0+) for the GPU path; SM_70+ for standard kernels. The pure-CPU
path runs anywhere Wasmtime runs — no GPU required.

**What's the license?**
Apache-2.0 — commercial use, modification, and redistribution all permitted.
See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

---

## 🤝 Get involved

- **Developers** — substantive changes go through the lightweight [RFC process](rfcs/README.md); bug fixes just open a PR. Start with [CONTRIBUTING.md](CONTRIBUTING.md).
- **Companies** — the design-partner program is open: early access and named credit in exchange for production validation.
- **Security researchers** — coordinated disclosure is documented in [SECURITY.md](SECURITY.md).

Reach the team at **`security@craton.com.ar`**.

---

<p align="center">
  <b>Craton TensorWasm</b> — GPU-accelerated serverless WebAssembly, in Rust.<br>
  <a href="https://github.com/craton-co/craton-tensor-wasm">github.com/craton-co/craton-tensor-wasm</a><br>
  Apache-2.0 &nbsp;·&nbsp; © 2026 Craton Software Company
</p>
