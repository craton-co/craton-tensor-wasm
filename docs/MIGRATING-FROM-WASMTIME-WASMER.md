<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Migrating from Wasmtime / Wasmer to Craton TensorWasm

v0.4 *"Migrating from Wasmtime/Wasmer to TensorWasm guide"* from
[`PATH-TO-V1.md`](PATH-TO-V1.md) (Documentation workstream).
Written for developers and operators who already run upstream
Wasmtime (as a library or via `wasmtime-cli`), Wasmer, or a Wasmer
Edge / Spin / custom Wasmtime-based FaaS, and who are evaluating
moving some or all of that workload onto Craton TensorWasm.

Not a marketing piece. The goal is an honest answer to "should I
switch?" — including the cases where the answer is "no, stay on
what you have." TensorWasm is a thin opinionated serverless layer
over upstream Wasmtime; not a replacement for the runtime you
already trust, but the right runtime when your problem shape
includes GPU dispatch, multi-tenant isolation, an HTTP gateway,
and audit-grade observability without rolling your own.

If you only read one section: skip to
[When TensorWasm is NOT the right choice](#5-when-tensorwasm-is-not-the-right-choice).

## Contents

[1. Who this guide is for](#1-who-this-guide-is-for) ·
[2. What TensorWasm IS and ISN'T](#2-what-tensorwasm-is-and-isnt) ·
[3. Feature matrix](#3-side-by-side-feature-matrix) ·
[4. When TensorWasm is right](#4-when-tensorwasm-is-the-right-choice) ·
[5. When TensorWasm is NOT right](#5-when-tensorwasm-is-not-the-right-choice) ·
[6. Library-user migration](#6-library-user-migration-embedded-wasmtime) ·
[7. CLI-user migration](#7-cli-user-migration-wasmtime-run--wasmer-run) ·
[8. Server-user migration](#8-server-user-migration-wasmer-edge--spin--custom-faas) ·
[9. GPU migration](#9-gpu-migration-no-precedent--raw-cuda--triton--cudarc--wgpu) ·
[10. Operational diff](#10-operational-diff--what-you-give-up-what-you-gain) ·
[11. What stays the same](#11-what-stays-the-same) ·
[12. FAQs](#12-faqs) ·
[13. Where to go next](#13-where-to-go-next)

## 1. Who this guide is for

Three personas, each with a different migration shape.

### 1.1 The library user

You import `wasmtime` (or `wasmer`) as a crate in a Rust service —
`Engine`, `Module`, `Linker`, `Instance::call_typed(...)` from your
own request handler. The Wasm runtime is a building block, not the
whole product.

**Likely outcome.** For most pure-CPU library use, stay where you
are. `tensor-wasm-exec` is deliberately narrower than `wasmtime`'s
Rust API. Switch if you want the bundled HTTP gateway, snapshots,
tenant registry, and WASI-GPU host bridge; otherwise the closest
TensorWasm equivalent is a step down in flexibility. See
[§6](#6-library-user-migration-embedded-wasmtime).

### 1.2 The CLI user

You run `wasmtime run foo.wasm` (or `wasmer run foo.wasm`) from a
shell, a Makefile, or CI. You may also use AOT compile.

**Likely outcome.** `tensor-wasm run` is feature-parity for the
*launch-a-`.wasm`-and-watch-it-finish* case **with one caveat**:
the v0.1.0 CLI only invokes `() -> ()` exports. If your guest
takes arguments, TensorWasm accepts and validates `--args` JSON
but does *not* pass values through — `call_export` only supports
the no-arg signature (verified in
[`crates/tensor-wasm-cli/src/cmd/run.rs`](../crates/tensor-wasm-cli/src/cmd/run.rs)).
Argument lowering is a v0.2 follow-up; do not migrate arg-taking
CLI workloads yet. See
[§7](#7-cli-user-migration-wasmtime-run--wasmer-run).

### 1.3 The server user

You operate Wasmer Edge, Spin, Fermyon Cloud, workerd on bare
metal, or a homegrown Wasmtime-based FaaS where every request
lands in a fresh instance. You care about cold-start, per-tenant
isolation, auth, audit, metrics, and a deployment story.

**Likely outcome.** This is the persona TensorWasm was built for.
The HTTP API ([`API.md`](../crates/tensor-wasm-api/API.md)),
snapshot subsystem ([`COLD-START.md`](COLD-START.md)), tenant
registry, audit log ([`AUDIT-LOG.md`](AUDIT-LOG.md)), SLO +
Grafana dashboard ([`SLO.md`](SLO.md),
[`dashboards/`](dashboards/)), and Helm-chart deployment
([`tutorials/production-deployment.md`](tutorials/production-deployment.md))
exist so you do not have to rebuild this glue per deployment.
See [§8](#8-server-user-migration-wasmer-edge--spin--custom-faas).

## 2. What TensorWasm IS and ISN'T

Read this before any code sample. The biggest source of
disappointed users would be thinking TensorWasm *replaces*
Wasmtime instead of *building on* it.

**TensorWasm IS:** a serverless runtime built on upstream Wasmtime
(pinned at `25.x` per
[`WASMTIME-UPGRADE.md`](WASMTIME-UPGRADE.md)), wrapping it with
HTTP gateway, snapshots, tenant registry, audit, metrics; a
GPU-aware Wasm runtime via `wasi:cuda/host@0.2.0`
([`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit)) — the only such
surface we know of that ships with a runtime rather than as a
research patch; Apache-2.0, self-hosted; compatible with the same
`.wasm` modules upstream Wasmtime runs.

**TensorWasm IS NOT:** a Wasmtime fork
([`WASMTIME-FORK.md`](WASMTIME-FORK.md)); a replacement for
Wasmtime-as-library (`tensor-wasm-exec` is narrower than
`wasmtime::Linker`); a Wasmer-LLVM competitor on tight loops
([`BENCHMARKING.md`](BENCHMARKING.md#where-tensorwasm-wins-where-it-wont));
an edge-network platform; WASI Preview 3 yet (P2 only at v1.0).

## 3. Side-by-side feature matrix

Thirteen rows. "Wasmtime (library)" is `cargo add wasmtime`;
"Wasmtime (CLI)" is `wasmtime-cli`; "Wasmer" is the unified
`wasmer-cli` + Wasmer Edge combo (they share the compiled artifact
format).

| Capability | Wasmtime (library) | Wasmtime (CLI) | Wasmer | Craton TensorWasm |
|---|---|---|---|---|
| WASI Preview 2 | yes | yes | yes | yes |
| WASI Preview 3 / async components | pending upstream | pending upstream | pending upstream | not yet (v2 per [PATH-TO-V1](PATH-TO-V1.md#anti-goals--what-v10-does-not-promise)) |
| Component model | yes | yes | yes | yes |
| GPU dispatch (WASI-GPU) | no | no | no | yes (CUDA today; AMD / Intel / Apple deferred to v2) |
| HTTP API gateway | no (BYO) | no | WebSocket gateway only | yes (axum, see [`API.md`](../crates/tensor-wasm-api/API.md)) |
| Multi-tenant context isolation | no (BYO) | no | no | yes (`TenantRegistry` + per-context CUDA streams; optional MPS) |
| Cold-start snapshots | manual `Module::serialize` / `deserialize` | via `--allow-precompiled` | Wasmer Universal artifacts (`.wasmu`) | yes (streaming zstd + bincode, see [`COLD-START.md`](COLD-START.md)) |
| Audit log | n/a | n/a | n/a | yes (structured JSONL, see [`AUDIT-LOG.md`](AUDIT-LOG.md)) |
| Rate limiting per token | n/a | n/a | n/a | yes (token-bucket per bearer, see [`API.md`](../crates/tensor-wasm-api/API.md) "Per-token rate limiting") |
| Authentication / scoped tokens | n/a | n/a | n/a | yes (bearer + `tenant=...` scope) |
| Prometheus metrics | n/a (BYO) | n/a | n/a | yes (`GET /metrics`, exposition v0.0.4) |
| SLOs published | n/a | n/a | n/a | yes ([`SLO.md`](SLO.md)) |
| Maintainer / project size | Bytecode Alliance (large) | Bytecode Alliance | Wasmer Inc | Craton Software Company (small) |

Cells marked **n/a** are out of scope for that runtime shape, not
verdicts; the matrix catalogues what ships *bundled*. "Yes" is not
"best-in-class" — TensorWasm inherits Wasmtime's component-model
and WASI P2 unchanged. The GPU row is the only column where
TensorWasm is uniquely "yes"; everything else is also-have.

## 4. When TensorWasm is the right choice

Pick TensorWasm if at least two of these hold. Single-criterion
matches usually have a better answer elsewhere
([§5](#5-when-tensorwasm-is-not-the-right-choice)).

- **You need a GPU-aware Wasm runtime.** No other production-shaped
  option exists today. WebGPU shaders are not the same model (no
  `cuLaunchKernel`-style explicit dispatch from inside the sandbox);
  `wasmtime-cuda`-style patches exist in research repos but do not
  ship as tested runtimes. TensorWasm's
  [`wasi:cuda/host@0.2.0`](../wit/wasi-cuda.wit) gives the guest
  `load_ptx`, `launch`, `sync`, and `last_error`, with bounds
  checks before any driver call and a back-pressure semaphore.
  **Caveat:** v0.1.0 reaches the driver only for zero-argument
  launches; non-empty kernel args return `KernelArgsUnsupported`
  ([`RISKS.md`](RISKS.md)); v0.2 removes the restriction
  ([`PATH-TO-V1.md`](PATH-TO-V1.md#v020--real-cuda)).
- **You want a self-hosted serverless gateway without rolling your
  own.** The [`tensor-wasm-api`](../crates/tensor-wasm-api/API.md)
  axum gateway ships with bearer auth, per-tenant scoped tokens, a
  64 MiB body cap, a token-bucket rate limiter per bearer,
  structured audit records, and OpenTelemetry tracing with W3C
  `traceparent` propagation.
- **Multi-tenant isolation matters.** `TenantRegistry` enforces
  per-tenant CUDA contexts (optionally MPS-backed,
  [`MPS-SETUP.md`](MPS-SETUP.md)) and a quota gate; the gateway
  authorizes `X-TensorWasm-Tenant` against the bearer's scope
  before any executor work runs.
- **You want operability without re-deriving it.** Reference
  Grafana dashboard ([`dashboards/`](dashboards/)), SLOs with
  burn-rate alerts ([`SLO.md`](SLO.md)), runbooks
  ([`runbooks/`](runbooks/)), Helm chart
  ([`deploy/helm/tensor-wasm/`](../deploy/helm/tensor-wasm/)), and
  the
  [production-deployment tutorial](tutorials/production-deployment.md)
  ship with the runtime.

## 5. When TensorWasm is NOT the right choice

- **Fastest possible pure-CPU Wasm execution.** Use **Wasmer-LLVM**.
  LLVM compiles 5-20x slower than Cranelift but runs tight inner
  loops faster. Published loss in
  [`BENCHMARKING.md`](BENCHMARKING.md#where-tensorwasm-wins-where-it-wont).
- **A Wasm library to embed in your own service.** Use **Wasmtime
  directly**. `tensor-wasm-exec` is built for the serverless
  lifecycle; long-lived "one module, many calls" loses the
  snapshot benefit and gives you a narrower API than
  `wasmtime::Linker`.
- **Cloudflare-network-scale edge.** Use **workerd on Cloudflare**.
  TensorWasm is a single-host runtime; the network is what makes
  Workers fast at the edge.
- **WASI Preview 3 / async components today.** Wait. P2 only at
  v1.0 ([`PATH-TO-V1.md`](PATH-TO-V1.md#out-of-scope--deferred-to-v20)).
- **AMD / Intel / Apple GPU backends today.** Wait. v1.0 is NVIDIA
  CUDA only; v2 will ship vendor abstraction.
- **A stable Rust toolchain.** We are pinned to
  `nightly-2026-04-03` through v1.0 with a quarterly bump cadence.
  Stable Rust is a v2 effort.

## 6. Library-user migration (embedded Wasmtime)

Most library users should not migrate (see [§1.1](#11-the-library-user)).
The examples below give the ones who decide they want it a runnable
starting point.

### 6.1 The Wasmtime library shape you have today

Typical Wasmtime embedding against the current TensorWasm pin
(`wasmtime 25.x`; the same shape works in `wasmtime 45.x` with
minor renames):

```rust
// Cargo.toml: wasmtime = "25"
use wasmtime::{Config, Engine, Linker, Module, Store};

fn run_once(wasm: &[u8]) -> anyhow::Result<()> {
    let mut config = Config::new();
    config.async_support(true);
    config.epoch_interruption(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wasm)?;
    let mut linker: Linker<()> = Linker::new(&engine);
    wasmtime_wasi::add_to_linker_sync(&mut linker, |s| s)?;
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    instance.get_typed_func::<(), ()>(&mut store, "_start")?.call(&mut store, ())?;
    Ok(())
}
```

You own the `Config`, `Linker`, `Store` state type, and import
table.

### 6.2 The equivalent TensorWasm shape

The executor flattens that into a single call. You give up `Linker`
customization (TensorWasm pre-wires WASI P2 + `wasi:cuda/host@0.2.0`)
and gain tenant binding, back-pressure semaphore, and snapshot
hooks for free:

```rust
// Cargo.toml: tensor-wasm-core = "0.1"; tensor-wasm-exec = "0.1"
use std::sync::Arc;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

async fn run_once(wasm: &[u8]) -> anyhow::Result<()> {
    let engine = Arc::new(TensorWasmEngine::new()?);
    let executor = TensorWasmExecutor::new(engine);
    let id = executor
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), wasm)
        .await?;
    let call = executor.call_export_with_args(id, "_start", &[]).await;
    let _ = executor.terminate(id).await;
    call?;
    Ok(())
}
```

Signature mapped from
[`crates/tensor-wasm-exec/src/executor.rs`](../crates/tensor-wasm-exec/src/executor.rs)
(`pub async fn spawn_instance(&self, cfg: SpawnConfig, wasm: &[u8]) -> Result<InstanceId, ExecError>`).

### 6.3 Or — wrap your module as a TensorWasm function via HTTP

The most typical TensorWasm shape is deploy via HTTP, invoke
remotely:

```bash
cargo run --release --bin tensor-wasm -- serve --addr 0.0.0.0:8080 &
WASM_B64=$(base64 -w0 < my_module.wasm)
RESP=$(curl -sf -X POST http://localhost:8080/functions \
  -H 'content-type: application/json' -H 'x-tensor-wasm-tenant: 1' \
  -d "{\"name\":\"my-module\",\"wasm_b64\":\"${WASM_B64}\"}")
ID=$(echo "$RESP" | jq -r .id)
curl -sf -X POST "http://localhost:8080/functions/${ID}/invoke" \
  -H 'content-type: application/json' -H 'x-tensor-wasm-tenant: 1' \
  -d '{}'
# => {"result":"ok","function_id":"<ID>"}
```

Full surface in [`API.md`](../crates/tensor-wasm-api/API.md); for
production deployment with mTLS, Prometheus, Grafana, and audit
log see
[`tutorials/production-deployment.md`](tutorials/production-deployment.md).

### 6.4 Migration checklist (library)

Confirm you actually want to switch ([§1.1](#11-the-library-user)).
Then: replace `wasmtime::Engine` with `TensorWasmEngine`; replace
`Linker` + `Module::new` + `Store` + `instantiate` with
`TensorWasmExecutor::spawn_instance`; replace `get_typed_func` +
`func.call` with `executor.call_export_with_args(id, "<export>", &[])`
(the legacy `call_export(id, "<export>")` no-args shim is
`#[deprecated]` since 0.3.7, removal target v0.4 — see
[§ Typed exports](#typed-exports-v036--v037)); drop any
custom host imports beyond WASI P2 + `wasi:cuda` (import table is
fixed in v0.1.0); replace `Module::serialize`/`deserialize`
warm-cache with the snapshot subsystem
([`COLD-START.md`](COLD-START.md)).

## 7. CLI-user migration (`wasmtime run` / `wasmer run`)

### 7.1 Side-by-side (pure-CPU, no args)

```bash
wasmtime run path/to/foo.wasm                       # Wasmtime
wasmer run path/to/foo.wasm                         # Wasmer (Cranelift, default)
tensor-wasm run path/to/foo.wasm --export _start    # Craton TensorWasm
```

`tensor-wasm run` defaults the export to `main`; pass `--export
_start` for the WASI convention. Behavior on success is identical
(exit 0, guest stdout reaches your terminal).

### 7.2 The argument caveat (v0.1.0)

```bash
wasmtime run foo.wasm --invoke 'add(1, 2)'     # values reach the guest
wasmer run foo.wasm --invoke add -- 1 2        # values reach the guest
# Craton TensorWasm: JSON is PARSED AND VALIDATED, but values do
# NOT reach the guest. Executor only invokes `() -> ()` in v0.1.0.
tensor-wasm run foo.wasm --export add --args '[1.0, 2.0]'
```

Verified in
[`crates/tensor-wasm-cli/src/cmd/run.rs`](../crates/tensor-wasm-cli/src/cmd/run.rs);
argument lowering tracked for v0.2
([`PATH-TO-V1.md`](PATH-TO-V1.md#v020--real-cuda)). If your CLI
workload requires arg passthrough, defer migration.

### 7.3 AOT compile

| Runtime | Compile | Load |
|---|---|---|
| Wasmtime CLI | `wasmtime compile foo.wasm -o foo.cwasm` | `wasmtime run --allow-precompiled foo.cwasm` |
| Wasmer CLI | `wasmer compile foo.wasm -o foo.wasmu` | `wasmer run foo.wasmu` |
| TensorWasm CLI | `tensor-wasm snapshot save ...` | `tensor-wasm snapshot restore ...` |

The TensorWasm snapshot is **not** `.wasmu`- or `.cwasm`-compatible
— it is a TensorWasm-internal streaming zstd + bincode payload
([`COLD-START.md`](COLD-START.md)). See [§12](#12-faqs) on artifact
compatibility.

### 7.4 Migration checklist (CLI)

Replace `wasmtime run X.wasm` / `wasmer run X.wasm` with
`tensor-wasm run X.wasm --export <name>`. If your workload uses
non-empty args, stop and re-evaluate at v0.2. If you used AOT
artifacts (`.cwasm`, `.wasmu`), switch to `tensor-wasm snapshot
save` for warm-cache; the formats are not interchangeable. Wire
`TENSOR_WASM_LOG` to your preferred filter (default `warn`).

## 8. Server-user migration (Wasmer Edge / Spin / custom FaaS)

The migration path the rest of the docs are written to support.

### 8.1 Spin component → TensorWasm function

Spin's deployment unit is a `.wasm` component plus a TOML
manifest; TensorWasm's is a `.wasm` module uploaded via
`POST /functions` and invoked via `POST /functions/{id}/invoke`.
Wasm artifact is portable; manifest is replaced by the deploy
call. Side by side:

```bash
# Spin
spin new -t http-rust my-component && cd my-component
spin build && spin up                       # → http://localhost:3000/

# TensorWasm
cargo build --release --target wasm32-wasip1 -p my-component
WASM=target/wasm32-wasip1/release/my_component.wasm
WASM_B64=$(base64 -w0 < "${WASM}")
RESP=$(curl -sf -X POST http://localhost:8080/functions \
  -H 'authorization: Bearer dev-token' -H 'content-type: application/json' \
  -H 'x-tensor-wasm-tenant: 1' \
  -d "{\"name\":\"my-component\",\"wasm_b64\":\"${WASM_B64}\"}")
ID=$(echo "$RESP" | jq -r .id)
curl -sf -X POST "http://localhost:8080/functions/${ID}/invoke" \
  -H 'authorization: Bearer dev-token' -H 'content-type: application/json' \
  -H 'x-tensor-wasm-tenant: 1' -d '{}'
```

Differences: TensorWasm calls `_start` (or `main`) per request; it
does *not* implement Spin's `spin-http` trigger model where the
component handles raw HTTP, so you refactor that handler to a
top-level `_start` (same constraint Wasmer Edge users face moving
back to a plain runtime). TensorWasm's `X-TensorWasm-Tenant`
scoping has no Spin analogue; per-tenant routing comes for free.

### 8.2 Wasmer Edge → TensorWasm

Wasmer Edge wraps a `.wasm` in a deploy manifest and runs it behind
Wasmer's hosted gateway. TensorWasm is the self-hosted analogue —
same `.wasm`, you bring the host. The HTTP shape differs (Wasmer
Edge is WebSocket-first; TensorWasm is HTTP request/response), so
the *client* changes; the artifact does not.

### 8.3 Production deployment

Once you have a working dev-mode flow, the production migration is
not "configure TensorWasm" — it is "deploy the Helm chart following
[`tutorials/production-deployment.md`](tutorials/production-deployment.md),"
which covers GPU scheduling, mTLS at the ingress, Prometheus
scrape, burn-rate alert rules, audit-log durability, and backup
hygiene. Chart value reference at
[`deploy/helm/tensor-wasm/README.md`](../deploy/helm/tensor-wasm/README.md).

### 8.4 Migration checklist (server)

Confirm your Wasm artifact builds against `wasm32-wasip1` (WASI P2;
P3 is not supported); refactor any Spin `spin-http` handler to a
plain `_start` / `main` entry point; decide your token-scope
strategy ([`tutorials/production-deployment.md`](tutorials/production-deployment.md)
§4.1 covers three patterns); walk the production-deployment
tutorial end-to-end on a staging cluster before cutting traffic
over; import the reference Grafana dashboard
([`dashboards/`](dashboards/)) and apply the PrometheusRule
([`SLO.md`](SLO.md)); validate backups via
[`BACKUP-RESTORE.md`](BACKUP-RESTORE.md) §7 *before* calling the
deployment production.

## 9. GPU migration (no precedent — raw CUDA / Triton / cudarc / wgpu)

No other runtime ships "sandboxed CUDA from inside a Wasm guest,"
so there is no like-for-like migration. Closest shapes are raw
CUDA C++ over `cuLaunchKernel`, Triton's Python dispatcher,
`cudarc` from a native Rust binary, or `wgpu` over Vulkan / Metal
/ DX12.

### 9.1 The mental model

Every existing case you might be coming from is *trusted code
talking to a trusted driver* — no sandbox. Moving to TensorWasm
means accepting one: your code is now a Wasm guest, and every CUDA
call goes through `wasi:cuda/host@0.2.0`, which bounds-checks
pointers against the guest's linear memory *before* any driver call
([`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit)). The benefit is
isolation between mutually-distrusting workloads on one host; the
cost is the bounds-check overhead (few hundred ns + back-pressure
semaphore) and the v0.1.0 `KernelArgsUnsupported` limitation
([`PATH-TO-V1.md`](PATH-TO-V1.md#v020--real-cuda) for the v0.2 fix).

### 9.2 From raw CUDA C++ to wasi:cuda

You had:

```cpp
cuModuleLoadDataEx(&mod, ptx, 0, nullptr, nullptr);
cuModuleGetFunction(&k, mod, "my_kernel");
cuLaunchKernel(k, gx,gy,gz, bx,by,bz, smem, stream, args, nullptr);
cuStreamSynchronize(stream);
```

You move to (Rust guest compiled to `wasm32-wasip1` — bindings ship
with the SDK):

```rust
extern "C" {
    fn wasi_cuda_load_ptx(p: i32, plen: i32, e: i32, elen: i32) -> i64;
    fn wasi_cuda_launch(
        k: i64, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
        smem: u32, args: i32, alen: i32,
    ) -> i32;
    fn wasi_cuda_sync() -> i32;
}

pub fn run(ptx: &[u8], entry: &str) -> Result<(), i32> {
    let kid = unsafe { wasi_cuda_load_ptx(
        ptx.as_ptr() as i32, ptx.len() as i32,
        entry.as_ptr() as i32, entry.len() as i32) };
    if kid < 0 { return Err(kid as i32); }
    let rc = unsafe { wasi_cuda_launch(kid, 1,1,1, 32,1,1, 0, 0,0) };
    if rc != 0 { return Err(rc); }
    let rc = unsafe { wasi_cuda_sync() };
    if rc != 0 { return Err(rc); }
    Ok(())
}
```

Wire-level constants in
[`crates/tensor-wasm-wasi-gpu/src/abi.rs`](../crates/tensor-wasm-wasi-gpu/src/abi.rs);
Component-Model form in
[`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit). Negative return codes
are stable across TensorWasm versions. Host install: read
[`CUDA-SETUP.md`](CUDA-SETUP.md) — the "SM-level compatibility
matrix" catches most first-time errors (PTX/device-cc mismatch
surfaces as `MalformedPtx`, code `-4`).

### 9.3 From Triton / cudarc / wgpu

The migration shape is similar: lift your dispatch loop into a Wasm
guest, replace your runtime's launch call with `wasi_cuda_launch`,
and re-link against TensorWasm's import table. Key differences:
**Triton** does kernel autotuning at compile time; TensorWasm's
opt-in `auto-offload` ([`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md)) detects
a narrower pattern set. **cudarc** is the same crate family
TensorWasm uses on the host (a `cust` → `cudarc` migration is
tracked for v0.2,
[`PATH-TO-V1.md`](PATH-TO-V1.md#1-cust-successor-gates-v02)); the
kernel source is reusable, only the launch surface changes.
**wgpu** is portable across NVIDIA / AMD / Intel / Apple; TensorWasm
v1.0 is NVIDIA only — wait for v2 if cross-vendor portability is
load-bearing.

### 9.4 Migration checklist (GPU)

Read [`CUDA-SETUP.md`](CUDA-SETUP.md) end-to-end and run its §9
verification script on the target host; read
[`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md) to know which patterns the JIT
pipeline offloads for you vs which you must emit PTX for yourself;
determine whether your kernels need parameters, and if yes gate
migration on v0.2 ([`PATH-TO-V1.md`](PATH-TO-V1.md#v020--real-cuda));
stand up MPS if you need spatial sharing across tenants
([`MPS-SETUP.md`](MPS-SETUP.md)); measure against the
[`BENCHMARKING.md`](BENCHMARKING.md) dimension-3 recipe
(`dispatch/serial` vs raw `cuLaunchKernel`) so you have a baseline
before rollout.

## 10. Operational diff — what you give up, what you gain

**Give up:** the upstream release cadence (Wasmtime ships a minor
roughly every 30 days; TensorWasm bumps quarterly per
[`WASMTIME-UPGRADE.md`](WASMTIME-UPGRADE.md), with a 7-day CVE
shortcut — you will be 1-2 minors behind); the Bytecode Alliance /
Wasmer Inc community size and corresponding issue response times
([`MAINTAINERS.md`](../MAINTAINERS.md) is the current roster);
flexibility of the embedding surface (`tensor-wasm-exec` is
narrower than `wasmtime::Linker`); a managed-service escape hatch
(self-hosted only at v1.0).

**Gain:** the bundled HTTP gateway with auth, scoped tokens, rate
limiting, audit, metrics, tracing
([`API.md`](../crates/tensor-wasm-api/API.md)); a snapshot
subsystem with cold-disk numbers and a regression gate
([`COLD-START.md`](COLD-START.md),
[`PERFORMANCE.md`](PERFORMANCE.md)); a WASI-GPU surface that
bounds-checks before the CUDA driver
([`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit)); a reference
deployment story
([`tutorials/production-deployment.md`](tutorials/production-deployment.md));
labelled gaps (every "modeled" number in
[`PERFORMANCE.md`](PERFORMANCE.md), every `TODO (v0.5)` in
[`SLO.md`](SLO.md), every "v0.1.0 contract" in
[`RISKS.md`](RISKS.md)).

**Stays the same in the middle:** the `.wasm` artifact
([§11](#11-what-stays-the-same)); Cranelift codegen quality
unchanged (per-loop perf for pure-CPU compute is within ~5% of
upstream Wasmtime per
[`BENCHMARKING.md`](BENCHMARKING.md#vs-wasmtime-upstream)'s
dimension-1 expectation — **note:** this is a stated bug-bar in
[`BENCHMARKING.md`](BENCHMARKING.md), not yet a published measured
comparison; first external comparisons are scheduled for the v0.5
milestone per
[`PATH-TO-V1.md`](PATH-TO-V1.md#performance)).

## 11. What stays the same

The most important sentence in this guide: **your `.wasm` modules
are 100% compatible**. A module that loads in upstream Wasmtime
loads in TensorWasm. No re-compile, no toolchain change, no SDK
rewrite for the guest side. Same `.wasm` binary (TensorWasm wraps
`wasmtime::Module::new`; validation is `wasmparser::validate`);
same Cranelift codegen, JIT, SIMD lowering, trap semantics —
per-loop perf on pure CPU within ~5% of upstream Wasmtime per
[`BENCHMARKING.md`](BENCHMARKING.md#vs-wasmtime-upstream); same
WASI P2 import surface; same component-model support, inherited.

What does *not* carry over: Wasmer Universal artifacts (`.wasmu`)
are Wasmer's format — ship source `.wasm`. Wasmtime `.cwasm`
precompiled artifacts are engine-specific — use the TensorWasm
snapshot subsystem. Custom host imports beyond WASI P2 + `wasi:cuda`
need a code change; the import table is fixed in v0.1.0.

## 12. FAQs

- **Is TensorWasm a Wasmtime fork?** No. We depend on upstream
  `wasmtime` as a crate via `[workspace.dependencies]` in
  [`Cargo.toml`](../Cargo.toml). Rationale in
  [`WASMTIME-FORK.md`](WASMTIME-FORK.md): forking would create a
  permanent maintenance burden, diverge from upstream JIT
  improvements, and introduce subtle correctness risk at the
  CLIF-rewrite boundary. The simplified IR
  ([`tensor_wasm_jit::detector::BlockIR`](../crates/tensor-wasm-jit/src/detector.rs))
  that picks GPU-offload candidates is built on `wasmparser`.
- **Will TensorWasm track Wasmtime releases?** Yes, quarterly, per
  [`WASMTIME-UPGRADE.md`](WASMTIME-UPGRADE.md). Minor bumps batch
  into one PR per calendar quarter; patches roll forward
  opportunistically; CVEs override the cadence with a 7-day target.
  Major Wasmtime bumps go through an RFC and never land in the
  same TensorWasm release as a TensorWasm major.
- **Can I use Wasmer LLVM-compiled `.wasmu` files?** No. `.wasmu`
  is Wasmer's compiled artifact format, tied to Wasmer's engine.
  TensorWasm runs Wasmtime, which has its own serialization format
  (`Module::serialize`). Bring the source `.wasm`; for warm-cache
  use `tensor-wasm snapshot save / restore`
  ([`COLD-START.md`](COLD-START.md)).
- **Can I use Wasmtime's `.cwasm` precompiled artifacts?** No —
  engine-specific format. TensorWasm configures Wasmtime through
  its own `TensorWasmEngine` builder, so `.cwasm` files compiled
  against a vanilla `wasmtime::Engine` are not portable. Ship
  `.wasm` source and snapshot inside TensorWasm.
- **Does my Wasmtime CLI script work as-is?** Mostly. `tensor-wasm
  run X.wasm --export _start` is the closest analogue. The
  `--args` flag exists but does not pass values to the guest in
  v0.1.0; see [§7.2](#72-the-argument-caveat-v010). For AOT
  (`wasmtime compile`), use `tensor-wasm snapshot save` — same
  goal, different format.
- **What's the perf overhead vs raw Wasmtime?** Within ~5% on
  pure-CPU compute, expected by design. This is the
  [`BENCHMARKING.md`](BENCHMARKING.md#vs-wasmtime-upstream)
  bug-bar — larger gaps are bugs to file. First published external
  comparisons are scheduled for v0.5
  ([`PATH-TO-V1.md`](PATH-TO-V1.md#performance)); until then the
  number is design intent, not a measured publication.
- **WASI Preview 3?** Not yet, v2 roadmap. v1.0 of TensorWasm ships
  P2 only.
- **AMD / Intel / Apple GPUs?** v2 roadmap. v1.0 is NVIDIA CUDA
  only; the WIT leaves room for vendor abstraction.
- **Managed-service offering?** No, self-hosted only at v1.0.
- **How do I undo this?** Symmetric. The same `.wasm` runs on
  Wasmtime or Wasmer; pull the source artifact out, point your old
  service at it, and you are back. TensorWasm snapshots and audit
  logs do not lock you in.
- **Can I run TensorWasm without CUDA?** Yes. All CUDA paths are
  feature-gated; on a no-CUDA host the wasi:cuda host functions
  return `AbiError::NotAvailable` (code `-1`) and the rest of the
  runtime is unaffected. Default developer-laptop configuration;
  see the [5-minute quickstart](../README.md#5-minute-quickstart).

## 13.bis Inter-version migration

### Typed exports (v0.3.6 → v0.3.7)

v0.3.6 shipped `TensorWasmExecutor::call_export(id, export)` as the only
guest-invocation entry point: it ran the export as a `() -> ()` typed
call, discarded any return value, and surfaced success / error via
`Result<(), ExecError>`. The signature could not pass arguments and
could not surface multi-value results.

B6.2 (v0.3.7) lands `call_export_with_args` as the primary entry point:

```rust
pub async fn call_export_with_args(
    &self,
    id: InstanceId,
    export: &str,
    args: &[WasmArg],
) -> Result<serde_json::Value, ExecError>
```

`WasmArg` is a small `Copy` enum mirroring the four core wasm value
types (`i32`, `i64`, `f32`, `f64`). The return value is the export's
result list serialised as a JSON array — empty for the historical
`() -> ()` shape, populated for richer signatures. The deadline,
admission-control, drop-guard, and span instrumentation contracts are
unchanged.

`call_export` and its sibling `call_export_then_terminate` are now thin
back-compat wrappers — both **`#[deprecated]` since 0.3.7**, both
slated for removal in **v0.4**. Compiling against them today emits a
`deprecated` warning pointing back at this section.

#### Migration

For every external `call_export(id, "name")` call site:

```diff
- exec.call_export(id, "name").await?;
+ exec.call_export_with_args(id, "name", &[]).await?;
```

The empty-slice fast path inside `call_export_with_args` takes the
same `func.typed::<(), ()>()` branch the legacy method used, so the
runtime cost is identical (no extra `Val` allocations on the hot
path). The discarded return value is the only behavioural diff — the
wrapper above maps it to `()` for callers that don't need it.

For `call_export_then_terminate(id, "name")`:

```diff
- exec.call_export_then_terminate(id, "name").await?;
+ exec.call_export_with_args_then_terminate(id, "name", &[]).await?;
```

Same `AutoTerminateGuard` lifecycle (success, error, *and* Future-drop
all clean up the registry entry); the discarded `serde_json::Value`
return is again the only diff.

In-tree callers (CLI `run` / `bench`, HTTP `/invoke`, every
`tensor-wasm-exec/tests/*.rs` integration test) migrated in B6.2 +
B6.2-follow-up. External embedders should plan the same swap before
the v0.4 bump.

## 13. Where to go next

In order: the 5-minute quickstart
([`README.md`](../README.md#5-minute-quickstart)); conceptual
onboarding ([`GETTING-STARTED.md`](GETTING-STARTED.md)); HTTP API
reference ([`API.md`](../crates/tensor-wasm-api/API.md));
production deployment
([`tutorials/production-deployment.md`](tutorials/production-deployment.md));
benchmarking your migration ([`BENCHMARKING.md`](BENCHMARKING.md))
— write up a `comparison.json` per its schema before and after the
cutover; CUDA setup if migrating a GPU workload
([`CUDA-SETUP.md`](CUDA-SETUP.md)); roadmap to see what is still in
motion ([`PATH-TO-V1.md`](PATH-TO-V1.md)).

## Related

- [`PATH-TO-V1.md`](PATH-TO-V1.md) — v0.4 documentation workstream
  this guide satisfies; anti-goals; v2 deferrals.
- [`WASMTIME-FORK.md`](WASMTIME-FORK.md) — explicit "we are not a
  fork" position.
- [`WASMTIME-UPGRADE.md`](WASMTIME-UPGRADE.md) — quarterly cadence
  policy; CVE shortcut; what forces a major bump.
- [`BENCHMARKING.md`](BENCHMARKING.md) — competitor recipes;
  fair-fight constraints; anti-cheating checklist.
- [`PERFORMANCE.md`](PERFORMANCE.md) — committed baseline and
  regression-gate policy.
- [`RISKS.md`](RISKS.md) — v0.1.0 known limitations.
- [`crates/tensor-wasm-api/API.md`](../crates/tensor-wasm-api/API.md)
  — HTTP surface, token grammar, error envelope.
- [`crates/tensor-wasm-wasi-gpu/src/abi.rs`](../crates/tensor-wasm-wasi-gpu/src/abi.rs)
  and [`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit) — WASI-GPU
  surface (raw ABI + Component-Model).
- [`tutorials/production-deployment.md`](tutorials/production-deployment.md)
  — end-to-end Helm-chart deployment.
- [`CUDA-SETUP.md`](CUDA-SETUP.md), [`MPS-SETUP.md`](MPS-SETUP.md),
  [`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md) — GPU operational guides.
- [`COLD-START.md`](COLD-START.md) — snapshot subsystem;
  restore-vs-`Module::deserialize` distinction.
- [`MIGRATION-v0-to-v1.md`](MIGRATION-v0-to-v1.md) — the *other*
  migration doc (TensorWasm v0.x to v1.0).

_Status: v0.4 release. Examples use the workspace-pinned Wasmtime
`25.x` API; same shape works with `wasmtime 45.x` modulo minor
renames. Re-validate when the v0.5 external comparisons land (the
"~5% overhead vs upstream Wasmtime" claim in
[§10](#10-operational-diff--what-you-give-up-what-you-gain) and
[§12](#12-faqs) becomes measured) and when v0.2 removes the v0.1.0
kernel-args limitation in
[§9](#9-gpu-migration-no-precedent--raw-cuda--triton--cudarc--wgpu)._
