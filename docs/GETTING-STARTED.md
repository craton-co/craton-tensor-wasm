# Getting Started with Craton TensorWasm

Welcome to **TensorWasm** — a runtime for deploying WebAssembly functions with first-class GPU acceleration. This guide takes you from a clean checkout to invoking your first deployed function in about fifteen minutes.

## Audience

This guide is written for **Rust developers** who want to deploy Wasm functions that can offload work to a GPU. You should be comfortable with `cargo`, basic shell usage, and have a passing familiarity with HTTP APIs. Prior CUDA experience is helpful but not required — TensorWasm can auto-offload many SIMD-shaped Rust loops without you ever writing a kernel by hand.

If you're operating TensorWasm in production rather than developing functions for it, jump straight to [DEPLOYMENT.md](./DEPLOYMENT.md).

## Prerequisites

| Requirement | Why | Notes |
|---|---|---|
| **Rust toolchain** | Building TensorWasm and your Wasm guests | `rustup` will pick up the nightly channel automatically from `rust-toolchain.toml`. |
| **CUDA 12.0+** | GPU offload (optional) | Only required if you want GPU acceleration. See [CUDA-SETUP.md](./CUDA-SETUP.md). |
| **Docker** | Observability stack | Used by the bundled `docker-compose.yml` to spin up Jaeger, Prometheus, and Grafana. |

Verify your toolchain:

```sh
rustup --version
cargo --version
nvcc --version   # only if you want GPU offload
docker --version
```

If `nvcc` is missing, that's fine — TensorWasm will fall back to CPU execution for kernels that haven't been explicitly compiled to PTX.

## Hello, TensorWasm

The five steps below walk you from `git clone` to a deployed function answering HTTP requests.

### 1. Clone and build the workspace

```sh
git clone https://github.com/craton-co/craton-tensor-wasm.git
cd craton-tensor-wasm
cargo build --workspace
```

The first build pulls a handful of large dependencies (`wasmtime`, the CUDA bindings, the OpenTelemetry stack) and takes 5–10 minutes on a modern laptop. Subsequent builds are incremental.

If you hit build errors related to CUDA, see [BUILD.md](./BUILD.md) for the `CUDA_ROOT` and `CUDA_ARCH` overrides.

### 2. Write a hello-world Wasm module

In a sibling directory to your `tensor-wasm` checkout:

```sh
cargo new --lib hello
cd hello
rustup target add wasm32-wasip1
```

Replace `src/lib.rs` with:

```rust
#[no_mangle]
pub extern "C" fn add(a: f32, b: f32) -> f32 {
    a + b
}
```

Then add to `Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib"]
```

Build it:

```sh
cargo build --target wasm32-wasip1
```

You now have `target/wasm32-wasip1/debug/hello.wasm`. For richer examples — including GPU kernels — read the [Wasm Developer Guide](./WASM-DEVELOPER-GUIDE.md).

### 3. Run it locally with the `tensor-wasm` CLI

From your `tensor-wasm` checkout:

```sh
cargo run --bin tensor-wasm -- run ../hello/target/wasm32-wasip1/debug/hello.wasm
```

This loads the module into an in-process runtime and runs it once. Use this loop for fast iteration during development — there's no HTTP layer, no scheduler, just the engine.

For the full CLI surface, see [CLI.md](./CLI.md).

### 4. Start the HTTP server

To expose your function over the network, start the API gateway:

```sh
cargo run --bin tensor-wasm -- serve --addr 0.0.0.0:8080
```

You should see startup logs ending in something like:

```
tensor-wasm-api listening on 0.0.0.0:8080
```

The server is **stateless** at the gateway tier — durability lives in snapshots, which we'll cover in [DEPLOYMENT.md](./DEPLOYMENT.md).

### 5. Deploy and invoke

In a second shell:

```sh
# Deploy
curl -X POST http://localhost:8080/functions \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @../hello/target/wasm32-wasip1/debug/hello.wasm

# Response includes a function id, e.g. {"id":"fn_abc123"}

# Invoke
curl -X POST http://localhost:8080/functions/fn_abc123/invoke \
  -H 'Content-Type: application/json' \
  -d '{"export":"add","args":[2.0,3.0]}'

# => {"result": 5.0}
```

You've just deployed and invoked your first TensorWasm function. For the full HTTP surface — async invocation, streaming, batching — see [API.md](../crates/tensor-wasm-api/API.md).

## Observability

TensorWasm emits OpenTelemetry traces and Prometheus metrics out of the box. The fastest way to see them is the bundled compose stack:

```sh
docker compose up -d jaeger prometheus grafana
TENSOR_WASM_OTLP_ENDPOINT=http://localhost:4317 \
TENSOR_WASM_LOG=debug \
  cargo run --bin tensor-wasm -- serve --addr 0.0.0.0:8080
```

Then open `http://localhost:16686` for Jaeger traces of every invocation, including per-host-call spans for kernel launches.

The `TENSOR_WASM_LOG` env var follows the `tracing-subscriber` directive format: `TENSOR_WASM_LOG=info,tensor_wasm_wasm=debug` is a useful default during development.

For the full metrics catalog and dashboard layout, see [OBSERVABILITY.md](./OBSERVABILITY.md).

## Next steps

Now that you have an end-to-end loop working, here are the docs to read next:

- **[WASM-DEVELOPER-GUIDE.md](./WASM-DEVELOPER-GUIDE.md)** — write real compute functions, including hand-tuned GPU kernels via `wasi:cuda/host@0.2.0`.
- **[AUTO-OFFLOAD.md](./AUTO-OFFLOAD.md)** — learn when and how TensorWasm automatically promotes hot SIMD loops to GPU kernels, no PTX required.
- **[API.md](../crates/tensor-wasm-api/API.md)** — the full HTTP API: deploy, invoke, list, delete, snapshots.
- **[CUDA-SETUP.md](./CUDA-SETUP.md)** — installing CUDA, picking an `sm_*` arch, and validating your GPU.
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** — running TensorWasm in production, capacity planning, and disaster recovery.

Welcome aboard.
