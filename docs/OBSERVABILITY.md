# Craton TensorWasm — Observability

Craton TensorWasm emits structured `tracing` spans and events across every crate in the workspace, with optional OTLP export gated behind the `otlp` feature on `tensor-wasm-core`. Traces stitch across HTTP boundaries via the W3C `traceparent` header so an external caller's trace context flows all the way down into executor and GPU spans. This document describes the stack, the span schema, and how to wire a local collector for development.

## Stack

- `tracing` for span and event emission across all TensorWasm crates.
- `tracing-subscriber` for filtering (`EnvFilter`) and human-readable fmt output.
- `tracing-opentelemetry` for forwarding spans into the OpenTelemetry pipeline.
- `opentelemetry-otlp` (with the `grpc-tonic` transport) for shipping spans to a collector.
- Jaeger, Honeycomb, or Grafana Tempo on the consumer side — anything that speaks OTLP works.

## Span schema

Every span listed below is part of TensorWasm's public observability contract. Renaming or removing one is a breaking change for downstream dashboards and alerting.

| Span | Target | Required attrs | Optional attrs |
|---|---|---|---|
| `http.request` | tower-http | method, uri, version, traceparent | request_id |
| `tensor_wasm_exec::executor::spawn_instance` | tensor-wasm-exec | tenant, instance_id | wasm_bytes |
| `tensor_wasm_exec::executor::call_export` | tensor-wasm-exec | instance, export | — |
| `tensor_wasm_exec::executor::terminate` | tensor-wasm-exec | instance | — |
| `wasi_cuda.load_ptx` | tensor-wasm-wasi-gpu | instance, ptx_bytes, entry_bytes | — |
| `wasi_cuda.launch` | tensor-wasm-wasi-gpu | instance, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem | — |
| `wasi_cuda.sync` | tensor-wasm-wasi-gpu | instance | — |

Required attributes must be present on every span instance; if a value is genuinely unavailable, prefer the sentinel `"unknown"` over silently dropping the field, so log-based queries don't miss rows.

## Parent-child relationships

Typical call tree for a single invocation through the API gateway:

```
http.request
└── tensor_wasm_exec::executor::spawn_instance
    ├── tensor_wasm_exec::executor::call_export
    │   ├── wasi_cuda.load_ptx
    │   ├── wasi_cuda.launch
    │   └── wasi_cuda.sync
    └── tensor_wasm_exec::executor::terminate
```

A guest that never touches the GPU produces the same shape minus the `wasi_cuda.*` children. A guest that calls an export multiple times produces one `call_export` span per call, each with its own GPU subtree.

## Local Jaeger setup

```sh
# Start Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Run TensorWasm with OTLP enabled
TENSOR_WASM_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run --bin tensor-wasm --features tensor-wasm-core/otlp -- run example.wasm
```

Then open <http://localhost:16686> to see traces. The service name defaults to `tensor-wasm`; override with `OTEL_SERVICE_NAME` if you run multiple TensorWasm instances against one collector.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `TENSOR_WASM_LOG` | `info` | tracing-subscriber filter directive |
| `TENSOR_WASM_OTLP_ENDPOINT` | (unset) | OTLP collector endpoint (preferred) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | Fallback OTLP endpoint |
| `OTEL_SERVICE_NAME` | `tensor-wasm` | Service name attribute |

`TENSOR_WASM_LOG` accepts the full `EnvFilter` directive syntax, so `TENSOR_WASM_LOG=tensor_wasm_exec=debug,wasmtime=warn,info` is valid. When both `TENSOR_WASM_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_ENDPOINT` are set, the TensorWasm-specific variable wins.

To enable OTLP export end-to-end, build the gateway with the `otlp`
feature on `tensor-wasm-core` and set at least
`TENSOR_WASM_OTLP_ENDPOINT`. Without the feature the gateway still
participates in W3C propagation — inbound `traceparent` headers are
parsed and the trace id surfaces on the `x-trace-id` response header —
but no spans are exported to a collector, so cross-process trace
visualisation in Jaeger / Tempo will be empty.

## Headers and W3C propagation

The API gateway extracts the incoming `traceparent` header and uses it as the parent span context for the request's `http.request` span. If the header is missing or malformed, a fresh root context is created. Outgoing requests from TensorWasm back to other services should propagate `traceparent` so the trace stays connected; the v0.1 client does not do this automatically — set the header manually for now. The `tracedebug` extractor on the API surface logs the resolved context at `debug` level, which is useful when a trace seems to be silently rooting itself.

Every response from the gateway carries an `x-trace-id` header whose
value is the 32-character lowercase hex representation of the trace
the request joined (either the inbound `traceparent` trace id, or the
fresh root assigned by the gateway when no header was supplied). The
header is omitted only when no `tracing_opentelemetry` subscriber is
installed in the process (e.g. unit tests that bypass `init` /
`init_with_otlp`); operators correlating a captured response with an
OTLP backend should always see a populated value in production.

### Propagation hop diagram

The trace id flows through every layer in the request-handling stack
and back out to the caller. Each arrow is a parent/child relationship
in the resulting span tree:

```
caller (sends `traceparent: 00-<trace_id>-<span_id>-01`)
   │
   ▼
[tower] trace_layer_with_propagation
   │   extracts parent context via TraceContextPropagator
   │   opens `http.request` span (parent: caller's span context)
   ▼
[tower] inject_trace_id_header
   │   reads Span::current().context() on the way back out and
   │   stamps `x-trace-id: <trace_id>` on the response
   ▼
[axum] http.invoke_function   (child of http.request)
   │   fields: function_id, tenant
   ▼
[axum] invoke.run            (child of http.invoke_function)
   │   fields: tenant, function_id, wasm_bytes_len
   ▼
[tensor-wasm-exec] tensor_wasm_exec::executor::spawn_instance
   │   fields: tenant, instance_id
   │
   ├── [tensor-wasm-snapshot] SnapshotReader::restore         (when warm-starting)
   │   fields: input_len
   │
   ├── [tensor-wasm-snapshot] restore_to_gpu                  (cuda only)
   │   fields: input_len, device_index
   │
   └── [tensor-wasm-exec] tensor_wasm_exec::executor::call_export
       │   fields: instance, export
       │
       ├── [tensor-wasm-wasi-gpu] wasi_cuda.load_ptx
       ├── [tensor-wasm-wasi-gpu] wasi_cuda.launch
       │   │
       │   └── [tensor-wasm-wasi-gpu] wasi_cuda.dispatch       (DispatchFuture::poll)
       │
       └── [tensor-wasm-wasi-gpu] wasi_cuda.sync
   ▼
[axum] response  →  caller (receives `x-trace-id: <trace_id>`)
```

The async invoke path (`POST /functions/{id}/invoke-async`) inserts
one additional hop, `async_invoke.job`, between the route handler and
`invoke.run`. The job span is opened via
`tracing::Instrument::instrument` on the `tokio::spawn` future so the
trace id carries across the spawn boundary; without that the executor
spans would orphan from the inbound HTTP request.

Verified under N=64 concurrent async-invoke requests on a 4-worker-thread
runtime (`crates/tensor-wasm-api/tests/trace_concurrent_load_test.rs`):
the four W4.1 named spans (`http.request`, `http.invoke_function_async`,
`async_invoke.job`, `invoke.run`) each fire exactly once per request
(total = 4 × N) with no orphan parent ids and no `x-trace-id` header
collisions across the response set. The test guards against two failure
modes the v0.3.2 audit (Problem #11) flagged: the `Instrument` wrap on
`tokio::spawn` double-instrumenting the spawned future (would surface as
count > 4 × N), and `#[instrument]` macros silently dropping spans under
contention (count < 4 × N).

### Cross-crate propagation example

A single invocation that reaches the GPU produces a connected trace across
three crates. With `TENSOR_WASM_OTLP_ENDPOINT=http://localhost:4317` set and a guest
that calls `wasi_cuda_load_ptx` + `wasi_cuda_launch`, you should see this in
Jaeger as one trace:

```
trace_id=abc...                         (from inbound `traceparent`)
└── http.request                        [tensor-wasm-api]            POST /functions/fn_x/invoke
    ├── tensor_wasm_exec::executor::spawn_instance  [tensor-wasm-exec]      tenant=t_42, instance_id=i_99
    └── tensor_wasm_exec::executor::call_export     [tensor-wasm-exec]      instance=i_99, export=run
        ├── wasi_cuda.load_ptx               [tensor-wasm-wasi-gpu]  instance=i_99, ptx_bytes=2048, entry_bytes=10
        ├── wasi_cuda.launch                 [tensor-wasm-wasi-gpu]  instance=i_99, kernel=k_3, grid_x=64, ...
        └── wasi_cuda.sync                   [tensor-wasm-wasi-gpu]  instance=i_99
```

The same `trace_id` flows through all three crates because each layer takes
the parent context from its caller: `tensor-wasm-api`'s tower middleware extracts
the W3C header into the request scope, `tensor-wasm-exec` opens its spans as
children of the active context inside the request task, and `tensor-wasm-wasi-gpu`'s
host functions open theirs as children of the active executor span. Removing
any of these layers from the chain (e.g. running `tensor-wasm run` from the CLI
without a parent context) collapses the trace to start at `spawn_instance`,
but the parent-child relationships among the inner spans are preserved.

## Metrics complement

This file documents traces only; for metrics see `tensor_wasm_core::metrics::TensorWasmMetrics` (Prometheus text exposition via `tensor-wasm-api`'s `GET /metrics`). Traces and metrics share label conventions where they overlap — a `tenant` attribute on a span and a `tenant` label on a counter mean the same thing and can be joined in tools like Grafana.

## Cross-references

- `tensor-wasm-core/src/telemetry.rs` — `init` and (gated) `init_with_otlp`
- `crates/tensor-wasm-api/src/middleware.rs` — `trace_layer_with_propagation`
- `SECURITY.md` for the threat model around trace data leakage
- `docs/PERFORMANCE.md` for performance impact of the OTLP exporter

_Status: S20 of the plan. Re-baseline span schema before v0.2 — span names are part of the public observability contract._

## Reference Grafana dashboard

The metrics half of this story is rendered by the reference Grafana dashboard committed at [`docs/dashboards/tensor-wasm-overview.json`](dashboards/tensor-wasm-overview.json), with import instructions and the full metric inventory in [`docs/dashboards/README.md`](dashboards/README.md). The dashboard renders the five SLIs defined in [`docs/SLO.md`](SLO.md) as a top-row stat strip, then drills into HTTP traffic, per-tenant capacity, snapshot capture/restore, JIT cache hit ratio, and back-pressure permit utilization. It targets a generic Prometheus datasource via the `${DS_PROMETHEUS}` variable so a single JSON file imports cleanly into any Grafana — no Mimir/Cortex-specific features. Panels whose backing metric is in the TODO column (per the dashboard README) render "No data" until W2.3 ships the HTTP request counter and duration histogram; tenant-row panels degrade to a single aggregate series until per-tenant labelling lands on the existing gauges.
