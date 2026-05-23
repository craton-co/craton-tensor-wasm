# Project Bali — Observability

Project Bali emits structured `tracing` spans and events across every crate in the workspace, with optional OTLP export gated behind the `otlp` feature on `bali-core`. Traces stitch across HTTP boundaries via the W3C `traceparent` header so an external caller's trace context flows all the way down into executor and GPU spans. This document describes the stack, the span schema, and how to wire a local collector for development.

## Stack

- `tracing` for span and event emission across all Bali crates.
- `tracing-subscriber` for filtering (`EnvFilter`) and human-readable fmt output.
- `tracing-opentelemetry` for forwarding spans into the OpenTelemetry pipeline.
- `opentelemetry-otlp` (with the `grpc-tonic` transport) for shipping spans to a collector.
- Jaeger, Honeycomb, or Grafana Tempo on the consumer side — anything that speaks OTLP works.

## Span schema

Every span listed below is part of Bali's public observability contract. Renaming or removing one is a breaking change for downstream dashboards and alerting.

| Span | Target | Required attrs | Optional attrs |
|---|---|---|---|
| `http.request` | tower-http | method, uri, version, traceparent | request_id |
| `bali_exec::executor::spawn_instance` | bali-exec | tenant, instance_id | wasm_bytes |
| `bali_exec::executor::call_export` | bali-exec | instance, export | — |
| `bali_exec::executor::terminate` | bali-exec | instance | — |
| `wasi_cuda.load_ptx` | bali-wasi-gpu | instance, ptx_bytes, entry_bytes | — |
| `wasi_cuda.launch` | bali-wasi-gpu | instance, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem | — |
| `wasi_cuda.sync` | bali-wasi-gpu | instance | — |

Required attributes must be present on every span instance; if a value is genuinely unavailable, prefer the sentinel `"unknown"` over silently dropping the field, so log-based queries don't miss rows.

## Parent-child relationships

Typical call tree for a single invocation through the API gateway:

```
http.request
└── bali_exec::executor::spawn_instance
    ├── bali_exec::executor::call_export
    │   ├── wasi_cuda.load_ptx
    │   ├── wasi_cuda.launch
    │   └── wasi_cuda.sync
    └── bali_exec::executor::terminate
```

A guest that never touches the GPU produces the same shape minus the `wasi_cuda.*` children. A guest that calls an export multiple times produces one `call_export` span per call, each with its own GPU subtree.

## Local Jaeger setup

```sh
# Start Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Run Bali with OTLP enabled
BALI_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run --bin bali --features bali-core/otlp -- run example.wasm
```

Then open <http://localhost:16686> to see traces. The service name defaults to `bali`; override with `OTEL_SERVICE_NAME` if you run multiple Bali instances against one collector.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `BALI_LOG` | `info` | tracing-subscriber filter directive |
| `BALI_OTLP_ENDPOINT` | (unset) | OTLP collector endpoint (preferred) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | Fallback OTLP endpoint |
| `OTEL_SERVICE_NAME` | `bali` | Service name attribute |

`BALI_LOG` accepts the full `EnvFilter` directive syntax, so `BALI_LOG=bali_exec=debug,wasmtime=warn,info` is valid. When both `BALI_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_ENDPOINT` are set, the Bali-specific variable wins.

## Headers and W3C propagation

The API gateway extracts the incoming `traceparent` header and uses it as the parent span context for the request's `http.request` span. If the header is missing or malformed, a fresh root context is created. Outgoing requests from Bali back to other services should propagate `traceparent` so the trace stays connected; the v0.1 client does not do this automatically — set the header manually for now. The `tracedebug` extractor on the API surface logs the resolved context at `debug` level, which is useful when a trace seems to be silently rooting itself.

## Metrics complement

This file documents traces only; for metrics see `bali_core::metrics::BaliMetrics` (Prometheus text exposition via `bali-api`'s `GET /metrics`). Traces and metrics share label conventions where they overlap — a `tenant` attribute on a span and a `tenant` label on a counter mean the same thing and can be joined in tools like Grafana.

## Cross-references

- `bali-core/src/telemetry.rs` — `init` and (gated) `init_with_otlp`
- `crates/bali-api/src/middleware.rs` — `trace_layer_with_propagation`
- `SECURITY.md` for the threat model around trace data leakage
- `docs/PERFORMANCE.md` for performance impact of the OTLP exporter

_Status: S20 of the plan. Re-baseline span schema before v0.2 — span names are part of the public observability contract._
