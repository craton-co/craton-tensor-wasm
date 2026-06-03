# tensor-wasm-core

Foundational crate for Craton TensorWasm providing shared primitives used across every other TensorWasm crate. Defines the project-wide `TensorWasmError` type, a Prometheus metrics registry (`TensorWasmMetrics`), tracing/telemetry initialisation helpers, and strongly-typed newtypes such as `TenantId`, `InstanceId`, and `KernelId`. Has no internal dependencies on other TensorWasm crates and sits at the bottom of the workspace dependency graph.

## Cargo features

| Feature | Default | Description |
| ------- | ------- | ----------- |
| `otlp`  | off     | Enables the `telemetry::init_with_otlp` entry point and pulls in `opentelemetry`, `opentelemetry-otlp`, `opentelemetry_sdk`, `tracing-opentelemetry`, and `tokio`. With this feature off the crate has no async runtime dependency and the OTLP exporter is not built. |

The default build is intentionally minimal: it links no async runtime and no gRPC client, so downstream binaries that don't need OTLP export stay small.

Enable from a workspace member with:

```toml
tensor-wasm-core = { workspace = true, features = ["otlp"] }
```

See [docs/BUILD.md](https://github.com/craton-co/craton-tensor-wasm/blob/main/docs/BUILD.md) for the project-wide flag taxonomy and how the workspace composes feature unification across crates.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `thiserror` — derive macro for the `TensorWasmError` enum.
- `anyhow` — flexible error type for fallible bootstrap paths.
- `prometheus-client` — backing store for the `TensorWasmMetrics` registry.
- `tracing` — structured event emission used throughout the workspace.
- `tracing-subscriber` — subscriber/formatter wiring for telemetry init helpers.
- `serde` — derive support for serialisable newtypes and config structs.

Optional dependencies (pulled in only by `--features otlp`):
- `opentelemetry`, `opentelemetry-otlp`, `opentelemetry_sdk` — OTLP exporter plumbing.
- `tracing-opentelemetry` — bridges `tracing` spans into the OTel SDK.
- `tokio` — async runtime required by the OTLP gRPC exporter.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
