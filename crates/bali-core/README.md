# bali-core

Foundational crate for Project Bali providing shared primitives used across every other Bali crate. Defines the project-wide `BaliError` type, a Prometheus metrics registry (`BaliMetrics`), tracing/telemetry initialisation helpers, and strongly-typed newtypes such as `TenantId`, `InstanceId`, and `KernelId`. Has no internal dependencies on other Bali crates and sits at the bottom of the workspace dependency graph.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `thiserror` — derive macro for the `BaliError` enum.
- `anyhow` — flexible error type for fallible bootstrap paths.
- `prometheus-client` — backing store for the `BaliMetrics` registry.
- `tracing` — structured event emission used throughout the workspace.
- `tracing-subscriber` — subscriber/formatter wiring for telemetry init helpers.
- `serde` — derive support for serialisable newtypes and config structs.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
