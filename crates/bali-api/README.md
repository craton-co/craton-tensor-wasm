# bali-api

HTTP serverless API gateway for Project Bali, built on axum. Exposes REST endpoints for deploying modules, invoking instances, scraping metrics, and health-checking the node, wraps them in Tower middleware for timeouts, per-tenant rate limiting, and request tracing, and provides the axum app builder and listener wiring that production deployments hook into.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime hosting the HTTP server.
- `axum` — web framework providing the router and extractors.
- `tower` — middleware abstractions stacked onto the router.
- `tower-http` — ready-made middleware (timeout, trace, compression).
- `hyper` — underlying HTTP/1 and HTTP/2 transport.
- `thiserror` — derive macro for API error variants.
- `tracing` — structured spans/events for request handling.
- `serde` — derive support for request/response DTOs.
- `serde_json` — JSON encoding of API payloads.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
