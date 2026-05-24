# tensor-wasm-api

HTTP serverless API gateway for Craton TensorWasm, built on axum. Exposes REST endpoints for deploying modules, invoking instances synchronously and asynchronously, polling job status, scraping metrics, and health-checking the node. Wraps the routes in a Tower middleware stack covering request tracing, per-request timeouts, a process-wide concurrency cap, a 64 MiB body limit, bearer-token authentication, and tenant-header scoping. Ships the axum app builder and listener wiring that production deployments hook into.

## Security surface

- **Body limit (64 MiB).** Every inbound request is capped via `tower_http::limit::RequestBodyLimitLayer`. Larger bodies are rejected with `413 Payload Too Large` before any handler runs.
- **Bearer-token auth via `TENSOR_WASM_API_TOKENS`.** A comma-separated allowlist of accepted tokens. Empty/unset puts the gateway in dev mode (warn-once on startup, requests pass through). When set, callers must send `Authorization: Bearer <token>`.
- **Tenant scoping via `X-TensorWasm-Tenant` header.** The header is parsed as a `u64` and threaded through to the executor. Absent header defaults to tenant `0`; set `TENSOR_WASM_API_REQUIRE_TENANT=1` to make the header mandatory.

See [`API.md`](API.md) for the full wire-format reference.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime hosting the HTTP server.
- `axum` — web framework providing the router and extractors.
- `tower` — middleware abstractions stacked onto the router.
- `tower-http` — ready-made middleware (timeout, trace, body-limit).
- `hyper` — underlying HTTP/1 and HTTP/2 transport.
- `thiserror` — derive macro for API error variants.
- `tracing` — structured spans/events for request handling.
- `tracing-opentelemetry` — bridges tracing spans into OpenTelemetry contexts so the W3C `traceparent` header stitches traces across services.
- `opentelemetry` — propagator API used by the trace layer to extract incoming `traceparent` headers.
- `serde` — derive support for request/response DTOs.
- `serde_json` — JSON encoding of API payloads.
- `wasmparser` — full structural validation of inbound Wasm modules at deploy time.
- `base64` — decoding `wasm_b64` deploy payloads.
- `dashmap` — concurrent in-memory function and job registries.
- `uuid` — server-assigned function and job identifiers.

Internal crate dependencies:
- `tensor-wasm-core` — error envelope mapping, shared metrics, type primitives.
- `tensor-wasm-exec` — drives `TensorWasmExecutor` for the synchronous and async invoke paths.
