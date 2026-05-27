# tensor-wasm-api

HTTP serverless API gateway for Craton TensorWasm, built on axum. Exposes REST endpoints for deploying modules, invoking instances synchronously and asynchronously, polling job status, scraping metrics, and health-checking the node. Wraps the routes in a Tower middleware stack covering request tracing, per-request timeouts, a process-wide concurrency cap, a 64 MiB body limit, bearer-token authentication, and tenant-header scoping. Ships the axum app builder and listener wiring that production deployments hook into.

## Security surface

- **Body limit (64 MiB).** Every inbound request is capped via `axum::extract::DefaultBodyLimit::max`. Larger bodies are rejected with `413 Payload Too Large` before any handler runs.
- **CORS allowlist.** Cross-origin browser callers are rejected by default; widen via `TENSOR_WASM_API_CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com`. The layer admits `GET`/`POST`/`DELETE` and the standard `Authorization`, `Content-Type`, `X-TensorWasm-Tenant`, and `Traceparent` headers.
- **Bearer-token auth via `TENSOR_WASM_API_TOKENS`.** A comma-separated allowlist of accepted tokens. Empty/unset puts the gateway in dev mode (warn-once on startup, requests pass through). When set, callers must send `Authorization: Bearer <token>`.
- **Tenant scoping via `X-TensorWasm-Tenant` header.** The header is parsed as a `u64` and threaded through to the executor. Absent header defaults to tenant `0`; set `TENSOR_WASM_API_REQUIRE_TENANT=1` to make the header mandatory.
- **Snapshot HMAC key via `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` (hex, 64 chars).** Optional. When set, the upcoming `/snapshot/save` and `/snapshot/restore` routes will HMAC-SHA256 sign on save and verify on restore. Malformed-but-set is a hard startup error so a typo never silently degrades to the no-signing path. The routes themselves are not yet wired (v0.4); the env knob is parsed today so manifests can be staged ahead of time.
- **Snapshot strict-verify via `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE` (`true`/`false`, default `false`).** Optional. When `true`, snapshot restore refuses v2 (unsigned) blobs even when a signing key is configured.

See [`API.md`](API.md) for the full wire-format reference.

## What ships in 0.3.x

The W2.x hardening wave layers the following on top of the core deploy/invoke surface:

- **W1.4 Rate limiting.** Per-token leaky bucket; over-limit callers receive `429 Too Many Requests` with `Retry-After`.
- **W2.1 Scoped tokens.** Per-tenant authorization; cross-tenant access surfaces as a `403` envelope.
- **W2.2 Audit log.** Structured JSON records every authenticated request through a sink-pluggable writer.
- **W2.3 HTTP metrics.** Prometheus exposition at `/metrics` covering request counts, duration histograms, and per-route labels.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime hosting the HTTP server.
- `axum` — web framework providing the router and extractors.
- `tower` — middleware abstractions stacked onto the router.
- `tower-http` — ready-made middleware (timeout, trace, CORS). Body-limit uses axum's native `DefaultBodyLimit::max` rather than the tower-http layer.
- `subtle` — constant-time byte comparison used by bearer-token allowlist lookup.
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
