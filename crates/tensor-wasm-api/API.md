# Craton TensorWasm HTTP API Reference

The `tensor-wasm-api` crate exposes the public REST surface of a [TensorWasm](../../ARCHITECTURE.md)
node. This document covers every endpoint the gateway serves. For build
instructions see [`docs/BUILD.md`](../../docs/BUILD.md); for the system-wide
architecture see [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

All requests and responses use JSON unless otherwise noted. Identifiers are
[RFC 4122] UUIDv4 strings.

[RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122

A machine-readable OpenAPI 3.0 description ships alongside this document at
[`openapi.json`](openapi.json).

## Conventions

### Base URL

A TensorWasm node listens on a single `host:port`. There is no API versioning prefix
yet — once the surface stabilises (S25) the prefix `/v1/` will be introduced
and the unprefixed routes deprecated.

### Authentication

The gateway accepts bearer-token authentication. At startup it reads a
comma-separated allowlist of accepted tokens from the environment variable
`TENSOR_WASM_API_TOKENS`. Every subsequent request must carry an
`Authorization: Bearer <token>` header whose token is in the allowlist;
mismatched or missing headers produce `401 Unauthorized` with `kind: "unauthorized"`.

If `TENSOR_WASM_API_TOKENS` is unset or empty, the gateway runs in **dev mode**:
authentication is disabled, a single `tracing::warn!` event is emitted at
startup (`TENSOR_WASM_API_TOKENS empty; API accepts all requests (dev mode)`), and
every request is allowed through. Dev mode is intended for local development
and integration tests; production deployments must always set the allowlist.

Example:

```bash
export TENSOR_WASM_API_TOKENS=secret-prod-token,canary-token
curl -H 'Authorization: Bearer secret-prod-token' http://localhost:8080/healthz
```

### Tenant scoping

Every request may carry an `X-TensorWasm-Tenant: <u64>` header. The value is parsed
as an unsigned 64-bit integer and forwarded to the executor as the owning
`TenantId` for any instance the request spawns.

* Absent header: defaults to tenant `0`.
* Header present but not a valid `u64`: `400 Bad Request` with
  `kind: "missing_tenant"`.
* If `TENSOR_WASM_API_REQUIRE_TENANT=1` was set at startup, the header is mandatory
  — absent requests are rejected with the same `kind`.

### Request limits

Every inbound request body is capped at **64 MiB** by
`tower_http::limit::RequestBodyLimitLayer`. Larger bodies are rejected with
`413 Payload Too Large` before any handler runs. The cap is global; it is
not user-tunable in this release.

### Error envelope

Every non-2xx response carries the same JSON envelope:

```json
{
  "error": {
    "kind": "<stable machine-readable identifier>",
    "message": "<human-readable description>"
  }
}
```

`kind` strings are part of the public contract; `message` strings are not and
may change between patch releases. Known `kind` values:

| `kind`             | HTTP | Meaning                                                                  |
|--------------------|------|--------------------------------------------------------------------------|
| `invalid_json`     | 400  | Request body could not be parsed as JSON, or shape wrong.                |
| `invalid_name`     | 400  | `name` field empty or whitespace-only.                                   |
| `invalid_base64`   | 400  | `wasm_b64` field is not valid standard base64.                           |
| `invalid_wasm`     | 400  | Decoded Wasm bytes fail `wasmparser::validate` (short, bad magic, etc.). |
| `missing_export`   | 400  | Module is missing both `_start` and `main`.                              |
| `missing_tenant`   | 400  | `X-TensorWasm-Tenant` header missing/garbled when required.                    |
| `unauthorized`     | 401  | Missing or unrecognised bearer token.                                    |
| `not_found`        | 404  | Requested function or job id does not exist.                             |
| `body_too_large`   | 413  | Inbound body exceeds the 64 MiB cap (often rendered as bare 413).        |
| `invoke_timeout`   | 504  | Invocation exceeded its per-call deadline.                               |
| `instance_not_found` | 404 | Executor lost track of an instance mid-call (rare).                     |
| `wasmtime`         | 500  | Underlying wasmtime call failed (trap, host error, etc.).                |
| `internal`         | 500  | Unexpected server-side failure.                                          |

---

## `POST /functions`

Deploy a new Wasm module. The module is decoded, validated with
`wasmparser::validate`, and stored in the in-memory registry as an
`Arc<[u8]>` so concurrent invocations share a single allocation.

**Request body**

| Field      | Type   | Required | Description                                                                                                             |
|------------|--------|----------|-------------------------------------------------------------------------------------------------------------------------|
| `name`     | string | yes      | Non-empty display name (free-form).                                                                                     |
| `wasm_b64` | string | yes      | Base64-encoded Wasm module bytes (standard alphabet, padded). Decoded value must validate as a complete Wasm module.    |

**Success — `200 OK`**

```json
{ "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479" }
```

**Errors** — `400 Bad Request` with `kind` of `invalid_json`, `invalid_name`,
`invalid_base64`, or `invalid_wasm`.

**Example**

```bash
WASM_B64=$(wat2wasm -o - module.wat | base64 -w0)
curl -s -X POST http://localhost:8080/functions \
  -H 'authorization: Bearer $TOKEN' \
  -H 'content-type: application/json' \
  -H 'x-tensor-wasm-tenant: 1' \
  -d "{\"name\":\"hello\",\"wasm_b64\":\"$WASM_B64\"}"
```

---

## `DELETE /functions/{id}`

Remove a deployed function. Idempotent on success: a second delete returns
`404`.

**Success — `204 No Content`** (empty body)

**Errors** — `404 Not Found` with `kind` of `not_found`.

**Example**

```bash
curl -X DELETE http://localhost:8080/functions/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

---

## `POST /functions/{id}/invoke`

Invoke a deployed function synchronously. Spawns a fresh executor instance
with a 30-second deadline, calls `_start` (falling back to `main`), and
terminates the instance before returning. The owning tenant is taken from
the `X-TensorWasm-Tenant` header (defaulting to `0`).

**Request body** — any JSON value, threaded through as the invocation
argument payload. An empty object `{}` is the conventional choice.

**Success — `200 OK`**

```json
{ "result": "ok", "function_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479" }
```

**Errors** — `404 Not Found` if the function id is unknown;
`400 Bad Request` (`missing_export`) if neither `_start` nor `main` is
exported; `504 Gateway Timeout` (`invoke_timeout`) if the call exceeds the
30-second deadline; `500 Internal Server Error` (`wasmtime`) for other
runtime failures.

**Example**

```bash
curl -X POST http://localhost:8080/functions/$ID/invoke \
  -H 'authorization: Bearer $TOKEN' \
  -H 'content-type: application/json' \
  -d '{}'
```

---

## `POST /functions/{id}/invoke-async`

Fire-and-forget invocation. Records a `Pending` job, spawns the
spawn/call/terminate flow onto a Tokio task, and returns immediately. The
caller polls `GET /jobs/{id}` to learn when the invocation finishes.

**Request body** — any JSON value (see `/invoke` above).

**Success — `202 Accepted`**

```json
{ "job_id": "5b3aa6c8-1f4f-4d23-bf01-2b1e10e7a4c9" }
```

**Errors** — `404 Not Found` if the function id is unknown.

### Async lifecycle

After receiving the 202 the caller polls `GET /jobs/{id}` until `status`
transitions out of `pending`:

* `status: "pending"` — job is queued or in flight; poll again.
* `status: "completed"` — `result` carries the same `{ "function_id", "result" }`
  shape the synchronous `/invoke` returns.
* `status: "failed"` — `result` carries `{ "kind": "...", "message": "..." }`
  mirroring the synchronous error envelope.

**Example**

```bash
curl -X POST http://localhost:8080/functions/$ID/invoke-async \
  -H 'authorization: Bearer $TOKEN' \
  -H 'content-type: application/json' \
  -d '{}'
```

---

## `GET /jobs/{id}`

Poll an async-invocation job.

**Success — `200 OK`**

```json
{
  "id": "5b3aa6c8-1f4f-4d23-bf01-2b1e10e7a4c9",
  "function_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "pending",
  "created_unix_ms": 1716491220123
}
```

`status` is one of `"pending"`, `"completed"`, or `"failed"`. Completed jobs
carry a `result` field with the invocation payload; failed jobs carry a
`result` field with the `{kind, message}` error envelope.

**Errors** — `404 Not Found` if the job id is unknown.

**Example**

```bash
curl http://localhost:8080/jobs/$JOB_ID
```

---

## `GET /metrics`

Prometheus text-format exposition of the shared `TensorWasmMetrics` registry from
`tensor-wasm-core`. Every counter the executor and the kernel layer publish is
exposed.

**Success — `200 OK`**, `Content-Type: text/plain; version=0.0.4`

```
# HELP tensor_wasm_active_instances Currently live Wasm instances.
# TYPE tensor_wasm_active_instances gauge
tensor_wasm_active_instances 0
# HELP tensor_wasm_kernel_dispatches_total ...
# TYPE tensor_wasm_kernel_dispatches_total counter
tensor_wasm_kernel_dispatches_total 0
...
```

**Example**

```bash
curl http://localhost:8080/metrics
```

---

## `GET /healthz`

Liveness probe — returns `200` as long as the process is serving.

**Success — `200 OK`**

```json
{ "status": "ok" }
```

**Example**

```bash
curl http://localhost:8080/healthz
```

---

## Middleware

Every route is wrapped in the standard tower stack assembled by
`tensor_wasm_api::server::build_router`:

* [`trace_layer_with_propagation`](src/middleware.rs) — emits a `tracing`
  span per request with method, URI, response status, and a `traceparent`
  field. Stitches incoming W3C `traceparent` headers into the parent
  OpenTelemetry context so traces correlate across services.
* [`body_limit_layer(64 MiB)`](src/middleware.rs) — rejects oversized
  bodies with `413` before any handler runs.
* [`timeout_layer(30s)`](src/middleware.rs) — fails slow requests with
  `408 Request Timeout`.
* [`concurrency_limit_layer(64)`](src/middleware.rs) — caps in-flight
  requests process-wide. Per-tenant buckets land in a follow-up release.
* [`bearer_auth`](src/middleware.rs) — enforces `TENSOR_WASM_API_TOKENS`. Dev
  mode pass-through when the allowlist is empty.
* [`tenant_scope`](src/middleware.rs) — parses `X-TensorWasm-Tenant` into a
  `TenantId` extension and applies the `TENSOR_WASM_API_REQUIRE_TENANT` policy.

The stack composition lives in `server.rs` so individual middleware can be
re-used by integration tests and benchmarks. See
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) for how the gateway sits relative
to the rest of the system.

## Stability

Endpoint paths, HTTP status codes, and `error.kind` values are part of the
public contract. Response field names are stable; additional fields may be
added in a forward-compatible manner. The `result` payload of `/invoke` and
`/jobs/{id}` is stable as currently shaped (`{ "function_id", "result" }`);
once the executor surfaces Wasm return values directly, the `result` field
will become the JSON-encoded return value of the entry point.

---

*The HTTP surface, validation, error envelope, and registries are wired
directly to the production `TensorWasmExecutor`. For ahead-of-time context see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) and [`docs/BUILD.md`](../../docs/BUILD.md).*
