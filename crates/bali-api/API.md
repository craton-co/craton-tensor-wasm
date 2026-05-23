# Bali HTTP API Reference (S17)

The `bali-api` crate exposes the public REST surface of a [Bali](../../ARCHITECTURE.md)
node. This document covers every endpoint in the S17 scaffold. For build
instructions see [`docs/BUILD.md`](../../docs/BUILD.md); for the system-wide
architecture see [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

All requests and responses use JSON unless otherwise noted. Identifiers are
[RFC 4122] UUIDv4 strings.

[RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122

## Conventions

### Base URL

A Bali node listens on a single `host:port`. There is no API versioning prefix
in S17 — once the surface stabilises (S25) the prefix `/v1/` will be introduced
and the unprefixed routes deprecated.

### Authentication

S17 has no authentication; the gateway trusts every caller. S20 introduces
tenant-scoped bearer tokens.

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

| `kind`            | HTTP | Meaning                                                    |
|-------------------|------|------------------------------------------------------------|
| `invalid_json`    | 400  | Request body could not be parsed as JSON, or shape wrong.  |
| `invalid_name`    | 400  | `name` field empty or whitespace-only.                     |
| `invalid_base64`  | 400  | `wasm_b64` field is not valid standard base64.             |
| `too_short`       | 400  | Decoded Wasm bytes < 8 bytes — no module header possible.  |
| `not_wasm`        | 400  | First four bytes are not the Wasm `\0asm` magic.           |
| `not_found`       | 404  | Requested function or job id does not exist.               |
| `internal`        | 500  | Unexpected server-side failure.                            |

---

## `POST /functions`

Deploy a new Wasm module. The module is stored in the in-memory registry; in
S17 it is not yet instantiated.

**Request body**

| Field      | Type   | Required | Description                                                                                                                                       |
|------------|--------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`     | string | yes      | Non-empty display name (free-form, not validated beyond non-emptiness).                                                                           |
| `wasm_b64` | string | yes      | Base64-encoded Wasm module bytes (standard alphabet, padded). Decoded value must be at least 8 bytes and begin with the `\0asm` magic.            |

**Success — `200 OK`**

```json
{ "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479" }
```

**Errors** — `400 Bad Request` with `kind` of `invalid_json`, `invalid_name`,
`invalid_base64`, `too_short`, or `not_wasm`.

**Example**

```bash
WASM_B64=$(printf '\0asm\x01\x00\x00\x00\x00' | base64)
curl -s -X POST http://localhost:8080/functions \
  -H 'content-type: application/json' \
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

Invoke a deployed function synchronously. Real Wasm execution wires through
`bali_exec::executor::BaliExecutor` in S20; in S17 this endpoint returns a
placeholder result so clients can be developed against the production HTTP
shape.

**Request body** — any JSON value, used as the (placeholder) argument
payload. An empty object `{}` is the conventional choice.

**Success — `200 OK`**

```json
{ "result": "ok", "function_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479" }
```

In S20 the `result` field becomes the JSON-encoded return value of the Wasm
entry point.

**Errors** — `404 Not Found` if the function id is unknown.

**Example**

```bash
curl -X POST http://localhost:8080/functions/$ID/invoke \
  -H 'content-type: application/json' \
  -d '{}'
```

---

## `POST /functions/{id}/invoke-async`

Fire-and-forget invocation. Returns a job id the caller can poll via
`GET /jobs/{id}`.

**Request body** — any JSON value (see `/invoke` above).

**Success — `200 OK`**

```json
{ "job_id": "5b3aa6c8-1f4f-4d23-bf01-2b1e10e7a4c9" }
```

**Errors** — `404 Not Found` if the function id is unknown.

**Example**

```bash
curl -X POST http://localhost:8080/functions/$ID/invoke-async \
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

`status` is one of `"pending"`, `"completed"`, or `"failed"`. When the job
finishes a `result` field carries the JSON-encoded payload:

```json
{
  "id": "5b3aa6c8-1f4f-4d23-bf01-2b1e10e7a4c9",
  "function_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "completed",
  "result": { "value": 42 },
  "created_unix_ms": 1716491220123
}
```

**Errors** — `404 Not Found` if the job id is unknown.

**Example**

```bash
curl http://localhost:8080/jobs/$JOB_ID
```

---

## `GET /metrics`

Prometheus text-format exposition. S17 returns a scaffold; S20 wires the
full `BaliMetrics` registry from `bali-core`.

**Success — `200 OK`**, `Content-Type: text/plain; version=0.0.4`

```
# bali metrics scaffold
# real exposition wires in S20
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
`bali_api::server::build_router`:

* [`trace_layer`](src/middleware.rs) — emits a `tracing` span per request with
  method, URI, and response status. Classifier treats `5xx` as failures.
* [`timeout_layer(30s)`](src/middleware.rs) — fails slow requests with
  `408 Request Timeout`.
* [`concurrency_limit_layer(64)`](src/middleware.rs) — caps in-flight requests
  process-wide. S20 replaces this with per-tenant buckets.

The stack composition lives in `server.rs` so individual middleware can be
re-used by integration tests and benchmarks. See
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) for how the gateway sits relative
to the rest of the system.

## Stability

Endpoint paths, HTTP status codes, and `error.kind` values are part of the
public contract. Response field names are stable; additional fields may be
added in a forward-compatible manner. Anything documented as "placeholder"
(notably `invoke`'s response shape) is **not** stable until S20.

---

*S17 is the HTTP scaffold: routing, validation, error envelope, and in-memory
registries. Real Wasm execution — driving `bali-exec::executor::BaliExecutor`
through the registries here — wires in S20. For ahead-of-time context see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) and [`docs/BUILD.md`](../../docs/BUILD.md).*
