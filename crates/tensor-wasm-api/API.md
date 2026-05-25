# Craton TensorWasm HTTP API Reference

The `tensor-wasm-api` crate exposes the public REST surface of a [TensorWasm](../../ARCHITECTURE.md)
node. This document covers every endpoint the gateway serves. For build
instructions see [`docs/BUILD.md`](../../docs/BUILD.md); for the system-wide
architecture see [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

All requests and responses use JSON unless otherwise noted. Identifiers are
[RFC 4122] UUIDv4 strings.

[RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122

A machine-readable OpenAPI 3.0 description ships alongside this document at
[`openapi.json`](openapi.json). The canonical OpenAPI 3.1 spec — validated
against the live router on every CI run — lives at
[`openapi/tensor-wasm-api.yaml`](../../openapi/tensor-wasm-api.yaml); see
`crates/tensor-wasm-api/tests/openapi_validation_test.rs` and the `openapi`
job in `.github/workflows/ci.yml` for the parity contract.

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

#### Per-tenant scopes (`TENSOR_WASM_API_TOKENS` entry forms)

Each entry in `TENSOR_WASM_API_TOKENS` carries an optional `:tenant=...`
clause that restricts the token to a subset of tenants. Three entry shapes
are accepted in the same env var; mixing forms is allowed.

| Entry shape                       | Meaning                                                  | Status                                          |
|-----------------------------------|----------------------------------------------------------|-------------------------------------------------|
| `token`                           | Bare token; coerced to wildcard scope (`tenant=*`).      | **Deprecated**, scheduled for removal in v1.0.  |
| `token:tenant=*`                  | Explicit wildcard scope — addresses every tenant.        | Stable.                                         |
| `token:tenant=1,2,3`              | Token may address tenants `1`, `2`, or `3` only.         | Stable.                                         |

Comma-separated tenant ids may include surrounding whitespace; the parser
splits on commas at the top level and re-glues continuation lists back into
the owning entry. Tenant ids that are neither a `u64` nor `*` (e.g.
`tenant=1,foo`) cause that single entry to be dropped at startup with a
`tracing::warn!`; the remaining entries continue to work.

When a route that binds to a tenant receives a request whose token is not
scoped to the bound tenant, the gateway returns:

```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{ "error": { "kind": "tenant_scope_denied", "message": "bearer token is not scoped to tenant 7; extend the token's tenant= clause in TENSOR_WASM_API_TOKENS" } }
```

Tenant scoping is enforced inside the route handlers (after middleware
auth + rate-limit pass), so 403 is returned with no Wasm spawn, no
executor work, and no metrics charged for the rejected invocation.

The routes that perform a tenant-scope check are the per-tenant
invocation paths:

* `POST /functions/{id}/invoke`
* `POST /functions/{id}/invoke-async`

Routes that do not bind to a tenant (`POST /functions`, `DELETE
/functions/{id}`, `GET /jobs/{id}`, `GET /healthz`, `GET /metrics`)
do not perform the check — they are allowed for any token that passes
the bearer-auth allowlist.

#### Migration from bare tokens

Bare entries are accepted for backwards compatibility and emit a one-shot
deprecation warning at startup, naming the count of bare entries observed:

```
WARN  tensor_wasm_api::middleware: bare bearer tokens in TENSOR_WASM_API_TOKENS are deprecated; switch to `token:tenant=...` (or `token:tenant=*` for the current wildcard behaviour) — bare entries are scheduled for removal in v1.0
```

To migrate, append `:tenant=*` to each entry to preserve current
behaviour:

```diff
- export TENSOR_WASM_API_TOKENS=secret-prod-token,canary-token
+ export TENSOR_WASM_API_TOKENS=secret-prod-token:tenant=*,canary-token:tenant=*
```

Or take advantage of the new surface to issue per-tenant tokens:

```bash
export TENSOR_WASM_API_TOKENS=prod-tenant-7:tenant=7,prod-tenant-8:tenant=8,admin:tenant=*
```

Bare entries are scheduled for removal in **v1.0**; configurations that
still rely on them at v1.0 will fail to start. The deprecation period
opens in v0.4 with this release.

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

### Per-token rate limiting

The gateway applies a per-bearer-token QPS + burst rate limit using an
in-process token bucket. The limit is configured via two environment
variables read at startup:

| Env var                              | Default | Meaning                                             |
|--------------------------------------|---------|-----------------------------------------------------|
| `TENSOR_WASM_API_RATE_LIMIT_QPS`     | unset   | Steady-state requests per second admitted per token.|
| `TENSOR_WASM_API_RATE_LIMIT_BURST`   | unset   | Bucket capacity per token (maximum burst).          |

* If **either** variable is unset, `0`, or unparseable, the limiter is
  **disabled** — equivalent to v0.1 behaviour.
* If one variable is set but the other is not, the missing knob defaults
  to `qps=100` / `burst=200` respectively.
* When enabled, a missing or unauthenticated request never consumes a
  permit — the rate-limit layer runs *after* bearer auth.
* In dev mode (empty allowlist) every request shares a single synthetic
  bucket; the limiter still throttles the total dev-mode request rate.

Rejected requests return `429 Too Many Requests` with the standard
`{ "error": { "kind": "rate_limited", "message": ... } }` envelope and a
`Retry-After` header. Per RFC 9110 §10.2.3 we emit the value as a
non-negative integer number of seconds (never a date). The integer is the
ceiling of `(1 - bucket_tokens) / qps`, clamped to at least `1` so even
high-QPS configurations produce an actionable back-off.

Example response:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 1

{ "error": { "kind": "rate_limited", "message": "per-token rate limit exceeded; retry after 1s" } }
```

Buckets are per-token, sharded by `dashmap`, and recover on a lazy
refill-on-take schedule. No external store (Redis, etc.) is required.

### Audit log

Every state-mutating API call (`POST /functions`, `DELETE
/functions/{id}`, `POST /functions/{id}/invoke`, `POST
/functions/{id}/invoke-async`) emits a single JSON line to the audit
sink. Read-only calls (`GET /healthz`, `GET /metrics`, `GET /jobs/{id}`)
emit nothing — they are intentionally suppressed to keep the stream
useful.

The destination sink is selected at server startup by
`TENSOR_WASM_API_AUDIT_LOG`:

| Value                     | Resulting sink                                                |
|---------------------------|----------------------------------------------------------------|
| (unset) or `stdout`       | One JSON object per line on stdout (the default).             |
| `none`                    | Audit disabled. Useful when Prometheus + OTel are sufficient. |
| `file:/path/to/audit.log` | Append-only JSONL file at the given path.                     |

Each record carries the actor (bearer-token id + scope), the action,
the resource (function id + tenant), the outcome (status code +
`error.kind` for non-2xx), an end-to-end latency in milliseconds, and
a per-request UUID also stamped into the request extensions so handlers
can correlate their own logs. When an XFCC-aware reverse proxy fronts
the gateway (see [`docs/deployment/mtls.md`](../../docs/deployment/mtls.md)),
the parsed `Subject` from `X-Forwarded-Client-Cert` is recorded as
`client_cert_subject`.

The wire-format schema, sample records, log-rotation guidance, latency
budget, and the mTLS / XFCC trust boundary are documented end-to-end in
[`docs/AUDIT-LOG.md`](../../docs/AUDIT-LOG.md).

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
| `tenant_scope_denied` | 403 | Bearer token is not scoped to the `X-TensorWasm-Tenant` named in the request. |
| `not_found`        | 404  | Requested function or job id does not exist.                             |
| `body_too_large`   | 413  | Inbound body exceeds the 64 MiB cap (often rendered as bare 413).        |
| `rate_limited`     | 429  | Per-token QPS + burst exceeded; response carries `Retry-After: <secs>`.  |
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
  mode pass-through when the allowlist is empty. On success inserts an
  `AuthContext { token_id, scope }` into the request extensions for
  downstream middleware and handlers to consume. The `scope` field is the
  per-tenant [`TokenScope`](src/token_scope.rs) parsed from the
  `token:tenant=...` clause; bare entries get the wildcard scope with a
  one-shot startup deprecation warning. Routes that bind to a tenant call
  `AuthContext::authorize_tenant` and return `403 tenant_scope_denied`
  when the scope does not cover the bound tenant.
* [`tenant_scope`](src/middleware.rs) — parses `X-TensorWasm-Tenant` into a
  `TenantId` extension and applies the `TENSOR_WASM_API_REQUIRE_TENANT` policy.
* [`rate_limit`](src/rate_limit.rs) — token-bucket per `AuthContext.token_id`.
  Reads `TENSOR_WASM_API_RATE_LIMIT_QPS` and `TENSOR_WASM_API_RATE_LIMIT_BURST`;
  no-op when either is unset or zero. Returns `429` + `Retry-After` on
  bucket-empty.
* [`audit_log_middleware`](src/audit.rs) — emits one JSON record per
  state-mutating call to the sink selected by `TENSOR_WASM_API_AUDIT_LOG`
  (default stdout, `file:/path` for an append-only file, `none` to
  disable). Runs innermost so the recorded outcome reflects the final
  status code; suppresses read-only routes via a path-shape filter.

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
