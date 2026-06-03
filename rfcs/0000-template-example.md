<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# RFC 0000 (EXAMPLE — NOT ACCEPTED): Tenant quota export endpoint

> **This file is a worked example showing what a filled-in RFC looks
> like.** It is **not** an accepted RFC, the design is intentionally
> hypothetical, and no code should be written against it. It lives at
> the top of `rfcs/` (not under `accepted/`) for exactly this reason.
> Copy [`TEMPLATE.md`](TEMPLATE.md), not this file, when drafting your
> own RFC.

- **Author(s):** Example Author <example@craton.com.ar>
- **Status:** Example (not real)
- **Created:** 2026-05-24
- **Discussion PR:** (none — this is a teaching example)
- **Related:** Hypothetical follow-up to the observability workstream
  in [`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md) v0.3.

## Summary

Add a read-only HTTP endpoint, `GET /tenants/{id}/quota`, that returns
the tenant's current quota allocation and usage as JSON. The endpoint
is intended for dashboards and operator tooling; it changes no
behaviour, mutates no state, and is opt-in via the existing bearer
token model with a new `tenant:read` scope.

## Motivation

Today, the only way for an operator to inspect a tenant's quota is to
scrape Prometheus metrics and reconstruct the allocation from
`tensor_wasm_tenant_quota_bytes_total` minus
`tensor_wasm_tenant_quota_bytes_used`. That requires the operator to
have Prometheus scrape access, know the metric names, and trust the
scrape interval. It also doesn't surface the original *configured*
quota — only the current snapshot.

Two operator reports (issue #NNN, design-partner feedback recorded in
the v0.3 sync notes) asked for a direct query. The v0.3 milestone in
[`docs/PATH-TO-V1.md`](../docs/PATH-TO-V1.md) already commits to
shipping a `tensor-wasm-cli observe` subcommand; this endpoint is what
that subcommand calls under the hood for the per-tenant view.

## Detailed design

### HTTP surface

```
GET /tenants/{tenant_id}/quota
Authorization: Bearer <token>
```

Response, `200 OK`, `content-type: application/json`:

```json
{
  "tenant_id": "acme-prod",
  "quota": {
    "gpu_memory_bytes": 8589934592,
    "wasm_memory_bytes": 1073741824,
    "concurrent_invocations": 64
  },
  "usage": {
    "gpu_memory_bytes": 3145728000,
    "wasm_memory_bytes": 671088640,
    "concurrent_invocations": 12
  },
  "as_of": "2026-05-24T15:42:11.214Z"
}
```

`as_of` is the monotonic timestamp at which the snapshot was read from
`TenantRegistry`. Callers polling this endpoint should treat the
response as a point-in-time reading; there is no subscription / SSE
variant in this RFC.

### Auth and scoping

The endpoint requires a bearer token with a new `tenant:read` scope,
or the legacy `tenant: *` scope for backwards compatibility (the same
rule the rest of the API follows during the v0.4 scoped-tokens
rollout — see the v0.4 milestone). Tokens without either scope receive
`403 Forbidden` with the standard problem-details body.

### Errors

| Status | Condition | Body |
|---|---|---|
| `401` | Missing or malformed bearer | standard auth error |
| `403` | Token lacks `tenant:read` for this tenant | standard auth error |
| `404` | `tenant_id` unknown to the registry | `{"error":"tenant_not_found"}` |
| `503` | `TenantRegistry` lock contended past 50 ms timeout | retry-with-backoff |

### Implementation sketch

- New handler in `crates/tensor-wasm-api/src/routes/tenants.rs`,
  ~80 LOC.
- Wires through to the existing `TenantRegistry::snapshot(tenant_id)`
  method (already public, currently used only by metrics).
- One new integration test under
  `crates/tensor-wasm-api/tests/quota_endpoint.rs`: happy path,
  unknown tenant, missing scope, malformed token.
- OpenAPI spec gains one path and one schema; the existing
  CI step that round-trips a generated client picks it up.

### Rollout

Single PR. No migration. Default build includes the route; tokens
issued before v0.x' have the `tenant: *` legacy scope and continue to
work. New tokens minted with `tensor-wasm-cli auth issue` gain a
`--scope tenant:read` flag.

## Drawbacks

- **Adds a stable HTTP route to the v1.0 surface.** Once shipped, the
  response schema becomes part of the SemVer contract.
- **Duplicates information already available via metrics.** Two ways
  to read the same number means two ways for them to disagree if a
  refactor only touches one path. The test plan above covers this with
  an assertion that the endpoint reading equals the corresponding
  metric reading at the same instant.
- **Lock contention concern.** `TenantRegistry::snapshot` takes a read
  lock; under heavy registration churn, polling this endpoint could
  add measurable contention. Mitigated by the 50 ms timeout and the
  `503` retry hint, but worth measuring before declaring the path
  GA-ready.

## Rationale and alternatives

### Alternative A: expose only via the existing `/metrics` endpoint

**What it is.** Add `tensor_wasm_tenant_quota_configured_bytes` as a
new Prometheus gauge alongside the existing
`tensor_wasm_tenant_quota_bytes_total` and
`tensor_wasm_tenant_quota_bytes_used`. Operators scrape with Prometheus
and query with PromQL.

**Why rejected.** The reports motivating this RFC came from operators
who do not have Prometheus scrape access for policy reasons. Forcing
them through a metric pipeline doesn't solve the problem. It also
doesn't give the `as_of` timestamp, which matters for incident
post-mortems.

**What would change the calculus.** If we discover that all operators
who need this data also have Prometheus, the metric-only path is
strictly less code and we should prefer it.

### Alternative B: streaming server-sent events for live updates

**What it is.** `GET /tenants/{id}/quota/stream` returns an SSE stream
emitting a JSON event whenever the tenant's usage crosses a configured
threshold.

**Why rejected.** Too much scope for one RFC; the polling endpoint
covers 90% of the use case at 10% of the complexity. SSE introduces a
new long-lived connection class to the API gateway, which interacts
with the rate-limiting design landing in v0.4 and would benefit from
its own RFC.

**What would change the calculus.** If polling overhead shows up in
the v0.3 dashboard work, an SSE follow-up RFC becomes worth writing.

### Alternative C: do nothing

**What it is.** Operators continue to scrape Prometheus, and the
`tensor-wasm-cli observe` subcommand reads metrics directly instead of
going through the API.

**Why rejected.** Pushes the metric-pipeline dependency into the CLI,
which would prevent operators from using `observe` against a fresh
deployment that hasn't wired up its scrape config yet. Also keeps the
`as_of` gap.

## Unresolved questions

- *Does the response include a `quota_class` field naming the policy
  the quota was sourced from (default, override, override-pending)?*
  Proposed: no, defer to a follow-up that designs the policy surface
  end-to-end. Open until the v0.4 quota-policy work scopes.
- *What is the polling rate limit?* Proposed: same as the per-token
  default landing in v0.4 (no special case). Open until v0.4 RFC
  numbers settle.
- *Do we add a `since=<timestamp>` query param to skip responses where
  nothing changed?* Proposed: no, keep this endpoint stateless; clients
  can compare responses themselves. Open if dashboard polling proves
  expensive.

## Prior art

- **Kubernetes `ResourceQuota` status subresource.** Reports
  `status.used` and `status.hard` per quota object. We mirror the
  used/configured split. Difference: k8s exposes via the apiserver
  watch API; we deliberately stay on polling for v1.0.
- **AWS Lambda `GetAccountSettings`.** Returns
  `AccountLimit` + `AccountUsage`. Schema is similar; AWS's
  per-function variant (`GetFunctionConcurrency`) is the rough
  equivalent of this per-tenant call.
- **Wasmtime `Engine::config`.** Read-only inspection of configured
  limits is a known-good pattern; this RFC extends the idea to the
  multi-tenant runtime layer.

## Future possibilities

- **Aggregate `/quotas` endpoint** listing every tenant's snapshot in
  one call, for operators with many tenants. Out of scope here because
  it raises auth questions (does `tenant:read` for all tenants imply
  `tenant:list`?) that deserve their own RFC.
- **Historical quota usage** via a time-series sub-resource. Would
  require persistent storage; the current `TenantRegistry` is
  in-memory.
- **Quota change notifications** as the SSE alternative above.
- **A `quota` field on the existing `/functions/{id}/invoke` response**
  so callers see remaining headroom inline. Cheap to add later if
  demand materialises.
