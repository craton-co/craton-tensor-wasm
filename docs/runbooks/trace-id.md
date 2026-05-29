<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Runbook: I have a trace id, how do I find the related logs?

This is not an alert runbook — nobody pages on a trace id. It's the
companion to the burn-rate and latency runbooks: when one of those
pages an operator, the operator usually wants to pivot from a single
captured request to the full set of logs and downstream spans
associated with that request. This file is the recipe for that pivot.

## What's a trace id, and where do I get one?

Every response from the TensorWasm API gateway carries an `x-trace-id`
HTTP response header. The value is a 32-character lowercase hex string
identifying the W3C trace the request joined — either the trace id from
the inbound `traceparent` header sent by an upstream caller, or a fresh
root assigned by the gateway when no `traceparent` was supplied. Sample:

```text
x-trace-id: 0af7651916cd43dd8448eb211c80319c
```

The same trace id appears as the `trace_id` field on every span the
gateway emits for that request, and as the `trace_id` field on every
log line emitted from within those spans (when the
`tracing_subscriber::fmt::Layer` is configured with
`with_current_span(true)`, which is the default in `init_with_otlp`).

If the header is missing entirely the gateway is most likely running
without a `tracing_opentelemetry` subscriber installed — check that the
binary was built with `--features tensor-wasm-core/otlp` and that
`init_with_otlp` ran at startup. The propagator-only path
(`install_w3c_propagator`) is enabled unconditionally in
`build_router`, but the trace id only resolves to a non-zero value when
an OTel layer is active.

## Step 1: confirm the trace id and grab it

If a user reported the failure with a screenshot or HAR file, the
`x-trace-id` header is in the response. From a `curl -i` capture:

```sh
curl -i -X POST http://gateway:8080/functions/$ID/invoke -d '...' \
  | grep -i '^x-trace-id'
```

If you are reproducing the failure yourself, send a fresh request and
note the header:

```sh
curl -sS -D - -X POST http://gateway:8080/functions/$ID/invoke -d '...' \
  -o /dev/null | grep -i '^x-trace-id'
```

Save the 32-char hex value as a shell variable:

```sh
TRACE_ID=0af7651916cd43dd8448eb211c80319c
```

## Step 2: pull the logs

The gateway writes structured logs to stdout / journald. The trace id
is included on every line emitted from within the request's span tree
(handler, executor, snapshot, dispatch), so a single grep across the
log stream is enough to recover the full timeline:

```sh
# journald
journalctl -u tensor-wasm --since "10 min ago" -o cat \
  | grep -F "$TRACE_ID"

# container stdout
docker logs --since 10m tensor-wasm 2>&1 | grep -F "$TRACE_ID"

# k8s pod
kubectl logs -n tensor-wasm deploy/tensor-wasm --since=10m \
  | grep -F "$TRACE_ID"
```

Order the matches by `created_at` if your subscriber emits one (the
default JSON formatter does). The first line is typically the
`http.request` span open from the tower trace layer; the last is the
response stamp from the audit middleware.

## Step 3: open the trace in your OTLP backend

If `OTEL_EXPORTER_OTLP_ENDPOINT` is set and the collector is healthy,
the same trace id maps to a single distributed trace in Jaeger,
Tempo, or Honeycomb. Paste the hex string into the backend's
"Find by trace id" box. The expected span tree is documented in
`docs/OBSERVABILITY.md` § "Propagation hop diagram" — verify it
matches what you see; missing hops usually mean the corresponding
crate is feature-gated out of the deployed binary
(e.g. `wasi_cuda.dispatch` is absent on no-CUDA hosts).

If the backend cannot find the trace, the most likely causes — in
descending order — are:

1. Collector is down or unreachable. `tensor-wasm` logs an
   `exporter: ...` error from the batch exporter when it can't push
   spans. Restart the collector or fix the network path.
2. Trace id was captured from a request that completed before the
   batch exporter flushed. The default flush interval is 5 s; wait
   that long, or shorten it via the OTel SDK env vars
   (`OTEL_BSP_SCHEDULE_DELAY`).
3. Subscriber is not actually wired. Confirm the binary was started
   with `init_with_otlp`, not plain `init`. The two are mutually
   exclusive — see `tensor-wasm-core::telemetry`.

## Step 4: pivot to metrics

The trace id alone does not carry tenant or function id. Read those
from the matched log lines (the `tensor` and `function_id` fields on
the `http.invoke_function` span) and use them to scope the Prometheus
queries documented in `docs/runbooks/invoke-latency-spike.md` and
`docs/runbooks/dispatch-latency-spike.md`. The dashboards in
`docs/dashboards/tensor-wasm-overview.json` accept the same
`tenant` label for per-tenant drill-down.

## Related docs

- `docs/OBSERVABILITY.md` — span schema, propagation hop diagram,
  env vars
- `docs/runbooks/invoke-latency-spike.md` — pages on slow `/invoke`
- `docs/runbooks/dispatch-latency-spike.md` — pages on slow dispatch
- `crates/tensor-wasm-api/src/trace_propagation.rs` — implementation
  reference for the propagator install + response-header injection
