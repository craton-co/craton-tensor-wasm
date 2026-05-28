# Streaming HTTP invocations (`/invoke-stream`)

Roadmap feature #2: Server-Sent Events / chunked-transfer responses from
Wasm-hosted LLM and token-streaming workloads.

> **Status:** v0.3.7 lands the scaffold — route, WIT contract, and
> `StreamingContext` channel surface — but does **not** yet drive the
> executor through a streaming invocation. v0.4 wires actual mid-
> execution streaming. Clients that hit the route today receive a
> single `event: scaffold` frame carrying
> `{"status":"not_yet_wired"}` and then end-of-stream.

## Motivation

LLM token decoding, audio-frame inference, and any workload that emits
intermediate output benefits from server-pushed bytes — the alternative
is `GET /jobs/{id}` poll loops whose latency floor is one round-trip per
poll. Adding streaming on a dedicated `/invoke-stream` URL keeps the
existing synchronous `/invoke` route's single-shot JSON envelope intact
while giving streaming workloads a first-class wire path.

## Wire shape

The route is `POST /functions/{id}/invoke-stream`. Request body matches
`/invoke` (currently empty / ignored — schema reserved for the v0.4
argument-passing landing). Response framing is negotiated from the
request's `Accept` header.

### SSE branch — `Accept: text/event-stream`

The response carries `Content-Type: text/event-stream` and uses the
standard SSE framing:

```
event: chunk
data: <chunk bytes, base64 or UTF-8 depending on content>

event: chunk
data: ...

event: end
data:
```

Each chunk the guest emits via `wasi:tensor/host.emit-chunk` becomes
one `data:` frame. The stream terminates with `event: end` when the
guest returns. A `keep-alive` comment line is injected on idle so
HTTP/2 proxies and load balancers don't reap the connection.

### Chunked-transfer branch — default

Any other `Accept` (or none) selects `Content-Type:
application/octet-stream` with `Transfer-Encoding: chunked`. Each
guest-emitted chunk is forwarded verbatim as one HTTP chunk frame. No
SSE prefix; clients consume raw bytes.

This is the lower-overhead path for non-browser consumers — e.g. a
Python `requests.post(..., stream=True)` reader that wants bytes
straight from the wire.

## Caps

Two hard caps live in `crates/tensor-wasm-wasi-gpu/src/streaming.rs`
and are documented in `wit/wasi-tensor.wit`:

| Constant                  | Value   | Purpose                                                      |
|---------------------------|---------|--------------------------------------------------------------|
| `MAX_CHUNK_BYTES`         | 64 KiB  | Single-`emit-chunk` cap; matches typical HTTP encoder buffer |
| `MAX_TOTAL_STREAM_BYTES`  | 64 MiB  | Per-invocation total. Trips the `-2` error code on overflow  |

A guest that exceeds the total cap receives `-2` from `emit-chunk` and
must stop emitting. The gateway never truncates mid-chunk: bytes that
are accepted are forwarded in full.

## Host contract

The WIT contract lives at
`crates/tensor-wasm-wasi-gpu/wit/wasi-tensor.wit`:

```
package wasi:tensor@0.1.0;

interface host {
    emit-chunk: func(bytes: list<u8>) -> s32;
    flush: func() -> s32;
}

world tensor-streaming { import host; }
```

Negative return codes:

* `-1` — streaming not enabled for this invocation (route was not
  `/invoke-stream`, or the gateway didn't attach a receiver).
* `-2` — guest tried to emit past the documented size cap.
* `-3` — downstream client disconnected (HTTP receiver dropped).

The host side is `tensor_wasm_wasi_gpu::streaming::StreamingContext`,
a clone-able value owning a `tokio::sync::mpsc::Sender<Vec<u8>>`. The
gateway holds the matching `Receiver` and drives the SSE / chunked
response body off it.

## v0.4 wiring (T34, landed)

1. `InstanceState` in `tensor-wasm-exec` carries a `StreamingContext`
   field. Default is `StreamingContext::disabled()`; spawns through
   `/invoke-stream` install a real channel-backed context via the new
   `SpawnConfig::with_streaming(ctx)` builder.
2. `tensor_wasm_wasi_gpu::add_streaming_to_linker` is invoked from
   `TensorWasmExecutor::spawn_instance` when `SpawnConfig::streaming`
   is `Some`; non-streaming spawns retain the empty-imports
   `Instance::new_async` path verbatim so existing callers are
   unaffected.
3. The `invoke_function_stream` handler in
   `crates/tensor-wasm-api/src/routes.rs` builds the `(tx, rx)` pair,
   spawns the executor call onto a Tokio task, and converts `rx` into
   an `axum::response::sse::Sse` (SSE branch) or
   `Body::from_stream` (chunked-transfer branch). A `oneshot::channel`
   carries the terminal status (success / deadline_elapsed /
   wasm_error) so the writer can emit a final `event: done` /
   `event: error` frame.
4. OpenAPI: the operation description now references the
   `event: chunk` / `event: done` / `event: error` framing instead of
   the v0.3.7 `event: scaffold` placeholder.

The URL, method, and response framing are unchanged between v0.3.7
and v0.4 — only the body content swapped from the scaffold marker to
real guest output.

## Security

* **Per-chunk byte payload sanitisation.** Per the threat model, the
  host does NOT sanitise chunk payloads on the server side — bytes
  flow guest→client verbatim. Sanitisation (ASCII control / ANSI
  escape stripping) is the client's responsibility; the CLI's T18
  receive-side scrubber handles incoming text. The reason: any
  server-side filter on byte payloads would prevent legitimate
  binary streaming (Parquet pages, protobuf-encoded events,
  image tiles).
* **Downstream disconnect.** The gateway monitors the
  `mpsc::Sender::send` result for `SendError`; on receiver drop the
  guest's next `emit-chunk` returns `-3` and the executor's deadline
  guard tears the instance down.
* **Total-bytes cap.** `MAX_TOTAL_STREAM_BYTES` bounds the per-
  invocation byte budget so a runaway guest cannot exhaust gateway
  heap.
* **Tenant isolation.** The route inherits the bearer-auth / tenant-
  scope envelope from `/invoke` (same middleware stack via
  `build_router_with_audit`'s `invoke_router`). Audit log records use
  the same `function_id` / `tenant` / `actor` shape — operators see
  streaming invocations alongside synchronous ones in
  `docs/AUDIT-LOG.md`.

## Testing

* Host-side: `crates/tensor-wasm-wasi-gpu/tests/streaming_scaffold.rs`
  exercises the four error codes (`-1`, `-2`, `-3`, success) on the
  `StreamingContext` directly.
* API-side:
  `crates/tensor-wasm-api/tests/streaming_invoke_scaffold.rs` POSTs
  to `/invoke-stream` and asserts the SSE / chunked content-types
  and the `scaffold` event payload.
