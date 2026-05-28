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

## v0.4 implementation plan

1. `InstanceState` in `tensor-wasm-exec` grows a `StreamingContext`
   field, set via a new `SpawnConfig::with_streaming(sender)` builder.
2. `tensor_wasm_wasi_gpu::add_streaming_to_linker` is called from the
   executor's linker build path, gated on the new spawn-config field.
3. The route handler in `crates/tensor-wasm-api/src/routes.rs` swaps
   the scaffold `event: scaffold` body for a real
   `mpsc::Receiver<Vec<u8>>` drain, instrumented with a per-chunk
   sanitiser (see Security below).
4. OpenAPI: the v0.3.7 entry already documents the SSE / chunked
   shapes; v0.4 only refines the response-body schema.

The URL, method, and response framing are pinned by v0.3.7 — no
breaking changes between v0.3.7 and v0.4.

## Security

* **Log-injection / ANSI-escape stripping.** Guest-emitted bytes flow
  through a `sanitize_path`-equivalent filter before they hit the SSE
  / chunked response body. The filter strips ASCII control bytes
  (`\x00`-`\x1F` except `\t`, `\n`, `\r`) and 7-bit ANSI escape
  sequences (`\x1B[...m`) so a hostile guest cannot smuggle escape
  sequences through an operator's `journalctl` window when the request
  hits a debugging proxy. This filter lives in v0.4; the v0.3.7
  scaffold returns only host-controlled bytes, so the filter is
  deliberately not yet on the response path.
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
