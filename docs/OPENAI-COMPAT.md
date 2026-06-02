# OpenAI-compatible inference gateway

**Status:** v0.4 wiring landed (T41). The handlers translate OpenAI
requests through to the internal invoke pipeline via a configurable
`model → function_uuid` map. The v0.3.5 scaffold's `501
openai_not_yet_wired` shell is gone; the URL surface, request shapes,
and error-envelope contract the scaffold committed to are preserved.

The TensorWasm API gateway exposes two OpenAI-compatible inference
routes alongside its native `/functions/{id}/invoke` surface, so that
off-the-shelf OpenAI SDKs (Python `openai`, Node `openai`, LangChain,
LlamaIndex, …) can target a TensorWasm deployment without
modification.

## Route surface

| Method | Path                    | Status today (v0.4)                                                                 |
| ------ | ----------------------- | ----------------------------------------------------------------------------------- |
| POST   | `/v1/completions`       | Wired (T41). Resolve `model` → function, marshal `prompt`, stream / buffer reply.   |
| POST   | `/v1/chat/completions`  | Wired (T41). Resolve `model` → function, marshal `messages`, stream / buffer reply. |

Both routes accept the request shapes documented in the OpenAI REST
reference:

- <https://platform.openai.com/docs/api-reference/completions/create>
- <https://platform.openai.com/docs/api-reference/chat/create>

The Rust mirrors of those shapes live in
`crates/tensor-wasm-api/src/openai.rs` (`CompletionsRequest`,
`ChatCompletionsRequest`, `ChatMessage`). The OpenAPI spec at
`openapi/tensor-wasm-api.yaml` carries the same shapes under the
`openai-compat` tag.

## Scope

The v0.4 wire-up preserves the three commitments the v0.3.5 scaffold
locked in:

1. **The URL surface.** `POST /v1/completions` and `POST
   /v1/chat/completions`, exactly as documented in the scaffold.
2. **The request shape.** Every documented OpenAI field is accepted
   (`#[serde(default)]`); v0.4 added semantic validation for the
   `model` field (404 on miss) but does not reject any field the
   scaffold accepted.
3. **The error envelope.** OpenAI SDKs parse the four-field
   `{ "message", "type", "param", "code" }` body verbatim and will not
   look at the gateway's native `{ "kind", "message" }` shell. The
   wire-up keeps the OpenAI envelope on every error path that v0.4
   reaches.

T41-specific behaviour:

* `model_not_found` is returned with HTTP 404 and `param: "model"`
  whenever `req.model` is not present in the operator-configured
  model map (see *Operator configuration* below).
* Token-count fields in `usage` are zeros — v0.4 does not wire a
  tokenizer. v0.5 lands a real counter.
* Streaming is plumbed through the same `StreamingContext` the T34
  `/invoke-stream` route uses; one OpenAI `data: { ... }` SSE frame
  per emitted chunk + terminal `data: [DONE]\n\n`.

## Operator configuration

Wire-up the gateway to OpenAI clients by setting the
`TENSOR_WASM_API_OPENAI_MODEL_MAP` environment variable to a
comma-separated list of `model_id:function_uuid` pairs.

```bash
export TENSOR_WASM_API_OPENAI_MODEL_MAP='gpt-3.5-turbo:00000000-0000-4000-8000-000000000001,gpt-4:00000000-0000-4000-8000-000000000002'
```

Each `model_id` is the string OpenAI SDKs put in the `model` field;
each `function_uuid` is a UUID returned by `POST /functions` at deploy
time. Empty / unset means "no models configured" — every OpenAI
request fails with `404 model_not_found`. The map is read once at
startup; restart the gateway to pick up new aliases.

A YAML config-file alternative is on the v0.5 roadmap; the env var is
the only supported mechanism in v0.4.

## Wire-format examples

### Non-streaming completions (T41)

```http
POST /v1/completions HTTP/1.1
Authorization: Bearer my-token
Content-Type: application/json

{ "model": "gpt-3.5-turbo", "prompt": "Hello", "stream": false }

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "cmpl-<uuid>",
  "object": "text_completion",
  "created": 1748469000,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "text": "Hello, world!",
      "index": 0,
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

### Streaming chat completions (T41)

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer my-token
Content-Type: application/json

{ "model": "gpt-4", "messages": [{"role":"user","content":"Hi"}], "stream": true }

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1748469000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"H"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1748469000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"i"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1748469000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

### Unknown model (T41)

```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "error": {
    "message": "model `gpt-unknown` is not configured in TENSOR_WASM_API_OPENAI_MODEL_MAP; ask your operator to add a `gpt-unknown:<function_uuid>` entry",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Malformed body

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": {
    "message": "Failed to parse the request body as JSON: …",
    "type": "invalid_request_error",
    "param": null,
    "code": "openai_invalid_request"
  }
}
```

## v0.4 wiring (T41, landed)

T41 closed the four chunks the scaffold reserved:

1. **`model` → function resolution.** The
   `TENSOR_WASM_API_OPENAI_MODEL_MAP` env var (format:
   `model:uuid,model:uuid,...`) maps each OpenAI model identifier to a
   deployed `FunctionRecord`. Unknown models return `404` with
   `type: "invalid_request_error"`, `code: "model_not_found"`,
   `param: "model"`. See
   [`crates/tensor-wasm-api/src/openai_translator.rs`](../crates/tensor-wasm-api/src/openai_translator.rs).
2. **Tenant inference.** The OpenAI routes ride the same
   `tenant_scope` middleware T2 wired on the protected stack. Absent
   `X-TensorWasm-Tenant` resolves to `TenantId(0)` under the default
   policy; the bearer token's scope is then enforced via
   `AuthContext::authorize_tenant` BEFORE any translator work.
3. **Argv marshalling.** For v0.4 the translator passes an empty
   args vector and calls the guest's `_start` (`() -> ()`) export.
   Guests communicate the response by emitting bytes through the T34
   `wasi:tensor/host.emit-chunk` host function; the handler drains the
   matching receiver and surfaces every chunk as either a buffered
   string (`stream: false`) or an OpenAI SSE delta frame
   (`stream: true`). The prompt length is preserved on the
   `TranslatedRequest` struct (`prompt_len_hint`) so a future revision
   can promote it to a typed `i32` arg once the
   host-pre-fills-guest-memory plumbing lands; v0.4 deliberately keeps
   the export signature `_start() -> ()` so the existing WASI command
   guests link cleanly.
4. **SSE streaming.** When `stream: true`, the handler returns
   `text/event-stream` and writes one OpenAI `data: { ... }` SSE frame
   per emitted chunk, terminated by a `data: [DONE]\n\n` line. The
   plumbing reuses T34's `StreamingContext::with_channel` +
   `SpawnConfig::with_streaming` end-to-end.

### Configuration knob

Set `TENSOR_WASM_API_OPENAI_MODEL_MAP` to a comma-separated list of
`model:function_uuid` pairs. Empty / unset means "no models
configured" — every OpenAI request then surfaces 404 model_not_found.
A YAML config-file alternative is deferred to v0.5.

### Deferred to v0.5

* **Tokenizer.** `usage.{prompt,completion,total}_tokens` ship as
  zeros until a tokenizer lands. SDKs that compute billing from the
  usage block will see zero; the field is present so the response
  shape matches the OpenAI public contract.
* **Multimodal content.** Image / audio parts inside a chat message's
  `content` array are silently dropped — only text parts survive into
  the assembled prompt.
* **YAML config file.** The env var is the only supported map
  configuration.

## Security note: token scoping

OpenAI SDKs send `Authorization: Bearer <api_key>` but **never** an
`X-TensorWasm-Tenant` header. The gateway's native routes derive the
tenant from that header (via the `tenant_scope` middleware); the
OpenAI routes cannot, because the header is absent on the wire.

The OpenAI routes are mounted *outside* the `tenant_scope` middleware
in `crates/tensor-wasm-api/src/server.rs` for that reason — the layer
would otherwise reject every OpenAI request as `missing_tenant` 400.
**Tenant resolution comes from the bearer token's `TokenScope`
instead**: a scoped token (`mykey:tenant=7`) implies tenant 7; a
wildcard token implies the default tenant (0) with a one-shot warning.

Operators wiring OpenAI clients should provision one bearer token per
tenant in `$TENSOR_WASM_API_TOKENS` (`"sk-tenant7:tenant=7"`, etc.).
The token's `:tenant=` clause is the **only** source of tenant
identity for `/v1/...` routes; SDKs that try to forward
`X-TensorWasm-Tenant` will have the header silently ignored.

Bearer auth itself still runs on `/v1/...` routes: an unauthenticated
OpenAI client receives `401`, not `501`. Rate-limit and audit-log
middleware also run, so the operator-facing observability surface
remains uniform with the native routes.

## References

- OpenAI API reference: <https://platform.openai.com/docs/api-reference>
- Source: `crates/tensor-wasm-api/src/openai.rs`
- Spec: `openapi/tensor-wasm-api.yaml` (`openai-compat` tag)
- Tests: `crates/tensor-wasm-api/tests/openai_scaffold_test.rs`
- Token scope: `crates/tensor-wasm-api/src/token_scope.rs`
