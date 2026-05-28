# OpenAI-compatible inference gateway

**Status:** v0.3.5 ships the **scaffold** (this document). The wiring step
that translates between OpenAI requests and the native invoke pipeline
lands in v0.4 — see `docs/PATH-TO-V1.md` for the milestone exit criteria.

The TensorWasm API gateway exposes two OpenAI-compatible inference
routes alongside its native `/functions/{id}/invoke` surface, so that
off-the-shelf OpenAI SDKs (Python `openai`, Node `openai`, LangChain,
LlamaIndex, …) can target a TensorWasm deployment without
modification.

## Route surface

| Method | Path                    | Status today | v0.4 behaviour                                   |
| ------ | ----------------------- | ------------ | ------------------------------------------------ |
| POST   | `/v1/completions`       | `501` scaffold | Resolve `model` → function, marshal `prompt`     |
| POST   | `/v1/chat/completions`  | `501` scaffold | Resolve `model` → function, marshal `messages`   |

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

The scaffold's job is to commit three things to the public contract:

1. **The URL surface.** Clients can begin integrating against the
   gateway URL today; the v0.4 wiring step will not move the routes.
2. **The request shape.** Every documented OpenAI field is accepted
   (`#[serde(default)]`); the v0.4 wiring step may add semantic
   validation but will not reject any field the scaffold accepts.
3. **The error envelope.** OpenAI SDKs parse the four-field
   `{ "message", "type", "param", "code" }` body verbatim and will not
   look at the gateway's native `{ "kind", "message" }` shell. The
   scaffold returns the OpenAI shape from the start so SDK error
   paths exercise the same code today and after v0.4.

The scaffold does **not** validate semantics (model existence,
`max_tokens` upper bounds, etc.) — those land in v0.4.

## Wire-format examples

### Scaffold response (today)

```http
POST /v1/completions HTTP/1.1
Authorization: Bearer my-token
Content-Type: application/json

{ "model": "tensor-wasm-llama", "prompt": "Hello" }

HTTP/1.1 501 Not Implemented
Content-Type: application/json

{
  "error": {
    "message": "OpenAI-compatible /v1/completions endpoint is a scaffold; …",
    "type": "not_implemented",
    "param": null,
    "code": "openai_not_yet_wired"
  }
}
```

Clients should branch on `error.code == "openai_not_yet_wired"`, not on
the human-readable `message`.

### Malformed body (today)

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

## v0.4 implementation plan

The wiring step lands on top of this scaffold in four chunks:

1. **`model` → function resolution.** A new env-driven allowlist
   (`TENSOR_WASM_OPENAI_MODEL_ALIASES=gpt-3.5-turbo=<uuid>,…`) maps
   each OpenAI model identifier to a deployed `FunctionRecord`.
   Unknown models return `404` with the OpenAI envelope (`type:
   "model_not_found"`).
2. **Tenant inference.** Tenant scope comes from the bearer token's
   `:tenant=` clause (see `crates/tensor-wasm-api/src/token_scope.rs`).
   A wildcard token routes to tenant 0 with a one-shot warning.
3. **Argv marshalling.** The translator serialises the OpenAI request
   into the wasm guest's `_start` argv as a single JSON blob. The guest
   sees one `arg0` containing the prompt or messages array; v0.4
   tightens this once a stable in-tree schema is published.
4. **SSE streaming.** When `stream: true`, the handler returns
   `text/event-stream` and writes one OpenAI `data:` chunk per token
   the guest emits via the `wasi:io/streams` host export.

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
