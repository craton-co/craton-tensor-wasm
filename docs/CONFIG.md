# Configuration Reference

> Every env var consumed by `tensor-wasm`. This list was generated from a source
> sweep of `crates/*/src` (all `std::env::var` / `std::env::var_os` call sites
> and the `ENV_*` constants they read) at **v0.3.7**. Vars marked
> *documented-only* appear in operator docs but are **not** read by the Rust
> source at this version — see the notes on each.

## API gateway (`tensor-wasm-api`)

### Auth / hardening
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_API_TOKENS | (empty = dev-mode) | comma-separated list | Bearer token allowlist; empty/unset enables dev-mode (no auth). See `docs/SECURITY.md`. (`middleware.rs`) |
| TENSOR_WASM_API_ALLOW_DEV_MODE | (unset = dev-mode off) | flag (`1`/`true`) | Explicit opt-in that permits dev-mode (no-auth) operation when `TENSOR_WASM_API_TOKENS` is empty; unset leaves dev-mode disabled. (`middleware.rs`) |
| TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS | (empty) | comma-separated list | Bearer tokens authorised to publish kernels; empty disables the publish surface. (`middleware.rs`) |
| TENSOR_WASM_API_REQUIRE_TENANT | (unset = not required) | flag (`1`) | When `1`, requests must carry the tenant header. (`middleware.rs`) |
| TENSOR_WASM_API_TRUSTED_HOSTS | (empty = allow any) | comma-separated list | Allowed `Host` headers; empty allows any (logs a startup warning when also unauthenticated). (`middleware.rs`) |
| TENSOR_WASM_API_TRUSTED_XFCC_PROXIES | (empty = never trust) | comma-separated CIDR list | Peers whose `X-Forwarded-Client-Cert` header is honoured. (`audit.rs`) |
| TENSOR_WASM_API_CORS_ALLOWED_ORIGINS | (empty = no CORS) | comma-separated list | Origins for `Access-Control-Allow-Origin`. (`middleware.rs`) |
| TENSOR_WASM_API_METRICS_TOKEN | (empty = open) | string | Bearer token guarding the metrics endpoint; empty leaves it unauthenticated. (`server.rs`) |

### Rate limiting
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_API_RATE_LIMIT_QPS | 0 = token layer disabled | u32 | Per-token QPS cap. Both QPS and BURST unset, either `0`, or either unparseable disables the per-token backstop layer; the per-tenant default layer is unaffected. (`rate_limit.rs`) |
| TENSOR_WASM_API_RATE_LIMIT_BURST | 0 = token layer disabled | u32 | Per-token burst. If one of the pair is set and the other unset, the unset side falls back to its built-in default. (`rate_limit.rs`) |

### Snapshots
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_API_SNAPSHOT_HMAC_KEY | (empty) | 32-byte hex (64 chars) | HMAC key for signed snapshots. (`config.rs`) |
| TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE | false | bool | Refuse v2 (unsigned) snapshots when true. (`config.rs`) |

### Kernel registry
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_API_KERNEL_HMAC_KEY | (unset = registry off) | 32-byte hex (64 chars) | HMAC-SHA256 key used to verify inbound kernel manifests. Unset or malformed leaves the registry unconfigured (`/kernels` returns `503 kernel_registry_not_configured`). (`routes.rs`) |
| TENSOR_WASM_API_KERNEL_REGISTRY_DIR | (unset = in-memory) | path | When set, persists the registry to disk rooted at this path; unset keeps the legacy in-memory registry. (`routes.rs`) |

### OpenAI-compatible API
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_API_OPENAI_MODEL_MAP | (empty) | model-map string | Maps OpenAI model names onto deployed models for the compat endpoint. See `docs/OPENAI-COMPAT.md`. (`openai_translator.rs`) |

## Runtime / tenant (`tensor-wasm-tenant`)
| Variable | Default | Type | Purpose |
|---|---|---|---|
| CUDA_MPS_PIPE_DIRECTORY | (unset = probe `/tmp/nvidia-mps`) | path | NVIDIA MPS control-daemon pipe directory; the registry probes this (else the default path) to decide MPS vs. per-tenant contexts. Read via `var_os`. (`registry.rs`, exported as `MPS_PIPE_DIRECTORY_ENV`) |

## JIT / GPU (`tensor-wasm-jit`)
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_PLIRON_PIPELINE | (empty = off) | flag (non-empty = on) | Enables the experimental pLIRON MLIR pipeline. (`detector.rs`) |
| TENSOR_WASM_PTXAS | (unset) | path | Override for the `ptxas` binary path; consumed only by the JIT test harness. (`tests/ptx_validates.rs`) |
| CUDA_ARCH | none (required for GPU builds) | SM level (`sm_75`, `sm_80`, `sm_89`, `sm_90`, …) | *Documented-only at runtime:* target compute capability for PTX emission. Heavily used by build/deploy tooling (`docs/CUDA-SETUP.md`, `docs/DEPLOYMENT.md`), but **not** read via `std::env::var` in the Rust source at v0.3.7. |

## Telemetry / logging (`tensor-wasm-core`)
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_LOG | info (warn for the CLI) | `tracing_subscriber` EnvFilter directive | Preferred log-filter directive. Falls back to `RUST_LOG`, then to the built-in default. (`telemetry.rs`, `tensor-wasm-cli/src/main.rs`) |
| RUST_LOG | (unset) | EnvFilter directive | Standard fallback log filter, consulted only when `TENSOR_WASM_LOG` is unset. (`telemetry.rs`) |
| TENSOR_WASM_OTLP_ENDPOINT | (unset = OTLP off) | URL | Preferred OTLP collector endpoint. Consumed as the primary lookup name passed to `init_with_otlp` (gated on the `otlp` feature). (`telemetry.rs`) |
| OTEL_EXPORTER_OTLP_ENDPOINT | (unset) | URL | Standard OTLP fallback endpoint; used only when the TensorWasm-specific var is unset. Final fallback is the hardcoded `http://localhost:4317`. (`telemetry.rs::resolve_otlp_endpoint`) |
| OTEL_SERVICE_NAME | tensor-wasm | string | *Documented-only:* `docs/OBSERVABILITY.md` advertises this as the service-name override, but the exporter sets the service name to the hardcoded `"tensor-wasm"` and does **not** read this var at v0.3.7. |

## CLI (`tensor-wasm-cli`)
| Variable | Default | Type | Purpose |
|---|---|---|---|
| TENSOR_WASM_TOKEN | (empty) | string | Bearer token used by `--server` calls; warns on plaintext to a non-loopback host. (`cmd/mod.rs`) |
| TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC | false | flag (clap `env`, bool) | Opt-in that silences the dev-mode public-bind safety gate (does not enable auth; the recurring "no auth + public bind" warning still fires). (`cmd/serve.rs`) |
| TENSOR_WASM_REQUIRE_KEY_PERMS | (unset = lenient) | flag (`1`) | When `1`, refuse snapshot key files whose mode is group/other-accessible. (`cmd/snapshot.rs`) |

## Build-time variables (compile-time only)
These are not runtime env vars an operator sets; they are read by
`tensor-wasm-core/build.rs` during compilation (or supplied by cargo) and baked
into the binary via `env!`, surfacing in the `tensor_wasm_build_info` metric and
`--version`:

| Variable | Source | Purpose |
|---|---|---|
| PROFILE | cargo (build script env) | Re-emitted as `TENSOR_WASM_PROFILE`; records the compile profile. |
| TARGET | cargo (build script env) | Re-emitted as `TENSOR_WASM_TARGET`; records the target triple. |
| TENSOR_WASM_GIT_SHA / TENSOR_WASM_RUSTC_VERSION / TENSOR_WASM_PROFILE / TENSOR_WASM_TARGET | baked by `build.rs` | Compile-time `env!` constants for build-info labels — not read from the environment at runtime. |
