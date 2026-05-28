# Configuration Reference

> Every env var consumed by `tensor-wasm`. Generated from `crates/*/src/config.rs` + `crates/tensor-wasm-cli/src/cmd/serve.rs`.

## API gateway (`tensor-wasm-api`)
| Variable | Default | Purpose |
|---|---|---|
| TENSOR_WASM_API_TOKENS | (empty = dev-mode) | Comma-separated bearer tokens; see `docs/SECURITY.md` |
| TENSOR_WASM_API_TRUSTED_HOSTS | (empty = allow any) | Comma-separated allowed Host headers |
| TENSOR_WASM_API_TRUSTED_XFCC_PROXIES | (empty = never trust) | CIDR allowlist for peers whose XFCC header is honoured |
| TENSOR_WASM_API_CORS_ALLOWED_ORIGINS | (empty = no CORS) | Comma-separated origins for `Access-Control-Allow-Origin` |
| TENSOR_WASM_API_RATE_LIMIT_QPS | 0 = disabled | Per-token QPS cap |
| TENSOR_WASM_API_RATE_LIMIT_BURST | 0 | Per-token burst |
| TENSOR_WASM_API_AUDIT_LOG | (empty) | Path to JSONL audit sink |
| TENSOR_WASM_API_SNAPSHOT_HMAC_KEY | (empty) | 32-byte hex HMAC key for signed snapshots |
| TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE | false | Refuse v2 (unsigned) snapshots |

## Runtime (`tensor-wasm-exec`)
TODO: enumerate.

## Telemetry (`tensor-wasm-core`)
| Variable | Default | Purpose |
|---|---|---|
| TENSOR_WASM_LOG | info | `tracing_subscriber` EnvFilter |
| OTEL_EXPORTER_OTLP_ENDPOINT | (off) | OTLP gRPC endpoint |

## CLI (`tensor-wasm-cli`)
| Variable | Default | Purpose |
|---|---|---|
| TENSOR_WASM_TOKEN | (empty) | Bearer token used by `--server` calls; warns on plaintext to non-loopback |

TODO: complete by sweeping all `std::env::var(...)` sites in workspace.
