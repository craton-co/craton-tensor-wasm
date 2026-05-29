# Craton TensorWasm CLI

The `tensor-wasm` binary is the developer-facing entry point to Craton TensorWasm. It wraps the same `tensor-wasm-exec` engine that powers the server (see [API.md](../crates/tensor-wasm-api/API.md)) so anything that runs against a deployed function can also be exercised locally without standing up infrastructure.

The CLI is built as part of the workspace — see [BUILD.md](./BUILD.md) for prerequisites and feature flags. After `cargo build -p tensor-wasm-cli` you will find the binary at `target/<profile>/tensor-wasm` (`tensor-wasm.exe` on Windows).

```
tensor-wasm --help
```

prints the top-level synopsis. Every subcommand also supports `--help` for its own flags.

## Global behaviour

- Logging is configured via `TENSOR_WASM_LOG` (which uses `tracing-subscriber`'s `EnvFilter` directive syntax); `RUST_LOG` is honoured as a fallback. The default level is `warn`; set `TENSOR_WASM_LOG=tensor_wasm_exec=debug` to drill into the executor or `TENSOR_WASM_LOG=info` to surface routine progress. Diagnostics are written to stderr so stdout stays clean for command output (critical for `--output json`). **Security warning:** setting the level to `trace` (notably `reqwest=trace`) causes `reqwest` to log outbound request headers, including the `Authorization: Bearer <token>` header set by `TENSOR_WASM_TOKEN`. Do not enable trace-level logging in production; the CLI does not currently install a tracing field-redaction layer.
- Exit codes follow the Unix convention: `0` on success, non-zero on any user or runtime error. Errors print to stderr with a chained-cause summary courtesy of `anyhow`. The `snapshot` and `kernel` subcommands additionally use `2` for local validation failures (bad path, malformed key, oversized archive) and `3` for "the API endpoint is not yet shipped" so CI can tell those apart from a generic failure (`1`).
- Arguments and outputs that involve guest data use JSON. Use `--args '[1.0, 2.0]'`-style values; non-array JSON is rejected with a clear message.

### Global flags

These flags are accepted on every subcommand (they are `global` clap flags):

- `--tenant <u64>`: tenant id advertised on outbound API requests via the `X-TensorWasm-Tenant` header. Defaults to `0`, which suppresses the header entirely for backwards compatibility.
- `--ca-cert <PATH>`: trust an additional PEM-encoded private CA root for outbound HTTPS. The certificate is *added alongside* the system trust store, not instead of it, so use this for an internal/self-signed CA rather than `--insecure`. The file must be PEM (not DER) and contain a `BEGIN CERTIFICATE` block, or the CLI fails fast with an actionable error.
- `--insecure`: disable TLS certificate verification entirely (`danger_accept_invalid_certs`). **Security hazard:** this exposes the connection to man-in-the-middle attacks and can leak the `TENSOR_WASM_TOKEN` bearer credential. A loud warning is logged on *every* invocation. Intended only for local dev against a throwaway cert; prefer `--ca-cert` everywhere else.

### Environment variables

- `TENSOR_WASM_TOKEN`: when set (and non-empty after trimming), sent as `Authorization: Bearer <token>` on every outbound request. If a token is configured and the target URL is a non-loopback `http://` endpoint, the CLI logs a one-shot warning before the token leaves the process (it does not refuse — operators may legitimately be on a trusted private network).
- `TENSOR_WASM_LOG` / `RUST_LOG`: `EnvFilter` directive for log verbosity (see above; default `warn`).
- `TENSOR_WASM_REQUIRE_KEY_PERMS`: when set to `1` on Unix, `snapshot save`/`snapshot restore`/`kernel publish`/`kernel verify` *refuse* (hard error, exit `2`) to use an HMAC key file that is group- or world-readable, instead of merely warning. Lets security-conscious deployments fail closed on a leaked-readable signing key.

## Subcommands

### `tensor-wasm run <file.wasm> [--export <name>] [--args <json>]`

Run a Wasm module locally against an in-process `TensorWasmEngine`.

- `<file.wasm>`: path to the module to execute. Must exist and be readable.
- `--export <name>`: function to invoke. Defaults to `main`.
- `--args <json>`: arguments to forward to the guest, encoded as a JSON array. Each element is converted to the closest-fitting wasm value type before being passed to `call_export_with_args`:

  | JSON literal | Wasm value type |
  |---|---|
  | integer in `i32` range (e.g. `1`, `-2147483648`) | `i32` |
  | integer outside `i32` range (e.g. `2147483648`) | `i64` |
  | non-integer numeric (e.g. `2.5`) | `f64` |
  | anything else (string, array, null) | rejected with a parse error |

  The export's declared signature must accept the resulting parameter types or wasmtime returns an error. `f32` cannot be selected from a JSON literal unambiguously; build a Wasm wrapper that demotes from `f64` if you need 32-bit floats from the CLI.

Examples:

```bash
# Legacy () -> () export — prints `ok`.
tensor-wasm run tests/wasm-fixtures/noop.wasm --export noop

# (i32, i32) -> i32 adder — prints `3`.
tensor-wasm run tests/wasm-fixtures/adder.wasm --export add --args '[1, 2]'

# (f64) -> f64 doubler — prints `3.0`.
tensor-wasm run tests/wasm-fixtures/doubler.wasm --export double --args '[1.5]'
```

On success the command prints the export's result list. An empty result list (the `() -> ()` case) collapses to the literal `ok` for stable scripting; a single-element result unwraps to the scalar (so a `-> i32` adder prints `3`, not `[3]`); multi-element results print as a JSON array. On failure the chained-cause stack is written to stderr and the process exits non-zero. This subcommand exercises the same compile-and-spawn path that `tensor-wasm-api`'s `POST /functions/{id}/invoke` handler uses, so local runs are a faithful reproduction of server behaviour.

### `tensor-wasm deploy <file.wasm> --server <url> [--name <name>] [--output <text|json>]`

Upload a Wasm module to a TensorWasm server.

- `<file.wasm>`: path to the artefact to deploy. Capped at 64 MiB to match the server's request-body limit.
- `--server <url>`: base URL of the target server (e.g. `http://localhost:8080`). Must use `http://` or `https://` and have a non-empty host.
- `--name <name>`: tenant-supplied display name. Defaults to the file stem when omitted.
- `--output <text|json>`: output format. `text` (default) prints the assigned function id bare; `json` prints a stable machine-readable envelope `{"id":"<...>"}` (compact, one line) for scripting / CI.

`tensor-wasm deploy` streams the Wasm bytes through a base64 encoder and `POST`s `{"name": ..., "wasm_b64": ...}` to `/functions` on the target server. On success the response carries the assigned function id, which is printed to stdout for piping into subsequent `tensor-wasm invoke` calls.

### `tensor-wasm invoke <id> --server <url> [--export <name>] [--args <json>] [--output <text|json>]`

Call a deployed function by id.

- `<id>`: the function identifier returned by an earlier `tensor-wasm deploy`. Validated locally against a strict identifier charset (`[A-Za-z0-9._-]`, rejecting the empty string and the traversal tokens `.` / `..`) before any request is made, so a fat-fingered id fails fast with a clear "invalid character" message rather than an opaque server round-trip.
- `--server <url>`: base URL of the target TensorWasm server.
- `--export <name>`: exported function to call on the deployed module. Defaults to `_start` (the WASI command convention); the server falls back to `main` when `_start` is absent.
- `--args <json>`: arguments forwarded to the function as a JSON array. The CLI validates that the value parses as a JSON array (non-array JSON is rejected) and embeds it as the `args` field; when omitted, `args` is an empty array.
- `--output <text|json>`: output format. `text` (default) pretty-prints the server response; `json` prints a stable machine-readable envelope `{"id":..., "export":..., "response":<parsed-or-raw>}` (compact, one line) for scripting / CI.

The subcommand issues a `POST /functions/{id}/invoke` against the server with the JSON body `{"export": "...", "args": [...]}` and prints the response to stdout. The wire envelope (including `args`) is fixed today, but **argument pass-through is not yet wired into the executor server-side** — the API handler deserialises the envelope strictly (to bound the DoS surface) but does not yet thread `args` into `call_export_with_args`. The shape is locked now so clients written against this CLI keep working once the server learns to consume `args`. Non-2xx responses surface as non-zero exits with the error envelope forwarded to stderr.

### `tensor-wasm bench <file.wasm> --export <name> [--n <iters>]`

Benchmark a Wasm export locally and print a P50 / P95 / P99 / max latency table. Each iteration spawns a fresh instance, invokes the export, and terminates — so the reported numbers are end-to-end (including cold start), not steady-state.

- `<file.wasm>`: path to the module.
- `--export <name>`: function to invoke per iteration. Defaults to `main`.
- `--n <iters>`: iteration count. Defaults to `100`. Must be at least 1.
- `--output <text|json>`: output format. `text` (default) prints the percentile table; `json` emits a machine-readable document of the same percentiles for CI perf gates.

Example:

```bash
tensor-wasm bench tests/wasm-fixtures/vector_add.wasm --export add --n 1000
```

Sample output:

```
bench: export=`add` iterations=1000
+-----------+--------------+
| percentile|       latency|
+-----------+--------------+
| P50       |     312.40 us |
| P95       |     589.10 us |
| P99       |       1.20 ms |
| max       |       4.83 ms |
+-----------+--------------+
```

Percentiles use the nearest-rank method on the sorted sample buffer. For a steady-state micro-benchmark (single instance, repeated calls), use the Criterion suite in `tensor-wasm-bench` — see [BUILD.md](./BUILD.md).

### `tensor-wasm snapshot save --instance <id> --output <out.tensor-wasm> --server <url>`<br />`tensor-wasm snapshot restore --input <in.tensor-wasm> --as-instance <id> --server <url>`

Capture or restore an instance's state from a `.tensor-wasm` archive **via the TensorWasm HTTP API** — both sub-actions are server-backed, not a local file format dance. Arguments are passed as named flags, not positionals.

`tensor-wasm snapshot save` `POST`s `/instances/{id}/snapshot` (the instance id is percent-encoded into the path) and streams the server-produced archive to `--output`. The download is written to a tempfile in the output's directory and atomically renamed on success, so a failed transfer never leaves a half-written snapshot at the target path. Flags:

- `--instance <id>`: identifier of the running instance to snapshot. Must be non-empty.
- `--output <path>`: where to write the resulting `.tensor-wasm` archive. The parent directory must exist.
- `--server <url>`: base URL of the target server.
- `--max-restore-bytes <bytes>`: cap on the number of bytes the CLI will accept from the server and write to disk. Defaults to 256 MiB; values above the default are clamped down so a malicious server cannot fill the operator's disk by streaming an unbounded response body.
- `--hmac-key-file <PATH>`: optional HMAC signing key (see below).

`tensor-wasm snapshot restore` streams the on-disk archive up to `POST /instances/restore` and prints the restored instance id from the server's `{"id": ...}` ack. Flags:

- `--input <path>`: path to the `.tensor-wasm` archive to upload. Must be a regular file.
- `--as-instance <id>`: identifier to assign to the restored instance. Validated against a strict identifier charset (`[A-Za-z0-9._-]`, rejecting empty and `.`/`..`) so it can be carried safely in the `X-TensorWasm-As-Instance` header.
- `--server <url>`: base URL of the target server.
- `--max-archive-bytes <bytes>`: cap on the *on-disk (compressed) archive* the CLI will upload. Defaults to 256 MiB; this bounds the compressed upload only — the decompressed footprint is enforced server-side and may be larger. The deprecated alias `--max-decompressed` is accepted for one release; prefer `--max-archive-bytes`.
- `--hmac-key-file <PATH>` / `--require-signature`: optional signature verification (see below).

Validation errors (missing/unwritable output, oversized archive, malformed key) are caught locally and exit `2` (`LOCAL_VALIDATION_FAILED`) before any network call. The `/instances/...` API routes are planned but not yet merged server-side; until they land, a route-miss `404` from the server is surfaced as exit `3` (`FEATURE_NOT_EXPOSED`) with a tracking-issue pointer rather than a silent success.

#### Signed snapshots: `--hmac-key-file` and `--require-signature`

Both `snapshot save` and `snapshot restore` accept `--hmac-key-file <PATH>` pointing at a 32-byte HMAC-SHA256 key. The file is interpreted as 64 hex characters when its trimmed length matches, otherwise as 32 raw bytes; any other length is rejected locally with exit code `2` (`LOCAL_VALIDATION_FAILED`) before the CLI dials the server. The hex-encoded key is forwarded as the `X-TensorWasm-Snapshot-HMAC-Key` request header — the server uses it to sign (on save) or verify (on restore) the archive. The CLI **refuses** to send the key over plaintext `http://` to a non-loopback host (exit `2`); use `https://` or a loopback target. `snapshot restore` additionally accepts `--require-signature`, which sends `X-TensorWasm-Snapshot-Require-Signature: true` so the server refuses to rehydrate any unsigned (v2) archive. On Unix, a group/world-readable key file logs a warning; set `TENSOR_WASM_REQUIRE_KEY_PERMS=1` to make that a hard error. See `docs/SNAPSHOT-FORMAT.md` for the on-disk layout of the signed frame.

### `tensor-wasm kernel publish|list|verify`

Publish, list, or verify entries in the signed kernel registry (roadmap feature #3). The server-side `/kernels` routes are gated behind the api crate's `kernel-registry-api` Cargo feature; a server built without it returns `503 kernel_registry_not_configured`, which the CLI surfaces as a normal error envelope. See `docs/KERNEL-REGISTRY.md` for the manifest schema and signing envelope.

`tensor-wasm kernel publish <name> <version> --ptx-file <PATH> --sm <SM> --key-file <PATH> [--publisher <id>] --server <url>` computes BLAKE3 over the PTX text, builds a `KernelManifest`, signs it with the HMAC-SHA256 key from `--key-file`, and `POST`s the bundle to `/kernels`. The server re-verifies the signature and digest before accepting.

- `<name>` / `<version>`: positional kernel name (e.g. `matmul.f32`) and SemVer-style version. Both must be non-empty.
- `--ptx-file <PATH>`: PTX text the manifest references. Capped at 16 MiB.
- `--sm <SM>`: compute capability the PTX targets (e.g. `80` for sm_80).
- `--key-file <PATH>`: 32-byte HMAC-SHA256 signing key, same on-disk format as `snapshot --hmac-key-file` (64 hex chars or 32 raw bytes).
- `--publisher <id>`: optional advisory publisher identifier baked into the manifest (not covered by the signature in v0.3.x). Defaults to `tensor-wasm-cli`.
- `--server <url>`: base URL of the target server.

`tensor-wasm kernel list --server <url>` GETs `/kernels` and renders the manifest table (`name@version  sm=<n>  publisher=<id>`). Empty registries print `(no kernels registered)`.

`tensor-wasm kernel verify <name@version> --manifest-file <PATH> --key-file <PATH>` is **local-only** — it reads a manifest JSON off disk, recomputes the HMAC under `--key-file`, and compares against the manifest's signature in constant time, also asserting the `name@version` selector matches. Useful for gating a build pipeline on a manifest verifying before it gets uploaded. No server is contacted.

### `tensor-wasm serve [--addr <host:port>] [--token <TOKEN>]... [--tenant-header-policy <optional|required>] [--cors-origin <ORIGIN>]... [--max-body-bytes <bytes>] [--allow-plaintext-public]`

Run the TensorWasm HTTP API gateway in-process: builds the axum router, binds it to `--addr`, and serves until Ctrl-C (also SIGTERM on Unix, for clean `docker stop` / pod termination). This is the entrypoint the README quickstart and `docs/GETTING-STARTED.md` point at. On a successful bind it prints `listening on http://<addr>`.

- `--addr <host:port>`: bind address. Defaults to `127.0.0.1:8080`. Production deployments should pass `0.0.0.0:<port>` (or a private subnet address) explicitly.
- `--token <TOKEN>`: bearer token the gateway accepts. Repeat to allowlist multiple tokens. If omitted entirely, the gateway falls back to reading `TENSOR_WASM_API_TOKENS` from the environment; empty/unset there means dev mode (auth disabled, with a startup warning).
- `--tenant-header-policy <optional|required>`: `X-TensorWasm-Tenant` header policy. `optional` (default) mirrors `TENSOR_WASM_API_REQUIRE_TENANT` unset; `required` mirrors `TENSOR_WASM_API_REQUIRE_TENANT=1` (requests without the header are rejected `400`).
- `--cors-origin <ORIGIN>`: origin to allow via CORS; repeatable. **Currently informational** — the api crate does not yet expose a programmatic CORS knob, so a non-empty value only logs a warning. Kept so the quickstart command parses cleanly.
- `--max-body-bytes <bytes>`: maximum inbound request body size. Defaults to 64 MiB to match the api crate's `MAX_REQUEST_BODY_BYTES`. **Currently informational** — passing a non-default value logs a warning and the gateway uses the compiled-in 64 MiB cap until the api crate grows a body-cap knob.
- `--allow-plaintext-public` (env: `TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC`): acknowledge knowingly exposing a dev-mode (no-auth) deployment on a non-loopback address. Required when `--addr` resolves to `0.0.0.0`, `::`, or any non-loopback IP **and** no token allowlist is configured; without it the CLI refuses to bind such a configuration. Setting it does NOT enable auth — it merely silences the safety gate, and a 60-second recurring warning keeps the misconfiguration visible in long-running logs.

The bind-safety gate runs before the wasmtime engine is initialised, so a misconfigured invocation fails fast with a single-line error naming both the address and the fix.

### `tensor-wasm metrics --server <url> [--output <text|json>]`

Fetch and print the `/metrics` endpoint of a TensorWasm server.

- `--server <url>`: base URL of the target server.
- `--output <text|json>`: output format. `text` (default) prints the raw Prometheus text exposition (pipe to `grep '^tensor_wasm_'` to filter to TensorWasm's own series); `json` emits a machine-readable document with the parsed samples for CI scripts.

Server-controlled label values are stripped of ASCII control bytes before printing so a malicious server cannot smuggle ANSI escapes into the operator's terminal.

### `tensor-wasm observe [--addr <url>] [--interval <secs>] [--output <text|json>]`

Live operator dashboard. Polls `GET /healthz` and `GET /metrics` against the target server on a fixed cadence and rewrites a single screen with the most actionable signals. Intended for on-call incident triage when neither a browser nor a Grafana session is available.

- `--addr <url>`: base URL of the target server. Defaults to `http://localhost:8080`. Must use `http://` or `https://` and have a non-empty host.
- `--interval <secs>`: refresh cadence, in seconds. Defaults to `2`. Must be at least `1`.
- `--output <text|json>`: output format. `text` (default) renders the in-place ANSI dashboard; `json` emits one machine-readable JSON document per tick as newline-delimited JSON (NDJSON) — in this mode the screen is not cleared between ticks, so the stream is appendable into a log pipeline.

Auth/tenant headers (`Authorization: Bearer ...`, `X-TensorWasm-Tenant`) are attached when configured, identical to every other HTTP-shaped subcommand. The refresh loop exits cleanly on Ctrl-C; per-tick fetch failures (network blips, server restart) are rendered into the board rather than aborting the loop.

Example:

```bash
TENSOR_WASM_TOKEN=devtoken tensor-wasm observe --addr https://tensor-wasm.example.com --interval 5
```

Sample output:

```
Craton TensorWasm — operator dashboard
target: http://localhost:8080   interval: 2s
--------------------------------------------------
liveness:   /healthz ok
uptime:     n/a
functions:  ?
jobs.active:?
instances:  3
gpu.memory: 1.00 GiB
--------------------------------------------------
endpoint                  req/s     p50      p95
/healthz                     4.50   n/a     n/a
/invoke                     10.00  10.0ms 275.0ms
--------------------------------------------------
Ctrl-C to exit.
```

Cells render as `?` (for counts) or `n/a` (for percentages, latencies) when the underlying Prometheus series is absent from the scrape — for example, `tensor_wasm_functions_total` and `tensor_wasm_jobs_active` are reserved series names not yet emitted by `tensor-wasm-core::metrics`, so they show `?` against current servers until they land. The dashboard never substitutes a misleading zero.

Prometheus parsing is done in-process with a small inline parser; no extra dependency is pulled in for the dashboard. The histogram percentiles use linear interpolation across the `_bucket` series, matching PromQL's `histogram_quantile()` for buckets that share a `path` label.

### `tensor-wasm completions <shell> [--out-dir <dir>]`

Emit a shell-completion script for the named shell. Supported values match `clap_complete::Shell`: `bash`, `zsh`, `fish`, `elvish`, `powershell`.

By default the script is written to stdout. Pass `--out-dir <dir>` to write it to a conventional filename inside `<dir>` instead — used to regenerate the committed scaffolding under [`crates/tensor-wasm-cli/completions/`](../crates/tensor-wasm-cli/completions/).

Wire-up examples:

```bash
# bash, system-wide
tensor-wasm completions bash | sudo tee /etc/bash_completion.d/tensor-wasm

# zsh, per-user
tensor-wasm completions zsh > ~/.zsh/completions/_tensor-wasm

# fish
tensor-wasm completions fish > ~/.config/fish/completions/tensor-wasm.fish

# PowerShell, current session
tensor-wasm completions powershell | Out-String | Invoke-Expression

# Regenerate the committed scaffolding
tensor-wasm completions bash --out-dir crates/tensor-wasm-cli/completions
tensor-wasm completions zsh  --out-dir crates/tensor-wasm-cli/completions
tensor-wasm completions fish --out-dir crates/tensor-wasm-cli/completions
```

### `tensor-wasm man [--out-dir <dir>]`

Generate roff(7) man pages for `tensor-wasm` and every subcommand, sourced from
the same clap definitions the help output uses. With no flags, the root page
is written to stdout. With `--out-dir <dir>`, the root page plus one
`tensor-wasm-<sub>.1` per subcommand is written under `<dir>` (this is how the
committed scaffolding under [`crates/tensor-wasm-cli/man/`](../crates/tensor-wasm-cli/man/)
is regenerated).

```bash
# Regenerate all committed man pages in one pass
tensor-wasm man --out-dir crates/tensor-wasm-cli/man

# Quick preview without committing
tensor-wasm man | man -l -
```

## Shell completions

Pre-generated completion scripts for bash, zsh, and fish live under
[`crates/tensor-wasm-cli/completions/`](../crates/tensor-wasm-cli/completions/).
That directory's [`README.md`](../crates/tensor-wasm-cli/completions/README.md)
covers per-OS install paths (system-wide vs per-user, Linux vs macOS, the
zsh `$fpath` story, and the fish `~/.config/fish/completions/` convention).

Short version:

| Shell | File                                                  | Install path (per-user)                          |
|-------|-------------------------------------------------------|--------------------------------------------------|
| bash  | `crates/tensor-wasm-cli/completions/tensor-wasm.bash` | `~/.local/share/bash-completion/completions/tensor-wasm` |
| zsh   | `crates/tensor-wasm-cli/completions/_tensor-wasm`     | `~/.zsh/completions/_tensor-wasm` (on `$fpath`)  |
| fish  | `crates/tensor-wasm-cli/completions/tensor-wasm.fish` | `~/.config/fish/completions/tensor-wasm.fish`    |

Regenerate after any clap-flag change with `tensor-wasm completions <shell>
--out-dir crates/tensor-wasm-cli/completions`.

## Man pages

Pre-generated `.1` man pages live under
[`crates/tensor-wasm-cli/man/`](../crates/tensor-wasm-cli/man/):

- `tensor-wasm.1` — root command + global flags
- `tensor-wasm-run.1`, `tensor-wasm-deploy.1`, `tensor-wasm-invoke.1`,
  `tensor-wasm-bench.1`, `tensor-wasm-snapshot.1`, `tensor-wasm-serve.1`,
  `tensor-wasm-metrics.1`, `tensor-wasm-observe.1`, `tensor-wasm-completions.1`,
  `tensor-wasm-man.1` — one per top-level subcommand

The `man --out-dir` generator walks the clap command tree, so it also emits a
`tensor-wasm-kernel.1` page for the `kernel` subcommand; that page is not yet
committed under `crates/tensor-wasm-cli/man/`, so re-running the regeneration
command below will add it on the next pass.

That directory's [`README.md`](../crates/tensor-wasm-cli/man/README.md) covers
per-OS install paths (Linux man-db, macOS, WSL on Windows) and the
`mandb` reindex step.

Regenerate after any clap-flag change with `tensor-wasm man --out-dir
crates/tensor-wasm-cli/man`.

## Cross-references

- [BUILD.md](./BUILD.md) — workspace build matrix and feature flags (the CLI is part of `cargo build --workspace`).
- [API.md](../crates/tensor-wasm-api/API.md) — REST surface that `tensor-wasm deploy` / `invoke` / `metrics` target.
- [AUTO-OFFLOAD.md](./AUTO-OFFLOAD.md) — JIT path triggered by `tensor-wasm run` and `tensor-wasm bench` when a guest is auto-offload-eligible.

## Stability

The CLI surface — subcommand names, required positional arguments, and the long-form flags listed above — is considered stable for the v0.1 release window. Short-form flag aliases and the machine-readable output format (the `--output json` envelopes) are not yet stable.
