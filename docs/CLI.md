# Craton TensorWasm CLI

The `tensor-wasm` binary is the developer-facing entry point to Craton TensorWasm. It wraps the same `tensor-wasm-exec` engine that powers the server (see [API.md](../crates/tensor-wasm-api/API.md)) so anything that runs against a deployed function can also be exercised locally without standing up infrastructure.

The CLI is built as part of the workspace — see [BUILD.md](./BUILD.md) for prerequisites and feature flags. After `cargo build -p tensor-wasm-cli` you will find the binary at `target/<profile>/tensor-wasm` (`tensor-wasm.exe` on Windows).

```
tensor-wasm --help
```

prints the top-level synopsis. Every subcommand also supports `--help` for its own flags.

## Global behaviour

- Logging is configured via `TENSOR_WASM_LOG` (which uses `tracing-subscriber`'s `EnvFilter` directive syntax). The default level is `info`; set `TENSOR_WASM_LOG=tensor_wasm_exec=debug` to drill into the executor or `TENSOR_WASM_LOG=warn` to quiet routine progress.
- Exit codes follow the Unix convention: `0` on success, non-zero on any user or runtime error. Errors print to stderr with a chained-cause summary courtesy of `anyhow`.
- Arguments and outputs that involve guest data use JSON. Use `--args '[1.0, 2.0]'`-style values; non-array JSON is rejected with a clear message.

## Subcommands

### `tensor-wasm run <file.wasm> [--export <name>] [--args <json>]`

Run a Wasm module locally against an in-process `TensorWasmEngine`.

- `<file.wasm>`: path to the module to execute. Must exist and be readable.
- `--export <name>`: function to invoke. Defaults to `main`.
- `--args <json>`: arguments to forward to the guest, encoded as a JSON array. Validated for shape only — the current executor invokes `() -> ()` exports, so values are ignored until S20 widens the call signature.

Example:

```bash
tensor-wasm run tests/wasm-fixtures/vector_add.wasm --export add --args '[1.0, 2.0]'
```

On success the command prints `ok`. On failure the chained-cause stack is written to stderr and the process exits non-zero. This subcommand exercises the same compile-and-spawn path that `tensor-wasm-api`'s `POST /functions/{id}/invoke` handler uses, so local runs are a faithful reproduction of server behaviour for the supported signatures.

### `tensor-wasm deploy <file.wasm> --server <url>`

Upload a Wasm module to a TensorWasm server.

- `<file.wasm>`: path to the artefact to deploy.
- `--server <url>`: base URL of the target server (e.g. `http://localhost:8080`). Must use `http://` or `https://` and have a non-empty host.

`tensor-wasm deploy` reads the Wasm bytes, base64-encodes them, and `POST`s them to `/functions` on the target server. On success the response includes the assigned function id, which is printed to stdout for piping into subsequent `tensor-wasm invoke` calls.

### `tensor-wasm invoke <id> --server <url> [--args <json>]`

Call a deployed function by id.

- `<id>`: the function identifier returned by an earlier `tensor-wasm deploy`.
- `--server <url>`: base URL of the target TensorWasm server.
- `--args <json>`: arguments forwarded to the function as a JSON array.

The subcommand issues a `POST /functions/{id}/invoke` against the server with the JSON body `{"export": "...", "args": [...]}` and prints the decoded response to stdout. Non-2xx responses surface as non-zero exits with the error body forwarded to stderr.

### `tensor-wasm bench <file.wasm> --export <name> [--n <iters>]`

Benchmark a Wasm export locally and print a P50 / P95 / P99 / max latency table. Each iteration spawns a fresh instance, invokes the export, and terminates — so the reported numbers are end-to-end (including cold start), not steady-state.

- `<file.wasm>`: path to the module.
- `--export <name>`: function to invoke per iteration. Defaults to `main`.
- `--n <iters>`: iteration count. Defaults to `100`. Must be at least 1.

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

### `tensor-wasm snapshot save <instance-id> <out.tensor-wasm>`<br />`tensor-wasm snapshot restore <in.tensor-wasm>`

Capture or restore an instance's state from a `.tensor-wasm` archive. `tensor-wasm snapshot save` writes the running instance's snapshot to the named output path via `tensor-wasm-snapshot::Writer`; `tensor-wasm snapshot restore` reads the archive, verifies the CRC32 integrity check, and rehydrates the instance through `tensor-wasm-snapshot::Reader`. Both commands surface validation errors (bad magic, version mismatch, CRC failure) as non-zero exits with a chained-cause message.

#### Signed snapshots: `--hmac-key-file` and `--require-signature`

Both `snapshot save` and `snapshot restore` accept `--hmac-key-file <PATH>` pointing at a 32-byte HMAC-SHA256 key. The file is interpreted as 64 hex characters when its trimmed length matches, otherwise as 32 raw bytes; any other length is rejected locally with exit code `2` (`LOCAL_VALIDATION_FAILED`) before the CLI dials the server. The hex-encoded key is forwarded as the `X-TensorWasm-Snapshot-HMAC-Key` request header — the actual `SnapshotWriter::with_hmac_sha256_key` / `SnapshotReader::with_hmac_sha256_key` plumbing lives server-side. `snapshot restore` additionally accepts `--require-signature`, which sends `X-TensorWasm-Snapshot-Require-Signature: true` so the server refuses to rehydrate any archive that does not carry an HMAC trailer (equivalent to calling `SnapshotReader::require_signature` on the server). See the `tensor-wasm-snapshot` crate's `FORMAT.md` for the on-disk layout of the signed v3 frame.

### `tensor-wasm metrics --server <url>`

Fetch and pretty-print the `/metrics` endpoint of a TensorWasm server. The output is the raw Prometheus text exposition; pipe to `grep '^tensor_wasm_'` to filter to TensorWasm's own series.

### `tensor-wasm observe [--addr <host:port>] [--interval <secs>]`

Live operator dashboard. Polls `GET /healthz` and `GET /metrics` against the target server on a fixed cadence and rewrites a single screen with the most actionable signals. Intended for on-call incident triage when neither a browser nor a Grafana session is available.

- `--addr <url>`: base URL of the target server. Defaults to `http://localhost:8080`. Must use `http://` or `https://` and have a non-empty host.
- `--interval <secs>`: refresh cadence, in seconds. Defaults to `2`. Must be at least `1`.

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
  `tensor-wasm-bench.1`, `tensor-wasm-snapshot.1`, `tensor-wasm-metrics.1`,
  `tensor-wasm-observe.1`, `tensor-wasm-completions.1`, `tensor-wasm-man.1` —
  one per top-level subcommand

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

The CLI surface — subcommand names, required positional arguments, and the long-form flags listed above — is considered stable for the v0.1 release window. Short-form flag aliases and machine-readable output formats (`--json`) are not yet stable.
