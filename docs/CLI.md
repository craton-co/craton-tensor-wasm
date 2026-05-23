# Bali CLI

The `bali` binary is the developer-facing entry point to Project Bali. It wraps the same `bali-exec` engine that powers the server (see [API.md](./API.md)) so anything that runs against a deployed function can also be exercised locally without standing up infrastructure.

The CLI is built as part of the workspace — see [BUILD.md](./BUILD.md) for prerequisites and feature flags. After `cargo build -p bali-cli` you will find the binary at `target/<profile>/bali` (`bali.exe` on Windows).

```
bali --help
```

prints the top-level synopsis. Every subcommand also supports `--help` for its own flags.

## Global behaviour

- Logging is opt-in via `RUST_LOG`. The default level is `warn`; set `RUST_LOG=info` for routine progress or `RUST_LOG=bali_exec=debug` to drill into the executor.
- Exit codes follow the Unix convention: `0` on success, non-zero on any user or runtime error. Errors print to stderr with a chained-cause summary courtesy of `anyhow`.
- Arguments and outputs that involve guest data use JSON. Use `--args '[1.0, 2.0]'`-style values; non-array JSON is rejected with a clear message.

## Subcommands

### `bali run <file.wasm> [--export <name>] [--args <json>]`

Run a Wasm module locally against an in-process `BaliEngine`.

- `<file.wasm>`: path to the module to execute. Must exist and be readable.
- `--export <name>`: function to invoke. Defaults to `main`.
- `--args <json>`: arguments to forward to the guest, encoded as a JSON array. Validated for shape only — the current executor invokes `() -> ()` exports, so values are ignored until S20 widens the call signature.

Example:

```bash
bali run tests/wasm-fixtures/vector_add.wasm --export add --args '[1.0, 2.0]'
```

On success the command prints `ok`. On failure the chained-cause stack is written to stderr and the process exits non-zero. This subcommand exercises the same compile-and-spawn path that `bali-api`'s `/v1/invoke` handler uses, so local runs are a faithful reproduction of server behaviour for the supported signatures.

### `bali deploy <file.wasm> --server <url>`

Upload a Wasm module to a Bali server.

- `<file.wasm>`: path to the artefact to deploy.
- `--server <url>`: base URL of the target server (e.g. `http://localhost:8080`). Must use `http://` or `https://` and have a non-empty host.

In S18 this is a stub that validates inputs and prints the planned action — the wire-level multipart upload arrives in S20 once `reqwest` lands as a workspace dependency. Existing scripts can wire `bali deploy` into their pipelines today and pick up the real upload behaviour without flag changes.

### `bali invoke <id> --server <url> [--args <json>]`

Call a deployed function by id.

- `<id>`: the function identifier returned by an earlier `bali deploy`.
- `--server <url>`: base URL of the target Bali server.
- `--args <json>`: arguments forwarded to the function as a JSON array.

S18 stub: validates the URL and JSON, then prints the planned `POST /v1/invoke/{id}`. Real transport ships in S20.

### `bali bench <file.wasm> --export <name> [--n <iters>]`

Benchmark a Wasm export locally and print a P50 / P95 / P99 / max latency table. Each iteration spawns a fresh instance, invokes the export, and terminates — so the reported numbers are end-to-end (including cold start), not steady-state.

- `<file.wasm>`: path to the module.
- `--export <name>`: function to invoke per iteration. Defaults to `main`.
- `--n <iters>`: iteration count. Defaults to `100`. Must be at least 1.

Example:

```bash
bali bench tests/wasm-fixtures/vector_add.wasm --export add --n 1000
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

Percentiles use the nearest-rank method on the sorted sample buffer. For a steady-state micro-benchmark (single instance, repeated calls), use the Criterion suite in `bali-bench` — see [BUILD.md](./BUILD.md).

### `bali snapshot save <instance-id> <out.bali>`<br />`bali snapshot restore <in.bali>`

Capture or restore an instance's state from a `.bali` archive. Both subcommands are stubs in S18 — they print `todo` and exit 0 so downstream tooling can wire them up against a stable surface. The real implementation backs onto `bali-snapshot::Writer` / `Reader` and lands in S20 alongside the HTTP transport.

### `bali metrics --server <url>`

Fetch and pretty-print the `/metrics` endpoint of a Bali server. S18 stub validates the URL and prints the planned request; S20 swaps in a real HTTP fetch and (optionally) filters output to `bali_*` series.

### `bali completions <shell>`

Emit a shell-completion script on stdout for the named shell. Supported values match `clap_complete::Shell`: `bash`, `zsh`, `fish`, `elvish`, `powershell`.

Wire-up examples:

```bash
# bash, system-wide
bali completions bash | sudo tee /etc/bash_completion.d/bali

# zsh, per-user
bali completions zsh > ~/.zsh/completions/_bali

# fish
bali completions fish > ~/.config/fish/completions/bali.fish

# PowerShell, current session
bali completions powershell | Out-String | Invoke-Expression
```

## Cross-references

- [BUILD.md](./BUILD.md) — workspace build matrix and feature flags (the CLI is part of `cargo build --workspace`).
- [API.md](./API.md) — REST surface that `bali deploy` / `invoke` / `metrics` target.
- [AUTO-OFFLOAD.md](./AUTO-OFFLOAD.md) — JIT path triggered by `bali run` and `bali bench` when a guest is auto-offload-eligible.
- [WASMTIME-FORK.md](./WASMTIME-FORK.md) — wasmtime patch set the CLI depends on transitively via `bali-exec`.

## Stability

The CLI surface — subcommand names, required positional arguments, and the long-form flags listed above — is considered stable for the S18 → S20 window. Stub subcommands will gain real implementations without renaming flags. Short-form flag aliases and machine-readable output formats (`--json`) are not yet stable.
