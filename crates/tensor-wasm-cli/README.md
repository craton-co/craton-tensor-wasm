# tensor-wasm-cli

Developer command-line interface for [Craton TensorWasm](https://github.com/craton-co/craton-tensor-wasm) — a GPU-accelerated serverless WebAssembly runtime. The crate ships a single binary, `tensor-wasm`, with subcommands for local execution, remote deployment, invocation, benchmarking, snapshotting, and Prometheus inspection.

## Install

From source against the workspace:

```bash
cargo install --path crates/tensor-wasm-cli
```

Or from a checkout:

```bash
cargo build --release -p tensor-wasm-cli
# binary at target/release/tensor-wasm
```

A binary distribution channel (Homebrew tap, signed release tarballs) is on the roadmap; `publish = false` in `Cargo.toml` keeps the crate off crates.io until that lands.

## Subcommands

| Command                | One-liner                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `tensor-wasm run`             | Execute a `.wasm` file locally against the in-process engine.                       |
| `tensor-wasm deploy`          | Upload a `.wasm` module (up to 64 MiB) to a remote TensorWasm server.                     |
| `tensor-wasm invoke`          | Call a deployed function by id; arguments forwarded as a JSON array.                |
| `tensor-wasm bench`           | Spawn/call/terminate `N` times locally and print P50/P95/P99/max latency.           |
| `tensor-wasm snapshot save`   | Capture a running instance to a `.tensor-wasm` archive (requires API; see notes).          |
| `tensor-wasm snapshot restore`| Restore an instance from a `.tensor-wasm` archive (requires API; see notes).               |
| `tensor-wasm kernel`          | Publish, list, or verify entries in the signed kernel registry (`publish`/`list` POST/GET `/kernels`; `verify` re-signs an on-disk manifest locally). |
| `tensor-wasm metrics`         | Fetch and print the Prometheus exposition page from a TensorWasm server.                  |
| `tensor-wasm observe`         | Live operator dashboard over `/healthz` + `/metrics` (refreshes in place).          |
| `tensor-wasm serve`           | Run the TensorWasm HTTP API gateway in-process; bind to `--addr` and serve until Ctrl-C.  |
| `tensor-wasm completions`     | Emit a shell-completion script for the named shell.                                 |
| `tensor-wasm man`             | Generate roff(7) man pages from the clap command tree.                              |

Use `tensor-wasm <subcommand> --help` for the authoritative flag list — that text is snapshot-tested with `insta` and reviewed on every change.

### Snapshot signing (`--hmac-key-file`, `--require-signature`)

Both `tensor-wasm snapshot save` and `tensor-wasm snapshot restore` accept `--hmac-key-file <PATH>` pointing at a 32-byte HMAC-SHA256 key. The file is read as 64 hex characters when it's exactly that length (surrounding whitespace stripped), otherwise as 32 raw bytes; mismatched-length files are rejected locally with exit code `2` before any network I/O. The hex-encoded key is forwarded to the server in the `X-TensorWasm-Snapshot-HMAC-Key` request header — actual sign/verify happens server-side via `tensor-wasm-snapshot`'s `with_hmac_sha256_key` builders. `snapshot restore` additionally accepts `--require-signature`, which sends `X-TensorWasm-Snapshot-Require-Signature: true` so the server refuses to rehydrate any archive that lacks an HMAC trailer. See the `tensor-wasm-snapshot` crate's `FORMAT.md` for the on-disk layout.

## Environment variables

| Variable      | Purpose                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------- |
| `TENSOR_WASM_TOKEN`  | When set, the CLI sends `Authorization: Bearer <token>` on every outbound request.           |
| `TENSOR_WASM_LOG`    | `tracing-subscriber` env-filter directive (e.g. `TENSOR_WASM_LOG=info,reqwest=warn`). Defaults to `warn`. |
| `RUST_LOG`    | Fallback consulted if `TENSOR_WASM_LOG` is unset.                                                   |

> **Security warning.** Setting `TENSOR_WASM_LOG` (or `RUST_LOG`) to `trace`
> — especially with `reqwest=trace` enabled — causes `reqwest` to log
> outbound request headers, **including the `Authorization: Bearer <token>`
> header**. Do not enable `trace`-level logging in production; restrict it to
> local debugging against a non-production token. The CLI does not currently
> install a tracing field-redaction layer.

## Global flags

* `--tenant <u64>` — when non-zero, the CLI attaches `X-TensorWasm-Tenant: <N>` to every outbound API request. Zero (the default) suppresses the header for backwards compatibility with older servers.

## Shell completions

```bash
# bash
tensor-wasm completions bash | sudo tee /etc/bash_completion.d/tensor-wasm

# zsh (ensure $fpath contains ~/.zsh/completions)
tensor-wasm completions zsh > ~/.zsh/completions/_tensor-wasm

# fish
tensor-wasm completions fish > ~/.config/fish/completions/tensor-wasm.fish

# PowerShell
tensor-wasm completions powershell | Out-String | Invoke-Expression

# elvish
tensor-wasm completions elvish > ~/.elvish/lib/tensor-wasm.elv
```

The full `clap_complete::Shell` matrix (bash, zsh, fish, powershell, elvish) is exercised by `cli_smoke.rs::completions_render_for_every_shell`.

## Exit codes

| Code | Meaning                                                                              |
| ---- | ------------------------------------------------------------------------------------ |
| `0`  | Success.                                                                             |
| `1`  | Generic runtime error (network failure, parse error, server 5xx, etc.).              |
| `2`  | Local-side validation failed on a snapshot subcommand (bad path, oversized file).    |
| `3`  | `tensor-wasm snapshot save/restore` received `404` from the server — feature not yet shipped. Track at <https://github.com/craton-co/craton-tensor-wasm/issues>. |

`tensor-wasm deploy` enforces a 64 MiB local cap on `--file` to match the server's request-body limit; oversized files exit `1` with a clear message before any HTTP I/O happens.

## Feature flags

The crate exposes no Cargo features; it compiles identically in every workspace configuration. See [`docs/BUILD.md`](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Further reading

* [`docs/CLI.md`](../../docs/CLI.md) — full operator guide (auth, multi-tenancy, observability, examples).
* [`docs/RISKS.md`](../../docs/RISKS.md) — security/abuse risk register; covers the 64 MiB cap, snapshot decompression bombs, and the bearer-auth threat model.

## License

Apache-2.0 © Craton Software Company. Security disclosures: <security@craton.com.ar>.
