# bali-cli

Developer command-line interface for Project Bali, shipping the `bali` binary with subcommands such as `bali run`, `bali deploy`, and `bali invoke`. In S1 the binary is a scaffold that simply prints a banner so the workspace builds cleanly; full subcommand wiring lands in S18.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async runtime backing CLI subcommands that hit the API.
- `clap` — argument parsing and subcommand dispatch.
- `clap_complete` — shell completion generation for the `bali` binary.
- `anyhow` — flexible error type for top-level CLI bailouts.
- `serde` — derive support for config files consumed by the CLI.
- `serde_json` — JSON output mode for machine-readable commands.
- `tracing` — structured spans/events for CLI runs.
- `tracing-subscriber` — subscriber/formatter wiring for CLI logging.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
