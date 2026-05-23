# bali-snapshot

Snapshot and restore subsystem for Project Bali, capturing combined Wasm linear memory and GPU device state to disk so cold starts are reduced to a memory-mapped load. Exposes `SnapshotWriter` to checkpoint a live instance and `SnapshotReader` to reconstitute it elsewhere, enabling fast scale-out and warm-start semantics for serverless workloads.

## Feature flags

This crate exposes no Cargo features; it compiles identically in every workspace configuration.

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root):
- `tokio` — async file I/O for streaming snapshots to disk.
- `thiserror` — derive macro for snapshot error variants.
- `tracing` — structured spans/events for checkpoint and restore.
- `serde` — derive support for the snapshot header format.
- `bincode` — compact binary encoding of the snapshot payload.
- `zstd` — compression for snapshot bodies on disk.

Internal crate dependencies are wired in by later sessions (this crate currently has none).
