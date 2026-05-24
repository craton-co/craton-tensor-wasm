# tensor-wasm-snapshot

Snapshot and restore subsystem for Craton TensorWasm, capturing combined Wasm linear memory and GPU device state to disk so cold starts are reduced to a memory-mapped load. Exposes `SnapshotWriter` to checkpoint a live instance and `SnapshotReader` to reconstitute it elsewhere, enabling fast scale-out and warm-start semantics for serverless workloads.

The on-disk wire layout (envelope, version history, size caps) is documented in [`FORMAT.md`](./FORMAT.md).

## Feature flags

| Feature | Default | Effect |
|---------|---------|--------|
| `cuda`  | off     | Enables the GPU-side restore path (`reader::restore_to_gpu` and `reader::RestoredOnGpu`). Pulls in [`cust`](https://docs.rs/cust) and materialises the snapshot's `gpu_memory` blob directly into a `UnifiedBuffer<u8>` prefetched to the target device. Off by default so the crate builds on CUDA-less hosts. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root unless noted):
- `tracing` — structured spans/events for checkpoint and restore.
- `serde` — derive support for the snapshot header format.
- `serde_bytes` — zero-copy byte-blob serialisation on the write path (crate-local pin).
- `bincode` — compact binary encoding of the snapshot payload (with `Options::with_limit` to bound restore-time allocation).
- `zstd` — streaming compression for snapshot bodies on disk.
- `crc32fast` — payload checksum (IEEE polynomial).
- `cust` *(optional, behind `cuda`)* — CUDA unified-memory bindings for `restore_to_gpu`.

Internal crate dependencies: `tensor-wasm-core` (errors, tenant/instance ID types).

## Hardening

The reader is the hardened side of the API and treats every input as untrusted:

- **Compressed-input cap**: rejected before zstd runs if larger than `limits::MAX_INPUT_BYTES` (~4 GiB).
- **Decompressed-stream cap**: streamed through `zstd::stream::read::Decoder` wrapped in `Read::take`, default ceiling `limits::MAX_DECOMPRESSED_BYTES` (256 MiB). Override per-reader via `SnapshotReader::with_max_decompressed`.
- **bincode allocation cap**: deserialised via `bincode::Options::with_limit` matching the decompressed cap, so a tampered `Vec<u8>` length field cannot drive the allocator past the ceiling.
- **Per-blob caps**: each memory blob has an explicit `limits::MAX_*_BYTES` ceiling (Wasm 1 GiB, GPU 4 GiB, registers 1 MiB).
- **CRC32**: payload-wide integrity check (over the three byte blobs in order) catches bit-flips that survive framing.
- **Magic + version**: refused before any further work, so foreign or stale blobs short-circuit.

## Platform

`tensor-wasm-snapshot` requires a 64-bit target (a `const _: () = assert!(usize::BITS >= 64, ...)` at the crate root enforces this at compile time). The size caps would silently truncate `usize` on 32-bit hosts.
