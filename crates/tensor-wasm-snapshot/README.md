# tensor-wasm-snapshot

Snapshot and restore subsystem for Craton TensorWasm, capturing combined Wasm linear memory and GPU device state to disk so cold starts are reduced to a memory-mapped load. Exposes `SnapshotWriter` to checkpoint a live instance and `SnapshotReader` to reconstitute it elsewhere, enabling fast scale-out and warm-start semantics for serverless workloads.

The on-disk wire layout (envelope, version history, size caps) is documented in [`FORMAT.md`](./FORMAT.md).

## Feature flags

| Feature | Default | Effect |
|---------|---------|--------|
| `cuda`  | off     | Enables the GPU-side restore path (`reader::restore_to_gpu`, `reader::restore_to_gpu_with`, and `reader::RestoredOnGpu`). Pulls in [`cust`](https://docs.rs/cust) and materialises the snapshot's `gpu_memory` blob directly into a `UnifiedBuffer<u8>` prefetched to the target device. `restore_to_gpu_with(&reader, ..)` lets callers supply a hardened `SnapshotReader` (cap / `require_signature` / HMAC key / max-age); `restore_to_gpu(..)` is a convenience wrapper that delegates through a default reader. Off by default so the crate builds on CUDA-less hosts. |
| `signed-snapshots` | **on** | Compiles in signature sign/verify for the wire-v3 trailer — **HMAC-SHA256** (`signature_kind = 1`, symmetric) and **Ed25519** (`signature_kind = 2`, asymmetric). Off does not break v2 reads — the reader still accepts v2 — but disables `SnapshotWriter::with_hmac_sha256_key`, `SnapshotWriter::with_ed25519_signing_key`, `SnapshotReader::with_ed25519_verifying_key`, and `SnapshotReader::require_signature`, so a v3 blob can no longer be produced or verified. Operators who genuinely do not want the codepath compiled in can `--no-default-features` it off; most should leave it on. |

See [docs/BUILD.md](../../docs/BUILD.md) for the project-wide flag taxonomy.

## Dependencies

External crates this crate depends on (pinned at workspace root unless noted):
- `tracing` — structured spans/events for checkpoint and restore.
- `serde` — derive support for the snapshot header format.
- `serde_bytes` — zero-copy byte-blob serialisation on the write path (crate-local pin).
- `bincode` (2.x, with the `serde` feature) — compact binary encoding of the snapshot payload, using `bincode::config::legacy()` for byte-identical wire compatibility with the 1.x default config. Restore-time allocation is bounded by a `Configuration::with_limit::<N>` const-generic ceiling sized to `limits::MAX_TOTAL_PAYLOAD_BYTES`.
- `zstd` — streaming compression for snapshot bodies on disk.
- `crc32fast` — payload checksum (IEEE polynomial).
- `hmac` + `sha2` + `subtle` + `zeroize` *(optional, behind `signed-snapshots`)* — HMAC-SHA256 v3 trailer signing/verification and constant-time signature comparison; `zeroize` scrubs the in-memory key on drop.
- `ed25519-dalek` *(optional, behind `signed-snapshots`)* — Ed25519 asymmetric v3 trailer signing/verification (publisher signs with a private key; verifiers hold only the public key).
- `cust` *(optional, behind `cuda`)* — CUDA unified-memory bindings for `restore_to_gpu`.

Internal crate dependencies: `tensor-wasm-core` (errors, tenant/instance ID types).

## Hardening

The reader is the hardened side of the API and treats every input as untrusted:

- **Compressed-input cap**: rejected before zstd runs if larger than `limits::MAX_INPUT_BYTES` (1 GiB).
- **Decompressed-stream cap**: streamed through `zstd::stream::read::Decoder` wrapped in `Read::take`, default ceiling `limits::MAX_DECOMPRESSED_BYTES` (256 MiB). Override per-reader via `SnapshotReader::with_max_decompressed`.
- **bincode allocation cap**: deserialised via `bincode::config::legacy().with_limit::<{ limits::MAX_TOTAL_PAYLOAD_BYTES }>()` (bincode 2.x compile-time const-generic limit), so a tampered `Vec<u8>` length field cannot drive the allocator past the static ceiling (sum of the per-blob caps + envelope slack). The per-blob caps below catch any oversized declared length that fits under the bincode ceiling.
- **Per-blob caps**: each memory blob has an explicit `limits::MAX_*_BYTES` ceiling (Wasm 1 GiB, GPU 4 GiB, registers 1 MiB).
- **CRC32**: payload-wide integrity check (over the three byte blobs in order) catches bit-flips that survive framing.
- **Magic + version**: refused before any further work, so foreign or stale blobs short-circuit.

## Threat model: authenticity vs integrity

The CRC32 in the v2 envelope is an **integrity** check: it catches storage bit-flips, accidental truncation, and most random framing damage. It is not an **authenticity** check — anyone who can write the snapshot bytes can write a matching CRC.

From v0.3.6 the crate ships an opt-in authenticity layer:

- **Wire v2** (the default writer output) is unchanged. Bytes on disk match exactly what every v0.1.0+ reader has always restored.
- **Wire v3** is `v2 + [trailer_magic: 4 bytes][signature_kind: u8][signature]` — a magic-prefixed trailer carrying either `HMAC-SHA256(key, v2_payload ++ trailer_magic ++ signature_kind)` (`signature_kind = 1`, 32-byte signature, 37-byte trailer) or an `Ed25519` signature over the same byte range (`signature_kind = 2`, 64-byte signature, 69-byte trailer). See [`FORMAT.md`](./FORMAT.md) for the byte-exact spec.
- **`SnapshotWriter::with_hmac_sha256_key(key)`** (symmetric) or **`SnapshotWriter::with_ed25519_signing_key(signing_key)`** (asymmetric) opts the writer into emitting v3. Without either, the writer emits v2 as before. Use Ed25519 when a single publisher signs and many verifiers — holding only the public key — must verify without being able to forge.
- **`SnapshotReader`** accepts both v2 and v3 by default; configure `with_hmac_sha256_key` and/or `with_ed25519_verifying_key` to verify the corresponding v3 trailers (each kind requires its matching key), while still allowing unsigned v2 through. **`SnapshotReader::require_signature()`** is the strict-mode switch — it rejects v2 entirely and is the end-state for any deployment whose snapshot store is reachable from a network the operator does not fully control.
- **Replay/rollback protection** — `SnapshotWriter::with_sequence_no(n)` / `with_nonce(bytes)` stamp the (now-live) `sequence_no` / `nonce` metadata fields; `SnapshotReader::with_min_sequence_no(floor)` / `with_expected_nonce(bytes)` enforce them. The intended pattern is a per-signing-key "track highest seen `sequence_no`" floor, which rejects replays of once-valid captures even inside the `with_max_age` freshness window. See [`FORMAT.md`](./FORMAT.md) § "Replay protection".

The opt-in design is deliberate: it preserves backward compatibility for existing v2 archives on disk and lets operators sequence the writer/reader/strict-mode rollout independently. The full migration playbook (provision key → configure reader → configure writer → flip to strict mode), the cross-tier ordering rule for key rotation, and the deployment-side env-var / CLI-flag surface are in [`docs/SNAPSHOT-COMPATIBILITY.md` — v2 → v3 migration](../../docs/SNAPSHOT-COMPATIBILITY.md#v2--v3-migration-signed-snapshots).

Operators handling snapshots received from untrusted peers (cross-tenant restore, snapshots fetched over the open internet, snapshots restored after a snapshot-store compromise) **should** configure a key and reach strict mode. Operators whose snapshot store is colocated with the runtime and protected by the same ACLs may stay on default-write v2 with no behavioural change.

## Platform

`tensor-wasm-snapshot` requires a 64-bit target (a `const _: () = assert!(usize::BITS >= 64, ...)` at the crate root enforces this at compile time). The size caps would silently truncate `usize` on 32-bit hosts.
