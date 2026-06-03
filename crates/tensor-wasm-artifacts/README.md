# tensor-wasm-artifacts

Unified content-addressed, HMAC-signed, on-disk artifact store primitive for Craton TensorWasm. Folds the JIT L2 disk cache (`tensor-wasm-jit::cache::DiskCache`) and the snapshot store (`tensor-wasm-snapshot::SnapshotWriter`/`SnapshotReader`) into a single signed-byte-blob abstraction so the two subsystems converge on one format, one audit surface, and one garbage-collection story. See [`docs/ARTIFACT-STORE.md`](../../docs/ARTIFACT-STORE.md) for the full design and the v0.4 migration plan.

## v0.3.7 status

Scaffold. The crate ships the [`ArtifactStore`] trait, an [`InMemoryArtifactStore`] for tests/fuzzers, and a fully-implemented [`DiskArtifactStore`]. The JIT L2 cache and the snapshot store **continue to use their own formats today** — the v0.4 follow-up migrates each to write through this primitive while remaining backward-compatible with v0.3.x readers via the legacy-magic fallback already baked into both crates.

## Format

```text
magic(16) || version(4) || content_hash(32) || zstd(payload) || hmac_tag(32)
```

- **magic** — `ARTIFACT_MAGIC` (`b"twasm-artifact01"`).
- **version** — little-endian `u32`; `ARTIFACT_VERSION = 1`.
- **content_hash** — BLAKE3 of the *uncompressed* payload. The `get` path recomputes this and rejects on mismatch.
- **zstd(payload)** — zstd-compressed body at `DEFAULT_ZSTD_LEVEL = 3` (matches the snapshot writer).
- **hmac_tag** — HMAC-SHA256 over `magic .. end-of-zstd`. Verified in constant time via `subtle::ConstantTimeEq`.

On disk a single artifact lives at `{content_hash_hex}.{key_fp_hex}.bin`, where `key_fp_hex` is the first 8 bytes of `blake3(hmac_key)`. Two stores in the same directory under different HMAC keys partition the namespace cleanly: a `get` against the wrong key returns `NotFound` (not `BadHmac`) because the verifier never opens the other store's file.

## Dependencies

- `blake3` — content addressing and key fingerprinting.
- `hmac` + `sha2` — HMAC-SHA256 signing (matches `tensor-wasm-snapshot`'s `signed-snapshots` trailer primitive).
- `subtle` — constant-time signature compare.
- `zstd` — body compression.
- `tempfile` — atomic temp-then-rename writes.
- `zeroize` — best-effort scrub of the HMAC key on drop.
- `parking_lot` — `Mutex` for the in-memory store.
- `tracing`, `thiserror`, `serde`, `serde_json` — supporting plumbing (`serde` + `serde_json` encode the `ArtifactMetadata` sidecar stored alongside each blob).

## Hardening

- **Magic + version** rejected before any keyed work (no HMAC oracle on foreign bytes).
- **HMAC-SHA256 constant-time compare** (no byte-by-byte timing leak).
- **Content-hash recomputation** as defence-in-depth — a header that disagrees with the decoded body is rejected even when the HMAC verifies.
- **Atomic writes** via `tempfile::NamedTempFile::persist` so a partial write never leaves a half-formed entry.
- **Key-fingerprinted filenames** so rotated keys partition the namespace and stale on-disk records become `NotFound` rather than HMAC failures.

## Roadmap

- v0.4: migrate `tensor-wasm-jit::cache::DiskCache` to write through `DiskArtifactStore`; migrate `tensor-wasm-snapshot` to wrap its bincode payload in this envelope rather than its bespoke v3 trailer.
- v0.4: shared garbage-collection / quota / `list`-based auditing CLI surface.
- v0.5: cross-tier key-rotation playbook (cohabits with the snapshot replay-protection matrix).

See [`docs/PATH-TO-V1.md` — Post-v0.3.6 strategic features, item #9](../../docs/PATH-TO-V1.md#post-v036-strategic-features) and [`docs/ARTIFACT-STORE.md`](../../docs/ARTIFACT-STORE.md).
