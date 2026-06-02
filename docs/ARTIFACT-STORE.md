# Unified Artifact Store

> Reference doc for `tensor-wasm-artifacts` — the v0.3.7 scaffold that
> folds the JIT L2 disk cache and the snapshot store into one
> content-addressed, HMAC-signed primitive. Tracked as roadmap feature
> #9 in [`PATH-TO-V1.md` § Post-v0.3.6 strategic features](PATH-TO-V1.md#post-v036-strategic-features).

## Motivation

TensorWasm has two on-disk subsystems that independently solve the same
problem — "give me a signed, integrity-checked, restartable byte blob
keyed by some hash" — with two different formats, two HMAC trailers,
two file-layout conventions, and two parallel audit surfaces:

| Concern | JIT L2 disk cache (`tensor-wasm-jit::cache::DiskCache`) | Snapshot store (`tensor-wasm-snapshot::SnapshotWriter`) |
|---|---|---|
| Magic | `b"TWJIT-KRNL-v2\0\0\0"` (16 bytes) | `0xBA11_5407` (4 bytes inside bincode) |
| Compression | None (PTX text only) | `zstd` (level 3 by default) |
| MAC | `blake3::keyed_hash` (32 bytes) | `HMAC-SHA256` v3 trailer (33 bytes incl. kind) |
| Key fingerprint in filename | yes (first 8 bytes of `blake3(hmac_key)`) | no (caller routes blobs externally) |
| Atomic write | yes (`tempfile::persist`) | up to the caller |
| Content addressing | no (keyed by `(tenant, blueprint, sm)`) | no (keyed by caller-chosen path) |

The two formats arrived independently — JIT first, snapshot second —
and converged in spirit but not in code. Every audit-pass against
either subsystem has to re-verify a near-identical set of properties
(magic, version, MAC, length bounds). Every operator runbook explains
the same disk-layout invariants twice. Every garbage-collection or
quota story has to be implemented twice.

Feature #9 folds the underlying machinery. The product layer keeps two
APIs (`KernelCache`, `SnapshotReader`/`Writer`) — those carry domain
semantics — but each lowers onto a shared `ArtifactStore` for the
disk-touching part.

## Format

A single artifact on disk looks like:

```text
magic(16) || version(4) || content_hash(32) || zstd(payload) || hmac_tag(32)
```

- `magic` — `b"twasm-artifact01"`. Deliberately distinct from both
  `TWJIT-KRNL-v2` and `0xBA11_5407` so an operator who mixes formats in
  the same directory gets cheap rejection on the wrong reader, not a
  silent HMAC failure.
- `version` — little-endian `u32`. `1` for this scaffold; bumped on any
  non-additive change.
- `content_hash` — BLAKE3 of the **uncompressed** payload. Recomputed
  by the verifier and cross-checked against this header value as
  defence-in-depth.
- `zstd(payload)` — streaming zstd at `DEFAULT_ZSTD_LEVEL = 3` (matches
  `tensor-wasm-snapshot::DEFAULT_ZSTD_LEVEL`).
- `hmac_tag` — HMAC-SHA256 over `magic .. end-of-zstd`. Verified in
  constant time. SHA256 (not BLAKE3-keyed) chosen so this crate aligns
  with the snapshot v3 trailer rather than the JIT cache's BLAKE3-keyed
  MAC — the migration plan converges JIT *onto* the SHA256 primitive.

Filenames are `{content_hash_hex}.{key_fp_hex}.bin` where `key_fp_hex`
is the first 8 bytes of `blake3(hmac_key)` rendered as hex. Two stores
in the same directory under different keys produce two distinct files
for an identical payload, and each store's `list()` only surfaces its
own entries.

## Threat model

The store defends against:

- **Storage tampering.** Any byte flipped inside the signed prefix
  invalidates the HMAC; verification is constant-time.
- **Key rotation / stale keys.** Old entries under a rotated key
  appear as `NotFound` to the new store (filename partition), not as
  silent HMAC failures.
- **Replay across stores.** A blob produced by store A cannot be
  injected into store B's namespace under a chosen content hash — the
  filename's key fingerprint encodes which key signed it.
- **Foreign / stale formats.** Magic + version are checked before any
  keyed work, so a v2 disk-cache file or a snapshot v3 blob accidentally
  dropped into the same directory short-circuits with `BadMagic` /
  `BadVersion` instead of failing the HMAC oracle.

It does **not** defend against:

- A malicious party with the HMAC key. The store is an integrity +
  authenticity layer over storage, not a confidentiality layer.
- Replay of the same `(hash, key)` pair across two TensorWasm
  deployments using the same key. Operators rotate per-deployment.

## Convergence plan — v0.4

The v0.3.7 scaffold ships the trait, an in-memory impl, and a disk
impl. The JIT L2 cache and the snapshot store continue to use their
own formats today. The v0.4 follow-up does the actual fold:

1. **`tensor-wasm-jit::cache::DiskCache` migrates** to write through
   `DiskArtifactStore`. The `TWJIT-KRNL-v2` magic stays readable
   (legacy-magic info-log path) so existing on-disk entries survive
   the upgrade and are rewritten under the unified envelope on next
   put. `CacheKey` becomes the content-derived payload header — the
   blueprint fingerprint, sm version, and `launch_geometry` are
   serialised inside the body so the outer envelope's `content_hash`
   keys remain stable across emitter-config knobs that should not
   affect identity.
2. **`tensor-wasm-snapshot` migrates** to wrap its bincode-encoded
   `Snapshot` in `DiskArtifactStore::put` / `get` rather than the
   bespoke v3 trailer. The on-disk magic changes; the v2 / v3 reader
   stays in the crate as a legacy decoder so existing operator
   snapshots can be migrated in place via a one-shot tool.
   - **Snapshot migration: LANDED in v0.3.7 (T40).**
     `SnapshotWriter::capture` now routes through the unified
     envelope (via `tensor_wasm_artifacts::encode_envelope_to_vec`)
     when the writer has an HMAC key configured, and
     `SnapshotReader::restore` auto-detects the envelope by its
     leading 16-byte magic and falls through to the legacy v3 / v2
     decoders otherwise. The `artifact-backing` feature is on by
     default; operators who depend on the legacy v3 wire format
     for writes can use `SnapshotWriter::with_legacy_envelope()` /
     `SnapshotWriter::capture_legacy()` or build the crate with
     `--no-default-features --features signed-snapshots`. See
     [`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md#v4-artifact-store-backed-default-in-v04--t40)
     for the detection ordering and opt-out semantics. The
     persistent-store methods (`capture_to_artifact_store` /
     `restore_from_artifact_store`) remain available under the same
     feature for callers that want content-addressed disk storage
     rather than just the framed envelope bytes.
3. **CLI surface.** A new `tensor-wasm artifact list|get|verify`
   command goes against the unified directory layout, so operators
   have one inspection command instead of two.
4. **GC + quota.** Once both consumers route through `ArtifactStore`,
   a single content-hash → metadata map (added in this same v0.4
   wave) powers shared garbage collection and per-tenant disk-quota
   accounting.

The migration is sequenced so each consumer can adopt independently —
the trait is fully usable today, and the v0.4 work is two PRs that
land sequentially without blocking each other.

## See also

- [`crates/tensor-wasm-artifacts/README.md`](../crates/tensor-wasm-artifacts/README.md) — the crate-level rundown.
- [`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) — the legacy snapshot v2/v3 wire format (preserved for backward compatibility).
- [`crates/tensor-wasm-jit/src/cache.rs`](../crates/tensor-wasm-jit/src/cache.rs) — the legacy JIT L2 disk-cache implementation.
- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md#post-v036-strategic-features) — roadmap feature #9 entry that this doc fulfils.
