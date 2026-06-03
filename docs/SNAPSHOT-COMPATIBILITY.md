<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Snapshot cross-version compatibility

This document records the compatibility promise Craton TensorWasm makes
about the on-disk snapshot format produced by
[`tensor-wasm-snapshot`](../crates/tensor-wasm-snapshot/), the version-to-behavior
matrix, and the procedure maintainers follow when bumping the format
version. It accompanies the wire-format spec in
[`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md);
that file describes what the bytes are, this file describes which versions
are required to read which bytes.

This document is the artifact behind the v0.5 exit criterion in
[`PATH-TO-V1.md`](./PATH-TO-V1.md): *"Cross-version snapshot compatibility
tested. Snapshots from v0.2, v0.3, v0.4 all restore cleanly under v0.5."*

## The promise

**v1.0 will read every snapshot produced by v0.5+.**

Concretely, once the v0.5 line freezes the snapshot format, every later
release on the v1.x line must accept a snapshot blob byte-for-byte
identical to one produced by any v0.5+ writer. The reverse direction is
explicitly **not** promised: an older reader may refuse a snapshot from a
newer writer (the version field is a hard match — see
[`FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md#version-history)).

In v0.x (pre-freeze), the promise is weaker: each minor bump may bump
`SNAPSHOT_VERSION` and refuse older blobs, with the upgrade path being
"re-capture from the live instance". The compat test suite under
[`crates/tensor-wasm-snapshot/tests/compat.rs`](../crates/tensor-wasm-snapshot/tests/compat.rs)
already exercises the v0.5 → v1.0 path: each released format version
contributes a golden fixture and the current reader must restore all of
them.

### Why a CRC32, not a signature

The on-wire `crc32` field is an integrity check, not a security primitive.
It catches storage bit-flips and accidental truncation; it does not
authenticate the source. From v0.3.6 onward an **optional** authenticator
gives operators a way to authenticate snapshot bytes; the CRC32 stays in
place as the cheap-and-always-on integrity check. Two authenticated wire
formats exist (see the matrix below and the
[v2 → v3 migration](#v2--v3-migration-signed-snapshots) section):

- **wire v3** — a magic-prefixed trailer appended after the zstd frame,
  carrying either an HMAC-SHA256 MAC (`signature_kind = 1`) or an
  Ed25519 signature (`signature_kind = 2`).
- **wire v4** — the unified content-addressed
  [`tensor-wasm-artifacts`](../crates/tensor-wasm-artifacts/) envelope
  (16-byte `b"twasm-artifact01"` magic prefix, BLAKE3 content hash,
  mandatory HMAC-SHA256). This is what an HMAC-keyed writer now emits by
  default — see the matrix and the default-write note below.

Snapshots restored from untrusted peers should either be paired with a
transport-layer signature (mTLS, signed manifest) or — preferred for
v0.3.6+ — carry a v3 trailer or a v4 envelope authenticated with a key
the reader holds.

## Format-version → behavior matrix

The matrix tracks **wire formats**, not just the inner `SNAPSHOT_VERSION`
field. v3 and v4 are distinguished by their framing (a magic-prefixed
trailer / a leading-magic envelope) rather than by a bumped inner version
byte: a v3 blob carries inner `version = 3`, and a v4 artifact envelope
wraps an inner `version = 2` payload. The reader dispatches purely by
magic — see [`FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) for the
exact byte layouts; this table is the cross-version compatibility view.

| Wire format | First release | Framing changes | Reader behavior |
|-------------|---------------|-----------------|-----------------|
| `1` | (pre-v0.1.0 development only) | Initial layout. No CRC, no enforced size caps. | Refused by every shipped reader; never reached `main` as a released format. |
| `2` | v0.1.0 preview | Added `Snapshot::crc32`; enforced per-blob size caps (`limits::MAX_*_BYTES`); zero-copy write path via `serde_bytes` (wire-identical to plain `Vec<u8>` bincode). Inner `version = 2`, bare `zstd(bincode(Snapshot))`. | Accepted by every shipped reader. Emitted by the default writer **only when no signing key is configured** (keyless fallback). |
| `3` *(opt-in, v0.3.6+)* | v0.3.6 | Appends a **37-byte** magic-prefixed trailer after the zstd frame: `[V3_TRAILER_MAGIC = b"S3T1": 4][signature_kind: u8: 1][signature: 32]`. `signature_kind = 1` ⇒ HMAC-SHA256; `signature_kind = 2` ⇒ Ed25519 (then the signature is 64 bytes, total trailer 69 bytes — `ED25519_TRAILER_LEN`). Inner `version = 3`; the bincode payload is otherwise byte-identical to v2. **(T8, BREAKING):** the trailer was 33 bytes pre-T8 (`[signature_kind][signature]`, no magic prefix) and used a ~1/256 single-byte sniff; the `S3T1` prefix replaces it with a ~1/2³² magic check. Pre-T8 v3 captures no longer parse and must be re-signed. | A reader with the `signed-snapshots` feature accepts `2` and `3`. HMAC v3 needs `SnapshotReader::with_hmac_sha256_key(key)`; Ed25519 v3 needs `with_ed25519_verifying_key(key)`. `require_signature()` rejects unsigned `2`/`4`-without-key. Emitted when a key is configured **and** the legacy envelope is forced (or `artifact-backing` is compiled out); an Ed25519 key always routes through this inline path. |
| `4` *(default-write when HMAC-keyed, v0.4 / T40)* | v0.4 (T40) | The unified `tensor-wasm-artifacts` envelope: `b"twasm-artifact01"(16) ‖ version(=1) ‖ blake3(payload)(32) ‖ zstd(bincode(Snapshot)) ‖ hmac_sha256(prefix)(32)`. The **inner** `Snapshot::version` is `2` (the outer envelope owns authentication, so no inner v3 trailer is written). Detected by the 16-byte leading magic, not by the inner version field. | Auto-detected by the leading magic before the legacy v3/v2 fall-through (requires `artifact-backing`). A tampered/wrong-key v4 blob is rejected here and is **not** retried as a legacy blob. Inner `version = 3` is rejected on this path; only `2` is accepted inside the envelope. |

Every entry adds a *new* format; existing rows are never removed. A reader
from release N must support every released format up to N once v0.5 freezes
the wire format. Until then, the table also lists "pre-freeze" rows where a
format is intentionally accepted by only one reader.

### Default-write behavior (corrected)

`artifact-backing` is a **default** cargo feature of `tensor-wasm-snapshot`
(`default = ["signed-snapshots", "artifact-backing"]`). As a result, the
default-write format depends on the writer's key configuration:

- **HMAC-keyed writer** (`SnapshotWriter::with_hmac_sha256_key(key)`,
  no `with_legacy_envelope()`): `SnapshotWriter::capture` routes through
  the **v4 artifact envelope** by default (T40 cutover).
- **Ed25519-keyed writer** (`with_ed25519_signing_key`): always emits the
  **inline v3** envelope (`signature_kind = 2`) — the artifact envelope
  authenticates with HMAC only, so an asymmetric signature has no slot
  there.
- **Keyless writer** (`SnapshotWriter::new()` with no signing key): falls
  back to the unsigned **v2** envelope — the v4 envelope mandates an HMAC
  key by construction. This keeps every in-tree keyless caller (proptest,
  mem-conformance, bench) emitting v2 unchanged.
- **Opt-out to legacy**: `SnapshotWriter::with_legacy_envelope()` or the
  per-call `SnapshotWriter::capture_legacy(state)` keep emitting the
  inline v2/v3 wire format even when `artifact-backing` is enabled.
  Building `--no-default-features --features signed-snapshots` disables v4
  emission at compile time.

This supersedes the previous doc's claim that "v2 is the default-write
through 0.3.6 / the default writer still emits v2" — an HMAC-keyed writer
now defaults to v4. The workspace version is `0.3.7`
([`Cargo.toml`](../Cargo.toml)); `FORMAT.md` labels the v4 cutover "default
in v0.4 — T40" and it is live today because `artifact-backing` ships in the
default feature set.

## Compat test architecture

The compatibility guarantee is enforced by the test suite in
[`crates/tensor-wasm-snapshot/tests/compat.rs`](../crates/tensor-wasm-snapshot/tests/compat.rs).
That file loads **golden fixtures** — checked-in binary snapshot blobs
produced by a frozen-snapshot generator at a fixed timestamp — and asserts:

1. The current `SnapshotReader::restore` returns `Ok(Snapshot)` for each
   golden fixture, with the expected `tenant_id`, `instance_id`, and per-blob
   lengths.
2. The raw on-wire magic + version bytes (first eight bytes of the
   decompressed payload) equal the constants in
   [`writer.rs`](../crates/tensor-wasm-snapshot/src/writer.rs)
   (`SNAPSHOT_MAGIC`, `SNAPSHOT_VERSION`). This guards against a silent
   format break: if someone bumps a constant without bumping the fixture,
   this test fails.
3. Bumping the on-wire version byte by one and re-compressing produces a
   blob the current reader **refuses** with a `version`-mentioning error.
   This guards against accidental weakening of the version check.

The fixtures themselves are produced by
[`examples/generate_golden.rs`](../crates/tensor-wasm-snapshot/examples/generate_golden.rs)
and live under
[`crates/tensor-wasm-snapshot/tests/fixtures/`](../crates/tensor-wasm-snapshot/tests/fixtures/).
They are checked in as opaque bytes — they are *not* regenerated on every
test run, because the entire point is that they encode a *frozen*
historical wire format.

### Generating the fixtures

From the repo root, after a clean checkout (or after a deliberate format
bump):

```sh
cargo run -p tensor-wasm-snapshot --example generate_golden -- \
    crates/tensor-wasm-snapshot/tests/fixtures
```

The generator writes:

- `tests/fixtures/golden_v0_1_0_minimal.snap` — empty bodies, fixed tenant
  `TenantId(0xA)`, instance `InstanceId(0xB)`.
- `tests/fixtures/golden_v0_1_0_with_wasm_memory.snap` — 4 KiB wasm memory
  (`i % 251` pattern), 1 KiB GPU memory (`(i*17) % 253` pattern), 256 B
  registers (`i ^ 0x5A`), tenant `TenantId(0xC0FFEE)`, instance
  `InstanceId(0xDEAD_BEEF_CAFE_F00D)`.

Both fixtures embed a **fixed** `metadata.created_unix_ms = 1_767_225_600_000`
(2026-01-01T00:00:00Z) so the bytes are stable across machines and runs.
This is the one place the generator diverges from
`SnapshotWriter::capture`: capture stamps the real wall clock, which would
make the golden bytes change every run. The generator hand-builds the
`Snapshot` struct so the timestamp is deterministic; every other framing
detail (bincode 1.x default config, zstd level 3) is identical to what the
production writer produces.

After running the generator, commit the two `.snap` files. Then un-ignore
the tests in `tests/compat.rs` (they are gated on `#[ignore = "golden
fixture not yet generated; run examples/generate_golden.rs first"]` so a
fresh checkout passes `cargo test` even before the fixtures land).

### Adding a new golden fixture when bumping the format version

When a future release bumps `SNAPSHOT_VERSION` (say from `2` → `3`) the
following changes happen in lockstep:

1. **Do not modify the existing fixtures.** `golden_v0_1_0_*.snap` must
   continue to decode under every future reader — that is the compat
   promise itself.
2. Add a new branch in `examples/generate_golden.rs` (or a sibling
   `examples/generate_golden_v0_2.rs`) that emits two new files:
   - `tests/fixtures/golden_v0_2_0_minimal.snap`
   - `tests/fixtures/golden_v0_2_0_with_wasm_memory.snap`
3. Add a new row to the [matrix](#format-version--behavior-matrix) above
   describing the wire change and which readers accept it.
4. Add new tests to `compat.rs` mirroring the existing `*_golden_restores`
   pair but loading the v0.2 fixtures. The v0.1.0 tests **must remain** —
   the v0.2 reader is required to read them.
5. Update `crates/tensor-wasm-snapshot/FORMAT.md` *Version history* table with
   the new version, the bytes it adds, and which readers accept it.
6. If the format change is *additive* (e.g. a new optional field that
   bincode treats as a trailing payload), the v0.2 reader can keep reading
   v0.1.0 blobs natively. If the format change is *structural* (e.g. a
   field reordering), the v0.2 reader needs an explicit version-2
   compatibility path; document the strategy in the FORMAT.md *Version
   history* row.

The combination of "old fixtures stay" + "new tests load old fixtures
under the new reader" is what makes the compat promise machine-checkable.
A future PR that breaks compatibility breaks the test suite.

## v2 → v3 migration (signed snapshots)

v0.3.6 introduces optional HMAC-SHA256 authentication for snapshots via
a new wire format, v3. A v3 blob is a v2-shaped zstd frame followed by a
**37-byte** magic-prefixed trailer (`[b"S3T1": 4][signature_kind: 1][sig:
32]`; see [`FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) for the
exact layout and the T8 history of the pre-T8 33-byte trailer). Because
the trailer sits *after* the zstd frame and the v0.3.6+ reader accepts
both v2 and v3, an operator can adopt signing **without** invalidating
existing v2 archives on the read path. The
recommended four-step rollout below sequences the cross-tier changes
so a v2 archive is never orphaned and so the strict-mode flip
(`require_signature = true`) only happens once every archive in scope
has been re-captured as v3.

### Step 1 — provision a 32-byte key

The HMAC key is opaque 32-byte material. Generate it once per
environment using one of:

```sh
# either of these is fine — both produce 32 bytes of CSPRNG output.
head -c 32 /dev/urandom | base64
openssl rand -hex 32
```

Distribute via the operator's existing secret-management plane
(Kubernetes Secret mounted as a file, HashiCorp Vault, AWS Secrets
Manager, environment variable injected by the orchestrator). The key
file format expected by `--hmac-key-file` is hex-encoded; the
`TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` env var expects hex as well.
**Treat the key with the same operational care as a TLS private key**:
losing it means losing the ability to verify any v3 archive signed
with it; leaking it means an attacker can forge v3 archives the reader
will accept.

### Step 2 — configure the readers (still accepts v2)

Roll the reader configuration first, before any writer starts emitting
v3. On the API tier, set `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` (hex) on
every gateway instance. On CLI-driven restore paths, pass
`--hmac-key-file PATH` to `tensor-wasm snapshot restore`. At this
point readers can verify v3 archives but still accept v2 (no
`require_signature`); existing archives keep restoring exactly as
before.

Verify Step 2 landed cleanly by restoring a known-good v2 archive
through the new reader configuration. The restore must succeed
unchanged; if it does not, the key plumbing has broken the unsigned
path and Steps 3–4 must be deferred until that is fixed.

### Step 3 — configure the writers (start emitting v3)

Once every reader in the deployment has the key, flip the writers.
- **Library callers**: pass `SnapshotWriter::with_hmac_sha256_key(key)`
  when constructing the writer.
- **CLI**: pass `--hmac-key-file PATH` to `tensor-wasm snapshot save`.
- **API tier**: the same `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` env var
  picked up in Step 2 is consumed by the writer half when the snapshot
  routes are exposed; until then the var is read into `AppConfig` and
  surfaces in the writer when the routes ship.

From this point forward, every newly captured snapshot is wire v3.
Existing v2 archives on disk are unchanged and still restore.

### Step 4 — flip the reader to strict mode (refuse v2)

When the operator has confirmed that every snapshot still in active
use has been re-captured as v3 — typically by waiting one full
snapshot-rotation cycle, or by listing the snapshot store and grepping
for the v2 magic — set the reader to refuse v2:

- **API tier**: set `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE=true`.
- **Library callers**: call `SnapshotReader::require_signature()` on
  the builder.
- **CLI restore**: pass `--require-signature` to
  `tensor-wasm snapshot restore`.

After this step, a v2 archive (or a v3 archive whose HMAC does not
verify under the configured key) is rejected at the reader; only
authenticated v3 restores succeed.

The four steps are deliberately ordered "reader-key first, writer-key
second, strict-mode last". Reversing any pair breaks the deployment:
emitting v3 before the readers know the key strands fresh captures;
flipping strict mode before all v2 archives are re-captured deletes
the restore path for archives that may still be the only copy of a
tenant's state.

### Key rotation

The same ordering rule applies to rotation: roll the *new* key to the
reader tier first as a second accepted key (the reader retains the
old key for a transitional window), then roll the new key to the
writer tier, then drop the old key from the reader tier once no v3
archive signed with the old key remains in scope. The reader's
multi-key support and the operator-facing tooling for it land
alongside the routes themselves; the env-var shape today carries a
single key and rotation is performed by deploying a new key in a
rolling restart once every v3 archive signed with the previous key
has been re-captured. Production deployments planning to rotate
sooner than the snapshot retention window should track this gap as a
follow-up against the v0.3.7 milestone.

## Ed25519 asymmetric signatures (wire v3, `signature_kind = 2`)

The v3 trailer also supports an **asymmetric** signature. It reuses the
same `V3_TRAILER_MAGIC` (`b"S3T1"`) and inner `version = 3` as the HMAC
trailer; only the `signature_kind` byte (`2`) and the signature length
(64 bytes, total trailer 69 bytes) differ. The signed message is
byte-identical in shape to the HMAC input — `prefix_bytes ‖
V3_TRAILER_MAGIC ‖ [signature_kind]` — so the whole v2-shaped frame, the
magic, and the kind byte are all authenticated, and an attacker cannot
rewrite the kind byte to `1` to attempt a cross-scheme downgrade.

Asymmetric signing fits the **single-publisher, many-verifier** case:
the publisher holds a private `ed25519_dalek::SigningKey`
(`SnapshotWriter::with_ed25519_signing_key`); verifiers hold only the
public `VerifyingKey` (`SnapshotReader::with_ed25519_verifying_key`), so
a compromised verifier cannot mint snapshots. HMAC, by contrast, requires
every verifier to hold the symmetric secret (which can also forge). The
reader verifies with `verify_strict` (not the permissive `verify`),
rejecting non-canonical signatures.

Compatibility notes:

- An Ed25519-keyed writer **always** emits the inline v3 envelope, never
  v4 — the artifact envelope authenticates with HMAC only. If both an
  Ed25519 and an HMAC key are configured on one writer, the Ed25519
  trailer wins (the writer never double-signs).
- The `SignatureKind` enum is `#[non_exhaustive]`, which is how Ed25519
  was added without a wire-format break. The reader's trailer detector
  probes both candidate lengths (69 then 37) and accepts the first whose
  magic *and* kind byte are self-consistent with the probed length.

## v4 artifact envelope (default for HMAC-keyed writers)

Wire v4 is the unified content-addressed envelope from
`tensor-wasm-artifacts`, gated by the (default) `artifact-backing`
feature. On-disk shape (byte layout authoritative in
[`FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md)):

```text
b"twasm-artifact01"(16) || version(=1) || blake3(payload)(32)
                        || zstd(bincode(Snapshot)) || hmac_sha256(prefix)(32)
```

- **Inner version is `2`.** The outer envelope owns authentication, so no
  inner v3 trailer is written; the reader rejects an inner `version = 3`
  on this path.
- **HMAC input order on the v3 *inline* path** is `prefix ‖
  V3_TRAILER_MAGIC ‖ [signature_kind]`; the v4 envelope instead HMACs the
  artifact-envelope prefix (magic ‖ version ‖ blake3 ‖ compressed payload)
  per the artifact crate. Both authenticate before any decompressed byte
  reaches the snapshot decoder ("authenticate then parse").
- **Reader detection order:** (1) leading `b"twasm-artifact01"` ⇒ v4;
  (2) `bytes[len-37..len-33] == b"S3T1"` ⇒ legacy v3; (3) otherwise ⇒
  legacy v2. No version field is consulted until the envelope has
  authenticated the bytes.
- The legacy v2/v3 decoders remain accepted on the read side
  **indefinitely** — every snapshot already on disk continues to load
  byte-for-byte unchanged.

A separate, also-`artifact-backing`-gated pair —
`SnapshotWriter::capture_to_artifact_store` /
`SnapshotReader::restore_from_artifact_store` — routes snapshots through a
persistent `DiskArtifactStore` (atomic-rename, content-addressed, key
-fingerprinted partitions). Its on-disk envelope is the same v4 shape; see
[`FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) §
"Artifact-store backing" and
[`ARTIFACT-STORE.md`](./ARTIFACT-STORE.md).

## Replay / rollback / freshness protection

The `SnapshotMetadata` fields `sequence_no: u64` and `nonce:
Option<[u8; 16]>` are **live** (no longer reserved-but-inert), and a
timestamp-freshness check is available. All three are **opt-in** and run
**after** authentication and the structural checks, so for an
authenticated v3/v4 blob the signature/MAC has already certified the
fields — an attacker cannot rewrite them without re-signing.

| Check | Writer opt-in | Reader opt-in | Rejection (`TensorWasmError`) |
|-------|---------------|---------------|-------------------------------|
| Sequence floor | `with_sequence_no(n)` (default `0`) | `with_min_sequence_no(floor)` | `Serialization("snapshot sequence_no … below floor …")` |
| Nonce match | `with_nonce(bytes)` (default `None`) | `with_expected_nonce(bytes)` | `Serialization("snapshot nonce mismatch …" / "… nonce missing …")` |
| Freshness | (uses `created_unix_ms`) | `with_max_age(duration)` (default `None`) | `SnapshotTooOld` when `now - created > max_age` |

**Operator pattern — per-signing-key high-water mark:** track a persistent
`last_seen` sequence number per signing key; construct the reader with
`with_min_sequence_no(last_seen)` (or `last_seen + 1` to forbid replay of
the exact last value) and, after a successful restore, set `last_seen =
max(last_seen, restored.metadata.sequence_no)`. This closes the rollback
window that `with_max_age` alone cannot: an attacker who replays a
once-valid capture *inside* the `max_age` window still trips the sequence
floor.

## Compatibility: old reader ↔ new blob

- **Old reader, new blob.** A pre-v0.4 reader (no `artifact-backing`) sees
  a v4 blob's leading `twasm-artifact01` magic as unknown input — it does
  not match the v2 bincode magic or the v3 trailer magic and is refused.
  A pre-v0.3.6 (v2-only) reader refuses a v3 blob because the inner
  `version = 3` is unknown to it. A reader built without
  `signed-snapshots` cannot verify a v3 trailer at all.
- **New reader, old blob.** A current reader auto-detects and accepts v2
  and (with the appropriate key) v3, and reads v4 when `artifact-backing`
  is compiled in — see the detection order above. Legacy v2/v3 reads are
  supported indefinitely.
- **Pre-T8 v3 blobs** (33-byte trailer, no `S3T1` prefix) fail the current
  classifier and must be re-signed with a current writer; v2 snapshots are
  unaffected by the T8 change.

## Migration paths supported

The reader does **not** attempt in-place migration of older snapshots: a
`version` mismatch is a hard error. The supported upgrade path is to
re-capture from the live instance under the new format. If a deployment
cannot do that (e.g. the source instance is long gone), the recommended
workaround is to run an older `tensor-wasm` binary against the old snapshot,
restore the instance, then re-capture under the current binary. A
`tensor-wasm-cli snapshot migrate` subcommand is **not** planned for v1.0; if
the use case proves common in beta deployments, it becomes a v1.x item.

The v2 → v3 and v3/v4 transitions are the explicit exceptions: the v3
trailer sits after a v2-shaped frame and the v4 envelope is detected by
its own leading magic, so a current reader accepts v2, v3, and v4 side by
side on the read path. No re-capture is *forced* at any of these format
bumps — operators re-capture on their own schedule and flip the reader to
strict mode once that schedule completes. (Pre-T8 v3 blobs are the one
exception that *does* force a re-sign; see
[old reader ↔ new blob](#compatibility-old-reader--new-blob).)

## Related docs

- [`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) —
  the wire-format spec (envelope, encoding, size caps, version history).
- [`crates/tensor-wasm-snapshot/README.md`](../crates/tensor-wasm-snapshot/README.md) —
  crate-level overview and hardening notes.
- [`PATH-TO-V1.md`](./PATH-TO-V1.md) — the v0.5 exit criterion this
  document satisfies.
- [`COLD-START.md`](./COLD-START.md) — how the snapshot subsystem is used
  on the restore hot path.
