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
authenticate the source. Snapshots restored from untrusted peers should be
paired with a transport-layer signature (mTLS, signed manifest). The
v1.0 line does not change this — adding payload signing is a v2.x item.

## Format-version → behavior matrix

| `SNAPSHOT_VERSION` | First release | Wire changes | Reader behavior |
|--------------------|---------------|--------------|-----------------|
| `1` | (pre-v0.1.0 development only) | Initial layout. No CRC, no enforced size caps. | Refused by every shipped reader; never reached `main` as a released format. |
| `2` *(current)* | v0.1.0 preview | Added `Snapshot::crc32`; enforced per-blob size caps (`limits::MAX_*_BYTES`); zero-copy write path via `serde_bytes` (wire-identical to plain `Vec<u8>` bincode). | Current `SnapshotReader::restore` accepts. |
| `3` *(future, v0.2 candidate)* | TBD | TBD — likely candidates: kernel-args metadata field, GPU-buffer chunking. | A v0.2+ reader will accept both `2` and `3`. |

Every entry adds a *new* version; existing rows are never removed. A
reader from release N must support every version `2 ..= N` listed above
once v0.5 freezes the wire format. Until then, the table also lists
"pre-freeze" rows where a version is intentionally accepted by only one
reader.

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

## Migration paths supported

The reader does **not** attempt in-place migration of older snapshots: a
`version` mismatch is a hard error. The supported upgrade path is to
re-capture from the live instance under the new format. If a deployment
cannot do that (e.g. the source instance is long gone), the recommended
workaround is to run an older `tensor-wasm` binary against the old snapshot,
restore the instance, then re-capture under the current binary. A
`tensor-wasm-cli snapshot migrate` subcommand is **not** planned for v1.0; if
the use case proves common in beta deployments, it becomes a v1.x item.

## Related docs

- [`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md) —
  the wire-format spec (envelope, encoding, size caps, version history).
- [`crates/tensor-wasm-snapshot/README.md`](../crates/tensor-wasm-snapshot/README.md) —
  crate-level overview and hardening notes.
- [`PATH-TO-V1.md`](./PATH-TO-V1.md) — the v0.5 exit criterion this
  document satisfies.
- [`COLD-START.md`](./COLD-START.md) — how the snapshot subsystem is used
  on the restore hot path.
