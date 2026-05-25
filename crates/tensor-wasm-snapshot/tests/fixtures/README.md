<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Snapshot golden fixtures

This directory holds binary snapshot blobs used by `tests/compat.rs` to prove
that the current `SnapshotReader::restore` accepts snapshots produced by past
versions of the writer. See [`docs/SNAPSHOT-COMPATIBILITY.md`](../../../../docs/SNAPSHOT-COMPATIBILITY.md)
for the compatibility promise and the procedure for adding a new fixture when
the format version bumps.

The blobs are **not regenerated on every test run** — their value is precisely
that they encode a *frozen* historical wire format. They are produced once by
[`examples/generate_golden.rs`](../../examples/generate_golden.rs) and checked
in verbatim.

## Files

| File | Format version | Bodies |
|------|----------------|--------|
| `golden_v0_1_0_minimal.snap` | `2` (as of v0.1.0) | empty wasm / gpu / registers |
| `golden_v0_1_0_with_wasm_memory.snap` | `2` (as of v0.1.0) | 4 KiB wasm, 1 KiB gpu, 256 B registers |

## Regenerating

From the repo root:

```sh
cargo run -p tensor-wasm-snapshot --example generate_golden -- \
    crates/tensor-wasm-snapshot/tests/fixtures
```

This overwrites the two files above and commits no other changes. If the
resulting bytes differ from what was previously checked in, that is a
deliberate format change — bump `SNAPSHOT_VERSION` and add a new fixture
filename under the new version rather than mutating the historical one.
