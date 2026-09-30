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

| File | Format version | Bodies | Regenerate? |
|------|----------------|--------|-------------|
| `golden_v0_1_0_minimal.snap` | `2` (as of v0.1.0) | empty wasm / gpu / registers | NO — frozen historical bytes |
| `golden_v0_1_0_with_wasm_memory.snap` | `2` (as of v0.1.0) | 4 KiB wasm, 1 KiB gpu, 256 B registers | NO — frozen historical bytes |
| `golden_current_minimal.snap` | current `SNAPSHOT_VERSION` | empty wasm / gpu / registers | YES — on every format bump |
| `golden_current_with_wasm_memory.snap` | current `SNAPSHOT_VERSION` | 4 KiB wasm, 1 KiB gpu, 256 B registers | YES — on every format bump |

### Frozen vs. current

The `golden_v0_1_0_*` fixtures are a **frozen** historical wire format. They no
longer decode under the current reader (the `Snapshot` payload grew the
`sequence_no` / `nonce` metadata fields after v0.1.0), so their round-trip tests
in `compat.rs` are `#[ignore]`d and serve only as documentation of that break.
They must **not** be regenerated — see the warning at the bottom of this file.

The `golden_current_*` fixtures are minted against the **current**
`SNAPSHOT_VERSION` and metadata layout and back the *un-ignored* cross-version
assertions in `compat.rs`. Those tests also rebuild the same bytes in-process
(deterministically, via a pinned timestamp), so they run even when the binary
fixtures are absent; when the binaries are present the tests additionally assert
they are byte-identical to the freshly built bytes, catching a stale checked-in
fixture after a format bump.

## Regenerating

From the repo root:

```sh
cargo run -p tensor-wasm-snapshot --example generate_golden -- \
    crates/tensor-wasm-snapshot/tests/fixtures
```

This writes all four files. The `golden_v0_1_0_*` historical fixtures are
emitted with the same (frozen) bodies and metadata as before; the
`golden_current_*` fixtures track the live format. If the `golden_v0_1_0_*`
bytes ever differ from what was previously checked in, that is an *accidental*
mutation of the historical record — do not commit it. The `golden_current_*`
bytes are expected to change on a deliberate format bump; commit those and let
`compat.rs`'s `assert_checked_in_matches` confirm they are in sync.
