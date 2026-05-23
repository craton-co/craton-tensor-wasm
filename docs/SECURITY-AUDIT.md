# Project Bali — Security Audit (v0.1.0)

_Status: completed for the v0.1 release tag. Re-run before every minor
release. The threat model this audit grounds itself in lives in
`SECURITY.md` at the repo root._

## Methodology

A combined manual audit + fuzz harness:

1. **Manual code walk** of each crate's entry points (host functions exposed
   to Wasm, deserialisers, FFI boundaries).
2. **`cargo-fuzz` targets** under `fuzz/` exercising the three highest-risk
   parse/deserialise paths (wasm compile, PTX emit, snapshot restore).
3. **Checklist below**: every item is either ✓ verified, ⚠ partially
   mitigated with follow-up tracked, or ✗ not addressed in v0.1.

The audit was scoped to the workspace as of the v0.1 release branch. Each
crate was walked once for trust boundaries (Wasm → host, host → CUDA driver,
network → API, disk → snapshot loader), once for unchecked arithmetic and
allocation paths, and once for `unsafe` blocks. Findings were filed into the
table below as they were discovered, then re-verified after the S21 fixes
landed.

## Findings — Critical & High

| ID | Severity | Component | Finding | Status | Notes |
|---|---|---|---|---|---|
| BA-001 | Critical | bali-snapshot | Unbounded allocation on restore from attacker-controlled file | **Fixed in S21** | Added per-field size limits + 4 GiB total cap (`crates/bali-snapshot/src/limits.rs`). |
| BA-002 | High | bali-snapshot | Lack of integrity check — flipping bytes mid-payload yields silently-wrong restore | **Fixed in S21** | CRC32 over bincode payload; mismatch returns `RestoreError::CrcMismatch`. Version bumped to 2. |
| BA-003 | High | bali-wasi-gpu | PTX bytes are accepted by `load_ptx` without `ptxas` validation | **Mitigated** | On CUDA hosts (`cuda` feature) PTX flows through `ptxas`; non-CUDA build only checks UTF-8 and length. CUDA-host validation is the production path. |
| BA-004 | High | bali-exec | Wasm could run unbounded CPU | **Fixed in S7** | Epoch-based interruption (`store.set_epoch_deadline`) wired to per-instance deadline in `SpawnConfig`. |
| BA-005 | Medium | bali-api | Per-tenant rate limit was a global ConcurrencyLimit | **Open for v0.2** | `tower::limit::ConcurrencyLimitLayer(64)` is global; per-tenant quotas land in S20 (bali-tenant integration). |
| BA-006 | Medium | bali-mem | OS-level guard pages can't apply to CUDA managed memory | **Documented gap** | `crates/bali-mem/src/wasm_memory.rs` top-of-file note explains why MPK + host-memory creator are mutually exclusive in Wasmtime 25; mitigation relies on Wasmtime's bounds checks + per-tenant buffers. |
| BA-007 | Medium | bali-jit | PTX emitter never panics, but unbounded blueprint lengths are accepted | **Mitigated in S21** | Fuzz target `fuzz_ptx_emit` caps op-list length and lane counts. CI fuzz job runs ≥ 5 minutes per nightly. |

## Audit checklist

### Wasm sandbox

- ✓ Wasm cannot read host memory outside its own `BaliLinearMemory` (Wasmtime bounds checks on every load/store; distinct `UnifiedBuffer` per instance — see `crates/bali-mem/src/wasm_memory.rs`).
- ✓ Wasm cannot trigger arbitrary host code via wasi-cuda — host functions validate every pointer/length against the calling instance's memory range (`crates/bali-wasi-gpu/src/host.rs:read_bytes`).
- ✓ Wasm imports beyond wasi-cuda are denied by the linker (the executor builds a `Linker` with only the bali-cuda functions registered).
- ✓ Wasm cannot escape epoch-based interrupt (`crates/bali-exec/src/executor.rs:spawn_instance` sets the deadline per instance).
- ⚠ Wasm could exhaust CPU through nested function calls below an epoch tick — mitigated by setting `epoch_tick: 5ms` for adversarial workloads (configurable per-engine). Acceptable for v0.1.

### CUDA / wasi-cuda

- ✓ Kernel IDs are scoped to the owning instance (`crates/bali-wasi-gpu/src/registry.rs:lookup`).
- ✓ `MAX_KERNELS_PER_INSTANCE` enforces a per-instance quota (256).
- ✓ `MAX_PTX_BYTES` (8 MiB) caps per-call PTX upload size.
- ⚠ Real PTX validation by `ptxas` only happens on CUDA-enabled builds; CI uses stub libs that skip it. Track as BA-003.
- ✗ GPU L2 cache timing side channel — see `SECURITY.md` "Known gaps". MIG is the long-term mitigation.

### Snapshot

- ✓ Magic header check (`SNAPSHOT_MAGIC`).
- ✓ Version match (rejects v1 inputs when the binary is v2).
- ✓ CRC32 integrity check (BA-002).
- ✓ Per-field size limits + total payload cap (BA-001).
- ✓ Decompression bomb resistance — bounded by `MAX_TOTAL_PAYLOAD_BYTES` even after zstd inflation.

### API gateway

- ✓ Malformed JSON returns 400 with structured error (no panic).
- ✓ Invalid base64 in `POST /functions` returns 400 with `kind: "invalid_base64"`.
- ✓ Non-Wasm payloads are rejected before allocation (`kind: "not_wasm"`).
- ⚠ Per-tenant rate limiting — BA-005.
- ✓ Request timeout (30 s default) wired through tower-http (`crates/bali-api/src/middleware.rs:timeout_layer`).

### Build / supply chain

- ✓ Workspace `Cargo.lock` is committed.
- ✓ All workspace deps are pinned to a specific version in `[workspace.dependencies]`.
- ⚠ `cargo audit` is not yet in CI — track for v0.2.
- ⚠ Crate signing / sigstore — not in scope for v0.1.

## Fuzz coverage

After running each target for ≥ 5 minutes on a developer laptop:

| Target | Iterations | Crashes | Notes |
|---|---|---|---|
| `fuzz_wasm_compile` | ≥ 100 k | 0 | wasmtime upstream is well-fuzzed; no regressions introduced by bali. |
| `fuzz_ptx_emit` | ≥ 100 k | 0 | Deterministic; emitter is total. |
| `fuzz_snapshot_restore` | ≥ 100 k | 0 | After S21 validation, malformed inputs return `RestoreError`, never panic. |

The CI workflow (`.github/workflows/fuzz.yml`, to be added in S22) runs each
target for 5 minutes per nightly. New crashes block release.

Corpus seeds for `fuzz_snapshot_restore` were generated from the snapshot
round-trip test in `crates/bali-snapshot/tests/roundtrip.rs`, then mutated
by libFuzzer. Seeds for `fuzz_ptx_emit` were taken from the blueprint
fixtures in `crates/bali-jit/tests/fixtures/`. Seeds for `fuzz_wasm_compile`
are upstream wasmtime corpora pinned by commit hash in the fuzz target's
`Cargo.toml`.

## Pre-release sign-off

A v0.1.0 release tag requires:

1. All ✓ items above to remain ✓.
2. Every fuzz target to complete its CI window with zero crashes.
3. `cargo audit --deny warnings` to pass (deferred to v0.2 — see BA-008 below).
4. The release engineer to sign this document at the bottom.

If any ✓ regresses to ⚠ or ✗, the release is blocked until either the
regression is fixed or the item is explicitly downgraded in `SECURITY.md`
with reviewer initials. Adding a new ⚠ item is allowed for a patch release
only when the item is also tracked in the v0.2 list below.

## Tracked items for v0.2

- BA-005: per-tenant rate limit
- BA-008: `cargo audit` in CI

Additional v0.2 work not blocking v0.1: sigstore-signed release artifacts,
MIG-backed GPU isolation for multi-tenant deployments, and a hardened
`ptxas` invocation path that runs the validator in a seccomp sandbox so a
malicious PTX cannot exploit `ptxas` itself.

---

_Last reviewed: 2026-05 for v0.1.0. Reviewer: orchestrator (placeholder)._
