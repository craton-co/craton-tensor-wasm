<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Wasmtime upgrade cadence policy

Wasmtime is the largest single dependency in this workspace and the
one most likely to drag the rest of the tree through a flag day. This
doc defines the cadence at which we bump it, the checklist a
maintainer follows for each bump, and the rules that turn a bump
into a TensorWasm release.

It is the artifact behind the API and ABI workstream entry in
[`PATH-TO-V1.md`](PATH-TO-V1.md): *"Wasmtime upgrade cadence policy
(quarterly minor bumps, major bumps case-by-case)."*

If you only read one section, skip to the
[bump checklist](#bump-checklist) — that is the concrete sequence a
maintainer follows when a new upstream minor lands.

## Contents

1. [Purpose](#purpose)
2. [Current pin](#current-pin)
3. [Cadence rules](#cadence-rules)
4. [Bump checklist](#bump-checklist)
5. [CVE shortcut](#cve-shortcut)
6. [Communicating bumps to users](#communicating-bumps-to-users)
7. [What forces a major bump](#what-forces-a-major-bump)
8. [Skip-version policy](#skip-version-policy)
9. [References](#references)

---

## Purpose

Wasmtime ships a new minor roughly every 30 days. TensorWasm cannot
follow that train one-for-one — every bump touches at least four
crates ([`tensor-wasm-exec`](../crates/tensor-wasm-exec),
[`tensor-wasm-mem`](../crates/tensor-wasm-mem),
[`tensor-wasm-wasi-gpu`](../crates/tensor-wasm-wasi-gpu),
[`tensor-wasm-jit`](../crates/tensor-wasm-jit)) and the snapshot
suite sometimes shifts bytes. So we batch.

"Supported" for a given Wasmtime version means: the workspace builds
and `cargo test --workspace` passes on the pinned nightly; the bench
corpus in
[`crates/tensor-wasm-bench`](../crates/tensor-wasm-bench) holds the
regression gate against
[`bench-results/baseline.json`](../bench-results/baseline.json); all
W1.3 snapshot fixtures load (see
[`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md)); and the
companion `wat`/`wasmparser`/`cranelift-codegen` pins match the
versions the Wasmtime release documents.

There is exactly one supported Wasmtime version on `main` at a time.
Backports to a previous v1.x line MAY pin an older Wasmtime as long
as that pin still gets upstream security patches; we do not maintain
support for unmaintained Wasmtime branches.

---

## Current pin

| Field | Value |
|---|---|
| Wasmtime version | `25.x` (the `25` major) |
| Last known-good patch | `25.0.3` |
| Set in | [`Cargo.toml`](../Cargo.toml) `[workspace.dependencies]` |
| Companion `cranelift-codegen` | `0.111` |
| Companion `wasmparser` | `0.218` |
| Companion `wat` | `1` (compatible across the `1.x` range) |
| Dependabot ignore | major bumps on `wasmtime` and `wasmtime-wasi` (see [`.github/dependabot.yml`](../.github/dependabot.yml)) |

Rationale for sitting on `25.x` at v0.1.0: Wasmtime 26 revises the
`LinearMemory` / `MemoryCreator` traits used by
[`tensor-wasm-mem::wasm_memory`](../crates/tensor-wasm-mem/src/wasm_memory.rs)
and ships a `cranelift-codegen` past 0.111 that breaks the IR shape
[`tensor-wasm-jit::clif_lower`](../crates/tensor-wasm-jit/src/clif_lower.rs)
consumes. The full justification lives in [`RISKS.md`](RISKS.md)
under "Wasmtime pin"; this doc is the operational follow-through.

The pin is a workspace dep so every member inherits via
`wasmtime.workspace = true`. Do not hard-pin a different version in
a member `Cargo.toml` — that has bitten us once and is a CI lint
candidate.

---

## Cadence rules

### Minor bumps — quarterly

We bump the Wasmtime minor **once per calendar quarter**, targeting
the last upstream release before the quarter closes:

| Quarter | Bump window | Default target |
|---|---|---|
| Q1 | weeks 11-13 | Wasmtime release dated in March |
| Q2 | weeks 24-26 | Wasmtime release dated in June |
| Q3 | weeks 37-39 | Wasmtime release dated in September |
| Q4 | weeks 50-52 | Wasmtime release dated in December |

If upstream did not ship a minor that quarter, we skip too — no
"backfill" bump to compensate. The bump runs through the
[checklist below](#bump-checklist) end to end on a feature branch
with PR title `chore(wasmtime): bump to <new>`.

### Patch bumps — opportunistic

Wasmtime patches (e.g. `25.0.3` → `25.0.4`) get batched into the
next minor PR by default. The exception is a CVE — see the
[CVE shortcut](#cve-shortcut). Otherwise the policy is: keep
Cargo.lock honest, do not open a patch-bump PR mid-quarter unless
something is broken.

### Major bumps — RFC required

A Wasmtime major bump (today: 25 → 26 → 27 → ...) goes through the
[RFC process](../rfcs/README.md):

1. An RFC PR under [`rfcs/`](../rfcs/) using
   [`rfcs/TEMPLATE.md`](../rfcs/TEMPLATE.md), listing the
   upstream-breaking changes that affect us and naming the migrator.
2. One-week comment window; a maintainer accepts or rejects.
3. On accept, an implementation PR labelled `wasmtime-major` lands.

Two hard rules:

- **A Wasmtime major never ships in the same TensorWasm release as a
  TensorWasm major.** Users should not absorb two surface changes at
  once with no way to bisect.
- **Wasmtime majors do not land during a beta or rc.** In the
  [`v0.5.0-beta`](PATH-TO-V1.md#v050-beta--external-validation) cycle
  or later, the bump waits for the next minor train.

### Security advisories — out of band

A Wasmtime security advisory is the one trigger that overrides the
quarterly cadence. See the [CVE shortcut](#cve-shortcut). The target
is: a patched workspace pin merged within **7 days** of upstream
publishing the advisory, with a hotfix TensorWasm patch release if the
last released TensorWasm version is affected.

---

## Bump checklist

Use this for a minor bump. The CVE shortcut truncates it to the
security-critical items; major bumps run the RFC plus this list at
higher rigor.

1. **Read the upstream release notes** at
   [`bytecodealliance/wasmtime/releases`](https://github.com/bytecodealliance/wasmtime/releases).
   Note every "breaking" line, every change to the companion
   `cranelift-codegen`/`wasmparser`/`wat` versions, and every change
   to the `LinearMemory` / `MemoryCreator` / `Store` / `Engine` /
   `Component` traits. Copy the list into the PR description.

2. **Update the workspace pin.** In [`Cargo.toml`](../Cargo.toml),
   bump `wasmtime`, `wasmtime-wasi`, and the companion pins to the
   versions the new Wasmtime release pulls in.

3. **Rebuild the lockfile** with precise updates to avoid unrelated
   churn:

   ```sh
   cargo update -p wasmtime --precise <new>
   cargo update -p wasmtime-wasi --precise <new>
   cargo update -p cranelift-codegen --precise <companion>
   cargo update -p wasmparser --precise <companion>
   ```

4. **Compile the full workspace** with default and CUDA feature
   sets:

   ```sh
   cargo build --workspace --all-targets
   cargo build --workspace --all-targets --features "cuda,unified-memory,mps,auto-offload"
   ```

   Likely fix sites on breakage:
   [`tensor-wasm-mem::wasm_memory`](../crates/tensor-wasm-mem/src/wasm_memory.rs)
   and
   [`tensor-wasm-jit::clif_lower`](../crates/tensor-wasm-jit/src/clif_lower.rs).

5. **Run the full test suite.** CI runs `cargo test --workspace`;
   locally also enumerate the feature-gated subcrates that the
   default workspace run skips:

   ```sh
   cargo test --workspace --all-targets
   cargo test -p tensor-wasm-mem --features "cuda,unified-memory"
   cargo test -p tensor-wasm-wasi-gpu --features "cuda"
   cargo test -p tensor-wasm-snapshot
   cargo test -p tensor-wasm-jit
   cargo test -p tensor-wasm-exec
   cargo test -p tensor-wasm-api
   cargo test -p tensor-wasm-cli
   ```

6. **WAT fixtures load.** All `.wat`/`.wasm` files under
   [`tests/wasm-fixtures/`](../tests/wasm-fixtures) and the inline WAT
   fixtures in [`crates/tensor-wasm-bench/benches/`](../crates/tensor-wasm-bench/benches)
   must still parse and instantiate. Snapshot e2e tests cover the
   typical path.

7. **Snapshot compat tests pass.** The W1.3 framework
   ([`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md)) keeps
   golden fixtures under
   `crates/tensor-wasm-snapshot/tests/fixtures/`. Every committed
   fixture must restore under the new Wasmtime without a format
   migration; a fixture failing here is the most common signal that
   a bump should be reclassified as a major.

8. **Bench delta.** Run the regression corpus:

   ```sh
   cargo bench -p tensor-wasm-bench
   ```

   Compare against
   [`bench-results/baseline.json`](../bench-results/baseline.json):

   - Median within the 10% `regress_pct_threshold` → pass.
   - Over threshold but within per-metric `tolerance_pct` → record
     in the PR with the explanation.
   - Outside tolerance → either investigate the regression or land a
     re-baseline first
     ([`bench-results/README.md`](../bench-results/README.md)).

   Never tighten the baseline in the same PR as a Wasmtime bump.

9. **Dependabot churn.** Wasmtime bumps drag transitive deps
   (`object`, `gimli`, `cap-std`); let Dependabot open the downstream
   PRs in the same week and merge serially.

10. **CHANGELOG entry** under the upcoming release in
    [`CHANGELOG.md`](../CHANGELOG.md) naming the old and new pins,
    linking the upstream release notes, and stating user-visible
    impact (or "no user-visible behavior change").

11. **Risk-register touch-up.** If the bump changed the pin
    rationale, update the "Wasmtime pin" section of
    [`RISKS.md`](RISKS.md).

12. **Open the PR** with title `chore(wasmtime): bump to <new>`.
    Body pastes the breaking-change list from step 1, the bench
    delta from step 8, and references the upstream release page.

A repeat-offender maintainer does a minor bump in about half a day.

---

## CVE shortcut

When upstream publishes a security advisory affecting our pinned
Wasmtime:

1. **Within 24 hours.** Open a tracking issue with the advisory
   link, the assessed exposure (is the affected code path reachable
   from defaults? from any feature flag?), and the proposed patched
   version.
2. **Within 72 hours.** Open the bump PR using a truncated
   checklist: steps 2, 3, 4, 5, 6, 7, 10, 11, 12 from
   [Bump checklist](#bump-checklist) are mandatory; step 8 (bench
   delta) is skipped for the security PR and run as a follow-up.
   A measured regression does not block the CVE patch.
3. **Within 7 days.** Patched PR merged. If the last released
   TensorWasm version is affected and the user-facing remediation is
   "upgrade TensorWasm," cut a TensorWasm patch release per
   [Communicating bumps to users](#communicating-bumps-to-users),
   tagged with the upstream CVE id.
4. **Within 14 days.** Re-baseline if the post-merge bench delta
   showed a real regression; otherwise close the follow-up.

The shortcut compresses wall-clock time without skipping the steps
that catch breakage in subsystems with no upstream advisory of their
own (snapshot restore, JIT lowering, WASI-GPU host functions).

---

## Communicating bumps to users

How a Wasmtime bump surfaces in the TensorWasm release stream
depends on what it changes:

- **Patch bump, no user-visible behavior change.** TensorWasm patch
  release. Release notes one-line the bump. No migration doc.
- **Minor bump.** TensorWasm **minor** release on v0.x. Aligns with
  [`PATH-TO-V1.md`](PATH-TO-V1.md#open-decisions-to-resolve-before-v10)
  Open Decision 8 — nightly-toolchain and Wasmtime cadence drive
  the v0.x minor cadence together. Release notes summarize upstream
  breaking changes and TensorWasm-side adapter shifts.
- **Major bump.** TensorWasm **major** at v1.0+ (during v0.x, still
  a minor only because v0.x is pre-stability). Always ships with
  `docs/MIGRATION-<old>-to-<new>.md` covering API-shape changes,
  silent behavior changes, and removed Wasmtime features users may
  have relied on transitively.
- **CVE patch.** TensorWasm patch release on every supported v0.x
  and v1.x line, naming the CVE id and linking the advisory. An
  unavoidable behavior change (rare) gets major-style migration
  treatment.

The [`CHANGELOG.md`](../CHANGELOG.md) release-notes template has a
"Wasmtime" subsection that must be present for every release that
touched the pin.

---

## What forces a major bump

A Wasmtime minor bump becomes a major (RFC required, restrictions
in [Cadence rules](#cadence-rules)) when any of these hold. None
are theoretical — each has happened on the Wasmtime release train.

- **Upstream removes a feature we depend on** — the `async`,
  `cranelift`, `component-model`, or `runtime` flags we enable in
  [`Cargo.toml`](../Cargo.toml).
- **Upstream removes or restructures async.** Shape changes to
  `Store::call_async`, the epoch interruption model, or the
  fuel-vs-epoch dichotomy force an audit of every async caller in
  [`tensor-wasm-exec`](../crates/tensor-wasm-exec) and ripple into
  [`tensor-wasm-api`](../crates/tensor-wasm-api).
- **Upstream changes the WASI Preview level.** We are on Preview 2
  via [`wit/`](../wit/) and
  [`tensor-wasm-wasi-gpu`](../crates/tensor-wasm-wasi-gpu); dropping
  it or defaulting to Preview 3 forces a WIT rewrite and a
  guest-fixture rebuild.
- **Upstream changes the `Engine` config API.** The
  `tensor-wasm-exec::engine` builder mirrors Wasmtime's `Config`
  closely; a breaking rename surfaces in user-facing config.
- **Upstream changes `LinearMemory` / `MemoryCreator`.** The
  already-confirmed reason 26 is a major (see
  [`RISKS.md`](RISKS.md)).
- **Upstream changes the serialization byte format.** If
  `Module::serialize` output bytes shift, all W1.3 fixtures need
  regeneration and
  [`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) governs
  the migrator-vs-break decision.
- **Upstream changes the component-model surface.** A breaking
  change to `Component::new` or the host adapter is a major.

Anything else — a new opt-in feature, a perf improvement, a default
flag flip we can override — is a minor.

---

## Skip-version policy

Yes, we can skip versions. The cadence does not require visiting
every Wasmtime release in order. The cost rises with the number
skipped because a run of skipped minors can collapse a deprecation
→ opt-out → removal sequence into one cold step.

- **One skipped minor: no extra ceremony.** Read the notes for both
  the skipped and target versions.
- **Two skipped: read both sets carefully.** Flag deprecation
  warnings in the skipped release that became hard errors in the
  target. The PR description must list those transitions.
- **Three or more skipped: treat as a major bump.** RFC required,
  even if upstream classifies each release as a minor individually.

The quarterly cadence is calibrated so we skip at most two minors
between bumps. If we ever fall behind enough to skip three, the
recovery bump is treated as a major regardless of upstream
classification.

---

## References

- Upstream releases:
  [`bytecodealliance/wasmtime/releases`](https://github.com/bytecodealliance/wasmtime/releases)
- Upstream security advisories:
  [`bytecodealliance/wasmtime/security/advisories`](https://github.com/bytecodealliance/wasmtime/security/advisories)
- RustSec for `wasmtime`:
  [`rustsec.org/packages/wasmtime`](https://rustsec.org/packages/wasmtime/)
- [`PATH-TO-V1.md`](PATH-TO-V1.md) — API/ABI workstream this policy
  satisfies; Open Decision 8 on release-cadence communication.
- [`RISKS.md`](RISKS.md) — original "Wasmtime pin" entry whose
  operational follow-through this doc is.
- [`WASMTIME-FORK.md`](WASMTIME-FORK.md) — why we did not fork
  Wasmtime; this cadence assumes we stay on the upstream line.
- [`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md) — W1.3
  framework exercised by [bump checklist](#bump-checklist) step 7.
- [`BENCHMARKING.md`](BENCHMARKING.md) — bench corpus and regression
  semantics for step 8; the ["vs
  Wasmtime"](BENCHMARKING.md#vs-wasmtime-upstream) recipe in
  particular: every bump tightens or relaxes the "TensorWasm overhead
  vs upstream Wasmtime" number we publish.
- [`rfcs/README.md`](../rfcs/README.md) — RFC process invoked for
  every Wasmtime major.
- [`.github/dependabot.yml`](../.github/dependabot.yml) — Dependabot
  ignores Wasmtime majors; patches and minors flow through normally.

---

_Status: written alongside the v0.1.0 fix wave as part of the v0.4
API/ABI workstream. Revisit at v0.5-beta: by then we will have lived
through one or two bumps under this policy and either the cadence
or the checklist will need tightening based on real experience._
