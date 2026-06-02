<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Software Bill of Materials (SBOM)

_Status: living document. Companion to
[`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md). Together they cover
the v1.0 supply-chain commitments described in
[`PATH-TO-V1.md`](PATH-TO-V1.md) (workstream "Supply chain", milestone
gate "SBOM (CycloneDX) attached to every release artifact" under
v1.0.0-rc1 → v1.0.0)._

Every published TensorWasm release ships a CycloneDX SBOM as a GitHub
release asset. This document explains what that file contains, what it
does **not** contain, how to regenerate it locally, and the contract
maintainers commit to. If you are auditing a release tarball, the SBOM
is the place to start; if you are integrating TensorWasm into a
downstream compliance pipeline (FedRAMP, SOC 2 control evidence,
internal security review), this file tells you what the SBOM is and
isn't authoritative for.

## Contents

1. [What CycloneDX is and why we publish one](#what-cyclonedx-is-and-why-we-publish-one)
2. [The contract](#the-contract)
3. [Release-asset filename convention](#release-asset-filename-convention)
4. [What is in the SBOM](#what-is-in-the-sbom)
5. [What is NOT in the SBOM](#what-is-not-in-the-sbom)
6. [Regenerating locally](#regenerating-locally)
7. [Verifying a release SBOM](#verifying-a-release-sbom)
8. [How CI generates and attaches it](#how-ci-generates-and-attaches-it)
9. [Related](#related)

---

## What CycloneDX is and why we publish one

[CycloneDX](https://cyclonedx.org/) is an OWASP-stewarded SBOM
specification: a JSON (or XML, or protobuf) document that enumerates
every component that went into a piece of software, each with a
version, a license, a package URL (`purl`), and — where the toolchain
can supply it — a cryptographic hash. The format is one of the two
that NTIA accepted as a "minimum element" SBOM standard (the other is
SPDX); we picked CycloneDX because the Rust tooling
([`cargo-cyclonedx`](https://crates.io/crates/cargo-cyclonedx)) is
mature, actively maintained, and emits machine-readable output that
downstream scanners (Trivy, Grype, Dependency-Track, GitHub's own
Dependabot) already consume.

Why bother. Three reasons:

1. **SLSA Level 3 target.** Our v1.0 supply-chain goal
   ([PATH-TO-V1.md, workstream Security](PATH-TO-V1.md#security)) is
   SLSA L3. An SBOM is a prerequisite — without one, the provenance
   attestation has nothing to attest *about*.
2. **Downstream packagers need it.** Debian, Nixpkgs, and Homebrew
   each run automated dependency-scanning against upstream SBOMs.
   Shipping one reduces friction for those repackagers and makes
   security advisories propagate faster (a new RustSec advisory on
   a transitive dep is matched against the SBOM, not re-derived from
   `Cargo.lock`).
3. **Customer compliance.** A growing number of TensorWasm consumers
   operate under regimes (US Executive Order 14028, EU Cyber Resilience
   Act, internal procurement frameworks) that require SBOM disclosure
   from infrastructure dependencies. Publishing one as a release asset
   means the answer to "do you have an SBOM?" is a URL, not a ticket.

The SBOM is generated from the same `Cargo.lock` that produced the
binary, on the same commit, in the same CI run. It is reproducible in
the same sense [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md)
defines: regenerating it from the same source tree with the same
pinned `cargo-cyclonedx` version produces byte-identical output.

---

## The contract

Maintainers commit to the following for every published release on
the v1.0 line and forward:

- **Attached.** A CycloneDX 1.5 JSON SBOM is uploaded as a release
  asset on the corresponding GitHub Release page, alongside the
  binaries and `.sha256` files. No release without an SBOM.
- **Reproducible.** Regenerating the SBOM from the release tag with
  the pinned `cargo-cyclonedx` version in
  [`.github/workflows/sbom.yml`](../.github/workflows/sbom.yml)
  produces a byte-identical file.
- **Signed.** Starting with the SLSA L3 milestone the SBOM is signed
  by the same cosign keyless flow that signs the binaries. Until then
  it ships unsigned but is covered by GitHub's HTTPS transport and the
  release-asset SHA published in the release notes.
- **Pinned tooling.** The version of `cargo-cyclonedx` used to
  generate it is pinned in the workflow and bumped deliberately (PR
  review, not Dependabot auto-merge). The pin lives in the workflow
  env var `CARGO_CYCLONEDX_VERSION`.

If a release goes out without an SBOM, that is a release-blocker
severity bug and gets a postmortem in
[`docs/SECURITY-AUDIT.md`](SECURITY-AUDIT.md). The point of the
contract is to let downstream auditors trust that the SBOM is a
faithful component manifest, not a marketing artifact.

---

## Release-asset filename convention

The SBOM for a release with tag `vX.Y.Z` is published as:

```text
tensor-wasm-cdx-v<X.Y.Z>.json
```

Examples:

- `tensor-wasm-cdx-v1.0.0.json` — the v1.0.0 release SBOM
- `tensor-wasm-cdx-v1.0.1.json` — a v1.0.1 patch release
- `tensor-wasm-cdx-v1.1.0-rc1.json` — a release candidate

Dev-branch SBOMs (produced by pushes to `dev` for verification) are
named with the short commit SHA instead of a version
(`tensor-wasm-cdx-v<sha8>.json`) and are uploaded as workflow
artifacts only — they are **not** attached to any release.

The `cdx` infix denotes the format (CycloneDX); we use it rather than
`.cdx.json` so the artifact name parses cleanly as a single token in
release tooling and changelogs.

---

## What is in the SBOM

For each component:

- **`name`** — the crate name as it appears on crates.io (e.g.
  `wasmtime`, `tokio`, `tensor-wasm-core`).
- **`version`** — the exact version resolved by `Cargo.lock`. No
  ranges; every entry is pinned to a single semver.
- **`purl`** — a [Package URL](https://github.com/package-url/purl-spec)
  of the form `pkg:cargo/<name>@<version>`, the universal identifier
  scanners use to match against vulnerability databases.
- **`hashes`** — for registry components, the SHA-256 from
  `Cargo.lock`'s `checksum =` line. Path and git dependencies have no
  registry checksum and are flagged as such.
- **`licenses`** — parsed from each crate's `Cargo.toml` license
  field. Multi-license expressions (`Apache-2.0 OR MIT`) are preserved
  as SPDX expressions.

The components fall into three categories:

1. **Workspace crates.** Every member listed in `[workspace.members]`
   of the root `Cargo.toml`: `tensor-wasm-core`, `tensor-wasm-mem`,
   `tensor-wasm-exec`, `tensor-wasm-wasi-gpu`, `tensor-wasm-jit`,
   `tensor-wasm-snapshot`, `tensor-wasm-tenant`, `tensor-wasm-api`,
   `tensor-wasm-cli`, `tensor-wasm-bench`. The workspace crate
   `tensor-wasm-bench` is included even though it is not shipped — it
   is part of the source tree at the released commit.
2. **Direct dependencies.** Everything declared in
   `[workspace.dependencies]` and each crate's `[dependencies]` /
   `[dev-dependencies]` / `[build-dependencies]`. CUDA-feature-gated
   deps (`cust`, `cudarc`, `ptx-builder`) appear even when the default
   build does not enable them, because they are part of the resolved
   dependency graph.
3. **Transitive dependencies.** Every crate in `Cargo.lock`. This is
   typically 300+ entries; the bulk of the SBOM is here.

The top-level `metadata.component` field identifies the workspace
itself (`tensor-wasm` at the released version). The `metadata.tools`
field records `cargo-cyclonedx` and the rustc version that produced
the manifest — useful when a downstream scanner asks "which tool
emitted this?" during compliance review.

---

## What is NOT in the SBOM

Equally important. The CycloneDX file describes the Rust dependency
graph; it does **not** describe the runtime environment. Specifically
out of scope:

- **System libraries.** `libc`, `libcuda.so` / `libcudart.so`,
  `libstdc++`, the dynamic linker — these are linked at runtime
  against whatever the host OS provides and are not part of the
  cargo dependency graph. Downstream packagers must derive their own
  manifest of system-package dependencies (Debian
  `debian/control` `Depends:` line, the Nix derivation's
  `buildInputs`, etc).
- **CUDA toolkit components.** PTX assembler, NVRTC, the driver, MPS
  daemons. These are external to the build; see
  [`CUDA-SETUP.md`](CUDA-SETUP.md) for the version matrix.
- **The Rust toolchain itself.** `rustc`, `cargo`, `llvm-tools` are
  pinned by [`rust-toolchain.toml`](../rust-toolchain.toml) and
  recorded in the reproducibility recipe rather than the SBOM. The
  toolchain version *is* embedded in `metadata.tools` so it can be
  cross-referenced, but rustc is not listed as a `component`.
- **Build-time-only tools that do not link in.** `cargo-deny`,
  `cargo-audit`, `cargo-fuzz`, `cargo-cyclonedx` itself. These are
  used during CI and developer workflows but never produce code that
  ends up in a release artifact.
- **The host kernel.** Out of scope by construction. Operators must
  reconcile against their own platform-security baseline.

If you need a complete supply-chain picture (Rust crates + system
libraries + toolchain), combine this SBOM with a runtime scan of the
deployed binary (e.g. `syft` against the release tarball or the
distroless container image from
[`REPRODUCIBLE-BUILDS.md` § Container builds](REPRODUCIBLE-BUILDS.md#container-builds)).

---

## Regenerating locally

The same generation step CI runs, in a single shell session:

```sh
# Pin the same version CI uses. Bumping this is a deliberate workflow
# change, not a casual local override.
cargo install cargo-cyclonedx --version "~0.5" --locked

# Generate. --top-level emits one SBOM rooted at the workspace rather
# than one per crate; --output-pattern bom writes `bom.json` to the
# workspace root.
cargo cyclonedx --format json --output-pattern bom --top-level

# Inspect the top-level component identifier and a few crates.
jq '.metadata.component | {name, version}' bom.json
jq '.components | length' bom.json
jq '.components[] | select(.name == "wasmtime")' bom.json
```

The output filename `bom.json` is intentional: it matches the
CycloneDX-recommended default and is what CI feeds into the renaming
step. If you want to produce a file with the release-asset name
yourself:

```sh
VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)
mv bom.json "tensor-wasm-cdx-v${VERSION}.json"
```

XML output is also available (`--format xml`) if a downstream tool
prefers it; the JSON form is what we publish.

---

## Verifying a release SBOM

To check that the SBOM attached to a release matches what you would
regenerate from the source at that tag:

```sh
# 1. Download both the release SBOM and the source.
gh release download v1.0.0 --pattern 'tensor-wasm-cdx-*.json' --repo craton-co/craton-tensor-wasm
git clone --depth 1 --branch v1.0.0 https://github.com/craton-co/craton-tensor-wasm
cd craton-tensor-wasm

# 2. Regenerate with the pinned tool version (same as CI).
cargo install cargo-cyclonedx --version "~0.5" --locked
cargo cyclonedx --format json --output-pattern bom --top-level

# 3. Compare. Reproducible-build commitment: byte-identical.
diff bom.json ../tensor-wasm-cdx-v1.0.0.json && echo "MATCH" || echo "MISMATCH"
```

If you get `MISMATCH`, the most common causes are: a different
`cargo-cyclonedx` patch version (the `~0.5` constraint allows 0.5.x
patch drift — pin the exact same patch CI used, visible in the release
workflow log), or a non-empty local `Cargo.lock` modification.

---

## How CI generates and attaches it

The workflow lives at
[`.github/workflows/sbom.yml`](../.github/workflows/sbom.yml). Summary:

| Trigger | Behaviour |
|---|---|
| `release: { types: [published] }` | Generate SBOM, validate JSON, upload as workflow artifact, attach as release asset via `softprops/action-gh-release@v2` |
| `push` to `dev` | Generate SBOM, validate JSON, upload as workflow artifact only (no release attachment) |
| `workflow_dispatch` | Same as `dev` push — for manual verification of the workflow itself |

The `dev`-push path exists so we catch SBOM-generation regressions
(a malformed `Cargo.toml`, a `cargo-cyclonedx` bug surfaced by a new
dependency) **before** a release tag is cut, not during the release.
If `sbom.yml` is red on `dev`, the next release is blocked.

The workflow does not need GPU runners, the CUDA toolkit, or the
nightly toolchain — `cargo-cyclonedx` operates on `Cargo.lock` and the
crate metadata, not on compiled code. It runs on `ubuntu-latest` with
stable Rust.

---

## Related

- [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md) — the
  reproducibility recipe and the SLSA L3 status table; the SBOM is one
  half of the supply-chain story documented there.
- [`PATH-TO-V1.md`](PATH-TO-V1.md) — the v1.0 milestone gate
  ("SBOM (CycloneDX) attached to every release artifact") this
  document satisfies.
- [`SECURITY.md`](../SECURITY.md) — CVE-disclosure process; an SBOM
  is what makes "is TensorWasm affected by RUSTSEC-XXXX-NNNN?"
  answerable without re-running cargo on every advisory.
- [`CUDA-SETUP.md`](CUDA-SETUP.md) — the system-library piece the SBOM
  explicitly does *not* cover.
- [`.github/workflows/sbom.yml`](../.github/workflows/sbom.yml) — the
  generation workflow itself.
- [CycloneDX specification](https://cyclonedx.org/specification/overview/)
  — upstream format reference.
- [`cargo-cyclonedx`](https://crates.io/crates/cargo-cyclonedx) — the
  tool we use; the README documents flags and output behaviour beyond
  what this doc covers.

---

_Status: living document. The contract section is binding for v1.0+.
If you find a release without an SBOM, that is a security-class bug
— file an issue and the maintainers will treat it as a release
regression._
