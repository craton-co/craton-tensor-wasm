<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — Reproducible Builds

_Status: living document. First written for the v1.0 PATH-TO-V1 gate
(see [`PATH-TO-V1.md`](PATH-TO-V1.md#v100-rc1--v100), v1.0 exit
criterion: "Reproducible builds documented. A reader can rebuild a
TensorWasm v1.0 artifact from source and get bit-identical output
(modulo timestamps).")._

This document is the recipe. If you follow it on Linux x86_64 with the
pinned toolchain and a clean source tree, two independent builds of a
TensorWasm release tag will produce binaries with identical sha256
digests. On Windows MSVC the story is messier but still achievable
since rustc 1.69; see [Known limitations](#known-limitations).

For the prerequisites and the regular (non-reproducibility-focused)
build matrix see [`BUILD.md`](BUILD.md). For release engineering, key
ownership, and what artifacts a release actually consists of, see
[`GOVERNANCE.md`](../GOVERNANCE.md) and the release workflow in
[`.github/workflows/release.yml`](../.github/workflows/release.yml).

## Contents

1. [Scope and the v1.0 commitment](#scope-and-the-v10-commitment)
2. [What's deterministic out of the box](#whats-deterministic-out-of-the-box)
3. [Sources of non-determinism we eliminate](#sources-of-non-determinism-we-eliminate)
4. [The reproducibility recipe](#the-reproducibility-recipe)
5. [Verification](#verification)
6. [Container builds](#container-builds)
7. [Known limitations](#known-limitations)
8. [Supply-chain attestation (SBOM and SLSA)](#supply-chain-attestation-sbom-and-slsa)
9. [CI integration](#ci-integration)
10. [Related](#related)

---

## Scope and the v1.0 commitment

"Bit-identical modulo timestamps" means: given the same git commit,
the same pinned toolchain, the same target triple, and the recipe in
[Section 4](#the-reproducibility-recipe), two independent builders on
two independent machines produce a binary whose sha256 matches. The
"modulo timestamps" carve-out covers timestamps that the user
intentionally injects (release metadata, container image creation
times) but not timestamps the toolchain bakes in by default — those
are nailed down by `SOURCE_DATE_EPOCH`.

The commitment **applies to**: the `tensor-wasm` CLI binary built from
`crates/tensor-wasm-cli`; each library `.rlib` under
`target/<triple>/release/deps/`; and the release tarballs/zips produced
by [`release.yml`](../.github/workflows/release.yml) **after** their
internal mtimes have been normalised (see
[Sources of non-determinism](#sources-of-non-determinism-we-eliminate)).

It does **not** apply to: rustdoc HTML output (timestamps baked into
the generated pages); local-developer incremental builds (the recipe
disables incremental); benchmark artifacts under `bench-out/`; or
anything built with `cargo install` from a non-pinned toolchain.

If we miss the commitment on a v1.x release, that is a release-blocker
severity bug and gets a CVE-class postmortem in
[`docs/SECURITY-AUDIT.md`](SECURITY-AUDIT.md). The point of the
commitment is to let downstream packagers (Debian, Nixpkgs, Homebrew),
distroless container builders, and security auditors verify that the
binary they got from the GitHub release matches the source tree they
inspected.

---

## What's deterministic out of the box

A surprising amount, on modern rustc. Given a pinned
`rust-toolchain.toml` (we pin `nightly-2026-04-03`), a committed
`Cargo.lock` that fixes the entire transitive dependency graph
(4,650 lines as of this writing), and `cargo build --locked`, the
following are deterministic without any extra effort:

- **Symbol mangling** — content-hash based, not name-based.
- **Codegen output for a single codegen unit.** Our `[profile.release]`
  in [`Cargo.toml`](../Cargo.toml) sets `codegen-units = 1`, which both
  improves LLVM optimization and removes inter-build variance from
  parallel codegen.
- **Compiler-internal HashMap iteration** — rustc uses `FxHashMap`
  (user-code `std::collections::HashMap` at runtime is *not* covered).
- **LTO output.** `lto = "thin"` has been deterministic since LLVM 11.
- **Debug info content.** With `debug = 1`, only line tables are
  emitted, and those are reproducible.

What is **not** deterministic without intervention: absolute paths
embedded in panic strings, debug info, and `file!()`; file timestamps
inside tarballs or zip archives; the PE timestamp in Windows
`.exe`/`.dll` files; and the ELF `NT_GNU_BUILD_ID` note on platforms
where the linker generates a random one. The recipe in
[Section 4](#the-reproducibility-recipe) addresses every item.

---

## Sources of non-determinism we eliminate

Each item below is something a vanilla `cargo build --release` would
get wrong, and the mitigation we apply.

### Timestamps in archive formats

`tar` and `zip` both embed file mtimes. The `release-binary` job in
[`release.yml`](../.github/workflows/release.yml) currently uses
plain `tar -czf` and `Compress-Archive`, which means the archives are
**not** reproducible today even though the inner binary is. The
mitigation is `SOURCE_DATE_EPOCH` plus, for tar specifically,
`--mtime=@$SOURCE_DATE_EPOCH --sort=name --owner=0 --group=0
--numeric-owner --pax-option=exthdr.name=%d/PaxHeaders/%f`. For zip,
use `zip -X` (omit extra file attributes) plus pre-touching every file
with `touch -d @$SOURCE_DATE_EPOCH`.

We have not yet plumbed this into the release workflow. Tracked as a
v1.0-rc1 follow-up — see [CI integration](#ci-integration) below.

### Build-path embedding

`file!()` macro calls, panic backtraces, and debug info all embed the
absolute path of the source file. On builder A this is
`/home/alice/work/tensor-wasm/crates/.../foo.rs`; on builder B,
`/builds/ci-12345/src/.../foo.rs`. They will not match.

Mitigation: rustc accepts `--remap-path-prefix=FROM=TO` (multiple
times). We remap two paths to fixed placeholders: the source tree
(to `/build/tensor-wasm`) and the cargo cache (to `/cargo`).

### Random ordering

Two historical suspects: HashMap iteration in proc-macros and
build-scripts (Rust's `std::collections::HashMap` uses a randomized
hasher; we don't believe any of our proc-macros iterate `HashMap`
into emitted tokens, but the audit-trail mitigation is
`RUSTC_BOOTSTRAP=1 RUST_HASHMAP_ENABLE_DETERMINISM=1`), and parallel
codegen ordering (already neutralised by `codegen-units = 1`; if a
future profile relaxes this, also pass `-Zthreads=1`).

### System time / locale / username

A handful of crates pick up `$USER`, `$HOSTNAME`, or `tzdata`. None of
our direct or transitive deps do as of v1.0; we grep for `env!("USER")`
and `env!("HOSTNAME")` on every dependency update (cargo-deny config in
`deny.toml`). `chrono` is pulled with `default-features = false` so
no TZ-database lookup happens at build time.

### Git-pinned sources

Two crates enter the workspace via `git = ...` pins rather than from
crates.io, both as of v0.3.1 (per [RFC 0001](../rfcs/0001-cuda-oxide-integration.md)).
Both pins are explicit revs (NOT branches), so a `cargo update` cannot
silently flip them and `cargo deny check sources` audits the URL +
rev pair on every CI run:

| Crate(s) | Repository | Pinned rev | Why git, not crates.io |
|---|---|---|---|
| `cuda-host`, `cuda-core`, `cuda-async` | `https://github.com/NVlabs/cuda-oxide` | `4a56e4220aab8ce5d085a411e7f806cebb647d14` (v0.1.0 tag) | NVlabs has not yet published these workspace members to crates.io; the crates.io `cuda-oxide` name is a different, unrelated 2018-era project. Re-evaluated at v0.4 per the RFC. |
| `pliron`, `pliron-derive` (transitive via cuda-oxide) | `https://github.com/vaivaswatha/pliron` | `b51e73b11648508188184451adebdcf63957b7fe` | Pliron is cuda-oxide's MLIR-like Rust-native IR framework; not yet on crates.io as a stable release. Mirrored from cuda-oxide's own `Cargo.toml` pin. |

`deny.toml` carries one `allow-git` entry per repository URL above with
a comment matching the table. When cuda-oxide bumps its Pliron pin in
a future release, the workspace `Cargo.toml` cuda-oxide rev AND the
`deny.toml` allowlist comment for `vaivaswatha/pliron` must be updated
in the same commit so the SBOM (`tensor-wasm-cdx-v<version>.json`) and
this doc stay in sync.

#### Policy: how a git-pinned dep gets in (and out)

The `cuda-oxide-backend` feature on `tensor-wasm-mem` is the first
TensorWasm feature whose enablement pulls a git-pinned dependency into
the resolved graph. Pliron — cuda-oxide's MLIR-like Rust-native IR
framework — is currently a `git`-pinned transitive dependency of
cuda-oxide because it has not yet published a stable release to
crates.io. The cuda-oxide-backend feature in v0.3.1 is dep-less (an
empty scaffold module per [RFC 0001](../rfcs/0001-cuda-oxide-integration.md)
"Rollout"); the actual git pin lands in the v0.4 parity work. This
section documents the policy so reviewers of that v0.4 PR know what
to look for.

A git pin is acceptable in this workspace **only if all four of these
hold**:

1. **The pin is an explicit revision (a 40-char commit SHA), not a
   branch or tag-name.** Branch pins can silently move; tag pins are
   safer but can still be force-pushed by an upstream that does not
   treat tags as immutable. A SHA cannot move. `cargo update` may not
   change a git revision pinned by SHA — that is what makes the pin
   reproducible. The resolved revision is captured in `Cargo.lock` for
   `--locked` builds.
2. **There is a comment in `Cargo.toml` directly above the dependency
   line linking to [RFC 0001](../rfcs/0001-cuda-oxide-integration.md)
   (or the successor RFC that justifies the pin) and naming the
   condition under which the pin is removed.** For Pliron, the removal
   condition is "Pliron publishes a stable release to crates.io". The
   comment is the contract; without it, a future cargo-update PR can
   re-pin to a newer SHA without an RFC discussion.
3. **There is a matching `allow-git` entry in `deny.toml` (under
   `[sources]`) for the repository URL.** `cargo deny check sources`
   audits this on every CI run via the `deny` job in
   [`.github/workflows/ci.yml`](../.github/workflows/ci.yml); if the
   workspace ever picks up an un-allowlisted git source the build
   fails. The `deny.toml` comment must match the `Cargo.toml` comment
   (same RFC link, same removal condition).
4. **The SBOM (`tensor-wasm-cdx-v<version>.json`, see
   [`SBOM.md`](SBOM.md)) records the resolved git URL + rev as the
   `purl` for the pinned crate.** `cargo-cyclonedx` does this by
   default for `git` sources; the check is that the SBOM contains a
   `pkg:cargo/...?vcs_url=git+https://...@<sha>` purl line for the
   pinned crate, not a bare `pkg:cargo/<name>@<version>` line that
   would imply a crates.io source.

When all four hold, the build is as reproducible as a crates.io
pin: two builders cloning the same SHA, running the same
`cargo build --locked` recipe (see
[Section 4](#the-reproducibility-recipe)), get bit-identical output.
The git protocol itself does not add non-determinism — cargo's
on-disk layout for git sources is content-addressed by SHA.

Reviewer checklist when a PR adds (or bumps) a git pin:

- [ ] `Cargo.toml` has a comment above the dep linking to the
      justifying RFC and naming the removal condition.
- [ ] `deny.toml` has a matching `allow-git` row with the same RFC
      link and removal condition in its comment.
- [ ] The pin is a 40-char SHA, not a branch or tag-name.
- [ ] `Cargo.lock` was regenerated in the same commit; the resolved
      revision matches.
- [ ] If this is a bump (not an add), the SBOM purl for the crate
      was regenerated and committed.
- [ ] The release notes / CHANGELOG entry calls out the bump.

When Pliron publishes to crates.io, the git pin in cuda-oxide's
`Cargo.toml` will be replaced with a normal version requirement; the
transitive git source for `vaivaswatha/pliron` will drop out of the
workspace's resolved graph; the `allow-git` row in `deny.toml` and the
table row above can be deleted; and this subsection (kept until then
as a forward-looking policy statement) can be reduced back to the
table-only treatment that the NVlabs/cuda-oxide pin gets today. The
open RFC question that tracks this is
[RFC 0001 "Unresolved questions"](../rfcs/0001-cuda-oxide-integration.md#unresolved-questions)
— "How does Pliron pin to a stable release vs git?".

### ELF `NT_GNU_BUILD_ID`

The linker can be configured to compute the build-ID from a hash of
the output bytes (`--build-id=sha1`) rather than from random bytes.
GNU ld and lld both support this; rustc passes the right flag when
`-C link-arg=-Wl,--build-id=sha1` is set. We add this to `RUSTFLAGS`
in the recipe below.

---

## The reproducibility recipe

The full recipe for a Linux x86_64 reproducible build of the
`tensor-wasm` CLI binary:

```sh
# 1. Clone at a tagged release. Shallow clones are fine; the recipe
#    does not depend on full git history.
git clone --depth 1 --branch v1.0.0 https://github.com/craton-co/craton-tensor-wasm
cd craton-tensor-wasm

# 2. Pick a deterministic timestamp. Use the commit timestamp of the
#    tag — this is the convention `SOURCE_DATE_EPOCH` was designed
#    for. (https://reproducible-builds.org/specs/source-date-epoch/)
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

# 3. Set RUSTFLAGS for the remappings and the deterministic build-ID.
#    The two --remap-path-prefix entries are critical: one for the
#    source tree, one for the cargo registry cache.
export RUSTFLAGS="\
  --remap-path-prefix=$(pwd)=/build/tensor-wasm \
  --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo \
  -C link-arg=-Wl,--build-id=sha1"

# 4. Disable incremental compilation. Incremental builds intentionally
#    embed cache-key metadata that varies between builds.
export CARGO_INCREMENTAL=0

# 5. Build with --locked. This is the load-bearing flag: it forces
#    cargo to refuse if Cargo.lock would have to change. Without
#    --locked, a new minor version of a dependency that appeared
#    between commit and build could silently land in your binary.
cargo build --workspace --release --locked --target x86_64-unknown-linux-gnu

# 6. The reproducible artifact:
ls -l target/x86_64-unknown-linux-gnu/release/tensor-wasm
sha256sum target/x86_64-unknown-linux-gnu/release/tensor-wasm
```

The pinned Rust toolchain (`rust-toolchain.toml` → `nightly-2026-04-03`)
is picked up automatically by `rustup` the first time `cargo` is invoked
in the workspace. You do not need to install it explicitly.

For Windows MSVC, the recipe is the same except for the path syntax and
the `--build-id` flag, which is ELF-only and ignored on PE. Use PowerShell:

```powershell
$env:SOURCE_DATE_EPOCH = (git log -1 --pretty=%ct)
$cwd = (Get-Location).Path
$env:RUSTFLAGS = "--remap-path-prefix=$cwd=/build/tensor-wasm --remap-path-prefix=$env:USERPROFILE\.cargo=/cargo"
$env:CARGO_INCREMENTAL = "0"
cargo build --workspace --release --locked --target x86_64-pc-windows-msvc
```

rustc 1.69 and later pass `/Brepro` to `link.exe` automatically when
`SOURCE_DATE_EPOCH` is set, which clears the PE timestamp field. See
the [Known limitations](#known-limitations) section for the caveats.

---

## Verification

The verification protocol is: build twice in independent scratch
directories, compare digests. Run the recipe in
[Section 4](#the-reproducibility-recipe) twice — once under
`/tmp/repro-a/src`, once under `/tmp/repro-b/src` — then:

```sh
sha256sum /tmp/repro-a/src/target/release/tensor-wasm > /tmp/sha-a
sha256sum /tmp/repro-b/src/target/release/tensor-wasm > /tmp/sha-b
diff <(awk '{print $1}' /tmp/sha-a) <(awk '{print $1}' /tmp/sha-b) \
  && echo "REPRODUCIBLE" || echo "MISMATCH"
```

Expected output:

```text
REPRODUCIBLE
```

If you get `MISMATCH`, the right next step is `diffoscope`, which
recursively decomposes both files and tells you exactly which bytes
differ:

```sh
diffoscope /tmp/repro-a/src/target/release/tensor-wasm \
           /tmp/repro-b/src/target/release/tensor-wasm \
           --html /tmp/diff.html
```

Common findings and what they mean:

| diffoscope output mentions | Cause | Fix |
|---|---|---|
| `.comment` section differs | Different LLVM/rustc patch versions | Re-check `rustc --version`; pin matches `rust-toolchain.toml` |
| Absolute paths in `.debug_str` | `--remap-path-prefix` missing or wrong | Re-read the recipe; both remaps must be present |
| `.note.gnu.build-id` differs | `--build-id=sha1` not applied | Check `RUSTFLAGS` made it through (`cargo build -vv`) |
| PE `IMAGE_FILE_HEADER.TimeDateStamp` differs | `SOURCE_DATE_EPOCH` not set or pre-1.69 toolchain | Update toolchain; verify env var is exported |
| Random bytes throughout | `CARGO_INCREMENTAL=0` missing | Set it; `cargo clean` and retry |

---

## Container builds

A reference Dockerfile that produces a reproducible `tensor-wasm`
binary. The image is multi-stage: a builder image with the pinned
toolchain, a distroless runtime image with only the binary.

```dockerfile
# syntax=docker/dockerfile:1.6

# ----- builder -----
FROM rust:1-slim-bookworm AS builder

# Pin the toolchain. The COPY of rust-toolchain.toml below triggers
# rustup to install nightly-2026-04-03 on first cargo invocation.
WORKDIR /build/tensor-wasm
COPY rust-toolchain.toml .
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY wit ./wit

# SOURCE_DATE_EPOCH is provided as a build argument by the caller.
# Use the commit timestamp:
#   docker build --build-arg SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct) ...
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="--remap-path-prefix=/build/tensor-wasm=/build/tensor-wasm --remap-path-prefix=/usr/local/cargo=/cargo -C link-arg=-Wl,--build-id=sha1"

RUN cargo build -p tensor-wasm-cli --release --locked --target x86_64-unknown-linux-gnu

# ----- runtime -----
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /build/tensor-wasm/target/x86_64-unknown-linux-gnu/release/tensor-wasm /usr/local/bin/tensor-wasm
USER nonroot
ENTRYPOINT ["/usr/local/bin/tensor-wasm"]
```

Build it with:

```sh
docker build --build-arg SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct) \
  -t tensor-wasm:v1.0.0 .
```

For a bit-identical *image* (not just binary), use `buildkit` with
`--output type=docker,...,dest=image.tar` and strip the manifest
timestamp; the full protocol is in the
[reproducible-builds.org Docker guide](https://reproducible-builds.org/docs/source-date-epoch/).

---

## Known limitations

The honest list of what this recipe does **not** achieve today.

### Windows MSVC PE timestamps

rustc 1.69+ does pass `/Brepro` to `link.exe` when `SOURCE_DATE_EPOCH`
is set, and this zeroes the PE header timestamp. In practice, we have
seen two cases where Windows binaries still differ:

1. **Embedded resources.** If `embed_manifest` or a `.rc` build script
   runs `rc.exe`, the resource section can pick up a build-time
   timestamp. We do not currently embed manifests; if that changes,
   we will need to pass `/n` to `rc.exe`.
2. **Linker version stamping.** Older MSVC linkers stamp the linker
   version into the PE header. The `/Brepro` flag is supposed to
   suppress this; we have seen at least one report against
   `link.exe 14.39` where it did not. Workaround: post-process the PE
   header with `pelf` or a custom tool to zero the offending field.

Treat Windows reproducibility as **best-effort** for v1.0. The Linux
build is the one we make a hard commitment on.

### Rustdoc HTML

`cargo doc` output is not reproducible. The generated HTML embeds:

- The exact rustdoc version in the page footer
- A timestamp in the `<meta>` generator tag
- Re-randomised CSS hashes on each build (rustdoc 1.78+ has an open
  upstream issue to fix this)

We make no reproducibility claim for the API reference docs. If you
need a stable docs hash for archival purposes, render the rustdoc
HTML once at release time and treat the rendered tarball as the
canonical artifact.

### Cargo's local registry cache

`~/.cargo/registry/cache` differs by user and by which registries
have been fetched. It is **not** part of the artifact reproducibility
commitment — the binary that comes out of `cargo build` is, but the
intermediate files in your cargo cache are not expected to match
between machines.

### Archive formats (tar, zip)

As noted in [Sources of non-determinism](#sources-of-non-determinism-we-eliminate),
the current release workflow does not yet normalise tarball/zip mtimes.
This means the **binary inside** the release archive is reproducible
under v1.0, but the archive itself is not. Tracked as a v1.0-rc1
follow-up.

---

## Supply-chain attestation (SBOM and SLSA)

Reproducibility is one half of the supply-chain story; an SBOM
(Software Bill of Materials) is the other. The PATH-TO-V1 workstream
calls out [SLSA Level 3](https://slsa.dev/spec/v1.0/levels) as the v1.0
target.

### SBOM generation

We generate a CycloneDX SBOM for every release and attach it to the
GitHub Release page alongside the binaries. The contract, the
file-naming convention (`tensor-wasm-cdx-v<version>.json`), the
"what's in it / what's not" boundaries, and the local-regeneration
recipe all live in [`SBOM.md`](SBOM.md) — that document is the
authoritative reference.

In short, the generation step is:

```sh
cargo install cargo-cyclonedx --version "~0.5" --locked
cargo cyclonedx --format json --output-pattern bom --top-level
```

and CI in [`.github/workflows/sbom.yml`](../.github/workflows/sbom.yml)
runs the same command on every release tag (and on pushes to `dev`
for verification). The pinned `cargo-cyclonedx` version is what makes
the output reproducible in the same sense the rest of this document
defines.

### SLSA Level 3 target

The v1.0 target from [`PATH-TO-V1.md`](PATH-TO-V1.md) is SLSA Level 3,
which requires:

| SLSA requirement | Status as of v1.0-rc1 |
|---|---|
| Build is scripted | Done — `.github/workflows/release.yml` |
| Provenance is generated | In progress — `slsa-github-generator` v2 |
| Provenance is signed | In progress — cosign keyless via GitHub OIDC |
| Build is reproducible | This document |
| Build is isolated | Done — GitHub-hosted runners, no self-hosted secrets |
| Source is two-person-reviewed | Done — branch-protection rule on `main` |

The two "in progress" items are tracked as v1.0-rc1 release-blockers.

---

## CI integration

The reproducibility commitment is only meaningful if CI proves it on
every release. A sketch of a GitHub Actions job that does this:

```yaml
# .github/workflows/reproducibility.yml
name: reproducibility-check
on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:

jobs:
  double-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: nightly-2026-04-03

      - name: Build A
        run: |
          export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
          export RUSTFLAGS="--remap-path-prefix=$(pwd)=/build/tensor-wasm \
            --remap-path-prefix=$HOME/.cargo=/cargo \
            -C link-arg=-Wl,--build-id=sha1"
          export CARGO_INCREMENTAL=0
          cargo build -p tensor-wasm-cli --release --locked --target-dir target-a

      - name: Build B (clean target dir)
        run: |
          export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
          export RUSTFLAGS="--remap-path-prefix=$(pwd)=/build/tensor-wasm \
            --remap-path-prefix=$HOME/.cargo=/cargo \
            -C link-arg=-Wl,--build-id=sha1"
          export CARGO_INCREMENTAL=0
          cargo build -p tensor-wasm-cli --release --locked --target-dir target-b

      - name: Compare digests
        run: |
          SHA_A=$(sha256sum target-a/release/tensor-wasm | awk '{print $1}')
          SHA_B=$(sha256sum target-b/release/tensor-wasm | awk '{print $1}')
          echo "A: $SHA_A"
          echo "B: $SHA_B"
          if [ "$SHA_A" != "$SHA_B" ]; then
            echo "::error::Reproducibility check FAILED"
            sudo apt-get install -y diffoscope
            diffoscope target-a/release/tensor-wasm target-b/release/tensor-wasm || true
            exit 1
          fi
          echo "REPRODUCIBLE: $SHA_A"
```

This is **not yet wired** into the release pipeline. Adding it is
tracked as a v1.0-rc1 follow-up, alongside the archive-mtime
normalisation noted above.

---

## Related

- [`BUILD.md`](BUILD.md) — the standard (non-reproducibility-focused)
  build matrix and feature flag reference
- [`SBOM.md`](SBOM.md) — the CycloneDX SBOM contract, filename
  convention, and local-regeneration recipe; the other half of the
  supply-chain story
- [`PATH-TO-V1.md`](PATH-TO-V1.md) — the v1.0 gate that this document
  satisfies
- [`GOVERNANCE.md`](../GOVERNANCE.md) — release engineering, key
  ownership, and the maintainer-side process for cutting a release
- [`SECURITY.md`](../SECURITY.md) — disclosure process; failures of
  the reproducibility commitment after v1.0 are security-class bugs
- [`.github/workflows/release.yml`](../.github/workflows/release.yml)
  — the current release pipeline (which this document will extend)
- [`CUDARC-SPIKE.md`](CUDARC-SPIKE.md) — context on the dependency
  migration and how lockfile pinning protects reproducibility across
  it
- [reproducible-builds.org](https://reproducible-builds.org/) — the
  upstream project; the `SOURCE_DATE_EPOCH` spec, the diffoscope tool,
  and the broader Linux-distro reproducibility ecosystem live there

---

_Status: living document. The recipe is correct for v1.0-rc1 as
of the last revision. If you encounter a reproducibility failure on
the pinned toolchain following the recipe exactly, file an issue with
the diffoscope output attached — that is the kind of regression we
treat as a release-blocker._
