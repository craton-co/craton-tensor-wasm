<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Craton TensorWasm — API Reference Publication Policy

_Status: living document. Tracks the v0.4 milestone item
"API reference auto-generated from rustdoc + OpenAPI, published per
release" from [`PATH-TO-V1.md`](PATH-TO-V1.md) (workstream
"Documentation"). Companion to [`SBOM.md`](SBOM.md), which describes
the parallel release-asset pipeline._

TensorWasm publishes two machine-generated reference surfaces alongside
every release: the Rust API rustdoc tree and a rendered HTML view of the
HTTP API's OpenAPI 3.1 spec. Both are produced by
[`.github/workflows/api-reference.yml`](../.github/workflows/api-reference.yml)
and shipped as a single static-site archive — a workflow artifact on
every dev-branch push, a release asset on every published tag. This
document explains what is in that archive, what is not, the URL
convention by which we expect it to be served, and how to regenerate the
inputs locally.

If you are integrating against TensorWasm: read the published reference,
not the source. If you are auditing a release: the archive is a frozen
snapshot of the reference at the tag, recomputable from source.

## Contents

1. [The pipeline at a glance](#the-pipeline-at-a-glance)
2. [Publication URL convention](#publication-url-convention)
3. [What is in the bundle](#what-is-in-the-bundle)
4. [What is NOT in the bundle](#what-is-not-in-the-bundle)
5. [Rendered HTML vs raw YAML](#rendered-html-vs-raw-yaml)
6. [Local regeneration](#local-regeneration)
7. [Versioning policy](#versioning-policy)
8. [Caveats and known limitations](#caveats-and-known-limitations)
9. [Related](#related)

---

## The pipeline at a glance

`.github/workflows/api-reference.yml` runs three jobs:

| Job             | Input                                  | Output                              |
|-----------------|----------------------------------------|-------------------------------------|
| `rustdoc`       | workspace `Cargo.toml` + crate sources | `target/doc/` HTML tree             |
| `openapi-html`  | `openapi/tensor-wasm-api.yaml`         | `site/api/index.html` (single file) |
| `bundle`        | both of the above                      | `tensor-wasm-api-reference-v<X>.zip` |

The `rustdoc` job runs `cargo doc --no-deps --workspace --release` with
`RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links"`. Any
warning — missing summary line, malformed example, dangling intra-doc
link — fails the build. This is intentional: the rustdoc gate in
`ci.yml` already runs on every PR, so a release-time failure means
something slipped through; the release-time job is the belt to that
suspenders.

The `openapi-html` job runs `redocly build-docs` to render the OpenAPI
3.1 YAML into a single self-contained HTML file with all CSS/JS inlined.
No CDN dependency at view time; the file works fully offline.

The `bundle` job downloads both artifacts, emits a one-screen landing
page (`site/index.html`) linking to each, and zips the whole tree as
`tensor-wasm-api-reference-v<VERSION>.zip`. On `release: published`
events the archive is attached to the GitHub Release via
[`softprops/action-gh-release@v2`](https://github.com/softprops/action-gh-release)
— the same action `sbom.yml` uses for the CycloneDX SBOM.

Triggers:

- `release: { types: [published] }` — the authoritative path; attaches
  the archive to the release page.
- `push: { branches: [dev] }` — preview build for the integration
  branch; uploaded as a workflow artifact only.
- `workflow_dispatch` — manual recompute (e.g. after a docs-only fix on
  a release branch that doesn't warrant a new tag).

## Publication URL convention

The archive is the source of truth. When (and if) GitHub Pages is
enabled for the repository, the contents are expected to be served at:

```
https://craton-co.github.io/craton-tensor-wasm/<version>/rustdoc/
https://craton-co.github.io/craton-tensor-wasm/<version>/api/
```

with `<version>` matching the tag-without-leading-`v` (`0.1.0`,
`0.2.0-beta.1`, etc., consistent with `tensor-wasm-cdx-v<VERSION>.json`
in `SBOM.md`). A `latest/` path is expected to redirect to the most
recent stable release (no pre-release tags). A top-level
`https://craton-co.github.io/craton-tensor-wasm/` index page lists every
version present.

**Caveat.** GitHub Pages is **not** currently enabled on the repository,
and the URLs above are aspirational. The static-site archive is
nonetheless published as a release asset today, so consumers can
download and self-host it. When Pages is enabled (or an alternative
host is chosen), this section is updated with the live URLs and a
companion `gh-pages` deploy job is added to the workflow. Tracking
that decision is left to the maintainer sync — it is not a
v0.4 exit-criterion blocker.

If you need to serve the bundle yourself, unzip it under any static
file root; all internal links are relative.

## What is in the bundle

### `rustdoc/`

The HTML rustdoc tree for every public item in every workspace crate
listed in the root `Cargo.toml` `[workspace] members`:

- [`tensor-wasm-core`](../crates/tensor-wasm-core) — core types
  (errors, results, IDs, tenant model).
- [`tensor-wasm-mem`](../crates/tensor-wasm-mem) — memory traits, host
  allocator, optional unified-memory backend.
- [`tensor-wasm-exec`](../crates/tensor-wasm-exec) — Wasmtime wrapper:
  engines, stores, instantiation, dispatch.
- [`tensor-wasm-snapshot`](../crates/tensor-wasm-snapshot) — streaming
  zstd + bincode snapshot capture / restore.
- [`tensor-wasm-api`](../crates/tensor-wasm-api) — axum HTTP gateway,
  request / response types, middleware.
- [`tensor-wasm-wasi-gpu`](../crates/tensor-wasm-wasi-gpu) — WASI-GPU
  host-function surface.
- [`tensor-wasm-jit`](../crates/tensor-wasm-jit) — auto-offload JIT
  blueprints and pattern matchers.
- [`tensor-wasm-tenant`](../crates/tensor-wasm-tenant) — tenant
  registry, quota gate, MPS coordination.

The `--no-deps` flag keeps the tree small and stable: only TensorWasm's
own surface ships, not every transitive crate's docs.

### `api/index.html`

The rendered HTML view of [`openapi/tensor-wasm-api.yaml`](../openapi/tensor-wasm-api.yaml).
Single self-contained file: every route, every schema, every example
the spec carries, with collapsible nav, request/response samples, and
search.

### `index.html`

A one-screen landing page linking to the two surfaces above and naming
the version. Generated by the `bundle` job from an inline heredoc; if
it grows beyond a screen, it is promoted to `templates/index.html.tmpl`
and substituted at build time.

## What is NOT in the bundle

- **Private items.** `cargo doc` is invoked without `--document-private-items`.
  Anything not `pub` is intentionally omitted; the bundle reflects the
  semver-stable surface, not internal implementation details.
- **CLI internals.** `tensor-wasm-cli` is a binary crate; its rustdoc
  is not part of the bundle because its public API is the command
  line, documented in [`CLI.md`](CLI.md) and the man pages under
  `crates/tensor-wasm-cli/man/`.
- **Bench harness.** `tensor-wasm-bench` is internal tooling; its types
  are not part of the supported surface.
- **Fuzz targets.** Excluded from the workspace at the manifest level
  (`exclude = ["fuzz"]`); never reached by `cargo doc --workspace`.
- **Transitive dependency docs.** `--no-deps` keeps the archive to
  TensorWasm-authored documentation only. Look up dependency docs
  on [docs.rs](https://docs.rs) — they are versioned there.
- **WIT-derived bindings docs.** The `wit/wasi-cuda.wit` interface is
  documented in-tree (`docs/WASM-DEVELOPER-GUIDE.md`); a future
  workstream may add a wit-bindgen render to this bundle, but it is
  not part of v0.4 scope.
- **Markdown docs from `docs/`.** This document, `PATH-TO-V1.md`,
  `BENCHMARKING.md`, etc. live in the source tree and on the repository
  website. They are not part of the per-release reference bundle; they
  evolve continuously and are not pinned to a tag.

## Rendered HTML vs raw YAML

The OpenAPI YAML at `openapi/tensor-wasm-api.yaml` is the canonical
machine-readable contract; the rendered HTML is a derived view. They
should never disagree — if they do, the YAML wins and the HTML is
considered stale until the next build.

What the rendered view adds:

- A browsable, searchable left-rail nav over every operation and
  schema.
- Generated request/response example payloads from the spec's
  `examples:` blocks.
- Auto-rendered "Try it" snippets (curl, fetch, several language
  clients) — view-only; the rendered HTML does not execute requests.
- Collapsed/expanded schema views that follow `$ref` resolutions
  without forcing the reader to do the JSON Pointer walk by hand.

What it does not add:

- New semantics. Any apparent behaviour visible only in the HTML and
  not the YAML is a Redocly rendering artefact and should be reported
  as a docs bug.
- Live request execution. The Redocly `build-docs` output is a
  static HTML page; there is no proxy, no `Try It Now` POST.

## Local regeneration

To rebuild what CI builds, on a machine with the pinned nightly
toolchain (see `rust-toolchain.toml`) and a recent Node:

```sh
# Rustdoc — same flags the workflow uses.
RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links" \
  cargo doc --no-deps --workspace --release

# Then open it in your browser:
cargo doc --no-deps --workspace --open

# OpenAPI HTML — no global install needed; npx pulls @redocly/cli on
# demand and caches it. The version pin matches the workflow.
npx --yes @redocly/cli@1 build-docs openapi/tensor-wasm-api.yaml \
  -o /tmp/tensor-wasm-api.html
xdg-open /tmp/tensor-wasm-api.html   # or `open` on macOS
```

To preview the assembled bundle locally:

```sh
mkdir -p site/rustdoc site/api
cp -R target/doc/. site/rustdoc/
npx --yes @redocly/cli@1 build-docs openapi/tensor-wasm-api.yaml \
  -o site/api/index.html
# Drop a minimal index.html in site/ if you want the landing page.
python3 -m http.server --directory site 8000
```

The CI bundle additionally generates the landing page from an inline
heredoc; reproducing it locally is optional.

## Versioning policy

Every published release ships a new copy of the bundle, keyed by the
version. Once published, a version's bundle is immutable: the CI job
that produced it can be re-run, but the resulting archive is expected
to be bit-identical (modulo a few non-deterministic Redocly bundle
fields). If a bug is found in a published version's docs, the fix
ships in the next release with a `CHANGELOG.md` entry; the old
version's archive is **not** rewritten in place.

This mirrors the source-code release policy in
[`PATH-TO-V1.md`](PATH-TO-V1.md): no in-place mutation of a tagged
release. If you need a fix to a historical version's docs, request a
patch release.

Beta and RC tags (`v0.4.0-beta.1`, `v1.0.0-rc1`) get their own bundles
under the same `<version>/` scheme; `latest/` always points at the most
recent stable (non-pre-release) version.

## Caveats and known limitations

- **GitHub Pages is not yet enabled.** The URL convention above is the
  intended target; today the canonical access path is the release-page
  asset. Resolving this is a maintainer-sync decision tracked
  separately from the v0.4 milestone.
- **Archive size.** rustdoc HTML for a non-trivial Rust workspace can
  run into tens of megabytes. The expected upper bound for v0.4-era
  TensorWasm is well under the GitHub Releases per-asset 2 GiB ceiling,
  so this is not a constraint today, but it is something to revisit
  if the workspace grows substantially or if `--document-private-items`
  is ever flipped on.
- **Broken-link checking is intra-crate only.** The
  `-D rustdoc::broken_intra_doc_links` gate catches `[Type]` references
  within a doc comment. Links to external URLs in `[text](https://…)`
  form are not validated; a separate link-checker pass (HTMLProofer or
  similar) is a candidate for a follow-up workstream but is out of
  scope here.
- **Nightly-pinned rustdoc.** The bundle is built with
  `nightly-2026-04-03` (the workspace's pinned toolchain). HTML output
  may differ between rustdoc versions; consumers should treat the
  archive as opaque HTML, not as a stable input format for tooling.
- **OpenAPI render is Redocly-specific.** Switching to rapidoc or
  another renderer is straightforward (one job swap), but the
  resulting HTML will look different and may break any deep links to
  specific schema anchors. The choice of Redocly is intentional —
  it matches the validator already pinned in `ci.yml`'s
  `openapi-validate` job, keeping the tool surface small.

## Related

- [`PATH-TO-V1.md`](PATH-TO-V1.md) — origin of the workstream item;
  see "Documentation" under "Per-area workstreams".
- [`SBOM.md`](SBOM.md) — sibling release-asset pipeline; same
  upload/attach pattern.
- [`REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md) — broader story
  for "the same input always produces the same release artifact".
- [`openapi/tensor-wasm-api.yaml`](../openapi/tensor-wasm-api.yaml) —
  the spec the HTML view renders.
- [`openapi/redocly.yaml`](../openapi/redocly.yaml) — the linter
  configuration; the renderer uses the same Redocly CLI version.
- [`.github/workflows/api-reference.yml`](../.github/workflows/api-reference.yml)
  — the pipeline this document describes.
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — the
  per-PR validators (`doc`, `openapi-validate`) that feed this
  release-time job.
