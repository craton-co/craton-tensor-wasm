# Release Engineering Runbook

> Process for tagging a TensorWasm release. Owner: @craton-co/release.

## Preconditions
- [ ] CI green on `dev` for ≥ 24h
- [ ] `cargo deny check sources advisories` clean
- [ ] `cargo audit` clean
- [ ] CHANGELOG `[Unreleased]` section finalised
- [ ] Version pins in `Cargo.toml`, `CITATION.cff` match the planned tag

## Release sequence
1. `git checkout -b release/vX.Y.Z dev`
2. Bump `workspace.package.version` and the 9 internal dep `version = "X.Y.Z"` entries in workspace Cargo.toml.
3. Update `CITATION.cff` (version + date-released).
4. Move CHANGELOG `[Unreleased]` content under `[X.Y.Z] - YYYY-MM-DD`.
5. PR `release/vX.Y.Z` → `dev` (CODEOWNERS gates).
6. After merge, tag `git tag -s vX.Y.Z -m "TensorWasm vX.Y.Z"`.
7. `git push origin dev vX.Y.Z` — release.yml workflow runs publish-dry-run + binary release + actual publish.

## Publish order
core → artifacts → tenant → jit → mem → wasi-gpu → snapshot → exec → api
(rationale: dependency topology — `tenant` precedes `mem` because `mem`
depends on `tensor-wasm-tenant`, and `artifacts` precedes both `jit` and
`snapshot` which depend on it; verify with
`cargo tree -e normal -p tensor-wasm-api`)

## Post-release
- [ ] Verify crates.io listings include LICENSE, README.
- [ ] Verify docs.rs build succeeds for each crate.
- [ ] Verify GitHub Release attaches all three platform binaries + SHA256.
- [ ] Bump `[Unreleased]` heading in CHANGELOG for next cycle.

## Security advisory release path
See `docs/runbooks/cve-disclosure-dry-run.md` for the embargoed-CVE flow. RC/patch releases follow the same sequence on a private fork until disclosure.

TODO: cosign signing of release artifacts (currently SHA256 only).
TODO: SBOM attachment to GitHub Release.
