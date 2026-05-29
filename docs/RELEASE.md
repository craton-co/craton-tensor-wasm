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

## Supply-chain attestations

- **SBOM (implemented).** Every published release ships a CycloneDX
  JSON SBOM, generated from the release commit's `Cargo.lock` and
  attached to the GitHub Release as an asset by the
  [`.github/workflows/sbom.yml`](../.github/workflows/sbom.yml)
  workflow. See [`docs/SBOM.md`](SBOM.md) for the contract, filename
  convention, and verification steps.
- **Artifact signing (planned / not yet implemented).** Release
  artifacts currently ship with SHA256 checksums (`.sha256`) only.
  Cosign keyless signing of binaries (and the SBOM) is planned for the
  SLSA L3 milestone; see [`docs/REPRODUCIBLE-BUILDS.md`](REPRODUCIBLE-BUILDS.md)
  and [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) for status. It is not yet
  wired into `release.yml`.
