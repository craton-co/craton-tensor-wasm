# Contributing to Craton TensorWasm

Thanks for your interest in TensorWasm. This document describes how to get a
development environment running, the conventions we follow for changes,
and how to coordinate with the maintainers.

## Code of Conduct

This project adopts the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
By participating you agree to uphold its terms. Enforcement issues go to
`security@craton.com.ar`.

## Development setup

### 1. Prerequisites

- **Git** with a configured `user.name` and `user.email` (used for DCO).
- **Rust toolchain** — `rustup` will install the pinned nightly
  automatically the first time you run a `cargo` command in the repo.
  The version is pinned in [`rust-toolchain.toml`](rust-toolchain.toml).
- **(Optional) CUDA 12.0+** — only required for the GPU-bound feature
  flags (`unified-memory`, `cuda`, `auto-offload`, `mps`). The default
  build path works on a CUDA-free host. See
  [`docs/CUDA-SETUP.md`](docs/CUDA-SETUP.md).
- **GNU Make** — the [`Makefile`](Makefile) wraps the cargo invocations
  used by CI so that local runs mirror the workflow.

### 2. Clone and build

```sh
git clone https://github.com/craton-co/craton-tensor-wasm
cd craton-tensor-wasm
make ci
```

`make ci` runs `cargo fmt --check`, `cargo clippy -- -D warnings`,
`cargo check --workspace`, and `cargo test --workspace`. It is the exact
set of checks the GitHub Actions workflow runs on every PR; running it
locally before pushing keeps the feedback loop short.

If you only want a fast smoke test, `make build` and `make test` are
both fine starting points.

### 3. CUDA-only tests

Tests that require an NVIDIA GPU are marked `#[ignore = "requires CUDA hardware"]`
and are exercised by the self-hosted CUDA CI runner. To run them locally
on a CUDA host, use `cargo test --workspace --features unified-memory --
--ignored`.

## Making a change

### Branches

Cut a topic branch off `main` for your work. We do not use long-lived
feature branches; aim for small, reviewable PRs.

### Commit messages

We keep commit messages concise, imperative, and present-tense. Look at
recent history (`git log --oneline`) for the prevailing style — a short
subject line summarising the change, followed by a body that explains
*why* if the diff is non-obvious. Examples from the current log:

```
S17-S22 audit + fix wave: real HTTP, OTel propagation, baseline, deny(missing_docs)
S3-S16 audit/fix orchestration + deferred-task implementations
```

For multi-crate work, scope the subject by the affected layer
(`tensor-wasm-mem: …`, `tensor-wasm-api: …`).

### Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Add a `Signed-off-by` line to every commit:

```sh
git commit -s -m "your message"
```

This adds `Signed-off-by: Your Name <you@example.com>` using the
identity from your git config. Unsigned commits will be rejected by the
PR check.

If you forgot to sign off, amend the most recent commit with
`git commit --amend -s`, or for a range use
`git rebase --signoff <base>..HEAD`.

### Pull requests

Open the PR against `main`. Use the
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
checklist — it covers lints, tests, docs, and the DCO requirement.

Reviewers look for:

- Tests added or updated for behaviour changes.
- New public items have rustdoc comments (the workspace lints
  `missing_docs` as an error).
- No churn unrelated to the stated change.
- Snapshot, CUDA, and observability-touching changes call out their
  cross-crate impact in the PR description.

## Issues

Bug reports and feature requests go through GitHub Issues. The two
templates under [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE/)
prompt for the information we usually need to triage.

For questions that don't fit either template, open a discussion thread
or email the maintainers (see [`MAINTAINERS.md`](MAINTAINERS.md)).

## Security disclosures

**Do not file security issues on the public tracker.** Email
`security@craton.com.ar` with a reproducer. See
[`SECURITY.md`](SECURITY.md) for the full process, supported versions,
and triage SLO.

## License

By submitting a contribution you agree it is licensed under the
[Apache License 2.0](LICENSE), the same terms as the rest of the
project.
