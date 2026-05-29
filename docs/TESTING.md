# Testing Conventions

> Owner: @craton-co/maintainers.

## Test layers

| Layer | Tool | Location | When run |
|---|---|---|---|
| Unit | `cargo test --lib` | `src/**/{tests.rs,mod.rs#[cfg(test)]}` | every PR |
| Integration | `cargo test --test ...` | `crates/*/tests/*.rs` | every PR |
| Property | `proptest` | mixed with unit/integration | every PR |
| Fuzz (short) | `cargo fuzz` | `fuzz/fuzz_targets/*.rs` | nightly cron |
| Fuzz (long) | `cargo fuzz` | same | weekly cron |
| CUDA conformance | `cargo test --features cuda -- --ignored` | gated tests | self-hosted runner |
| Benchmarks | `criterion` | `crates/*/benches/*.rs`, `crates/tensor-wasm-bench/benches/*.rs` | PR-triggered + nightly |

## #[ignore] policy
- CUDA-requiring tests: `#[ignore = "requires CUDA hardware"]`.
- Hardware-runner-only tests: `#[ignore = "requires self-hosted runner"]`.
- Flaky tests: not allowed — fix or delete; do NOT mark `#[ignore]`.

## Coverage target
85% line coverage on stable crates by 1.0 (measured via `cargo llvm-cov`).

Coverage is measured in CI by the
[`coverage.yml`](../.github/workflows/coverage.yml) workflow
(`cargo llvm-cov`), which uploads the `lcov.info` report to Codecov on
pushes to `main` and on pull requests. A direct link to the coverage
dashboard is planned and will be added here once the Codecov project
page is public.
