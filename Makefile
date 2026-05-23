.PHONY: help build test bench fmt fmt-check lint check doc clean ci ci-bench

help:
	@echo "Project Bali - Makefile targets:"
	@echo "  build      - Build all workspace crates"
	@echo "  test       - Run all workspace tests"
	@echo "  bench      - Run all workspace benchmarks"
	@echo "  fmt        - Format all code with rustfmt"
	@echo "  fmt-check  - Check formatting without modifying files"
	@echo "  lint       - Run clippy with warnings as errors"
	@echo "  check      - Run cargo check on all targets"
	@echo "  doc        - Build documentation (no deps)"
	@echo "  clean      - Remove build artifacts"
	@echo "  ci         - Run fmt-check, lint, check, and test"
	@echo "  ci-bench   - Run benches with CI-equivalent flags"

build:
	cargo build --workspace

test:
	cargo test --workspace

bench:
	cargo bench --workspace

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace --all-targets

doc:
	cargo doc --workspace --no-deps

clean:
	cargo clean

ci: fmt-check lint check test

ci-bench: ## Run benches with CI-equivalent flags
	cargo bench --workspace --no-run
	cargo bench --workspace -- --warm-up-time 1 --measurement-time 5
