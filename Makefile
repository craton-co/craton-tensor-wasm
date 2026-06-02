.PHONY: help build test bench fmt fmt-check lint check doc clean ci ci-bench ptx

help:
	@echo "Craton TensorWasm - Makefile targets:"
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
	@echo "  ptx        - Regenerate vector_add PTX fixtures via nvcc (needs CUDA)"

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

# Regenerate the vector_add PTX fixtures from kernels/vector_add.cu (fix #7).
# Hand-authored PTX is rejected by the CUDA 13 JIT; nvcc output is canonical.
# Requires the CUDA Toolkit (nvcc) — run on a CUDA host, then commit the
# regenerated *.ptx. sm_75 covers the RTX 2060 dev box and JITs forward onto
# newer GPUs; sm_80 is the canonical Ampere target.
ptx:
	@command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found: install the CUDA Toolkit (see docs/CUDA-SETUP.md)"; exit 2; }
	nvcc -ptx -arch=sm_75 -o kernels/vector_add_sm75.ptx kernels/vector_add.cu
	nvcc -ptx -arch=sm_80 -o kernels/vector_add.ptx      kernels/vector_add.cu
	cp kernels/vector_add_sm75.ptx crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add_sm75.ptx
	cp kernels/vector_add.ptx      crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add.ptx
	@echo "Regenerated vector_add PTX fixtures (sm_75 + sm_80) from kernels/vector_add.cu"

clean:
	cargo clean

ci: fmt-check lint check test

ci-bench:
	cargo bench --workspace --no-run
	cargo bench --workspace -- --warm-up-time 1 --measurement-time 5
