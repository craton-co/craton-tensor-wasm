# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the `tensor-wasm` CLI binary, which also serves the HTTP API
# via `tensor-wasm serve`. Stage 1 uses the official `rustlang/rust:nightly` image
# plus rustup to honour the workspace-pinned channel from `rust-toolchain.toml`.
# Stage 2 is distroless/cc so the runtime image carries no shell or package
# manager and remains small.
#
# Build context is the repository root:
#   docker build -f docker/tensor-wasm-api.Dockerfile -t tensor-wasm-api .

# --- Stage 1: builder -------------------------------------------------------
FROM rustlang/rust:nightly-bookworm-slim AS builder

WORKDIR /workspace

# Toolchain pin: copy rust-toolchain.toml first so rustup installs the exact
# nightly the workspace requires before we touch any other files. Subsequent
# cargo invocations in this stage will use that channel automatically.
COPY rust-toolchain.toml ./rust-toolchain.toml
RUN rustup show active-toolchain || rustup toolchain install

# Build deps for any C/C++ sys-crates we link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the workspace and build.
COPY . .
RUN cargo build --release -p tensor-wasm-cli \
    && cp target/release/tensor-wasm /usr/local/bin/tensor-wasm

# --- Stage 2: runtime -------------------------------------------------------
FROM gcr.io/distroless/cc-debian12 AS runtime

COPY --from=builder /usr/local/bin/tensor-wasm /usr/local/bin/tensor-wasm

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/tensor-wasm"]
CMD ["serve", "--addr", "0.0.0.0:8080"]
