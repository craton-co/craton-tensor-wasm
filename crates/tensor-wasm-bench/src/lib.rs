// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Benchmark harness crate for Craton TensorWasm.
//!
//! The crate itself exposes no public API — it exists as the home for the
//! Criterion benches in [`benches/`](../../crates/tensor-wasm-bench/benches/) and
//! their shared dev-dependencies. Each bench file targets one slice of the
//! runtime:
//!
//! - `benches/kernel_dispatch.rs` — back-pressure permit + dispatch-future
//!   overhead (serial and concurrent).
//! - `benches/cold_start.rs` — snapshot capture, in-memory restore, and a
//!   full capture→fs→restore disk round-trip.
//! - `benches/memory_bandwidth.rs` — host-side `copy_from_slice` over the
//!   guarded host buffer in sequential and fixed-stride patterns.
//! - `benches/jit_compile.rs` — PTX emit latency, blueprint fingerprint
//!   cost, and kernel-cache hit-vs-miss latency.
//! - `benches/e2e_inference.rs` — full axum router round-trip through
//!   `tensor-wasm-api` for the healthz, create-function, and invoke-not-found
//!   paths.
//! - `benches/tail_latency.rs` — hand-rolled 10 000-sample loop that
//!   captures P50/P95/P99/**P99.9**/max for `dispatch/serial/100`,
//!   `dispatch/concurrent_cap64/100`, `e2e/healthz/get`, and
//!   `e2e/invoke_not_found/post`. Output goes to stdout and (when run
//!   from the workspace root) to `bench-results/tail-latency.json`.
//!
//! See [`docs/PERFORMANCE.md`](../../../docs/PERFORMANCE.md) for the
//! published bench inventory, reference numbers, and the regression-gate
//! policy used by `.github/workflows/bench.yml`.
#![deny(missing_docs)]
