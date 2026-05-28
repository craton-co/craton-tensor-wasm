// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Measures the overhead of `/invoke-stream` vs `/invoke` for the
//! scaffold response shape. v0.3.7 stream just emits a single
//! `event: scaffold` frame; this bench pins the noise floor so v0.4's
//! actual streaming wire can be benchmarked against it.
//!
//! ## Placeholder status (B7.1 pending)
//!
//! The `/invoke-stream` route is in flight on the parallel B7.1 branch
//! and is **not yet** in this worktree's `build_router` surface. This
//! bench file is committed as a placeholder so the `[[bench]]` entry in
//! `Cargo.toml` resolves and a `cargo build --benches` invocation
//! against this crate compiles cleanly; the three bench functions
//! themselves contain `todo!()` so an attempt to actually run them
//! (`cargo bench --bench streaming_invoke`) panics with a clear
//! "route not yet wired" message rather than silently emitting numbers
//! against the legacy `/invoke` path that would mislead a later
//! comparison.
//!
//! When B7.1 lands and `/invoke-stream` is reachable on `build_router`,
//! replace each `todo!()` body with the corresponding router-driven
//! sample loop (pattern after `tail_latency.rs::measure_invoke_not_found`:
//! build the router once per group, oneshot the request inside `b.iter`,
//! consume the response body to completion). The `baseline.json` entries
//! for these groups carry `regression_check: false` until the same PR
//! captures real numbers and flips them on.
//!
//! ## Why three groups (post-B7.1)
//!
//! 1. **`invoke_stream/baseline_invoke`** — POST `/functions/{id}/invoke`
//!    with an empty body. The reference number any stream variant has to
//!    beat (or at least not lose to by much).
//! 2. **`invoke_stream/sse`** — POST `/functions/{id}/invoke-stream` with
//!    `Accept: text/event-stream`. Drives the SSE framing path.
//! 3. **`invoke_stream/chunked`** — POST `/functions/{id}/invoke-stream`
//!    with the default `Accept`. Drives the chunked-JSON fallback path.
//!
//! All three benches share a router with one pre-deployed function (set
//! up in each `bench_function`'s outer scope; the function id is read at
//! deploy time and reused inside the `b.iter` closure). The pre-deploy
//! avoids charging registry-growth allocator noise to the streaming
//! measurement.

#![allow(clippy::todo)]

use criterion::{criterion_group, criterion_main, Criterion};

/// Baseline `/invoke` measurement. Placeholder until B7.1 lands; see
/// module docs for the post-B7.1 wiring contract.
fn bench_invoke_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream");
    group.bench_function("baseline_invoke", |b| {
        b.iter(|| {
            // TODO(B7.1): once `/invoke-stream` lands, replace this with
            // the actual baseline POST /functions/{id}/invoke loop.
            // Pattern: build a router with one pre-deployed function via
            // `build_router_with_audit` (see `tail_latency.rs`), oneshot
            // a POST with an empty body, drain the response body with
            // `to_bytes`, assert StatusCode::OK.
            todo!("B7.1: wire /invoke baseline once /invoke-stream route is restored");
        });
    });
    group.finish();
}

/// SSE-mode `/invoke-stream` measurement. Placeholder until B7.1 lands.
fn bench_invoke_stream_sse(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream");
    group.bench_function("sse", |b| {
        b.iter(|| {
            // TODO(B7.1): POST /functions/{id}/invoke-stream with
            // `Accept: text/event-stream`. Consume the response body to
            // completion (the v0.3.7 stream emits a single
            // `event: scaffold` frame; the bench's job is to pin the
            // framing-overhead floor before the v0.4 real-stream lands).
            todo!("B7.1: wire /invoke-stream SSE path once route is restored");
        });
    });
    group.finish();
}

/// Chunked-JSON-mode `/invoke-stream` measurement. Placeholder until
/// B7.1 lands.
fn bench_invoke_stream_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("invoke_stream");
    group.bench_function("chunked", |b| {
        b.iter(|| {
            // TODO(B7.1): POST /functions/{id}/invoke-stream with the
            // default `Accept` (i.e. no SSE negotiation). Same body-drain
            // contract as the SSE variant above; this measures the
            // chunked-transfer-encoding fallback floor.
            todo!("B7.1: wire /invoke-stream chunked path once route is restored");
        });
    });
    group.finish();
}

criterion_group!(
    streaming,
    bench_invoke_baseline,
    bench_invoke_stream_sse,
    bench_invoke_stream_chunked,
);
criterion_main!(streaming);
