// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm bench` — local latency micro-benchmark.
//!
//! Loads a `.wasm` file, spawns a fresh instance per iteration, invokes the
//! named export, records the wall-clock duration, and finally prints a table
//! with P50/P95/P99 and max latencies. The bench measures end-to-end
//! `spawn + call + terminate` so it captures cold-start cost as well —
//! steady-state micro-bench (single instance, repeated calls) will follow in
//! a dedicated `--mode steady` flag.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{TensorWasmExecutor, SpawnConfig};
use clap::Args;

/// Arguments to `tensor-wasm bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Path to the `.wasm` file to benchmark.
    pub file: PathBuf,
    /// Name of the export to call on each iteration.
    #[arg(long, default_value = "main")]
    pub export: String,
    /// Number of iterations to run. Must be >= 1.
    #[arg(long, default_value_t = 100)]
    pub n: usize,
}

/// Entry point for `tensor-wasm bench`.
pub async fn run(args: BenchArgs) -> Result<()> {
    if args.n == 0 {
        anyhow::bail!("--n must be >= 1");
    }
    let wasm = std::fs::read(&args.file)
        .with_context(|| format!("reading wasm file {}", args.file.display()))?;
    let engine = Arc::new(TensorWasmEngine::new().context("constructing TensorWasmEngine")?);
    let executor = TensorWasmExecutor::new(engine);

    let mut samples: Vec<Duration> = Vec::with_capacity(args.n);
    for i in 0..args.n {
        let start = Instant::now();
        let id = executor
            .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &wasm)
            .await
            .with_context(|| format!("spawning instance on iter {i}"))?;
        executor
            .call_export(id, &args.export)
            .await
            .with_context(|| format!("calling `{}` on iter {i}", args.export))?;
        executor
            .terminate(id)
            .await
            .with_context(|| format!("terminating on iter {i}"))?;
        samples.push(start.elapsed());
    }

    print_table(&args.export, args.n, &mut samples);
    Ok(())
}

/// Compute and print the P50/P95/P99/max latency table.
///
/// `samples` is sorted in place. Pure function so unit tests can exercise the
/// math without spinning up an executor.
pub(crate) fn print_table(export: &str, n: usize, samples: &mut [Duration]) {
    samples.sort();
    let p = |q: f64| -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }
        let len = samples.len();
        // Nearest-rank percentile: index = ceil(q * len) - 1, clamped to
        // [0, len-1]. Using `usize` directly avoids the historical `isize`
        // round-trip that could overflow for very large `len`.
        let rank = (q * len as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(len - 1);
        samples[idx]
    };
    let max = samples.last().copied().unwrap_or_default();
    let p50 = p(0.50);
    let p95 = p(0.95);
    let p99 = p(0.99);

    println!("bench: export=`{}` iterations={}", export, n);
    println!("+-----------+--------------+");
    println!("| percentile|       latency|");
    println!("+-----------+--------------+");
    println!("| P50       | {:>12} |", fmt_dur(p50));
    println!("| P95       | {:>12} |", fmt_dur(p95));
    println!("| P99       | {:>12} |", fmt_dur(p99));
    println!("| max       | {:>12} |", fmt_dur(max));
    println!("+-----------+--------------+");
}

/// Format a [`Duration`] for the bench table. Picks the most-readable unit
/// (`us` / `ms` / `s`) without exposing trailing zero precision noise.
fn fmt_dur(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2} us", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_picks_nearest_rank() {
        // Samples 1..=100 ms; P50 = 50ms, P95 = 95ms, P99 = 99ms, max = 100ms.
        let mut samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        print_table("noop", samples.len(), &mut samples);
        assert_eq!(samples.last().copied(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn percentile_handles_single_sample() {
        let mut samples = vec![Duration::from_millis(7)];
        // Should not panic and should not under-/over-flow indexing.
        print_table("one", 1, &mut samples);
    }

    #[test]
    fn fmt_dur_picks_unit() {
        assert!(fmt_dur(Duration::from_nanos(500)).ends_with("ns"));
        assert!(fmt_dur(Duration::from_micros(5)).ends_with("us"));
        assert!(fmt_dur(Duration::from_millis(5)).ends_with("ms"));
        assert!(fmt_dur(Duration::from_secs(2)).ends_with("s"));
    }
}
