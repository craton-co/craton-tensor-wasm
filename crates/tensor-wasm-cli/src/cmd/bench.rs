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
use clap::Args;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

use super::OutputFormat;

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
    /// Output format: `text` (human-readable table, default) or `json`
    /// (machine-readable document for CI perf gates).
    ///
    /// `display_order` pinned so this sorts after the other local flags and
    /// before the global TLS flags, keeping the help layout stable.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, display_order = 800)]
    pub output: OutputFormat,
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
            .call_export_with_args(id, &args.export, &[])
            .await
            .with_context(|| format!("calling `{}` on iter {i}", args.export))?;
        executor
            .terminate(id)
            .await
            .with_context(|| format!("terminating on iter {i}"))?;
        samples.push(start.elapsed());
    }

    let summary = BenchSummary::from_samples(&args.export, args.n, &mut samples);
    match args.output {
        OutputFormat::Text => print_table(&summary),
        OutputFormat::Json => println!("{}", summary.to_json()),
    }
    Ok(())
}

/// Computed P50/P95/P99/max latency summary for a bench run.
///
/// Holds the durations as raw nanoseconds so the JSON renderer can emit
/// stable integer fields (machine-friendly) while the text renderer formats
/// them through [`fmt_dur`]. Pure to construct so unit tests can exercise the
/// math without spinning up an executor.
#[derive(Debug, Clone)]
pub(crate) struct BenchSummary {
    export: String,
    iterations: usize,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
}

impl BenchSummary {
    /// Sort `samples` in place and compute the percentile summary.
    pub(crate) fn from_samples(export: &str, n: usize, samples: &mut [Duration]) -> Self {
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
        Self {
            export: export.to_string(),
            iterations: n,
            p50: p(0.50),
            p95: p(0.95),
            p99: p(0.99),
            max: samples.last().copied().unwrap_or_default(),
        }
    }

    /// Render the summary as a machine-readable JSON document. Latencies are
    /// emitted both as nanosecond integers (`*_ns`, stable for CI thresholds)
    /// and as the human-readable string (`*_human`) the text table shows.
    pub(crate) fn to_json(&self) -> String {
        let doc = serde_json::json!({
            "export": self.export,
            "iterations": self.iterations,
            "latency": {
                "p50_ns": self.p50.as_nanos() as u64,
                "p95_ns": self.p95.as_nanos() as u64,
                "p99_ns": self.p99.as_nanos() as u64,
                "max_ns": self.max.as_nanos() as u64,
                "p50_human": fmt_dur(self.p50),
                "p95_human": fmt_dur(self.p95),
                "p99_human": fmt_dur(self.p99),
                "max_human": fmt_dur(self.max),
            }
        });
        // `to_string` (compact) keeps the line greppable / pipeable; callers
        // who want pretty output can pipe through `jq`.
        doc.to_string()
    }
}

/// Print the P50/P95/P99/max latency table for a computed [`BenchSummary`].
pub(crate) fn print_table(s: &BenchSummary) {
    println!("bench: export=`{}` iterations={}", s.export, s.iterations);
    println!("+-----------+--------------+");
    println!("| percentile|       latency|");
    println!("+-----------+--------------+");
    println!("| P50       | {:>12} |", fmt_dur(s.p50));
    println!("| P95       | {:>12} |", fmt_dur(s.p95));
    println!("| P99       | {:>12} |", fmt_dur(s.p99));
    println!("| max       | {:>12} |", fmt_dur(s.max));
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
        let s = BenchSummary::from_samples("noop", samples.len(), &mut samples);
        print_table(&s);
        assert_eq!(samples.last().copied(), Some(Duration::from_millis(100)));
        assert_eq!(s.p50, Duration::from_millis(50));
        assert_eq!(s.p95, Duration::from_millis(95));
        assert_eq!(s.p99, Duration::from_millis(99));
        assert_eq!(s.max, Duration::from_millis(100));
    }

    #[test]
    fn percentile_handles_single_sample() {
        let mut samples = vec![Duration::from_millis(7)];
        // Should not panic and should not under-/over-flow indexing.
        let s = BenchSummary::from_samples("one", 1, &mut samples);
        print_table(&s);
    }

    #[test]
    fn json_output_is_valid_and_carries_percentiles() {
        // The percentile math is already covered above; this pins the JSON
        // contract CI perf gates will parse. Samples 1..=100 ms so the
        // expected ns values are deterministic.
        let mut samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let summary = BenchSummary::from_samples("noop", samples.len(), &mut samples);
        let json = summary.to_json();

        // Must parse as a JSON object.
        let v: serde_json::Value = serde_json::from_str(&json).expect("bench --output json valid");
        assert_eq!(v["export"], "noop");
        assert_eq!(v["iterations"], 100);
        // P50 = 50ms = 50_000_000 ns.
        assert_eq!(v["latency"]["p50_ns"], 50_000_000u64);
        assert_eq!(v["latency"]["p95_ns"], 95_000_000u64);
        assert_eq!(v["latency"]["p99_ns"], 99_000_000u64);
        assert_eq!(v["latency"]["max_ns"], 100_000_000u64);
        // Human strings present for eyeballing.
        assert!(v["latency"]["p50_human"].is_string());
    }

    #[test]
    fn fmt_dur_picks_unit() {
        assert!(fmt_dur(Duration::from_nanos(500)).ends_with("ns"));
        assert!(fmt_dur(Duration::from_micros(5)).ends_with("us"));
        assert!(fmt_dur(Duration::from_millis(5)).ends_with("ms"));
        assert!(fmt_dur(Duration::from_secs(2)).ends_with("s"));
    }
}
