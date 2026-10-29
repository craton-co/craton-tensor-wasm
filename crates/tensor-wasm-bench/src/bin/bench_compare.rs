// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! CI regression-gate comparator (finding 6).
//!
//! Reads two `tail-latency.json`-shaped result documents — a committed
//! baseline and the current run — and exits non-zero when any *gated* metric
//! has regressed by more than the threshold. Diagnostic metrics are printed
//! with their delta but never fail the build. The gated-vs-diagnostic split is
//! [`tensor_wasm_bench::classify_metric`]; the per-metric regression math is
//! [`tensor_wasm_bench::check_regression`], both unit-tested in the lib.
//!
//! Built only under `--features regression-gate` (which pulls in `serde_json`
//! as a non-dev dependency so this binary can parse the result files) so the
//! default `cargo bench` / `cargo build` path stays serde_json-dev-only and
//! the hand-rolled JSON serialisers in `lib.rs` keep their reason to exist.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p tensor-wasm-bench --features regression-gate --bin bench_compare -- \
//!     bench-results/baseline.json bench-results/tail-latency.json [--threshold 0.10]
//! ```
//!
//! The documents are expected to carry a top-level `"metrics"` array whose
//! entries each have a `"metric"` string and a numeric `"p99_ns"` (the gated
//! statistic — P99 is the SLO-facing number; mean would hide tail
//! regressions). Metrics present in the current run but absent from the
//! baseline are reported as "new" and never fail. The exit code is `1` if any
//! gated metric regressed past the threshold, else `0`.

use std::collections::HashMap;
use std::process::ExitCode;

use serde_json::Value;
use tensor_wasm_bench::{check_regression, MetricClass};

/// The statistic compared for regressions. P99 over the mean so a tail
/// regression that leaves the mean flat is still caught.
const STAT_FIELD: &str = "p99_ns";

/// Default fail-on threshold: 10 % slower.
const DEFAULT_THRESHOLD: f64 = 0.10;

fn metrics_by_key(doc: &Value) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    if let Some(arr) = doc.get("metrics").and_then(Value::as_array) {
        for entry in arr {
            let Some(metric) = entry.get("metric").and_then(Value::as_str) else {
                continue;
            };
            let Some(stat) = entry.get(STAT_FIELD).and_then(Value::as_f64) else {
                continue;
            };
            out.insert(metric.to_string(), stat);
        }
    }
    out
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let baseline_path = args
        .next()
        .ok_or("usage: bench_compare <baseline.json> <current.json> [--threshold FRAC]")?;
    let current_path = args
        .next()
        .ok_or("usage: bench_compare <baseline.json> <current.json> [--threshold FRAC]")?;

    let mut threshold = DEFAULT_THRESHOLD;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--threshold" => {
                let v = args.next().ok_or("--threshold requires a value")?;
                threshold = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --threshold {v:?}: {e}"))?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let baseline_doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&baseline_path)
            .map_err(|e| format!("read {baseline_path}: {e}"))?,
    )
    .map_err(|e| format!("parse {baseline_path}: {e}"))?;
    let current_doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&current_path).map_err(|e| format!("read {current_path}: {e}"))?,
    )
    .map_err(|e| format!("parse {current_path}: {e}"))?;

    let baseline = metrics_by_key(&baseline_doc);
    let current = metrics_by_key(&current_doc);

    let mut any_failure = false;
    let mut keys: Vec<&String> = current.keys().collect();
    keys.sort();

    println!(
        "bench_compare: threshold = {:.1}% slower on {STAT_FIELD}",
        threshold * 100.0
    );
    for key in keys {
        let cur = current[key];
        let Some(&base) = baseline.get(key) else {
            println!("  NEW        {key}: {cur:.0} (no baseline — not gated)");
            continue;
        };
        let check = check_regression(key, base, cur, threshold);
        let tag = match (check.class, check.is_failing_regression) {
            (MetricClass::Gated, true) => "FAIL",
            (MetricClass::Gated, false) => "ok  ",
            (MetricClass::Diagnostic, _) => "diag",
        };
        println!(
            "  {tag}  {key}: {base:.0} -> {cur:.0} ({:+.1}%){}",
            check.delta_frac * 100.0,
            if check.is_failing_regression {
                "  <-- gated regression"
            } else {
                ""
            }
        );
        any_failure |= check.is_failing_regression;
    }

    Ok(any_failure)
}

fn main() -> ExitCode {
    match run() {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => {
            eprintln!("bench_compare: gated regression(s) detected — failing CI");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("bench_compare: {e}");
            ExitCode::FAILURE
        }
    }
}
