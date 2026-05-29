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
//! - `benches/dispatch_future_backends.rs` — F3/RFC 0001 comparison of the
//!   busy-poll `DispatchFuture` path against the `cuda-async` backend
//!   (the latter a documented stub until v0.4). Meaningful only with
//!   `--features cuda`; emits `DISPATCH_BACKEND` JSON lines and
//!   `bench-results/dispatch-future-backends.json`. Diagnostic, not gated.
//! - `benches/metrics_label_validation.rs` — `HttpRequestLabels::try_new`
//!   route-lookup cost against a 100-route allow-list (first/last/miss).
//!   Pins the `Vec` → `HashSet` migration. Diagnostic, not gated.
//! - `benches/call_export_args.rs` — `call_export_with_args` overhead vs
//!   the legacy no-args `call_export` shim (`call_export/noargs/*` and
//!   `call_export/args/two_i32`).
//! - `benches/streaming_invoke.rs` — `/invoke-stream` vs `/invoke` floor
//!   (`invoke_stream/baseline_invoke`, `/sse`, `/chunked`). Placeholder
//!   that emits skip lines until B7.1 wires the route.
//!
//! See [`docs/PERFORMANCE.md`](../../../docs/PERFORMANCE.md) for the
//! published bench inventory, reference numbers, and the regression-gate
//! policy used by `.github/workflows/bench.yml`.
//!
//! ## Shared bench helpers
//!
//! The crate is not entirely API-free: it carries the small set of items that
//! were previously copy-pasted across `benches/tail_latency.rs` and
//! `benches/dispatch_future_backends.rs`. Centralising them here lets a single
//! `#[cfg(test)] mod tests` block (compiled and run by `cargo test`, unlike a
//! `cfg(test)` module inside a `harness = false` bench target) cover the
//! percentile math and the JSON serialisation contract. The exported items are:
//!
//! - [`percentile_nearest_rank`] — the nearest-rank percentile both benches use.
//! - [`TailResult`] — the `tail_latency.rs` per-metric JSON row.
//! - [`TailStats`] — the `dispatch_future_backends.rs` per-backend JSON row.
//! - [`json_escape`] — minimal RFC 8259 string escaping shared by the
//!   hand-rolled serialisers (the crate's `serde_json` is a *dev*-dependency,
//!   so the non-test serialiser path cannot reach for it).

#![deny(missing_docs)]

use std::time::Duration;

/// Nearest-rank percentile (`samples[ceil(p * n) - 1]`).
///
/// `sorted` must be pre-sorted ascending; the function panics on an empty
/// slice to surface bench misconfiguration loudly rather than silently
/// emitting 0 ns, and panics if `p` is outside `0.0..=1.0`. A `p` of `0.0`
/// clamps the rank to 1 and so returns the minimum sample.
///
/// This is the algorithm documented in `benches/tail_latency.rs` (ISO/IEC
/// 25062, the method `hdrhistogram` documents): it always returns an
/// actually-observed value with no interpolation artefacts.
///
/// # Panics
///
/// Panics if `sorted` is empty or if `p` is not in `0.0..=1.0`.
#[must_use]
pub fn percentile_nearest_rank(sorted: &[Duration], p: f64) -> u128 {
    assert!(!sorted.is_empty(), "percentile of empty sample set");
    assert!((0.0..=1.0).contains(&p), "percentile p out of range");
    let n = sorted.len();
    let rank = ((p * n as f64).ceil() as usize).max(1);
    let idx = rank.min(n) - 1;
    sorted[idx].as_nanos()
}

/// Minimal RFC 8259 string escaping for the hand-rolled JSON serialisers.
///
/// `serde_json` is only a *dev*-dependency of this crate, so the serialiser
/// paths used at `cargo bench` time cannot depend on it. This helper escapes
/// the characters that would otherwise produce invalid JSON: backslash,
/// double-quote, and the C0 control characters (with the named short escapes
/// for `\n`, `\r`, `\t`, `\u{08}`, and `\u{0C}`, and `\uXXXX` for the rest).
#[must_use]
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// One row of the `benches/tail_latency.rs` structured JSON output.
///
/// Field names match the format documented in `bench-results/README.md` so
/// downstream parsers (Grafana import scripts, the v0.3 observability
/// dashboards) can consume it without translation. The `backend` field was
/// added in W4.4 (RFC 0001 Unresolved questions extension) and is the
/// compile-time `BACKEND_LABEL` of the run; it is always present today.
/// Downstream parsers that pre-date W4.4 should treat `backend` as optional
/// and fall back to `"unified-memory"` when it is missing — that matches the
/// W4.6 historical default.
#[derive(Debug, Clone)]
pub struct TailResult {
    /// `<group>/<id>` metric label, e.g. `dispatch/serial/100`.
    pub metric: String,
    /// Compile-time backend discriminator (`unified-memory` / `cudarc` /
    /// `cuda-oxide`).
    pub backend: &'static str,
    /// Number of raw observations the percentiles were computed over.
    pub samples: usize,
    /// 50th-percentile latency, nanoseconds.
    pub p50_ns: u128,
    /// 95th-percentile latency, nanoseconds.
    pub p95_ns: u128,
    /// 99th-percentile latency, nanoseconds.
    pub p99_ns: u128,
    /// 99.9th-percentile latency, nanoseconds.
    pub p99_9_ns: u128,
    /// Maximum observed latency, nanoseconds.
    pub max_ns: u128,
}

impl TailResult {
    /// Render this row as a single-line JSON object. String fields are escaped
    /// via [`json_escape`]; numeric fields are emitted verbatim.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"metric\":\"{}\",\"backend\":\"{}\",\"samples\":{},\"p50_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"p99_9_ns\":{},\"max_ns\":{}}}",
            json_escape(&self.metric),
            json_escape(self.backend),
            self.samples,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.p99_9_ns,
            self.max_ns,
        )
    }
}

/// One row of the `benches/dispatch_future_backends.rs` structured JSON
/// output: a tail-percentile summary for a single dispatch backend.
///
/// JSON fields mirror the [`TailResult`] schema plus a `backend`
/// discriminator and a `status` field (always `"measured"` for this row;
/// stub backends emit a separate skip line — see
/// `dispatch_future_backends.rs`) so the same parser handles measured and
/// skip lines.
#[derive(Debug, Clone)]
pub struct TailStats {
    /// Stable backend label (`busy-poll` / `cuda-async`).
    pub backend: String,
    /// Number of raw observations the percentiles were computed over.
    pub samples: usize,
    /// 50th-percentile latency, nanoseconds.
    pub p50_ns: u128,
    /// 95th-percentile latency, nanoseconds.
    pub p95_ns: u128,
    /// 99th-percentile latency, nanoseconds.
    pub p99_ns: u128,
    /// 99.9th-percentile latency, nanoseconds.
    pub p99_9_ns: u128,
    /// Maximum observed latency, nanoseconds.
    pub max_ns: u128,
}

impl TailStats {
    /// Render this row as a single-line JSON object with `"status":"measured"`.
    /// The `backend` string is escaped via [`json_escape`].
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"backend\":\"{}\",\"status\":\"measured\",\"samples\":{},\"p50_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"p99_9_ns\":{},\"max_ns\":{}}}",
            json_escape(&self.backend),
            self.samples,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.p99_9_ns,
            self.max_ns,
        )
    }
}

#[cfg(test)]
mod tests {
    // `super::*` re-exports the crate-root `use std::time::Duration;`, so no
    // separate `Duration` import is needed here.
    use super::*;

    fn durations(ns: impl IntoIterator<Item = u64>) -> Vec<Duration> {
        ns.into_iter().map(Duration::from_nanos).collect()
    }

    #[test]
    fn nearest_rank_known_distribution() {
        // 1..=100 ns sorted. Nearest-rank: rank(p, 100) = ceil(p*100) → idx-1.
        let samples = durations(1u64..=100);
        assert_eq!(percentile_nearest_rank(&samples, 0.50), 50);
        assert_eq!(percentile_nearest_rank(&samples, 0.95), 95);
        assert_eq!(percentile_nearest_rank(&samples, 0.99), 99);
        // P99.9 of 100 samples: ceil(99.9) = 100 → samples[99] = 100.
        assert_eq!(percentile_nearest_rank(&samples, 0.999), 100);
        // P100 → samples[99] = 100.
        assert_eq!(percentile_nearest_rank(&samples, 1.0), 100);
    }

    #[test]
    fn nearest_rank_p0_returns_min() {
        let samples = durations([7, 13]);
        // p=0.0 clamps the rank to 1 → samples[0].
        assert_eq!(percentile_nearest_rank(&samples, 0.0), 7);
    }

    #[test]
    fn nearest_rank_single_element() {
        let samples = durations([42]);
        // Every percentile of a one-element set is that element.
        assert_eq!(percentile_nearest_rank(&samples, 0.0), 42);
        assert_eq!(percentile_nearest_rank(&samples, 0.50), 42);
        assert_eq!(percentile_nearest_rank(&samples, 0.999), 42);
        assert_eq!(percentile_nearest_rank(&samples, 1.0), 42);
    }

    #[test]
    fn nearest_rank_is_monotonic_non_decreasing() {
        let samples = durations(1u64..=1000);
        let p50 = percentile_nearest_rank(&samples, 0.50);
        let p95 = percentile_nearest_rank(&samples, 0.95);
        let p99 = percentile_nearest_rank(&samples, 0.99);
        let p99_9 = percentile_nearest_rank(&samples, 0.999);
        let max = samples.last().unwrap().as_nanos();
        assert!(p50 <= p95);
        assert!(p95 <= p99);
        assert!(p99 <= p99_9);
        assert!(p99_9 <= max);
    }

    #[test]
    #[should_panic(expected = "percentile of empty sample set")]
    fn nearest_rank_empty_panics() {
        let _ = percentile_nearest_rank(&[], 0.50);
    }

    #[test]
    #[should_panic(expected = "percentile p out of range")]
    fn nearest_rank_p_out_of_range_panics() {
        let samples = durations([1, 2, 3]);
        let _ = percentile_nearest_rank(&samples, 1.5);
    }

    #[test]
    fn json_escape_handles_quotes_backslashes_and_controls() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb\tc"), "a\\nb\\tc");
        // Bare control char (U+0001) becomes a \u escape.
        assert_eq!(json_escape("\u{1}"), "\\u0001");
    }

    #[test]
    fn tail_result_json_round_trips_through_serde_json() {
        let r = TailResult {
            metric: "dispatch/serial/100".to_string(),
            backend: "unified-memory",
            samples: 10_000,
            p50_ns: 1,
            p95_ns: 2,
            p99_ns: 3,
            p99_9_ns: 4,
            max_ns: 5,
        };
        let s = r.to_json();
        // serde_json (a dev-dependency) parses the hand-rolled output, proving
        // the serialiser emits valid JSON with the documented field names.
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["metric"], "dispatch/serial/100");
        assert_eq!(v["backend"], "unified-memory");
        assert_eq!(v["samples"], 10_000);
        assert_eq!(v["p50_ns"], 1);
        assert_eq!(v["p95_ns"], 2);
        assert_eq!(v["p99_ns"], 3);
        assert_eq!(v["p99_9_ns"], 4);
        assert_eq!(v["max_ns"], 5);
    }

    #[test]
    fn tail_result_json_escapes_metric_with_quote() {
        // A metric containing a quote must still produce parseable JSON.
        let r = TailResult {
            metric: "weird\"metric".to_string(),
            backend: "cudarc",
            samples: 1,
            p50_ns: 0,
            p95_ns: 0,
            p99_ns: 0,
            p99_9_ns: 0,
            max_ns: 0,
        };
        let s = r.to_json();
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["metric"], "weird\"metric");
        assert_eq!(v["backend"], "cudarc");
    }

    #[test]
    fn tail_stats_json_round_trips_and_marks_measured() {
        let stats = TailStats {
            backend: "busy-poll".to_string(),
            samples: 64,
            p50_ns: 10,
            p95_ns: 20,
            p99_ns: 30,
            p99_9_ns: 40,
            max_ns: 50,
        };
        let s = stats.to_json();
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["backend"], "busy-poll");
        assert_eq!(v["status"], "measured");
        assert_eq!(v["samples"], 64);
        assert_eq!(v["p99_9_ns"], 40);
        assert_eq!(v["max_ns"], 50);
    }
}
