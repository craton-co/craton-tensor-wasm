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
//!   busy-poll `event.query()` `DispatchFuture` path against a
//!   `stream.synchronize()`-based `cuda-async` backend. Both drive real
//!   `cust` work; meaningful only with `--features cuda`. Emits
//!   `DISPATCH_BACKEND` JSON lines and
//!   `bench-results/dispatch-future-backends.json`. Diagnostic, not gated.
//! - `benches/metrics_label_validation.rs` — `HttpRequestLabels::try_new`
//!   route-lookup cost against a 100-route allow-list (first/last/miss).
//!   Pins the `Vec` → `HashSet` migration. Diagnostic, not gated.
//! - `benches/call_export_args.rs` — `call_export_with_args` overhead: a
//!   `hot/*` group (instance reused across the loop) isolates args
//!   marshalling, and an `envelope/*` group (spawn + invoke + terminate per
//!   sample) matches the `/invoke` per-request shape.
//! - `benches/streaming_invoke.rs` — `/invoke-stream` vs `/invoke` floor
//!   (`invoke_stream/baseline_invoke`, `/sse`, `/chunked`), driven against a
//!   pre-deployed `_start` no-op function through the live streaming route.
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

/// Whether a metric participates in the CI regression gate or is published
/// for diagnosis only (finding 6).
///
/// The gate fails CI when a [`Gated`](MetricClass::Gated) metric regresses by
/// more than the configured threshold. [`Diagnostic`](MetricClass::Diagnostic)
/// metrics are recorded and diffed by humans but never fail the build — they
/// are too noisy (micro-benches dominated by allocator jitter) or too
/// environment-dependent (disk round-trip) to gate on. This split is the
/// "mark which metrics are gated vs diagnostic" half of the regression-gate
/// methodology; the bench inventory in `docs/PERFORMANCE.md` carries the
/// authoritative per-metric mapping, and [`classify_metric`] encodes the
/// default rules a comparator applies when a metric is not explicitly listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricClass {
    /// Regressions beyond the threshold fail CI.
    Gated,
    /// Recorded and diffed, but never fails CI.
    Diagnostic,
}

/// Default gated-vs-diagnostic classification for a `<group>/<id>` metric key.
///
/// Rules (first match wins):
/// * `dispatch/real_sync*`, `dispatch/contended*`, and every
///   `*_steady_state*` / disk-round-trip / `dispatch_future_backends`
///   backend-comparison metric is **diagnostic** — they depend on the GPU,
///   the host filesystem, or are explicit A/B harnesses, none of which gate
///   cleanly across CI runners.
/// * the headline `dispatch/serial*`, `dispatch/concurrent*`,
///   `e2e/healthz*`, `e2e/invoke_not_found*`, and `jit_compile/*` metrics are
///   **gated** — they are CPU-bound, low-variance, and meaningful on any
///   runner.
/// * anything unrecognised defaults to **diagnostic** (fail-open for the
///   gate: a new, unclassified metric must be opted *in* to gating, so adding
///   a bench never accidentally breaks CI before its baseline is captured).
#[must_use]
pub fn classify_metric(metric: &str) -> MetricClass {
    let diagnostic_markers = [
        "real_sync",
        "contended",
        "steady_state",
        "disk_round_trip",
        "span_only",
        "secured",
        "metrics_label_validation",
        "memory_bandwidth",
    ];
    if diagnostic_markers.iter().any(|m| metric.contains(m)) {
        return MetricClass::Diagnostic;
    }
    let gated_prefixes = [
        "dispatch/serial",
        "dispatch/concurrent",
        "e2e/healthz",
        "e2e/invoke_not_found",
        "jit_compile/",
        "cold_start/capture",
    ];
    if gated_prefixes.iter().any(|p| metric.starts_with(p)) {
        return MetricClass::Gated;
    }
    MetricClass::Diagnostic
}

/// Outcome of comparing one metric's current value against its baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionCheck {
    /// The `<group>/<id>` metric key.
    pub metric: String,
    /// Gated or diagnostic (see [`MetricClass`]).
    pub class: MetricClass,
    /// Baseline value (e.g. `p99_ns` or mean ns), in the same unit as
    /// `current`.
    pub baseline: f64,
    /// Current run's value, same unit as `baseline`.
    pub current: f64,
    /// Fractional change `(current - baseline) / baseline`. Positive means
    /// slower (a regression); negative means faster.
    pub delta_frac: f64,
    /// `true` only when the metric is [`Gated`](MetricClass::Gated) AND
    /// `delta_frac` exceeds `threshold_frac`. Diagnostic metrics are never
    /// failing regardless of `delta_frac`.
    pub is_failing_regression: bool,
}

/// Compare one metric against its baseline at the given fractional threshold
/// (e.g. `0.10` for "fail on >10 % slower").
///
/// A non-positive `baseline` is treated as "no usable baseline" and yields a
/// `delta_frac` of `0.0` and `is_failing_regression == false` — a fresh
/// metric with a zero/placeholder baseline never fails the gate.
///
/// # Panics
///
/// Panics if `threshold_frac` is negative (a nonsensical gate config).
#[must_use]
pub fn check_regression(
    metric: &str,
    baseline: f64,
    current: f64,
    threshold_frac: f64,
) -> RegressionCheck {
    assert!(threshold_frac >= 0.0, "threshold_frac must be non-negative");
    let class = classify_metric(metric);
    let delta_frac = if baseline > 0.0 {
        (current - baseline) / baseline
    } else {
        0.0
    };
    let is_failing_regression =
        class == MetricClass::Gated && baseline > 0.0 && delta_frac > threshold_frac;
    RegressionCheck {
        metric: metric.to_string(),
        class,
        baseline,
        current,
        delta_frac,
        is_failing_regression,
    }
}

/// Resolve the compile-time backend label from the two backend feature
/// booleans, applying the documented priority order
/// `cuda-oxide` > `cudarc` > `unified-memory`.
///
/// Pulled out as a pure function (rather than only the `#[cfg]`-selected
/// `const` in `tail_latency.rs`) so the priority ordering is unit-testable
/// across all four combinations without recompiling under different feature
/// sets. [`backend_label`] wires the live `#[cfg]` flags into this function.
#[must_use]
pub const fn resolve_backend_label(cuda_oxide_backend: bool, cudarc_backend: bool) -> &'static str {
    if cuda_oxide_backend {
        "cuda-oxide"
    } else if cudarc_backend {
        "cudarc"
    } else {
        "unified-memory"
    }
}

/// The active compile-time backend label for this build, resolved via
/// [`resolve_backend_label`] over the live `cuda-oxide-backend` /
/// `cudarc-backend` feature flags. `tail_latency.rs` uses this so its
/// `BACKEND_LABEL` and the tested priority logic share one implementation.
#[must_use]
pub const fn backend_label() -> &'static str {
    resolve_backend_label(
        cfg!(feature = "cuda-oxide-backend"),
        cfg!(feature = "cudarc-backend"),
    )
}

/// Locate the workspace `bench-results/` directory, or `None` if neither the
/// cwd nor the manifest-relative candidate resolves.
///
/// Shared by `tail_latency.rs` and `dispatch_future_backends.rs` (which used
/// to carry byte-identical copies) so the lookup behaves the same under
/// `cargo bench` regardless of which bench ran, and so the path logic is
/// reachable from `cargo test` (it cannot be tested from inside a
/// `harness = false` bench, where `cfg(test)` is unset).
///
/// First tries `<cwd>/bench-results` (true when invoked from the workspace
/// root), then `<CARGO_MANIFEST_DIR>/../../bench-results` (true when `cargo
/// bench` is launched from inside the crate dir and changes cwd).
#[must_use]
pub fn workspace_bench_results_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("bench-results");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.join("..").join("..").join("bench-results");
    let canonical = std::fs::canonicalize(&candidate).ok()?;
    canonical.is_dir().then_some(canonical)
}

/// Render the full `tail-latency.json` document from a metric set.
///
/// Extracted from `tail_latency.rs::render_file` so the document-level JSON
/// validity (not just per-row validity) is provable from `cargo test` by
/// round-tripping the output through `serde_json` — the bench file's own
/// `cfg(test)` module never compiles under the bench profile. The schema is
/// documented in `bench-results/README.md`.
#[must_use]
pub fn render_tail_results_document(
    results: &[TailResult],
    backend: &str,
    samples: usize,
) -> String {
    let metrics = results
        .iter()
        .map(|r| format!("    {}", r.to_json()))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        concat!(
            "{{\n",
            "  \"// generated\": \"P99.9 tail-latency snapshot (see crates/tensor-wasm-bench/benches/tail_latency.rs).\",\n",
            "  \"// algorithm\": \"nearest-rank percentile per ISO/IEC 25062 over {} sorted samples per metric\",\n",
            "  \"// backend-axis\": \"W4.4 RFC 0001 extension -- run the bench once per backend (default / --features cudarc-backend / --features cuda-oxide-backend) and diff the three files; see docs/BENCHMARKING.md for the operator recipe\",\n",
            "  \"backend\": \"{}\",\n",
            "  \"metrics\": [\n{}\n  ]\n",
            "}}\n",
        ),
        samples,
        json_escape(backend),
        metrics,
    )
}

/// Render the full `dispatch-future-backends.json` document from a set of
/// already-serialised per-backend JSON lines (measured or skip).
///
/// Extracted from `dispatch_future_backends.rs::render_file` for the same
/// `cargo test`-visible document-validity reason as
/// [`render_tail_results_document`].
#[must_use]
pub fn render_backend_document(lines: &[String], samples: usize) -> String {
    let body = lines
        .iter()
        .map(|l| format!("    {l}"))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        concat!(
            "{{\n",
            "  \"// generated\": \"BusyPoll vs cuda-async DispatchFuture comparison (see crates/tensor-wasm-bench/benches/dispatch_future_backends.rs).\",\n",
            "  \"// rfc\": \"rfcs/0001-cuda-oxide-integration.md -- Unresolved question #3\",\n",
            "  \"// algorithm\": \"nearest-rank percentile per ISO/IEC 25062 over {} sorted samples per backend\",\n",
            "  \"backends\": [\n{}\n  ]\n",
            "}}\n",
        ),
        samples, body,
    )
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

    // ---- finding 7: json_escape edge cases ----

    #[test]
    fn json_escape_named_short_escapes_cr_bs_ff() {
        // \r, \b (U+0008), \f (U+000C) must use their named short escapes,
        // not the generic \uXXXX form.
        assert_eq!(json_escape("a\rb"), "a\\rb");
        assert_eq!(json_escape("a\u{08}b"), "a\\bb");
        assert_eq!(json_escape("a\u{0c}b"), "a\\fb");
        // And as a single combined string, parseable by serde_json.
        let combined = json_escape("\r\u{08}\u{0c}");
        let wrapped = format!("\"{combined}\"");
        let v: serde_json::Value = serde_json::from_str(&wrapped).expect("valid JSON");
        assert_eq!(v, serde_json::Value::String("\r\u{08}\u{0c}".to_string()));
    }

    #[test]
    fn json_escape_passes_multibyte_utf8_through_unchanged() {
        // Multi-byte UTF-8 (BMP + astral) must pass through verbatim — the
        // escaper only touches ASCII control chars, quote, and backslash.
        let s = "héllo — 日本語 — 😀 — Ωμέγα";
        assert_eq!(json_escape(s), s);
        // Round-trips as a JSON string value.
        let wrapped = format!("\"{}\"", json_escape(s));
        let v: serde_json::Value = serde_json::from_str(&wrapped).expect("valid JSON");
        assert_eq!(v, serde_json::Value::String(s.to_string()));
    }

    // ---- finding 7: render_file document-level validity ----

    fn sample_tail_result(metric: &str, backend: &'static str) -> TailResult {
        TailResult {
            metric: metric.to_string(),
            backend,
            samples: 10_000,
            p50_ns: 1,
            p95_ns: 2,
            p99_ns: 3,
            p99_9_ns: 4,
            max_ns: 5,
        }
    }

    #[test]
    fn tail_results_document_is_valid_json_with_metrics_array() {
        let results = vec![
            sample_tail_result("dispatch/serial/100", "unified-memory"),
            sample_tail_result("e2e/healthz/get", "unified-memory"),
        ];
        let doc = render_tail_results_document(&results, "unified-memory", 10_000);
        // Whole document parses — proves the comma-joining + header lines
        // produce a valid object, not just valid per-row fragments.
        let v: serde_json::Value = serde_json::from_str(&doc).expect("valid JSON document");
        assert_eq!(v["backend"], "unified-memory");
        let metrics = v["metrics"].as_array().expect("metrics is an array");
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0]["metric"], "dispatch/serial/100");
        assert_eq!(metrics[1]["metric"], "e2e/healthz/get");
        assert_eq!(metrics[0]["p99_9_ns"], 4);
    }

    #[test]
    fn tail_results_document_escapes_backend_label() {
        // A pathological backend label with a quote must not break the doc.
        let results = vec![sample_tail_result("dispatch/serial/100", "unified-memory")];
        let doc = render_tail_results_document(&results, "we\"ird", 10_000);
        let v: serde_json::Value = serde_json::from_str(&doc).expect("valid JSON document");
        assert_eq!(v["backend"], "we\"ird");
    }

    #[test]
    fn backend_document_is_valid_json_with_backends_array() {
        let lines = vec![
            TailStats {
                backend: "busy-poll".to_string(),
                samples: 10_000,
                p50_ns: 1,
                p95_ns: 2,
                p99_ns: 3,
                p99_9_ns: 4,
                max_ns: 5,
            }
            .to_json(),
            // A skip line shaped like the bench's own skip_json output.
            "{\"backend\":\"cuda-async\",\"status\":\"skipped\",\"reason\":\"unavailable\"}"
                .to_string(),
        ];
        let doc = render_backend_document(&lines, 10_000);
        let v: serde_json::Value = serde_json::from_str(&doc).expect("valid JSON document");
        let backends = v["backends"].as_array().expect("backends is an array");
        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0]["status"], "measured");
        assert_eq!(backends[1]["status"], "skipped");
    }

    // ---- finding 7: workspace_bench_results_dir ----

    #[test]
    fn workspace_bench_results_dir_resolves_or_returns_none() {
        // The function must never panic and must return either a real
        // directory or `None`. (Whether `bench-results/` exists depends on the
        // checkout; both outcomes are valid, but a returned path must be a
        // directory that ends in `bench-results`.)
        match workspace_bench_results_dir() {
            Some(dir) => {
                assert!(dir.is_dir(), "returned path must be an existing directory");
                assert_eq!(
                    dir.file_name().and_then(|n| n.to_str()),
                    Some("bench-results"),
                    "returned path must be the bench-results directory"
                );
            }
            None => { /* acceptable: no bench-results/ in this checkout */ }
        }
    }

    // ---- finding 7: BACKEND_LABEL cfg-priority ordering ----

    #[test]
    fn backend_label_priority_is_cuda_oxide_then_cudarc_then_unified() {
        // cuda-oxide wins over everything.
        assert_eq!(resolve_backend_label(true, true), "cuda-oxide");
        assert_eq!(resolve_backend_label(true, false), "cuda-oxide");
        // cudarc wins only when cuda-oxide is off.
        assert_eq!(resolve_backend_label(false, true), "cudarc");
        // Neither => the historical default.
        assert_eq!(resolve_backend_label(false, false), "unified-memory");
    }

    #[test]
    fn backend_label_matches_active_feature_set() {
        // The live label must equal what the priority function returns for the
        // compiled feature set — pins the const in `tail_latency.rs` to the
        // tested logic.
        let expected = resolve_backend_label(
            cfg!(feature = "cuda-oxide-backend"),
            cfg!(feature = "cudarc-backend"),
        );
        assert_eq!(backend_label(), expected);
    }

    // ---- finding 6: regression-gate comparator core ----

    #[test]
    fn classify_metric_gates_headline_diagnoses_the_rest() {
        assert_eq!(classify_metric("dispatch/serial/100"), MetricClass::Gated);
        assert_eq!(
            classify_metric("dispatch/concurrent_cap64/100"),
            MetricClass::Gated
        );
        assert_eq!(classify_metric("e2e/healthz/get"), MetricClass::Gated);
        assert_eq!(classify_metric("jit_compile/emit_text"), MetricClass::Gated);
        assert_eq!(classify_metric("cold_start/capture"), MetricClass::Gated);
        // Diagnostic markers.
        assert_eq!(
            classify_metric("dispatch/real_sync/1"),
            MetricClass::Diagnostic
        );
        assert_eq!(
            classify_metric("dispatch/contended_cap1/1000"),
            MetricClass::Diagnostic
        );
        assert_eq!(
            classify_metric("cold_start/restore_steady_state"),
            MetricClass::Diagnostic
        );
        assert_eq!(
            classify_metric("e2e/secured/invoke_not_found_authed"),
            MetricClass::Diagnostic
        );
        // Unknown defaults to diagnostic (fail-open for the gate).
        assert_eq!(
            classify_metric("some/brand_new_metric"),
            MetricClass::Diagnostic
        );
    }

    #[test]
    fn check_regression_fails_only_gated_metrics_over_threshold() {
        // Gated + 20% slower at a 10% threshold => failing.
        let c = check_regression("dispatch/serial/100", 100.0, 120.0, 0.10);
        assert_eq!(c.class, MetricClass::Gated);
        assert!((c.delta_frac - 0.20).abs() < 1e-9);
        assert!(c.is_failing_regression);

        // Gated but within threshold => not failing.
        let c = check_regression("dispatch/serial/100", 100.0, 105.0, 0.10);
        assert!(!c.is_failing_regression);

        // Gated and faster => not failing.
        let c = check_regression("dispatch/serial/100", 100.0, 80.0, 0.10);
        assert!(c.delta_frac < 0.0);
        assert!(!c.is_failing_regression);

        // Diagnostic metric, even a huge regression, never fails.
        let c = check_regression("dispatch/real_sync/1", 100.0, 1000.0, 0.10);
        assert_eq!(c.class, MetricClass::Diagnostic);
        assert!(!c.is_failing_regression);

        // Zero/placeholder baseline => never fails, delta 0.
        let c = check_regression("dispatch/serial/100", 0.0, 999.0, 0.10);
        assert_eq!(c.delta_frac, 0.0);
        assert!(!c.is_failing_regression);
    }

    #[test]
    #[should_panic(expected = "threshold_frac must be non-negative")]
    fn check_regression_rejects_negative_threshold() {
        let _ = check_regression("dispatch/serial/100", 100.0, 110.0, -0.1);
    }
}
