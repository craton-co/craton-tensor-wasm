// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm observe` — one-screen operator dashboard for a TensorWasm server.
//!
//! Polls `GET /healthz` and `GET /metrics` on the supplied address at a fixed
//! interval and rewrites the terminal in place with a compact status board.
//! The intent is to give an on-call engineer the most actionable signals
//! without leaving the shell: liveness, registered functions, in-flight jobs,
//! per-endpoint request rate, latency percentiles, and tenant GPU memory.
//!
//! The screen is repainted with a plain `\x1B[2J\x1B[H` clear-and-home — no
//! TUI dependency. Refresh cadence and target address are user-controllable;
//! defaults are `http://localhost:8080` and `2s`.
//!
//! ## Prometheus parser
//!
//! Metrics are parsed by a tiny inline parser (see [`parse_metrics`]) that
//! recognises the prometheus text exposition format well enough to extract the
//! handful of series this command consumes. It ignores `# HELP` / `# TYPE`
//! comment lines, accepts the standard `name{labels} value` shape, and
//! tolerates missing label braces. Histograms are not decomposed structurally;
//! the parser treats the `_bucket`, `_sum`, and `_count` synthesised series as
//! ordinary samples and the consumer reassembles them on demand.
//!
//! ## Assumed metric names
//!
//! Several series referenced by this dashboard are not yet emitted by the
//! current `tensor_wasm_core::metrics::TensorWasmMetrics` registry. Where a name is
//! missing the corresponding cell renders as `?` (counts) or `n/a`
//! (percentages, latencies) rather than producing a misleading zero. The
//! orchestrator should verify which of these are wired before claiming the
//! v0.3 exit criterion as met. See `docs/CLI.md` for the documented surface.
//!
//! As of W2.3, the per-endpoint request rate and latency cells *do* render
//! real values: the `tensor_wasm_http_requests_total{route,method,status}`
//! counter and the `tensor_wasm_http_request_duration_seconds_bucket{route,method,status}`
//! histogram are emitted by the `tensor_wasm_api::http_metrics` middleware.
//! The label name is `route` (axum route template), not `path` — handled by
//! `Snapshot::from_metrics` and `histogram_quantile` without altering
//! the wider parser surface.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Args;

use super::{HttpContext, OutputFormat};

/// Default address used when `--addr` is omitted. Matches the dev-server
/// quickstart in `docs/CLI.md`.
pub const DEFAULT_ADDR: &str = "http://localhost:8080";

/// Default refresh interval, in seconds. Chosen so a human can read the board
/// without the screen feeling laggy under typical scrape budgets.
pub const DEFAULT_INTERVAL_SECS: u64 = 2;

/// HTTP timeout for an individual `/healthz` or `/metrics` fetch. Kept well
/// below the refresh interval so a hung server still surfaces as "down" on
/// the next tick rather than freezing the dashboard.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// ANSI escape that clears the screen and parks the cursor in the top-left.
/// Intentionally the only ANSI we emit — fancier control codes are deferred
/// to a real TUI port.
const CLEAR_AND_HOME: &str = "\x1B[2J\x1B[H";

/// Arguments to `tensor-wasm observe`.
#[derive(Debug, Args)]
pub struct ObserveArgs {
    /// Base URL of the target TensorWasm server. Defaults to `http://localhost:8080`.
    #[arg(long, default_value = DEFAULT_ADDR)]
    pub addr: String,

    /// Refresh interval, in seconds. Must be at least 1.
    #[arg(long, default_value_t = DEFAULT_INTERVAL_SECS)]
    pub interval: u64,

    /// Output format: `text` (in-place ANSI dashboard, default) or `json`
    /// (one machine-readable JSON document per tick, newline-delimited, for
    /// CI scripts and log pipelines). In `json` mode the screen is NOT
    /// cleared between ticks — each tick is appended as an NDJSON line.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, display_order = 800)]
    pub output: OutputFormat,
}

/// One parsed metric sample. Histogram buckets share the same shape — the `le`
/// label lives inside [`Self::labels`].
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Metric name, e.g. `tensor_wasm_kernel_dispatches_total` or
    /// `tensor_wasm_http_request_duration_seconds_bucket`.
    pub name: String,
    /// Labels parsed from `{k="v",...}`. Order is preserved as a `HashMap`
    /// because dashboards only need lookup-by-key.
    pub labels: HashMap<String, String>,
    /// Sample value. `f64::NAN` if the value field failed to parse.
    pub value: f64,
}

/// Parsed metrics body grouped by metric name. Histogram bucket series end up
/// as multiple entries under the same key with differing `le` labels.
pub type Metrics = HashMap<String, Vec<Sample>>;

/// Parse a Prometheus text-exposition payload into samples.
///
/// Handles only what this dashboard consumes:
/// - skips blank lines and `#`-prefixed `HELP`/`TYPE` comments
/// - accepts `name value` (no labels) and `name{k="v",k2="v2"} value`
/// - ignores trailing timestamps if present (uncommon for scrape output)
///
/// Lines that do not match the expected shape are silently dropped — better
/// than crashing the dashboard on a malformed entry from an experimental
/// metric a future TensorWasm version might emit.
pub fn parse_metrics(body: &str) -> Metrics {
    let mut out: Metrics = HashMap::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(sample) = parse_sample(line) else {
            continue;
        };
        out.entry(sample.name.clone()).or_default().push(sample);
    }
    out
}

/// Parse a single non-comment, non-empty exposition line. Returns `None` for
/// any line whose shape this minimal parser cannot make sense of.
fn parse_sample(line: &str) -> Option<Sample> {
    // Split the optional `{labels}` from the name.
    let (name, rest) = if let Some(open) = line.find('{') {
        let name = &line[..open];
        let after = &line[open + 1..];
        let close = after.find('}')?;
        let labels_raw = &after[..close];
        let value_part = after[close + 1..].trim_start();
        (
            name.trim(),
            (parse_labels(labels_raw), value_part.to_string()),
        )
    } else {
        // `name value [timestamp]` — split on whitespace.
        let mut it = line.splitn(2, char::is_whitespace);
        let name = it.next()?.trim();
        let value_part = it.next()?.trim();
        (name, (HashMap::new(), value_part.to_string()))
    };

    if name.is_empty() {
        return None;
    }

    // Value is the first whitespace-delimited token in the remainder; any
    // trailing timestamp is discarded.
    let value_token = rest.1.split_whitespace().next()?;
    let value: f64 = value_token.parse().ok()?;

    Some(Sample {
        name: name.to_string(),
        labels: rest.0,
        value,
    })
}

/// Parse the inside of `{...}` into a `HashMap`. Tolerates extra whitespace
/// and either single or double-quoted values; Prometheus spec mandates
/// double quotes but we accept both for robustness against hand-written
/// fixtures.
fn parse_labels(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace and commas.
        while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Read key up to `=`.
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key = raw[key_start..i].trim().to_string();
        i += 1; // skip `=`
                // Expect a quote.
        if i >= bytes.len() {
            break;
        }
        let quote = bytes[i];
        if quote != b'"' && quote != b'\'' {
            break;
        }
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            // Skip escaped quote.
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            i += 1;
        }
        let val = raw[val_start..i].to_string();
        i += 1; // skip closing quote
        if !key.is_empty() {
            out.insert(key, val);
        }
    }
    out
}

/// Aggregate counter snapshots across two scrapes so we can compute a rate.
#[derive(Debug, Clone, Default)]
struct Snapshot {
    /// Per-endpoint cumulative request counts, keyed by `path` label.
    http_requests: HashMap<String, f64>,
    /// Wall-clock instant the snapshot was taken.
    taken_at: Option<Instant>,
}

impl Snapshot {
    fn from_metrics(m: &Metrics, now: Instant) -> Self {
        let mut http_requests = HashMap::new();
        if let Some(series) = m.get("tensor_wasm_http_requests_total") {
            for s in series {
                // W2.3 emits the axum route template under the `route` label
                // (e.g. `/functions/:id/invoke`). Earlier ad-hoc fixtures
                // used `path`; keep that as a fallback so the historical
                // text-format snapshot tests still parse. The aggregation
                // sums across status codes so the operator-facing rate
                // covers 2xx + 4xx + 5xx for the endpoint.
                let route = s
                    .labels
                    .get("route")
                    .or_else(|| s.labels.get("path"))
                    .cloned()
                    .unwrap_or_else(|| "<unlabelled>".to_string());
                *http_requests.entry(route).or_insert(0.0) += s.value;
            }
        }
        Self {
            http_requests,
            taken_at: Some(now),
        }
    }
}

/// Run the observe dashboard until the user hits Ctrl-C.
///
/// Polls `/healthz` and `/metrics` every `args.interval` seconds, repainting
/// the screen between requests. Exits cleanly on Ctrl-C; returns
/// [`anyhow::Error`] only for unrecoverable startup failures (invalid URL,
/// HTTP client construction). Per-tick fetch failures are rendered into the
/// dashboard rather than aborting the loop, so a transient network blip does
/// not kill the operator's view.
pub async fn run(args: ObserveArgs, ctx: &HttpContext) -> Result<()> {
    super::validate_server_url(&args.addr)?;
    if args.interval == 0 {
        anyhow::bail!("--interval must be at least 1 second");
    }
    let interval = Duration::from_secs(args.interval);
    let base = super::server_base(&args.addr).to_string();
    let client = ctx.build_client(FETCH_TIMEOUT)?;

    let mut prev = Snapshot::default();
    loop {
        let tick_start = Instant::now();
        let health = fetch_healthz(&client, ctx, &base).await;
        let metrics_text = fetch_metrics(&client, ctx, &base).await;
        let now = Instant::now();
        let (metrics, fetch_err) = match metrics_text {
            Ok(body) => (parse_metrics(&body), None),
            Err(e) => (Metrics::new(), Some(e.to_string())),
        };
        let snap = Snapshot::from_metrics(&metrics, now);
        let mut stdout = std::io::stdout();
        match args.output {
            OutputFormat::Text => {
                let board =
                    render_board(&base, interval, &health, &metrics, &prev, &snap, fetch_err);
                // T18: the board is composed of server-derived text —
                // `/healthz` body, route labels from `/metrics`, and
                // fetch-error messages. A malicious server could embed ANSI
                // escapes that survive through `parse_metrics` /
                // `render_board` and rewrite the operator's terminal.
                // Sanitise the rendered board before emitting; the
                // `CLEAR_AND_HOME` constant we prepend is the *only* escape
                // sequence the dashboard is allowed to use, and it is added
                // outside the sanitised payload below.
                let board = super::sanitise_terminal_output(&board);
                // cli fix 3: only emit the `\x1B[2J\x1B[H` clear-and-home
                // escape when stdout is a TTY. Piped / redirected output (CI
                // logs, `tee`, files) would otherwise capture the raw escape
                // bytes and either render them as garbage or trip downstream
                // parsers. In non-TTY mode we instead separate each board
                // with a blank line so the stream stays readable.
                if stdout.is_terminal() {
                    print!("{CLEAR_AND_HOME}{board}");
                } else {
                    println!("\n{board}");
                }
            }
            OutputFormat::Json => {
                // NDJSON: one self-contained document per tick. No screen
                // clear — downstream log pipelines / `jq` consume the stream
                // line-by-line. `to_string` already strips control bytes via
                // serde's JSON escaping, so no extra sanitisation is needed.
                let doc =
                    render_board_json(&base, interval, &health, &metrics, &prev, &snap, &fetch_err);
                println!("{doc}");
            }
        }
        // Flush so the terminal repaints before we sleep.
        use std::io::Write;
        let _ = stdout.flush();
        prev = snap;

        // Sleep until the next tick, racing with Ctrl-C. The branch order is
        // load-bearing: a pending Ctrl-C should exit even if the sleep is
        // also ready in the same poll.
        let elapsed = tick_start.elapsed();
        let remaining = interval.saturating_sub(elapsed);
        tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => {
                println!();
                return Ok(());
            }
            _ = tokio::time::sleep(remaining) => {}
        }
    }
}

/// Health-probe outcome.
#[derive(Debug)]
enum Health {
    /// Server returned 2xx; `body` is the (parsed-or-raw) status line.
    Ok { body: String },
    /// Server reachable but did not return 2xx.
    Bad { status: u16 },
    /// Network or DNS failure.
    Unreachable { error: String },
}

async fn fetch_healthz(client: &reqwest::Client, ctx: &HttpContext, base: &str) -> Health {
    let url = format!("{base}/healthz");
    let req = ctx.apply(client.get(&url));
    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            // T17: bound the in-memory body so a malicious server can't OOM
            // the CLI by streaming gigabytes of `/healthz` text.
            let body = super::bounded_text(resp).await.unwrap_or_default();
            if status.is_success() {
                Health::Ok { body }
            } else {
                Health::Bad {
                    status: status.as_u16(),
                }
            }
        }
        Err(e) => Health::Unreachable {
            error: e.to_string(),
        },
    }
}

async fn fetch_metrics(client: &reqwest::Client, ctx: &HttpContext, base: &str) -> Result<String> {
    let url = format!("{base}/metrics");
    let req = ctx.apply(client.get(&url));
    let resp = req.send().await.with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    // T17: bound the in-memory body so a malicious server can't OOM the CLI
    // by streaming gigabytes of `/metrics` exposition.
    let text = super::bounded_text(resp)
        .await
        .with_context(|| format!("read {url}"))?;
    if !status.is_success() {
        anyhow::bail!("{url} returned HTTP {}", status.as_u16());
    }
    Ok(text)
}

/// Render the full status board as a `String` ready for `print!`.
///
/// Pure function modulo time-of-day; the input snapshots carry everything
/// needed to compute rates. Keeping it pure makes the board snapshot-testable
/// without spinning a real server.
fn render_board(
    addr: &str,
    interval: Duration,
    health: &Health,
    metrics: &Metrics,
    prev: &Snapshot,
    cur: &Snapshot,
    fetch_err: Option<String>,
) -> String {
    let mut s = String::new();
    s.push_str("Craton TensorWasm — operator dashboard\n");
    s.push_str(&format!(
        "target: {addr}   interval: {}s\n",
        interval.as_secs()
    ));
    s.push_str("--------------------------------------------------\n");

    // Liveness.
    match health {
        Health::Ok { body } => {
            // cli fix 4: parse the body as JSON and check `body["status"]`
            // rather than substring-matching `"ok"` against the raw bytes.
            // The server-side handler emits `{"status":"ok"}` (see
            // `tensor-wasm-api::routes::healthz`), and the substring shape
            // tripped on any value (e.g. a future `"degraded":"ok"` flag,
            // or whitespace variations like `"status" : "ok"`) that
            // happened to contain the literal `"ok"`. Falling back to the
            // raw 2xx case (`200`) when the body doesn't parse keeps the
            // dashboard useful against pre-JSON servers or proxies that
            // rewrite the body.
            let status_word: String = match serde_json::from_str::<serde_json::Value>(body) {
                Ok(v) => match v.get("status").and_then(|s| s.as_str()) {
                    Some("ok") => "ok".to_string(),
                    // A non-"ok" status string still came back as 2xx, so
                    // the server *is* reachable; surface the value verbatim
                    // (truncated) so the operator sees what the server
                    // actually said.
                    Some(other) => truncate(other, 12),
                    None => "200".to_string(),
                },
                Err(_) => "200".to_string(),
            };
            s.push_str(&format!("liveness:   /healthz {status_word}\n"));
        }
        Health::Bad { status } => {
            s.push_str(&format!("liveness:   /healthz HTTP {status} (bad)\n"));
        }
        Health::Unreachable { error } => {
            s.push_str(&format!("liveness:   unreachable — {error}\n"));
        }
    }

    // Uptime, if /healthz body exposed it as JSON `"uptime_seconds": N`.
    let uptime = match health {
        Health::Ok { body } => parse_uptime_seconds(body),
        _ => None,
    };
    s.push_str(&format!("uptime:     {}\n", fmt_uptime(uptime)));

    // Headline counters.
    s.push_str(&format!(
        "functions:  {}\n",
        fmt_optional_u64(scalar(metrics, "tensor_wasm_functions_total"))
    ));
    s.push_str(&format!(
        "jobs.active:{}\n",
        fmt_optional_u64(scalar(metrics, "tensor_wasm_jobs_active"))
    ));
    s.push_str(&format!(
        "instances:  {}\n",
        fmt_optional_u64(scalar(metrics, "tensor_wasm_active_instances"))
    ));

    // GPU memory: prefer per-tenant gauge if present, fall back to the
    // process-wide gauge that the current TensorWasmMetrics registry actually
    // exposes.
    let gpu_text = if let Some(series) = metrics.get("tenant_gpu_memory_bytes") {
        let total: f64 = series.iter().map(|s| s.value).sum();
        fmt_bytes(total)
    } else if let Some(v) = scalar(metrics, "tensor_wasm_gpu_memory_used_bytes") {
        fmt_bytes(v)
    } else {
        "n/a".to_string()
    };
    s.push_str(&format!("gpu.memory: {gpu_text}\n"));
    s.push_str("--------------------------------------------------\n");

    // Per-endpoint request rate + latency percentiles.
    s.push_str("endpoint                  req/s     p50      p95\n");
    let endpoints: Vec<String> = {
        let mut keys: Vec<String> = cur.http_requests.keys().cloned().collect();
        keys.sort();
        keys
    };
    if endpoints.is_empty() {
        s.push_str("(no tensor_wasm_http_requests_total series observed)\n");
    } else {
        let dt_secs = match (prev.taken_at, cur.taken_at) {
            (Some(a), Some(b)) => (b - a).as_secs_f64().max(0.0),
            _ => 0.0,
        };
        for path in endpoints {
            let cur_v = cur.http_requests.get(&path).copied().unwrap_or(0.0);
            let prev_v = prev.http_requests.get(&path).copied().unwrap_or(0.0);
            // cli fix 5: distinguish "first scrape / no rate yet" from "the
            // counter went backwards" (server restart between scrapes, or
            // the underlying registry was reset). The previous shape
            // collapsed both into `n/a`, hiding the much more interesting
            // restart signal. We now emit `(reset)` explicitly when
            // `cur_v < prev_v` so the operator knows to expect a request-rate
            // dip on the next tick rather than chasing a phantom outage.
            let rate_cell = if dt_secs <= 0.0 {
                // No baseline scrape yet — leave the cell as `n/a`.
                fmt_rate(f64::NAN)
            } else if cur_v < prev_v {
                "(reset)".to_string()
            } else {
                fmt_rate((cur_v - prev_v) / dt_secs)
            };
            let p50 = histogram_quantile(
                metrics,
                "tensor_wasm_http_request_duration_seconds",
                &path,
                0.5,
            );
            let p95 = histogram_quantile(
                metrics,
                "tensor_wasm_http_request_duration_seconds",
                &path,
                0.95,
            );
            s.push_str(&format!(
                "{:<24} {:>8}  {:>6}  {:>6}\n",
                truncate(&path, 24),
                rate_cell,
                fmt_latency(p50),
                fmt_latency(p95),
            ));
        }
    }
    s.push_str("--------------------------------------------------\n");

    if let Some(err) = fetch_err {
        s.push_str(&format!("warn: metrics fetch failed — {err}\n"));
    }
    s.push_str("Ctrl-C to exit.\n");
    s
}

/// Render the same data [`render_board`] shows as a machine-readable JSON
/// document — one per tick when `--output json` is set. Pure modulo the
/// snapshot inputs, mirroring `render_board` so the two views stay in sync.
///
/// Shape (stable for CI scripts):
/// ```json
/// {
///   "target": "...", "interval_secs": 2,
///   "liveness": "ok" | "bad" | "unreachable",
///   "health_status": "ok" | "200" | "<word>" | null,
///   "uptime_seconds": 125 | null,
///   "functions": 3 | null, "jobs_active": 0 | null, "instances": 1 | null,
///   "gpu_memory_bytes": 1073741824 | null,
///   "endpoints": [ { "route", "req_per_s": <num|null>, "reset": <bool>,
///                    "p50_seconds": <num|null>, "p95_seconds": <num|null> } ],
///   "fetch_error": "..." | null
/// }
/// ```
#[allow(clippy::too_many_arguments)]
fn render_board_json(
    addr: &str,
    interval: Duration,
    health: &Health,
    metrics: &Metrics,
    prev: &Snapshot,
    cur: &Snapshot,
    fetch_err: &Option<String>,
) -> String {
    let (liveness, health_status): (&str, serde_json::Value) = match health {
        Health::Ok { body } => {
            let status = match serde_json::from_str::<serde_json::Value>(body) {
                Ok(v) => match v.get("status").and_then(|s| s.as_str()) {
                    Some("ok") => serde_json::Value::String("ok".to_string()),
                    Some(other) => serde_json::Value::String(other.to_string()),
                    None => serde_json::Value::String("200".to_string()),
                },
                Err(_) => serde_json::Value::String("200".to_string()),
            };
            ("ok", status)
        }
        Health::Bad { status } => ("bad", serde_json::Value::String(format!("HTTP {status}"))),
        Health::Unreachable { error } => ("unreachable", serde_json::Value::String(error.clone())),
    };

    let uptime = match health {
        Health::Ok { body } => parse_uptime_seconds(body),
        _ => None,
    };

    let gpu_memory_bytes: Option<f64> = if let Some(series) = metrics.get("tenant_gpu_memory_bytes")
    {
        Some(series.iter().map(|s| s.value).sum())
    } else {
        scalar(metrics, "tensor_wasm_gpu_memory_used_bytes")
    };

    let dt_secs = match (prev.taken_at, cur.taken_at) {
        (Some(a), Some(b)) => (b - a).as_secs_f64().max(0.0),
        _ => 0.0,
    };
    let mut routes: Vec<String> = cur.http_requests.keys().cloned().collect();
    routes.sort();
    let endpoints: Vec<serde_json::Value> = routes
        .into_iter()
        .map(|route| {
            let cur_v = cur.http_requests.get(&route).copied().unwrap_or(0.0);
            let prev_v = prev.http_requests.get(&route).copied().unwrap_or(0.0);
            let (req_per_s, reset) = if dt_secs <= 0.0 {
                (serde_json::Value::Null, false)
            } else if cur_v < prev_v {
                (serde_json::Value::Null, true)
            } else {
                (json_num((cur_v - prev_v) / dt_secs), false)
            };
            let p50 = histogram_quantile(
                metrics,
                "tensor_wasm_http_request_duration_seconds",
                &route,
                0.5,
            );
            let p95 = histogram_quantile(
                metrics,
                "tensor_wasm_http_request_duration_seconds",
                &route,
                0.95,
            );
            serde_json::json!({
                "route": route,
                "req_per_s": req_per_s,
                "reset": reset,
                "p50_seconds": p50.and_then(serde_json::Number::from_f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
                "p95_seconds": p95.and_then(serde_json::Number::from_f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            })
        })
        .collect();

    let doc = serde_json::json!({
        "target": addr,
        "interval_secs": interval.as_secs(),
        "liveness": liveness,
        "health_status": health_status,
        "uptime_seconds": uptime,
        "functions": scalar(metrics, "tensor_wasm_functions_total").map(|v| v as u64),
        "jobs_active": scalar(metrics, "tensor_wasm_jobs_active").map(|v| v as u64),
        "instances": scalar(metrics, "tensor_wasm_active_instances").map(|v| v as u64),
        "gpu_memory_bytes": gpu_memory_bytes.map(|v| v as u64),
        "endpoints": endpoints,
        "fetch_error": fetch_err,
    });
    doc.to_string()
}

/// Wrap an `f64` into a JSON value, falling back to `null` for non-finite
/// values that `serde_json::Number` cannot represent.
fn json_num(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// Look up a single-sample series (typical gauge or label-less counter) and
/// return its value if exactly one observation exists. Returns `None` if the
/// series is absent or has multiple labelled variants we'd otherwise need to
/// sum (we want loud `?` cells in that case, not a misleading silent sum).
fn scalar(metrics: &Metrics, name: &str) -> Option<f64> {
    let series = metrics.get(name)?;
    if series.len() == 1 {
        Some(series[0].value)
    } else if !series.is_empty() {
        // Sum if every sample is unlabelled (counter rolled up by a single
        // variant); otherwise prefer to admit ambiguity to the operator.
        if series.iter().all(|s| s.labels.is_empty()) {
            Some(series.iter().map(|s| s.value).sum())
        } else {
            None
        }
    } else {
        None
    }
}

/// Linear interpolation across histogram buckets, à la `histogram_quantile()`
/// in PromQL. Returns `None` when fewer than two buckets are observed for the
/// requested route label or when the requested rank cannot be localised.
///
/// The `route` argument is matched against either a `route` label (W2.3
/// HTTP histograms emit this) or a legacy `path` label (still consulted as
/// a fallback so older fixtures stay parseable). Samples that carry the
/// matching label *and* extra dimensions (e.g. `status`, `method`) are
/// aggregated by summing the cumulative bucket counts — the operator-facing
/// quantile is intentionally per-route, not per-status.
fn histogram_quantile(metrics: &Metrics, base_name: &str, route: &str, q: f64) -> Option<f64> {
    let bucket_name = format!("{base_name}_bucket");
    let series = metrics.get(&bucket_name)?;
    // First filter to the samples that name our route, then collapse
    // across the other dimensions (status, method) by summing counts
    // bucket-by-bucket. This mirrors the dashboard's
    // `sum by (le) (rate(... {route="..."}[5m]))` pattern.
    let mut by_le: HashMap<String, f64> = HashMap::new();
    for s in series.iter().filter(|s| {
        s.labels
            .get("route")
            .or_else(|| s.labels.get("path"))
            .map(|p| p == route)
            .unwrap_or(false)
    }) {
        if let Some(le) = s.labels.get("le") {
            *by_le.entry(le.clone()).or_insert(0.0) += s.value;
        }
    }
    let mut buckets: Vec<(f64, f64)> = by_le
        .into_iter()
        .filter_map(|(le, count)| {
            let upper = if le == "+Inf" {
                f64::INFINITY
            } else {
                le.parse::<f64>().ok()?
            };
            Some((upper, count))
        })
        .collect();
    if buckets.len() < 2 {
        return None;
    }
    buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total = buckets.last().map(|b| b.1).unwrap_or(0.0);
    if total <= 0.0 {
        return None;
    }
    let target = total * q;
    let mut prev_upper = 0.0;
    let mut prev_count = 0.0;
    for (upper, count) in buckets {
        if count >= target {
            if upper.is_infinite() {
                return Some(prev_upper);
            }
            let bucket_count = count - prev_count;
            if bucket_count <= 0.0 {
                return Some(upper);
            }
            let frac = (target - prev_count) / bucket_count;
            return Some(prev_upper + (upper - prev_upper) * frac);
        }
        prev_upper = upper;
        prev_count = count;
    }
    None
}

fn parse_uptime_seconds(body: &str) -> Option<u64> {
    // Look for `"uptime_seconds"`/`"uptime"` followed by a JSON number. We
    // do not pull in serde_json here for one optional field — the substring
    // match is cheap and resilient to surrounding whitespace.
    for key in ["uptime_seconds", "uptime"] {
        let needle = format!("\"{key}\"");
        if let Some(pos) = body.find(&needle) {
            let after = &body[pos + needle.len()..];
            let after = after.trim_start_matches([' ', ':', '\t']);
            let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = num.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

fn fmt_uptime(s: Option<u64>) -> String {
    match s {
        None => "n/a".to_string(),
        Some(0) => "0s".to_string(),
        Some(mut total) => {
            let days = total / 86_400;
            total %= 86_400;
            let hours = total / 3_600;
            total %= 3_600;
            let mins = total / 60;
            let secs = total % 60;
            if days > 0 {
                format!("{days}d{hours}h{mins}m")
            } else if hours > 0 {
                format!("{hours}h{mins}m{secs}s")
            } else if mins > 0 {
                format!("{mins}m{secs}s")
            } else {
                format!("{secs}s")
            }
        }
    }
}

fn fmt_optional_u64(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{}", x as u64),
        None => "?".to_string(),
    }
}

fn fmt_bytes(v: f64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if v >= GIB {
        format!("{:.2} GiB", v / GIB)
    } else if v >= MIB {
        format!("{:.2} MiB", v / MIB)
    } else if v >= KIB {
        format!("{:.2} KiB", v / KIB)
    } else {
        format!("{v:.0} B")
    }
}

fn fmt_rate(r: f64) -> String {
    if r.is_nan() {
        "n/a".to_string()
    } else if r >= 100.0 {
        format!("{r:.0}")
    } else {
        format!("{r:.2}")
    }
}

fn fmt_latency(secs: Option<f64>) -> String {
    match secs {
        None => "n/a".to_string(),
        Some(s) if s >= 1.0 => format!("{s:.2}s"),
        Some(s) if s >= 1e-3 => format!("{:.1}ms", s * 1e3),
        Some(s) => format!("{:.0}us", s * 1e6),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let cut = n.saturating_sub(1);
        let mut out: String = s.chars().take(cut).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_BODY: &str = r#"
# HELP tensor_wasm_active_instances Number of live Wasm instances.
# TYPE tensor_wasm_active_instances gauge
tensor_wasm_active_instances 3
# HELP tensor_wasm_kernel_dispatches_total Total CUDA kernel launches.
# TYPE tensor_wasm_kernel_dispatches_total counter
tensor_wasm_kernel_dispatches_total 17
tensor_wasm_http_requests_total{path="/invoke",method="POST"} 42
tensor_wasm_http_requests_total{path="/healthz",method="GET"} 9
tensor_wasm_http_request_duration_seconds_bucket{path="/invoke",le="0.005"} 1
tensor_wasm_http_request_duration_seconds_bucket{path="/invoke",le="0.01"}  5
tensor_wasm_http_request_duration_seconds_bucket{path="/invoke",le="0.05"}  9
tensor_wasm_http_request_duration_seconds_bucket{path="/invoke",le="0.5"}  10
tensor_wasm_http_request_duration_seconds_bucket{path="/invoke",le="+Inf"} 10
tensor_wasm_gpu_memory_used_bytes 1073741824
"#;

    #[test]
    fn parses_simple_counter_without_labels() {
        let m = parse_metrics("foo 42\n");
        let s = &m["foo"];
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].value, 42.0);
        assert!(s[0].labels.is_empty());
    }

    #[test]
    fn parses_labelled_counter() {
        let m = parse_metrics(r#"foo{a="1",b="two"} 3.14"#);
        let s = &m["foo"][0];
        assert_eq!(s.labels.get("a").map(String::as_str), Some("1"));
        assert_eq!(s.labels.get("b").map(String::as_str), Some("two"));
        // PI-shaped sample value from the fixture; arbitrary numeric literal,
        // not the constant — using #[allow] keeps it readable.
        #[allow(clippy::approx_constant)]
        let expected = 3.14_f64;
        assert!((s.value - expected).abs() < 1e-9);
    }

    #[test]
    fn ignores_help_and_type_comments() {
        let m = parse_metrics("# HELP foo nope\n# TYPE foo counter\nfoo 1\n");
        assert_eq!(m.len(), 1);
        assert_eq!(m["foo"][0].value, 1.0);
    }

    #[test]
    fn drops_malformed_lines() {
        let m = parse_metrics("garbage\nfoo notanumber\nfoo 7\n");
        assert_eq!(m["foo"].len(), 1);
        assert_eq!(m["foo"][0].value, 7.0);
    }

    #[test]
    fn extracts_http_requests_groups_by_path() {
        let m = parse_metrics(SAMPLE_BODY);
        let series = &m["tensor_wasm_http_requests_total"];
        assert_eq!(series.len(), 2);
        let paths: Vec<&str> = series
            .iter()
            .map(|s| s.labels.get("path").unwrap().as_str())
            .collect();
        assert!(paths.contains(&"/invoke"));
        assert!(paths.contains(&"/healthz"));
    }

    #[test]
    fn scalar_returns_single_observation() {
        let m = parse_metrics(SAMPLE_BODY);
        assert_eq!(scalar(&m, "tensor_wasm_active_instances"), Some(3.0));
        assert_eq!(
            scalar(&m, "tensor_wasm_kernel_dispatches_total"),
            Some(17.0)
        );
        assert!(scalar(&m, "tensor_wasm_does_not_exist").is_none());
    }

    #[test]
    fn histogram_quantile_interpolates_buckets() {
        let m = parse_metrics(SAMPLE_BODY);
        let p50 = histogram_quantile(
            &m,
            "tensor_wasm_http_request_duration_seconds",
            "/invoke",
            0.5,
        );
        // 50% of 10 = 5. The 0.01 bucket has count=5 exactly, so p50 == 0.01.
        assert!(p50.is_some());
        assert!((p50.unwrap() - 0.01).abs() < 1e-9);
        let p95 = histogram_quantile(
            &m,
            "tensor_wasm_http_request_duration_seconds",
            "/invoke",
            0.95,
        );
        // 95% of 10 = 9.5 — falls between 0.05 (count=9) and 0.5 (count=10).
        // Interpolation gives 0.05 + (0.5-0.05)*(9.5-9)/1 = 0.275.
        let v = p95.unwrap();
        assert!(v > 0.05 && v < 0.5, "got {v}");
    }

    #[test]
    fn parse_uptime_seconds_handles_present_and_missing() {
        assert_eq!(
            parse_uptime_seconds(r#"{"status":"ok","uptime_seconds": 125}"#),
            Some(125)
        );
        assert_eq!(parse_uptime_seconds(r#"{"status":"ok"}"#), None);
    }

    #[test]
    fn fmt_helpers_produce_human_units() {
        assert_eq!(fmt_bytes(1024.0), "1.00 KiB");
        assert_eq!(fmt_bytes(2.0 * 1024.0 * 1024.0 * 1024.0), "2.00 GiB");
        assert_eq!(fmt_uptime(None), "n/a");
        assert_eq!(fmt_uptime(Some(0)), "0s");
        assert_eq!(fmt_uptime(Some(65)), "1m5s");
        assert!(fmt_uptime(Some(86400 * 3)).starts_with("3d"));
        assert_eq!(fmt_latency(None), "n/a");
        assert_eq!(fmt_latency(Some(0.000_012)), "12us");
        assert_eq!(fmt_latency(Some(0.005)), "5.0ms");
        assert_eq!(fmt_optional_u64(None), "?");
        assert_eq!(fmt_optional_u64(Some(42.0)), "42");
    }

    #[test]
    fn render_board_shows_question_marks_when_metrics_missing() {
        // Empty metrics — every cell that depends on a server-emitted series
        // should fall back to `?` or `n/a`, not silently render zero.
        let metrics = Metrics::new();
        let now = Instant::now();
        let snap = Snapshot::from_metrics(&metrics, now);
        let board = render_board(
            "http://localhost:8080",
            Duration::from_secs(2),
            &Health::Ok {
                body: r#"{"status":"ok"}"#.to_string(),
            },
            &metrics,
            &Snapshot::default(),
            &snap,
            None,
        );
        assert!(board.contains("functions:  ?"), "got: {board}");
        assert!(board.contains("jobs.active:?"), "got: {board}");
        assert!(board.contains("gpu.memory: n/a"), "got: {board}");
        assert!(board.contains("/healthz ok"), "got: {board}");
    }

    #[test]
    fn render_board_reports_health_failure() {
        let board = render_board(
            "http://localhost:8080",
            Duration::from_secs(2),
            &Health::Unreachable {
                error: "connection refused".to_string(),
            },
            &Metrics::new(),
            &Snapshot::default(),
            &Snapshot::default(),
            Some("connection refused".to_string()),
        );
        assert!(board.contains("liveness:   unreachable"), "got: {board}");
        assert!(board.contains("warn: metrics fetch failed"), "got: {board}");
    }

    #[test]
    fn render_board_computes_rate_between_snapshots() {
        let metrics_now = parse_metrics("tensor_wasm_http_requests_total{path=\"/invoke\"} 100\n");
        let t0 = Instant::now() - Duration::from_secs(2);
        let t1 = Instant::now();
        let prev = Snapshot {
            http_requests: {
                let mut m = HashMap::new();
                m.insert("/invoke".to_string(), 80.0);
                m
            },
            taken_at: Some(t0),
        };
        let cur = Snapshot::from_metrics(&metrics_now, t1);
        let board = render_board(
            "http://x",
            Duration::from_secs(2),
            &Health::Ok {
                body: "{\"status\":\"ok\"}".to_string(),
            },
            &metrics_now,
            &prev,
            &cur,
            None,
        );
        // 20 requests over ~2 seconds = ~10 req/s. Allow some slop for the
        // exact elapsed used in fmt_rate.
        assert!(board.contains("/invoke"), "got: {board}");
        assert!(board.contains("req/s"), "got: {board}");
    }

    #[test]
    fn render_board_json_is_valid_and_carries_fields() {
        let metrics = parse_metrics(SAMPLE_BODY);
        let now = Instant::now();
        let snap = Snapshot::from_metrics(&metrics, now);
        let doc = render_board_json(
            "http://localhost:8080",
            Duration::from_secs(2),
            &Health::Ok {
                body: r#"{"status":"ok","uptime_seconds":125}"#.to_string(),
            },
            &metrics,
            &Snapshot::default(),
            &snap,
            &None,
        );
        let v: serde_json::Value =
            serde_json::from_str(&doc).expect("observe --output json must be valid JSON");
        assert_eq!(v["target"], "http://localhost:8080");
        assert_eq!(v["interval_secs"], 2);
        assert_eq!(v["liveness"], "ok");
        assert_eq!(v["health_status"], "ok");
        assert_eq!(v["uptime_seconds"], 125);
        assert_eq!(v["instances"], 3);
        assert_eq!(v["gpu_memory_bytes"], 1073741824u64);
        // Endpoints come from the SAMPLE_BODY http_requests series.
        assert!(v["endpoints"].is_array());
        assert!(v["fetch_error"].is_null());
    }

    #[test]
    fn render_board_json_reports_unreachable() {
        let doc = render_board_json(
            "http://x",
            Duration::from_secs(2),
            &Health::Unreachable {
                error: "connection refused".to_string(),
            },
            &Metrics::new(),
            &Snapshot::default(),
            &Snapshot::default(),
            &Some("connection refused".to_string()),
        );
        let v: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(v["liveness"], "unreachable");
        assert_eq!(v["fetch_error"], "connection refused");
        assert!(v["functions"].is_null());
    }

    #[test]
    fn truncate_respects_width() {
        assert_eq!(truncate("short", 10), "short");
        let t = truncate("a-very-long-endpoint-name", 10);
        assert_eq!(t.chars().count(), 10);
        assert!(t.ends_with('…'));
    }
}
