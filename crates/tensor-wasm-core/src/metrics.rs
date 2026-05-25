// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Prometheus metrics for the TensorWasm workspace.
//!
//! [`TensorWasmMetrics`] owns one `Registry` and a fixed set of metrics used by the
//! execution engine, the WASI-CUDA bridge, the snapshot subsystem, and the API
//! gateway. Construct it once at process startup and clone individual metric
//! handles into the components that emit them — the underlying atomics are
//! shared.

use std::sync::atomic::{AtomicI64, AtomicU64};
use std::sync::Arc;

use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

/// Default histogram buckets for kernel-launch latency, in seconds.
///
/// Calibrated for the expected range of CUDA kernel dispatches (10 µs–10 s).
/// Override by constructing [`TensorWasmMetrics`] with [`TensorWasmMetrics::with_buckets`].
pub const DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS: [f64; 14] = [
    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0,
];

/// Default histogram buckets for HTTP request duration, in seconds.
///
/// Standard `[1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]`
/// shape — covers the expected operating range of the API gateway, from the
/// ~30 µs floor of `GET /healthz` (with a single 1 ms first bucket as a
/// generous P99 ceiling for liveness probes) up to the 30 s per-request
/// timeout enforced by `tensor-wasm-api`'s middleware stack. Override by
/// constructing [`TensorWasmMetrics`] with [`TensorWasmMetrics::with_http_buckets`]
/// (or the combined [`TensorWasmMetrics::with_all_buckets`] constructor) if a
/// deployment needs finer/coarser resolution for its workload mix.
pub const DEFAULT_HTTP_DURATION_BUCKETS_SECONDS: [f64; 12] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Label set for `tensor_wasm_http_requests_total` and
/// `tensor_wasm_http_request_duration_seconds`.
///
/// Cardinality is bounded by the caller — the HTTP middleware in
/// `tensor-wasm-api` initialises a runtime allow-list of route templates at
/// startup and substitutes `"unknown"` for any unmatched path. `method` is
/// drawn from the small set of HTTP verbs the router accepts (`GET`, `POST`,
/// `DELETE`). `status` is the numeric status code rendered as a string —
/// also bounded by HTTP's three-digit code space.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpRequestLabels {
    /// Axum route template that matched the request (e.g. `/functions/:id/invoke`).
    /// Never the substituted value — see crate-level docs on cardinality.
    pub route: String,
    /// HTTP method (`GET`, `POST`, `DELETE`).
    pub method: String,
    /// Numeric HTTP status code rendered as decimal (e.g. `"200"`, `"401"`).
    pub status: String,
}

/// Label set for `tensor_wasm_http_requests_in_flight`.
///
/// Drops `status` (in-flight requests have not produced a status yet) and
/// keeps `route` + `method` for the same cardinality bound as
/// [`HttpRequestLabels`].
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpInFlightLabels {
    /// Axum route template that matched the request.
    pub route: String,
    /// HTTP method (`GET`, `POST`, `DELETE`).
    pub method: String,
}

/// Label set for `tensor_wasm_build_info`.
///
/// Standard Prometheus "info-style metric" labels: a gauge that is always
/// `1` and whose interesting payload is the label values themselves. The
/// label set is fixed for the lifetime of the process — there is exactly
/// one series, primed at registry construction — so cardinality is
/// trivially bounded. The values come from compile-time `env!()` lookups
/// driven by `tensor-wasm-core/build.rs`; see that script for the
/// source-of-truth contract per field.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BuildInfoLabels {
    /// Crate / binary semver, taken from `CARGO_PKG_VERSION` at compile
    /// time (the workspace-wide `[workspace.package] version` pin).
    pub version: String,
    /// `git rev-parse HEAD` at build time, or `"unknown"` for source
    /// tarballs and hermetic builds without `git` on `PATH`.
    pub git_sha: String,
    /// First line of `rustc --version` at build time, or `"unknown"` if
    /// the build script could not invoke `rustc`.
    pub rustc_version: String,
    /// Cargo profile the binary was compiled under (`debug`, `release`,
    /// `bench`, …), or `"unknown"` if the build script could not read
    /// `PROFILE` from its environment.
    pub profile: String,
    /// Rust target triple the binary was compiled for (e.g.
    /// `x86_64-unknown-linux-gnu`), or `"unknown"` if the build script
    /// could not read `TARGET` from its environment.
    pub target: String,
}

/// All TensorWasm metrics collected behind a single [`Registry`].
///
/// Clone metric handles into call sites — they are cheap atomic-shared
/// references, NOT separate counters.
#[derive(Debug, Clone)]
pub struct TensorWasmMetrics {
    inner: Arc<TensorWasmMetricsInner>,
}

#[derive(Debug)]
struct TensorWasmMetricsInner {
    registry: parking_lot::Mutex<Registry>,
    /// Live-instance count. Signed because the gauge has a balanced `inc`/`dec`
    /// access pattern and brief negative values during shutdown races are
    /// preferable to a silent wrap to `u64::MAX`.
    active_instances: Gauge<i64, AtomicI64>,
    /// GPU memory currently in use, in bytes.
    gpu_memory_used_bytes: Gauge<u64, AtomicU64>,
    kernel_dispatches_total: Counter<u64>,
    kernel_latency_seconds: Histogram,
    instance_spawns_total: Counter<u64>,
    instance_terminations_total: Counter<u64>,
    offload_success_total: Counter<u64>,
    offload_fallback_total: Counter<u64>,
    http_requests_total: Family<HttpRequestLabels, Counter<u64>>,
    http_request_duration_seconds: Family<HttpRequestLabels, Histogram, HttpDurationCtor>,
    http_requests_in_flight: Family<HttpInFlightLabels, Gauge<i64, AtomicI64>>,
    build_info: Family<BuildInfoLabels, Gauge<i64, AtomicI64>>,
}

/// Compile-time crate version, from `CARGO_PKG_VERSION`. Public so callers
/// outside this crate (e.g. the CLI's `--version` flag) can share one
/// source of truth with the `tensor_wasm_build_info` gauge.
pub const BUILD_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git commit SHA the crate was built from, or `"unknown"` for tarball
/// builds. Populated by `tensor-wasm-core/build.rs`.
pub const BUILD_GIT_SHA: &str = env!("TENSOR_WASM_GIT_SHA");

/// `rustc --version` output captured at build time, or `"unknown"` if
/// the build script could not invoke `rustc`. Populated by
/// `tensor-wasm-core/build.rs`.
pub const BUILD_RUSTC_VERSION: &str = env!("TENSOR_WASM_RUSTC_VERSION");

/// Cargo profile the crate was compiled under (`debug`, `release`, …),
/// or `"unknown"` if the build script could not read `PROFILE`.
/// Populated by `tensor-wasm-core/build.rs`.
pub const BUILD_PROFILE: &str = env!("TENSOR_WASM_PROFILE");

/// Rust target triple the crate was compiled for, or `"unknown"` if the
/// build script could not read `TARGET`. Populated by
/// `tensor-wasm-core/build.rs`.
pub const BUILD_TARGET: &str = env!("TENSOR_WASM_TARGET");

/// Build the [`BuildInfoLabels`] tuple from the compile-time constants
/// above. Exposed so test code and the CLI can produce the same labels
/// the metric carries without re-stating each `env!()` lookup.
pub fn current_build_info_labels() -> BuildInfoLabels {
    BuildInfoLabels {
        version: BUILD_VERSION.to_string(),
        git_sha: BUILD_GIT_SHA.to_string(),
        rustc_version: BUILD_RUSTC_VERSION.to_string(),
        profile: BUILD_PROFILE.to_string(),
        target: BUILD_TARGET.to_string(),
    }
}

/// Histogram constructor that captures the configured HTTP-duration bucket
/// list. Stored alongside the [`Family`] so newly-observed label combinations
/// produce a histogram with the same buckets as every other series in the
/// family.
#[derive(Clone, Debug)]
pub struct HttpDurationCtor {
    buckets: Arc<[f64]>,
}

impl prometheus_client::metrics::family::MetricConstructor<Histogram> for HttpDurationCtor {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.buckets.iter().copied())
    }
}

/// Snapshot of every numeric counter and gauge in [`TensorWasmMetrics`].
///
/// Each field is captured with `Relaxed` ordering, so the snapshot is *not*
/// a globally consistent point-in-time view — different fields can come from
/// different instants. It is still useful for diagnostics, smoke-test
/// assertions, and admin endpoints that want a single allocation-free read of
/// every gauge and counter without parsing the Prometheus text format.
///
/// The histogram is intentionally omitted because `prometheus-client` does
/// not expose a public accessor for the internal bucket state — to inspect
/// kernel latency, use [`TensorWasmMetrics::encode_text`] and parse the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorWasmMetricsStats {
    /// Current value of `tensor_wasm_active_instances`.
    pub active_instances: i64,
    /// Current value of `tensor_wasm_gpu_memory_used_bytes`.
    pub gpu_memory_used_bytes: u64,
    /// Current value of `tensor_wasm_kernel_dispatches_total`.
    pub kernel_dispatches_total: u64,
    /// Current value of `tensor_wasm_instance_spawns_total`.
    pub instance_spawns_total: u64,
    /// Current value of `tensor_wasm_instance_terminations_total`.
    pub instance_terminations_total: u64,
    /// Current value of `tensor_wasm_offload_success_total`.
    pub offload_success_total: u64,
    /// Current value of `tensor_wasm_offload_fallback_total`.
    pub offload_fallback_total: u64,
}

impl Default for TensorWasmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorWasmMetrics {
    /// Construct a fresh metrics registry with default histogram buckets.
    pub fn new() -> Self {
        Self::with_all_buckets(
            DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS.iter().copied(),
            DEFAULT_HTTP_DURATION_BUCKETS_SECONDS.iter().copied(),
        )
    }

    /// Construct a fresh metrics registry with caller-supplied kernel-latency
    /// histogram buckets. The HTTP-duration histogram keeps the default
    /// [`DEFAULT_HTTP_DURATION_BUCKETS_SECONDS`] bucket list.
    ///
    /// Buckets must be sorted ascending and finite; behaviour with unsorted or
    /// non-finite values is implementation-defined by `prometheus-client`.
    pub fn with_buckets(buckets: impl IntoIterator<Item = f64>) -> Self {
        Self::with_all_buckets(buckets, DEFAULT_HTTP_DURATION_BUCKETS_SECONDS.iter().copied())
    }

    /// Construct a fresh metrics registry with caller-supplied HTTP-duration
    /// histogram buckets. The kernel-latency histogram keeps the default
    /// [`DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS`] bucket list.
    pub fn with_http_buckets(http_buckets: impl IntoIterator<Item = f64>) -> Self {
        Self::with_all_buckets(
            DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS.iter().copied(),
            http_buckets,
        )
    }

    /// Construct a fresh metrics registry with caller-supplied buckets for
    /// both the kernel-latency and HTTP-duration histograms.
    pub fn with_all_buckets(
        kernel_buckets: impl IntoIterator<Item = f64>,
        http_buckets: impl IntoIterator<Item = f64>,
    ) -> Self {
        let mut registry = Registry::default();
        let active_instances: Gauge<i64, AtomicI64> = Gauge::default();
        let gpu_memory_used_bytes: Gauge<u64, AtomicU64> = Gauge::default();
        let kernel_dispatches_total: Counter<u64> = Counter::default();
        let kernel_latency_seconds = Histogram::new(kernel_buckets.into_iter());
        let instance_spawns_total: Counter<u64> = Counter::default();
        let instance_terminations_total: Counter<u64> = Counter::default();
        let offload_success_total: Counter<u64> = Counter::default();
        let offload_fallback_total: Counter<u64> = Counter::default();
        let http_requests_total: Family<HttpRequestLabels, Counter<u64>> = Family::default();
        let http_buckets_arc: Arc<[f64]> = http_buckets.into_iter().collect();
        let http_request_duration_seconds: Family<HttpRequestLabels, Histogram, HttpDurationCtor> =
            Family::new_with_constructor(HttpDurationCtor {
                buckets: http_buckets_arc,
            });
        let http_requests_in_flight: Family<HttpInFlightLabels, Gauge<i64, AtomicI64>> =
            Family::default();
        let build_info: Family<BuildInfoLabels, Gauge<i64, AtomicI64>> = Family::default();
        // Prime the single build-info series so it is observable on the
        // very first scrape (Family<...> emits nothing until at least
        // one label tuple has been touched). The value is `1` per the
        // Prometheus "info-style metric" convention — the payload is the
        // label set, not the number.
        build_info
            .get_or_create(&current_build_info_labels())
            .set(1);

        registry.register(
            "tensor_wasm_active_instances",
            "Number of currently live Wasm instances",
            active_instances.clone(),
        );
        registry.register(
            "tensor_wasm_gpu_memory_used_bytes",
            "Total GPU memory currently allocated to live instances, in bytes",
            gpu_memory_used_bytes.clone(),
        );
        registry.register(
            "tensor_wasm_kernel_dispatches",
            "Cumulative count of GPU kernel dispatches issued via wasi_cuda_launch",
            kernel_dispatches_total.clone(),
        );
        registry.register(
            "tensor_wasm_kernel_latency_seconds",
            "Histogram of kernel launch-to-completion latency in seconds",
            kernel_latency_seconds.clone(),
        );
        registry.register(
            "tensor_wasm_instance_spawns",
            "Cumulative count of Wasm instance spawns",
            instance_spawns_total.clone(),
        );
        registry.register(
            "tensor_wasm_instance_terminations",
            "Cumulative count of Wasm instance terminations",
            instance_terminations_total.clone(),
        );
        registry.register(
            "tensor_wasm_offload_success",
            "Cumulative count of GPU-offloaded basic blocks that completed successfully",
            offload_success_total.clone(),
        );
        registry.register(
            "tensor_wasm_offload_fallback",
            "Cumulative count of GPU offloads that deopted to the CPU fallback path",
            offload_fallback_total.clone(),
        );
        registry.register(
            "tensor_wasm_http_requests",
            "Cumulative count of HTTP requests served by the API gateway, \
             labelled by axum route template, method, and numeric status code",
            http_requests_total.clone(),
        );
        registry.register(
            "tensor_wasm_http_request_duration_seconds",
            "Histogram of HTTP request duration in seconds, labelled by axum \
             route template, method, and numeric status code",
            http_request_duration_seconds.clone(),
        );
        registry.register(
            "tensor_wasm_http_requests_in_flight",
            "Number of HTTP requests currently being served, labelled by axum \
             route template and method",
            http_requests_in_flight.clone(),
        );
        registry.register(
            "tensor_wasm_build_info",
            "Constant `1` gauge whose labels identify the binary build \
             (version, git_sha, rustc_version, profile, target). Prometheus \
             info-style metric: aggregate other series against it to label \
             dashboards with the live binary identity",
            build_info.clone(),
        );

        Self {
            inner: Arc::new(TensorWasmMetricsInner {
                registry: parking_lot::Mutex::new(registry),
                active_instances,
                gpu_memory_used_bytes,
                kernel_dispatches_total,
                kernel_latency_seconds,
                instance_spawns_total,
                instance_terminations_total,
                offload_success_total,
                offload_fallback_total,
                http_requests_total,
                http_request_duration_seconds,
                http_requests_in_flight,
                build_info,
            }),
        }
    }

    /// Number of currently live Wasm instances (gauge).
    pub fn active_instances(&self) -> &Gauge<i64, AtomicI64> {
        &self.inner.active_instances
    }

    /// GPU memory currently allocated to live instances, in bytes (gauge).
    pub fn gpu_memory_used_bytes(&self) -> &Gauge<u64, AtomicU64> {
        &self.inner.gpu_memory_used_bytes
    }

    /// Cumulative count of GPU kernel dispatches (counter).
    pub fn kernel_dispatches_total(&self) -> &Counter<u64> {
        &self.inner.kernel_dispatches_total
    }

    /// Histogram of kernel launch-to-completion latency in seconds.
    pub fn kernel_latency_seconds(&self) -> &Histogram {
        &self.inner.kernel_latency_seconds
    }

    /// Cumulative count of Wasm instance spawns (counter).
    pub fn instance_spawns_total(&self) -> &Counter<u64> {
        &self.inner.instance_spawns_total
    }

    /// Cumulative count of Wasm instance terminations (counter).
    pub fn instance_terminations_total(&self) -> &Counter<u64> {
        &self.inner.instance_terminations_total
    }

    /// Cumulative count of GPU offloads that completed successfully (counter).
    pub fn offload_success_total(&self) -> &Counter<u64> {
        &self.inner.offload_success_total
    }

    /// Cumulative count of GPU offloads that deopted to CPU (counter).
    pub fn offload_fallback_total(&self) -> &Counter<u64> {
        &self.inner.offload_fallback_total
    }

    /// Per-(route, method, status) HTTP request counter family.
    ///
    /// Increment via
    /// `metrics.http_requests_total().get_or_create(&labels).inc()`. Cardinality
    /// is bounded by the caller: see [`HttpRequestLabels`] for the contract
    /// (route is always the axum template, never the resolved id).
    pub fn http_requests_total(&self) -> &Family<HttpRequestLabels, Counter<u64>> {
        &self.inner.http_requests_total
    }

    /// Per-(route, method, status) HTTP request duration histogram family.
    ///
    /// Observe via
    /// `metrics.http_request_duration_seconds().get_or_create(&labels).observe(secs)`.
    /// Buckets are the [`DEFAULT_HTTP_DURATION_BUCKETS_SECONDS`] unless the
    /// registry was constructed via [`Self::with_http_buckets`] /
    /// [`Self::with_all_buckets`].
    pub fn http_request_duration_seconds(
        &self,
    ) -> &Family<HttpRequestLabels, Histogram, HttpDurationCtor> {
        &self.inner.http_request_duration_seconds
    }

    /// Per-(route, method) in-flight HTTP request gauge family.
    ///
    /// Increment on request entry, decrement on response. Backed by an
    /// `AtomicI64` so a brief negative reading during shutdown races is
    /// reported honestly rather than wrapping silently.
    pub fn http_requests_in_flight(&self) -> &Family<HttpInFlightLabels, Gauge<i64, AtomicI64>> {
        &self.inner.http_requests_in_flight
    }

    /// `tensor_wasm_build_info` info-style gauge family.
    ///
    /// Always carries a single primed sample with value `1` and the
    /// labels in [`current_build_info_labels`]. Operators aggregate
    /// other series against this gauge to surface the running binary's
    /// identity on dashboards and alert annotations:
    ///
    /// ```promql
    /// sum by (version, git_sha) (tensor_wasm_build_info)
    /// ```
    pub fn build_info(&self) -> &Family<BuildInfoLabels, Gauge<i64, AtomicI64>> {
        &self.inner.build_info
    }

    /// Snapshot every gauge and counter into a plain struct with one
    /// `Relaxed` load per field.
    ///
    /// Useful for diagnostic endpoints and tests that want to assert on
    /// numeric values without round-tripping through the Prometheus text
    /// format. The snapshot is *not* point-in-time consistent across fields —
    /// see [`TensorWasmMetricsStats`] for the consistency contract.
    pub fn stats(&self) -> TensorWasmMetricsStats {
        TensorWasmMetricsStats {
            active_instances: self.inner.active_instances.get(),
            gpu_memory_used_bytes: self.inner.gpu_memory_used_bytes.get(),
            // `Counter::get` requires the underlying atomic; the public API
            // is `Counter<u64>::get -> u64` via the inherent method.
            kernel_dispatches_total: self.inner.kernel_dispatches_total.get(),
            instance_spawns_total: self.inner.instance_spawns_total.get(),
            instance_terminations_total: self.inner.instance_terminations_total.get(),
            offload_success_total: self.inner.offload_success_total.get(),
            offload_fallback_total: self.inner.offload_fallback_total.get(),
        }
    }

    /// Render the registry as a Prometheus text-format exposition document.
    pub fn encode_text(&self) -> String {
        let mut out = String::new();
        let registry = self.inner.registry.lock();
        encode(&mut out, &registry).expect("text encoding into a String cannot fail");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_default() {
        let m = TensorWasmMetrics::new();
        // Initial counters are zero; encoding should mention every registered
        // metric name. NOTE: prometheus-client `Family<...>` metrics emit no
        // HELP/TYPE/sample lines until at least one label tuple has been
        // observed, so the three HTTP families
        // (`tensor_wasm_http_requests`, `tensor_wasm_http_request_duration_seconds`,
        // `tensor_wasm_http_requests_in_flight`) are deliberately excluded here
        // and covered by `http_request_families_observable` below, which
        // primes a label tuple before scraping.
        let s = m.encode_text();
        for name in [
            "tensor_wasm_active_instances",
            "tensor_wasm_gpu_memory_used_bytes",
            "tensor_wasm_kernel_dispatches",
            "tensor_wasm_kernel_latency_seconds",
            "tensor_wasm_instance_spawns",
            "tensor_wasm_instance_terminations",
            "tensor_wasm_offload_success",
            "tensor_wasm_offload_fallback",
        ] {
            assert!(s.contains(name), "missing metric {name} in:\n{s}");
        }
    }

    #[test]
    fn http_request_families_observable() {
        let m = TensorWasmMetrics::new();
        let labels = HttpRequestLabels {
            route: "/healthz".to_string(),
            method: "GET".to_string(),
            status: "200".to_string(),
        };
        m.http_requests_total().get_or_create(&labels).inc();
        m.http_request_duration_seconds()
            .get_or_create(&labels)
            .observe(0.002);
        let in_flight_labels = HttpInFlightLabels {
            route: "/healthz".to_string(),
            method: "GET".to_string(),
        };
        m.http_requests_in_flight()
            .get_or_create(&in_flight_labels)
            .inc();
        m.http_requests_in_flight()
            .get_or_create(&in_flight_labels)
            .dec();

        let s = m.encode_text();
        // Counter renders with `_total` suffix per OpenMetrics convention.
        assert!(
            s.contains("tensor_wasm_http_requests_total{route=\"/healthz\",method=\"GET\",status=\"200\"} 1"),
            "missing labelled counter sample in:\n{s}"
        );
        // Histogram count series carries the same labels.
        assert!(
            s.contains("tensor_wasm_http_request_duration_seconds_count"),
            "missing histogram count series in:\n{s}"
        );
        // In-flight gauge resolves to zero after balanced inc/dec.
        assert!(
            s.contains("tensor_wasm_http_requests_in_flight{route=\"/healthz\",method=\"GET\"} 0"),
            "missing in-flight gauge sample in:\n{s}"
        );
        // The expected `0.025` bucket is present (HTTP default bucket list).
        assert!(s.contains("le=\"0.025\""), "missing 0.025 bucket in:\n{s}");
    }

    #[test]
    fn http_request_buckets_overridable() {
        let m = TensorWasmMetrics::with_http_buckets([0.5f64, 1.0, 2.0]);
        let labels = HttpRequestLabels {
            route: "/x".to_string(),
            method: "GET".to_string(),
            status: "200".to_string(),
        };
        m.http_request_duration_seconds()
            .get_or_create(&labels)
            .observe(0.75);
        let s = m.encode_text();
        assert!(s.contains("le=\"0.5\""), "got:\n{s}");
        assert!(s.contains("le=\"1.0\""), "got:\n{s}");
        // Default HTTP bucket `0.025` must NOT appear under the custom list.
        assert!(!s.contains("le=\"0.025\""), "got:\n{s}");
    }

    #[test]
    fn gauge_increments_observable() {
        let m = TensorWasmMetrics::new();
        m.active_instances().inc();
        m.active_instances().inc();
        m.active_instances().dec();
        m.gpu_memory_used_bytes().inc_by(4096);
        let s = m.encode_text();
        // After two inc + one dec the gauge is 1.
        assert!(s.contains("tensor_wasm_active_instances 1"), "got:\n{s}");
        assert!(s.contains("tensor_wasm_gpu_memory_used_bytes 4096"), "got:\n{s}");
    }

    #[test]
    fn counter_increments_observable() {
        let m = TensorWasmMetrics::new();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        let s = m.encode_text();
        assert!(s.contains("tensor_wasm_kernel_dispatches_total 3"), "got:\n{s}");
    }

    #[test]
    fn histogram_observations_recorded() {
        let m = TensorWasmMetrics::new();
        m.kernel_latency_seconds().observe(0.0001);
        m.kernel_latency_seconds().observe(0.5);
        m.kernel_latency_seconds().observe(7.0);
        let s = m.encode_text();
        // Three observations: count == 3
        assert!(
            s.contains("tensor_wasm_kernel_latency_seconds_count 3"),
            "got:\n{s}"
        );
    }

    #[test]
    fn clone_shares_state() {
        let a = TensorWasmMetrics::new();
        let b = a.clone();
        a.kernel_dispatches_total().inc();
        b.kernel_dispatches_total().inc();
        let s = a.encode_text();
        assert!(s.contains("tensor_wasm_kernel_dispatches_total 2"), "got:\n{s}");
    }

    #[test]
    fn custom_buckets_accepted() {
        let m = TensorWasmMetrics::with_buckets([0.1f64, 1.0, 10.0]);
        m.kernel_latency_seconds().observe(0.5);
        let s = m.encode_text();
        // The bucket labels reflect the custom values.
        assert!(s.contains("le=\"0.1\""), "got:\n{s}");
        assert!(s.contains("le=\"1.0\""), "got:\n{s}");
    }

    // --- Per-getter assertion tests -----------------------------------------
    //
    // Each public getter on `TensorWasmMetrics` previously had its observable
    // behaviour covered only transitively through `encode_text`. The block
    // below asserts on the typed handle returned by each getter so a future
    // refactor that swaps an inner field (e.g. atomic type) is caught by the
    // failing test rather than by a broken Prometheus dashboard.

    #[test]
    fn getter_active_instances_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.active_instances().inc();
        assert_eq!(m.active_instances().get(), 1);
    }

    #[test]
    fn getter_gpu_memory_used_bytes_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.gpu_memory_used_bytes().inc_by(2048);
        assert_eq!(m.gpu_memory_used_bytes().get(), 2048);
    }

    #[test]
    fn getter_kernel_dispatches_total_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        assert_eq!(m.kernel_dispatches_total().get(), 2);
    }

    #[test]
    fn getter_kernel_latency_seconds_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.kernel_latency_seconds().observe(0.123);
        // Histogram has no public count accessor — verify via text encoding.
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_kernel_latency_seconds_count 1"),
            "got:\n{s}"
        );
    }

    #[test]
    fn getter_instance_spawns_total_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.instance_spawns_total().inc();
        assert_eq!(m.instance_spawns_total().get(), 1);
    }

    #[test]
    fn getter_instance_terminations_total_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.instance_terminations_total().inc();
        m.instance_terminations_total().inc();
        m.instance_terminations_total().inc();
        assert_eq!(m.instance_terminations_total().get(), 3);
    }

    #[test]
    fn getter_offload_success_total_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.offload_success_total().inc();
        assert_eq!(m.offload_success_total().get(), 1);
    }

    #[test]
    fn getter_offload_fallback_total_returns_live_handle() {
        let m = TensorWasmMetrics::new();
        m.offload_fallback_total().inc();
        m.offload_fallback_total().inc();
        assert_eq!(m.offload_fallback_total().get(), 2);
    }

    #[test]
    fn stats_returns_all_counters_at_once() {
        let m = TensorWasmMetrics::new();
        m.active_instances().inc();
        m.active_instances().inc();
        m.gpu_memory_used_bytes().inc_by(1024);
        m.kernel_dispatches_total().inc();
        m.instance_spawns_total().inc();
        m.instance_spawns_total().inc();
        m.instance_terminations_total().inc();
        m.offload_success_total().inc();
        m.offload_fallback_total().inc();
        let s = m.stats();
        assert_eq!(s.active_instances, 2);
        assert_eq!(s.gpu_memory_used_bytes, 1024);
        assert_eq!(s.kernel_dispatches_total, 1);
        assert_eq!(s.instance_spawns_total, 2);
        assert_eq!(s.instance_terminations_total, 1);
        assert_eq!(s.offload_success_total, 1);
        assert_eq!(s.offload_fallback_total, 1);
    }

    #[test]
    fn stats_initial_snapshot_is_all_zero() {
        let s = TensorWasmMetrics::new().stats();
        assert_eq!(
            s,
            TensorWasmMetricsStats {
                active_instances: 0,
                gpu_memory_used_bytes: 0,
                kernel_dispatches_total: 0,
                instance_spawns_total: 0,
                instance_terminations_total: 0,
                offload_success_total: 0,
                offload_fallback_total: 0,
            }
        );
    }

    // --- tensor_wasm_build_info -------------------------------------------
    //
    // The info-style gauge is primed at registry construction with the
    // compile-time build identity. These tests pin the three contracts
    // the rest of the stack (dashboards, alerts, UPGRADE.md verification)
    // depends on: the series name appears in scrape output, the value is
    // exactly `1`, and every documented label key is present.

    #[test]
    fn build_info_appears_in_encoded_text() {
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_build_info"),
            "missing tensor_wasm_build_info in:\n{s}"
        );
    }

    #[test]
    fn build_info_value_is_one() {
        let m = TensorWasmMetrics::new();
        let labels = current_build_info_labels();
        // Sanity: the primed handle reads back the value we set.
        assert_eq!(m.build_info().get_or_create(&labels).get(), 1);
        // And the Prometheus exposition agrees. The encoded line ends in
        // ` 1` after the closing brace of the label set.
        let s = m.encode_text();
        let line = s
            .lines()
            .find(|l| l.starts_with("tensor_wasm_build_info{"))
            .unwrap_or_else(|| panic!("no tensor_wasm_build_info sample line in:\n{s}"));
        assert!(
            line.ends_with(" 1"),
            "expected build_info value of 1, got line: {line}"
        );
    }

    #[test]
    fn build_info_carries_expected_label_keys() {
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        // Each documented label key must appear in the exposition.
        // OpenMetrics renders `key="value"`; matching on `key="` is
        // robust against the substituted value (which depends on the
        // host running the test).
        for key in ["version", "git_sha", "rustc_version", "profile", "target"] {
            let needle = format!("{}=\"", key);
            assert!(
                s.contains(&needle),
                "missing label key `{key}` on tensor_wasm_build_info in:\n{s}"
            );
        }
        // The `version` label must reflect CARGO_PKG_VERSION verbatim.
        let version_needle = format!("version=\"{}\"", env!("CARGO_PKG_VERSION"));
        assert!(
            s.contains(&version_needle),
            "expected `{version_needle}` in:\n{s}"
        );
    }

    #[test]
    fn build_info_constants_match_labels() {
        // The helper and the public constants are two paths to the same
        // truth; if they diverge the dashboards lie. Pin them together.
        let labels = current_build_info_labels();
        assert_eq!(labels.version, BUILD_VERSION);
        assert_eq!(labels.git_sha, BUILD_GIT_SHA);
        assert_eq!(labels.rustc_version, BUILD_RUSTC_VERSION);
        assert_eq!(labels.profile, BUILD_PROFILE);
        assert_eq!(labels.target, BUILD_TARGET);
        // None of the values are empty — the build script substitutes
        // "unknown" rather than the empty string on failure.
        assert!(!labels.version.is_empty());
        assert!(!labels.git_sha.is_empty());
        assert!(!labels.rustc_version.is_empty());
        assert!(!labels.profile.is_empty());
        assert!(!labels.target.is_empty());
    }
}
