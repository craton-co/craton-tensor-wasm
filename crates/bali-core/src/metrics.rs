//! Prometheus metrics for the Bali workspace.
//!
//! [`BaliMetrics`] owns one `Registry` and a fixed set of metrics used by the
//! execution engine, the WASI-CUDA bridge, the snapshot subsystem, and the API
//! gateway. Construct it once at process startup and clone individual metric
//! handles into the components that emit them — the underlying atomics are
//! shared.

use std::sync::Arc;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

/// Default histogram buckets for kernel-launch latency, in seconds.
///
/// Calibrated for the expected range of CUDA kernel dispatches (10 µs–10 s).
/// Override by constructing [`BaliMetrics`] with [`BaliMetrics::with_buckets`].
pub const DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS: [f64; 14] = [
    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0,
];

/// All Bali metrics collected behind a single [`Registry`].
///
/// Clone metric handles into call sites — they are cheap atomic-shared
/// references, NOT separate counters.
#[derive(Debug, Clone)]
pub struct BaliMetrics {
    inner: Arc<BaliMetricsInner>,
}

#[derive(Debug)]
struct BaliMetricsInner {
    registry: parking_lot::Mutex<Registry>,
    active_instances: Gauge<i64, std::sync::atomic::AtomicI64>,
    gpu_memory_used_bytes: Gauge<i64, std::sync::atomic::AtomicI64>,
    kernel_dispatches_total: Counter<u64>,
    kernel_latency_seconds: Histogram,
    instance_spawns_total: Counter<u64>,
    instance_terminations_total: Counter<u64>,
    offload_success_total: Counter<u64>,
    offload_fallback_total: Counter<u64>,
}

impl Default for BaliMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl BaliMetrics {
    /// Construct a fresh metrics registry with default histogram buckets.
    pub fn new() -> Self {
        Self::with_buckets(DEFAULT_KERNEL_LATENCY_BUCKETS_SECONDS.iter().copied())
    }

    /// Construct a fresh metrics registry with caller-supplied histogram buckets.
    ///
    /// Buckets must be sorted ascending and finite; behaviour with unsorted or
    /// non-finite values is implementation-defined by `prometheus-client`.
    pub fn with_buckets(buckets: impl IntoIterator<Item = f64>) -> Self {
        let mut registry = Registry::default();
        let active_instances: Gauge<i64, _> = Gauge::default();
        let gpu_memory_used_bytes: Gauge<i64, _> = Gauge::default();
        let kernel_dispatches_total: Counter<u64> = Counter::default();
        let kernel_latency_seconds = Histogram::new(buckets.into_iter());
        let instance_spawns_total: Counter<u64> = Counter::default();
        let instance_terminations_total: Counter<u64> = Counter::default();
        let offload_success_total: Counter<u64> = Counter::default();
        let offload_fallback_total: Counter<u64> = Counter::default();

        registry.register(
            "bali_active_instances",
            "Number of currently live Wasm instances",
            active_instances.clone(),
        );
        registry.register(
            "bali_gpu_memory_used_bytes",
            "Total GPU memory currently allocated to live instances, in bytes",
            gpu_memory_used_bytes.clone(),
        );
        registry.register(
            "bali_kernel_dispatches",
            "Cumulative count of GPU kernel dispatches issued via wasi_cuda_launch",
            kernel_dispatches_total.clone(),
        );
        registry.register(
            "bali_kernel_latency_seconds",
            "Histogram of kernel launch-to-completion latency in seconds",
            kernel_latency_seconds.clone(),
        );
        registry.register(
            "bali_instance_spawns",
            "Cumulative count of Wasm instance spawns",
            instance_spawns_total.clone(),
        );
        registry.register(
            "bali_instance_terminations",
            "Cumulative count of Wasm instance terminations",
            instance_terminations_total.clone(),
        );
        registry.register(
            "bali_offload_success",
            "Cumulative count of GPU-offloaded basic blocks that completed successfully",
            offload_success_total.clone(),
        );
        registry.register(
            "bali_offload_fallback",
            "Cumulative count of GPU offloads that deopted to the CPU fallback path",
            offload_fallback_total.clone(),
        );

        Self {
            inner: Arc::new(BaliMetricsInner {
                registry: parking_lot::Mutex::new(registry),
                active_instances,
                gpu_memory_used_bytes,
                kernel_dispatches_total,
                kernel_latency_seconds,
                instance_spawns_total,
                instance_terminations_total,
                offload_success_total,
                offload_fallback_total,
            }),
        }
    }

    /// Number of currently live Wasm instances (gauge).
    pub fn active_instances(&self) -> &Gauge<i64, std::sync::atomic::AtomicI64> {
        &self.inner.active_instances
    }

    /// GPU memory currently allocated to live instances, in bytes (gauge).
    pub fn gpu_memory_used_bytes(&self) -> &Gauge<i64, std::sync::atomic::AtomicI64> {
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
        let m = BaliMetrics::new();
        // Initial counters are zero; encoding should mention every registered metric name.
        let s = m.encode_text();
        for name in [
            "bali_active_instances",
            "bali_gpu_memory_used_bytes",
            "bali_kernel_dispatches",
            "bali_kernel_latency_seconds",
            "bali_instance_spawns",
            "bali_instance_terminations",
            "bali_offload_success",
            "bali_offload_fallback",
        ] {
            assert!(s.contains(name), "missing metric {name} in:\n{s}");
        }
    }

    #[test]
    fn gauge_increments_observable() {
        let m = BaliMetrics::new();
        m.active_instances().inc();
        m.active_instances().inc();
        m.active_instances().dec();
        m.gpu_memory_used_bytes().inc_by(4096);
        let s = m.encode_text();
        // After two inc + one dec the gauge is 1.
        assert!(s.contains("bali_active_instances 1"), "got:\n{s}");
        assert!(s.contains("bali_gpu_memory_used_bytes 4096"), "got:\n{s}");
    }

    #[test]
    fn counter_increments_observable() {
        let m = BaliMetrics::new();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        let s = m.encode_text();
        assert!(s.contains("bali_kernel_dispatches_total 3"), "got:\n{s}");
    }

    #[test]
    fn histogram_observations_recorded() {
        let m = BaliMetrics::new();
        m.kernel_latency_seconds().observe(0.0001);
        m.kernel_latency_seconds().observe(0.5);
        m.kernel_latency_seconds().observe(7.0);
        let s = m.encode_text();
        // Three observations: count == 3
        assert!(
            s.contains("bali_kernel_latency_seconds_count 3"),
            "got:\n{s}"
        );
    }

    #[test]
    fn clone_shares_state() {
        let a = BaliMetrics::new();
        let b = a.clone();
        a.kernel_dispatches_total().inc();
        b.kernel_dispatches_total().inc();
        let s = a.encode_text();
        assert!(s.contains("bali_kernel_dispatches_total 2"), "got:\n{s}");
    }

    #[test]
    fn custom_buckets_accepted() {
        let m = BaliMetrics::with_buckets([0.1f64, 1.0, 10.0]);
        m.kernel_latency_seconds().observe(0.5);
        let s = m.encode_text();
        // The bucket labels reflect the custom values.
        assert!(s.contains("le=\"0.1\""), "got:\n{s}");
        assert!(s.contains("le=\"1.0\""), "got:\n{s}");
    }
}
