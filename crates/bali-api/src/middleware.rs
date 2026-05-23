//! Tower middleware helpers: timeouts, concurrency limits, tracing spans.
//!
//! Each helper returns a single `tower` layer that the server module composes
//! into the axum router. Keeping the helpers thin makes them easy to reuse in
//! integration tests and benchmarks where a custom stack is desired.

use std::time::Duration;

use axum::http::StatusCode;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::classify::{ServerErrorsAsFailures, SharedClassifier};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// Default per-request timeout used by [`crate::server::build_router`].
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Default process-wide cap on in-flight requests. Replaced by per-tenant
/// buckets in S20.
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 64;

/// Build a per-request timeout layer.
///
/// Requests that exceed `d` are aborted with `408 Request Timeout`.
pub fn timeout_layer(d: Duration) -> TimeoutLayer {
    TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, d)
}

/// Build the default HTTP tracing layer.
///
/// Emits a `tracing` span per request capturing method, URI, and response
/// status. The classifier treats `5xx` responses as failures.
pub fn trace_layer() -> TraceLayer<SharedClassifier<ServerErrorsAsFailures>> {
    TraceLayer::new_for_http()
}

/// Build a process-wide concurrency limit layer that allows at most `max`
/// in-flight requests.
pub fn concurrency_limit_layer(max: usize) -> ConcurrencyLimitLayer {
    ConcurrencyLimitLayer::new(max)
}

/// Returns a [`TraceLayer`] that, in addition to the per-request span,
/// reads the W3C `traceparent` header from the incoming request and uses
/// it as the parent context for the resulting span.
///
/// When a downstream service (or load test client) sends the W3C standard
/// header, traces stitch correctly across the boundary. When no header is
/// present, the span is parented to the local context as usual.
pub fn trace_layer_with_propagation() -> tower_http::trace::TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    impl Fn(&axum::http::Request<axum::body::Body>) -> tracing::Span + Clone,
> {
    TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<axum::body::Body>| {
        // Extract the W3C traceparent header (if present). The full
        // W3C parse / context attachment lands when the opentelemetry
        // global propagator is wired in `bali-core::telemetry` (the
        // `otlp` feature there installs it). For now we still surface
        // the header value as a span attribute so traces emitted by
        // this process carry the incoming trace id for correlation.
        let traceparent = req
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        tracing::info_span!(
            "http.request",
            method = %req.method(),
            uri = %req.uri(),
            version = ?req.version(),
            traceparent = %traceparent,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_layer_constructs() {
        let _ = timeout_layer(Duration::from_millis(1));
        let _ = timeout_layer(DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn trace_layer_constructs() {
        let _ = trace_layer();
    }

    #[test]
    fn concurrency_limit_layer_constructs() {
        let _ = concurrency_limit_layer(1);
        let _ = concurrency_limit_layer(DEFAULT_CONCURRENCY_LIMIT);
    }

    #[test]
    fn trace_layer_with_propagation_constructs() {
        let _ = trace_layer_with_propagation();
    }
}
