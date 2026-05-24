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

/// Adapter that lets the OpenTelemetry `TextMapPropagator` read from an
/// `axum`/`http` [`HeaderMap`](axum::http::HeaderMap).
///
/// The propagator API is generic over an [`opentelemetry::propagation::Extractor`],
/// which expects `get(&str) -> Option<&str>` and `keys() -> Vec<&str>` semantics.
/// We provide the smallest possible bridge so we do not need the
/// `opentelemetry-http` crate as an additional dependency.
struct HeaderMapExtractor<'a>(&'a axum::http::HeaderMap);

impl<'a> opentelemetry::propagation::Extractor for HeaderMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Returns a [`TraceLayer`] that, in addition to the per-request span,
/// reads the W3C `traceparent` header from the incoming request and uses
/// it as the parent context for the resulting span.
///
/// When a downstream service (or load test client) sends the W3C standard
/// header, traces stitch correctly across the boundary. When no header is
/// present, the span is parented to the local context as usual.
///
/// The function delegates the actual parsing to the OpenTelemetry global
/// `TextMapPropagator` (installed by `bali-core::telemetry` under the
/// `otlp` feature). If no propagator is installed the extraction returns
/// an empty context and the span is parented locally — i.e. behaviour
/// degrades gracefully.
pub fn trace_layer_with_propagation() -> tower_http::trace::TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    impl Fn(&axum::http::Request<axum::body::Body>) -> tracing::Span + Clone,
> {
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<axum::body::Body>| {
        // Surface the raw traceparent value as a span field for log-based
        // correlation, even when no OTel propagator is installed.
        let traceparent = req
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let span = tracing::info_span!(
            "http.request",
            method = %req.method(),
            uri = %req.uri(),
            version = ?req.version(),
            traceparent = %traceparent,
        );

        // Extract the parent `opentelemetry::Context` from the incoming
        // headers via whichever `TextMapPropagator` was installed globally
        // (typically `TraceContextPropagator` for W3C `traceparent`). Then
        // attach it as the parent of the freshly-created tracing span via
        // the `OpenTelemetrySpanExt::set_parent` bridge.
        let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&HeaderMapExtractor(req.headers()))
        });
        span.set_parent(parent_cx);

        span
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
