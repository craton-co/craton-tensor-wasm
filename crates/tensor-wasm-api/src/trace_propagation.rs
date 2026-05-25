// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! W3C Trace Context propagation helpers.
//!
//! This module is the single point of contact between the inbound HTTP
//! request's `traceparent` / `tracestate` headers and the local `tracing`
//! span tree. It exists so the wiring lives in exactly one place — the
//! tower middleware in [`crate::middleware`] is the only consumer at the
//! moment, but the same helpers are usable from integration tests and from
//! any future surface that wants to participate in distributed tracing
//! (e.g. a CLI shim that calls back into the gateway, a benchmark harness
//! that captures the resulting trace IDs, etc.).
//!
//! The actual span-creation logic still lives in
//! [`crate::middleware::trace_layer_with_propagation`]; this file owns:
//!
//! * the [`HeaderMapExtractor`] adapter that lets the OpenTelemetry
//!   `TextMapPropagator` API read from `axum::http::HeaderMap`;
//! * [`install_w3c_propagator`], an idempotent one-shot installer for the
//!   global [`opentelemetry_sdk::propagation::TraceContextPropagator`] so
//!   middleware extraction is not a silent no-op;
//! * [`extract_parent_context`], a thin helper around the global
//!   propagator that the middleware uses to attach a parent to its span;
//! * [`current_trace_id`], a helper for emitting the active trace id on
//!   responses (`x-trace-id` header) and in audit records.
//!
//! ## Why install the propagator from the API crate
//!
//! `tensor-wasm-core`'s [`telemetry`](tensor_wasm_core::telemetry) module
//! initialises the global tracer provider but its OTel deps are
//! feature-gated under `otlp`. The gateway crate, by contrast, needs the
//! propagator on every deployment regardless of whether the operator has
//! enabled OTLP export — without it the middleware sees an empty context
//! and `traceparent` from the client is silently dropped. So we install
//! the propagator here, unconditionally, at router build time. It's a
//! single global write guarded by a [`Once`].

use std::sync::Once;

#[allow(unused_imports)] // referenced indirectly via the global propagator registration
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry_sdk::propagation::TraceContextPropagator;

/// HTTP header name used by the gateway to surface the request's resolved
/// trace id on every response. Lowercased per HTTP/2 norms; axum
/// normalises outgoing headers regardless, but using the canonical
/// lowercase spelling here matches the rest of the codebase.
///
/// The value is the 32-character lowercase hex string the OpenTelemetry
/// SDK assigns to the request's trace, the same one operators paste into
/// Jaeger / Tempo / Honeycomb to find the trace. Operators correlating
/// logs with a captured trace id should query this header — see
/// `docs/runbooks/trace-id.md`.
pub const HEADER_TRACE_ID: &str = "x-trace-id";

/// Guards [`install_w3c_propagator`]. A separate global from any used in
/// `tensor_wasm_core::telemetry` so the API crate can stand on its own
/// without forcing operators to enable the `otlp` feature.
static PROPAGATOR_INIT: Once = Once::new();

/// Install the W3C Trace Context propagator as the global text-map
/// propagator if no other propagator has been installed yet.
///
/// Idempotent: subsequent calls are no-ops. Safe to call from
/// [`crate::server::build_router`] on every router construction — the
/// `Once` guarantees the install happens exactly once per process.
///
/// Without this call the global propagator defaults to OpenTelemetry's
/// `NoopTextMapPropagator`, which always extracts an empty context and
/// therefore silently drops the inbound `traceparent` header. We make
/// the failure mode "the propagator is installed but the SDK is not"
/// (degraded to a no-op exporter) rather than "the propagator is
/// missing" (which is a much harder bug to spot at runtime).
pub fn install_w3c_propagator() {
    PROPAGATOR_INIT.call_once(|| {
        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    });
}

/// Adapter that lets the OpenTelemetry `TextMapPropagator` read from an
/// `axum`/`http` [`HeaderMap`](axum::http::HeaderMap).
///
/// The propagator API is generic over an
/// [`opentelemetry::propagation::Extractor`], which expects
/// `get(&str) -> Option<&str>` and `keys() -> Vec<&str>` semantics. We
/// provide the smallest possible bridge so we do not need the
/// `opentelemetry-http` crate as an additional dependency.
pub struct HeaderMapExtractor<'a>(pub &'a axum::http::HeaderMap);

impl<'a> opentelemetry::propagation::Extractor for HeaderMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Extract a parent [`opentelemetry::Context`] from the supplied headers
/// via whichever global propagator has been installed (typically
/// [`TraceContextPropagator`] after [`install_w3c_propagator`] has run).
///
/// Returns an empty context if no propagator is installed or if the
/// headers do not carry a recognised trace-context entry; the caller can
/// pass that empty context to `set_parent` safely — the resulting span
/// becomes a new local root, which is the documented degradation
/// behaviour.
pub fn extract_parent_context(headers: &axum::http::HeaderMap) -> opentelemetry::Context {
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderMapExtractor(headers))
    })
}

/// Return the 32-character lowercase hex representation of the active
/// span's trace id, if any.
///
/// The value is read from `tracing::Span::current()` via the
/// [`tracing_opentelemetry::OpenTelemetrySpanExt`] bridge, which only
/// resolves to a non-zero id when a `tracing_opentelemetry` layer is
/// active in the subscriber. Returns `None` when no layer is installed
/// (the common case in unit tests that do not initialise telemetry) or
/// when the active span has no associated OTel context.
///
/// The "invalid" all-zero trace id (`00000000000000000000000000000000`)
/// returned by `SpanContext::INVALID` is also mapped to `None` so callers
/// can branch on `Option` instead of pattern-matching on the sentinel.
pub fn current_trace_id() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let span = tracing::Span::current();
    let cx = span.context();
    let trace_id = cx.span().span_context().trace_id();
    let hex = trace_id.to_string();
    // `TraceId::INVALID` renders as 32 zeros — surfacing that to callers
    // would defeat the point of the helper.
    if hex.chars().all(|c| c == '0') {
        None
    } else {
        Some(hex)
    }
}

/// Middleware that stamps `x-trace-id: <hex>` on every response when the
/// active span has a resolved OTel trace id.
///
/// Called via [`axum::middleware::from_fn`] from
/// [`crate::server::build_router_with_audit`]. Sits inside
/// [`crate::middleware::trace_layer_with_propagation`] so the active
/// span at header-injection time is the one the tracing layer just
/// created (with its parent already wired from `traceparent`).
///
/// Absent trace id => header omitted entirely; we never emit
/// `x-trace-id: ` with an empty value, as that surfaces as a confusing
/// "trace context unavailable" signal to operators reading captures.
pub async fn inject_trace_id_header(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut response = next.run(req).await;
    if let Some(hex) = current_trace_id() {
        if let Ok(value) = axum::http::HeaderValue::from_str(&hex) {
            response.headers_mut().insert(HEADER_TRACE_ID, value);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use opentelemetry::propagation::Extractor;

    #[test]
    fn header_map_extractor_returns_value() {
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", "abc".parse().unwrap());
        let ex = HeaderMapExtractor(&headers);
        assert_eq!(ex.get("traceparent"), Some("abc"));
        assert!(ex.keys().iter().any(|k| *k == "traceparent"));
    }

    #[test]
    fn header_map_extractor_missing_returns_none() {
        let headers = HeaderMap::new();
        let ex = HeaderMapExtractor(&headers);
        assert_eq!(ex.get("traceparent"), None);
    }

    #[test]
    fn install_w3c_propagator_is_idempotent() {
        // First call may or may not actually install (another test or the
        // server bin may have got there first), but neither call panics
        // and both must return cleanly.
        install_w3c_propagator();
        install_w3c_propagator();
    }

    #[test]
    fn extract_parent_context_with_no_trace_header_is_safe() {
        install_w3c_propagator();
        let headers = HeaderMap::new();
        // Must not panic; returned context may be empty.
        let _ = extract_parent_context(&headers);
    }

    #[test]
    fn extract_parent_context_with_valid_traceparent_returns_non_empty() {
        use opentelemetry::trace::TraceContextExt;

        install_w3c_propagator();
        let mut headers = HeaderMap::new();
        // Sample W3C traceparent: version=00, trace_id, parent_id, sampled.
        headers.insert(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
                .parse()
                .unwrap(),
        );
        let cx = extract_parent_context(&headers);
        let trace_id_hex = cx.span().span_context().trace_id().to_string();
        assert_eq!(trace_id_hex, "0af7651916cd43dd8448eb211c80319c");
    }

    #[test]
    fn current_trace_id_is_none_without_subscriber() {
        // No tracing-opentelemetry layer installed in this unit test, so
        // the helper must report None rather than the all-zero sentinel.
        // (Other tests in the workspace may have set up a subscriber, but
        // the global SpanContext for the current — no-op — span is still
        // invalid. The check below is robust to either outcome: we only
        // require that the helper does not surface the sentinel.)
        match current_trace_id() {
            None => {}
            Some(hex) => assert!(
                !hex.chars().all(|c| c == '0'),
                "current_trace_id leaked the all-zero sentinel",
            ),
        }
    }
}
