// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! HTTP request metrics middleware.
//!
//! Emits the three series referenced by [`docs/SLO.md`](../../../docs/SLO.md)
//! and the reference Grafana dashboard
//! ([`docs/dashboards/README.md`](../../../docs/dashboards/README.md)):
//!
//! * `tensor_wasm_http_requests_total{route, method, status}` — counter
//! * `tensor_wasm_http_request_duration_seconds_bucket{route, method, status, le=...}` — histogram
//! * `tensor_wasm_http_requests_in_flight{route, method}` — gauge
//!
//! All three families live on the shared
//! [`tensor_wasm_core::metrics::TensorWasmMetrics`] registry so a single
//! `GET /metrics` scrape sees the HTTP series next to the kernel-latency and
//! tenant gauges. The middleware itself is a thin
//! [`axum::middleware::from_fn`] body — the production stack composes it via
//! [`http_metrics_middleware`] in [`crate::server::build_router`], placed
//! *outside* `bearer_auth` so the counter increments for `401 Unauthorized`
//! responses as well (per the SLO doc's `availability_http` SLI, which counts
//! every HTTP response).
//!
//! ## Route-template extraction
//!
//! Axum exposes the matched route template via
//! [`axum::extract::MatchedPath`], which the routing layer inserts into the
//! request extensions before invoking any per-route layer. Our middleware is
//! installed via `Router::layer`, which wraps each route's terminal endpoint
//! — at the point our middleware runs, `MatchedPath` is therefore already
//! present in the extensions whenever the request matched a route. For
//! requests that did not match a route (axum's built-in 404 path), the
//! extension is absent and the middleware labels the metric as `"unknown"`
//! to avoid leaking arbitrary path strings into the label space.
//!
//! ## Cardinality control
//!
//! Label cardinality is bounded by a *runtime allow-list* of route templates
//! initialised at startup ([`RouteAllowList`]). Templates not in the
//! allow-list collapse to `"unknown"`. This choice over a compile-time list
//! is deliberate: the router is built dynamically (`build_router` /
//! `build_router_with_audit` / `build_router_with_full_config`), so a
//! compile-time list would either duplicate the route registry (drift risk)
//! or require a const-fn router builder (not what axum offers in 0.7). The
//! allow-list is constructed from the same string literals the router uses
//! — see [`DEFAULT_ROUTE_ALLOWLIST`] — and any new route added to
//! `build_router_with_audit` MUST also be added to that constant. Failing
//! to do so is not a security or correctness issue (the metric still
//! emits, under `"unknown"`), but it does lose per-route resolution in
//! dashboards. CI does not enforce parity today; the orchestrator should
//! flag any router change that is not accompanied by an
//! [`DEFAULT_ROUTE_ALLOWLIST`] update.
//!
//! ## Performance budget
//!
//! Target overhead is < 1 µs per request. The middleware:
//!
//! * reads the `MatchedPath` extension (one `HashMap::get` lookup),
//! * one [`HashSet::contains`] probe against the allow-list,
//! * two `Family::get_or_create` calls (one read-locked
//!   `HashMap::get` on the steady-state hot path; one write-locked insert
//!   on the cold path when a new label combination first appears),
//! * one atomic `fetch_add` for the counter,
//! * one atomic `fetch_add` + `fetch_sub` for the in-flight gauge,
//! * one `Histogram::observe` which is a single bucket-array sweep.
//!
//! ## In-flight gauge balancing
//!
//! The in-flight gauge is managed by an RAII `InFlightGuard`: construction
//! increments the gauge and the guard's `Drop` decrements it. Because the
//! guard lives on the middleware future's stack across `next.run(req).await`,
//! the gauge is decremented on *every* exit path — normal completion, an
//! early return, a panic, and crucially **future cancellation** (a client
//! disconnect or an outer `tower::timeout` layer dropping the metrics future
//! mid-`await`). The timeout/load-shed layers in `build_router` sit *outside*
//! this middleware, so cancellation is the realistic leak path; the guard
//! closes it. This replaces the previous manual `inc()`/`dec()` pair, which
//! leaked `tensor_wasm_http_requests_in_flight{route,method}` permanently
//! whenever the future was dropped between the two calls.
//!
//! No heap allocations on the steady-state path beyond what
//! `prometheus-client` itself does on a cold-cache `get_or_create`. The
//! `String` label clones are unavoidable today — `prometheus-client 0.24`
//! does not yet support borrowed label values in the
//! `prometheus_client::metrics::family::Family` API; this is a known
//! cost we accept in exchange for the type-safe label set.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
use tensor_wasm_core::metrics::{HttpInFlightLabels, HttpRequestLabels, TensorWasmMetrics};

/// Label substituted when a request did not match any route template (axum's
/// built-in 404 fallback) or when the matched template is not on the
/// configured allow-list.
///
/// Kept distinct from any real route template so dashboards can flag the
/// `route="unknown"` series as a sign that either a real route is missing
/// from [`DEFAULT_ROUTE_ALLOWLIST`] or that scanners are probing the
/// gateway.
pub const UNKNOWN_ROUTE: &str = "unknown";

/// Default allow-list of axum route templates emitted by
/// [`crate::server::build_router_with_audit`]. Kept in sync by hand with the
/// `.route(...)` calls in that function; CI does not enforce parity (see
/// module docs for the cardinality-control rationale).
pub const DEFAULT_ROUTE_ALLOWLIST: &[&str] = &[
    "/healthz",
    "/metrics",
    "/functions",
    "/functions/:id",
    "/functions/:id/invoke",
    "/functions/:id/invoke-async",
    "/functions/:id/invoke-stream",
    "/jobs/:id",
    // OpenAI-compat shim (B4.9). Scaffold routes that return 501 today
    // and wire model → function translation in v0.4. See
    // `crates/tensor-wasm-api/src/openai.rs` for the handlers and
    // `docs/OPENAI-COMPAT.md` for the rollout plan.
    "/v1/completions",
    "/v1/chat/completions",
    // Kernel registry routes (B6.4). Only mounted under
    // `--features kernel-registry-api`; listed unconditionally here so
    // the route label resolves on that build axis instead of collapsing
    // to `"unknown"`. Listing them on the default build is harmless —
    // the allow-list is a label-cardinality guard, not a router, so a
    // template that never matches simply never appears as a `route`
    // value. See `crates/tensor-wasm-api/src/kernels.rs` for the
    // handlers and `build_router_full` for the mount.
    "/kernels",
    "/kernels/:name/:version",
];

/// Set of route templates that the metrics middleware is willing to emit as
/// the `route` label value. Anything not in the set collapses to
/// [`UNKNOWN_ROUTE`].
///
/// Cloned into request extensions at server start so the
/// [`http_metrics_middleware`] body does not need to capture the allow-list
/// via a type-erased closure. The set is wrapped in [`Arc`] so the per-request
/// clone is a single refcount bump.
#[derive(Debug, Clone, Default)]
pub struct RouteAllowList {
    /// Allowed route templates. `Arc<HashSet<&'static str>>` because the
    /// templates are compile-time string literals (see
    /// [`DEFAULT_ROUTE_ALLOWLIST`]) — no heap allocation per probe.
    inner: Arc<HashSet<&'static str>>,
}

impl RouteAllowList {
    /// Construct an allow-list from [`DEFAULT_ROUTE_ALLOWLIST`].
    pub fn new_default() -> Self {
        Self::from_static(DEFAULT_ROUTE_ALLOWLIST)
    }

    /// Construct an allow-list from an explicit `&'static str` slice. Used by
    /// integration tests that want to assert on a non-default route set.
    pub fn from_static(routes: &'static [&'static str]) -> Self {
        let set: HashSet<&'static str> = routes.iter().copied().collect();
        Self {
            inner: Arc::new(set),
        }
    }

    /// `true` if `route` is on the allow-list.
    pub fn allows(&self, route: &str) -> bool {
        self.inner.contains(route)
    }
}

/// Bundle passed via request extensions to [`http_metrics_middleware`].
///
/// Composed of the shared metrics registry plus the route allow-list. Cloned
/// per-request — both fields are cheap [`Arc`]-backed handles.
#[derive(Clone, Debug)]
pub struct HttpMetricsLayerConfig {
    /// Shared metrics registry. The middleware mutates the three HTTP-metric
    /// families on this handle.
    pub metrics: Arc<TensorWasmMetrics>,
    /// Allow-list of route templates that may appear as the `route` label.
    pub routes: RouteAllowList,
}

/// Resolve the route-template label for a request.
///
/// Reads the [`MatchedPath`] extension that axum's router inserts before
/// invoking per-route layers; falls back to [`UNKNOWN_ROUTE`] if the
/// extension is absent (no matching route) or if the matched template is
/// not on the allow-list (cardinality guard).
pub fn route_label(req: &Request, routes: &RouteAllowList) -> String {
    let template = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str())
        .unwrap_or(UNKNOWN_ROUTE);
    if routes.allows(template) {
        template.to_string()
    } else {
        UNKNOWN_ROUTE.to_string()
    }
}

/// RAII guard that owns one balanced `inc()`/`dec()` of the in-flight gauge.
///
/// Mirrors the [`JobsActiveGuard`](crate::routes) idiom: construction
/// increments the gauge for `(route, method)` and [`Drop`] decrements it, so
/// the gauge is balanced on every exit path. Critically, because the guard is
/// held on the middleware future's stack across `next.run(req).await`, the
/// `Drop` also fires when that future is *cancelled* — i.e. a client
/// disconnect, or an outer `tower::timeout` layer dropping the future
/// mid-`await`. That cancellation path is exactly what the previous manual
/// `inc()`/`dec()` pair leaked (`dec()` was never reached when the future was
/// dropped between the two calls).
///
/// The guard owns a `dec` closure rather than the gauge directly. The
/// closure captures the owned gauge handle returned by
/// `Family::get_or_create_owned` (which is cheap — `prometheus-client`'s
/// gauge wraps `Arc<AtomicI64>`, a single refcount bump — and crucially does
/// *not* hold the `Family`'s internal `RwLock`, which must never be held
/// across an `.await`). Capturing via a closure keeps this module free of a
/// direct `prometheus-client` dependency: the concrete `Gauge<i64,
/// AtomicI64>` type is inferred at the call site and never named here.
struct InFlightGuard<F: FnMut()> {
    dec: F,
}

impl<F: FnMut()> InFlightGuard<F> {
    /// Build a guard from an already-incremented gauge's `dec` closure. The
    /// caller increments before constructing so the inc/dec pair is visibly
    /// balanced at the call site; the guard owns the matching `dec()` and
    /// runs it in `Drop`. Keep the guard alive across `next.run(req).await`.
    fn new(dec: F) -> Self {
        Self { dec }
    }
}

impl<F: FnMut()> Drop for InFlightGuard<F> {
    /// Decrement the in-flight gauge. Runs on normal completion, early return,
    /// panic, and future cancellation/timeout — guaranteeing the gauge is
    /// balanced for every request that incremented it.
    fn drop(&mut self) {
        (self.dec)();
    }
}

/// Tower middleware that emits the three HTTP-metric families per request.
///
/// Operationally:
///
/// 1. Resolve the route template via [`route_label`] (one extension lookup).
/// 2. Snapshot [`Instant::now`] as the request-start time.
/// 3. Construct an `InFlightGuard` for `(route, method)` (increments the
///    in-flight gauge; its `Drop` decrements it on every exit path,
///    including cancellation/timeout).
/// 4. Run the inner service.
/// 5. Drop the guard, decrementing the in-flight gauge.
/// 6. Compute elapsed seconds via [`Instant::elapsed`] (saturating).
/// 7. Increment the request counter and observe the duration histogram for
///    `(route, method, status)`.
///
/// If the [`HttpMetricsLayerConfig`] extension is absent (e.g. a test
/// driver that bypassed the production router), the middleware degrades to
/// a no-op pass-through — no panics, no metric emission. This mirrors the
/// audit-log middleware's behaviour.
pub async fn http_metrics_middleware(req: Request, next: Next) -> Response {
    // Snapshot the config; if the production router was bypassed, fall
    // through to a plain pass-through so misconfigured test drivers do not
    // panic.
    let Some(cfg) = req.extensions().get::<HttpMetricsLayerConfig>().cloned() else {
        return next.run(req).await;
    };

    let route = route_label(&req, &cfg.routes);
    let method = req.method().as_str().to_string();
    let in_flight_labels = HttpInFlightLabels::new(route.clone(), method.clone());

    // `get_or_create_owned` returns an owned `Gauge` (cheap: `Gauge` wraps
    // `Arc<AtomicI64>`) so we can release the family's internal RwLock
    // before `next.run(req).await` — holding any lock across an `.await`
    // is the canonical recipe for deadlock under Tokio.
    let in_flight = cfg
        .metrics
        .http_requests_in_flight()
        .get_or_create_owned(&in_flight_labels);
    // Increment now, then hand the matching `dec()` to an RAII guard. Because
    // the guard lives on this future's stack across the `.await` below, the
    // gauge is balanced on every exit path — normal completion, early return,
    // panic, and (the realistic leak path) cancellation when an outer timeout
    // layer or a client disconnect drops this future mid-`await`.
    in_flight.inc();
    let _in_flight_guard = InFlightGuard::new(move || {
        // `Gauge::dec` returns the previous value; discard it so the closure
        // is `FnMut()` rather than `FnMut() -> i64`.
        in_flight.dec();
    });
    let start = Instant::now();

    // Run the inner service. If this future is dropped here (timeout /
    // disconnect), `_in_flight_guard`'s `Drop` still runs and decrements the
    // gauge — see the module-level "In-flight gauge balancing" docs.
    let response = next.run(req).await;

    let elapsed = start.elapsed();
    // Saturating: `Duration::as_secs_f64` is already total-ordered for
    // finite durations; `Instant::elapsed` is wall-clock-monotonic so
    // there is no negative case. The `.max(0.0)` belt is paranoia for
    // platforms where the monotonic guarantee leaks (none today, but
    // cheap).
    let elapsed_secs = elapsed.as_secs_f64().max(0.0);

    let status = response.status().as_u16().to_string();
    let labels = HttpRequestLabels::new(route, method, status);
    cfg.metrics
        .http_requests_total()
        .get_or_create(&labels)
        .inc();
    cfg.metrics
        .http_request_duration_seconds()
        .get_or_create(&labels)
        .observe(elapsed_secs);

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;

    // `Arc` is only imported through the parent module; the no-extension
    // fallback test does not need it.

    #[test]
    fn route_allowlist_default_covers_known_routes() {
        let allow = RouteAllowList::new_default();
        for r in [
            "/healthz",
            "/metrics",
            "/functions",
            "/functions/:id",
            "/functions/:id/invoke",
            "/functions/:id/invoke-async",
            "/functions/:id/invoke-stream",
            "/jobs/:id",
        ] {
            assert!(allow.allows(r), "default allow-list missing {r}");
        }
    }

    #[test]
    fn route_allowlist_rejects_unlisted() {
        let allow = RouteAllowList::new_default();
        assert!(!allow.allows("/secret"));
        assert!(!allow.allows("/functions/abc-123/invoke"));
        assert!(!allow.allows(""));
    }

    #[test]
    fn route_label_falls_back_to_unknown_when_extension_missing() {
        // `MatchedPath` is constructed only by axum's router (its tuple
        // field is `pub(crate)`), so the unit-level test here can only
        // cover the no-extension fallback. The end-to-end behaviour —
        // matched template passing through, unlisted template collapsing
        // — is exercised in `tests/http_metrics_test.rs` against a real
        // router.
        let allow = RouteAllowList::new_default();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/anything")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(route_label(&req, &allow), UNKNOWN_ROUTE);
    }
}
