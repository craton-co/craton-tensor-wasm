// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Prometheus metrics for the TensorWasm workspace.
//!
//! [`TensorWasmMetrics`] owns one `Registry` and a fixed set of metrics used by the
//! execution engine, the WASI-CUDA bridge, the snapshot subsystem, and the API
//! gateway. Construct it once at process startup and clone individual metric
//! handles into the components that emit them — the underlying atomics are
//! shared.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::atomic::{AtomicI64, AtomicU64};
use std::sync::{Arc, OnceLock};

use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use thiserror::Error;

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
///
/// Fields are `Cow<'static, str>` so the validated-input constructor
/// [`HttpRequestLabels::try_new`] can hand back borrowed `&'static str`
/// pointers for the closed sets (HTTP verbs, allow-listed route templates,
/// 3-digit status codes) without per-request `String` allocations.
/// Construction via the public fields is still permitted as an escape
/// hatch — the validated path is preferred.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` patterns and
/// cannot construct via a struct literal that exhaustively names every
/// field. Use [`HttpRequestLabels::new`] (or the validated
/// [`HttpRequestLabels::try_new`]) so adding a new label field in a
/// future minor release is non-breaking.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
#[non_exhaustive]
pub struct HttpRequestLabels {
    /// Axum route template that matched the request (e.g. `/functions/:id/invoke`).
    /// Never the substituted value — see crate-level docs on cardinality.
    pub route: Cow<'static, str>,
    /// HTTP method (`GET`, `POST`, `DELETE`).
    pub method: Cow<'static, str>,
    /// Numeric HTTP status code rendered as decimal (e.g. `"200"`, `"401"`).
    pub status: Cow<'static, str>,
}

/// Process-global allow-list of axum route templates that may appear in
/// `HttpRequestLabels::route`.
///
/// The HTTP `Family<HttpRequestLabels, ...>` metrics insert a new series
/// the first time any label tuple is observed and never evict — so if a
/// caller ever lets a raw URL path (with unbounded path parameters) leak
/// into the `route` label the registry grows without bound and eventually
/// OOMs the process. The allow-list is the structural defence: the
/// validated constructor [`HttpRequestLabels::try_new`] consults the
/// registered allow-list and rejects any unknown route with
/// [`LabelError::UnknownRoute`].
///
/// The intent is for the API binary to call
/// [`register_route_allowlist`] exactly once at startup, listing every
/// axum route template the router serves. Library and test code that
/// never starts the API can leave the allow-list unregistered — in that
/// state the validator is backward-compatible and accepts any route, so
/// existing tests continue to work unchanged.
///
/// The stored strings are `&'static str` so the validator can hand them
/// back to callers as zero-copy `Cow::Borrowed` and the `Family` map
/// keys do not pay a per-request allocation for the closed route set.
///
/// Lookup is `O(1)` via an internal `HashSet<&'static str>` populated at
/// construction. The declaration-order list is kept alongside for
/// callers (e.g. tests, route-introspection dashboards) that need to
/// iterate the registered set.
#[derive(Clone, Debug)]
pub struct RouteAllowlist {
    /// Declaration-order view of the registered routes. Preserved for
    /// callers that need to enumerate the allow-list deterministically.
    ordered: Vec<&'static str>,
    /// Hash-indexed view used by [`Self::lookup`] to keep the per-request
    /// validator at `O(1)` regardless of allow-list size.
    set: HashSet<&'static str>,
}

impl PartialEq for RouteAllowlist {
    fn eq(&self, other: &Self) -> bool {
        // `ordered` is the source of truth: two allow-lists are equal
        // iff they list the same routes in the same order. The hash
        // index is a derived field — comparing it would double-count
        // and `HashSet`'s `PartialEq` is also order-insensitive, so
        // including it adds no information.
        self.ordered == other.ordered
    }
}

impl Eq for RouteAllowlist {}

impl RouteAllowlist {
    /// Construct a fresh allow-list from a slice of static route templates.
    ///
    /// Returned in an [`Arc`] so the caller can clone the handle cheaply
    /// into [`register_route_allowlist`] and into any test that wants to
    /// inspect the registered set.
    ///
    /// The internal `HashSet` index used by [`Self::lookup`] is built
    /// once here, so a process startup that registers a 100+ route
    /// allow-list pays the hashing cost exactly once instead of on
    /// every request.
    pub fn new(routes: &[&'static str]) -> Arc<Self> {
        let ordered = routes.to_vec();
        let set = ordered.iter().copied().collect();
        Arc::new(Self { ordered, set })
    }

    /// Look up `route` in the allow-list. Returns the matching
    /// `&'static str` so the caller can attach it to a label without
    /// allocating, or `None` if the route is not registered.
    ///
    /// `O(1)` average via the internal `HashSet` — no linear scan of
    /// the registered routes, so a 100+ route allow-list does not
    /// inflate per-request latency.
    pub fn lookup(&self, route: &str) -> Option<&'static str> {
        // `HashSet::get` over `&'static str` keyed by `&str` returns
        // a `Option<&&'static str>`; deref once to recover the
        // `&'static str` pointer the caller can hand back as
        // `Cow::Borrowed` without allocating.
        self.set.get(route).copied()
    }

    /// Return the registered route templates in declaration order.
    pub fn routes(&self) -> &[&'static str] {
        &self.ordered
    }
}

/// Process-global storage for the route allow-list. Set exactly once
/// via [`register_route_allowlist`] (the API binary does this at
/// startup); read on every label validation. `None` (the default) means
/// "no allow-list registered" — the validator falls through to accept
/// any route for backward compatibility with library callers and tests.
static ROUTE_ALLOWLIST: OnceLock<Arc<RouteAllowlist>> = OnceLock::new();

/// Register the process-global HTTP route allow-list.
///
/// Intended to be called exactly once at API-binary startup, before the
/// first request hits the metrics middleware. Returns
/// [`LabelError::AllowlistAlreadyRegistered`] if called a second time —
/// the allow-list is immutable for the lifetime of the process so
/// dashboards and alert rules can rely on a stable set of `route` label
/// values.
///
/// Test code that needs an allow-list should construct a fresh
/// [`RouteAllowlist`] inline and call [`HttpRequestLabels::try_new_with_allowlist`]
/// directly to avoid contending on the process-global slot.
pub fn register_route_allowlist(routes: &[&'static str]) -> Result<(), LabelError> {
    let list = RouteAllowlist::new(routes);
    ROUTE_ALLOWLIST
        .set(list)
        .map_err(|_| LabelError::AllowlistAlreadyRegistered)
}

/// Read-only accessor for the process-global allow-list, primarily for
/// test introspection. Returns `None` if no allow-list has been
/// registered.
pub fn registered_route_allowlist() -> Option<Arc<RouteAllowlist>> {
    ROUTE_ALLOWLIST.get().cloned()
}

/// Errors returned by [`HttpRequestLabels::try_new`] when a candidate
/// label tuple would inflate metric cardinality past the bounded set.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` arms so new
/// validation error variants added in a future minor release do not break
/// downstream code. The enum has no `Default` impl — construct variants
/// explicitly.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LabelError {
    /// The supplied `route` value is not present in the registered
    /// [`RouteAllowlist`]. Carries the offending string for diagnostic
    /// logging; callers MUST NOT attach the raw value to any unbounded
    /// metric series (doing so reintroduces the very cardinality leak
    /// the allow-list exists to prevent).
    #[error("route `{route}` is not in the registered HTTP route allow-list")]
    UnknownRoute {
        /// The candidate route string that failed validation.
        route: String,
    },

    /// The supplied `method` is not one of the nine HTTP verbs accepted
    /// by the validator (`GET`, `POST`, `PUT`, `DELETE`, `PATCH`,
    /// `HEAD`, `OPTIONS`, `TRACE`, `CONNECT`).
    #[error("HTTP method `{method}` is not in the allowed verb set")]
    InvalidMethod {
        /// The candidate method string that failed validation.
        method: String,
    },

    /// The supplied numeric status code is outside the HTTP standard
    /// range of `100..=599`.
    #[error("HTTP status code {status} is outside the valid range 100..=599")]
    InvalidStatus {
        /// The candidate numeric status code that failed validation.
        status: u16,
    },

    /// [`register_route_allowlist`] was called more than once. The
    /// allow-list is immutable for the lifetime of the process.
    #[error("HTTP route allow-list has already been registered")]
    AllowlistAlreadyRegistered,
}

/// HTTP verbs that [`HttpRequestLabels::try_new`] accepts. Stored as
/// `&'static str` so a successful match yields a zero-copy
/// `Cow::Borrowed` for the `method` label.
const ALLOWED_HTTP_METHODS: &[&str] = &[
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT",
];

/// Decimal renderings of HTTP status codes `100..=599`, captured in a
/// single static table so [`HttpRequestLabels::try_new`] can hand back a
/// zero-copy `Cow::Borrowed(&'static str)` for any valid status. The
/// table is indexed as `STATUS_STR[code as usize - 100]`.
static STATUS_STR: once_cell_table::StatusTable = once_cell_table::StatusTable::new();

mod once_cell_table {
    //! Lazily-initialised `&'static str` table for HTTP status codes
    //! `100..=599`. Constructed on first access via [`std::sync::OnceLock`]
    //! so the strings live for `'static` without baking a 500-entry
    //! `const` table into the binary.

    use std::sync::OnceLock;

    /// Lookup table of `&'static str` decimal renderings of HTTP status
    /// codes `100..=599`. Initialised lazily on first lookup.
    pub(crate) struct StatusTable(OnceLock<Vec<&'static str>>);

    impl StatusTable {
        /// Construct an empty (uninitialised) table. The inner `Vec`
        /// is populated on the first call to [`Self::get`].
        pub(crate) const fn new() -> Self {
            Self(OnceLock::new())
        }

        /// Return the `&'static str` decimal rendering of `code`, or
        /// `None` if `code` is outside the `100..=599` range. The
        /// returned reference is good for the lifetime of the
        /// process.
        pub(crate) fn get(&self, code: u16) -> Option<&'static str> {
            // Range-check *before* `get_or_init` so an out-of-range first
            // lookup (e.g. code 0 or 999) returns `None` without triggering
            // the one-time 500-entry table build.
            if !(100..=599).contains(&code) {
                return None;
            }
            let table = self.0.get_or_init(|| {
                (100..=599)
                    .map(|n: u16| {
                        // `Box::leak` is intentional: the table is
                        // populated exactly once, lives for the
                        // lifetime of the process, and the leaked
                        // memory is bounded at 500 small strings.
                        let s: Box<str> = n.to_string().into_boxed_str();
                        &*Box::leak(s)
                    })
                    .collect()
            });
            Some(table[(code - 100) as usize])
        }
    }
}

impl HttpRequestLabels {
    /// Construct a [`HttpRequestLabels`] from already-validated label
    /// components.
    ///
    /// This is the non-validating constructor — useful when the caller
    /// has already proven the values are bounded (e.g. tests, or code
    /// that constructed them by some path other than user input). For
    /// the validated/allocation-free path use [`Self::try_new`] instead.
    /// Exists primarily so callers do not have to name every field of a
    /// `#[non_exhaustive]` struct directly.
    pub fn new(
        route: impl Into<Cow<'static, str>>,
        method: impl Into<Cow<'static, str>>,
        status: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            route: route.into(),
            method: method.into(),
            status: status.into(),
        }
    }

    /// Build a validated [`HttpRequestLabels`] from caller-supplied
    /// route / method / status values.
    ///
    /// Validation rules:
    ///
    /// * `route` is looked up in the process-global allow-list
    ///   registered via [`register_route_allowlist`]. If no allow-list
    ///   has been registered the route is accepted as-is (backward
    ///   compatibility for tests and library callers that do not run
    ///   the API binary). If an allow-list IS registered and `route`
    ///   is not in it, returns [`LabelError::UnknownRoute`].
    /// * `method` must be one of the nine HTTP verbs in
    ///   [`ALLOWED_HTTP_METHODS`]. Comparison is case-sensitive
    ///   (uppercase) — the API middleware normalises before calling.
    /// * `status` must be in the standard HTTP range `100..=599`.
    ///
    /// On success the returned struct embeds the matched
    /// `&'static str` for `method` and `status` (and for `route` when
    /// the allow-list is registered) as a `Cow::Borrowed`, avoiding
    /// per-request allocations on the hot path.
    pub fn try_new(route: &str, method: &str, status: u16) -> Result<Self, LabelError> {
        Self::try_new_with_allowlist(route, method, status, ROUTE_ALLOWLIST.get())
    }

    /// Like [`Self::try_new`] but consults an explicit allow-list
    /// instead of the process-global slot. Useful in test code that
    /// wants deterministic behaviour without touching the
    /// `OnceLock`-backed global.
    ///
    /// Pass `None` to skip the route allow-list check entirely
    /// (mirrors the "no allow-list registered" path of
    /// [`Self::try_new`]).
    pub fn try_new_with_allowlist(
        route: &str,
        method: &str,
        status: u16,
        allowlist: Option<&Arc<RouteAllowlist>>,
    ) -> Result<Self, LabelError> {
        let route_cow: Cow<'static, str> = match allowlist {
            Some(list) => match list.lookup(route) {
                Some(matched) => Cow::Borrowed(matched),
                None => {
                    return Err(LabelError::UnknownRoute {
                        route: route.to_string(),
                    });
                }
            },
            None => Cow::Owned(route.to_string()),
        };

        let method_cow: Cow<'static, str> =
            match ALLOWED_HTTP_METHODS.iter().copied().find(|&m| m == method) {
                Some(matched) => Cow::Borrowed(matched),
                None => {
                    return Err(LabelError::InvalidMethod {
                        method: method.to_string(),
                    });
                }
            };

        let status_cow: Cow<'static, str> = match STATUS_STR.get(status) {
            Some(s) => Cow::Borrowed(s),
            None => return Err(LabelError::InvalidStatus { status }),
        };

        Ok(HttpRequestLabels {
            route: route_cow,
            method: method_cow,
            status: status_cow,
        })
    }
}

/// Label set for `tensor_wasm_http_requests_in_flight`.
///
/// Drops `status` (in-flight requests have not produced a status yet) and
/// keeps `route` + `method` for the same cardinality bound as
/// [`HttpRequestLabels`].
///
/// Fields are `Cow<'static, str>` so callers that already hold a
/// `&'static str` (the common case — axum route templates and HTTP verbs
/// are compile-time constants) can hand it in as a zero-copy
/// `Cow::Borrowed` instead of paying a `String` allocation on every
/// request. The sibling [`HttpRequestLabels`] uses the same pattern.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` patterns and
/// construct via [`HttpInFlightLabels::new`] (or `..Default::default()`
/// — not available, this struct has no `Default` impl) so adding a new
/// label field in a future minor release is non-breaking.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
#[non_exhaustive]
pub struct HttpInFlightLabels {
    /// Axum route template that matched the request.
    pub route: Cow<'static, str>,
    /// HTTP method (`GET`, `POST`, `DELETE`).
    pub method: Cow<'static, str>,
}

impl HttpInFlightLabels {
    /// Construct a [`HttpInFlightLabels`] from already-validated label
    /// components. Exists so callers do not have to name every field of
    /// a `#[non_exhaustive]` struct and can pass `&'static str` literals
    /// without writing `Cow::Borrowed(...)` themselves.
    pub fn new(route: impl Into<Cow<'static, str>>, method: impl Into<Cow<'static, str>>) -> Self {
        Self {
            route: route.into(),
            method: method.into(),
        }
    }
}

/// Label set for `tensor_wasm_gpu_memory_bytes_per_tenant`.
///
/// Single-label family keyed by the tenant id. Cardinality is bounded
/// by the tenant count, **not** by user input: a `tenant_id` is allocated
/// server-side by `tensor-wasm-api` when a tenant first appears and is
/// never recycled within the lifetime of a node (see
/// [`crate::types::TenantId`]). The encoded label value is the
/// `Display` form of [`crate::types::TenantId`] (e.g. `"T#42"`) — the
/// same string the rest of the workspace uses in span attributes and
/// audit log entries, so dashboards and traces can join on it without
/// conversion.
///
/// Operators concerned about long-term cardinality growth under tenant
/// churn should pair scrapes with a Prometheus retention policy or a
/// recording rule that drops series for tenants absent from the
/// registry — the gauge itself does not eagerly forget. The matching
/// per-tenant breakdown of `tensor_wasm_jobs_active` planned for v0.4
/// is expected to reuse this exact label shape so a single relabel rule
/// covers both series.
///
/// Fields are `Cow<'static, str>` so callers that hold the tenant
/// rendering as an owned `String` (the common case — `TenantId::Display`
/// allocates) can hand it in as `Cow::Owned`, while tests and codepaths
/// that already hold a `&'static str` ID can avoid the allocation
/// entirely. The sibling [`HttpRequestLabels`] uses the same pattern.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` patterns and
/// construct via [`TenantLabels::new`] so adding a new dimension to the
/// tenant breakdown in a future minor release is non-breaking.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
#[non_exhaustive]
pub struct TenantLabels {
    /// Server-allocated tenant id rendered via [`crate::types::TenantId`]'s
    /// `Display` impl (e.g. `"T#42"`). Never user-supplied — the API
    /// layer assigns these monotonically when a tenant first appears.
    pub tenant_id: Cow<'static, str>,
}

impl TenantLabels {
    /// Construct labels from any string-like tenant id rendering.
    ///
    /// Accepts `&'static str`, `String`, or `Cow<'static, str>` so
    /// callers that already hold a static literal (test fixtures) and
    /// callers that produce a fresh `String` via `TenantId::to_string`
    /// share one entry point without an extra clone. Prefer
    /// [`Self::from_tenant_id`] when the caller has a typed
    /// [`crate::types::TenantId`] — that path is the canonical one and
    /// guarantees the `T#<u64>` rendering dashboards depend on.
    pub fn new(tenant_id: impl Into<Cow<'static, str>>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
        }
    }

    /// Construct labels from a typed tenant id, enforcing the canonical
    /// `T#<u64>` rendering. Prefer this over [`Self::new`] for any code
    /// that has a [`crate::types::TenantId`] in hand — it removes the
    /// possibility of a caller accidentally formatting the id with a
    /// different prefix and silently splitting a tenant's series into
    /// two label values on the dashboard.
    pub fn from_tenant_id(id: crate::types::TenantId) -> Self {
        // PERF: hot-path allocation. Audit flagged this as re-formatting
        // `T#{id}` on every call; a `DashMap<TenantId, &'static str>` cache
        // (cf. `StatusTable` at line ~292 for the `Box::leak` pattern) would
        // amortise the format + heap allocation to once per distinct tenant.
        // Deferred: `dashmap` is not a dep of `tensor-wasm-core` and adding it
        // exceeds the scope of a self-contained perf change.
        Self::new(format!("T#{}", id.get()))
    }
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
///
/// **Non-exhaustive**: callers MUST use `..` in `match` patterns and
/// construct via [`BuildInfoLabels::new`] so adding a new build-info
/// dimension (e.g. a `host_arch` label) in a future minor release is
/// non-breaking. The crate-provided helper
/// [`current_build_info_labels`] is the production source of values.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
#[non_exhaustive]
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

impl BuildInfoLabels {
    /// Construct a [`BuildInfoLabels`] from any combination of
    /// `&'static str`, `String`, or `Cow<'static, str>` per field.
    ///
    /// Each argument accepts `impl Into<Cow<'static, str>>` so the
    /// CLI's `--version` plumbing (`&'static str` literals from
    /// `env!()`) and the build-info helper (`String` produced by
    /// `to_string()`) can share one entry point without an extra
    /// `to_string()` round-trip. The internal storage stays `String`
    /// to keep the struct trivially serializable and to match the
    /// existing `current_build_info_labels()` constructor.
    pub fn new(
        version: impl Into<Cow<'static, str>>,
        git_sha: impl Into<Cow<'static, str>>,
        rustc_version: impl Into<Cow<'static, str>>,
        profile: impl Into<Cow<'static, str>>,
        target: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            version: version.into().into_owned(),
            git_sha: git_sha.into().into_owned(),
            rustc_version: rustc_version.into().into_owned(),
            profile: profile.into().into_owned(),
            target: target.into().into_owned(),
        }
    }
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
    /// Number of jobs currently `Pending` in the API-layer job registry.
    /// Signed for the same balanced-inc/dec reason as `active_instances`:
    /// brief negative readings during a shutdown race are honest
    /// telemetry, while wrap-to-`u64::MAX` is a silent dashboard lie.
    jobs_active: Gauge<i64, AtomicI64>,
    /// Per-tenant GPU memory accounting (bytes currently reserved).
    /// `u64` matches the underlying tenant counter
    /// (`TenantContext::bytes_in_use`) and the existing single-series
    /// total at `tensor_wasm_gpu_memory_used_bytes`.
    gpu_memory_bytes_per_tenant: Family<TenantLabels, Gauge<u64, AtomicU64>>,
    /// Per-tenant CPU (linear-memory / host) memory accounting (bytes
    /// currently reserved). Mirrors [`Self::gpu_memory_bytes_per_tenant`]
    /// in shape and `u64` width (the underlying
    /// `TenantContext::bytes_in_use` counter). Updated by
    /// `tensor-wasm-tenant`'s `publish_memory_gauge` on every
    /// `consume_bytes` / `release_bytes` transition.
    cpu_memory_bytes_per_tenant: Family<TenantLabels, Gauge<u64, AtomicU64>>,
    /// Cumulative count of streaming chunks emitted via
    /// `wasi:tensor/host.emit-chunk` and successfully forwarded onto
    /// the SSE / chunked-transfer response body of
    /// `POST /functions/{id}/invoke-stream`. Incremented from the
    /// `tensor-wasm-api` route handler on every chunk that passes
    /// the per-stream cap and reaches the wire — rejected emits
    /// (`-1`, `-2`, `-3`) do NOT bump the counter.
    ///
    /// Single series in v0.4; the per-tenant breakdown reuses the
    /// `TenantLabels` shape and will land alongside the per-tenant
    /// streaming-bytes histogram in a future revision.
    streaming_chunks_emitted_total: Counter<u64>,
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
///
/// **Non-exhaustive**: callers MUST use `..` in struct-pattern matches and
/// construct via `Self { ..Default::default() }` once a `Default` impl is
/// added, or by snapshotting an existing [`TensorWasmMetrics`] via
/// [`TensorWasmMetrics::stats`]. Adding a new counter field in a future
/// minor release is non-breaking under this attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
        Self::with_all_buckets(
            buckets,
            DEFAULT_HTTP_DURATION_BUCKETS_SECONDS.iter().copied(),
        )
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
        let kernel_latency_seconds = Histogram::new(kernel_buckets);
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
        let jobs_active: Gauge<i64, AtomicI64> = Gauge::default();
        let gpu_memory_bytes_per_tenant: Family<TenantLabels, Gauge<u64, AtomicU64>> =
            Family::default();
        let cpu_memory_bytes_per_tenant: Family<TenantLabels, Gauge<u64, AtomicU64>> =
            Family::default();
        let streaming_chunks_emitted_total: Counter<u64> = Counter::default();
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
        registry.register(
            "tensor_wasm_jobs_active",
            "Number of async-invocation jobs currently in `Pending` state \
             in the API-layer job registry. Incremented when \
             `POST /functions/:id/invoke-async` accepts a job; decremented \
             when the job transitions to `Completed` or `Failed`. \
             v0.3.x emits a single series; per-tenant breakdown is the \
             v0.4 follow-up and will reuse the `TenantLabels` shape",
            jobs_active.clone(),
        );
        registry.register(
            "tensor_wasm_gpu_memory_bytes_per_tenant",
            "Per-tenant GPU memory currently reserved, in bytes. Updated \
             by `tensor-wasm-tenant` on every `consume_bytes` / \
             `release_bytes` accounting transition. Additive to the \
             single-series `tensor_wasm_gpu_memory_used_bytes` total; \
             `sum by () (tensor_wasm_gpu_memory_bytes_per_tenant)` is \
             expected to track that total within scrape jitter",
            gpu_memory_bytes_per_tenant.clone(),
        );
        registry.register(
            "tensor_wasm_cpu_memory_bytes_per_tenant",
            "Per-tenant CPU (linear-memory / host) memory currently \
             reserved, in bytes. Updated by `tensor-wasm-tenant` on every \
             `consume_bytes` / `release_bytes` accounting transition. The \
             CPU counterpart to `tensor_wasm_gpu_memory_bytes_per_tenant`",
            cpu_memory_bytes_per_tenant.clone(),
        );
        registry.register(
            "tensor_wasm_streaming_chunks_emitted",
            "Cumulative count of streaming chunks emitted via \
             `wasi:tensor/host.emit-chunk` and successfully forwarded \
             onto the SSE / chunked-transfer response body of \
             `POST /functions/{id}/invoke-stream`. Rejected emits \
             (caps exceeded, receiver dropped) are NOT counted",
            streaming_chunks_emitted_total.clone(),
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
                jobs_active,
                gpu_memory_bytes_per_tenant,
                cpu_memory_bytes_per_tenant,
                streaming_chunks_emitted_total,
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

    /// Number of async-invocation jobs currently in `Pending` state (gauge).
    ///
    /// Incremented from `tensor-wasm-api`'s `invoke_function_async` handler
    /// when a `JobRecord` is inserted into the registry; decremented from
    /// the spawned background task once the job resolves to `Completed` or
    /// `Failed`. Signed for the same balanced-inc/dec reason as
    /// [`Self::active_instances`]. Single series in v0.3.x; the v0.4
    /// follow-up will switch to a `Family<TenantLabels, ...>` mirroring
    /// the shape of [`Self::gpu_memory_bytes_per_tenant`].
    pub fn jobs_active(&self) -> &Gauge<i64, AtomicI64> {
        &self.inner.jobs_active
    }

    /// Per-tenant GPU memory currently reserved, in bytes (gauge family).
    ///
    /// Set via
    /// `metrics.gpu_memory_bytes_per_tenant().get_or_create(&labels).set(bytes)`
    /// from the tenant subsystem on every `consume_bytes` /
    /// `release_bytes` transition. Additive to the pre-existing
    /// single-series total at
    /// `tensor_wasm_gpu_memory_used_bytes`; the per-tenant family is the
    /// breakdown, not a replacement. See [`TenantLabels`] for the
    /// cardinality contract.
    pub fn gpu_memory_bytes_per_tenant(&self) -> &Family<TenantLabels, Gauge<u64, AtomicU64>> {
        &self.inner.gpu_memory_bytes_per_tenant
    }

    /// Per-tenant CPU (linear-memory / host) memory currently reserved, in
    /// bytes (gauge family).
    ///
    /// Set via
    /// `metrics.cpu_memory_bytes_per_tenant().get_or_create(&labels).set(bytes)`
    /// from the tenant subsystem on every `consume_bytes` /
    /// `release_bytes` transition. The CPU counterpart to
    /// [`Self::gpu_memory_bytes_per_tenant`]; see [`TenantLabels`] for the
    /// cardinality contract.
    pub fn cpu_memory_bytes_per_tenant(&self) -> &Family<TenantLabels, Gauge<u64, AtomicU64>> {
        &self.inner.cpu_memory_bytes_per_tenant
    }

    /// Cumulative count of streaming chunks emitted via
    /// `wasi:tensor/host.emit-chunk` and successfully forwarded onto
    /// the `/invoke-stream` HTTP response body (counter).
    ///
    /// Incremented from the `tensor-wasm-api` route handler on every
    /// chunk that crosses the host→client boundary. Cap rejections
    /// (`-1`, `-2`, `-3` return codes) do NOT bump this counter — the
    /// metric measures real bytes-on-wire throughput, not guest emit
    /// attempts.
    pub fn streaming_chunks_emitted_total(&self) -> &Counter<u64> {
        &self.inner.streaming_chunks_emitted_total
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
        // PERF: scrape payloads are in the tens of KB; pre-sizing to 8 KiB
        // avoids 2-3 reallocations per scrape on the typical hot path while
        // costing only a single page of slack for the smallest registries.
        let mut out = String::with_capacity(8 * 1024);
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
        // `tensor_wasm_http_requests_in_flight`,
        // `tensor_wasm_gpu_memory_bytes_per_tenant`) are deliberately excluded
        // here and covered by `http_request_families_observable` /
        // `gpu_memory_per_tenant_family_observable_after_set` below, which
        // prime a label tuple before scraping. `tensor_wasm_jobs_active`
        // is single-series (no labels) so it does appear on the first
        // scrape with its initial value of zero.
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
            "tensor_wasm_jobs_active",
        ] {
            assert!(s.contains(name), "missing metric {name} in:\n{s}");
        }
    }

    #[test]
    fn http_request_families_observable() {
        let m = TensorWasmMetrics::new();
        let labels = HttpRequestLabels {
            route: Cow::Borrowed("/healthz"),
            method: Cow::Borrowed("GET"),
            status: Cow::Borrowed("200"),
        };
        m.http_requests_total().get_or_create(&labels).inc();
        m.http_request_duration_seconds()
            .get_or_create(&labels)
            .observe(0.002);
        let in_flight_labels = HttpInFlightLabels {
            route: Cow::Borrowed("/healthz"),
            method: Cow::Borrowed("GET"),
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
            route: Cow::Borrowed("/x"),
            method: Cow::Borrowed("GET"),
            status: Cow::Borrowed("200"),
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
        assert!(
            s.contains("tensor_wasm_gpu_memory_used_bytes 4096"),
            "got:\n{s}"
        );
    }

    #[test]
    fn counter_increments_observable() {
        let m = TensorWasmMetrics::new();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        m.kernel_dispatches_total().inc();
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_kernel_dispatches_total 3"),
            "got:\n{s}"
        );
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
        assert!(
            s.contains("tensor_wasm_kernel_dispatches_total 2"),
            "got:\n{s}"
        );
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

    // --- tensor_wasm_jobs_active + gpu_memory_bytes_per_tenant -----------
    //
    // The jobs_active gauge is single-series so it appears in the encoded
    // output on the very first scrape (its initial value `0` is emitted).
    // The per-tenant family follows the W2.3 pattern: it emits nothing
    // until at least one label tuple has been observed, so the
    // observable test below primes a tenant before scraping.

    #[test]
    fn jobs_active_initial_zero_in_text() {
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_jobs_active 0"),
            "missing initial-zero jobs_active sample in:\n{s}"
        );
    }

    #[test]
    fn jobs_active_inc_dec_observable() {
        let m = TensorWasmMetrics::new();
        m.jobs_active().inc();
        m.jobs_active().inc();
        m.jobs_active().dec();
        assert_eq!(m.jobs_active().get(), 1);
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_jobs_active 1"),
            "expected jobs_active 1 after two inc + one dec, got:\n{s}"
        );
    }

    #[test]
    fn gpu_memory_per_tenant_family_observable_after_set() {
        // Family<...> metrics emit nothing until a label tuple is touched
        // (same contract as the W2.3 HTTP families). Prime two distinct
        // tenants and assert both series appear with the expected values.
        let m = TensorWasmMetrics::new();
        let t1 = TenantLabels {
            tenant_id: Cow::Borrowed("T#1"),
        };
        let t2 = TenantLabels {
            tenant_id: Cow::Borrowed("T#2"),
        };
        m.gpu_memory_bytes_per_tenant().get_or_create(&t1).set(4096);
        m.gpu_memory_bytes_per_tenant().get_or_create(&t2).set(8192);
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id=\"T#1\"} 4096"),
            "missing tenant T#1 sample in:\n{s}"
        );
        assert!(
            s.contains("tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id=\"T#2\"} 8192"),
            "missing tenant T#2 sample in:\n{s}"
        );
    }

    #[test]
    fn gpu_memory_per_tenant_family_silent_until_observed() {
        // Mirror W2.3 http_request_families_observable contract: the
        // family must NOT appear in the exposition until a label tuple
        // has been touched. This pin keeps a future "prime at startup"
        // refactor from silently breaking the cardinality story (no
        // empty/zero series for never-seen tenants).
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        assert!(
            !s.contains("tensor_wasm_gpu_memory_bytes_per_tenant{"),
            "per-tenant gauge should be silent before observation; got:\n{s}"
        );
    }

    #[test]
    fn gpu_memory_per_tenant_total_existing_metric_preserved() {
        // The new family is additive — the pre-existing single-series
        // `tensor_wasm_gpu_memory_used_bytes` total must still appear
        // on every scrape (dashboards built on W2.5 depend on it).
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_gpu_memory_used_bytes"),
            "single-series total must be preserved alongside the per-tenant family; got:\n{s}"
        );
    }

    #[test]
    fn cpu_memory_per_tenant_family_observable_after_set() {
        // Mirror of `gpu_memory_per_tenant_family_observable_after_set`
        // for the CPU counterpart family. Family<...> metrics emit nothing
        // until a label tuple is touched; prime two tenants and assert
        // both series appear with the expected values.
        let m = TensorWasmMetrics::new();
        let t1 = TenantLabels {
            tenant_id: Cow::Borrowed("T#1"),
        };
        let t2 = TenantLabels {
            tenant_id: Cow::Borrowed("T#2"),
        };
        m.cpu_memory_bytes_per_tenant().get_or_create(&t1).set(4096);
        m.cpu_memory_bytes_per_tenant().get_or_create(&t2).set(8192);
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_cpu_memory_bytes_per_tenant{tenant_id=\"T#1\"} 4096"),
            "missing tenant T#1 sample in:\n{s}"
        );
        assert!(
            s.contains("tensor_wasm_cpu_memory_bytes_per_tenant{tenant_id=\"T#2\"} 8192"),
            "missing tenant T#2 sample in:\n{s}"
        );
    }

    #[test]
    fn cpu_memory_per_tenant_family_silent_until_observed() {
        // The CPU family must NOT appear in the exposition until a label
        // tuple has been touched — same cardinality contract as the GPU
        // family and the W2.3 HTTP families.
        let m = TensorWasmMetrics::new();
        let s = m.encode_text();
        assert!(
            !s.contains("tensor_wasm_cpu_memory_bytes_per_tenant{"),
            "per-tenant CPU gauge should be silent before observation; got:\n{s}"
        );
    }

    #[test]
    fn cpu_and_gpu_per_tenant_families_are_independent_series() {
        // Setting the CPU family for a tenant must not perturb the GPU
        // family for the same tenant (the M1 regression: both writing the
        // same series). Assert the two series coexist with distinct values.
        let m = TensorWasmMetrics::new();
        let t = TenantLabels {
            tenant_id: Cow::Borrowed("T#7"),
        };
        m.cpu_memory_bytes_per_tenant().get_or_create(&t).set(1024);
        m.gpu_memory_bytes_per_tenant().get_or_create(&t).set(2048);
        let s = m.encode_text();
        assert!(
            s.contains("tensor_wasm_cpu_memory_bytes_per_tenant{tenant_id=\"T#7\"} 1024"),
            "CPU series missing or wrong value in:\n{s}"
        );
        assert!(
            s.contains("tensor_wasm_gpu_memory_bytes_per_tenant{tenant_id=\"T#7\"} 2048"),
            "GPU series missing or wrong value in:\n{s}"
        );
    }

    // --- BuildInfoLabels::new --------------------------------------------
    //
    // The struct's fields are `pub`, so the literal-construction path is
    // already exercised by `build_info_constants_match_labels`. The two
    // tests below pin the ergonomic constructor: it must round-trip its
    // arguments verbatim and accept any `impl Into<Cow<'static, str>>`
    // payload (the `String` case is the load-bearing one — without that
    // the `to_string()` from `current_build_info_labels` would not
    // compile against `new()`).

    #[test]
    fn build_info_labels_new_round_trips() {
        let labels = BuildInfoLabels::new(
            "1.2.3",
            "deadbeef",
            "rustc 1.99.0",
            "release",
            "x86_64-unknown-linux-gnu",
        );
        assert_eq!(labels.version, "1.2.3");
        assert_eq!(labels.git_sha, "deadbeef");
        assert_eq!(labels.rustc_version, "rustc 1.99.0");
        assert_eq!(labels.profile, "release");
        assert_eq!(labels.target, "x86_64-unknown-linux-gnu");
    }

    #[test]
    fn build_info_labels_string_field_accepts_owned() {
        // The constructor accepts `impl Into<Cow<'static, str>>` so a
        // freshly-allocated `String` (e.g. from `format!`) flows through
        // without an extra `.as_str()` round-trip on the caller. This
        // pin keeps a future tightening of the bound (say, to
        // `&'static str`) from silently breaking the
        // `current_build_info_labels()` plumbing that hands in
        // `BUILD_VERSION.to_string()` and friends.
        let owned = String::from("foo");
        let labels = BuildInfoLabels::new(
            owned,
            String::from("bar"),
            String::from("baz"),
            String::from("qux"),
            String::from("quux"),
        );
        assert_eq!(labels.version, "foo");
        assert_eq!(labels.git_sha, "bar");
        assert_eq!(labels.rustc_version, "baz");
        assert_eq!(labels.profile, "qux");
        assert_eq!(labels.target, "quux");
    }

    // --- TenantLabels::from_tenant_id ------------------------------------
    //
    // The struct's `tenant_id` field is a freely-mutable `String`, so the
    // contract "the value is always the `T#<u64>` Display form of a
    // `TenantId`" is enforced only by convention at call sites. The typed
    // constructor below is the load-bearing one — it removes the
    // possibility of a caller formatting the id with a different prefix
    // and splitting a tenant's series into two label values.

    #[test]
    fn tenant_labels_from_tenant_id_renders_canonical_form() {
        use crate::types::TenantId;
        let labels = TenantLabels::from_tenant_id(TenantId(42));
        assert_eq!(labels.tenant_id, "T#42");
        // The Display form of the typed id agrees with the label string.
        assert_eq!(labels.tenant_id, TenantId(42).to_string());
    }

    #[test]
    fn tenant_labels_new_accepts_string_and_static_str() {
        // Mirrors `BuildInfoLabels::new`: any `impl Into<Cow<'static, str>>`
        // payload flows through. The `&'static str` path is exercised by
        // the existing `gpu_memory_per_tenant_*` tests; here we cover the
        // `String` case so a future `Cow<'static, str>` migration of the
        // field can keep the same surface.
        let from_static = TenantLabels::new("T#1");
        let from_owned = TenantLabels::new(String::from("T#2"));
        assert_eq!(from_static.tenant_id, "T#1");
        assert_eq!(from_owned.tenant_id, "T#2");
    }

    // --- RouteAllowlist hash lookup --------------------------------------
    //
    // The internal storage migrated from a `Vec<&'static str>` linear
    // scan to a `HashSet<&'static str>` to keep `lookup()` at `O(1)`
    // even when the allow-list has hundreds of routes. The tests below
    // pin the post-migration contract: lookups still return the
    // `&'static str` pointer from the registered set (so callers can
    // attach it as `Cow::Borrowed`), and the declaration-order accessor
    // is preserved for callers that need to iterate.

    #[test]
    fn route_allowlist_lookup_finds_registered_route() {
        let list = RouteAllowlist::new(&["/healthz", "/v1/chat/completions", "/metrics"]);
        // Lookup returns Some for every registered route.
        assert_eq!(list.lookup("/healthz"), Some("/healthz"));
        assert_eq!(
            list.lookup("/v1/chat/completions"),
            Some("/v1/chat/completions"),
        );
        assert_eq!(list.lookup("/metrics"), Some("/metrics"));
        // Unknown route is rejected.
        assert_eq!(list.lookup("/not-a-route"), None);
        // Case sensitivity is preserved (the registered routes are the
        // exact axum templates, not a case-insensitive bag).
        assert_eq!(list.lookup("/HEALTHZ"), None);
    }

    #[test]
    fn route_allowlist_routes_preserves_declaration_order() {
        // The hash index used by `lookup` is order-insensitive; the
        // `routes()` accessor must still return the registered routes
        // in the order the caller supplied them so any dashboard /
        // diagnostic dump stays stable.
        let registered: &[&'static str] = &["/c", "/a", "/b"];
        let list = RouteAllowlist::new(registered);
        assert_eq!(list.routes(), registered);
    }

    #[test]
    fn route_allowlist_lookup_returns_static_pointer() {
        // The whole reason for storing `&'static str` (instead of
        // `String`) is so the validator can hand back a `Cow::Borrowed`
        // without allocating. Pin that contract: the pointer returned by
        // `lookup` is the *same* `&'static str` the caller registered,
        // not a freshly-allocated copy.
        const ROUTE: &str = "/healthz";
        let list = RouteAllowlist::new(&[ROUTE]);
        let matched = list.lookup("/healthz").expect("registered");
        assert!(
            std::ptr::eq(matched.as_ptr(), ROUTE.as_ptr()),
            "lookup should return the registered `&'static str`, got a new pointer",
        );
    }

    #[test]
    fn route_allowlist_lookup_handles_large_allowlists() {
        // The migration's motivation is a 100+ route allow-list. Build
        // one and confirm both hit and miss paths still resolve
        // correctly (the hash index is the only code path under test
        // here — timing is exercised by
        // `crates/tensor-wasm-bench/benches/metrics_label_validation.rs`).
        let routes: Vec<&'static str> = (0..128)
            .map(|i| -> &'static str { Box::leak(format!("/route_{i}").into_boxed_str()) })
            .collect();
        let list = RouteAllowlist::new(&routes);
        // Every registered route is found.
        for r in &routes {
            assert_eq!(list.lookup(r), Some(*r));
        }
        // A route not in the list is rejected even at large sizes.
        assert_eq!(list.lookup("/route_999999"), None);
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

#[cfg(test)]
mod tests_b51 {
    //! Additional coverage added in batch B5.1 (per docs/ROADMAP.md).
    //!
    //! The tests in this module pin the `RouteAllowlist` direct-surface
    //! contract, the `StatusTable` boundary behaviour, and the validation
    //! paths of `HttpRequestLabels::try_new` / `try_new_with_allowlist` —
    //! all of which were previously covered only transitively through the
    //! HTTP metrics middleware in `tensor-wasm-api`. Pinning them inside
    //! `tensor-wasm-core` means a future change to the cardinality
    //! contract breaks here, in the crate that owns the type, rather than
    //! in a downstream test that happens to exercise the code path.
    //!
    //! Tests that mutate the process-global `ROUTE_ALLOWLIST` are gated
    //! with `#[ignore]` so they do not race against one another or
    //! against the rest of the suite. Run them explicitly with
    //! `cargo test -p tensor-wasm-core -- --ignored process_global` if you
    //! want to exercise the one-shot registration path.

    use super::*;

    // --- RouteAllowlist direct surface ----------------------------------

    #[test]
    fn route_allowlist_direct_surface_lookup_hit_and_miss() {
        let alw = RouteAllowlist::new(&["/healthz", "/metrics"]);
        // Hit: returns the matched `&'static str` (the validator hands
        // this back to callers as a zero-copy `Cow::Borrowed`).
        assert_eq!(alw.lookup("/healthz"), Some("/healthz"));
        assert_eq!(alw.lookup("/metrics"), Some("/metrics"));
        // Miss: not in the allow-list.
        assert_eq!(alw.lookup("/unknown"), None);
        // Lookup is case-sensitive (`/Healthz` is NOT the same template
        // as `/healthz`).
        assert_eq!(alw.lookup("/Healthz"), None);
        // `.routes()` returns the registered slice in declaration order.
        assert_eq!(alw.routes(), &["/healthz", "/metrics"]);
    }

    // --- StatusTable boundary behaviour ---------------------------------
    //
    // The static lives at `super::STATUS_STR` and exposes a `get(code)
    // -> Option<&'static str>` that returns the decimal rendering of the
    // status code (e.g. `"100"`, `"599"`) when the code is in the
    // standard `100..=599` HTTP range, and `None` otherwise. The tests
    // below pin both ends of the range and a value well outside it.

    #[test]
    fn status_table_boundaries() {
        // In-range values render as their decimal string.
        assert_eq!(STATUS_STR.get(100), Some("100"));
        assert_eq!(STATUS_STR.get(599), Some("599"));
        // Out-of-range values return `None` (the fallback bucket).
        assert_eq!(STATUS_STR.get(99), None);
        assert_eq!(STATUS_STR.get(600), None);
        assert_eq!(STATUS_STR.get(u16::MAX), None);
    }

    // --- HttpRequestLabels::try_new* validation -------------------------

    #[test]
    #[ignore = "process-global allowlist; mutually exclusive with `register_route_allowlist_is_one_shot`"]
    fn try_new_rejects_unknown_route() {
        // Note: this test mutates the process-global `ROUTE_ALLOWLIST`
        // via `register_route_allowlist`. It is gated with `#[ignore]`
        // so it cannot race with `register_route_allowlist_is_one_shot`
        // or with any other test that also registers the global. Run
        // it explicitly via `cargo test -- --ignored` against a fresh
        // process. The `try_new_with_allowlist` path (exercised by
        // `try_new_rejects_lowercase_method` and
        // `try_new_rejects_status_outside_1xx_5xx`) covers the same
        // validation logic without touching the global, and is the
        // recommended path for non-binary callers.
        let _ = register_route_allowlist(&["/known"]);
        let err = HttpRequestLabels::try_new("/unknown", "GET", 200).unwrap_err();
        match err {
            LabelError::UnknownRoute { route } => assert_eq!(route, "/unknown"),
            other => panic!("expected UnknownRoute, got {other:?}"),
        }
    }

    #[test]
    fn try_new_rejects_lowercase_method() {
        // The validator's method check is case-sensitive (the API
        // middleware normalises to uppercase before calling). Verify
        // that `"get"` fails with `InvalidMethod` carrying the offending
        // string, while the corresponding uppercase value passes.
        let alw = RouteAllowlist::new(&["/x"]);
        let err =
            HttpRequestLabels::try_new_with_allowlist("/x", "get", 200, Some(&alw)).unwrap_err();
        match err {
            LabelError::InvalidMethod { method } => assert_eq!(method, "get"),
            other => panic!("expected InvalidMethod, got {other:?}"),
        }
        // Sanity: the uppercase form is accepted (proves the failure
        // really is the case sensitivity, not some other validation).
        assert!(
            HttpRequestLabels::try_new_with_allowlist("/x", "GET", 200, Some(&alw)).is_ok(),
            "uppercase `GET` must pass the validator",
        );
    }

    #[test]
    fn try_new_rejects_status_outside_1xx_5xx() {
        // The validator accepts the standard HTTP range `100..=599` and
        // rejects anything else with `InvalidStatus` carrying the
        // offending numeric code. Exercise both boundary misses.
        let alw = RouteAllowlist::new(&["/x"]);
        for bad in [99u16, 600u16] {
            let err = HttpRequestLabels::try_new_with_allowlist("/x", "GET", bad, Some(&alw))
                .unwrap_err();
            match err {
                LabelError::InvalidStatus { status } => assert_eq!(status, bad),
                other => panic!("expected InvalidStatus for {bad}, got {other:?}"),
            }
        }
    }

    #[test]
    #[ignore = "process-global allowlist; mutually exclusive with other tests that touch ROUTE_ALLOWLIST"]
    fn register_route_allowlist_is_one_shot() {
        // The process-global slot is intentionally write-once so
        // dashboards and alert rules can rely on a stable `route` label
        // set for the lifetime of the process. The second call must
        // fail with `AllowlistAlreadyRegistered` regardless of whether
        // the first call won the race (a previous test in this process
        // may already have registered a different list — that still
        // forces the second call here to fail with the same error,
        // which is what we assert).
        let _ = register_route_allowlist(&["/foo"]);
        let err = register_route_allowlist(&["/bar"]).unwrap_err();
        assert!(
            matches!(err, LabelError::AllowlistAlreadyRegistered),
            "expected AllowlistAlreadyRegistered, got {err:?}",
        );
    }

    // NOTE: `default_http_metrics_config_matches_new` from the B5.1
    // task description lives in `tensor-wasm-api`, not in this crate —
    // `HttpMetricsLayerConfig` is defined in
    // `crates/tensor-wasm-api/src/http_metrics.rs` and is not reachable
    // from `tensor-wasm-core` (which would otherwise introduce a
    // reverse dependency on the API layer). The analogous test belongs
    // in `tensor-wasm-api`'s own test suite; tracked separately.
}
