// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tracing initialisation helpers.
//!
//! The TensorWasm workspace uses the `tracing` crate for structured logging. This
//! module exposes two entry points:
//!
//! * [`init`] — wires a `tracing_subscriber` pipeline to stderr in either
//!   human-friendly pretty format or line-delimited JSON.
//! * [`init_with_otlp`] (gated on the `otlp` feature) — additionally exports
//!   spans to an OTLP collector.
//!
//! Each entry point guards itself with its own `std::sync::Once`. Mixing the
//! two in the same process is a configuration bug: whichever runs first wins,
//! and the other returns [`OtlpInitError::AlreadyInitialized`] (for
//! `init_with_otlp`) or `false` (for `init`).

use std::sync::Once;
#[cfg(feature = "otlp")]
use std::sync::OnceLock;

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Log-level filter accepted by [`init`].
///
/// `Auto` reads the `TENSOR_WASM_LOG` environment variable (or `RUST_LOG` if `TENSOR_WASM_LOG`
/// is unset), falling back to `info` if neither is set. The other variants pin
/// the level explicitly, primarily for tests.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` arms so a future
/// minor release that adds a level (e.g. a hypothetical `Off` variant)
/// does not break downstream code. The enum has no `Default` impl —
/// construct variants explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogLevel {
    /// Read from environment (`TENSOR_WASM_LOG` first, then `RUST_LOG`).
    Auto,
    /// Errors only.
    Error,
    /// Errors and warnings.
    Warn,
    /// Errors, warnings, and operational info events.
    Info,
    /// Adds developer-level diagnostic spans.
    Debug,
    /// Adds every span and event (very verbose).
    Trace,
}

impl LogLevel {
    fn as_directive(self) -> Option<&'static str> {
        match self {
            LogLevel::Auto => None,
            LogLevel::Error => Some("error"),
            LogLevel::Warn => Some("warn"),
            LogLevel::Info => Some("info"),
            LogLevel::Debug => Some("debug"),
            LogLevel::Trace => Some("trace"),
        }
    }
}

/// Guards [`init`]. Distinct from [`INIT_OTLP`] so the two entry points can
/// each report accurate "did I initialise?" status independently.
static INIT: Once = Once::new();

/// Guards [`init_with_otlp`]. See [`INIT`].
#[cfg(feature = "otlp")]
static INIT_OTLP: Once = Once::new();

/// Records the outcome of the one-and-only successful or failed
/// `init_with_otlp` body so that subsequent callers see the **same**
/// `Result` instead of a stale `Ok(false)`.
///
/// Filled exactly once from inside the [`INIT_OTLP`] `call_once` closure.
/// Subsequent invocations of [`init_with_otlp`] read this slot and replay
/// the recorded outcome (mapping the first-call success to `Ok(false)` to
/// preserve the "did this call perform initialisation?" contract).
#[cfg(feature = "otlp")]
static INIT_OTLP_RESULT: OnceLock<Result<(), OtlpInitError>> = OnceLock::new();

/// Initialise the global tracing subscriber.
///
/// `level` controls the verbosity filter; pass [`LogLevel::Auto`] for the
/// usual env-driven behaviour. `json` selects between human-friendly pretty
/// formatting (`false`) and line-delimited JSON suitable for log shippers
/// (`true`).
///
/// This is safe to call multiple times — only the first call wins. Subsequent
/// calls are silently ignored.
///
/// Returns `true` if this call actually installed the subscriber, `false`
/// if a previous call (or another component in the same process) had already
/// installed one. The previous version of this function reported `true`
/// even when the underlying `try_init()` had failed because another
/// subscriber was already global — callers now get an accurate signal.
///
/// # Panics
///
/// Does not panic in normal use. If the supplied `EnvFilter` directive is
/// malformed, the function falls back to `info`.
pub fn init(level: LogLevel, json: bool) -> bool {
    // `try_init_succeeded` records the outcome of `try_init` *inside* the
    // `call_once` closure so we can distinguish a real successful install
    // from "another global subscriber was already there". The outer variable
    // stays `false` if (a) this isn't the first call to `init`, or (b) the
    // first call lost the race to install the global subscriber.
    let mut try_init_succeeded = false;
    INIT.call_once(|| {
        let filter = build_filter(level);
        let registry = tracing_subscriber::registry().with(filter);
        let try_init_result = if json {
            let json_layer = fmt::layer()
                .json()
                .with_target(true)
                .with_current_span(true)
                .with_span_list(false);
            registry.with(json_layer).try_init()
        } else {
            let pretty_layer = fmt::layer().with_target(true).with_ansi(true).compact();
            registry.with(pretty_layer).try_init()
        };
        try_init_succeeded = try_init_result.is_ok();
    });
    try_init_succeeded
}

/// Build the standard `EnvFilter` shared by [`init`] and [`init_with_otlp`].
///
/// Extracted so the two entry points cannot drift apart silently — both honour
/// `TENSOR_WASM_LOG` (preferred) and `RUST_LOG` (fallback) when `Auto` is passed, and
/// fall back to `info` for malformed directives.
fn build_filter(level: LogLevel) -> EnvFilter {
    match level.as_directive() {
        Some(directive) => EnvFilter::try_new(directive).unwrap_or_else(|_| EnvFilter::new("info")),
        None => {
            // Prefer TENSOR_WASM_LOG, then RUST_LOG, then info.
            if let Ok(v) = std::env::var("TENSOR_WASM_LOG") {
                EnvFilter::try_new(&v).unwrap_or_else(|_| EnvFilter::new("info"))
            } else {
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent() {
        // The process-wide invariant: across the lifetime of the process, at
        // most one call to `init` can return `true`. We make two calls in a row
        // and assert that at most one returns `true`; depending on test
        // execution order another test may have initialised already, in which
        // case both calls here return `false`.
        let first = init(LogLevel::Info, false);
        let second = init(LogLevel::Debug, true);
        assert!(!(first && second), "init reported success twice");
        // Stronger: a *third* call must always be false because by this point
        // either `first` installed a subscriber, or one was already installed.
        let third = init(LogLevel::Info, false);
        assert!(!third, "init succeeded after at least one prior call");
    }

    #[test]
    fn log_level_directives() {
        assert_eq!(LogLevel::Error.as_directive(), Some("error"));
        assert_eq!(LogLevel::Warn.as_directive(), Some("warn"));
        assert_eq!(LogLevel::Info.as_directive(), Some("info"));
        assert_eq!(LogLevel::Debug.as_directive(), Some("debug"));
        assert_eq!(LogLevel::Trace.as_directive(), Some("trace"));
        assert_eq!(LogLevel::Auto.as_directive(), None);
    }

    #[test]
    fn log_levels_are_distinct() {
        assert_ne!(LogLevel::Info, LogLevel::Debug);
        assert_ne!(LogLevel::Trace, LogLevel::Auto);
    }

    // --- B5.1: `build_filter` env-var fallback contract -----------------
    //
    // `build_filter(LogLevel::Auto)` is the path the API binary and CLI
    // take in production: it consults `TENSOR_WASM_LOG` first, then
    // `RUST_LOG`, then falls back to `info`. The two tests below pin
    // both the malformed-input and the unset-env paths against the
    // `info` default so a regression that silently drops the fallback
    // (and leaves the subscriber filtering at the much-louder default
    // `trace`-equivalent, or panics on bad input) breaks here.
    //
    // `temp_env::with_var` scopes the mutation to the closure body and
    // restores the prior value when it returns, so the tests cannot
    // leak env-var state into the rest of the suite. We also scope
    // `RUST_LOG` so a developer-laptop default does not perturb the
    // assertion.

    #[test]
    fn build_filter_falls_back_to_info_on_malformed_env() {
        // A malformed directive must NOT panic — `build_filter` is
        // called from the very first lines of `init`, before any
        // diagnostics are wired up, so a panic here would be the
        // hardest possible failure mode. The fallback path returns the
        // default `info` filter.
        temp_env::with_vars(
            [
                ("TENSOR_WASM_LOG", Some("not-a-valid-filter")),
                // Also clear RUST_LOG so the local developer env (which
                // may set it to `debug`) does not perturb the assertion.
                // Explicit `None::<&str>` annotation keeps the array
                // homogeneous in the `Option<&str>` element type.
                ("RUST_LOG", None::<&str>),
            ],
            || {
                let filter = build_filter(LogLevel::Auto);
                let rendered = format!("{filter}");
                assert!(
                    rendered.contains("info"),
                    "expected `info` fallback for malformed env directive, got: {rendered}",
                );
            },
        );
    }

    #[test]
    fn build_filter_uses_default_when_env_unset() {
        // Neither `TENSOR_WASM_LOG` nor `RUST_LOG` set — the default
        // `info` filter must apply. This pins the contract documented
        // on `LogLevel::Auto`.
        temp_env::with_vars(
            [
                ("TENSOR_WASM_LOG", None::<&str>),
                ("RUST_LOG", None::<&str>),
            ],
            || {
                let filter = build_filter(LogLevel::Auto);
                let rendered = format!("{filter}");
                assert!(
                    rendered.contains("info"),
                    "expected `info` default with both env vars unset, got: {rendered}",
                );
            },
        );
    }
}

/// Initialise the global tracing subscriber with an additional OTLP exporter.
///
/// `otlp_env_var` is an environment variable name to read for the endpoint
/// (e.g. `"TENSOR_WASM_OTLP_ENDPOINT"`). If unset, falls back to the standard
/// `OTEL_EXPORTER_OTLP_ENDPOINT` and finally to `http://localhost:4317`.
///
/// Behaviour without the `otlp` feature: this function is not available
/// (gated out). Call [`init`] instead.
///
/// Like [`init`], this is safe to call multiple times — only the first call
/// performs initialisation. It uses a *separate* `Once` from [`init`], so a
/// previous call to [`init`] does not silently swallow OTLP setup; instead the
/// caller gets [`OtlpInitError::AlreadyInitialized`] and can react (typically
/// by logging that OTLP is unavailable).
///
/// Returns `Ok(true)` if this call performed the initialisation, `Ok(false)`
/// if it was a duplicate `init_with_otlp` call following a successful first
/// call, and `Err(OtlpInitError::AlreadyInitialized)` if `init` ran first. If
/// the first `init_with_otlp` call recorded an error (`Exporter(_)` or
/// `AlreadyInitialized`), every subsequent call replays that same error
/// instead of returning `Ok(false)` — without this the inconsistent global
/// state described in M5 would be invisible to the caller.
///
/// **Ordering guarantee (M5 TOCTOU fix):** on the error path, the
/// OpenTelemetry global tracer provider and propagator are NOT mutated.
/// `tracing_subscriber::try_init` runs first, and only on its success do we
/// call `opentelemetry::global::set_tracer_provider` /
/// `set_text_map_propagator`. A failed `try_init` (e.g. another subscriber
/// raced in between the `INIT.is_completed()` check and `try_init`) drops
/// the freshly-built provider without touching the global slot.
#[cfg(feature = "otlp")]
pub fn init_with_otlp(
    level: LogLevel,
    json: bool,
    otlp_env_var: &str,
) -> Result<bool, OtlpInitError> {
    // Reject the case where a plain `init()` already grabbed the global
    // subscriber slot. Without this check the OTLP pipeline would silently
    // fail to install while we returned `Ok(true)`.
    if INIT.is_completed() {
        return Err(OtlpInitError::AlreadyInitialized);
    }

    // `performed` flips to `true` *only* on the call that actually runs the
    // `call_once` body to completion successfully. All other paths (cached
    // outcome from a prior call, or a failure recorded in this call) leave
    // it `false` and either return `Ok(false)` (duplicate success) or a
    // cloned `Err(_)` from `INIT_OTLP_RESULT`.
    let mut performed = false;
    INIT_OTLP.call_once(|| {
        let outcome = run_otlp_init(level, json, otlp_env_var);
        performed = outcome.is_ok();
        // Record the verbatim outcome so subsequent callers see the same
        // result instead of `Ok(false)` masking a real failure.
        let _ = INIT_OTLP_RESULT.set(outcome);
    });

    if performed {
        return Ok(true);
    }

    // Either this is a repeat caller (the closure above did nothing) or the
    // closure ran but recorded an `Err`. In both cases replay the cached
    // outcome. `INIT_OTLP_RESULT` is guaranteed populated once `call_once`
    // has returned, but defensively fall back to `Ok(false)` if it isn't
    // (unreachable in practice — the closure always calls `set` exactly
    // once before returning).
    match INIT_OTLP_RESULT.get() {
        Some(Ok(())) => Ok(false),
        Some(Err(e)) => Err(e.clone()),
        None => Ok(false),
    }
}

/// Body of [`init_with_otlp`], lifted out so the `call_once` closure can
/// short-circuit via `?` and so the ordering between
/// `tracing_subscriber::try_init` and the OpenTelemetry global mutations is
/// obvious.
///
/// **Ordering invariant (the M5 TOCTOU fix):** we install the tracing
/// subscriber via `try_init` *before* calling
/// `opentelemetry::global::set_tracer_provider` or
/// `set_text_map_propagator`. The OpenTelemetry globals are
/// last-writer-wins and have no `try_*` variant, so once they are mutated
/// there is no way to roll them back. If we wrote them first and then
/// `try_init` failed (because another subscriber raced in between the
/// `INIT.is_completed()` check at the top of `init_with_otlp` and the
/// `try_init` call here), the caller would see `Err(AlreadyInitialized)`
/// while the OTel globals had been silently replaced with a freshly-built
/// provider that nothing was reading from. By doing `try_init` first we
/// guarantee that on the error path the OTel globals are untouched and the
/// freshly-built `SdkTracerProvider` is dropped.
#[cfg(feature = "otlp")]
fn run_otlp_init(level: LogLevel, json: bool, otlp_env_var: &str) -> Result<(), OtlpInitError> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::{fmt, prelude::*};

    let endpoint = std::env::var(otlp_env_var)
        .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .unwrap_or_else(|_| "http://localhost:4317".to_string());

    // Build OTLP exporter (tonic-grpc).
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .map_err(|e| OtlpInitError::Exporter(format!("{e:?}")))?;

    // Build the provider and derive a tracer. Crucially, neither
    // `SdkTracerProvider::builder().build()` nor `provider.tracer(...)`
    // mutates OpenTelemetry global state — they're plain constructors. The
    // global mutation (`set_tracer_provider`) is deferred until *after*
    // `try_init` succeeds so a failed `try_init` leaves the OTel globals
    // untouched.
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name("tensor-wasm")
                .build(),
        )
        .build();
    let tracer = provider.tracer("tensor-wasm");

    // Attempt `try_init` FIRST. This is the operation with the most
    // failure modes (another subscriber may have raced in between the
    // outer `INIT.is_completed()` check and now), so any error here must
    // happen *before* we touch any OpenTelemetry global.
    let filter = build_filter(level);
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let registry = tracing_subscriber::registry().with(filter).with(otel_layer);
    let try_init_result = if json {
        registry.with(fmt::layer().json()).try_init()
    } else {
        registry.with(fmt::layer().compact()).try_init()
    };
    // If `try_init` failed it means *something else* in the process won
    // the race to set the global subscriber between our
    // `INIT.is_completed()` check at the top of `init_with_otlp` and now.
    // Surface that as `AlreadyInitialized`. Because we have NOT yet
    // touched the OTel globals, dropping `provider` here is sufficient
    // cleanup — no roll-back required.
    if try_init_result.is_err() {
        drop(provider);
        return Err(OtlpInitError::AlreadyInitialized);
    }

    // `try_init` succeeded: we own the global tracing subscriber. Now —
    // and only now — install the OTel globals. Both are last-writer-wins
    // and infallible.
    opentelemetry::global::set_tracer_provider(provider);

    // Install the W3C Trace Context propagator alongside the tracer
    // provider so any embedder using `init_with_otlp` from a non-API
    // entry point (CLI, bench harness) still extracts inbound
    // `traceparent` headers when it sets up its own HTTP surface.
    // The API gateway separately calls
    // `tensor_wasm_api::install_w3c_propagator` from
    // `build_router_with_audit`; both calls converge on the same
    // global via OpenTelemetry's `set_text_map_propagator`, which is
    // last-writer-wins but safe to call repeatedly with the same
    // propagator type.
    opentelemetry::global::set_text_map_propagator(
        opentelemetry_sdk::propagation::TraceContextPropagator::new(),
    );

    Ok(())
}

/// Errors from [`init_with_otlp`].
///
/// `Clone` is implemented so the first call's outcome can be cached in
/// [`INIT_OTLP_RESULT`] and replayed verbatim to every subsequent caller.
/// Without that, repeat callers would silently see `Ok(false)` even when
/// the original initialisation had failed.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` arms so a future
/// minor release that adds a failure mode (e.g. a separate
/// `PropagatorRejected` variant) does not break downstream code. The
/// enum has no `Default` impl — construct variants explicitly.
#[cfg(feature = "otlp")]
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum OtlpInitError {
    /// The OTLP exporter could not be built.
    #[error("OTLP exporter build failed: {0}")]
    Exporter(String),
    /// A global tracing subscriber was already installed — usually because
    /// [`init`] was called earlier in the process. Pick one of the two entry
    /// points and call it exactly once.
    #[error("a global tracing subscriber has already been installed")]
    AlreadyInitialized,
}

#[cfg(all(test, feature = "otlp"))]
mod otlp_tests {
    use super::*;

    #[test]
    fn init_with_otlp_does_not_panic() {
        // We don't expect an OTLP collector to be running in CI, so the call
        // may return Ok(true), Ok(false), Err(Exporter(_)), or
        // Err(AlreadyInitialized) depending on what other tests ran first.
        // The contract under test here is just "this call doesn't panic and
        // returns a typed value".
        let _ = init_with_otlp(LogLevel::Info, false, "TENSOR_WASM_TEST_OTLP_ENDPOINT");
    }

    #[test]
    fn already_initialized_variant_displays() {
        // The variant must produce a human-readable message so operators can
        // diagnose the misconfiguration from logs.
        let e = OtlpInitError::AlreadyInitialized;
        let s = format!("{e}");
        assert!(
            s.to_lowercase().contains("already"),
            "AlreadyInitialized message should mention 'already', got: {s}",
        );
    }

    #[test]
    fn already_initialized_when_init_ran_first() {
        // If plain `init` succeeded earlier (or runs now), `init_with_otlp`
        // must refuse with `AlreadyInitialized` rather than silently no-op
        // with `Ok(false)`. Run `init` here to guarantee the precondition
        // regardless of test execution order.
        let _ = init(LogLevel::Info, false);
        assert!(INIT.is_completed());
        let err = init_with_otlp(LogLevel::Info, false, "TENSOR_WASM_TEST_OTLP_ENDPOINT_2")
            .expect_err("init_with_otlp must fail once plain init has run");
        assert!(matches!(err, OtlpInitError::AlreadyInitialized));
    }
}
