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

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Log-level filter accepted by [`init`].
///
/// `Auto` reads the `TENSOR_WASM_LOG` environment variable (or `RUST_LOG` if `TENSOR_WASM_LOG`
/// is unset), falling back to `info` if neither is set. The other variants pin
/// the level explicitly, primarily for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// if it was a duplicate `init_with_otlp` call (the OTLP exporter is already
/// running), and `Err(OtlpInitError::AlreadyInitialized)` if `init` ran first.
#[cfg(feature = "otlp")]
pub fn init_with_otlp(
    level: LogLevel,
    json: bool,
    otlp_env_var: &str,
) -> Result<bool, OtlpInitError> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::{fmt, prelude::*};

    // Reject the case where a plain `init()` already grabbed the global
    // subscriber slot. Without this check the OTLP pipeline would silently
    // fail to install while we returned `Ok(true)`.
    if INIT.is_completed() {
        return Err(OtlpInitError::AlreadyInitialized);
    }

    let mut performed = false;
    let mut init_err: Option<OtlpInitError> = None;
    INIT_OTLP.call_once(|| {
        let endpoint = std::env::var(otlp_env_var)
            .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
            .unwrap_or_else(|_| "http://localhost:4317".to_string());

        // Build OTLP exporter (tonic-grpc).
        let exporter_result = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build();
        let exporter = match exporter_result {
            Ok(e) => e,
            Err(e) => {
                init_err = Some(OtlpInitError::Exporter(format!("{e:?}")));
                return;
            }
        };

        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                opentelemetry_sdk::Resource::builder()
                    .with_service_name("tensor-wasm")
                    .build(),
            )
            .build();
        let tracer = provider.tracer("tensor-wasm");
        opentelemetry::global::set_tracer_provider(provider);

        let filter = build_filter(level);
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let registry = tracing_subscriber::registry().with(filter).with(otel_layer);
        let try_init_result = if json {
            registry.with(fmt::layer().json()).try_init()
        } else {
            registry.with(fmt::layer().compact()).try_init()
        };
        // If `try_init` failed it means *something else* in the process won
        // the race to set the global subscriber between our `INIT.is_completed()`
        // check above and now. Surface that as `AlreadyInitialized` so the
        // caller sees a consistent failure mode.
        if try_init_result.is_err() {
            init_err = Some(OtlpInitError::AlreadyInitialized);
            return;
        }
        performed = true;
    });

    if let Some(e) = init_err {
        return Err(e);
    }
    Ok(performed)
}

/// Errors from [`init_with_otlp`].
#[cfg(feature = "otlp")]
#[derive(Debug, thiserror::Error)]
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
