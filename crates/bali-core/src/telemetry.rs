//! Tracing initialisation helpers.
//!
//! The Bali workspace uses the `tracing` crate for structured logging. This
//! module exposes a single entry point, [`init`], that wires a
//! `tracing_subscriber` pipeline to stderr — either in human-friendly pretty
//! format or in line-delimited JSON. Crate-level call sites then emit spans
//! and events with `tracing::info!`, `tracing::instrument`, etc.

use std::sync::Once;

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Log-level filter accepted by [`init`].
///
/// `Auto` reads the `BALI_LOG` environment variable (or `RUST_LOG` if `BALI_LOG`
/// is unset), falling back to `info` if neither is set. The other variants pin
/// the level explicitly, primarily for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Read from environment (`BALI_LOG` first, then `RUST_LOG`).
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

static INIT: Once = Once::new();

/// Initialise the global tracing subscriber.
///
/// `level` controls the verbosity filter; pass [`LogLevel::Auto`] for the
/// usual env-driven behaviour. `json` selects between human-friendly pretty
/// formatting (`false`) and line-delimited JSON suitable for log shippers
/// (`true`).
///
/// This is safe to call multiple times — only the first call wins. Subsequent
/// calls are silently ignored. Returns `true` if this call performed the
/// initialisation, `false` if it was a no-op.
///
/// # Panics
///
/// Does not panic in normal use. If the supplied `EnvFilter` directive is
/// malformed, the function falls back to `info`.
pub fn init(level: LogLevel, json: bool) -> bool {
    let mut performed = false;
    INIT.call_once(|| {
        let filter = match level.as_directive() {
            Some(directive) => {
                EnvFilter::try_new(directive).unwrap_or_else(|_| EnvFilter::new("info"))
            }
            None => {
                // Prefer BALI_LOG, then RUST_LOG, then info.
                if let Ok(v) = std::env::var("BALI_LOG") {
                    EnvFilter::try_new(&v).unwrap_or_else(|_| EnvFilter::new("info"))
                } else {
                    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
                }
            }
        };

        let registry = tracing_subscriber::registry().with(filter);
        if json {
            let json_layer = fmt::layer()
                .json()
                .with_target(true)
                .with_current_span(true)
                .with_span_list(false);
            let _ = registry.with(json_layer).try_init();
        } else {
            let pretty_layer = fmt::layer().with_target(true).with_ansi(true).compact();
            let _ = registry.with(pretty_layer).try_init();
        }
        performed = true;
    });
    performed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent() {
        // First call should perform initialisation, subsequent calls should be no-ops.
        // Note: depending on test execution order another test may have initialised
        // already, in which case the first call here is itself a no-op. We assert the
        // weaker invariant that *at most one* call returns true across the lifetime
        // of the process.
        let first = init(LogLevel::Info, false);
        let second = init(LogLevel::Debug, true);
        assert!(!(first && second), "init reported success twice");
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
/// `otlp_endpoint` is an environment variable name to read for the endpoint
/// (e.g. `"BALI_OTLP_ENDPOINT"`). If unset, falls back to the standard
/// `OTEL_EXPORTER_OTLP_ENDPOINT` and finally to `http://localhost:4317`.
///
/// Behaviour without the `otlp` feature: this function is not available
/// (gated out). Call `init` instead.
///
/// Like `init`, this is safe to call multiple times — only the first call
/// performs initialisation.
///
/// Returns `true` if this call performed the initialisation, `false` if it
/// was a no-op.
#[cfg(feature = "otlp")]
pub fn init_with_otlp(
    level: LogLevel,
    json: bool,
    otlp_env_var: &str,
) -> Result<bool, OtlpInitError> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let mut performed = false;
    let mut init_err: Option<OtlpInitError> = None;
    INIT.call_once(|| {
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
                    .with_service_name("bali")
                    .build(),
            )
            .build();
        let tracer = provider.tracer("bali");
        opentelemetry::global::set_tracer_provider(provider);

        // Combine with the standard fmt subscriber.
        let filter = match level.as_directive() {
            Some(directive) => {
                EnvFilter::try_new(directive).unwrap_or_else(|_| EnvFilter::new("info"))
            }
            None => {
                if let Ok(v) = std::env::var("BALI_LOG") {
                    EnvFilter::try_new(&v).unwrap_or_else(|_| EnvFilter::new("info"))
                } else {
                    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
                }
            }
        };

        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let registry = tracing_subscriber::registry().with(filter).with(otel_layer);
        if json {
            let _ = registry.with(fmt::layer().json()).try_init();
        } else {
            let _ = registry.with(fmt::layer().compact()).try_init();
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
}

#[cfg(all(test, feature = "otlp"))]
#[cfg(test)]
mod otlp_tests {
    use super::*;

    #[test]
    fn init_with_otlp_does_not_panic() {
        // We don't expect an OTLP collector to be running in CI, so the call
        // may return Ok(true), Ok(false), or Err(Exporter(_)). Any is fine.
        let _ = init_with_otlp(LogLevel::Info, false, "BALI_TEST_OTLP_ENDPOINT");
    }
}
