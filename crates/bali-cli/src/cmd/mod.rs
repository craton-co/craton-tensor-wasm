//! Subcommand implementations for the `bali` CLI.
//!
//! Each module owns its own clap-derive arg struct and a `run` entry point
//! returning [`anyhow::Result<()>`]. The top-level `main` dispatcher dispatches
//! into these modules; tests in `tests/cli_smoke.rs` exercise them through the
//! built binary.

use anyhow::Result;

pub mod bench;
pub mod completions;
pub mod deploy;
pub mod invoke;
pub mod metrics;
pub mod run;
pub mod snapshot;

/// Lightweight URL validator used by the HTTP-shaped subcommands (`deploy`,
/// `invoke`, `metrics`).
///
/// Accepts any string that starts with `http://` or `https://` and contains a
/// non-empty host component. Returns an `anyhow::Error` otherwise so the CLI
/// fails fast with a clear message instead of dispatching a request to garbage.
pub(crate) fn validate_server_url(url: &str) -> Result<()> {
    let rest = if let Some(r) = url.strip_prefix("http://") {
        r
    } else if let Some(r) = url.strip_prefix("https://") {
        r
    } else {
        anyhow::bail!("--server must start with http:// or https://, got `{url}`");
    };
    // Host is everything up to the first `/` or `?`. It must be non-empty.
    let host_end = rest.find(['/', '?']).unwrap_or(rest.len());
    if rest[..host_end].is_empty() {
        anyhow::bail!("--server has no host component: `{url}`");
    }
    Ok(())
}

/// Render a non-2xx HTTP response from the Bali API as a human-readable
/// `anyhow` error.
///
/// The server's error envelope is documented in `bali-api/src/routes.rs` as:
///
/// ```json
/// { "error": { "kind": "<machine>", "message": "<human>" } }
/// ```
///
/// If the body parses into that shape we surface both fields; otherwise we
/// fall back to the raw body so the user still gets *something* actionable.
/// The returned error always includes the HTTP status code.
pub(crate) fn render_error_response(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    #[derive(serde::Deserialize)]
    struct Envelope {
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        kind: String,
        message: String,
    }

    if let Ok(env) = serde_json::from_str::<Envelope>(body) {
        anyhow::anyhow!(
            "server returned {} ({}): {}",
            status.as_u16(),
            env.error.kind,
            env.error.message
        )
    } else {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            anyhow::anyhow!("server returned {} with empty body", status.as_u16())
        } else {
            anyhow::anyhow!("server returned {}: {}", status.as_u16(), trimmed)
        }
    }
}

/// Trim any trailing `/` from the configured server URL so callers can safely
/// concatenate `"/path"` suffixes without producing `//path`.
pub(crate) fn server_base(url: &str) -> &str {
    url.trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_http_and_https() {
        assert!(validate_server_url("http://localhost:8080").is_ok());
        assert!(validate_server_url("https://bali.example.com/api").is_ok());
    }

    #[test]
    fn rejects_missing_scheme() {
        assert!(validate_server_url("localhost:8080").is_err());
    }

    #[test]
    fn rejects_empty_host() {
        assert!(validate_server_url("http:///path").is_err());
    }

    #[test]
    fn render_error_extracts_envelope_fields() {
        let body = r#"{"error":{"kind":"invalid_name","message":"name must be non-empty"}}"#;
        let err = render_error_response(reqwest::StatusCode::BAD_REQUEST, body);
        let s = format!("{err}");
        assert!(s.contains("400"), "missing status: {s}");
        assert!(s.contains("invalid_name"), "missing kind: {s}");
        assert!(s.contains("name must be non-empty"), "missing message: {s}");
    }

    #[test]
    fn render_error_falls_back_to_raw_body() {
        let err = render_error_response(reqwest::StatusCode::BAD_GATEWAY, "upstream exploded");
        let s = format!("{err}");
        assert!(s.contains("502"), "missing status: {s}");
        assert!(s.contains("upstream exploded"), "missing body: {s}");
    }

    #[test]
    fn render_error_handles_empty_body() {
        let err = render_error_response(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "");
        let s = format!("{err}");
        assert!(s.contains("500"));
        assert!(s.to_lowercase().contains("empty"), "expected `empty`: {s}");
    }

    #[test]
    fn server_base_strips_trailing_slashes() {
        assert_eq!(server_base("http://x:1/"), "http://x:1");
        assert_eq!(server_base("http://x:1"), "http://x:1");
        assert_eq!(server_base("http://x:1///"), "http://x:1");
    }
}
