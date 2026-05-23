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
/// `invoke`, `metrics`) while a real HTTP client is still pending.
///
/// Accepts any string that starts with `http://` or `https://` and contains a
/// non-empty host component. Returns an `anyhow::Error` otherwise so the CLI
/// fails fast with a clear message instead of silently dispatching a request
/// to garbage.
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
}
