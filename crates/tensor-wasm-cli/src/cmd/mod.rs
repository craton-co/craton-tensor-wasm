// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Subcommand implementations for the `tensor-wasm` CLI.
//!
//! Each module owns its own clap-derive arg struct and a `run` entry point
//! returning [`anyhow::Result<()>`]. The top-level `main` dispatcher dispatches
//! into these modules; tests in `tests/cli_smoke.rs` exercise them through the
//! built binary.

use std::time::Duration;

use anyhow::{Context, Result};

pub mod bench;
pub mod completions;
pub mod deploy;
pub mod invoke;
pub mod man;
pub mod metrics;
pub mod observe;
pub mod run;
pub mod snapshot;

/// Environment variable read by [`HttpContext::from_env`] to obtain a bearer
/// token for outbound API requests. See `docs/CLI.md` for the operator guide.
pub(crate) const TENSOR_WASM_TOKEN_ENV: &str = "TENSOR_WASM_TOKEN";

/// HTTP request header carrying the caller's tenant id. Mirrors the
/// `X-TensorWasm-Tenant` constant defined server-side in `tensor-wasm-api`.
pub(crate) const TENANT_HEADER: &str = "X-TensorWasm-Tenant";

/// Per-process HTTP context shared by every subcommand that talks to a TensorWasm
/// server.
///
/// Centralises the construction of the `reqwest::Client` so that bearer-token
/// auth (`TENSOR_WASM_TOKEN` env var, mapped to `Authorization: Bearer ...`) and the
/// `X-TensorWasm-Tenant` header are applied uniformly across `deploy`, `invoke`,
/// `metrics`, and `snapshot` — i.e. every command that issues an HTTP request.
///
/// The `tenant` field follows the convention documented in
/// [`crate::Cli::tenant`]: a value of `0` suppresses the header entirely so
/// upgrading clients do not start sending headers servers may not yet handle.
#[derive(Debug, Clone)]
pub(crate) struct HttpContext {
    /// Bearer token loaded from [`TENSOR_WASM_TOKEN_ENV`], if set.
    token: Option<String>,
    /// Tenant id; `0` means "do not attach the header".
    tenant: u64,
}

impl HttpContext {
    /// Build a context from the supplied `--tenant` value and the process
    /// environment. Missing / empty `TENSOR_WASM_TOKEN` is treated as "no auth".
    pub(crate) fn from_env(tenant: u64) -> Self {
        let token = std::env::var(TENSOR_WASM_TOKEN_ENV)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
        Self { token, tenant }
    }

    /// Construct a [`reqwest::Client`] with the supplied request timeout. The
    /// per-request headers (auth, tenant) are applied per-call via
    /// [`Self::apply`] rather than baked into the client so each subcommand
    /// can still override the timeout cheaply.
    pub(crate) fn build_client(&self, timeout: Duration) -> Result<reqwest::Client> {
        reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("building HTTP client")
    }

    /// Attach `Authorization` and `X-TensorWasm-Tenant` headers to a request builder
    /// when configured. Returns the (possibly unchanged) builder so callers
    /// can chain.
    pub(crate) fn apply(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(t) = &self.token {
            req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {t}"));
        }
        if self.tenant != 0 {
            req = req.header(TENANT_HEADER, self.tenant.to_string());
        }
        req
    }
}

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

/// Render a non-2xx HTTP response from the TensorWasm API as a human-readable
/// `anyhow` error.
///
/// The server's error envelope is documented in `tensor-wasm-api/src/routes.rs` as:
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
        assert!(validate_server_url("https://tensor-wasm.example.com/api").is_ok());
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

    use std::sync::Mutex;

    // Serialize env-mutating tests: parallel `set_var` / `remove_var` is UB
    // since process env is a global. Wrapping them in a Mutex is the
    // standard fix for tests that genuinely need to poke env vars.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn http_context_no_token_no_tenant_is_noop() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(TENSOR_WASM_TOKEN_ENV);
        let ctx = HttpContext::from_env(0);
        assert!(ctx.token.is_none());
        assert_eq!(ctx.tenant, 0);
    }

    #[test]
    fn http_context_picks_up_token_and_tenant() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(TENSOR_WASM_TOKEN_ENV, "  abc123  ");
        let ctx = HttpContext::from_env(42);
        // Whitespace is trimmed.
        assert_eq!(ctx.token.as_deref(), Some("abc123"));
        assert_eq!(ctx.tenant, 42);
        std::env::remove_var(TENSOR_WASM_TOKEN_ENV);
    }

    #[test]
    fn http_context_treats_empty_token_as_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(TENSOR_WASM_TOKEN_ENV, "   ");
        let ctx = HttpContext::from_env(0);
        assert!(ctx.token.is_none());
        std::env::remove_var(TENSOR_WASM_TOKEN_ENV);
    }

    #[test]
    fn apply_adds_authorization_header_when_token_set() {
        let ctx = HttpContext {
            token: Some("abc".to_string()),
            tenant: 0,
        };
        // Build an off-stack RequestBuilder so we can inspect headers.
        let client = reqwest::Client::new();
        let req = ctx.apply(client.get("http://x")).build().unwrap();
        assert_eq!(req.headers().get("authorization").unwrap(), "Bearer abc");
        assert!(req.headers().get(TENANT_HEADER).is_none());
    }

    #[test]
    fn apply_adds_tenant_header_when_nonzero() {
        let ctx = HttpContext {
            token: None,
            tenant: 7,
        };
        let client = reqwest::Client::new();
        let req = ctx.apply(client.get("http://x")).build().unwrap();
        assert_eq!(req.headers().get(TENANT_HEADER).unwrap(), "7");
        assert!(req.headers().get("authorization").is_none());
    }

    #[test]
    fn apply_omits_tenant_when_zero() {
        let ctx = HttpContext {
            token: None,
            tenant: 0,
        };
        let client = reqwest::Client::new();
        let req = ctx.apply(client.get("http://x")).build().unwrap();
        assert!(req.headers().get(TENANT_HEADER).is_none());
    }
}
