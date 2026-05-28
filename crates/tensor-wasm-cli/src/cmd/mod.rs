// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Subcommand implementations for the `tensor-wasm` CLI.
//!
//! Each module owns its own clap-derive arg struct and a `run` entry point
//! returning [`anyhow::Result<()>`]. The top-level `main` dispatcher dispatches
//! into these modules; tests in `tests/cli_smoke.rs` exercise them through the
//! built binary.

use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;

pub mod bench;
pub mod completions;
pub mod deploy;
pub mod invoke;
pub mod kernel;
pub mod man;
pub mod metrics;
pub mod observe;
pub mod run;
pub mod serve;
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
/// The `tenant` field follows the convention documented for the CLI's
/// `--tenant` flag (`src/main.rs`): a value of `0` suppresses the header
/// entirely so upgrading clients do not start sending headers servers may
/// not yet handle.
/// The struct is `pub` only so integration tests under `tests/` can
/// observe credential-safety behaviour through the lib surface. The type
/// is `#[doc(hidden)]` because it is NOT stable public API — the only
/// supported consumer is the in-tree `tensor-wasm` binary plus the
/// in-tree tests.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HttpContext {
    /// Bearer token loaded from [`TENSOR_WASM_TOKEN_ENV`], if set.
    token: Option<String>,
    /// Tenant id; `0` means "do not attach the header".
    tenant: u64,
}

impl HttpContext {
    /// Build a context from the supplied `--tenant` value and the process
    /// environment. Missing / empty `TENSOR_WASM_TOKEN` is treated as "no auth".
    pub fn from_env(tenant: u64) -> Self {
        let token = std::env::var(TENSOR_WASM_TOKEN_ENV)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
        Self { token, tenant }
    }

    /// Test-only constructor that bypasses the process environment so
    /// integration tests can build an `HttpContext` with a known token
    /// without racing other tests that mutate `TENSOR_WASM_TOKEN`. Marked
    /// `#[doc(hidden)]` so it doesn't leak into the public API surface.
    #[doc(hidden)]
    pub fn from_env_for_test_with_token(token: impl Into<String>, tenant: u64) -> Self {
        Self {
            token: Some(token.into()),
            tenant,
        }
    }

    /// Test-only constructor accepting an optional token. Mirrors
    /// [`Self::from_env_for_test_with_token`] but lets a test exercise the
    /// "no token configured" branch deterministically.
    #[doc(hidden)]
    pub fn from_env_for_test_with_token_optional(token: Option<String>, tenant: u64) -> Self {
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
    ///
    /// When a bearer token is configured and the request URL is non-loopback
    /// `http://`, a one-shot `tracing::warn!` fires so an operator who
    /// accidentally points the CLI at a plaintext endpoint sees a clear
    /// signal in their logs. We do NOT refuse — operators may legitimately
    /// be on a trusted private network; the HMAC-key case in `snapshot.rs`
    /// is stricter (refuse) because the snapshot signing key is far more
    /// sensitive than a bearer token (which can be rotated cheaply).
    pub fn apply(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(t) = &self.token {
            // Try to inspect the URL on the in-flight builder so we can warn
            // before the token leaves the process. `try_clone` + `build` is
            // the only stable way to peek at the builder's URL today.
            if let Some(cloned) = req.try_clone() {
                if let Ok(built) = cloned.build() {
                    let scheme = built.url().scheme();
                    let host = built.url().host_str().unwrap_or("");
                    if scheme == "http" && !is_loopback_host(host) {
                        warn_plaintext_token_once();
                    }
                }
            }
            req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {t}"));
        }
        if self.tenant != 0 {
            req = req.header(TENANT_HEADER, self.tenant.to_string());
        }
        req
    }
}

/// Latch so the plaintext-token warning fires at most once per process.
/// `OnceLock<()>` rather than `OnceLock<bool>` because we only need
/// "have we fired yet?" semantics — `get().is_some()` is the answer.
static PLAINTEXT_TOKEN_WARNED: OnceLock<()> = OnceLock::new();

/// Test-only accessor returning whether the plaintext-token warning has
/// been emitted by the current process. Used by `tests/url_credential_safety.rs`
/// to verify the warn-once gating without needing a tracing subscriber.
///
/// `OnceLock` has no public `reset`, so callers cannot observe the
/// transition more than once per process. Tests that depend on the
/// pre-warning state must therefore run before anything that triggers the
/// gate; the integration test in `tests/url_credential_safety.rs` is the
/// only consumer and it owns its own test binary, so isolation is
/// guaranteed.
///
/// Marked `#[doc(hidden)]` rather than `#[cfg(test)]` because integration
/// tests under `tests/` are compiled as *external* consumers of this lib
/// crate, so a `cfg(test)` symbol would not be visible to them. The
/// `doc(hidden)` attribute keeps the function out of rustdoc / IDE
/// autocomplete so it cannot be mistaken for stable API.
#[doc(hidden)]
pub fn plaintext_token_warned_for_test() -> bool {
    PLAINTEXT_TOKEN_WARNED.get().is_some()
}

/// Emit the one-shot plaintext-token warning. The `OnceLock::set` call is
/// the gate: a second call returns `Err` and is silently dropped, which is
/// the desired "warn once" semantics.
fn warn_plaintext_token_once() {
    if PLAINTEXT_TOKEN_WARNED.set(()).is_ok() {
        tracing::warn!(
            "sending TENSOR_WASM_TOKEN bearer auth over plaintext http:// to a \
             non-loopback host; rotate the token if this URL was unexpected \
             and prefer https:// for production endpoints"
        );
    }
}

/// Loopback / dev-only hosts where plaintext http:// is acceptable. We match
/// `localhost`, `127.x.y.z` (any IPv4 loopback), and `::1` (IPv6 loopback).
/// IPv6 hosts arrive bracketed in URLs but reqwest's `Url::host_str` returns
/// them unbracketed, so we compare against `::1` directly.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") || host == "::1" {
        return true;
    }
    if let Ok(addr) = host.parse::<std::net::IpAddr>() {
        return addr.is_loopback();
    }
    false
}

/// Lightweight URL validator used by the HTTP-shaped subcommands (`deploy`,
/// `invoke`, `metrics`).
///
/// Accepts any string that starts with `http://` or `https://` and contains a
/// non-empty host component. Returns an `anyhow::Error` otherwise so the CLI
/// fails fast with a clear message instead of dispatching a request to garbage.
///
/// URLs containing embedded credentials (`http://user:pass@host`) are also
/// rejected — `reqwest` forwards the userinfo as a Basic-auth header, and
/// every subcommand that formats `{url}` into an error message would
/// otherwise echo the password verbatim into logs and CI output. Auth
/// belongs in the `TENSOR_WASM_TOKEN` env var (see [`HttpContext::from_env`]).
///
/// `pub` rather than `pub(crate)` so the integration test in
/// `tests/url_credential_safety.rs` can exercise the credential-safety
/// branches directly without spawning the full binary.
pub fn validate_server_url(url: &str) -> Result<()> {
    let rest = if let Some(r) = url.strip_prefix("http://") {
        r
    } else if let Some(r) = url.strip_prefix("https://") {
        r
    } else {
        anyhow::bail!("--server must start with http:// or https://, got `{url}`");
    };
    // Host is everything up to the first `/` or `?`. It must be non-empty.
    let host_end = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..host_end];
    if authority.is_empty() {
        anyhow::bail!("--server has no host component: `{url}`");
    }
    // Userinfo lives between the scheme and the first `/` or `?`, separated
    // from the host by a literal `@`. We deliberately do NOT echo the URL
    // back here because the offending password is part of the string — the
    // error message must not leak the credential a careless operator just
    // pasted into their shell history.
    if authority.contains('@') {
        anyhow::bail!(
            "URLs with embedded credentials are not allowed; pass auth via TENSOR_WASM_TOKEN env var"
        );
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

/// Extract `(scheme, host)` from a server URL the CLI accepts (`http://...` or
/// `https://...`). Strips a `:port` suffix from the host so `is_loopback_host`
/// only sees the hostname. Returns `None` if the URL lacks a recognised
/// scheme — callers should have already passed `validate_server_url` so this
/// is purely defensive.
///
/// Note: this is a deliberately lightweight scheme/host extractor, not a
/// full URL parser. The CLI already calls [`validate_server_url`] before
/// any HTTP code path runs, so by the time we get here the input is known
/// to be `http://…` or `https://…` with a non-empty authority and no
/// embedded userinfo (`@`). That lets us shortcut the parse without
/// pulling in the `url` crate.
pub(crate) fn extract_scheme_host(url: &str) -> Option<(&str, &str)> {
    let (scheme, rest) = if let Some(r) = url.strip_prefix("http://") {
        ("http", r)
    } else if let Some(r) = url.strip_prefix("https://") {
        ("https", r)
    } else {
        return None;
    };
    let host_end = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..host_end];
    // Strip `:port`. IPv6 literals would be bracketed (`[::1]:8080`); the
    // simple `rfind(':')` works for IPv4 / hostnames; for the bracketed
    // IPv6 case the host portion runs from `[` to `]` and the port (if
    // any) starts after `]`. We handle both shapes here so a future
    // `https://[::1]:8443` URL is classified correctly.
    let host = if let Some(stripped) = authority.strip_prefix('[') {
        // `[ipv6]` or `[ipv6]:port`. Find the closing bracket.
        let close = stripped.find(']')?;
        &stripped[..close]
    } else if let Some(colon) = authority.rfind(':') {
        &authority[..colon]
    } else {
        authority
    };
    Some((scheme, host))
}

// ---------------------------------------------------------------------------
// Response-body size cap (T17, DoS hardening)
// ---------------------------------------------------------------------------
//
// Multiple HTTP paths in the CLI used to call `reqwest::Response::text()` /
// `bytes()` with no upper bound. A malicious or buggy server could stream
// gigabytes into the CLI's RAM that way. The helpers below buffer a response
// body into `Vec<u8>` while enforcing [`MAX_RESPONSE_BODY_BYTES`]; any client
// path that legitimately needs more should stream to disk through
// `reqwest::Response::bytes_stream()` instead (see `snapshot.rs` for an
// example).

/// Hard cap (in bytes) on the size of in-memory response bodies the CLI
/// will buffer. Set defensively — any client path that legitimately needs
/// more should stream to disk instead. 16 MiB is comfortably larger than
/// any legitimate Prometheus exposition, kernel-manifest list, or JSON
/// envelope the API server emits today, and small enough that 8x parallel
/// requests cannot exhaust a CI runner with 256 MiB free.
///
/// `pub` (not `pub(crate)`) — and `#[doc(hidden)]` — purely so the
/// integration test in `tests/bounded_response.rs` can reference the
/// exact constant rather than hard-coding `16 << 20`. It is NOT
/// considered stable API.
#[doc(hidden)]
pub const MAX_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Typed error returned by [`bounded_bytes`] / [`bounded_text`].
///
/// `anyhow::Error` auto-converts from any `std::error::Error + Send + Sync +
/// 'static`, so callers can keep their existing `?` ergonomics — the typed
/// shape is here so a test (or future structured-error work) can match
/// `ResponseTooLarge` without scraping a message string.
///
/// `pub` + `#[doc(hidden)]` so the integration test in
/// `tests/bounded_response.rs` can `match` on the variant. NOT stable
/// API — external consumers should depend on the CLI binary.
#[doc(hidden)]
#[derive(Debug)]
pub enum ApiClientError {
    /// The server announced (via `Content-Length`) or streamed more bytes
    /// than the CLI is willing to buffer in memory. `actual` is what we
    /// observed — either the declared `Content-Length` (when we bailed
    /// fast) or the running total accumulated when the cap tripped.
    ResponseTooLarge {
        /// Either the declared `Content-Length` or the running byte count
        /// at the moment the cap tripped, whichever is the trigger.
        actual: u64,
        /// The cap that was violated. Always equals
        /// [`MAX_RESPONSE_BODY_BYTES`] today, threaded through so the
        /// error message stays accurate if the constant ever changes.
        limit: usize,
    },
    /// `reqwest` failed mid-stream (connection reset, TLS error, etc.).
    /// The underlying error is preserved so the original cause chains
    /// through `anyhow`'s error reporter.
    Stream(reqwest::Error),
    /// The body decoded as bytes successfully but was not valid UTF-8
    /// (only reachable through [`bounded_text`]).
    Utf8(std::string::FromUtf8Error),
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiClientError::ResponseTooLarge { actual, limit } => write!(
                f,
                "response body exceeds in-memory cap: server reported {actual} bytes, \
                 client limit is {limit} bytes ({} MiB); refusing to buffer",
                limit / (1024 * 1024)
            ),
            ApiClientError::Stream(e) => write!(f, "streaming response body: {e}"),
            ApiClientError::Utf8(e) => write!(f, "response body is not valid UTF-8: {e}"),
        }
    }
}

impl std::error::Error for ApiClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApiClientError::Stream(e) => Some(e),
            ApiClientError::Utf8(e) => Some(e),
            ApiClientError::ResponseTooLarge { .. } => None,
        }
    }
}

/// Read a [`reqwest::Response`] body to a `Vec<u8>`, rejecting bodies
/// larger than [`MAX_RESPONSE_BODY_BYTES`].
///
/// Two layers of defence:
///   1. If the server advertised a `Content-Length` larger than the cap,
///      bail FAST without reading a single byte off the socket — this
///      stops a malicious server from forcing the CLI to consume its
///      cap's worth of bandwidth before erroring.
///   2. Otherwise, accumulate `bytes_stream()` chunks into a `Vec<u8>`
///      while tripwiring `total > MAX_RESPONSE_BODY_BYTES` after each
///      chunk. Hitting the trip also emits a `tracing::warn!` so an
///      operator running with `RUST_LOG=warn` sees the source URL.
#[doc(hidden)]
pub async fn bounded_bytes(
    resp: reqwest::Response,
) -> std::result::Result<Vec<u8>, ApiClientError> {
    let limit = MAX_RESPONSE_BODY_BYTES;
    // Fast-fail on declared Content-Length. A server that streams more than
    // it declared will still trip the per-chunk check below.
    if let Some(declared) = resp.content_length() {
        if declared > limit as u64 {
            tracing::warn!(
                declared,
                limit,
                "refusing to buffer response: Content-Length exceeds CLI cap"
            );
            return Err(ApiClientError::ResponseTooLarge {
                actual: declared,
                limit,
            });
        }
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(ApiClientError::Stream)?;
        // Use `checked_add` so a pathological server claiming a chunk of
        // `usize::MAX` bytes can't overflow `buf.len() + chunk.len()` past
        // the cap check.
        let new_total = buf.len().saturating_add(chunk.len());
        if new_total > limit {
            tracing::warn!(
                running_total = new_total as u64,
                limit,
                "response body cap tripped mid-stream"
            );
            return Err(ApiClientError::ResponseTooLarge {
                actual: new_total as u64,
                limit,
            });
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Same as [`bounded_bytes`] but decodes the buffered body as UTF-8.
///
/// Almost every CLI call site needs a `String` for `render_error_response`
/// / `serde_json::from_str`, so this is the canonical helper to reach for.
#[doc(hidden)]
pub async fn bounded_text(resp: reqwest::Response) -> std::result::Result<String, ApiClientError> {
    let bytes = bounded_bytes(resp).await?;
    String::from_utf8(bytes).map_err(ApiClientError::Utf8)
}

// ---------------------------------------------------------------------------
// Terminal-output sanitisation (T18, terminal-hijack hardening)
// ---------------------------------------------------------------------------
//
// Several CLI paths take a server-returned `String` (snapshot restore id,
// invoke fallback body, metrics text, kernel-list rows, observe-board cells)
// and `println!` it verbatim. A malicious server can embed ANSI escape
// sequences (`ESC [ …`) which rewrite the user's terminal title bar, hide
// subsequent output, smuggle in a CR-overwritten "yes" answer to a later
// prompt, or otherwise hijack the operator's terminal session. The helper
// below scrubs server-controlled bytes before they reach the terminal.

/// Strip ASCII control bytes (`< 0x20`, except `\n`, `\r`, `\t`) and the
/// DEL byte (`0x7F`) from a server-returned string before printing it to
/// the user's terminal. Prevents ANSI escape injection, title-bar
/// rewriting, and "smuggle a CR" attacks via malicious responses.
///
/// Multi-byte UTF-8 sequences are preserved untouched — only US-ASCII
/// control bytes are filtered. The replacement is `?` so the operator
/// sees that something was stripped.
///
/// `pub` + `#[doc(hidden)]` so the integration test in
/// `tests/sanitise_terminal_output.rs` can exercise the helper directly.
/// NOT considered stable API — external consumers should depend on the
/// CLI binary.
#[doc(hidden)]
pub fn sanitise_terminal_output(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                '?'
            } else {
                c
            }
        })
        .collect()
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

    // cli fix 6: env-mutating tests now go through the `temp-env` crate
    // rather than the bare `std::env::set_var` / `remove_var` APIs. Two
    // reasons:
    //   * From Rust 2024 onwards `std::env::set_var` is marked `unsafe`, so
    //     the previous shape stops compiling on the workspace's pinned
    //     nightly without an `unsafe { ... }` wrapper that papers over the
    //     real soundness concern.
    //   * Even before 2024, parallel `set_var` from multiple test threads
    //     is documented UB. The hand-rolled `Mutex<()>` previously here
    //     serialised *these three* tests against each other, but did
    //     nothing about other tests in the same binary (or the binary's
    //     dynamically-loaded deps) reading or writing env at the same time.
    //     `temp_env::with_var` owns a process-global mutex internally that
    //     covers every `with_var` call across the test binary, and it
    //     restores the previous value on scope exit so a panicked test
    //     can't leak state into the next.

    #[test]
    fn http_context_no_token_no_tenant_is_noop() {
        // `None` as the value asks `temp-env` to ensure the var is *unset*
        // for the duration of the closure, restoring whatever was there
        // before on return.
        temp_env::with_var(TENSOR_WASM_TOKEN_ENV, None::<&str>, || {
            let ctx = HttpContext::from_env(0);
            assert!(ctx.token.is_none());
            assert_eq!(ctx.tenant, 0);
        });
    }

    #[test]
    fn http_context_picks_up_token_and_tenant() {
        temp_env::with_var(TENSOR_WASM_TOKEN_ENV, Some("  abc123  "), || {
            let ctx = HttpContext::from_env(42);
            // Whitespace is trimmed.
            assert_eq!(ctx.token.as_deref(), Some("abc123"));
            assert_eq!(ctx.tenant, 42);
        });
    }

    #[test]
    fn http_context_treats_empty_token_as_unset() {
        temp_env::with_var(TENSOR_WASM_TOKEN_ENV, Some("   "), || {
            let ctx = HttpContext::from_env(0);
            assert!(ctx.token.is_none());
        });
    }

    #[test]
    fn apply_adds_authorization_header_when_token_set() {
        let ctx = HttpContext {
            token: Some("abc".to_string()),
            tenant: 0,
        };
        // Build an off-stack RequestBuilder so we can inspect headers.
        // Use https:// so this test doesn't trip the plaintext-token warn-once
        // latch and inadvertently flip the gate state observed by the
        // sibling tests in `tests/url_credential_safety.rs`.
        let client = reqwest::Client::new();
        let req = ctx.apply(client.get("https://x")).build().unwrap();
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
        let req = ctx.apply(client.get("https://x")).build().unwrap();
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
        let req = ctx.apply(client.get("https://x")).build().unwrap();
        assert!(req.headers().get(TENANT_HEADER).is_none());
    }

    #[test]
    fn validate_server_url_rejects_userinfo() {
        // Use distinctive credential values (`hunter2`, `alice42`) so the
        // "must not leak" assertions below can grep for them without
        // colliding with words that legitimately appear in the rejection
        // message (the spec-mandated text says "pass auth via …", so
        // looking for the literal substring "pass" would be ambiguous).
        let err = validate_server_url("http://alice42:hunter2@host").unwrap_err();
        let s = format!("{err}");
        assert!(
            s.contains("embedded credentials"),
            "message should mention embedded credentials: {s}"
        );
        // The error must NOT echo the credential values back — that's the
        // whole point of routing this through a generic message instead of
        // `{url}`.
        assert!(
            !s.contains("hunter2"),
            "must not leak the password value: {s}"
        );
        assert!(
            !s.contains("alice42"),
            "must not leak the username value: {s}"
        );
    }

    #[test]
    fn validate_server_url_rejects_userinfo_no_password() {
        // `http://user@host` (no password) is still userinfo and must be
        // rejected, otherwise an operator could smuggle a token-like
        // username past the check.
        assert!(validate_server_url("http://token@host").is_err());
    }

    #[test]
    fn is_loopback_host_classifies_dev_addresses() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("LOCALHOST"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.10.20.30"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("example.com"));
        assert!(!is_loopback_host("10.0.0.1"));
        assert!(!is_loopback_host(""));
    }

    #[test]
    fn sanitise_terminal_output_strips_ansi_csi() {
        // ESC `[` `3` `1` `m` is a common SGR colour escape. After
        // sanitisation the ESC byte must be replaced with `?` so the
        // sequence can no longer be recognised by the terminal emulator
        // as a control directive.
        let out = sanitise_terminal_output("hello\x1b[31mworld");
        assert!(!out.contains('\x1b'), "ESC byte survived: {out:?}");
        assert!(out.contains("hello"));
        assert!(out.contains("world"));
    }

    #[test]
    fn sanitise_terminal_output_preserves_unicode_and_whitespace() {
        // Newline, carriage return, tab, and any multi-byte UTF-8
        // sequence must pass through untouched — the helper must NOT
        // mangle legitimate human-readable output.
        let input = "héllo\n\tworld\r";
        assert_eq!(sanitise_terminal_output(input), input);
    }

    #[test]
    fn sanitise_terminal_output_replaces_del() {
        // 0x7F (DEL) is technically `c.is_control()` per the Unicode
        // tables and is not one of the three whitespace exceptions, so
        // it must be replaced with `?`.
        assert_eq!(sanitise_terminal_output("a\x7Fb"), "a?b");
    }

    #[test]
    fn extract_scheme_host_handles_common_shapes() {
        assert_eq!(
            extract_scheme_host("http://localhost:8080"),
            Some(("http", "localhost"))
        );
        assert_eq!(
            extract_scheme_host("https://example.com/api?x=1"),
            Some(("https", "example.com"))
        );
        assert_eq!(
            extract_scheme_host("http://127.0.0.1"),
            Some(("http", "127.0.0.1"))
        );
        // IPv6 with port and without.
        assert_eq!(
            extract_scheme_host("http://[::1]:8443"),
            Some(("http", "::1"))
        );
        assert_eq!(extract_scheme_host("http://[::1]"), Some(("http", "::1")));
        assert_eq!(extract_scheme_host("ftp://x"), None);
    }
}
