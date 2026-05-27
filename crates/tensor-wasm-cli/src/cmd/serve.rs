// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm serve` — run the HTTP API gateway in-process.
//!
//! Stands up an axum router built from `tensor_wasm_api::build_router_with_full_config`,
//! binds it to `--addr`, and serves until Ctrl-C. Flags map onto the
//! existing API config surface (auth allowlist, tenant header policy,
//! request body cap); knobs not yet exposed by the api crate fall back to
//! the same `TENSOR_WASM_API_*` environment variables `from_env()` reads.
//!
//! This is the entrypoint the README quickstart and `docs/GETTING-STARTED.md`
//! point at; previously the only path to a running gateway was via embedding
//! the library or running an integration test harness, so any change to the
//! flag set here must keep the quickstart command (`tensor-wasm serve --addr
//! 0.0.0.0:8080`) working with no extra setup.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Args;
use tensor_wasm_api::{
    build_router_with_full_config, AppState, AuthConfig, RateLimitConfig, RateLimiter,
    TenantConfig,
};

/// Default bind address used when `--addr` is omitted. Matches the loopback
/// quickstart in `docs/GETTING-STARTED.md`; production deployments should
/// pass `0.0.0.0:<port>` (or a private subnet address) explicitly.
pub const DEFAULT_ADDR: &str = "127.0.0.1:8080";

/// Default request-body cap, mirroring the api crate's
/// `MAX_REQUEST_BODY_BYTES` constant. Surfacing the number here (rather than
/// re-exporting the constant) keeps the CLI help text readable while still
/// matching what the gateway actually enforces.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Policy for the `X-TensorWasm-Tenant` request header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TenantHeaderPolicy {
    /// Header is optional; absence defaults to `TenantId(0)`. Mirrors the
    /// gateway's behaviour when `TENSOR_WASM_API_REQUIRE_TENANT` is unset.
    Optional,
    /// Header is mandatory; requests without it are rejected `400`.
    /// Mirrors `TENSOR_WASM_API_REQUIRE_TENANT=1`.
    Required,
}

impl TenantHeaderPolicy {
    fn into_config(self) -> TenantConfig {
        TenantConfig {
            require_header: matches!(self, TenantHeaderPolicy::Required),
        }
    }
}

/// Arguments to `tensor-wasm serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Address to bind the HTTP server to (e.g. `127.0.0.1:8080`, `0.0.0.0:8080`).
    #[arg(long, default_value = DEFAULT_ADDR)]
    pub addr: SocketAddr,

    /// Bearer token accepted by the gateway. Repeat to allowlist multiple
    /// tokens. Each value is treated as a wildcard-scope token (equivalent to
    /// `token:tenant=*` in the env-driven `TENSOR_WASM_API_TOKENS` allowlist).
    /// If omitted entirely, the gateway falls back to reading
    /// `TENSOR_WASM_API_TOKENS` from the environment; empty/unset there =
    /// dev mode (auth disabled with a startup warning).
    #[arg(long = "token", value_name = "TOKEN")]
    pub tokens: Vec<String>,

    /// Policy for the `X-TensorWasm-Tenant` header. `optional` (default)
    /// mirrors `TENSOR_WASM_API_REQUIRE_TENANT` unset; `required` mirrors
    /// `TENSOR_WASM_API_REQUIRE_TENANT=1`.
    #[arg(long, value_enum, default_value_t = TenantHeaderPolicy::Optional)]
    pub tenant_header_policy: TenantHeaderPolicy,

    /// Origin to allow via CORS. Repeat for multiple origins. NOTE: the
    /// api crate does not yet expose a programmatic CORS knob, so this flag
    /// is currently informational — the gateway's CORS surface is wired
    /// via tower-http defaults inside `tensor-wasm-api`. The flag is kept
    /// so the README quickstart command parses cleanly; once
    /// `tensor_wasm_api::CorsConfig` lands (H18-H20 follow-up) this will
    /// thread through to the router builder.
    #[arg(long = "cors-origin", value_name = "ORIGIN")]
    pub cors_origins: Vec<String>,

    /// Maximum inbound request body size, in bytes. Defaults to 64 MiB to
    /// match the api crate's `MAX_REQUEST_BODY_BYTES`. NOTE: the api crate
    /// does not yet expose a programmatic body-cap knob via
    /// `build_router_with_full_config`; passing a non-default value here
    /// emits a warning and the router uses the compiled-in 64 MiB cap.
    /// Wired through once the api crate grows a `BodyLimitConfig`.
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    pub max_body_bytes: usize,
}

/// Entry point for `tensor-wasm serve`.
pub async fn run(args: ServeArgs) -> Result<()> {
    let addr = args.addr;

    // Auth: if --token was passed, build an explicit allowlist; otherwise
    // fall back to the env-driven loader so operators who already export
    // TENSOR_WASM_API_TOKENS (the documented production knob) don't have to
    // re-declare them on the command line.
    let auth = if args.tokens.is_empty() {
        AuthConfig::from_env()
    } else {
        AuthConfig::from_tokens(args.tokens.iter().map(|s| s.as_str()))
    };

    let tenant = args.tenant_header_policy.into_config();

    // Per-token rate limiter: the api crate currently only exposes
    // RateLimitConfig::from_env / ::disabled. Mirror the env-driven default
    // so `TENSOR_WASM_API_RATE_LIMIT_QPS` keeps working when set; otherwise
    // the limiter is a pass-through.
    let limiter = RateLimiter::new(RateLimitConfig::from_env());

    if args.max_body_bytes != DEFAULT_MAX_BODY_BYTES {
        tracing::warn!(
            target: "tensor_wasm_cli::serve",
            requested = args.max_body_bytes,
            actual = DEFAULT_MAX_BODY_BYTES,
            "--max-body-bytes is currently informational; gateway uses the \
             compiled-in 64 MiB cap until tensor-wasm-api exposes a \
             BodyLimitConfig",
        );
    }
    if !args.cors_origins.is_empty() {
        tracing::warn!(
            target: "tensor_wasm_cli::serve",
            count = args.cors_origins.len(),
            "--cors-origin is currently informational; tensor-wasm-api does \
             not yet expose a CorsConfig (H18-H20 follow-up)",
        );
    }

    let state = AppState::try_new().context("initialising AppState (wasmtime engine)")?;
    let router = build_router_with_full_config(state, auth, tenant, limiter);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding TCP listener on {addr}"))?;

    // Print the canonical scheme://host:port line the README quickstart
    // greps for; keep it on a single line so log filters can latch onto it.
    println!("listening on http://{addr}");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum::serve")?;

    Ok(())
}

/// Resolves when the process should shut down cleanly. Currently just Ctrl-C;
/// once we add SIGTERM handling for systemd/kubernetes drains this is the
/// place to extend.
async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::error!(
            target: "tensor_wasm_cli::serve",
            error = %e,
            "failed to install Ctrl-C handler; serve loop will run until killed",
        );
        // Park forever so the caller's `with_graceful_shutdown` never fires
        // a spurious shutdown on a handler-install error.
        std::future::pending::<()>().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Tiny harness to exercise the clap derive without standing up the
    /// whole `Cli` enum.
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        args: ServeArgs,
    }

    #[test]
    fn defaults_parse() {
        let p = Probe::try_parse_from(["serve"]).expect("defaults parse");
        assert_eq!(p.args.addr.to_string(), "127.0.0.1:8080");
        assert!(p.args.tokens.is_empty());
        assert_eq!(p.args.tenant_header_policy, TenantHeaderPolicy::Optional);
        assert!(p.args.cors_origins.is_empty());
        assert_eq!(p.args.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }

    #[test]
    fn parses_addr_override() {
        let p = Probe::try_parse_from(["serve", "--addr", "0.0.0.0:9000"]).expect("parses");
        assert_eq!(p.args.addr.to_string(), "0.0.0.0:9000");
    }

    #[test]
    fn rejects_garbage_addr() {
        assert!(Probe::try_parse_from(["serve", "--addr", "not-an-address"]).is_err());
    }

    #[test]
    fn collects_repeated_tokens() {
        let p =
            Probe::try_parse_from(["serve", "--token", "a", "--token", "b"]).expect("parses");
        assert_eq!(p.args.tokens, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn collects_repeated_cors_origins() {
        let p = Probe::try_parse_from([
            "serve",
            "--cors-origin",
            "https://app.example",
            "--cors-origin",
            "https://admin.example",
        ])
        .expect("parses");
        assert_eq!(p.args.cors_origins.len(), 2);
    }

    #[test]
    fn parses_tenant_header_policy() {
        let p =
            Probe::try_parse_from(["serve", "--tenant-header-policy", "required"]).expect("parses");
        assert_eq!(p.args.tenant_header_policy, TenantHeaderPolicy::Required);
        let cfg = p.args.tenant_header_policy.into_config();
        assert!(cfg.require_header);
    }
}
